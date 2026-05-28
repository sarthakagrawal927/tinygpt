// train_f16.wgsl — matmul variant with packed-half STORAGE on B (weights),
// f32 STORAGE on A (activations), f32 COMPUTE everywhere. Compiled separately
// from train.wgsl because the binding inner type for g1 differs (`array<u32>`
// here, `array<f32>` there); the host-side bind-group layout is identical
// across all *.wgsl files in this set.
//
// Why this kernel: matmul on transformer training is bandwidth-bound on
// M-series GPUs. Weights (B) are loaded K times per output across the
// workgroup; halving the weight storage to f16 roughly halves the B
// bandwidth without disturbing the f32 numerics of A or the accumulator.
// Estimated win on bandwidth-bound shapes: ~1.5-2× over the f32 vec4
// path. Numerical drift: a single matmul accumulates K f32×f32 products
// with B unpacked from f16, so the relative error is at most O(K * eps_f16)
// per output ≈ K × 5e-4 / sqrt(K) ≈ sqrt(K) × 5e-4 (with random-sign
// rounding), which for K=1024 is ~1.6%. The numerics gate in ops.ts
// validates this empirically before activating the path.
//
// Algorithm: 16×16 workgroup, 4×4 register block, 64×64 output tile — same
// as matmul_blocked_vec4, just with the B-side load swapped for an
// `unpack2x16float` lane-select. A stays scalar f32 (no vec4) because
// activations are produced by other f32 kernels in the same submit and we
// don't want a separate pack-pass per step on the activation stream.
//
// Layout requirements:
//   K and N must be even (each pair of consecutive f16 in B packs into one
//   u32 along the N-contiguous axis). All preset matmul shapes satisfy this.

struct P {
  a: u32, b: u32, c: u32, d: u32,
  fa: f32, fb: f32, fc: f32, fd: f32,
};

// Access mode MUST be read_write here, not read — see the long comment in
// train_vec4.wgsl. The shared bind-group layout in ops.ts declares all six
// storage buffers as read-write; declaring `read` here causes silent wrong
// reads on Chromium/Apple even though only `read` is needed.
@group(0) @binding(0) var<storage, read_write> g0: array<f32>;      // A [M,K] f32
@group(0) @binding(1) var<storage, read_write> g1: array<u32>;      // B [K,N] packed f16 (K*N/2 u32)
@group(0) @binding(2) var<storage, read_write> g2: array<f32>;      // C [M,N] f32
@group(0) @binding(3) var<storage, read_write> g3: array<f32>;
@group(0) @binding(4) var<storage, read_write> g4: array<f32>;
@group(0) @binding(5) var<storage, read_write> g5: array<f32>;
@group(0) @binding(6) var<uniform> p: P;

// Shared tiles hold UNPACKED f32 values. Unpacking from packed-half happens
// once per global load; the inner K-loop reads f32 from shared memory.
var<workgroup> mb_tileA: array<array<f32, 16>, 64>;
var<workgroup> mb_tileB: array<array<f32, 64>, 16>;

@compute @workgroup_size(16, 16)
fn matmul_blocked_f16(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = p.a; let K = p.b; let N = p.c;
  let halfN = N / 2u;
  let blockRow = wid.x * 64u;
  let blockCol = wid.y * 64u;
  let lrow = lid.x; let lcol = lid.y;
  let tid = lrow * 16u + lcol;

  var acc: array<array<f32, 4>, 4>;
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      acc[i][j] = 0.0;
    }
  }

  let nTiles = (K + 15u) / 16u;
  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    let kBase = t * 16u;

    // Load A: each of the 256 threads loads 4 scalars (64 rows × 16 cols
    // per K-tile, 256 threads × 4 = 1024 scalars; here we load 1 per
    // thread × 4 iterations through staggered indexing for coverage).
    // Simpler: thread (tid) loads tile[row=tid/16, col=tid%16] for the 16×16
    // bottom corner, then the other 3 quadrants by adding 16 to row.
    {
      let row = tid / 16u;
      let col = tid % 16u;
      let kCol = kBase + col;
      for (var rb: u32 = 0u; rb < 4u; rb = rb + 1u) {
        let tileR = row + rb * 16u;
        let aRow = blockRow + tileR;
        var v: f32 = 0.0;
        if (aRow < M && kCol < K) {
          v = g0[aRow * K + kCol];
        }
        mb_tileA[tileR][col] = v;
      }
    }

    // Load B: same 16×64 tile staffing. Each thread reads a packed pair
    // from g1 and unpacks one of the two halves depending on column parity.
    // 16 K-rows × 64 N-cols = 1024 scalars; 256 threads × 4 each.
    {
      let row = tid / 16u;
      let col = tid % 16u;
      let bRow = kBase + row;
      for (var cb: u32 = 0u; cb < 4u; cb = cb + 1u) {
        let tileC = col + cb * 16u;
        let bCol = blockCol + tileC;
        var v: f32 = 0.0;
        if (bRow < K && bCol < N) {
          let bWordCol = bCol / 2u;
          let bIsHigh = (bCol & 1u) == 1u;
          let pair = unpack2x16float(g1[bRow * halfN + bWordCol]);
          v = select(pair.x, pair.y, bIsHigh);
        }
        mb_tileB[row][tileC] = v;
      }
    }
    workgroupBarrier();

    let myA0 = lrow * 4u;
    let myB0 = lcol * 4u;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let a0 = mb_tileA[myA0 + 0u][k];
      let a1 = mb_tileA[myA0 + 1u][k];
      let a2 = mb_tileA[myA0 + 2u][k];
      let a3 = mb_tileA[myA0 + 3u][k];
      let b0 = mb_tileB[k][myB0 + 0u];
      let b1 = mb_tileB[k][myB0 + 1u];
      let b2 = mb_tileB[k][myB0 + 2u];
      let b3 = mb_tileB[k][myB0 + 3u];
      acc[0][0] = acc[0][0] + a0 * b0; acc[0][1] = acc[0][1] + a0 * b1;
      acc[0][2] = acc[0][2] + a0 * b2; acc[0][3] = acc[0][3] + a0 * b3;
      acc[1][0] = acc[1][0] + a1 * b0; acc[1][1] = acc[1][1] + a1 * b1;
      acc[1][2] = acc[1][2] + a1 * b2; acc[1][3] = acc[1][3] + a1 * b3;
      acc[2][0] = acc[2][0] + a2 * b0; acc[2][1] = acc[2][1] + a2 * b1;
      acc[2][2] = acc[2][2] + a2 * b2; acc[2][3] = acc[2][3] + a2 * b3;
      acc[3][0] = acc[3][0] + a3 * b0; acc[3][1] = acc[3][1] + a3 * b1;
      acc[3][2] = acc[3][2] + a3 * b2; acc[3][3] = acc[3][3] + a3 * b3;
    }
    workgroupBarrier();
  }

  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    let outRow = blockRow + lrow * 4u + i;
    if (outRow >= M) { continue; }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let outCol = blockCol + lcol * 4u + j;
      if (outCol < N) {
        g2[outRow * N + outCol] = acc[i][j];
      }
    }
  }
}

// ===========================================================================
// Backward dA = dY @ W^T (matmulAbt variant with f16-packed weight)
//
// matmulAbt's convention: output[m, n] = sum_k A[m, k] * B[n, k]. For our
// backward pass with B = W (the weight stored as [orig_K, orig_N] row-major,
// contiguous on orig_N), the call is matmulAbt(dY, W, M=orig_M, K=orig_N,
// N=orig_K). So matmulAbt's K is the inner dim (= orig_N = the contiguous
// packing axis), and matmulAbt's N is the row count of W.
//
// In this kernel:
//   bRow ∈ [0, matmulAbt_N) = [0, orig_K) — a row in W
//   bCol ∈ [0, matmulAbt_K) = [0, orig_N) — a column in W (packed axis)
//   W[bRow, bCol] = unpack2x16float(g1[bRow * halfK + bCol/2]).{x or y}
//
// Same blocked4 algorithm as matmul_abt_blocked in train.wgsl (16×16
// workgroup, 4×4 register block per thread, 64×64 output tile). Only the
// B-side global load is changed to unpack from packed-half storage.
var<workgroup> mab_tileA_f16: array<array<f32, 16>, 64>;  // [m][k]
var<workgroup> mab_tileB_f16: array<array<f32, 64>, 16>;  // [k][n] but loaded from W[n,k]

@compute @workgroup_size(16, 16)
fn matmul_abt_blocked_f16(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = p.a; let K = p.b; let N = p.c;  // matmulAbt convention: output [M, N], inner K
  let halfK = K / 2u;
  let blockRow = wid.x * 64u;
  let blockCol = wid.y * 64u;
  let lrow = lid.x; let lcol = lid.y;

  var acc: array<array<f32, 4>, 4>;
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      acc[i][j] = 0.0;
    }
  }

  let nTiles = (K + 15u) / 16u;
  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    let kBase = t * 16u;

    // Load A[blockRow + lrow*4 + i, kBase + lcol] into mab_tileA_f16 — f32 scalar.
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let aRow = blockRow + lrow * 4u + i;
      let aCol = kBase + lcol;
      var v: f32 = 0.0;
      if (aRow < M && aCol < K) { v = g0[aRow * K + aCol]; }
      mab_tileA_f16[lrow * 4u + i][lcol] = v;
    }

    // Load B[blockCol + lcol*4 + j, kBase + lrow] — B is W with [N, K] view,
    // stored as packed-half along K. Each thread reads 4 elements (one row of
    // the B-tile slice it owns); unpack-select per element.
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let bRow = blockCol + lcol * 4u + j;   // row in W (orig_K direction)
      let bCol = kBase + lrow;               // column in W (orig_N, packed)
      var v: f32 = 0.0;
      if (bRow < N && bCol < K) {
        let bColPair = bCol / 2u;
        let bIsHigh = (bCol & 1u) == 1u;
        let pair = unpack2x16float(g1[bRow * halfK + bColPair]);
        v = select(pair.x, pair.y, bIsHigh);
      }
      mab_tileB_f16[lrow][lcol * 4u + j] = v;
    }
    workgroupBarrier();

    let myA0 = lrow * 4u;
    let myB0 = lcol * 4u;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let a0 = mab_tileA_f16[myA0 + 0u][k];
      let a1 = mab_tileA_f16[myA0 + 1u][k];
      let a2 = mab_tileA_f16[myA0 + 2u][k];
      let a3 = mab_tileA_f16[myA0 + 3u][k];
      let b0 = mab_tileB_f16[k][myB0 + 0u];
      let b1 = mab_tileB_f16[k][myB0 + 1u];
      let b2 = mab_tileB_f16[k][myB0 + 2u];
      let b3 = mab_tileB_f16[k][myB0 + 3u];
      acc[0][0] = acc[0][0] + a0 * b0; acc[0][1] = acc[0][1] + a0 * b1;
      acc[0][2] = acc[0][2] + a0 * b2; acc[0][3] = acc[0][3] + a0 * b3;
      acc[1][0] = acc[1][0] + a1 * b0; acc[1][1] = acc[1][1] + a1 * b1;
      acc[1][2] = acc[1][2] + a1 * b2; acc[1][3] = acc[1][3] + a1 * b3;
      acc[2][0] = acc[2][0] + a2 * b0; acc[2][1] = acc[2][1] + a2 * b1;
      acc[2][2] = acc[2][2] + a2 * b2; acc[2][3] = acc[2][3] + a2 * b3;
      acc[3][0] = acc[3][0] + a3 * b0; acc[3][1] = acc[3][1] + a3 * b1;
      acc[3][2] = acc[3][2] + a3 * b2; acc[3][3] = acc[3][3] + a3 * b3;
    }
    workgroupBarrier();
  }

  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    let outRow = blockRow + lrow * 4u + i;
    if (outRow >= M) { continue; }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let outCol = blockCol + lcol * 4u + j;
      if (outCol < N) {
        g2[outRow * N + outCol] = acc[i][j];
      }
    }
  }
}

// Pack a f32 weight buffer into a packed-f16 storage buffer. Reads from g0
// (length M*N f32) and writes to g1 (length M*N/2 u32). Stride along the
// last (N) axis. N must be even.
//
// Dispatched on weight load (after importState) and after each AdamW update
// to the corresponding weight tensor. Cost is one read + one write per
// f32 element — cheap relative to the matmul that follows.
@compute @workgroup_size(64)
fn pack_to_f16(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  // Each thread writes ONE u32 = two consecutive f16. Total threads needed:
  // M*N/2. Dispatch ceil((M*N/2) / 64) workgroups.
  let totalPairs = p.a * (p.b / 2u);  // p.a=M (row count), p.b=N (col count)
  let idx = gid.x;
  if (idx >= totalPairs) { return; }
  let pairCol = idx % (p.b / 2u);
  let row = idx / (p.b / 2u);
  let baseF32 = row * p.b + pairCol * 2u;
  let lo = g0[baseF32];
  let hi = g0[baseF32 + 1u];
  g1[idx] = pack2x16float(vec2<f32>(lo, hi));
}
