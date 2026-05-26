// train_vec4.wgsl — vec4-aligned matmul variants for the training pipeline.
// Compiled separately from train.wgsl because the binding inner type differs
// (array<vec4<f32>> here, array<f32> there). Same g0-g5 + p uniform layout
// from the host's perspective — storage buffers don't care about view type.
//
// Algorithm identical to matmul_blocked_*.wgsl (16×16 workgroup, 4×4 register
// block, 64×64 output tile), just issues 128-bit memory transactions for A
// and B. C remains array<f32> (write-only, doesn't benefit from vec4).
//
// Measured on M-series: 1.37× faster than scalar blocked4 at 2048³.
// Requires K and N to be multiples of 4 — all preset matmul shapes satisfy this.

struct P {
  a: u32, b: u32, c: u32, d: u32,
  fa: f32, fb: f32, fc: f32, fd: f32,
};

// Access mode MUST be read_write here, not read — the shared bind-group
// layout in ops.ts declares all six storage buffers as type: "storage"
// (read-write). WGSL `var<storage, read>` requires the layout to be
// `read-only-storage`. Mismatching the two yields silently-wrong reads on
// some implementations (Chromium/Apple included) rather than a validation
// error — exactly the symptom we hit on the first attempt: standalone bench
// (which created its own auto-layout) passed; train.wgsl bind-group layout
// integration produced loss → 88. Keep these read_write even though the
// kernel only reads from g0 and g1.
@group(0) @binding(0) var<storage, read_write> g0: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> g1: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> g2: array<f32>;
@group(0) @binding(3) var<storage, read_write> g3: array<f32>;
@group(0) @binding(4) var<storage, read_write> g4: array<f32>;
@group(0) @binding(5) var<storage, read_write> g5: array<f32>;
@group(0) @binding(6) var<uniform> p: P;

// ===========================================================================
// matmul (forward): C = A @ B,  A:[M,K]  B:[K,N]
// ===========================================================================

var<workgroup> mb_tileA: array<array<f32, 16>, 64>;
var<workgroup> mb_tileB: array<array<f32, 64>, 16>;

@compute @workgroup_size(16, 16)
fn matmul_blocked_vec4(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = p.a; let K = p.b; let N = p.c;
  let K4 = K / 4u; let N4 = N / 4u;
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
    let kBase4 = t * 4u;
    // Load A: 256 vec4s per K-tile (64 rows × 4 vec4s).
    {
      let row = tid / 4u;
      let col4 = tid % 4u;
      let aRow = blockRow + row;
      var v: vec4<f32>;
      if (aRow < M && kBase + col4 * 4u < K) {
        v = g0[aRow * K4 + kBase4 + col4];
      } else {
        v = vec4<f32>(0.0);
      }
      mb_tileA[row][col4 * 4u + 0u] = v.x;
      mb_tileA[row][col4 * 4u + 1u] = v.y;
      mb_tileA[row][col4 * 4u + 2u] = v.z;
      mb_tileA[row][col4 * 4u + 3u] = v.w;
    }
    // Load B: 256 vec4s per K-tile (16 rows × 16 vec4s).
    {
      let row = tid / 16u;
      let col4 = tid % 16u;
      let bRow = kBase + row;
      let bCol4 = blockCol / 4u + col4;
      var v: vec4<f32>;
      if (bRow < K && bCol4 < N4) {
        v = g1[bRow * N4 + bCol4];
      } else {
        v = vec4<f32>(0.0);
      }
      mb_tileB[row][col4 * 4u + 0u] = v.x;
      mb_tileB[row][col4 * 4u + 1u] = v.y;
      mb_tileB[row][col4 * 4u + 2u] = v.z;
      mb_tileB[row][col4 * 4u + 3u] = v.w;
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
