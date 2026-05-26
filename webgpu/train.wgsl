// train.wgsl — the WebGPU training kernels (Phase 5).
//
// One module, one binding layout: six generic storage buffers (g0..g5) plus a
// params uniform (p). Every kernel reads/writes whichever slots it needs — so
// adding a kernel never means touching the host-side bind-group plumbing.
// `p` carries up to four u32 (dims) and four f32 (eps, scale, ...).
//
// Naive kernels — correct first; tiling is a later optimisation. Stage 1 added
// the matmuls; stage 2 adds layernorm, GELU, and the elementwise ops.

struct P {
  a: u32, b: u32, c: u32, d: u32,
  fa: f32, fb: f32, fc: f32, fd: f32,
};

@group(0) @binding(0) var<storage, read_write> g0: array<f32>;
@group(0) @binding(1) var<storage, read_write> g1: array<f32>;
@group(0) @binding(2) var<storage, read_write> g2: array<f32>;
@group(0) @binding(3) var<storage, read_write> g3: array<f32>;
@group(0) @binding(4) var<storage, read_write> g4: array<f32>;
@group(0) @binding(5) var<storage, read_write> g5: array<f32>;
@group(0) @binding(6) var<uniform> p: P;

// --- matmul family ---------------------------------------------------------
// C = A @ B    g0=A[M,K] g1=B[K,N] g2=C[M,N]   p.a=M p.b=K p.c=N
@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x; let col = gid.y;
  if (row >= p.a || col >= p.c) { return; }
  var acc = 0.0;
  for (var k = 0u; k < p.b; k = k + 1u) {
    acc = acc + g0[row * p.b + k] * g1[k * p.c + col];
  }
  g2[row * p.c + col] = acc;
}

// Thread-blocked + workgroup-shared-tiled matmul. Same bind layout as
// `matmul` above — A=g0, B=g1, C=g2, p.a=M p.b=K p.c=N — but each thread
// computes a 4×4 register block of output values, and the 16×16 workgroup
// outputs a 64×64 tile. Standalone-kernel measurement: 5.18× at 2048×2048.
//
// Dispatch with workgroups = ceil(M/64) × ceil(N/64).
var<workgroup> mb_tileA: array<array<f32, 16>, 64>;
var<workgroup> mb_tileB: array<array<f32, 64>, 16>;

@compute @workgroup_size(16, 16)
fn matmul_blocked(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = p.a; let K = p.b; let N = p.c;
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
    // Cooperative load: 4 A-elements and 4 B-elements per thread.
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let aRow = blockRow + lrow * 4u + i;
      let aCol = kBase + lcol;
      var v: f32 = 0.0;
      if (aRow < M && aCol < K) { v = g0[aRow * K + aCol]; }
      mb_tileA[lrow * 4u + i][lcol] = v;
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let bRow = kBase + lrow;
      let bCol = blockCol + lcol * 4u + j;
      var v: f32 = 0.0;
      if (bRow < K && bCol < N) { v = g1[bRow * N + bCol]; }
      mb_tileB[lrow][lcol * 4u + j] = v;
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

// C = A @ Bᵀ   g0=A[M,K] g1=B[N,K] g2=C[M,N]   (dA = dC @ Bᵀ)
@compute @workgroup_size(16, 16)
fn matmul_abt(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x; let col = gid.y;
  if (row >= p.a || col >= p.c) { return; }
  var acc = 0.0;
  for (var k = 0u; k < p.b; k = k + 1u) {
    acc = acc + g0[row * p.b + k] * g1[col * p.b + k];
  }
  g2[row * p.c + col] = acc;
}

// Thread-blocked variant of matmul_abt. Same A-load pattern as matmul_blocked
// (row-major); B's load pattern flips because we want B[col, k] not B[k, col].
// For the tileB load, threads pull rows of B that correspond to the output
// columns we're writing.
var<workgroup> mab_tileA: array<array<f32, 16>, 64>;
var<workgroup> mab_tileB: array<array<f32, 64>, 16>; // [k][n] but loaded from B[n,k]

@compute @workgroup_size(16, 16)
fn matmul_abt_blocked(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = p.a; let K = p.b; let N = p.c;
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
    // Load A[blockRow + lrow*4 + i, kBase + lcol] into mab_tileA — same as forward.
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let aRow = blockRow + lrow * 4u + i;
      let aCol = kBase + lcol;
      var v: f32 = 0.0;
      if (aRow < M && aCol < K) { v = g0[aRow * K + aCol]; }
      mab_tileA[lrow * 4u + i][lcol] = v;
    }
    // Load B[blockCol + lcol*4 + j, kBase + lrow] — B is [N, K] in abt convention.
    // Place in mab_tileB[lrow][lcol*4 + j] so the inner loop reads B-row-aligned.
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let bRow = blockCol + lcol * 4u + j; // this is a column of output, but a row of B[N,K]
      let bCol = kBase + lrow;
      var v: f32 = 0.0;
      if (bRow < N && bCol < K) { v = g1[bRow * K + bCol]; }
      mab_tileB[lrow][lcol * 4u + j] = v;
    }
    workgroupBarrier();

    let myA0 = lrow * 4u;
    let myB0 = lcol * 4u;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let a0 = mab_tileA[myA0 + 0u][k];
      let a1 = mab_tileA[myA0 + 1u][k];
      let a2 = mab_tileA[myA0 + 2u][k];
      let a3 = mab_tileA[myA0 + 3u][k];
      let b0 = mab_tileB[k][myB0 + 0u];
      let b1 = mab_tileB[k][myB0 + 1u];
      let b2 = mab_tileB[k][myB0 + 2u];
      let b3 = mab_tileB[k][myB0 + 3u];
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

// C = Aᵀ @ B   g0=A[K,M] g1=B[K,N] g2=C[M,N]   (dB = Aᵀ @ dC)
@compute @workgroup_size(16, 16)
fn matmul_atb(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x; let col = gid.y;
  if (row >= p.a || col >= p.c) { return; }
  var acc = 0.0;
  for (var k = 0u; k < p.b; k = k + 1u) {
    acc = acc + g0[k * p.a + row] * g1[k * p.c + col];
  }
  g2[row * p.c + col] = acc;
}

// Thread-blocked variant of matmul_atb. A is [K, M] so we read A[k, row]
// (column-major access pattern from M's perspective). B is [K, N] like
// forward but indexed by k (rows of B = K).
var<workgroup> mat_tileA: array<array<f32, 64>, 16>; // [k][m]
var<workgroup> mat_tileB: array<array<f32, 64>, 16>; // [k][n]

@compute @workgroup_size(16, 16)
fn matmul_atb_blocked(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = p.a; let K = p.b; let N = p.c;
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
    // Load A[kBase + lrow, blockRow + lcol*4 + i] — A is [K, M].
    // Place into mat_tileA[lrow][lcol*4 + i] so inner loop reads A[k][m].
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let aRow = kBase + lrow;
      let aCol = blockRow + lcol * 4u + i;
      var v: f32 = 0.0;
      if (aRow < K && aCol < M) { v = g0[aRow * M + aCol]; }
      mat_tileA[lrow][lcol * 4u + i] = v;
    }
    // Load B[kBase + lrow, blockCol + lcol*4 + j] — same K-row indexing.
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let bRow = kBase + lrow;
      let bCol = blockCol + lcol * 4u + j;
      var v: f32 = 0.0;
      if (bRow < K && bCol < N) { v = g1[bRow * N + bCol]; }
      mat_tileB[lrow][lcol * 4u + j] = v;
    }
    workgroupBarrier();

    let myA0 = lrow * 4u;
    let myB0 = lcol * 4u;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      // mat_tileA[k][m] — pull row m of the output block from k-row of A.
      let a0 = mat_tileA[k][myA0 + 0u];
      let a1 = mat_tileA[k][myA0 + 1u];
      let a2 = mat_tileA[k][myA0 + 2u];
      let a3 = mat_tileA[k][myA0 + 3u];
      let b0 = mat_tileB[k][myB0 + 0u];
      let b1 = mat_tileB[k][myB0 + 1u];
      let b2 = mat_tileB[k][myB0 + 2u];
      let b3 = mat_tileB[k][myB0 + 3u];
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

// --- elementwise -----------------------------------------------------------
// c = a + b    g0=a g1=b g2=c   p.a=n   (residual add)
@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.a) { return; }
  g2[i] = g0[i] + g1[i];
}

// y += bias (broadcast over rows)   g0=y[rows,D] g1=bias[D]   p.a=rows p.b=D
@compute @workgroup_size(64)
fn bias_add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.a * p.b) { return; }
  g0[i] = g0[i] + g1[i % p.b];
}

// db[d] = sum over rows of dy[row,d]   g0=dy[rows,D] g1=db[D]   p.a=rows p.b=D
@compute @workgroup_size(64)
fn bias_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dcol = gid.x;
  if (dcol >= p.b) { return; }
  var s = 0.0;
  for (var n = 0u; n < p.a; n = n + 1u) { s = s + g0[n * p.b + dcol]; }
  g1[dcol] = s;
}

// erf via the Abramowitz & Stegun 7.1.26 approximation (~1e-7 accurate) — WGSL
// has no erf builtin, and the C++ model uses the exact (erf-based) GELU.
fn erf(x: f32) -> f32 {
  let s = sign(x);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
          - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return s * y;
}

const INV_SQRT2: f32 = 0.70710678118;
const INV_SQRT_2PI: f32 = 0.39894228040;

// y = GELU(x)   g0=x g1=y   p.a=n
@compute @workgroup_size(64)
fn gelu_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.a) { return; }
  let v = g0[i];
  g1[i] = 0.5 * v * (1.0 + erf(v * INV_SQRT2));
}

// dx = dy * GELU'(x)   g0=x g1=dy g2=dx   p.a=n
@compute @workgroup_size(64)
fn gelu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.a) { return; }
  let v = g0[i];
  let cdf = 0.5 * (1.0 + erf(v * INV_SQRT2));
  let pdf = INV_SQRT_2PI * exp(-0.5 * v * v);
  g2[i] = g1[i] * (cdf + v * pdf);
}

// --- layernorm -------------------------------------------------------------
// y = gamma * (x - mean) / sqrt(var + eps) + beta, over the last dim D.
// g0=x g1=gamma g2=beta g3=y g4=mean g5=rstd   p.a=N rows, p.b=D, p.fa=eps
@compute @workgroup_size(64)
fn layernorm_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= p.a) { return; }
  let base = row * p.b;
  var mu = 0.0;
  for (var d = 0u; d < p.b; d = d + 1u) { mu = mu + g0[base + d]; }
  mu = mu / f32(p.b);
  var v = 0.0;
  for (var d = 0u; d < p.b; d = d + 1u) {
    let diff = g0[base + d] - mu;
    v = v + diff * diff;
  }
  v = v / f32(p.b);
  let rs = 1.0 / sqrt(v + p.fa);
  g4[row] = mu;
  g5[row] = rs;
  for (var d = 0u; d < p.b; d = d + 1u) {
    g3[base + d] = g1[d] * ((g0[base + d] - mu) * rs) + g2[d];
  }
}

// dx for layernorm.   g0=x g1=gamma g2=mean g3=rstd g4=dy g5=dx   p.a=N p.b=D
@compute @workgroup_size(64)
fn layernorm_dx(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= p.a) { return; }
  let base = row * p.b;
  let mu = g2[row];
  let rs = g3[row];
  var mean_dxhat = 0.0;
  var mean_dxhat_xhat = 0.0;
  for (var d = 0u; d < p.b; d = d + 1u) {
    let xhat = (g0[base + d] - mu) * rs;
    let dxhat = g4[base + d] * g1[d];
    mean_dxhat = mean_dxhat + dxhat;
    mean_dxhat_xhat = mean_dxhat_xhat + dxhat * xhat;
  }
  mean_dxhat = mean_dxhat / f32(p.b);
  mean_dxhat_xhat = mean_dxhat_xhat / f32(p.b);
  for (var d = 0u; d < p.b; d = d + 1u) {
    let xhat = (g0[base + d] - mu) * rs;
    let dxhat = g4[base + d] * g1[d];
    g5[base + d] = rs * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
  }
}

// dgamma / dbeta for layernorm — one invocation per feature, summed over rows.
// g0=x g1=mean g2=rstd g3=dy g4=dgamma g5=dbeta   p.a=N p.b=D
@compute @workgroup_size(64)
fn layernorm_dgb(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dcol = gid.x;
  if (dcol >= p.b) { return; }
  var dg = 0.0;
  var db = 0.0;
  for (var n = 0u; n < p.a; n = n + 1u) {
    let dy = g3[n * p.b + dcol];
    let xhat = (g0[n * p.b + dcol] - g1[n]) * g2[n];
    dg = dg + dy * xhat;
    db = db + dy;
  }
  g4[dcol] = dg;
  g5[dcol] = db;
}

// --- causal multi-head attention (the SDPA core; projections are matmul) ----
// Layout: q,k,v,ctx are [B,T,C] with C = H*hd; attn,dscores are [B,H,T,T].
// One invocation per (b,h,t). p.a=B p.b=T p.c=C p.d=H, p.fa = 1/sqrt(hd).

// attn = softmax(causal(q.kᵀ * scale))   g0=q g1=k g2=attn
//
// Score array sized for ctx up to 1024 (Behemoth). Per-invocation private
// memory ≈ 4 KB — well within Apple's limit. The previous fixed size of 256
// silently broke at ctx > 256.
@compute @workgroup_size(64)
fn attn_softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let idx = gid.x;
  if (idx >= B * H * T) { return; }
  let b = idx / (H * T); let rem = idx % (H * T);
  let h = rem / T; let t1 = rem % T;
  let hd = C / H; let off = h * hd; let scale = p.fa;

  var sc: array<f32, 1024>;
  var maxv = -1e30;
  let qb = (b * T + t1) * C + off;
  for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
    var s = 0.0;
    let kb = (b * T + t2) * C + off;
    for (var d = 0u; d < hd; d = d + 1u) { s = s + g0[qb + d] * g1[kb + d]; }
    s = s * scale;
    sc[t2] = s;
    if (s > maxv) { maxv = s; }
  }
  var sum = 0.0;
  for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
    sc[t2] = exp(sc[t2] - maxv);
    sum = sum + sc[t2];
  }
  let inv = 1.0 / sum;
  let arow = ((b * H + h) * T + t1) * T;
  for (var t2 = 0u; t2 < T; t2 = t2 + 1u) { g2[arow + t2] = 0.0; }
  for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) { g2[arow + t2] = sc[t2] * inv; }
}

// Flash-Attention-1-style fused softmax + value: same output as
// attn_softmax → attn_value but in one kernel pass, saving the round-trip
// of the attention matrix through global memory between the two kernels.
//
// Inputs:  g0=q[B,T,C]  g1=k[B,T,C]  g2=v[B,T,C]  g3=attn[B,H,T,T]  g4=ctx[B,T,C]
// Params:  p.a=B  p.b=T  p.c=C  p.d=H  p.fa = 1/sqrt(hd)
//
// We still write attn into g3 because the backward kernels read it. (FA2
// recomputes attention on backward; we don't, yet — that's an open item.)
// But we no longer have to read attn back into another kernel to compute
// ctx — it's accumulated in the same pass.
//
// Algorithm:
//   1. Pass 1 over K: compute all scores, track running max.
//   2. Pass 2 over K: write softmax(scores) into attn AND multiply by V
//      while accumulating into ctx, all in one loop.
//
// TODO(fa2-backward): the g3 = attn[B,H,T,T] writeback below exists only
// because the current backward path (matmul_abt_blocked, attn_softmax_bwd,
// attn_value_bwd) reads the full attention matrix from global memory. Once
// the FA2 backward kernel lands and recomputes attention on-the-fly (with
// the saved per-row max + sum statistics), every `g3[arow + t2] = ...`
// store in this kernel can be deleted — and so can the B*H*T*T attention
// allocation in ops.ts. Trigger condition: FA2 backward is wired through
// gpu_model.ts and parity-checked against the reference. Until then, the
// writeback is load-bearing.
@compute @workgroup_size(64)
fn attn_fused_sv(@builtin(global_invocation_id) gid: vec3<u32>) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let idx = gid.x;
  if (idx >= B * H * T) { return; }
  let b = idx / (H * T); let rem = idx % (H * T);
  let h = rem / T; let t1 = rem % T;
  let hd = C / H; let off = h * hd; let scale = p.fa;

  var sc: array<f32, 1024>;
  var maxv = -1e30;
  let qb = (b * T + t1) * C + off;
  // Pass 1: scores + max.
  for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
    var s = 0.0;
    let kb = (b * T + t2) * C + off;
    for (var d = 0u; d < hd; d = d + 1u) { s = s + g0[qb + d] * g1[kb + d]; }
    s = s * scale;
    sc[t2] = s;
    if (s > maxv) { maxv = s; }
  }
  // Pass 2a: sum.
  var sum = 0.0;
  for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
    sc[t2] = exp(sc[t2] - maxv);
    sum = sum + sc[t2];
  }
  let inv = 1.0 / sum;

  // Write the (normalised) attention probabilities — backward kernels need them.
  let arow = ((b * H + h) * T + t1) * T;
  for (var t2 = 0u; t2 < T; t2 = t2 + 1u) { g3[arow + t2] = 0.0; }
  for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) { g3[arow + t2] = sc[t2] * inv; }

  // Pass 2b: ctx[t1, :hd] = sum_{t2 <= t1} softmax(t2) * v[t2, :hd].
  // Zero the output, then accumulate. Per-thread output is just hd floats.
  let cb = (b * T + t1) * C + off;
  for (var d = 0u; d < hd; d = d + 1u) { g4[cb + d] = 0.0; }
  for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
    let a = sc[t2] * inv;
    let vb = (b * T + t2) * C + off;
    for (var d = 0u; d < hd; d = d + 1u) {
      g4[cb + d] = g4[cb + d] + a * g2[vb + d];
    }
  }
}

// ctx = attn @ v   g0=attn g1=v g2=ctx
@compute @workgroup_size(64)
fn attn_value(@builtin(global_invocation_id) gid: vec3<u32>) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let idx = gid.x;
  if (idx >= B * H * T) { return; }
  let b = idx / (H * T); let rem = idx % (H * T);
  let h = rem / T; let t1 = rem % T;
  let hd = C / H; let off = h * hd;
  let arow = ((b * H + h) * T + t1) * T;
  let cb = (b * T + t1) * C + off;
  for (var d = 0u; d < hd; d = d + 1u) {
    var acc = 0.0;
    for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
      acc = acc + g0[arow + t2] * g1[(b * T + t2) * C + off + d];
    }
    g2[cb + d] = acc;
  }
}

// dscores = softmax-backward(dattn), dattn = dctx @ vᵀ
// g0=dctx g1=v g2=attn g3=dscores
@compute @workgroup_size(64)
fn attn_dscores(@builtin(global_invocation_id) gid: vec3<u32>) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let idx = gid.x;
  if (idx >= B * H * T) { return; }
  let b = idx / (H * T); let rem = idx % (H * T);
  let h = rem / T; let t1 = rem % T;
  let hd = C / H; let off = h * hd;
  let arow = ((b * H + h) * T + t1) * T;
  let cb = (b * T + t1) * C + off;

  var dattn: array<f32, 256>;
  var dot = 0.0;
  for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
    var da = 0.0;
    let vb = (b * T + t2) * C + off;
    for (var d = 0u; d < hd; d = d + 1u) { da = da + g0[cb + d] * g1[vb + d]; }
    dattn[t2] = da;
    dot = dot + da * g2[arow + t2];
  }
  for (var t2 = 0u; t2 < T; t2 = t2 + 1u) { g3[arow + t2] = 0.0; }
  for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
    g3[arow + t2] = g2[arow + t2] * (dattn[t2] - dot);
  }
}

// dq[t1] = scale * sum_{t2<=t1} dscores[t1,t2] * k[t2]   g0=dscores g1=k g2=dq
@compute @workgroup_size(64)
fn attn_dq(@builtin(global_invocation_id) gid: vec3<u32>) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let idx = gid.x;
  if (idx >= B * H * T) { return; }
  let b = idx / (H * T); let rem = idx % (H * T);
  let h = rem / T; let t1 = rem % T;
  let hd = C / H; let off = h * hd; let scale = p.fa;
  let arow = ((b * H + h) * T + t1) * T;
  let qb = (b * T + t1) * C + off;
  for (var d = 0u; d < hd; d = d + 1u) {
    var acc = 0.0;
    for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
      acc = acc + g0[arow + t2] * g1[(b * T + t2) * C + off + d];
    }
    g2[qb + d] = acc * scale;
  }
}

// dk[t2] = scale * sum_{t1>=t2} dscores[t1,t2] * q[t1]   g0=dscores g1=q g2=dk
@compute @workgroup_size(64)
fn attn_dk(@builtin(global_invocation_id) gid: vec3<u32>) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let idx = gid.x;
  if (idx >= B * H * T) { return; }
  let b = idx / (H * T); let rem = idx % (H * T);
  let h = rem / T; let t2 = rem % T;
  let hd = C / H; let off = h * hd; let scale = p.fa;
  let kb = (b * T + t2) * C + off;
  for (var d = 0u; d < hd; d = d + 1u) {
    var acc = 0.0;
    for (var t1 = t2; t1 < T; t1 = t1 + 1u) {
      let s = g0[((b * H + h) * T + t1) * T + t2];
      acc = acc + s * g1[(b * T + t1) * C + off + d];
    }
    g2[kb + d] = acc * scale;
  }
}

// dv[t2] = sum_{t1>=t2} attn[t1,t2] * dctx[t1]   g0=attn g1=dctx g2=dv
@compute @workgroup_size(64)
fn attn_dv(@builtin(global_invocation_id) gid: vec3<u32>) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let idx = gid.x;
  if (idx >= B * H * T) { return; }
  let b = idx / (H * T); let rem = idx % (H * T);
  let h = rem / T; let t2 = rem % T;
  let hd = C / H; let off = h * hd;
  let vb = (b * T + t2) * C + off;
  for (var d = 0u; d < hd; d = d + 1u) {
    var acc = 0.0;
    for (var t1 = t2; t1 < T; t1 = t1 + 1u) {
      let a = g0[((b * H + h) * T + t1) * T + t2];
      acc = acc + a * g1[(b * T + t1) * C + off + d];
    }
    g2[vb + d] = acc;
  }
}

// FA2-aware backward: dscores from q/k/L instead of from the cached attn.
// Recovers P = exp(q·k·scale − L[t1]) inside the kernel, so the FA2 forward
// can drop its second-pass attn writeback entirely. Same output shape as
// attn_dscores; same downstream consumers (attn_dq / attn_dk).
//
// g0=q g1=k g2=L g3=dctx g4=v g5=dscores   p.a=B p.b=T p.c=C p.d=H p.fa=1/sqrt(hd)
//
// One invocation per (b, h, t1). Walks K once to compute P*dP per t2 and
// the row-wide D = sum_{t2 <= t1} P · dP, then writes dscores[t1, t2] =
// P · (dP − D). Stash array sized for ctx ≤ 1024 (Behemoth-ready, matches
// the bump made to attn_softmax earlier this session).
@compute @workgroup_size(64)
fn attn_dscores_fa2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let idx = gid.x;
  if (idx >= B * H * T) { return; }
  let b = idx / (H * T); let rem = idx % (H * T);
  let h = rem / T; let t1 = rem % T;
  let hd = C / H; let off = h * hd; let scale = p.fa;
  let cb = (b * T + t1) * C + off;
  let Lt1 = g2[(b * H + h) * T + t1];

  // Pass 1: walk K once, build P[t2] and dP[t2] arrays, accumulate D.
  var Pcache: array<f32, 1024>;
  var dPcache: array<f32, 1024>;
  var D: f32 = 0.0;
  for (var t2: u32 = 0u; t2 <= t1; t2 = t2 + 1u) {
    var s: f32 = 0.0;
    let kb = (b * T + t2) * C + off;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
      s = s + g0[cb + d] * g1[kb + d];
    }
    let P = exp(s * scale - Lt1);
    var dP: f32 = 0.0;
    let vb = (b * T + t2) * C + off;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
      dP = dP + g3[cb + d] * g4[vb + d];
    }
    Pcache[t2] = P;
    dPcache[t2] = dP;
    D = D + P * dP;
  }

  // Pass 2: zero the row, then write dS = P · (dP − D) for causal entries.
  let arow = ((b * H + h) * T + t1) * T;
  for (var t2: u32 = 0u; t2 < T; t2 = t2 + 1u) { g5[arow + t2] = 0.0; }
  for (var t2: u32 = 0u; t2 <= t1; t2 = t2 + 1u) {
    g5[arow + t2] = Pcache[t2] * (dPcache[t2] - D);
  }
}

// FA2-aware backward: dv from q/k/L instead of from the cached attn.
// Same algorithm as attn_dv (per-K-row, walks Q rows ≥ t2), but recomputes
// P inline from q · k · scale − L[t1].
//
// g0=q g1=k g2=L g3=dctx g4=dv   p.a=B p.b=T p.c=C p.d=H p.fa=1/sqrt(hd)
@compute @workgroup_size(64)
fn attn_dv_fa2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let idx = gid.x;
  if (idx >= B * H * T) { return; }
  let b = idx / (H * T); let rem = idx % (H * T);
  let h = rem / T; let t2 = rem % T;
  let hd = C / H; let off = h * hd; let scale = p.fa;
  let vb = (b * T + t2) * C + off;
  let kb = (b * T + t2) * C + off;

  for (var d: u32 = 0u; d < hd; d = d + 1u) {
    var acc: f32 = 0.0;
    for (var t1: u32 = t2; t1 < T; t1 = t1 + 1u) {
      // Recompute S, then P = exp(S − L[t1]).
      var s: f32 = 0.0;
      let qb = (b * T + t1) * C + off;
      for (var dd: u32 = 0u; dd < hd; dd = dd + 1u) {
        s = s + g0[qb + dd] * g1[kb + dd];
      }
      let Lt1 = g2[(b * H + h) * T + t1];
      let P = exp(s * scale - Lt1);
      acc = acc + P * g3[(b * T + t1) * C + off + d];
    }
    g4[vb + d] = acc;
  }
}

// --- embeddings, cross-entropy, optimizer (stage 4) ------------------------

// x = tok_emb[id] + pos_emb[t]   g0=tok_emb[V,C] g1=pos_emb[Tctx,C]
//   g2=ids[N] (int values as f32)  g3=x[N,C]   p.a=N p.b=C p.c=T
@compute @workgroup_size(64)
fn embed_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let N = p.a; let C = p.b; let T = p.c;
  let i = gid.x;
  if (i >= N * C) { return; }
  let n = i / C; let c = i % C;
  let id = u32(g2[n]);
  let t = n % T;
  g3[i] = g0[id * C + c] + g1[t * C + c];
}

// dtok[v,c] = sum over rows whose token id == v of dx[n,c]
// g0=dx[N,C] g1=ids[N] g2=dtok[V,C]   p.a=N p.b=C p.c=V
@compute @workgroup_size(64)
fn embed_tok_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
  let N = p.a; let C = p.b; let V = p.c;
  let i = gid.x;
  if (i >= V * C) { return; }
  let v = i / C; let c = i % C;
  var s = 0.0;
  for (var n = 0u; n < N; n = n + 1u) {
    if (u32(g1[n]) == v) { s = s + g0[n * C + c]; }
  }
  g2[i] = s;
}

// dpos[t,c] = sum over batch of dx[(b*T+t),c]   g0=dx[N,C] g1=dpos[T,C]
//   p.a=N p.b=C p.c=T
@compute @workgroup_size(64)
fn embed_pos_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
  let N = p.a; let C = p.b; let T = p.c;
  let i = gid.x;
  if (i >= T * C) { return; }
  let t = i / C; let c = i % C;
  var s = 0.0;
  var n = t;
  loop {
    if (n >= N) { break; }
    s = s + g0[n * C + c];
    n = n + T;
  }
  g1[i] = s;
}

// Per row: softmax over the vocab, the loss, and dlogits = (softmax - onehot)/N.
// g0=logits[N,V] g1=targets[N] g2=dlogits[N,V] g3=loss[N]   p.a=N p.b=V
@compute @workgroup_size(64)
fn cross_entropy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let N = p.a; let V = p.b;
  let n = gid.x;
  if (n >= N) { return; }
  let base = n * V;
  var mx = g0[base];
  for (var v = 1u; v < V; v = v + 1u) {
    let x = g0[base + v];
    if (x > mx) { mx = x; }
  }
  var sum = 0.0;
  for (var v = 0u; v < V; v = v + 1u) { sum = sum + exp(g0[base + v] - mx); }
  let tgt = u32(g1[n]);
  g3[n] = -((g0[base + tgt] - mx) - log(sum));
  let invN = 1.0 / f32(N);
  for (var v = 0u; v < V; v = v + 1u) {
    let prob = exp(g0[base + v] - mx) / sum;
    var onehot = 0.0;
    if (v == tgt) { onehot = 1.0; }
    g2[base + v] = (prob - onehot) * invN;
  }
}

// In-place AdamW step. g0=param g1=grad g2=m g3=v
//   p.a=count p.b=step  p.fa=lr p.fb=weight_decay  (betas/eps fixed)
@compute @workgroup_size(64)
fn adamw(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.a) { return; }
  let b1 = 0.9; let b2 = 0.95; let eps = 1e-8;
  let step = f32(p.b);
  let g = g1[i];
  let m = b1 * g2[i] + (1.0 - b1) * g;
  let v = b2 * g3[i] + (1.0 - b2) * g * g;
  g2[i] = m;
  g3[i] = v;
  let mhat = m / (1.0 - pow(b1, step));
  let vhat = v / (1.0 - pow(b2, step));
  g0[i] = g0[i] - p.fa * (mhat / (sqrt(vhat) + eps) + p.fb * g0[i]);
}
