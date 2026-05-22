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
