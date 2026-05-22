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

// --- causal multi-head attention (the SDPA core; projections are matmul) ----
// Layout: q,k,v,ctx are [B,T,C] with C = H*hd; attn,dscores are [B,H,T,T].
// One invocation per (b,h,t). p.a=B p.b=T p.c=C p.d=H, p.fa = 1/sqrt(hd).

// attn = softmax(causal(q.kᵀ * scale))   g0=q g1=k g2=attn
@compute @workgroup_size(64)
fn attn_softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let idx = gid.x;
  if (idx >= B * H * T) { return; }
  let b = idx / (H * T); let rem = idx % (H * T);
  let h = rem / T; let t1 = rem % T;
  let hd = C / H; let off = h * hd; let scale = p.fa;

  var sc: array<f32, 256>;
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
