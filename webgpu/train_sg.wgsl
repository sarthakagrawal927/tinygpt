// train_sg.wgsl — subgroup-using variants of the reduction-heavy kernels.
// Compiled separately from train.wgsl because `enable subgroups;` is a
// module-level directive and the base train.wgsl needs to work on devices
// that lack the feature. ops.ts dispatches into this module only when
// the device advertises the `subgroups` feature.
//
// Convention: one WORKGROUP per row (vs. one thread per row in train.wgsl).
// Each workgroup uses all 64 threads to cooperatively reduce that row.
// On big d_model (Mega / Behemoth) this turns serial 1280-element scans
// into 20-element-per-thread scans + a single subgroupAdd — the regime
// where this lever actually pays off.

enable subgroups;

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

// Within the workgroup, subgroups produce one partial sum each. We fold
// those down to a single value via shared memory. Max subgroups per
// workgroup of 64 is 64 (sg_size = 1) — set the array big enough for the
// worst case.
const WG: u32 = 64u;
var<workgroup> sg_partial_sum: array<f32, 64>;
var<workgroup> sg_partial_max: array<f32, 64>;
var<workgroup> wg_mu: f32;
var<workgroup> wg_rs: f32;
var<workgroup> wg_sum: f32;
var<workgroup> wg_max: f32;

// LayerNorm forward — one workgroup per row, all 64 threads collaborate.
// g0=x[N,D] g1=gamma[D] g2=beta[D] g3=y[N,D] g4=mean[N] g5=rstd[N]
// p.a=N p.b=D p.fa=eps
@compute @workgroup_size(WG)
fn layernorm_forward_sg(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sid: u32,
  @builtin(subgroup_size) sgSize: u32,
) {
  let row = wid.x;
  let tid = lid.x;
  let D = p.b;
  if (row >= p.a) { return; }
  let base = row * D;
  let invD = 1.0 / f32(D);
  let nSg = (WG + sgSize - 1u) / sgSize;
  let sgId = tid / sgSize;

  // Pass 1 — sum.
  var localSum: f32 = 0.0;
  var i = tid;
  loop {
    if (i >= D) { break; }
    localSum = localSum + g0[base + i];
    i = i + WG;
  }
  let sgSum = subgroupAdd(localSum);
  if (sid == 0u) { sg_partial_sum[sgId] = sgSum; }
  workgroupBarrier();
  if (tid == 0u) {
    var s: f32 = 0.0;
    for (var k: u32 = 0u; k < nSg; k = k + 1u) { s = s + sg_partial_sum[k]; }
    wg_mu = s * invD;
  }
  workgroupBarrier();
  let mu = wg_mu;

  // Pass 2 — variance.
  var localVar: f32 = 0.0;
  i = tid;
  loop {
    if (i >= D) { break; }
    let diff = g0[base + i] - mu;
    localVar = localVar + diff * diff;
    i = i + WG;
  }
  let sgVar = subgroupAdd(localVar);
  if (sid == 0u) { sg_partial_sum[sgId] = sgVar; }
  workgroupBarrier();
  if (tid == 0u) {
    var v: f32 = 0.0;
    for (var k: u32 = 0u; k < nSg; k = k + 1u) { v = v + sg_partial_sum[k]; }
    wg_rs = 1.0 / sqrt(v * invD + p.fa);
    g4[row] = mu;
    g5[row] = wg_rs;
  }
  workgroupBarrier();
  let rs = wg_rs;

  // Pass 3 — apply.
  i = tid;
  loop {
    if (i >= D) { break; }
    g3[base + i] = g1[i] * ((g0[base + i] - mu) * rs) + g2[i];
    i = i + WG;
  }
}

// Cross-entropy with subgroup reductions over the vocab (V=256 for the
// byte-level model, so each thread handles 4 elements with WG=64). Same
// shape: per-row softmax + loss + gradient.
// g0=logits[N,V] g1=targets[N] g2=dlogits[N,V] g3=loss[N]   p.a=N p.b=V
@compute @workgroup_size(WG)
fn cross_entropy_sg(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sid: u32,
  @builtin(subgroup_size) sgSize: u32,
) {
  let n = wid.x;
  let tid = lid.x;
  let V = p.b;
  if (n >= p.a) { return; }
  let base = n * V;
  let nSg = (WG + sgSize - 1u) / sgSize;
  let sgId = tid / sgSize;

  // Pass 1 — max.
  var localMax: f32 = -3.4e38;
  var i = tid;
  loop {
    if (i >= V) { break; }
    let x = g0[base + i];
    if (x > localMax) { localMax = x; }
    i = i + WG;
  }
  let sgMax = subgroupMax(localMax);
  if (sid == 0u) { sg_partial_max[sgId] = sgMax; }
  workgroupBarrier();
  if (tid == 0u) {
    var m: f32 = sg_partial_max[0];
    for (var k: u32 = 1u; k < nSg; k = k + 1u) {
      let v = sg_partial_max[k];
      if (v > m) { m = v; }
    }
    wg_max = m;
  }
  workgroupBarrier();
  let mx = wg_max;

  // Pass 2 — exp + sum.
  var localSum: f32 = 0.0;
  i = tid;
  loop {
    if (i >= V) { break; }
    localSum = localSum + exp(g0[base + i] - mx);
    i = i + WG;
  }
  let sgSum2 = subgroupAdd(localSum);
  if (sid == 0u) { sg_partial_sum[sgId] = sgSum2; }
  workgroupBarrier();
  if (tid == 0u) {
    var s: f32 = 0.0;
    for (var k: u32 = 0u; k < nSg; k = k + 1u) { s = s + sg_partial_sum[k]; }
    wg_sum = s;
    let tgt = u32(g1[n]);
    g3[n] = -((g0[base + tgt] - mx) - log(s));
  }
  workgroupBarrier();
  let sum = wg_sum;

  // Pass 3 — gradient.
  let invN = 1.0 / f32(p.a);
  let tgt = u32(g1[n]);
  i = tid;
  loop {
    if (i >= V) { break; }
    let prob = exp(g0[base + i] - mx) / sum;
    var onehot: f32 = 0.0;
    if (i == tgt) { onehot = 1.0; }
    g2[base + i] = (prob - onehot) * invN;
    i = i + WG;
  }
}
