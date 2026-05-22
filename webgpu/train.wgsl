// train.wgsl — matmul forward + the two backward variants (Phase 5).
//
// Forward:   C = A @ B           dA = dC @ Bᵀ        dB = Aᵀ @ dC
//
// Each is one entry point. dA reuses an "A times B-transposed" matmul; dB
// reuses an "A-transposed times B" matmul. Naive one-element-per-invocation —
// correct first; tiling is a later optimisation. 16x16 workgroups.
//
// `dims` carries (M, K, N) for whatever product the entry point computes.

struct Dims {
  M: u32,
  K: u32,
  N: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

// C = A @ B        A:[M,K]  B:[K,N]  C:[M,N]
@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= dims.M || col >= dims.N) { return; }
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
    acc = acc + A[row * dims.K + k] * B[k * dims.N + col];
  }
  C[row * dims.N + col] = acc;
}

// C = A @ Bᵀ       A:[M,K]  B:[N,K]  C:[M,N]   (used for dA = dC @ Bᵀ)
@compute @workgroup_size(16, 16)
fn matmul_abt(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= dims.M || col >= dims.N) { return; }
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
    acc = acc + A[row * dims.K + k] * B[col * dims.K + k];
  }
  C[row * dims.N + col] = acc;
}

// C = Aᵀ @ B       A:[K,M]  B:[K,N]  C:[M,N]   (used for dB = Aᵀ @ dC)
@compute @workgroup_size(16, 16)
fn matmul_atb(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= dims.M || col >= dims.N) { return; }
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
    acc = acc + A[k * dims.M + row] * B[k * dims.N + col];
  }
  C[row * dims.N + col] = acc;
}
