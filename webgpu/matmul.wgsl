// matmul.wgsl — WebGPU compute shader for matrix multiply (Phase 5).
//
// C = A @ B      A:[M,K]  B:[K,N]  C:[M,N]   (all row-major f32)
//
// One invocation computes one output element. Workgroups are 16x16; the host
// dispatches ceil(M/16) x ceil(N/16) of them. This is the naive version —
// correct first; workgroup-shared-memory tiling is a later optimization.
//
// Acceptance: output equals the WASM matmul within tolerance, and is
// measurably faster on a large matrix (see webgpu/kernels.ts).

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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= dims.M || col >= dims.N) {
    return; // guard the ragged edge when M/N are not multiples of 16
  }
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
    acc = acc + A[row * dims.K + k] * B[k * dims.N + col];
  }
  C[row * dims.N + col] = acc;
}
