// matmul_f16packed.wgsl — matmul with half-precision STORAGE, full-precision
// COMPUTE. The on-GPU buffers for A and B hold packed half-precision values
// (two f16 per u32 via pack2x16float / unpack2x16float). The shader unpacks
// at read time, multiplies + accumulates in f32, and writes the f32 result.
//
// Why this matters: matmul on big language models is bandwidth-bound on
// M-series GPUs. Halving the weight-buffer footprint roughly halves the
// memory traffic per inner-loop step, which translates to ~2x throughput on
// the kernels where bandwidth dominates. Compute precision stays f32, so
// the math is identical to the f32 reference within rounding (max abs error
// scales like the standard f16 ~3e-4 relative).
//
// No shader-f16 extension is required — pack2x16float and unpack2x16float
// are core WGSL built-ins. This makes the path available on every WebGPU
// device, including Playwright's headless Chromium.
//
// Layout convention (same as matmul.wgsl):
//   C = A @ B,  A: [M, K], B: [K, N], C: [M, N], all row-major.
//   K and N are required to be even (each pair of consecutive f16 lives in
//   one u32). The host packer guarantees that with end-padding when needed.

struct Dims {
  M: u32,
  K: u32,
  N: u32,
  _pad: u32,
};

// A and B are packed: each u32 holds two consecutive f16 values along the
// last (contiguous) dimension. The buffer length for A is M*K/2 u32s, etc.
@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= dims.M || col >= dims.N) {
    return;
  }

  let K = dims.K;
  let N = dims.N;
  let halfK = K / 2u;
  let halfN = N / 2u;

  // col is in f32 terms; locate the u32 word that holds it and whether we
  // want the low or high lane.
  let bWordCol = col / 2u;
  let bIsHigh = (col & 1u) == 1u;

  var acc: f32 = 0.0;
  // Walk K two at a time — one u32 of A holds A[row, 2k] and A[row, 2k+1].
  for (var kPair: u32 = 0u; kPair < halfK; kPair = kPair + 1u) {
    let aPair = unpack2x16float(A[row * halfK + kPair]);
    // Two B values from two different rows (rows 2*kPair and 2*kPair+1),
    // both at column `col`. Each lives in its own u32 word (bWordCol),
    // possibly in the low or high lane.
    let bRow0 = unpack2x16float(B[(2u * kPair) * halfN + bWordCol]);
    let bRow1 = unpack2x16float(B[(2u * kPair + 1u) * halfN + bWordCol]);
    let b0 = select(bRow0.x, bRow0.y, bIsHigh);
    let b1 = select(bRow1.x, bRow1.y, bIsHigh);
    acc = acc + aPair.x * b0 + aPair.y * b1;
  }

  C[row * dims.N + col] = acc;
}
