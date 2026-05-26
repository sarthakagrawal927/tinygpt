// matmul_tiled_f16.wgsl — workgroup-shared-memory tiling + packed-half
// storage on A and B. Stacks the two independent wins:
//
//   - Tiling: each global load is amortized across 16 multiply-accumulates
//     via var<workgroup> shared memory. Cuts the global-read count by ~16×.
//   - f16-packed storage: A and B hold two f16 per u32 via pack2x16float;
//     halves bandwidth pressure on the global loads we still do.
//
// Combined effect on bandwidth-bound matmuls should be roughly the product
// of the two standalone wins. Accumulation stays in f32 inside the shader
// for numerical safety; only storage is packed-half.
//
// K and N must be even (the packed layout puts two f16 in one u32 along
// the contiguous axis).

const TILE: u32 = 16u;

struct Dims {
  M: u32,
  K: u32,
  N: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> A: array<u32>; // packed: M*K/2 u32s
@group(0) @binding(1) var<storage, read> B: array<u32>; // packed: K*N/2 u32s
@group(0) @binding(2) var<storage, read_write> C: array<f32>; // M*N f32s
@group(0) @binding(3) var<uniform> dims: Dims;

// Shared tiles hold UNPACKED f32 values — unpack on load, accumulate in fp32.
// This is the right tradeoff: pack/unpack cost is paid once per global load,
// shared memory ops happen 16 times per load.
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.x;
  let col = gid.y;
  let lrow = lid.x;
  let lcol = lid.y;

  let M = dims.M;
  let K = dims.K;
  let N = dims.N;
  let halfK = K / 2u;
  let halfN = N / 2u;

  var acc: f32 = 0.0;
  let nTiles = (K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    // Each thread loads ONE element of A and ONE element of B into shared.
    // Both A and B come from packed-u32 storage, so we figure out which
    // half-lane holds the element we want and unpack just that one.
    let aCol = t * TILE + lcol;
    let bRow = t * TILE + lrow;

    // A[row, aCol] — locate the u32 word and lane.
    var aVal: f32 = 0.0;
    if (row < M && aCol < K) {
      let aWordCol = aCol / 2u;
      let aIsHigh = (aCol & 1u) == 1u;
      let pair = unpack2x16float(A[row * halfK + aWordCol]);
      aVal = select(pair.x, pair.y, aIsHigh);
    }
    tileA[lrow][lcol] = aVal;

    // B[bRow, col] — locate the u32 word and lane.
    var bVal: f32 = 0.0;
    if (bRow < K && col < N) {
      let bWordCol = col / 2u;
      let bIsHigh = (col & 1u) == 1u;
      let pair = unpack2x16float(B[bRow * halfN + bWordCol]);
      bVal = select(pair.x, pair.y, bIsHigh);
    }
    tileB[lrow][lcol] = bVal;

    workgroupBarrier();

    // Inner product across the K-tile from shared memory.
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      acc = acc + tileA[lrow][k] * tileB[k][lcol];
    }

    workgroupBarrier();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}
