// attention_fa2.wgsl — Flash Attention 2 forward (causal, no dropout).
//
// One workgroup per (batch, head, Q-tile of Br=16 rows). 16 threads per
// workgroup — one thread per Q row in the tile. Each thread maintains its
// own online-softmax state (m_i, l_i) and partial output O_i[hd] in
// registers; the workgroup cooperatively walks K/V in Bc=16-row blocks
// loaded into shared memory.
//
// Improvement over `attn_fused_sv` (which lives in train.wgsl):
//   - attn_fused_sv allocates `array<f32, 1024>` of scores PER THREAD. With
//     ctx=1024 that's 4 KB private memory per invocation; Apple drivers
//     spill it. FA2 holds Q in shared mem and only the running softmax
//     state in registers — O(hd) per thread, not O(T).
//   - attn_fused_sv has one thread per Q row; FA2 has 16 threads per Q tile
//     loading K/V cooperatively, so K/V global reads are amortised 16×
//     across the tile (one load, 16 dot products).
//   - Memory traffic: K and V are streamed once per Q tile rather than
//     once per Q row.
//
// This kernel's bind layout matches train.wgsl's six-storage-buffer + uniform
// convention. Same `P` struct, same g0..g5 ordering. Drops directly into the
// existing `attentionForward` dispatch path once integration is wired up.
//
// Bindings (forward only):
//   g0 = q[B, T, C]            input
//   g1 = k[B, T, C]            input
//   g2 = v[B, T, C]            input
//   g3 = attn[B, H, T, T]      output (kept so the existing backward kernels
//                              can still read it — FA2 backward will recompute
//                              attention on the fly, and at that point we can
//                              drop this writeback entirely)
//   g4 = ctx[B, T, C]          output
//   p.a = B  p.b = T  p.c = C  p.d = H   p.fa = 1/sqrt(hd)
//
// Constraints / caveats (see docs/fa2_forward_notes.md for the full writeup):
//   - hd (= C/H) must satisfy hd <= MAX_HD (128). Every preset currently
//     uses hd in {32, 48, 64}.
//   - T does NOT need to be a multiple of 16; the boundary K block masks
//     out-of-range columns with -inf scores.
//   - Causal mask is exact: K blocks strictly past the tile's last Q row are
//     skipped entirely; the boundary block applies the mask per (q_row, t2).

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

// Tile sizes. Br = Q rows per tile, Bc = K rows per block. Equal here
// because that makes the cooperative load (16 threads × hd) trivial: each
// thread loads exactly one row of K and one row of V per block.
const BR: u32 = 16u;
const BC: u32 = 16u;
// Maximum head dim the kernel will accept. Workgroup memory is sized at
// this; threads only iterate over the actual `hd` at runtime. Every preset
// today uses hd in {32, 48, 64}; 64 keeps total workgroup memory at
// 3 × 16 × 64 × 4 = 12 KB (well within the typical 16-32 KB limit). Raise
// to 128 once we have a preset that needs it, and re-verify against
// device.limits.maxComputeWorkgroupStorageSize.
const MAX_HD: u32 = 64u;

// Workgroup-shared tiles. One tile of 16 Q rows lives here for the whole
// kernel; K and V tiles are loaded once per block.
var<workgroup> Qtile: array<array<f32, MAX_HD>, BR>;
var<workgroup> Ktile: array<array<f32, MAX_HD>, BC>;
var<workgroup> Vtile: array<array<f32, MAX_HD>, BC>;
// Final m / l values per Q row in the tile — written by each thread at the
// end of the K loop so the second pass (the one that writes attn[] for the
// existing backward to consume) can read them back.
var<workgroup> m_final: array<f32, BR>;
var<workgroup> l_final: array<f32, BR>;

// Forward kernel.
// Workgroup dispatch: x = ceil(T / BR), y = B * H.
@compute @workgroup_size(BR)
fn fa2_forward(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let B = p.a; let T = p.b; let C = p.c; let H = p.d;
  let hd = C / H;
  let scale = p.fa;

  let bh = wid.y;
  let b = bh / H;
  let h = bh % H;
  let q_tile = wid.x;                // which 16-row chunk of Q
  let q_start = q_tile * BR;         // first Q row in this tile

  let lane = lid.x;                  // 0..BR-1; the Q row this thread owns
  let q_row = q_start + lane;        // absolute Q row index (may be >= T)
  let q_valid = q_row < T;

  let off = h * hd;                  // head-channel offset within C

  // --- 1. Load Q tile cooperatively. Each thread loads its own row. -------
  // Out-of-range Q rows fill with zero — their outputs will be discarded.
  for (var d: u32 = 0u; d < hd; d = d + 1u) {
    var v: f32 = 0.0;
    if (q_valid) { v = g0[(b * T + q_row) * C + off + d]; }
    Qtile[lane][d] = v;
  }
  workgroupBarrier();

  // --- 2. Per-thread online-softmax state. --------------------------------
  // O_i is held in a private array of size MAX_HD; the loop only touches the
  // first `hd` entries.
  var m_i: f32 = -1.0e30;
  var l_i: f32 = 0.0;
  var O_i: array<f32, MAX_HD>;
  for (var d: u32 = 0u; d < hd; d = d + 1u) { O_i[d] = 0.0; }

  // Last Q row index in this tile (clamped to T-1). The causal mask says we
  // only need K rows up to q_end inclusive; past that, every Q row in the
  // tile would mask the K row out anyway.
  // (When the whole tile is past T, q_end will be < 0 in signed terms — we
  // handle that by skipping the loop entirely below.)
  var q_end: u32 = 0u;
  if (q_start < T) {
    let last = q_start + BR - 1u;
    if (last < T) { q_end = last; } else { q_end = T - 1u; }
  }

  let nKBlocks = (T + BC - 1u) / BC;

  // --- 3. Walk K/V in blocks of BC=16. -----------------------------------
  if (q_start < T) {
    for (var kj: u32 = 0u; kj < nKBlocks; kj = kj + 1u) {
      let k_start = kj * BC;
      // Causal skip: if this K block starts past every Q row in the tile,
      // every score in it is -inf and we can skip the load + compute.
      if (k_start > q_end) { break; }

      // (3a) Cooperative load of K and V tiles. Each thread loads row `lane`.
      let kv_row = k_start + lane;
      let kv_valid = kv_row < T;
      for (var d: u32 = 0u; d < hd; d = d + 1u) {
        var kv: f32 = 0.0;
        var vv: f32 = 0.0;
        if (kv_valid) {
          kv = g1[(b * T + kv_row) * C + off + d];
          vv = g2[(b * T + kv_row) * C + off + d];
        }
        Ktile[lane][d] = kv;
        Vtile[lane][d] = vv;
      }
      workgroupBarrier();

      // (3b) Per-thread: compute scores S[0..BC-1] for this block against
      // *my* Q row, find the block max, then merge with running m_i / l_i.
      // We only do this for valid Q rows.
      if (q_valid) {
        var S: array<f32, BC>;
        var m_block: f32 = -1.0e30;
        for (var jj: u32 = 0u; jj < BC; jj = jj + 1u) {
          let t2 = k_start + jj;
          var s: f32 = -1.0e30;
          // Mask: out-of-range T columns AND causal mask (t2 > q_row).
          if (t2 < T && t2 <= q_row) {
            s = 0.0;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
              s = s + Qtile[lane][d] * Ktile[jj][d];
            }
            s = s * scale;
          }
          S[jj] = s;
          if (s > m_block) { m_block = s; }
        }

        // Online-softmax merge. m_new = max(m_i, m_block).
        // alpha = exp(m_i - m_new) rescales prior accumulation. beta_j = exp(S_j - m_new).
        // l_new = alpha * l_i + sum_j beta_j;  O_new = alpha * O_i + sum_j beta_j * V_j.
        let m_new = max(m_i, m_block);
        // Guard: if m_block was still -inf (entire block masked — shouldn't
        // happen because we break above, but be defensive), skip the merge.
        if (m_new > -1.0e29) {
          let alpha = exp(m_i - m_new);
          var l_new = alpha * l_i;
          // Rescale O_i in place.
          for (var d: u32 = 0u; d < hd; d = d + 1u) { O_i[d] = O_i[d] * alpha; }
          // Add per-column contributions.
          for (var jj: u32 = 0u; jj < BC; jj = jj + 1u) {
            if (S[jj] > -1.0e29) {
              let pj = exp(S[jj] - m_new);
              l_new = l_new + pj;
              for (var d: u32 = 0u; d < hd; d = d + 1u) {
                O_i[d] = O_i[d] + pj * Vtile[jj][d];
              }
            }
          }
          m_i = m_new;
          l_i = l_new;
        }
      }
      workgroupBarrier();
    }
  }

  // --- 4. Normalise and write ctx. ---------------------------------------
  if (q_valid && l_i > 0.0) {
    let inv = 1.0 / l_i;
    let cb = (b * T + q_row) * C + off;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
      g4[cb + d] = O_i[d] * inv;
    }
  } else if (q_valid) {
    // Q row had no unmasked K columns at all (shouldn't happen for causal
    // attention since t2 = q_row is always valid for q_row < T, but be safe).
    let cb = (b * T + q_row) * C + off;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { g4[cb + d] = 0.0; }
  }

  // Stash final m_i, l_i in workgroup memory so we can recompute attn[] in
  // a second pass without extra global memory.
  m_final[lane] = m_i;
  l_final[lane] = l_i;

  // Save L = m + log(l) so the FA2 backward kernels can reconstruct
  // P = exp(S - L) from q/k without reading the materialised attn matrix.
  // Buffer shape: g5 = L[B, H, T], one f32 per (b, h, q_row).
  if (q_valid) {
    let L_idx = (b * H + h) * T + q_row;
    if (l_i > 0.0) {
      g5[L_idx] = m_i + log(l_i);
    } else {
      // Q row had no unmasked K columns; output is 0; downstream uses
      // -inf as the marker. Match what naïve attention does in this corner.
      g5[L_idx] = -3.4e38;
    }
  }
  workgroupBarrier();

  // The second-pass attn writeback that used to live here is GONE.
  //
  // Earlier the forward kernel re-walked K blocks to materialise the full
  // [B,H,T,T] attention matrix into g3, because the existing backward
  // kernels (attn_dscores, attn_dv) read it. Now that ops.ts dispatches
  // the FA2-aware backward (attn_dscores_fa2 + attn_dv_fa2) whenever this
  // forward runs (hd ≤ 64), backward reconstructs P = exp(S − L[t1]) from
  // q/k/L on the fly — it never touches g3. So we skip the writeback
  // entirely and save BR × T mul-adds per workgroup plus the O(B·H·T²)
  // global memory traffic that used to land there. The half of the FA2
  // memory + time win that lever 10b's negative-result entry had been
  // waiting for.
  //
  // The g3 binding still exists (the shared bind layout has six storage
  // slots; ops.ts still passes the attn tensor for shape parity), it just
  // never gets touched in this path. Future work: stop allocating attn at
  // all when this kernel is the chosen forward.
}
