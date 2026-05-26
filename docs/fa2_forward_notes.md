# Flash Attention 2 — forward pass notes

Implementation log for task #47 (FA2 in WGSL), forward-only. Backward will
be a separate session — see "What's still needed" at the bottom.

Files in this change:

- `webgpu/attention_fa2.wgsl` — the new kernel (entry point `fa2_forward`).
- `tests/test_fa2_parity.mjs` — algorithm-level parity vs the naive reference.

## The algorithm

Standard causal scaled-dot-product attention is

```
S = Q Kᵀ / √d                      // [T, T]
S = causal_mask(S)                 // S[i, j] = -inf for j > i
P = softmax(S, axis=-1)            // [T, T]
O = P V                            // [T, d]
```

The existing `attn_fused_sv` in `train.wgsl` collapses softmax and the
P·V matmul into a single pass but still allocates the full row of scores
(`array<f32, 1024>`) in per-invocation private memory. At ctx=1024 that's
4 KB **per thread**, which Apple WebGPU spills to global memory.

FA2 keeps Q in shared memory, walks K and V in blocks, and maintains the
softmax state online — never materialising a full row of scores.

### Notation

- `Br = Bc = 16` — Q-row tile size and K/V-row block size.
- `m_i` — running per-row max log-score.
- `l_i` — running per-row softmax denominator.
- `O_i` — running per-row partial output (`Σ exp(s − m) · v`, length `hd`).
- `α = exp(m_old − m_new)` — rescale factor when the max changes.

### Loop (per (b, h, Q-tile))

1. Load `Qtile[Br][hd]` into workgroup-shared memory, cooperatively.
2. Each thread `lane ∈ [0, Br)` initialises `m_i = −∞`, `l_i = 0`,
   `O_i = 0` for its Q row.
3. For each K block `kj`:
   - **Causal skip.** If `kj·Bc > q_end_of_tile`, every score in this and
     all later blocks is `−∞`; break.
   - Cooperatively load `Ktile[Bc][hd]` and `Vtile[Bc][hd]`.
   - Each thread, against its Q row:
     - Compute `S[jj] = Q · Ktile[jj] · scale` for `jj ∈ [0, Bc)`,
       writing `−∞` whenever the position is out-of-range or strictly
       above the causal diagonal.
     - `m_block = max_j S[j]`, `m_new = max(m_i, m_block)`.
     - `α = exp(m_i − m_new)`; rescale `O_i *= α`, `l_i *= α`.
     - For each valid `jj`: `p = exp(S[jj] − m_new)`; `l_i += p`;
       `O_i += p · Vtile[jj]`.
     - Update `m_i = m_new`.
4. After the loop, `O_i /= l_i` and write to `ctx[b, q_row]`.
5. **Compatibility second pass.** Re-walk K blocks; for each valid
   `(q_row, t2)` compute the score and write
   `attn[b, h, q_row, t2] = exp(S − m_final) / l_final` so the existing
   backward kernels still find the attention matrix where they expect it.

The math of the online merge is the standard FA1/FA2 identity:

```
softmax(concat(A, B)) ↔ softmax-merge of the per-block (m_A, l_A) and (m_B, l_B).
```

I'm doing FA2's loop order (outer = Q-tile, inner = K-block) rather than FA1's
(outer = K-block, inner = Q-tile). The FA2 order means each thread only
writes its `O_i` to global memory once, at the very end — no atomic
accumulation across workgroups.

## Why this is better than `attn_fused_sv`

Two real wins, one that's a wash for now.

1. **Private-memory pressure goes from O(T) to O(hd).** `attn_fused_sv`
   declares `var sc: array<f32, 1024>` per invocation; FA2 only holds
   `m_i`, `l_i`, and `O_i[hd]` per thread (so 1 + 1 + hd floats, ≤ 66
   floats at hd=64 vs. 1024 in the old kernel). On Apple drivers, the
   old version provably spills; FA2 fits in registers.

2. **K and V are read once per Q tile, not once per Q row.** `attn_fused_sv`
   has one thread per Q row and each thread re-reads every K and V row
   independently. FA2 has 16 threads per Q tile cooperatively loading
   K/V into shared memory — that's a 16× reduction in global K/V reads.
   This is where the FA2 paper's "I/O complexity" win comes from.

3. **What does NOT improve until ctx ≥ 256:** for tiny ctx (Tiny preset has
   ctx=32, Trial has ctx=64), the existing kernel already fits comfortably
   and the dispatch overhead of a per-tile workgroup dominates. Expected
   crossover is around ctx=128–256, with the gap widening at Mega
   (ctx=512) and the projected Behemoth path (ctx ≥ 1024).

Per AGENTS.md "Safety rules for heavy GPU / compile loops" I did **not**
run a benchmark sweep to measure the crossover. That's a separate
single-shot measurement session once the kernel is integrated.

## Parity verification

`tests/test_fa2_parity.mjs` checks the algorithm (not the WGSL — see
caveats) against the naive `softmax(causal(QKᵀ/√d)) V` reference, in
plain JS. Six shapes, including:

- T = 16 (one full Q tile, T == Br, T % Bc == 0)
- T = 8 (single partial tile)
- T = 20 (boundary K block has the mask)
- T = 256 (ctx=256, the regime where FA2 starts to pay off)
- hd = 64 (Behemoth-shaped head dim, equal to MAX_HD)

Result: all twelve checks pass, with `max |ctx − ctx_ref| ≈ 5.96e-8`
(≈ 1 f32 ULP — comes from the order in which `exp(·) · v[d]` terms are
summed) and `max |attn − attn_ref| = 0` (the second-pass attn writeback
uses the same summation order as the naive reference, so it's bit-exact).

```
$ node tests/test_fa2_parity.mjs
ok   fa2 ctx   [B=1 T=16 C=32 H=4 hd=8]   maxAbsDiff=5.96e-8
...
ok   fa2 attn  [B=1 T=256 C=64 H=2 hd=32] maxAbsDiff=0.00e+0
ALL PASS
```

### Why not a live WGSL parity dispatch

Node doesn't ship WebGPU; every existing WGSL parity test in this repo
runs through a headed Chromium + Playwright (`browser/src/webgpu-test.ts`
driven by `browser/webgpu_test.mjs`). Spinning that harness up just to
add one assertion would also require touching `ops.ts` to expose the new
kernel — and per the partition for this turn, both of those files are
off-limits (another agent is working in parallel).

So the live WGSL parity assertion will go into `webgpu-test.ts` in the
integration session, alongside the `ops.ts` wiring. The block to add
mirrors the existing `// --- attention (stage 3) ---` section:

```ts
// Add to browser/src/webgpu-test.ts after the existing attention check.
{
  const B = 2, T = 48, C = 24, H = 3;     // exercise multi-block walk
  const q = rand(B*T*C), k = rand(B*T*C), v = rand(B*T*C);
  const qt = GpuTensor.fromData(dev, q);
  const kt = GpuTensor.fromData(dev, k);
  const vt = GpuTensor.fromData(dev, v);
  const fa2 = ops.attentionForwardFA2(qt, kt, vt, B, T, C, H);   // new method
  const ref = ops.attentionForward(qt, kt, vt, B, T, C, H);      // existing FA1-style
  check("fa2 forward ctx",  maxError(await fa2.ctx.download(),  await ref.ctx.download())  < 1e-4, "");
  check("fa2 forward attn", maxError(await fa2.attn.download(), await ref.attn.download()) < 1e-4, "");
}
```

Manual smoke test in the meantime: open the page in Chrome with
WebGPU enabled, hit the existing "WebGPU kernel parity tests" route
(`/webgpu-test.html`) after wiring, and confirm `ALL PASS`. The
end-to-end gate stays `tests/test_webgpu_train.mjs` (50 WASM steps vs
50 WebGPU steps with the same seed, <5% loss drift).

## Caveats / known issues

- **`MAX_HD = 64`.** Workgroup memory is sized at compile time, so we
  bound `hd` at 64. Every current preset is well within that
  (byte-tinygpt-v0 = 32, small/Behemoth = 64). Bumping this needs both
  a constant change and a re-verification against
  `device.limits.maxComputeWorkgroupStorageSize` — at 64 we sit at
  3 × 16 × 64 × 4 = 12 KB of workgroup memory; at 128 we'd hit 24 KB,
  which is over Apple's 16 KB default on some drivers.
- **T does not need to be a multiple of 16.** Out-of-range columns get
  `-∞` scores and are dropped by the softmax. Verified by the T=20 case.
- **The second `attn` pass costs roughly 1× the score work over again.**
  That's the price of keeping the existing backward kernels unchanged.
  Once FA2 backward is shipped (next session), this pass can be deleted
  and the kernel becomes purely log-sum-exp + ctx out.
- **No dropout, no relative-position bias.** This project's model uses
  neither, so neither is implemented.
- **Subgroup primitives not used.** A future pass could use
  `subgroupShuffle` to skip the workgroup barrier between K-block
  loads and score computation, but the basic kernel doesn't need it.
  Existing subgroup-using kernels live in `train_sg.wgsl` — see the
  pattern there.

## What's still needed for full FA2

This delivery is **forward only**. To finish FA2 the next session needs:

1. **Backward kernel with recomputation.** The standard FA2 backward
   recomputes attention on the fly from Q, K, V and the saved log-sum-exp
   `L = m_final + log(l_final)`. We'd write `L` instead of the full attn
   matrix in the forward, then the backward walks K/V blocks twice:
   once to recompute `P` for the dV contribution, once to compute `dS`
   and accumulate dQ / dK. The existing `attn_dscores`, `attn_dq`,
   `attn_dk`, `attn_dv` kernels can be retired.
2. **Drop the second `attn` writeback pass.** Once (1) lands, the FA2
   forward stops writing `attn[B,H,T,T]` entirely — saves O(B·H·T²) of
   global memory, which is the other half of the FA2 memory win.
3. **Integration into `ops.ts`.** Add `attentionForwardFA2` (and once
   backward lands, switch `attentionForward` itself to it). Dispatch:
   workgroup count `x = ceil(T / 16)`, `y = B · H`.
4. **End-to-end gate.** Re-run `tests/test_webgpu_train.mjs` and the
   `webgpu-test.ts` overfit gate (150 training steps on a tiny model
   driving loss from ~ln(256) to <0.5) after integration.
5. **Single-shot benchmark.** One Mega-preset (ctx=512) step on FA1 vs
   FA2; the win should be visible in step time. Document it like the
   vec4 writeup in `browser/devlog.html`. **Do not loop** —
   AGENTS.md "Safety rules" applies.

If the next agent gets a NaN out of the backward, the first thing to
check is the `L = m + log(l)` reconstruction at extremely-negative `m`:
when an entire row was masked (shouldn't happen for causal attention with
`q_row < T`, but the boundary tile has `q_row ≥ T` lanes that produce
`l = 0`), `L = -∞` and downstream `exp(s − L)` blows up. The forward
kernel already returns zero ctx for those lanes; the backward needs to
match.
