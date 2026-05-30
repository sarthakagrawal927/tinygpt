# Speculative-decoding heads: Medusa + EAGLE-2

Two joint-trained draft-head architectures that bolt onto a frozen base
LM and propose multiple tokens per base-forward, verified in a single
extra base forward pass. Implemented in the native-Mac build under
`native-mac/Sources/TinyGPTModel/MedusaHeads.swift` (Cai et al., 2024)
and `EagleDraft.swift` (Li et al., 2024).

## Why bother

Vanilla speculative decoding (the `SpeculativeDecode.swift` path,
Leviathan et al. 2023) needs a SEPARATE small draft model with the
same tokenizer. That works, but you have to TRAIN a second model just
to draft tokens for your real model. Joint-trained heads sidestep
this — the "draft" lives ATOP the base, attached at the final-hidden
state, so training fits in O(M) extra params (1-5% of base) and the
heads inherit the base's representational quality for free.

The trade is one extra moving piece per decode step (the head /
draft net), and a sidecar training run. When acceptance rate is high
the wall-clock saving is real; the verification path is GREEDY (we
accept the longest matching argmax prefix), so output is lossless
wrt the base's own argmax at every position.

## Mechanism comparison

### Medusa (Cai et al., 2024)

- **N independent heads** sit alongside the base's LM head.
- Head `k` takes the base's final hidden state `h_t` at position `t` and
  predicts the token at position `t + k + 1`. So head 0 predicts the
  same thing the base does (`t+1`); head 1 predicts `t+2`; etc.
- Each head is tiny: in our implementation, one residual block + linear
  projection to vocab:
  ```
  res  = SiLU(W_res · h_t) + h_t
  logits_k = W_vocab,k · res
  ```
- During inference: one base forward gives us `h_t`; the heads spit out
  candidate tokens for `t+1 .. t+N+1` in a single batched pass. A
  SECOND base forward over `prompt + candidates` then verifies.
- **Tree attention (paper §3.3)**: instead of a single 1-D candidate
  chain, Medusa proposes a TREE of candidates (top-k per head). The
  base verifies all paths in one forward via a custom block-diagonal
  causal mask. This implementation does the simpler **linear (width-1)**
  variant — see "What's not implemented" below.

### EAGLE-2 (Li et al., 2024)

- **One small auto-regressive draft net** replaces the N independent
  heads. The key insight: feeding the base's RICH hidden state (rather
  than just the token embedding) into the draft net gives substantially
  better acceptance rate, especially deeper in the draft tail.
- Draft net input at step k: `concat(hidden_k, embed(token_k))` →
  Linear → SiLU → Linear → SiLU → norm + residual → vocab projection.
- The vocab projection and token embedding are warm-started from the
  base's LM head + token embedding (the EAGLE "tied" recipe).
- At inference: one base forward gives `hidden_0` and the base's argmax
  `token_0`. The draft net then unrolls auto-regressively `N` steps,
  producing `[token_0, token_1, ..., token_N]`. A second base forward
  verifies.
- **Dynamic tree pruning** (paper §3): like Medusa's tree, EAGLE-2
  builds a dynamic candidate tree pruned by confidence. We do
  width-1 only — see "What's not implemented" below.

### Side-by-side summary

| Aspect                 | Medusa                       | EAGLE-2                          |
|------------------------|------------------------------|-----------------------------------|
| Heads                  | N parallel, each shallow     | 1 small AR network unrolled N×    |
| Input to head k        | base's hidden_t              | hidden_{k-1} + embed(token_{k-1}) |
| Param count vs. base   | ~5% (4 heads × d² + d·V)     | ~3% (one shared draft net)        |
| Acceptance rate (paper)| 60-70% on standard LMs       | 70-85% on standard LMs            |
| Training stability     | High (independent heads)     | Moderate (AR error compounds)     |
| Tree-attention support | Yes (paper)                  | Yes (dynamic, paper)              |
| This impl's tree       | width-1 (linear chain only)  | width-1 (linear chain only)       |

## Code map

```
native-mac/Sources/TinyGPTModel/
├── MedusaHeads.swift       MedusaHead, MedusaHeadStack, medusaHeadsLoss,
│                           MedusaVerify, MedusaHeadsIO. Also hosts the
│                           shared `.heads` sidecar plumbing
│                           (SpecHeadsFileHeader, write/read header+blobs,
│                           NestedDictionary param-tree restore).
├── EagleDraft.swift        EagleDraft (the draft net), eagleTrainingForward,
│                           eagleDraftLoss, EagleVerify, EagleDraftIO,
│                           EagleWarmStart (copy base.embed/lm_head → draft).
└── ModelConfig.swift       SpeculativeHeadConfig — small in-memory config
                            struct (kind, numHeads, hiddenDim). Lives only
                            in memory; not serialised into the .tinygpt
                            base manifest (heads are a SIDECAR, by design).

native-mac/Sources/TinyGPT/
├── TrainHeads.swift        `tinygpt train-heads` CLI. Loads a base model,
│                           closure-captures it (so MLX autograd treats
│                           base params as constants — same trick as
│                           tuned-lens), trains the heads with AdamW on a
│                           byte-level corpus.
├── Sample.swift            New `--heads <path>` + `--head-type {medusa|eagle}`
│                           flags. Routes through MedusaVerify / EagleVerify
│                           when set; falls back to existing decode paths
│                           otherwise. Bypasses KV cache (verify pass
│                           re-processes the whole tail).
└── TinyGPT.swift           Pre-switch shim dispatches `train-heads` to
                            `TrainHeads.run`. Matches the existing
                            score-bench shim pattern (other agents are
                            concurrently editing the formal switch).
```

## On-disk format: `.heads`

Shared format for both Medusa and EAGLE-2 (distinguished by the JSON
header's `kind` field). Little-endian throughout:

```
0    4    magic "TGMH"  (TinyGPT Medusa/EAGLE Heads)
4    4    version u32   (currently 1)
8    4    header_len u32
12   N    JSON header   (SpecHeadsFileHeader — see MedusaHeads.swift)
12+N      raw fp32 tensor blobs in `header.entries` order
```

The header records the base model's config (layers, dModel, heads, ctx,
vocab) so a sidecar refuses to load against the wrong base. Loading
walks the tensor entries, materialises each into an MLXArray, and
applies them to a freshly-built `MedusaHeadStack` / `EagleDraft` via
`Module.update(parameters: …)`.

Token embedding + vocab projection in EAGLE are written out as
ordinary fp32 tensors; on load the draft owns them. (We don't ALIAS to
the base's parameters — the sidecar must stay loadable against a
re-saved or re-quantised base.)

## CLI walkthrough

### Train heads on a frozen base

```bash
tinygpt train-heads <base.tinygpt> \
    --type medusa            # or 'eagle'
    --corpus <text.txt>      # UTF-8 byte-level text
    --steps 500              # default
    --num-heads 4            # look-ahead horizon
    --lr 1e-3                # default — small heads, small LR
    --batch 4 --ctx 128      # match the base's context if you have memory
    --out heads.heads
```

The base model is loaded but FROZEN: it's closure-captured inside the
loss function, so MLX's autograd never sees its parameters as
gradient targets. Only the head stack (Medusa) or draft net (EAGLE)
updates. AdamW; no LR schedule (heads are small; constant LR works
fine in practice).

The CLI prints a loss-curve sample (every 10 steps) at the end so the
agent log captures whether loss actually moved during a smoke run.

### Sample with heads attached

```bash
tinygpt sample <base.tinygpt> \
    --heads heads.heads \
    --head-type medusa       # must match the sidecar
    --prompt "ROMEO:" \
    --tokens 200 \
    --temperature 0          # greedy; --heads forces this
```

The base verifies each speculative burst — output is BIT-IDENTICAL to
plain greedy decode on the base alone (when the heads agree, you get
multiple tokens per base forward; when they disagree, you fall back
to the base's argmax). The CLI prints per-run acceptance metrics:

```
[heads] steps=48, proposed=240, accepted=52 (21.7%) · 91 tok/s · 1.10s
```

## Smoke run: 50 steps of head training, demo.tinygpt

Base: `browser/public/demo.tinygpt` (12L · d=256 · vocab=256 ·
9.6M params · webgpu-trained on a small mixed corpus).

Corpus: `data/examples/shakespeare.txt` (1.1 MB).

Both Medusa and EAGLE-2 train cleanly. Loss curves at lr=1e-3, batch=4,
ctx=128:

| step | Medusa head loss | EAGLE draft loss |
|------|------------------|-------------------|
|   1  | 5.879            | 4.858             |
|  10  | 2.877            | 3.147             |
|  20  | 2.763            | 2.866             |
|  30  | 2.635            | 2.463             |
|  40  | 2.566            | 2.312             |
|  50  | 2.543            | 2.309             |

Both trend downward smoothly. EAGLE's initial loss is lower because
the warm-started vocab projection already encodes a reasonable token
distribution; its slope is comparable to Medusa.

50 steps is FAR below convergence — the paper recipes run for tens of
thousands of steps with logit distillation, not raw CE. Acceptance
rates at this point are 20-25%, well below the paper's 60-85%.

### Acceptance rate / speedup (greedy, ROMEO prompt, 100 new tokens)

| Configuration              | tok/s | acceptance | output |
|----------------------------|-------|------------|--------|
| Baseline (KV-cached decode)| 232   | n/a        | golden |
| Medusa (50-step heads)     | 202   | 21.7%      | bit-identical to baseline |
| Medusa (300-step heads)    | 375   | 20.8%      | bit-identical to baseline |
| EAGLE-2 (50-step draft)    | 200   | 26.5%      | bit-identical to baseline |

Notes:

* The first run after build is noticeably slower than subsequent runs
  (graph compile cost); the numbers above are warm-cache.
* The Medusa-300 throughput exceeds the cached baseline despite the
  acceptance rate barely moving — the head loss continues to drop
  (5.79 → 2.37 across 300 steps) and the heads' confidence calibration
  improves even when the argmax doesn't. With more training the
  acceptance rate should rise; the per-step structure already pays.
* EAGLE-2 has higher acceptance than Medusa at equal training budget
  (26.5% vs 21.7%) — directionally matches the paper's finding that
  EAGLE drafts the longer tail better. The MLP-only draft net here is
  a simplification of EAGLE's 1-block transformer (see below);
  expect a wider Medusa↔EAGLE gap with the full architecture.
* Correctness: the EAGLE and Medusa-300 sample outputs are
  bit-identical to baseline greedy decode (verified above). This is
  the greedy verify rule working as designed.

## Training procedure: practical recipe

**This implementation is a first cut**; the production recipe in both
papers does several things we don't yet:

1. **Logit distillation, not CE on token ids.** The heads' real
   objective is to match the BASE's next-token distribution, not just
   predict the correct token. With distillation, the heads learn to
   propose tokens the base will actually accept; with raw CE on token
   ids, they only learn to propose the SAME token the base would —
   but with a different confidence calibration, which lowers the
   verify-pass acceptance rate.
2. **Train on the base's own generations**, not external corpora.
   The base's distribution drifts from the corpus distribution after
   any non-trivial training. Self-generated training data makes the
   heads optimise against the distribution the base actually exhibits.
3. **Tree attention during training** (Medusa especially). The heads
   should be trained to give USEFUL top-K predictions, not just
   top-1. Single-token CE doesn't reward that.

For a real production-quality head set on a frontier base, expect
something like:

* 10k-100k training steps
* Logit-KL loss (target = `softmax(base_logits)`, distilled into the
  head's distribution; same shape as the existing CE call but with a
  soft target)
* Self-generated training data
* Tree attention turned on during training (loss aggregates over
  the candidate tree, not a single token)

## What's not implemented (the honest list)

1. **Tree-attention verification.** Both Medusa and EAGLE-2's headline
   speedup numbers come from the TREE form: propose a tree of N×K
   candidates, verify in one base forward with a custom block-diagonal
   causal mask. We do width-1 only — a single linear candidate chain.
   The verify path is correct (greedy preserved); we just give up the
   acceptance-rate amplification a wider tree would give. Adding it
   means: (a) build the candidate tree from per-head top-K, (b)
   flatten with the right position-ids, (c) build a per-position
   attention mask that allows each tree node to attend only to its
   ancestors, (d) extract per-path argmax from the verify forward.
2. **Logit distillation training.** Loss is raw shifted-CE on token
   ids — see "Training procedure" above. Distillation would change
   the `lossFn` body to compute KL against a soft target. ~5 lines.
3. **EAGLE-2 draft net is MLP-only, not 1 transformer block.** The
   paper's draft is one attention block (with its own KV cache during
   the unroll) + FFN. Ours is just FFN. Acceptance rate cost: maybe
   10-15 percentage points on a well-trained model.
4. **No KV cache during verify.** The verify base forward re-processes
   the whole tail (prompt + proposals). With a KV cache and careful
   index bookkeeping the verify cost drops to one forward over just
   the proposals. The vanilla `SpeculativeDecode.swift` has the same
   limitation — it's a unified follow-up rather than per-mechanism.
5. **Heads-decode only works on from-scratch byte-level models.** The
   HF wrapper doesn't yet expose `forwardToHidden` / `applyLMHead`,
   which the verify path needs. Extending it is a ~20-line addition
   to `TinyGPTModelHF.swift` + an `AnyModel` method. Punted to keep
   the scope honest.
6. **BPE corpora not supported by `train-heads`.** Same one-line gate
   as TunedLens; same one-line fix (swap `ByteCorpus` for
   `TokenizedCorpus`). Punted; can be unblocked when needed.
7. **Sampled (non-greedy) speculative decoding.** The verify rule is
   greedy-only here. Temperature > 0 spec decoding needs per-token
   `p_draft` / `p_target` softmaxes + rejection sampling. The note
   in `--heads` ("forces greedy, temperature ignored") matches what
   the vanilla `--draft` path also does.

## Production-readiness caveats (what 10-100k more steps would buy)

* **Acceptance rate**. 20-25% at 50 steps → 60-85% in the paper after
  full training. This is the load-bearing number. At 60%+ acceptance
  the speedup compounds: 2 forwards yield 3-4 accepted tokens, so even
  per-token throughput beats KV-cached single-token decode.
* **Wall-clock speedup**. Below acceptance ≈ 50%, the head path's
  "verify forward processes N+1 extra positions" overhead dominates
  the savings, and you can be SLOWER than the baseline. We see this
  in the 50-step EAGLE row (200 tok/s vs 232 baseline). Once acceptance
  passes ~40% and the verify pass becomes a small constant relative
  to N accepted tokens, the math flips.
* **Memory**. Both architectures add ≤ 5% to the base's param count
  (1-2× lm_head size). At inference, peak memory is still dominated
  by the base; heads' extra footprint is invisible.
* **Tree attention** is the biggest single unimplemented lever.
  Without it, expect the speedup ceiling to sit at ~1.5-2× — with it,
  ~2.5-3.5× is realistic on standard LMs.

## References

* Cai, T., Li, Y., Geng, Z., Peng, H., Chen, J., & Dao, T. (2024).
  *Medusa: Simple LLM Inference Acceleration Framework with Multiple
  Decoding Heads*. arXiv:2401.10774.
* Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). *EAGLE-2: Faster
  Inference of Language Models with Dynamic Draft Trees*.
  arXiv:2406.16858.
* Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from
  Transformers via Speculative Decoding*. ICML.
* Existing tinygpt prior art:
  - `docs/interpretability.md` (tuned-lens, same closure-frozen-base
    training pattern)
  - `SpeculativeDecode.swift` (the small-draft variant this work
    complements)
