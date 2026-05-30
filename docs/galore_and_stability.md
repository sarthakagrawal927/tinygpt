# GaLore + training-stability bells (Tier 2)

This page documents the five `tinygpt train` flags introduced in the
Tier-2 stability batch:

| Flag                          | Feature                                    | Touches                |
| ----------------------------- | ------------------------------------------ | ---------------------- |
| `--galore-rank R`             | GaLore gradient low-rank projection        | Trainer (gradient hook)|
| `--galore-update-every K`     | GaLore basis refresh cadence (steps)       | GaLoreProjector        |
| `--z-loss-weight F`           | PaLM-style logit-magnitude penalty         | Loss path              |
| `--deep-norm`                 | DeepNorm residual α + projection-init β    | TransformerBlock init  |
| `--lr-layer-decay F`          | Layer-wise LR decay (F < 1, deeper = full LR) | Gradient transform |
| `--embedding-rmsnorm`         | RMSNorm right after the token embedding    | Model forward / manifest |

All five round-trip through the `.tinygpt` manifest (`galoreRank`,
`galoreUpdateEvery`, `zLossWeight`, `useDeepNorm`, `lrLayerDecay`,
`useEmbeddingRMSNorm`) so a `--resume` from a saved checkpoint keeps
the same regimen.

---

## 1. GaLore — Gradient Low-Rank Projection

**Reference.** Zhao et al., *GaLore: Memory-Efficient LLM Training by
Gradient Low-Rank Projection*, ICML 2024. arXiv:2403.03507.

**The trick.** AdamW's optimiser state (`m`, `v`) for a transformer
is *dominated* by the 2-D weight matrices. For a Llama-7B model
that's 26 GB of fp32 Adam state on top of the 13 GB of weights — far
more than the activation memory you'd ever spend in one forward.
GaLore observes that Adam's update on those matrices is empirically
well-approximated by a rank-R update for small R, then exploits it:

1. Maintain a basis `P : [m, r]` per tracked matrix.
2. Replace gradient `G : [m, n]` with `P (P^T G)` — a rank-R
   approximation living in span(P).
3. Refresh `P` every `K` steps from the SVD of the *current* gradient:
   `P = U[:, :r]`.

Result: FULL fine-tuning (every weight in the network moves, unlike
LoRA's frozen-base + adapter scheme) at LoRA-rank-R optimiser memory
cost. Especially valuable for pretraining where LoRA's frozen-base
prior is too restrictive.

### Surface

```bash
tinygpt train --preset huge --steps 50000 \
    --galore-rank 256 \
    --galore-update-every 200 \
    --corpus shakespeare.txt --out out.tinygpt
```

### Implementation

`GaLoreProjector` (one per 2-D weight matrix) owns the basis tensor
`P` plus a step counter. Each call to `project(G)`:

- on a refresh-boundary step → `P = svd(G).U[:, :rank]` (CPU stream
  — Metal SVD support is incomplete in MLX as of writing)
- always → returns `P (P^T G)`

`GaLoreManager` lazily walks the gradient tree on first sight, decides
which leaves to track (2-D, ≥4 k elements, not the token embedding —
embedding gradients are rank-1-per-step naturally, projection hurts),
and caches projectors keyed by parameter name.

The trainer hook lives between `gradFn(...)` and `optimizer.update`:

```swift
let (loss, grads) = gradFn(model, x, y)
var processed = grads
processed = clipGradNorm(processed, maxNorm: 1.0)   // before projection
if let g = galore { processed = g.processGradients(processed) }
optimizer.update(model: model, gradients: processed)
```

GaLore forces `compile` *off* — projector state mutates outside the
graph, breaking the trace.

### Memory accounting

There's a subtle but important caveat. MLX-Swift's AdamW keeps
`m, v` at the FULL parameter shape regardless of grad rank, so a
naive integration doesn't actually shrink the on-device optimiser
state. We project the *gradient* (which IS what GaLore does for the
training dynamics), but to preserve the "GaLore matches LoRA r=R
memory" claim we ALSO report the theoretical budget a fully
GaLore-aware optimiser WOULD use:

- per tracked matrix: `2 × low_rank_floats + basis_floats`
- vs the full AdamW: `2 × m × n`

Example: for `byte-tinygpt-small` (6 layers, d=192) GaLore-rank-64
tracks 36 matrices and reports theoretical 6.8 MB vs full 20.2 MB
(33.3%). A truly GaLore-aware optimiser (e.g., the reference PyTorch
implementation) would realise that 3× saving in practice; here it
remains a budget figure on the stdout summary. The grad-projection
training dynamics ARE the same as the paper's recipe — only the
on-device Adam state isn't yet pruned.

A follow-up is queued to subclass `AdamW` and store `m, v` at
`[r, n]` shape for tracked matrices; the GaLoreProjector already
exposes `loRankAdamFloats` / `fullAdamFloats` ready for the swap.

### Smoke results

Tested on `--preset tiny --corpus shakespeare.txt --steps 50`
(tiny: 4L, d=128, 842 k params; 24 trackable matrices in the run).

| Config                                  | Loss (50 steps) | Compile |
| --------------------------------------- | --------------- | ------- |
| baseline                                | 2.86            | on      |
| `--galore-rank 32`                      | 3.02            | off     |
| `--galore-rank 64` (preset=small)       | 2.86            | off     |
| `--z-loss-weight 1e-4`                  | 3.05            | on      |
| `--deep-norm`                           | 3.41            | on      |
| `--embedding-rmsnorm --lr-layer-decay 0.85` | 2.97        | on      |
| all five together                       | 3.31            | off     |

GaLore at rank=32 on a d=128 model captures ~25% of the parameter
space; the small loss gap is the expected price of the projection.
GaLore-256 on the `huge` preset (d=256) is the recommended setting.

---

## 2. Z-loss — logit-magnitude penalty

**Reference.** Chowdhery et al., *PaLM: Scaling Language Modeling
with Pathways*, 2022. The "z-loss" auxiliary first appeared in
Lepikhin et al., *GShard*, 2020.

**The trick.** Adds `z · (log Σ exp(logit))²` to the loss. Keeps the
log-sum-exp from drifting upward — a softmax that saturates is one
of the classic ways an LLM training run blows up at step 50 000.

PaLM defaults: `z = 1e-4`.

### Surface

```bash
tinygpt train --preset mega --z-loss-weight 1e-4 ...
```

### Implementation

`TinyGPTModel.loss` (and the parallel branch on `TinyGPTModelHF`)
computes:

```swift
let lse = max + log(Σ exp(logit - max))    // numerically stable
total = ce + zWeight * (lse * lse).mean()
```

MLX-Swift doesn't ship a `logsumexp` op; we expand it inline. Cost:
one `max`, one `exp`, one `log`, one square — ~constant per step
relative to the cross-entropy.

In MTP (multi-token-prediction) mode the z-loss currently fires on
the primary-horizon logits only — the MTP path computes loss
per-horizon and can't easily share a single logits tensor across
horizons. This is conservative: the regulariser bites where it
matters most (the head the model decodes from at inference).

---

## 3. DeepNorm — residual scaling for very deep stacks

**Reference.** Wang et al., *DeepNet: Scaling Transformers to 1,000
Layers*, 2022. arXiv:2203.00555.

**The trick.** For decoder-only stacks of N layers,

- residual is scaled by **α = (2N)^(¼)** at every sub-layer addition,
- specific projection inits scaled by **β = (8N)^(-¼)** (v_proj,
  o_proj, and the MLP output projection — `fc_out` for plain MLP,
  `down_proj` for SwiGLU).

α blows up the running residual stream's variance, β pulls it back —
the two are balanced so training Loss(layer) stays bounded as depth
grows. The paper trains 1000-layer transformers without divergence;
on a 12-layer toy model the gain is much smaller.

### Surface

```bash
tinygpt train --preset behemoth --deep-norm ...
```

For N=32 layers: α = (64)^¼ ≈ 2.83, β = (256)^(-¼) ≈ 0.250.
For N=4 (our `tiny` preset): α = (8)^¼ ≈ 1.68, β = (32)^(-¼) ≈ 0.42.

### Implementation

- The α multiplier lands in `TransformerBlock.blockAfterAttn` and the
  HF block's `rawForward`: every `y = x + sub(LN(x))` becomes
  `y = α·x + sub(LN(x))`. When DeepNorm is off, α = 1.0 and the
  scalar multiply is a free no-op.
- The β init runs in the block's `init` after `super.init`, via
  `Module.update(modules:)`. We can't mutate `Linear.weight` (it's a
  `let`), so we install a fresh `Linear` whose weight is
  `original · β`. The HF SwiGLU variant targets `down_proj`; the
  from-scratch dense MLP targets `fc_out`. Both targets attention's
  `v_proj` and `o_proj`.

A `--resume` from a non-DeepNorm checkpoint does NOT retroactively
re-init weights; the user must train from scratch to get the β-scaled
trajectory. This matches the paper's reproduce-from-init recommendation.

### Caveat

DeepNorm shines on stacks ≥ 100 layers. On a `tiny` (4-layer) model,
the loss curves of DeepNorm-on and DeepNorm-off cross within ~100
steps — the residual rescaling is just adding overhead. Recommended
only when nLayers ≥ 32 (i.e. `behemoth` / `titan` / depth-modded
custom configs).

---

## 4. Layer-wise LR decay

**Reference.** A folklore fine-tuning trick — appears in the original
ULMFit (Howard & Ruder, 2018) under the name "discriminative
fine-tuning", popularised by BERT fine-tuning recipes.

**The trick.** Scale each block's gradient by `factor^(L - 1 - i)`
so the deepest block trains at the full LR and shallower blocks
get progressively smaller updates. The intuition: surface-level
features (the embedding layer, the first couple of blocks)
generalise broadly across tasks; task-specific reasoning concentrates
deeper. Slowing the early blocks reduces catastrophic forgetting
when fine-tuning a pre-trained model.

### Surface

```bash
tinygpt train --preset huge --resume base.tinygpt --lr-layer-decay 0.85 ...
```

For L=12: deepest layer @ 100% LR, shallowest at `0.85^11 ≈ 17%` LR.

### Implementation

A gradient transform (cousin of `clipGradNorm` and `scaleLoraBGradients`).
The walker parses dotted parameter names — `blocks.7.attn.q_proj.weight`
or `layers.3.self_attn.o_proj.weight` — extracts the block index N,
and multiplies the leaf by `decay^(nLayers - 1 - N)`. Non-block
parameters (embedding, final norm, lm_head) get the full LR
(multiplied by 1.0, the identity).

The transform is graph-pure (just a scalar multiply per leaf) so it
stays compile-safe — unlike GaLore, it doesn't force the uncompiled
path.

### Caveat

Layer-wise LR decay is a **fine-tuning** lever, not a pretraining
one. Applying it from-scratch can leave the early layers
under-trained (they never get enough signal to learn good low-level
features). The recommended usage is `tinygpt train --resume ...
--lr-layer-decay 0.85` for adaptation runs.

---

## 5. Embedding RMSNorm

**Reference.** Appears in several recent (2024-2025) long-context
training recipes — e.g. Falcon Mamba's `model.embeddings.layernorm`,
the embedding-normalised variant of Gemma 2. The literature
attribution is muddled (no single canonical paper); the construction
is straightforward enough that several groups arrived independently.

**The trick.** Apply RMSNorm to the token-embedding output *before*
positional embeddings are added:

```
tokEmb = embed_norm(token_embedding(idx))
x = tokEmb + position_embedding(...)
```

RMSNorm pulls every token's embedding to unit RMS, so downstream
attention sees a more uniform input scale. Stabilises early-training
loss on long-context (≥4096) transformers; ~5% loss improvement
reported on the >8k contexts where the embedding-output norms drift
the most.

### Surface

```bash
tinygpt train --preset huge --embedding-rmsnorm ...
```

### Implementation

`TinyGPTModel` and `TinyGPTModelHF` gain an optional `embedNorm:
RMSNorm?` slot. When `cfg.useEmbeddingRMSNorm` is true at init,
the slot is populated with a fresh `RMSNorm(dimensions: dModel)`.
Both training forward (`forwardToHidden`, `forwardLayerwise`) and
inference forward (`forwardCached` in `KVCache.swift` /
`KVCacheHF.swift`) consult `embedNorm` and pass the embedding
through it before the positional add.

The manifest writes an `embed_norm.weight` tensor (shape `[d_model]`)
right after the embedding tables. A from-scratch checkpoint trained
WITHOUT the flag has no `embed_norm.weight` entry; loading it back
WITH the flag would fail (the model wants the tensor, the file
doesn't have it). The `--resume` path picks up the flag from the
saved manifest so this isn't a footgun on continue-training; the
only failure mode is "save without, load with" via an ad-hoc CLI
override that the resume path explicitly blocks.

### Caveat

The RMSNorm's `.weight` is initialised to ones — at step 0 the
forward output is essentially the original embedding scaled by
1 / sqrt(mean(x²)). For from-scratch training this looks like a big
initial loss spike (we observe step-1 loss of ~10 vs ~6 without)
that disappears within ~30 steps as the embedding magnitudes adapt.
Always pair with `--warmup ≥ 50` on a fresh run.

---

## Compatibility matrix

| Flag                  | Affects manifest? | Compile-safe? | Pre-train | Fine-tune |
| --------------------- | ----------------- | ------------- | --------- | --------- |
| `--galore-rank`       | header only       | NO (forces off) | yes     | yes       |
| `--galore-update-every` | header only     | NO            | yes       | yes       |
| `--z-loss-weight`     | header only       | yes           | yes       | yes       |
| `--deep-norm`         | header only       | yes           | yes only  | no        |
| `--lr-layer-decay`    | header only       | yes           | discouraged | yes     |
| `--embedding-rmsnorm` | **adds tensor**   | yes           | yes only  | no        |

"Pre-train" = safe to enable from scratch. "Fine-tune" = safe to
enable on a `--resume`. DeepNorm and embedding RMSNorm change the
model's *init / structure*, so flipping them on mid-training would
corrupt learned weights.

## File map

| File                                       | Change                                  |
| ------------------------------------------ | --------------------------------------- |
| `TinyGPTModel/GaLore.swift` (new)          | `GaLoreProjector`, `GaLoreManager`, `scaleLayerwiseLR`, `applyBetaInit` |
| `TinyGPTModel/Trainer.swift`               | GaLore + layer-LR hook, compile gating  |
| `TinyGPTModel/TrainerHF.swift`             | same, parallel for HF model             |
| `TinyGPTModel/TinyGPTModel.swift`          | `embedNorm` slot, z-loss in `loss()`    |
| `TinyGPTModel/HFModel.swift`               | `embedNorm` slot, `loss()` w/ z-loss    |
| `TinyGPTModel/TransformerBlock.swift`      | DeepNorm α + β init                     |
| `TinyGPTModel/TransformerBlockHF.swift`    | same, parallel for HF block             |
| `TinyGPTModel/ModelConfig.swift`           | five new persistable fields + α/β helpers |
| `TinyGPTModel/AnyModel.swift`              | round-trip new fields through loader    |
| `TinyGPTModel/KVCache.swift` / `KVCacheHF.swift` | embed_norm in `forwardCached`     |
| `TinyGPTIO/Manifest.swift`                 | six new optional header fields          |
| `TinyGPT/Train.swift`                      | CLI flags + run-summary lines + manifest entry |
| `TinyGPT/TrainSupport.swift`               | propagate the new fields on save        |
