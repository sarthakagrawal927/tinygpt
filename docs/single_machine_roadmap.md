# Single-machine techniques — ROI-ranked + exhaustive landscape

A complete inventory of model techniques that **run on one Mac (or in a
browser tab)**, with rough what-it-is explanations and ROI ranking for
TinyGPT specifically.

## How to read this

**Filter** — single-machine only. Anything requiring a GPU cluster
(ZeRO/FSDP, tensor parallelism, large RLHF runs) is in the "skip"
section at the bottom.

**Two views of the same landscape:**

- **Tiers 1-4** are the ROI ranking of *training-or-product-shaping*
  techniques (what to build next). Higher tier = better ROI for us.
- **The category sections** that follow Tier 4 are an *exhaustive
  taxonomy* of everything else — optimizers, data, interpretability,
  browser perf, etc. — with shorter explanations and no ROI tier
  because they're orthogonal to the main pipeline.

**Status legend:**

- 🟢 shipped · 🟡 partial · ⬜ not yet built · 🟣 considered + parked

---

## TL;DR — what to read first

This doc is the master plan for what to build, what to skip, and what
can't be built right now. Six parts:

| Part | What it covers |
|---|---|
| **Part 1** | Tier 1-4 ROI ranking of techniques worth building |
| **Part 2** | Orthogonal categories — optimizers, data, tokenization, interpretability, browser perf |
| **Part 3** | The top-10 recommended order |
| **Part 4** | Open-source datasets we'd actually use (pretrain / SFT / DPO / eval) |
| **Part 5** | Recent research, 2024-2026 (web-verified, with arxiv links) |
| **Part 6** | **The phased roadmap** — 7 weeks of sequenced work, ready to execute |
| **Part 7** | **What we can't add right now** — categorized blockers |

**One-line answer to "what do I build first?"** Phase 1 of Part 6:
NEFTune + gradient clipping + LoRA+ + persistent tokenized cache + the
browser-side benchmark runner. ~3 days, all small wins.

---

# PART 1 — ROI-ranked next-build queue

## Tier 1 — high ROI, build next

1-3 days each, visible product or educational improvement.

### 1.1 Knowledge distillation ⬜

A small "student" model is trained to match a large "teacher" model's
*logits* (full probability distribution), not just hard ground-truth
tokens. The student learns *what the teacher would have said*,
including the relative probabilities of plausible alternatives — a
much richer training signal than next-token classification alone.
5M-param student distilled from 100M-param teacher often retains
70-90% of the teacher's quality at 20× smaller — the canonical path
to "tiny models that are actually good." Two models in memory; KL
divergence loss; backprop through student only.

**Effort:** ~2 days. **ROI: very high.**

### 1.2 Sequence packing for SFT 🟣

SFT data is full of short examples; naively batching them wastes 90%
of positions on padding. Sequence packing concatenates many short
examples into one long sequence with a block-diagonal attention mask
preventing cross-example contamination. 5-10× SFT throughput on
Dolly-15k-shaped data.

**Effort:** ~1 day (custom mask via `MLXFast.scaledDotProductAttention`).
**ROI: high.**

### 1.3 QLoRA training (combine int4 base + LoRA) 🟡

Quantize the base to int4 (frozen), train fp16 LoRA on top. We have
int4 inference and LoRA training as separate paths; combining them
6× the memory budget — fine-tune 30B HF models instead of 13B on a
48 GB Mac.

**Effort:** ~1 day. **ROI: high.**

### 1.4 ORPO (Odds-Ratio Preference Optimization) ⬜

A 2024 alignment recipe that merges SFT and DPO into a single
training pass. Loss = SFT cross-entropy + a preference-aware
log-odds-ratio term. No separate reference model needed (saves ~½
of DPO's memory). Iterates faster than SFT+DPO separately;
comparable final quality on most benchmarks.

**Effort:** ~1 day (new loss function on top of existing SFT path).
**ROI: high — the modern simplest path to instruction-following.**

### 1.5 SimPO (reference-free DPO) ⬜

DPO without the reference model. Replaces the `logπ_pol - logπ_ref`
ratio with a length-normalized log-probability target. Half the
memory of DPO; ~equivalent final quality on published benchmarks.

**Effort:** ~half day (change the loss function in our DPO trainer).
**ROI: high — frees half the GPU memory for bigger batch sizes.**

### 1.6 NEFTune (noisy embeddings fine-tune) ⬜

During SFT/DPO, add small uniform noise to the embedding outputs
(not the weights, just the embeddings at forward time). Reported
+5-10 points on instruction-following benchmarks. Roughly 3 lines
of code. The least-effort biggest-impact one-line trick of 2024.

**Effort:** ~half day. **ROI: very high per minute spent.**

### 1.7 Gradient checkpointing ⬜

Don't save every layer's activations during forward; re-compute them
during backward. Trade ~30% extra compute for ~√L activation memory
reduction. Unlocks training Behemoth (404M) and Titan (1.3B) at full
batch.

**Effort:** ~2 days. **ROI: high if we want Behemoth/Titan training.**

### 1.8 Speculative decoding ⬜

Small "draft" model proposes K tokens; main model verifies K in one
forward pass. Accepted tokens are free, rejected ones cost one extra
forward. 2-4× sample throughput at no quality cost.

**Effort:** ~2 days. **ROI: high for browser sample UX.**

### 1.9 Browser-side benchmark runner ⬜

The leaderboard infrastructure exists but currently reads pre-computed
scores. A "Run benchmark on your loaded model" button in the browser
closes the submission loop end-to-end.

**Effort:** ~half day (worker plumbing). **ROI: high product win.**

---

## Tier 2 — medium ROI

Solid additions that aren't critical-path.

### 2.1 KTO (Kahneman-Tversky Optimization) ⬜

DPO needs (chosen, rejected) pairs; KTO needs only single examples
labeled "good" or "bad." Much more data available (thumbs up/down
flows). Quality comparable to DPO on most benchmarks.

**Effort:** ~half day (different loss in DPO trainer). **ROI: medium-high
when paired data is scarce.**

### 2.2 IPO (Identity Preference Optimization) ⬜

DPO variant with stronger regularization toward the reference,
designed for small (~1K-pair) datasets where vanilla DPO overfits.

**Effort:** ~half day. **ROI: medium.**

### 2.3 DoRA (Weight-Decomposed LoRA) ⬜

Decompose each weight into magnitude + direction; LoRA the direction,
train magnitude scalars separately. ~5-10% better than vanilla LoRA
at the same rank.

**Effort:** ~1 day. **ROI: medium (free quality on every fine-tune).**

### 2.4 GaLore (Gradient Low-Rank Projection) ⬜

Projects gradients to a low-rank subspace before applying the
optimizer; same memory as LoRA but performs full fine-tuning (all
weights move, not just adapters). Especially useful for pretraining
where LoRA is too restrictive.

**Effort:** ~1.5 days. **ROI: medium-high — full-finetune at LoRA cost.**

### 2.5 VeRA (Vector-based Random Adapters) ⬜

LoRA variant with frozen random projection matrices and trainable
diagonal scalars only. ~10× smaller adapter than LoRA; comparable
quality on most tasks.

**Effort:** ~1 day. **ROI: medium (extreme adapter-size compression).**

### 2.6 LoftQ (LoRA-Friendly Quantization) ⬜

Initialize LoRA adapters to compensate for the quantization error
of the base — A and B are chosen so `A·B` approximates the original
fp32 weight minus its int4-quantized version. Improves QLoRA quality
vs naive initialization.

**Effort:** ~1 day (pairs with QLoRA in 1.3). **ROI: medium.**

### 2.7 AWQ / GPTQ quantization readers ⬜

AWQ and GPTQ are popular int4 storage *formats*, different from
MLX-Swift's built-in quantize. Many HF models ship in AWQ/GPTQ
already (e.g., `TheBloke/Llama-2-7B-AWQ`).

**Effort:** ~1 day per format. **ROI: medium (expands model menu).**

### 2.8 HQQ (Half-Quadratic Quantization) ⬜

Recent (2024) calibration-free int4 quantization. Comparable quality
to GPTQ but faster (~minutes vs hours to quantize a 7B model). No
calibration data needed.

**Effort:** ~1.5 days. **ROI: medium.**

### 2.9 Sliding window attention ⬜

Each token attends only to the last N tokens (e.g., 512). Used by
Mistral. Lets us train at ctx=4096 with memory that otherwise
allows only ctx=512.

**Effort:** ~1 day. **ROI: medium (only matters for long contexts).**

### 2.10 ALiBi position bias ⬜

Alternative to RoPE. Add a position-distance penalty to attention
scores instead of rotating Q/K. Extrapolates to longer contexts at
inference than seen during training — RoPE doesn't generalize as
well.

**Effort:** ~1 day. **ROI: medium.**

### 2.11 KV cache quantization (KIVI) ⬜

Compress K and V tensors in the KV cache to int8 or int4. At long
contexts, the cache is what runs out of memory. KIVI (2024) is a
recent technique with minimal quality loss.

**Effort:** ~1 day. **ROI: medium (long-context inference).**

### 2.12 Multi-Token Prediction (MTP) ⬜

DeepSeek-V3 / Meta MTP variant. Predict the next K tokens at each
position during training (extra heads), not just K=1. Better
training signal + enables speculative decoding via the same heads
at inference. ~10% perplexity improvement reported.

**Effort:** ~2 days. **ROI: medium-high — both a training and
inference win.**

### 2.13 Multi-Query Attention (MQA) 🟢

Extreme GQA — one K/V head shared across all Q heads. Smaller KV
cache, faster decode. **Already supported** via our `nKvHeads`
config; setting `nKvHeads: 1` activates MQA.

**Effort:** zero. **ROI: free.**

### 2.14 Streaming-LLM attention sink ⬜

Keep the first 4 tokens permanently in attention, then slide a window
over the rest. Enables infinite-context streaming without quality
collapse (which naive sliding window suffers from). Particularly
useful for long chat.

**Effort:** ~1 day (mask change + cache management). **ROI: medium.**

### 2.15 Prefix / prompt caching ⬜

When the same prompt prefix is used across many generations, cache
its KV state once and reuse. Common patterns: system prompts,
few-shot examples, RAG retrieved chunks. 5-50× latency improvement
for cache-hit prompts.

**Effort:** ~1 day. **ROI: medium-high (matters once we have a UI
that does long system prompts).**

### 2.16 Prefix tuning / soft prompts ⬜

Instead of LoRA-ing the weights, train a small set of "virtual
tokens" prepended to every prompt. Base fully frozen; ~10K params
vs LoRA's ~100K-1M. Works well for narrow tasks.

**Effort:** ~1 day. **ROI: medium (niche but tiny adapter).**

---

## Tier 3 — niche or specialized

Build when there's a specific use case.

### 3.1 Constitutional AI / RLAIF ⬜

Use a stronger model as judge to critique + revise weaker model
outputs. Anthropic's approach to alignment without humans. Gated on
access to a strong local judge.

**Effort:** ~3 days. **ROI: low for us (no strong local judge).**

### 3.2 GPTQ quantization (from-scratch) ⬜

Layer-by-layer int4 quantization using Hessian information.
Marginally better than MLX's built-in `quantize`; the diff is ~1-2%
perplexity.

**Effort:** ~3 days. **ROI: low (diminishing returns).**

### 3.3 SmoothQuant ⬜

Equalize activation/weight magnitudes before int8 quantization for
cleaner calibration. Less relevant since modern int4 methods (HQQ,
LoftQ) work without this preprocessing.

**Effort:** ~2 days. **ROI: low.**

### 3.4 Quantization-aware training (QAT) ⬜

During training, simulate int4 quantization in the forward (with
straight-through estimator gradients). Model learns to be quantization-
robust. ~2-3% better post-quantization perplexity vs PTQ.

**Effort:** ~3 days. **ROI: low at our scale.**

### 3.5 Pruning (unstructured) ⬜

Zero out the smallest weights. 50% sparsity often retainable. But
Mac/browser GPUs don't efficiently exploit unstructured sparsity —
runtime savings are marginal vs quantization.

**Effort:** ~2 days. **ROI: low.**

### 3.6 Structured pruning (head/layer removal) ⬜

Remove whole attention heads or layers. Real speedup (vs unstructured's
marginal). 10-20% smaller. Tricky because removed pieces cascade.

**Effort:** ~3 days. **ROI: low-medium (distillation usually wins).**

### 3.7 LASER (selective rank reduction) ⬜

Drop the *highest* singular-value components of specific MLP layers.
Counterintuitively often *improves* downstream quality (acts like a
regularizer + removes "memorized" noise). One-line post-training fix
for some quality issues.

**Effort:** ~half day (just SVD + truncate selected layers).
**ROI: medium-low; cheap to try.**

### 3.8 Cross-layer KV sharing ⬜

Have multiple transformer layers share the same K/V tensors.
Smaller KV cache. Used in You Only Cache Once (YOCO), Llama-3.2
small variants.

**Effort:** ~2 days. **ROI: low-medium.**

### 3.9 RLHF / PPO ⬜

The original RLHF: train a reward model, PPO against it. ChatGPT's
original method. DPO is the modern replacement at 1/5 the complexity
for 80-90% of the quality lift.

**Effort:** ~1 week. **ROI: low — DPO does the job.**

### 3.10 Medusa-style speculative decoding ⬜

Speculative decoding where the model has extra "Medusa heads"
predicting future tokens directly. No separate draft model, but
requires training the heads. (Tier 1 vanilla speculative decoding
is simpler.)

**Effort:** ~3 days. **ROI: low.**

### 3.11 EAGLE-2 (better speculative decoding) ⬜

2024 variant of speculative decoding with 2-3× higher acceptance
rates than vanilla. Uses the target model's hidden states (not just
logits) to guide the draft.

**Effort:** ~3 days. **ROI: low (incremental over vanilla
speculative).**

### 3.12 Mixture of Experts (MoE) 🟣

N expert MLPs per layer, router picks top-K per token. Modern
frontier models (Mixtral, GPT-4) are MoE. At our scale, an 8×16M
MoE behaves like 128M capacity in 32M compute. Educational
value > practical value at 100M scale. Parked in
`docs/parked_multi_model.md`.

**Effort:** ~1 week. **ROI: high educationally, medium practically.**

### 3.13 Differential attention ⬜

2024 paper. Replaces standard attention with the *difference* of two
attention maps (subtract noise pattern). Modest perplexity wins,
better long-context recall. Not yet widely adopted.

**Effort:** ~2 days. **ROI: low.**

### 3.14 Mixture of Depths (MoD) ⬜

Router decides which tokens go through each transformer layer (some
skip layers entirely). Saves compute on "easy" tokens. Recent
(2024).

**Effort:** ~3 days. **ROI: low-medium.**

### 3.15 LayerDrop ⬜

Randomly skip layers during training; at inference, optionally use
fewer layers for faster decode. Mild regularizer + post-hoc model
shrinking.

**Effort:** ~1 day. **ROI: low.**

### 3.16 ReLoRA ⬜

Periodically merge LoRA into base + restart training with fresh
LoRA. Accumulates "high-rank" effective updates over many merges.
Closes some of the gap between LoRA and full fine-tune.

**Effort:** ~2 days. **ROI: low-medium.**

### 3.17 AdaLoRA ⬜

Adaptive rank assignment — automatically allocates higher rank to
layers that need it, lower to layers that don't. Slightly better
than fixed-rank LoRA at the same param budget.

**Effort:** ~2 days. **ROI: low.**

### 3.18 LoRA+ ⬜

Different learning rates for the A and B matrices of LoRA (B's LR
is ~16× larger). ~5-10% quality improvement at zero memory cost.

**Effort:** ~half day. **ROI: medium-low.**

### 3.19 PISSA initialization ⬜

LoRA initialized using the principal singular vectors of the base
weight (instead of random). Faster convergence — sometimes 2× faster
to reach the same loss.

**Effort:** ~1 day. **ROI: medium-low.**

### 3.20 LoRA-FA (frozen A) ⬜

Train only B in LoRA; freeze A. Half the params, ~80% of the
quality. Useful when memory is extreme.

**Effort:** ~half day. **ROI: low.**

### 3.21 RsLoRA (rank-stabilized LoRA) ⬜

Different scaling for the LoRA delta (`α/√r` instead of `α/r`).
Improves quality at high ranks (r > 64) where standard LoRA scaling
hurts.

**Effort:** ~half day. **ROI: low (most fine-tunes use r ≤ 16).**

---

## Tier 4 — skip (not worth it for us)

- **fp16 mixed-precision training** — bf16 is strictly better
  (Tier 1 already shipped)
- **ZeRO / FSDP / pipeline parallelism** — multi-device only
- **State space models (Mamba, RWKV)** — different architecture
  entirely; ~2-3 week port; better as a side project
- **PagedAttention / continuous batching** — multi-user inference
- **Tree attention / lookahead decoding** — marginal over
  speculative decoding
- **Adapter modules (Houlsby/Pfeiffer)** — LoRA's older cousin,
  superseded
- **BitFit** — train biases only; quality is poor
- **Hyena / long-conv** — different architecture
- **fp8 training** — needs H100/Blackwell hardware

---

# PART 2 — Categories not on the main ROI line

These are orthogonal to "what to build next" but matter for completeness.
Compact entries.

## Optimizers (we currently use AdamW)

- **AdamW 🟢** — what we have. Standard. Memory: 2× params (m + v).
- **Lion ⬜** — sign-based optimizer; ~½ the memory of Adam. Sometimes
  matches Adam quality. Worth trying for big models where Adam's
  memory dominates.
- **Sophia ⬜** — second-order optimizer (uses Hessian estimates).
  Reported 2× faster convergence. More code complexity.
- **Muon ⬜** — 2024 optimizer. Orthogonalizes gradients via
  Newton-Schulz iterations. Big wins on small-scale benchmarks
  (Karpathy's nanoGPT speedrun adopted it). Worth a benchmark.
- **Adafactor ⬜** — sublinear-memory Adam variant. Trades some
  quality for much less memory.
- **BAdam ⬜** — block coordinate descent for full fine-tune at
  LoRA memory cost. Different mechanism from GaLore.
- **LISA ⬜** — Layer-wise Importance Sampled AdamW. Train only a
  random subset of layers per step. Memory savings + sometimes
  better quality.

## Training stability tricks

- **Gradient clipping ⬜** — clip gradient norm to a fixed value
  (~1.0 typical). Prevents loss spikes. Probably the cheapest +
  most universal stability lever. MLX-Swift has `MLX.clipNorm`;
  needs wiring into Trainer.
- **Z-loss / auxiliary loss ⬜** — add a small penalty term on
  logsumexp magnitudes. Stabilizes training at scale.
- **Embedding RMSNorm ⬜** — apply RMSNorm to the token embedding
  output. Helps with input distribution drift.
- **Layer-wise learning rate decay ⬜** — smaller LR for lower
  layers. Stabilizes fine-tuning.
- **Warmup curves beyond linear ⬜** — cosine warmup, exp warmup.
  Marginal vs linear.
- **DeepNorm ⬜** — residual scaling that improves stability of
  very deep models. Matters past ~50 layers.
- **Embedding tying 🟢** — already a config flag (`tieEmbeddings`).

## Data techniques

- **Curriculum learning ⬜** — order examples easy → hard. Modest
  gains; needs a difficulty metric.
- **DoReMi ⬜** — learns optimal data domain mixing ratios via a
  reference model. Useful when training on a mix (FineWeb + Wiki +
  code).
- **Data quality filtering ⬜** — perplexity-based filtering with
  a reference model. Drop the highest-PPL docs (likely noise).
- **Deduplication ⬜** — drop near-duplicate documents.
  FineWeb-edu is already deduped; matters for raw web scrapes.
- **Hard example mining ⬜** — oversample high-loss examples.
- **Importance sampling ⬜** — sample based on token-level
  importance scores.
- **Self-instruct ⬜** — use the model to generate its own training
  data; bootstrap from a small seed set.
- **Evol-instruct ⬜** — iteratively complicate prompts to grow
  instruction data quality.
- **Distillation-based synthesis ⬜** — generate data with a
  bigger model, train smaller on it. Pairs with knowledge
  distillation (Tier 1.1).
- **Document-level shuffling 🟢** — implicit in our random batch
  sampling.
- **Sample packing ⬜** — different from sequence packing (Tier
  1.2): combine examples from *different* sources into one batch
  to avoid intra-source correlation.

## Tokenization

- **Byte-level (vocab=256) 🟢** — what we have on the from-scratch
  path.
- **HF BPE / SentencePiece via swift-transformers 🟢** — shipped
  for from-scratch (Tier 1) and HF models.
- **BPE-dropout ⬜** — randomly merge less often during encoding;
  regularizer.
- **Train our own BPE on our corpus ⬜** — `tokenizers` Rust crate
  via Python wrapper. ~5% perplexity improvement at same step count
  vs using a foreign tokenizer.
- **Vocabulary trimming ⬜** — drop unused BPE tokens. Shrinks the
  embedding matrix.
- **tiktoken adoption ⬜** — OpenAI's tokenizer. Format isn't
  natively in swift-transformers but is reproducible.
- **Subword regularization ⬜** — present multiple valid
  tokenizations of the same text during training. Robustness
  technique.

## Interpretability tools (educational value)

- **Logit lens ⬜** — at every layer, project hidden state to vocab
  via the LM head, see what token the model would predict if
  forced to stop there. Reveals layer-wise prediction emergence.
- **Activation patching ⬜** — replace one example's hidden state
  with another's at a specific layer; see what changes downstream.
  The mechanistic-interpretability primitive.
- **Linear probes ⬜** — train a small linear classifier on hidden
  states for a specific property. Detects what each layer "knows."
- **Attention heat maps ⬜** — visualize attention weights per head
  per position. Browser playground could ship this.
- **Top-k token-by-token ⬜** — show the top 5 alternatives at each
  generation position with their probabilities. Already partially
  in the browser playground; could be expanded.
- **Per-layer ablation ⬜** — zero out one layer at inference, see
  how much quality drops. Identifies critical vs redundant layers.
- **Sparse autoencoders for interpretability ⬜** — Anthropic-style
  feature decomposition. Substantial build.
- **Knowledge editing (ROME / MEMIT) ⬜** — surgical weight edits to
  modify specific facts.

## Inference optimizations (single-user, single-GPU)

- **KV cache 🟢** — shipped (both arch paths).
- **Flash Attention forward 🟢** — `MLXFast.scaledDotProductAttention`
  + WGSL FA2 in browser.
- **Flash Attention backward 🟢** — same.
- **Quantized inference (int4/int8) 🟢** — via `MLXNN.quantize`.
- **Speculative decoding** — Tier 1.8.
- **Prefix caching** — Tier 2.15.
- **Streaming attention sink** — Tier 2.14.
- **KV cache quantization** — Tier 2.11.
- **Multi-Token Prediction inference path** — Tier 2.12.
- **Token elimination ⬜** — drop low-probability past tokens
  from the KV cache. Trade slight quality for shorter effective
  cache.
- **Continuous batching ⬜** — multi-user only; not us.
- **PagedAttention ⬜** — multi-user only.
- **Tree decoding ⬜** — sample a tree, prune; better than top-k
  for some tasks.

## Browser-side performance (already largely explored in `perf_quest.md`)

- **f16 storage 🟢** — shipped.
- **Blocked 4×4 matmul 🟢** — shipped.
- **WebGPU subgroups ⬜** — better matmul via subgroup ops.
  Chrome only.
- **WebGPU cooperative matrix ⬜** — hardware matmul intrinsics.
  Chrome-experimental.
- **WebNN integration ⬜** — Chrome's neural-network API. Could
  delegate inference to native backend.
- **Memory64 🟢** — shipped (lifts 4 GB heap cap).
- **OPFS persistence 🟢** — shipped.

## Architecture variants (not yet covered)

- **Standard transformer (RoPE + RMSNorm + SwiGLU + GQA) 🟢** —
  what we have.
- **MoE 🟣** — Tier 3.12.
- **Sliding window** — Tier 2.9.
- **Sparse attention (BigBird, Longformer) ⬜** — pattern-based
  sparse masks for very long context. Tier 4 unless we go past
  ctx=8192.
- **Linear attention (Performer, Linformer, Reformer) ⬜** —
  O(N) attention via kernel tricks. Quality usually worse than
  flash attention at moderate contexts.
- **State space models (Mamba, RWKV) ⬜** — Tier 4 (whole
  different family).
- **Hybrid attention/SSM (Jamba, Samba) ⬜** — interleave
  attention + SSM layers. Compromise architecture.
- **Multi-token prediction heads** — Tier 2.12.
- **Mixture of Depths** — Tier 3.14.
- **Differential attention** — Tier 3.13.
- **YOCO (cross-layer KV)** — Tier 3.8.
- **Pre-norm vs post-norm ⬜** — pre-norm is standard; post-norm
  is more stable at very large scale (used by GPT-2 original).
  Toggle in our `TransformerBlock`.

## Adapter / PEFT taxonomy summary

Compact table of the LoRA family for cross-reference:

| Name | Idea | Status | Tier |
|---|---|---|---|
| LoRA | Rank-r `A·B` delta | 🟢 | shipped |
| Multi-LoRA composition | Stack N adapters | 🟢 | shipped |
| DoRA | Magnitude + direction decomp | ⬜ | 2.3 |
| QLoRA | Int4 base + fp16 LoRA | 🟡 | 1.3 |
| VeRA | Frozen random A, B; scalars | ⬜ | 2.5 |
| GaLore | Low-rank gradient projection | ⬜ | 2.4 |
| LoftQ | Quantization-aware LoRA init | ⬜ | 2.6 |
| LoRA+ | Different LR for A vs B | ⬜ | 3.18 |
| LoRA-FA | Freeze A | ⬜ | 3.20 |
| RsLoRA | Different scaling at high r | ⬜ | 3.21 |
| ReLoRA | Periodic merge + restart | ⬜ | 3.16 |
| AdaLoRA | Adaptive per-layer rank | ⬜ | 3.17 |
| PISSA | SVD-based initialization | ⬜ | 3.19 |
| NEFTune | Noise on embeddings | ⬜ | 1.6 |
| Prefix tuning | Soft prompt tokens | ⬜ | 2.16 |
| Prompt tuning | Soft prompt; smaller variant | ⬜ | (subsumed) |
| IA³ | Element-wise scaling | ⬜ | 4 (skip) |
| BitFit | Train biases only | ⬜ | 4 (skip) |
| Adapter (Houlsby) | Bottleneck MLPs | ⬜ | 4 (skip) |

## Cross-cutting infrastructure (lifts ROI of everything)

- **Browser-side benchmark runner** — Tier 1.9.
- **Real submission upload flow** — drag-drop + auto-score; OPFS
  first, R2 later.
- **TinyGPT-as-library API** — surface the four primitives
  (`forward_backward`, `optim_step`, `sample`, `save_state`) as a
  public Python/TS API. Parked, see
  `memory/project_tinker_parked.md`.
- **Persistent tokenized cache ⬜** — write `.tokens` files alongside
  `.txt` corpora. Saves the 30-min BPE-encode cost on every run.
- **Real CI ⬜** — GitHub Actions: build + test on every PR.
- **`tinygpt eval` benchmark harness ⬜** — `tinygpt eval --bench
  tinystories-ppl path/to/model.tinygpt` would replace the
  separate `score_gallery.mjs` scripts.

---

# PART 3 — recommended order

Filtering for highest ROI + most likely to surprise-and-delight at our
scale, in order:

1. **NEFTune** (Tier 1.6, ~half day) — 3-line embedding noise +5-10
   points on instruction-following. The single highest-impact-per-minute
   item on this entire document.
2. **Magpie synthetic SFT pipeline** (Part 5 addition, ~1 day) — use
   a public aligned LLM to synthesize SFT data without seeds. Models
   trained on Magpie data surpass Llama-3-8B-Instruct (ICLR 2025).
   Eliminates our dependency on Dolly/no_robots specifically.
3. **Sequence packing** (Tier 1.2, ~1 day) — already on the roadmap.
4. **Knowledge distillation** (Tier 1.1, ~2 days) — the educational +
   leaderboard play, validated by April 2026 survey.
5. **SimPO** (Tier 1.5, ~half day) — reference-free preference
   training, half the DPO memory.
6. **Evolution Strategies trainer** (Part 5 addition, ~3 days) — the
   "genuinely novel for our scale" play. Beats RL on long-horizon
   rewards; lower memory; parallelizable on CPU. May be the right
   alignment recipe for resource-constrained settings.
7. **QLoRA training** (Tier 1.3, ~1 day) — 6× fine-tune ceiling.
8. **ORPO** (Tier 1.4, ~1 day) — merge SFT + DPO into one pass.
9. **Browser-side benchmark runner** (Tier 1.9, ~half day) — closes
   the leaderboard loop.
10. **Gradient clipping** (training-stability section, ~half day) — we
    should already have this; it's standard hygiene.

≈ **7 days of focused work for the entire Tier 1 + top training-stability
items** — and at the end of that we have: a noise-regularized SFT path,
4× SFT speedup, distillation working, ref-free preference training,
combined int4-base + LoRA training, single-pass SFT+DPO, in-browser
benchmark scoring, and proper gradient clipping. That's the "polish
the post-training pipeline" tier.

---

# PART 4 — open-source datasets

Verified URLs, sizes, and licenses for everything in the
pretrain → SFT → DPO pipeline. Picked for "this fits on a 48 GB Mac"
or "we can stream a slice."

## Pretraining corpora

| Dataset | Size | License | Notes |
|---|---|---|---|
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1.3 T tokens (default), `sample-10BT` / `100BT` / `350BT` subsets, 114 total | ODC-BY-1.0 | **Our pick.** Educational-quality web text; each doc has a 2.5-5.06 quality score. CC-MAIN snapshots from 2013-2025. |
| [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 15 T tokens | ODC-BY-1.0 | Broader, less-filtered parent of FineWeb-edu. Use when fineweb-edu becomes a bottleneck. |
| [togethercomputer/RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) | 30 T+ tokens | Apache 2.0 + various | RedPajama's open Llama-equivalent. Mostly Common Crawl. |
| [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) | 627 B tokens | various | Deduplicated subset of RedPajama-V1. |
| [allenai/c4](https://huggingface.co/datasets/allenai/c4) | 750 GB | ODC-BY | Cleaned Common Crawl (T5's pretraining data). |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | ~20 GB English | CC-BY-SA | Clean encyclopedic prose. Pairs well with web corpora. |
| [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | ~25 B tokens | Apache 2.0 | Synthetic educational text generated by Mixtral. Used to train SmolLM. |
| [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | ~600 B tokens | Apache 2.0 | SmolLM's pretraining mix — Cosmopedia + FineWeb-edu + Stack. |
| [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | ~2 GB | CDLA-Permissive-1.0 | Designed for sub-100M-param models specifically. Vocabulary capped at ~1500 words; the 1M-param coherence threshold. |
| [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted) | ~825 GB | various | The Pile minus copyrighted portions. Pythia's training data. |
| [bigscience/roots](https://huggingface.co/datasets/bigscience/roots) | 1.6 T tokens | varies | BLOOM's pretraining corpus. Multilingual. |

## SFT (supervised fine-tuning)

| Dataset | Size | License | Notes |
|---|---|---|---|
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 15 K pairs | CC-BY-SA-3.0 | Hand-written, high quality. Our first-run pick. |
| [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | 10 K pairs | CC-BY-NC-4.0 | Hand-written, broader topic coverage than Dolly. |
| [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | 939 K pairs | ODC-BY-1.0 + mixed | 18 datasets blended. Includes Persona-MATH, WildChat-GPT4, FLAN-v2, Evol-CodeAlpaca, OpenAssistant-Guanaco, WildGuardMix, etc. **The kitchen-sink SFT set.** |
| [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | ~1 M conversations | mixed | Distilled from GPT-4. Higher quality per pair than Tulu. |
| [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | ~518 K | MIT | Slimmer Orca; distilled instructions. |
| [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | 52 K | CC-BY-NC-4.0 | GPT-3.5-generated; classic. Lower quality per pair. |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | ~10 K conversations | Apache 2.0 | Multi-turn, human-labeled. |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 200 K conversations | MIT | Large-scale multi-turn chat. |

## Preference data (DPO / KTO / SimPO / ORPO)

| Dataset | Size | License | Notes |
|---|---|---|---|
| [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 187 K paired (61K train_prefs split) | MIT | **Our DPO pick.** Six splits: train_sft / train_prefs / train_gen × test/train. GPT-4 judgments. TruthfulQA-decontaminated. |
| [argilla/dpo-mix-7k](https://huggingface.co/datasets/argilla/dpo-mix-7k) | 7 K paired | mixed | Smaller, cleaner. Good for small DPO runs. |
| [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 60 K paired | MIT | UltraFeedback with corrected labels (~hundred fixed). |
| [allenai/llama-3.1-tulu-3-8b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture) | 270 K paired | ODC-BY | Tulu-3's preference mix; uses on-policy DPO data. |
| [anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) | 160 K paired | MIT | Human-labeled "helpful" and "harmless" preferences. Slow to grow but human-grade. |
| [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | 40 K | CC-BY-4.0 | Single-label good/bad (use with KTO). |
| [argilla/distilabel-capybara-dpo-7k-binarized](https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized) | 7 K paired | mixed | Smaller curated set. |

## Code data

| Dataset | Size | License | Notes |
|---|---|---|---|
| [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | 3.1 B files (~67 TB raw) | various | Largest permissive-license code corpus. Use subsets. |
| [codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) | ~110 GB | various | Pre-filtered Github code. |
| [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) | ~3 TB | various | Deduplicated v1. |
| [nickrosh/Evol-Instruct-Code-80k-v1](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) | 80 K | MIT | Code SFT data via Evol-Instruct. |
| [HuggingFaceH4/CodeAlpaca_20K](https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K) | 20 K | CC-BY-4.0 | Hand-cleaned code instructions. |

## Math + reasoning

| Dataset | Size | License | Notes |
|---|---|---|---|
| [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | 395 K | MIT | Math instruction-following. |
| [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 860 K | Apache 2.0 | Olympiad-style math + Chain-of-Thought. |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 8.5 K | MIT | Grade-school math word problems. The standard math benchmark. |
| [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math) | 12.5 K | MIT | MATH dataset; competition-level math. |

## Eval / benchmark datasets

| Benchmark | Source | Notes |
|---|---|---|
| **MMLU** | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | 57-task knowledge benchmark. Standard for "general capability." |
| **HellaSwag** | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | Common-sense continuation. |
| **TruthfulQA** | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Truthfulness eval. |
| **ARC** | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | Grade-school science MC. |
| **GSM8K** | (see math row above) | Math word problems. |
| **MT-Bench** | [HuggingFaceH4/mt_bench_prompts](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts) | LLM-judge instruction-following. |
| **AlpacaEval** | [tatsu-lab/alpaca_eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) | Pairwise GPT-4 judgments. |
| **IFEval** | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | Instruction-following with verifiable constraints. Rule-based eval — no LLM judge. |

---

# PART 5 — recent research (2024-2025 highlights)

The 2024-era body of work that informed Tier 1-3 above. Each entry is
just enough context to know whether to dig in.

## Alignment / preference

- **DPO** — Rafailov et al., NeurIPS 2023.
  "[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)."
  The paper that displaced PPO/RLHF for most labs.
- **KTO** — Ethayarajh et al., 2024.
  "[Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)."
  Single-label preference; no pairs needed.
- **ORPO** — Hong et al., 2024.
  "[Reference-Free Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691)."
  Merges SFT + DPO into one loss. Drops the reference model.
- **SimPO** — Meng et al., 2024.
  "[Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)."
  Reference-free DPO. Length-normalized objective.
- **IPO** — Azar et al., 2023.
  "[A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)."
  Identity preference; stronger regularization than DPO.
- **CPO** — Xu et al., 2024.
  "[Contrastive Preference Optimization](https://arxiv.org/abs/2401.08417)."
  Combines DPO with a behavior-cloning term.
- **NEFTune** — Jain et al., NeurIPS 2023.
  "[Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)."
  3 lines of code; +5-10 points on AlpacaEval.

## Parameter-efficient fine-tuning

- **DoRA** — Liu et al., 2024.
  "[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)."
  Magnitude + direction decomposition; consistently beats LoRA.
- **GaLore** — Zhao et al., 2024.
  "[GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)."
  Full fine-tuning at LoRA memory.
- **LoftQ** — Li et al., ICLR 2024.
  "[LoftQ: LoRA-Fine-Tuning-Aware Quantization](https://arxiv.org/abs/2310.08659)."
  Quantization-aware LoRA initialization.
- **VeRA** — Kopiczko et al., ICLR 2024.
  "[VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454)."
  Shared random matrices; 10× smaller than LoRA.
- **PISSA** — Meng et al., 2024.
  "[PiSSA: Principal Singular Values and Singular Vectors Adaptation](https://arxiv.org/abs/2404.02948)."
  SVD-based LoRA initialization for faster convergence.
- **LoRA+** — Hayou et al., ICML 2024.
  "[LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354)."
  Different LRs for A and B.
- **rsLoRA** — Kalajdzievski, 2023.
  "[A Rank Stabilization Scaling Factor](https://arxiv.org/abs/2312.03732)."

## Quantization

- **GPTQ** — Frantar et al., ICLR 2023.
  "[GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)."
- **AWQ** — Lin et al., MLSys 2024.
  "[AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)."
- **HQQ** — Badri & Shaji, 2024.
  "[Half-Quadratic Quantization](https://mobiusml.github.io/hqq_blog/)."
  Calibration-free int4.
- **KIVI** — Liu et al., 2024.
  "[KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)."
- **BitNet b1.58** — Ma et al., 2024.
  "[The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)."
  Ternary weights ({-1, 0, 1}) from scratch. Training-from-scratch; not post-training.

## Inference / efficiency

- **Speculative decoding** — Leviathan et al., ICML 2023.
  "[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)."
- **Medusa** — Cai et al., 2024.
  "[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)."
- **EAGLE / EAGLE-2** — Li et al., 2024.
  "[EAGLE-2: Faster Inference of Language Models](https://arxiv.org/abs/2406.16858)."
- **StreamingLLM** — Xiao et al., ICLR 2024.
  "[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)."

## Architecture variants

- **Multi-Token Prediction (MTP)** — Gloeckle et al., ICML 2024.
  "[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)."
  Used by DeepSeek-V3 and Meta. Predict K tokens per position.
- **Differential Transformer** — Microsoft 2024.
  "[Differential Transformer](https://arxiv.org/abs/2410.05258)."
  Subtract a noise attention pattern from the signal one.
- **Mixture of Depths** — Raposo et al., 2024.
  "[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)."
  Route tokens through fewer layers.
- **Mamba / Mamba-2** — Gu & Dao, 2023/2024.
  "[Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)."
- **LASER** — Sharma et al., ICLR 2024.
  "[The Truth is in There: Improving Reasoning with Layer-Selective Rank Reduction](https://arxiv.org/abs/2312.13558)."
  Counterintuitively beneficial selective rank truncation.

## Optimizers

- **Sophia** — Liu et al., 2023.
  "[Sophia: A Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342)."
- **Lion** — Chen et al., NeurIPS 2023.
  "[Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)."
- **Muon** — Jordan, 2024.
  Newton-Schulz orthogonalization on gradients.
  [Notes by Keller Jordan](https://kellerjordan.github.io/posts/muon/).
  Adopted by the nanoGPT speedrun community for big wins.
- **GaLore** — see PEFT section above.
- **LISA** — Pan et al., 2024.
  "[LISA: Layerwise Importance Sampling](https://arxiv.org/abs/2403.17919)."

## Distillation

- **Soft targets distillation** — Hinton et al., 2015. The original.
- **MiniLLM** — Gu et al., ICLR 2024.
  "[MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)."
  KL-divergence variants for distilling LLMs.
- **Distilling Step-by-Step** — Hsieh et al., ACL 2023.
  "[Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301)."
  Distill reasoning chains.

## Synthetic data + curriculum

- **Self-Instruct** — Wang et al., 2023.
  "[Self-Instruct: Aligning LMs with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)."
- **Evol-Instruct** — Xu et al., 2024 (WizardLM).
  "[WizardLM: Empowering LLMs to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)."
- **DoReMi** — Xie et al., NeurIPS 2023.
  "[DoReMi: Optimizing Data Mixtures](https://arxiv.org/abs/2305.10429)."
  Learns optimal data-domain ratios.
- **TinyStories** — Eldan & Li, 2023.
  "[TinyStories: How Small Can Language Models Be?](https://arxiv.org/abs/2305.07759)."
  Established the sub-100M coherence threshold on a constrained vocabulary.

## 2025-era — reasoning + RLVR + verifier-based training

The biggest single shift in 2025 was the move from "preference
alignment is the post-training step" to "reasoning + RL on verifiable
rewards is the post-training step." If we're documenting current state
honestly, this section matters more than the alignment papers above.

- **DeepSeek-R1 / R1-Zero** — DeepSeek-AI, Jan 2025.
  "[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning](https://arxiv.org/abs/2501.12948)."
  Showed that RL with verifiable rewards (math/code grading) on a
  strong base produces emergent chain-of-thought reasoning. R1-Zero
  is the variant that skipped SFT entirely and went straight from
  base to RL. Single most-cited result of 2025.
- **GRPO (Group Relative Policy Optimization)** — DeepSeek, 2024-2025.
  PPO variant used by R1 that doesn't need a separate value
  network. Computes advantage from a group of rollouts' relative
  rewards. Less memory than PPO, fits on smaller hardware. See
  R1 paper's training-recipe appendix.
- **RLVR (Reinforcement Learning from Verifiable Rewards)** —
  the umbrella name for: math problems → grade with verifier;
  code → grade by running tests; reasoning → grade by deterministic
  checker. No human or LLM judge needed. Allen AI's Tulu-3 paper
  was the first open recipe; DeepSeek-R1 scaled it dramatically.
- **OpenAI o1 / o3** — proprietary but reframed the field. The
  "test-time compute" thesis: hand the model more inference compute
  (chain-of-thought tokens) and it gets noticeably smarter. Started
  the "reasoning model" category.
- **Reasoning distillation** — DeepSeek-R1-Distill series:
  reasoning traces from R1 are used as SFT data for smaller models.
  The 7B / 14B / 32B distilled variants approach much-bigger
  non-reasoning models on math + code benchmarks. Big practical
  win for the "small model that's actually useful" play.
- **OpenThoughts** — community open-source dataset of reasoning
  traces (huggingface.co/datasets/open-thoughts/OpenThoughts-114k).
  Built to be the open analog of R1's reasoning data.
- **Test-time compute scaling** — separate axis from training
  scaling. Spend more inference tokens (CoT) for better outputs.
  Has its own scaling-law literature. Notable: Snell et al., 2024.
  "[Scaling LLM Test-Time Compute Optimally Can Be More Effective
  Than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)."
- **DeepSeek-V3** — DeepSeek-AI, Dec 2024.
  "[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)."
  671B-MoE (37B active). Multi-Token Prediction during training.
  Major open-MoE recipe — sets the bar for 2025-era frontier
  MoE training.
- **Qwen3** — Alibaba, 2025. Latest in the Qwen family (4B-397B).
  Newer dense + MoE variants; supported by Tinker per their page.

### What "2025-era post-training" looks like at our scale

The R1-class recipe at the 100M-1B scale is hard but interesting:

1. **Pretrain** (still the same) — FineWeb-edu etc.
2. **SFT on reasoning traces** — use OpenThoughts or
   R1-distill data. Trains the model to emit `<think>...</think>`
   blocks before answers. ~6× more tokens per example than vanilla
   SFT because reasoning is verbose.
3. **RL with verifiable rewards** — for any task with a programmatic
   verifier (math, code, format-following), use GRPO to RL the
   model on those tasks. Requires a verifier per task. Hard at our
   scale because RL eats compute.

Distillation from R1-Distill-7B / 14B into a 100M model is
plausible at our scale; full RLVR is probably a stretch. Tracked as
a future tier.

### What 2025 datasets to know about

| Dataset | Use | Notes |
|---|---|---|
| [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | Reasoning SFT | Open-source reasoning traces, R1-style |
| [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | SFT (already listed) | The 2024-Q4 open-recipe gold standard |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | Distillation teacher | (And siblings: 1.5B, 14B, 32B). If we want to do reasoning distillation, these are the obvious teachers. |
| [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | RLVR / math | Math problems with verifiable answers |
| [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Math + CoT (already listed) | Used by many 2025 reasoning models |

## 2025-2026 web-verified additions

After running real web/arxiv searches (May 2026), these are the items
not in my training data that matter for our scale:

### Reasoning RL — DAPO supersedes GRPO

**DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)** —
[ByteDance Seed & Tsinghua, March 2025](https://arxiv.org/abs/2503.14476).
Beat DeepSeek-R1 by 3 points on AIME 2024 with 50% of R1's training
compute. Four techniques layered on top of GRPO: **clip higher**
(asymmetric PPO clipping bounds), **dynamic sampling** (rejection
sample bad rollouts before they enter the gradient), **token-level
policy gradients** (per-token advantage rather than per-sequence),
**overlong reward shaping** (handle generations that hit the token
limit). Open source at
[BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO).

Practical implication for us: **GRPO is the right Tier 3 mental model;
DAPO is the right Tier 3 implementation.** If we ever build RLVR, copy
DAPO not vanilla GRPO.

### Evolution Strategies are competitive again (genuinely surprising)

**Evolution Strategies at Scale** —
[Qiu et al., Sept 2025](https://arxiv.org/abs/2509.24372). First
successful application of ES to billion-parameter LLM fine-tuning at
full parameter scale (no dimensionality reduction). Findings that
matter at our scale:

- **Beats PPO/DPO** on long-horizon and delayed-reward tasks
- **More robust to reward hacking** than RL
- **Better training stability** — no value network to tune
- Tolerates very high-dimensional parameter spaces
- Open source: [VsonicV/es-fine-tuning-paper](https://github.com/VsonicV/es-fine-tuning-paper)

**Why this matters specifically for TinyGPT:** ES is parallelizable
across CPU workers (no GPU needed for the rollouts) and has lower
per-step memory than PPO/GRPO. At our resource-constrained scale it
could plausibly out-perform DPO/SimPO for instruction-following at
the same wall-clock budget. Worth a real benchmark.

**Adds a new Tier 1 item** (call it 1.10): **Evolution Strategies
alignment trainer** — ~3 days, novel-for-our-scale, low memory.

### Synthetic SFT data — Magpie is the new standard

**Magpie: Alignment Data Synthesis from Scratch** —
[Xu et al., ICLR 2025](https://arxiv.org/abs/2406.08464). Generates
high-quality SFT data by prompting aligned LLMs with **just their
pre-query templates**. No seeds, no prompt engineering, no human
labels. Models SFT'd on Magpie data **surpass Llama-3-8B-Instruct**
(which used 10M SFT pairs from heavy SFT + DPO). On AlpacaEval,
ArenaHard, and WildBench.

Released: [Magpie-Align/Magpie-Llama-3.3-1M](https://huggingface.co/Magpie-Align)
— 1M synthetic SFT pairs from Llama-3.3-70B, Jan 2025.

**Adds a new Tier 1 item** (1.11): **wire a Magpie pipeline** — point
at any open-weights aligned model (Llama-3.3-8B-Instruct works), get
high-quality SFT data we don't have to curate. ~1 day.

### FP4 training is real and matters

Three papers in 2025 established that **fully-quantized FP4 training**
(weights + activations + gradients all in FP4) reaches BF16-comparable
quality:

- [Wang et al., Jan 2025](https://arxiv.org/abs/2501.17116) — "Optimizing LLM Training Using FP4 Quantization." Demonstrated at 13B params on 100B tokens.
- [Microsoft & Nvidia FP4-All-the-Way, May 2025](https://arxiv.org/abs/2505.19115) — 200B-token validation.
- [Quartet II / NVFP4, Jan 2026](https://arxiv.org/abs/2601.22813) — improved gradient estimator.

The format that wins is **NVFP4**: blocks of 16 FP4 values share a
scale factor; stochastic rounding on backward+update, round-to-nearest
on forward. Key empirical threshold: when gradient norm drops below
~√3 × quantization noise, FP4 training stops working — caps how deep
into training you can stay in FP4.

**For us:** Mac M-series GPUs don't have native FP4 ops yet (we'd
simulate). bf16 → FP4 is a ~3-4× memory savings on top of bf16's 2×.
But the dependency on hardware FP4 means this is **parked** for
TinyGPT until Apple silicon supports it.

### Distillation has matured into a survey topic

- [On-Policy Distillation Survey, April 2026](https://arxiv.org/abs/2604.00626) — confirms distillation is "the dominant technique for transferring frontier capabilities into smaller, deployable student models." Validates our Tier 1.1.
- **MiniPLM** ([Gu et al., NeurIPS 2024](https://openreview.net/forum?id=tJHDw8XfeC)) — distillation for **pre-training**, not just post-training. Distill a small base FROM a big base. Novel.
- **Knowledge Distillation with Training Wheels** ([Feb 2025](https://arxiv.org/abs/2502.17717)) — student can "request help" from teacher at test time. Interesting research direction but not yet practical.

### Pretraining corpora — Dolma is the OLMo alternative

[allenai/dolma](https://huggingface.co/datasets/allenai/dolma) —
3 trillion tokens. OLMo's training data. Includes web (Common Crawl +
Refined Web + C4) + academic (arXiv + Semantic Scholar) + code
(StarCoder) + Reddit + StackExchange + books. **More diverse mix than
FineWeb** (which is pure web). Smaller (3T vs FineWeb's 15T) but
higher per-token diversity.

Pair with FineWeb-edu (filtered quality) for our pretraining mix.
Updated table in Part 4 reflects this.

### 2026 small-model landscape (relevant peers)

The competitive scale for "small model that's actually useful" has
shifted up since our project started:

- **SmolLM3-3B** — fully open instruct + reasoning model. Beats
  Llama-3.2-3B and Qwen2.5-3B on 12 benchmarks. Trained on
  ~600B tokens (SmolLM corpus = Cosmopedia + FineWeb-edu + Stack).
- **Qwen3.5-0.8B** — multimodal (text + vision) from scratch.
  Apache 2.0. Released Feb-March 2026.
- **Phi-4-mini-instruct** — Microsoft. Data-quality-over-scale
  thesis. Beats GPT-4o on MATH; beats Llama-3.2-3B across all
  benchmarks.
- **Gemma-3n-E2B-IT** — Google. On-device-focused. 2B with
  multi-modal.

Implication: **the leaderboard's "browser-trainable small model"
niche is now distinctively educational + open-process, not
performance-competitive with 2026 commercial small models.** The
leaderboard product narrative should emphasize "every byte of training
code is here" + "trains in a tab" rather than "competes with Phi-4."

### Tools we should know about

- **[Unsloth](https://github.com/unslothai/unsloth)** — fine-tuning
  framework that fits 8B models on 12 GB consumer GPUs via custom
  Triton kernels + memory tricks. The reference for "what a serious
  consumer-grade fine-tune stack looks like." Not Mac/MLX-Swift, but
  worth studying for technique transfer.
- **[Argilla Distilabel](https://github.com/argilla-io/distilabel)** —
  Python framework for synthetic SFT/DPO data generation pipelines.
  Wraps Magpie, DEITA, UltraFeedback recipes.
- **DEITA** ([Liu et al., 2024](https://arxiv.org/abs/2312.15685))
  — instruction-tuning data quality framework. Scores complexity ×
  quality × diversity for instruction sets.

## Honest note on the cutoff

My knowledge cutoff is **January 2026**. That means:

- ✅ Solid: everything through ~end of 2025 (R1, V3, Qwen3, Tulu-3,
  GRPO, the alignment-recipes era)
- ⚠️ Patchy: late 2025 / early 2026 — I know there were ongoing
  developments in reasoning RL, agent frameworks, and longer-context
  but specifics are spottier
- ❌ Unknown: anything from Feb 2026 onward

**If you have specific recent papers, datasets, or techniques in mind
that aren't listed**, point me at the URL or name and I'll fold them
in. The above is "what I know is current"; not "what is current."

## Survey / overview reads

- **State of GPT** (Karpathy, 2023) — still the cleanest mental model
  of pretrain → SFT → RM → PPO. We're skipping RM/PPO for DPO.
- **A Survey of LLMs** (Zhao et al., 2024) —
  "[arXiv 2303.18223](https://arxiv.org/abs/2303.18223)." Continuously updated.
- **HuggingFace Alignment Handbook** —
  [github.com/huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook).
  The reference recipes for SFT/DPO at the 7B scale.
- **AllenAI Tulu-3 paper** — Lambert et al., 2024.
  "[Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)."
  The open recipe most relevant to our pipeline shape.
- **SmolLM blog post** (Hugging Face, 2024) —
  [blog](https://huggingface.co/blog/smollm). 135M / 360M / 1.7B
  fully-open small models with their training recipe.

---

# PART 6 — the phased roadmap (sequential, executable)

7 weeks of sequenced work. Each phase is ~1 week of focused effort with
one concrete deliverable. You can stop after any phase and still have
shipped something useful.

## Phase 1 — foundation polish (~3 days of quick wins)

| Item | Days | Bucket |
|---|---:|---|
| NEFTune (noisy embeddings) | 0.5 | quality |
| Gradient clipping | 0.5 | stability |
| LoRA+ (different LR for A vs B) | 0.5 | quality |
| Persistent tokenized corpus cache | 0.5 | dev velocity |
| Browser-side benchmark runner | 0.5 | new option (browser) |
| Investigate Mega-bf16 OOM + add guardrails | 1.5 | stability |

**Deliverable:** every post-training step gets the NEFTune bump;
gallery models can be benchmarked from the browser; resumed runs
don't re-tokenize the corpus.

## Phase 2 — useful post-trained model (~5 days)

| Item | Days | Bucket |
|---|---:|---|
| Magpie SFT pipeline | 1 | new option |
| SimPO (reference-free DPO) | 0.5 | perf |
| ORPO (merge SFT + DPO) | 1 | new option |
| KTO (single-label preference) | 0.5 | new option |
| Sequence packing for SFT | 1 | perf |
| Run Huge → SFT → DPO end-to-end, score | 1 | shipping |

**Deliverable:** a Huge-100M model that follows instructions
reasonably, with concrete leaderboard scores.

## Phase 3 — inference unlock (~5 days)

| Item | Days | Bucket |
|---|---:|---|
| Speculative decoding | 2 | perf |
| KV cache quantization | 1 | perf + long context |
| Prefix / prompt caching | 1 | perf |
| StreamingLLM attention sink | 1 | new option |

**Deliverable:** browser playground feels 2-4× snappier; long-context
sampling becomes practical.

## Phase 4 — knowledge distillation, the educational headliner (~6 days)

| Item | Days | Bucket |
|---|---:|---|
| Knowledge distillation trainer | 2 | learning + new option |
| Distill Mega-instruct → 5M student | 1 | shipping |
| Reasoning distillation from R1-Distill | 2 | learning + new option |
| Compare: distillation vs same-size from scratch | 1 | shipping |

**Deliverable:** a 5M-param model that punches above its weight on
the leaderboard. Case study: tiny models reproducibly competing.

## Phase 5 — MoE + distillation combined (~7 days, the bet you're excited about)

This is its own phase because it's the highest-leverage capability
unlock that fits locally. **More effective capability per gigabyte
of memory than the dense equivalent.**

| Item | Days | Bucket |
|---|---:|---|
| MoE architecture: router + expert MLP + load-balance loss | 3 | learning + new option |
| Train a tiny MoE from scratch on a known-good corpus (sanity) | 1 | learning |
| Distill from open-MoE teacher (DeepSeek-V3-Distill family or Mixtral-class) into our 2B / 8-expert MoE | 2 | new option |
| Compare: 2B MoE (4 GB at bf16) vs dense 500M baseline at same per-token compute | 1 | shipping |

**Deliverable:** a 2B-total / ~500M-active MoE that fits in ~4 GB
of memory and outperforms a 500M dense model at the same per-token
compute cost. The "we run a much-bigger-effective model locally"
artifact.

## Phase 6 — new training paradigms + bigger models (~5 days)

| Item | Days | Bucket |
|---|---:|---|
| Evolution Strategies trainer | 3 | new option |
| Multi-Token Prediction | 2 | perf + new option |
| Gradient checkpointing | 2 | new option (bigger models) |

**Deliverable:** real benchmark numbers comparing ES vs DPO; the
first Behemoth/Titan training run that fits in memory.

## Phase 7 — browser-side performance frontier (~5 days)

| Item | Days | Bucket |
|---|---:|---|
| WebGPU subgroups | 2 | perf (browser) |
| WebGPU cooperative matrix | 2 | perf (browser) |
| WebNN integration as fallback | 1 | new option (browser reach) |

**Deliverable:** speedup curve extends from 12.1× into 15-20× on
Chrome; capability pills advertise active perf paths.

## Phase 8 — interpretability tools (~3 days, browser playground)

| Item | Days | Bucket |
|---|---:|---|
| Logit lens visualization | 1 | learning |
| Attention heatmap UI | 1 | learning |
| Per-layer ablation tool | 1 | learning |
| Activation patching | 1.5 | learning |

**Deliverable:** playground gets an "Inspect" tab alongside Sample /
Train / Fine-tune.

## Phase 9 — quantization + small-model story (~5 days)

| Item | Days | Bucket |
|---|---:|---|
| QLoRA training (int4 base + LoRA) | 1 | new option |
| DoRA | 1 | quality |
| AWQ reader | 1 | new option |
| HQQ quantization | 1.5 | perf |
| LASER selective rank reduction | 0.5 | perf + new option |

**Deliverable:** every gallery model ships in three sizes (fp32 /
bf16 / int4) with documented quality tradeoffs.

## Phase 10 — architecture menu (~6 days, educational)

| Item | Days | Bucket |
|---|---:|---|
| Sliding window attention | 1 | new option |
| ALiBi position bias | 1 | learning |
| Mixture of Depths | 2 | learning |
| Differential attention | 1.5 | learning |
| YOCO cross-layer KV sharing | 1 | new option |

**Deliverable:** five attention variants implemented alongside the
standard one — the "every modern architectural idea has a one-file
implementation here" story.

## The cut-points that matter

- **Stop at Phase 2 (~8 days):** you have a useful 100M
  instruction-following model + real leaderboard numbers. V1 done.
- **Stop at Phase 5 (~26 days):** add inference perf + distillation
  + the MoE-distill big-model-locally artifact. The story is
  complete and compelling. **This is where I'd cut for the HN launch.**
- **Stop at Phase 7 (~36 days):** add ES + browser perf frontier.
  Everything that's both novel and at-our-scale.
- **Everything after Phase 7 is polish + educational deepening.**

---

# PART 7 — what we can't add right now

Categorized blockers. These are NOT the "skip" items (Tier 4) — those
are things we deliberately won't build because better alternatives
exist. These are things we'd build but **can't** for external reasons.

## Blocked by hardware

| Item | Why blocked | Unblock condition |
|---|---|---|
| **Distributed training (ZeRO, FSDP, pipeline parallelism, tensor parallelism)** | Single device only; nothing to parallelize across | Buy/rent a multi-GPU cluster — not the project's scope |
| **Native FP4 training** | Mac M-series GPU lacks FP4 tensor ops | Apple ships FP4 support (rumored on future M-series; not current) |
| **Native FP8 training** | Same — no FP8 ops on Apple silicon | Same |
| **Hardware-accelerated MoE routing** | Apple silicon doesn't have specialized sparse-routing ops | Same |
| **ANE (Apple Neural Engine) acceleration of training** | ANE is inference-only; not exposed for training | Apple opens ANE training APIs (no public roadmap) |

## Blocked by external library state

| Item | Why blocked | Unblock condition |
|---|---|---|
| **Gradient checkpointing as first-class** | MLX-Swift doesn't expose it yet (would write custom forward — possible but invasive) | MLX-Swift adds API (tracked upstream); or we ship a custom impl as Phase 6 |
| **Fast BPE encoding** | swift-transformers BPE is single-threaded; 2 GB corpus takes ~30 min | Wait for swift-transformers improvements OR write a Rust-backed encoder via FFI |
| **Native int4 / int8 matmul on browser WebGPU** | WebGPU doesn't yet have quantized matmul extensions | Wait for WebGPU spec (subgroup / coop-matrix extensions in Phase 7 help) |
| **AWQ / GPTQ / GGUF model loading** | No Swift readers exist yet | We could write them (Phase 9) — just hasn't been done |

## Blocked by budget / cost

| Item | Why blocked | Unblock condition |
|---|---|---|
| **Tinker / managed cloud training APIs** | Usage-based pricing; not affordable for solo project | Project becomes funded |
| **Large-scale synthetic data generation via GPT-4 / Claude API** | $1K-$10K to generate Magpie-scale (~1M pairs) of frontier-quality SFT data | Use open-weights teachers instead (Magpie pipeline does this) |
| **Multi-TB dataset downloads** | Bandwidth + disk for full Common Crawl / Pile | Stream subsets (the HF importer does this); full corpora not needed at our scale |
| **Strong local judge model for Constitutional AI / RLAIF** | No 70B+ model fits + runs at usable speed on a single Mac | Hardware grows OR use a smaller (worse) judge with explicit caveat |

## Blocked by knowledge cutoff

| Item | Why blocked | Unblock condition |
|---|---|---|
| **Anything published after January 2026** | My training cutoff | User pastes URLs / paper names; I fold them in |
| **Late-2025 / early-2026 alignment recipes I haven't seen** | Patchy coverage of Nov 2025 onward | Same |
| **Cutting-edge benchmark / dataset releases** | Same | Same — see how DeepSeek-R1, DAPO, Magpie all needed web search to verify |

## Blocked by integration scope

| Item | Why blocked | Unblock condition |
|---|---|---|
| **Full RLHF / PPO pipeline with reward model training** | Real cost is 5× the code of DPO + 10× the iteration time; usually skipped at our scale | DPO already covers 80-90% of the value (Tier 3.9) |
| **Mass-scale Constitutional AI / RLAIF** | Requires generating + judging millions of model outputs | Smaller-scale exploration possible if needed |
| **State space models (Mamba/Mamba-2)** | Whole different architecture; ~2-3 week port; reuses almost nothing | Become a separate side-project (Tier 4) |
| **Diffusion language models** | Different paradigm; whole new codebase | Side-project |

## The honest summary

- **What we CAN build but haven't:** everything in Tiers 1-3 of Part 1
  + the optimizers / data / interpretability / browser-perf items in
  Part 2 + the phased plan in Part 6. ~50 distinct items, ~10 weeks
  of focused work total.
- **What we CAN'T build right now:** the items in Part 7 above. The
  blockers are real, but **none of them prevent us from shipping a
  genuinely useful artifact** at the 100M-1B scale on one Mac.
- **What we COULD build but probably shouldn't:** Tier 4 items
  (fp16, ZeRO at single-device, RLHF/PPO, etc.) — superseded by
  better alternatives.

---

## Doc readiness checklist

- ✅ Exhaustive landscape (Tiers 1-4 + orthogonal categories)
- ✅ Recent research with arxiv links (2024-2026, web-verified)
- ✅ Open-source datasets with URLs + licenses
- ✅ Phased executable roadmap (Part 6)
- ✅ "What we can't add right now" (Part 7)
- ✅ Honest knowledge-cutoff acknowledgment
- ✅ Cross-references to other docs (training_phases, memory_tradeoffs,
  leaderboard, perf_quest)

**This doc is ready** to be the master reference for "what's worth
building on TinyGPT." Update path: when new research lands or items
ship, edit in place (the file is the source of truth; the chat is
ephemeral).

---

## Cross-reference

- [`docs/training_phases.md`](training_phases.md) — pretrain → SFT →
  DPO pipeline (current form)
- [`docs/memory_tradeoffs.md`](memory_tradeoffs.md) — bf16, grad accum,
  grad checkpointing
- [`docs/leaderboard.md`](leaderboard.md) — benchmark framework
- [`docs/parked_multi_model.md`](parked_multi_model.md) — MoE park
- [`docs/perf_research.md`](perf_research.md), `docs/perf_quest.md` —
  browser-side performance levers
- [`docs/precision.md`](precision.md) — the fp32/fp16/bf16 study
