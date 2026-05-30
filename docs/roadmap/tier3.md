# Roadmap — Tier 3 (niche or specialized)

Build when there's a specific use case.

Status legend: 🟢 shipped · 🟡 partial · ⬜ not yet built · 🟣 parked.

## 3.1 Constitutional AI / RLAIF ⬜

Use a stronger model as judge to critique + revise weaker model
outputs. Anthropic's approach to alignment without humans. Gated on
access to a strong local judge.

**Effort:** ~3 days. **ROI: low for us (no strong local judge).**

## 3.2 GPTQ quantization (from-scratch) 🟢

Layer-by-layer int4 quantization using Hessian information.
Marginally better than MLX's built-in `quantize`; the diff is ~1-2%
perplexity.

**Effort:** ~3 days. **ROI: low (diminishing returns).**

## 3.3 SmoothQuant 🟢

Equalize activation/weight magnitudes before int8 quantization for
cleaner calibration. Less relevant since modern int4 methods (HQQ,
LoftQ) work without this preprocessing.

**Effort:** ~2 days. **ROI: low.**

## 3.4 Quantization-aware training (QAT) 🟢

During training, simulate int4 quantization in the forward (with
straight-through estimator gradients). Model learns to be quantization-
robust. ~2-3% better post-quantization perplexity vs PTQ.

**Effort:** ~3 days. **ROI: low at our scale.**

## 3.5 Pruning (unstructured) 🟢

Zero out the smallest weights. 50% sparsity often retainable. But
Mac/browser GPUs don't efficiently exploit unstructured sparsity —
runtime savings are marginal vs quantization.

**Effort:** ~2 days. **ROI: low.**

## 3.6 Structured pruning (head/layer removal) 🟢

Remove whole attention heads or layers. Real speedup (vs unstructured's
marginal). 10-20% smaller. Tricky because removed pieces cascade.

**Effort:** ~3 days. **ROI: low-medium (distillation usually wins).**

## 3.7 LASER (selective rank reduction) 🟢

Drop the *highest* singular-value components of specific MLP layers.
Counterintuitively often *improves* downstream quality (acts like a
regularizer + removes "memorized" noise). One-line post-training fix
for some quality issues.

**Effort:** ~half day (just SVD + truncate selected layers).
**ROI: medium-low; cheap to try.**

## 3.8 Cross-layer KV sharing 🟡

Have multiple transformer layers share the same K/V tensors.
Smaller KV cache. Used in You Only Cache Once (YOCO), Llama-3.2
small variants.

**Effort:** ~2 days. **ROI: low-medium.**

## 3.9 RLHF / PPO ⬜

The original RLHF: train a reward model, PPO against it. ChatGPT's
original method. DPO is the modern replacement at 1/5 the complexity
for 80-90% of the quality lift.

**Effort:** ~1 week. **ROI: low — DPO does the job.**

## 3.10 Medusa-style speculative decoding 🟢

Speculative decoding where the model has extra "Medusa heads"
predicting future tokens directly. No separate draft model, but
requires training the heads. (Tier 1 vanilla speculative decoding
is simpler.)

**Effort:** ~3 days. **ROI: low.**

## 3.11 EAGLE-2 (better speculative decoding) 🟢

2024 variant of speculative decoding with 2-3× higher acceptance
rates than vanilla. Uses the target model's hidden states (not just
logits) to guide the draft.

**Effort:** ~3 days. **ROI: low (incremental over vanilla
speculative).**

## 3.12 Mixture of Experts (MoE) 🟢

N expert MLPs per layer, router picks top-K per token. Modern
frontier models (Mixtral, GPT-4) are MoE. At our scale, an 8×16M
MoE behaves like 128M capacity in 32M compute. Educational
value > practical value at 100M scale. See [`docs/moe.md`](../moe.md).
Parked notes in [`docs/archive/parked_multi_model.md`](../archive/parked_multi_model.md).

**Effort:** ~1 week. **ROI: high educationally, medium practically.**

## 3.13 Differential attention 🟢

2024 paper. Replaces standard attention with the *difference* of two
attention maps (subtract noise pattern). Modest perplexity wins,
better long-context recall. Not yet widely adopted.

**Effort:** ~2 days. **ROI: low.**

## 3.14 Mixture of Depths (MoD) 🟢

Router decides which tokens go through each transformer layer (some
skip layers entirely). Saves compute on "easy" tokens. Recent
(2024).

**Effort:** ~3 days. **ROI: low-medium.**

## 3.15 LayerDrop 🟢

Randomly skip layers during training; at inference, optionally use
fewer layers for faster decode. Mild regularizer + post-hoc model
shrinking.

**Effort:** ~1 day. **ROI: low.**

## 3.16 ReLoRA ⬜

Periodically merge LoRA into base + restart training with fresh
LoRA. Accumulates "high-rank" effective updates over many merges.
Closes some of the gap between LoRA and full fine-tune.

**Effort:** ~2 days. **ROI: low-medium.**

## 3.17 AdaLoRA 🟢

Adaptive rank assignment — automatically allocates higher rank to
layers that need it, lower to layers that don't. Slightly better
than fixed-rank LoRA at the same param budget.

**Effort:** ~2 days. **ROI: low.**

## 3.18 LoRA+ 🟢

Different learning rates for the A and B matrices of LoRA (B's LR
is ~16× larger). ~5-10% quality improvement at zero memory cost.

**Effort:** ~half day. **ROI: medium-low.**

## 3.19 PISSA initialization 🟢

LoRA initialized using the principal singular vectors of the base
weight (instead of random). Faster convergence — sometimes 2× faster
to reach the same loss.

**Effort:** ~1 day. **ROI: medium-low.**

## 3.20 LoRA-FA (frozen A) 🟢

Train only B in LoRA; freeze A. Half the params, ~80% of the
quality. Useful when memory is extreme.

**Effort:** ~half day. **ROI: low.**

## 3.21 RsLoRA (rank-stabilized LoRA) 🟢

Different scaling for the LoRA delta (`α/√r` instead of `α/r`).
Improves quality at high ranks (r > 64) where standard LoRA scaling
hurts.

**Effort:** ~half day. **ROI: low (most fine-tunes use r ≤ 16).**
