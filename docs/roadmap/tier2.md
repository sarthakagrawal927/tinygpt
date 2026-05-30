# Roadmap — Tier 2 (medium ROI)

Solid additions that aren't critical-path.

Status legend: 🟢 shipped · 🟡 partial · ⬜ not yet built · 🟣 parked.

## 2.1 KTO (Kahneman-Tversky Optimization) 🟢

DPO needs (chosen, rejected) pairs; KTO needs only single examples
labeled "good" or "bad." Much more data available (thumbs up/down
flows). Quality comparable to DPO on most benchmarks.

**Effort:** ~half day (different loss in DPO trainer). **ROI: medium-high
when paired data is scarce.**

## 2.2 IPO (Identity Preference Optimization) ⬜

DPO variant with stronger regularization toward the reference,
designed for small (~1K-pair) datasets where vanilla DPO overfits.

**Effort:** ~half day. **ROI: medium.**

## 2.3 DoRA (Weight-Decomposed LoRA) 🟢

Decompose each weight into magnitude + direction; LoRA the direction,
train magnitude scalars separately. ~5-10% better than vanilla LoRA
at the same rank. See [`docs/lora_guide.md`](../lora_guide.md).

**Effort:** ~1 day. **ROI: medium (free quality on every fine-tune).**

## 2.4 GaLore (Gradient Low-Rank Projection) 🟢

Projects gradients to a low-rank subspace before applying the
optimizer; same memory as LoRA but performs full fine-tuning (all
weights move, not just adapters). Especially useful for pretraining
where LoRA is too restrictive.

**Effort:** ~1.5 days. **ROI: medium-high — full-finetune at LoRA cost.**

## 2.5 VeRA (Vector-based Random Adapters) 🟢

LoRA variant with frozen random projection matrices and trainable
diagonal scalars only. ~10× smaller adapter than LoRA; comparable
quality on most tasks.

**Effort:** ~1 day. **ROI: medium (extreme adapter-size compression).**

## 2.6 LoftQ (LoRA-Friendly Quantization) 🟢

Initialize LoRA adapters to compensate for the quantization error
of the base — A and B are chosen so `A·B` approximates the original
fp32 weight minus its int4-quantized version. Improves QLoRA quality
vs naive initialization.

**Effort:** ~1 day (pairs with QLoRA in 1.3). **ROI: medium.**

## 2.7 AWQ / GPTQ quantization readers 🟡

AWQ and GPTQ are popular int4 storage *formats*, different from
MLX-Swift's built-in quantize. Many HF models ship in AWQ/GPTQ
already (e.g., `TheBloke/Llama-2-7B-AWQ`).

**Effort:** ~1 day per format. **ROI: medium (expands model menu).**

## 2.8 HQQ (Half-Quadratic Quantization) 🟢

Recent (2024) calibration-free int4 quantization. Comparable quality
to GPTQ but faster (~minutes vs hours to quantize a 7B model). No
calibration data needed.

**Effort:** ~1.5 days. **ROI: medium.**

## 2.9 Sliding window attention 🟢

Each token attends only to the last N tokens (e.g., 512). Used by
Mistral. Lets us train at ctx=4096 with memory that otherwise
allows only ctx=512.

**Effort:** ~1 day. **ROI: medium (only matters for long contexts).**

## 2.10 ALiBi position bias 🟢

Alternative to RoPE. Add a position-distance penalty to attention
scores instead of rotating Q/K. Extrapolates to longer contexts at
inference than seen during training — RoPE doesn't generalize as
well.

**Effort:** ~1 day. **ROI: medium.**

## 2.11 KV cache quantization (KIVI) 🟢

Compress K and V tensors in the KV cache to int8 or int4. At long
contexts, the cache is what runs out of memory. KIVI (2024) is a
recent technique with minimal quality loss.

**Effort:** ~1 day. **ROI: medium (long-context inference).**

## 2.12 Multi-Token Prediction (MTP) 🟢

DeepSeek-V3 / Meta MTP variant. Predict the next K tokens at each
position during training (extra heads), not just K=1. Better
training signal + enables speculative decoding via the same heads
at inference. ~10% perplexity improvement reported. See
[`docs/mtp.md`](../mtp.md).

**Effort:** ~2 days. **ROI: medium-high — both a training and
inference win.**

## 2.13 Multi-Query Attention (MQA) 🟢

Extreme GQA — one K/V head shared across all Q heads. Smaller KV
cache, faster decode. **Already supported** via our `nKvHeads`
config; setting `nKvHeads: 1` activates MQA.

**Effort:** zero. **ROI: free.**

## 2.14 Streaming-LLM attention sink 🟢

Keep the first 4 tokens permanently in attention, then slide a window
over the rest. Enables infinite-context streaming without quality
collapse (which naive sliding window suffers from). Particularly
useful for long chat.

**Effort:** ~1 day (mask change + cache management). **ROI: medium.**

## 2.15 Prefix / prompt caching 🟢

When the same prompt prefix is used across many generations, cache
its KV state once and reuse. Common patterns: system prompts,
few-shot examples, RAG retrieved chunks. 5-50× latency improvement
for cache-hit prompts.

**Effort:** ~1 day. **ROI: medium-high (matters once we have a UI
that does long system prompts).**

## 2.16 Prefix tuning / soft prompts ⬜

Instead of LoRA-ing the weights, train a small set of "virtual
tokens" prepended to every prompt. Base fully frozen; ~10K params
vs LoRA's ~100K-1M. Works well for narrow tasks.

**Effort:** ~1 day. **ROI: medium (niche but tiny adapter).**
