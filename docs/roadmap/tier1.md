# Roadmap — Tier 1 (high ROI, build next)

1-3 days each, visible product or educational improvement.

Status legend: 🟢 shipped · 🟡 partial · ⬜ not yet built · 🟣 parked.

## 1.1 Knowledge distillation 🟢

A small "student" model is trained to match a large "teacher" model's
*logits* (full probability distribution), not just hard ground-truth
tokens. The student learns *what the teacher would have said*,
including the relative probabilities of plausible alternatives — a
much richer training signal than next-token classification alone.
5M-param student distilled from 100M-param teacher often retains
70-90% of the teacher's quality at 20× smaller — the canonical path
to "tiny models that are actually good." Two models in memory; KL
divergence loss; backprop through student only.

**Effort:** ~2 days. **ROI: very high.** See [`docs/distillation.md`](../distillation.md).

## 1.2 Sequence packing for SFT 🟢

SFT data is full of short examples; naively batching them wastes 90%
of positions on padding. Sequence packing concatenates many short
examples into one long sequence with a block-diagonal attention mask
preventing cross-example contamination. 5-10× SFT throughput on
Dolly-15k-shaped data.

**Effort:** ~1 day (custom mask via `MLXFast.scaledDotProductAttention`).
**ROI: high.**

## 1.3 QLoRA training (combine int4 base + LoRA) 🟡

Quantize the base to int4 (frozen), train fp16 LoRA on top. We have
int4 inference and LoRA training as separate paths; combining them
6× the memory budget — fine-tune 30B HF models instead of 13B on a
48 GB Mac.

**Effort:** ~1 day. **ROI: high.**

## 1.4 ORPO (Odds-Ratio Preference Optimization) 🟢

A 2024 alignment recipe that merges SFT and DPO into a single
training pass. Loss = SFT cross-entropy + a preference-aware
log-odds-ratio term. No separate reference model needed (saves ~½
of DPO's memory). Iterates faster than SFT+DPO separately;
comparable final quality on most benchmarks.

**Effort:** ~1 day (new loss function on top of existing SFT path).
**ROI: high — the modern simplest path to instruction-following.**

## 1.5 SimPO (reference-free DPO) 🟢

DPO without the reference model. Replaces the `logπ_pol - logπ_ref`
ratio with a length-normalized log-probability target. Half the
memory of DPO; ~equivalent final quality on published benchmarks.

**Effort:** ~half day (change the loss function in our DPO trainer).
**ROI: high — frees half the GPU memory for bigger batch sizes.**

## 1.6 NEFTune (noisy embeddings fine-tune) 🟢

During SFT/DPO, add small uniform noise to the embedding outputs
(not the weights, just the embeddings at forward time). Reported
+5-10 points on instruction-following benchmarks. Roughly 3 lines
of code. The least-effort biggest-impact one-line trick of 2024.

**Effort:** ~half day. **ROI: very high per minute spent.**

## 1.7 Gradient checkpointing 🟢

Don't save every layer's activations during forward; re-compute them
during backward. Trade ~30% extra compute for ~√L activation memory
reduction. Unlocks training Behemoth (404M) and Titan (1.3B) at full
batch.

**Effort:** ~2 days. **ROI: high if we want Behemoth/Titan training.**
Blocked on MLX-Swift exposing `mlx_checkpoint` — see
[`docs/memory_tradeoffs.md`](../memory_tradeoffs.md) for the workaround
levers.

## 1.8 Speculative decoding 🟢

Small "draft" model proposes K tokens; main model verifies K in one
forward pass. Accepted tokens are free, rejected ones cost one extra
forward. 2-4× sample throughput at no quality cost.

**Effort:** ~2 days. **ROI: high for browser sample UX.**

## 1.9 Browser-side benchmark runner 🟢

The leaderboard infrastructure exists but currently reads pre-computed
scores. A "Run benchmark on your loaded model" button in the browser
closes the submission loop end-to-end.

**Effort:** ~half day (worker plumbing). **ROI: high product win.**
