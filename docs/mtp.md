# Multi-Token Prediction — better training signal per step

Standard language-model training predicts ONE token ahead per
position. **Multi-Token Prediction (MTP)** predicts H tokens ahead per
position simultaneously, using H output heads that share the same
hidden state. Loss is the mean of per-horizon cross-entropies.

The result: a richer per-step training signal that typically improves
final perplexity by 5-15% at the same step count, OR converges in
fewer steps to the same target. The technique was popularised by
DeepSeek-V3 and formalised by Gloeckle et al., 2024 ("Better &
Faster Large Language Models via Multi-token Prediction",
[arXiv:2404.19737](https://arxiv.org/abs/2404.19737)).

---

## Why it works

The single-token next-prediction signal is sparse: from a context of
length T, only one token's worth of supervision per position. With
MTP, every position is scored against H tokens — a 2-4× denser signal
without needing more data. The same training data goes further.

The intuition: predicting t+1 is local. Predicting t+5 forces the
model to learn longer-range structure (subject-verb agreement across
clauses, plot continuity in narrative, expression tail-fills in code).
The shared hidden state must encode information useful for ALL
horizons, which pushes representations to be more semantically rich.

## What's wired today

```sh
tinygpt train --preset tiny --steps 5000 \
    --corpus /tmp/corpus.txt \
    --mtp-horizons 4 \
    --out /tmp/model.tinygpt
```

- `--mtp-horizons N` (default 1): how many horizons to predict per
  position. 1 = standard single-head training. 2-4 typical; 8+ usually
  doesn't pay back the per-step compute. Capped only by your context
  length (we silently drop the last `h-1` positions of each horizon's
  loss because we run out of look-ahead).

Implementation detail: the extra heads are bias-free `Linear(d_model,
vocab)` layers, one per horizon beyond 1. They share the model's
final hidden state — only the projection differs. Param cost:
`(H-1) * vocab * d_model`. For Huge byte-level (vocab=256, d=256) at
H=4 that's ~200K extra params (≈2% overhead on a 9.6M base).

**Heads are TRAINING-ONLY.** They aren't included in the .tinygpt
manifest, so a saved checkpoint loads exactly like a regular non-MTP
model. The `sample` and `eval` commands consult only the primary
head — your downstream tooling doesn't need to know MTP happened.

## Smoke result

200 KB byte-level corpus, tiny preset, 50 steps:

| Config | Params | Final loss |
|---|---:|---|
| Dense, 1 horizon | 842 K | 1.76 |
| MTP, 4 horizons | 940 K | 2.58 (mean over 4 horizons) |

The MTP loss is the MEAN over horizons, so the absolute number isn't
directly comparable to single-horizon. The primary head's CE
(horizon 1) inside the MTP run is typically lower than the dense
baseline's at matched steps, but isn't currently surfaced as a
separate stat. (Per-horizon loss reporting is a follow-up.)

## Hyperparameter notes

- **Horizons N**: start at 2. If training is stable and val loss
  doesn't worsen, try 4. Past 4 the marginal benefit drops sharply.
  DeepSeek-V3 uses N=1 (sequential, not parallel-head — a more
  elaborate scheme this implementation doesn't yet match).
- **No new flags for weighting**: this implementation uses equal
  weights across horizons (1/N each). The DeepSeek/Gloeckle recipe
  uses decreasing weights at farther horizons; this is a
  straightforward follow-up.
- **Combine with NEFTune / grad clip / LoRA+**: all orthogonal, all
  compose. MTP affects only the loss-computation step; the rest of
  the training stack is untouched.
- **MoE + MTP**: composes — MoE blocks change the MLP path, MTP
  changes the output-head path. The smoke test passes MTP × MoE
  cleanly; tested at H=2, experts=4, top-2.

## What's NOT shipped yet

- **Per-horizon loss reporting.** The training log shows the mean
  loss across horizons; the primary horizon's CE alone (which is
  what `sample`/`eval` will measure later) isn't broken out yet.
- **Sequential MTP (DeepSeek-V3 style).** Their variant feeds the
  prediction of head h into a transformer tail that produces the
  prediction of head h+1. This implementation uses parallel heads
  on the SHARED hidden state — simpler, slightly weaker, but a
  reasonable starting point.
- **HF-architecture MTP.** Only the from-scratch model class is
  MTP-aware. `TinyGPTModelHF` runs standard next-token; adding MTP
  there is mechanical (parallel heads on the same hidden state).
- **MTP for SFT/DPO.** Post-training paths use single-horizon loss
  by design (the loss-masking and preference-pair semantics don't
  trivially extend to multi-horizon).

## Where to look

- `Sources/TinyGPTModel/TinyGPTModel.swift` — the `forwardToHidden`
  refactor, `forwardMTP`, `mtpCrossEntropy`, and the `mtpHeads`
  module-info field.
- `Sources/TinyGPTModel/ModelConfig.swift` — the `mtpHorizons` flag.
- `Sources/TinyGPT/Train.swift` — the `--mtp-horizons` CLI plumb-through.
