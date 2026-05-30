# Roadmap — Tier 5 (2026 frontier — research artifacts)

Tiers 1-4 cover the **2017-2024 transformer toolbox**: every published
technique that fits on a single Mac. Tier 5 is the **2026 frontier** —
items that exist at the lab scale today and that we want to attempt at
small (single-Mac) scale, with honest expectations about negative
results.

These are research-flavoured items, not product-flavoured. The
"output" of each is typically a paper-shaped artifact + reproducible
code + a position on a qualifying scaling curve, NOT a polished UX
feature. Effort estimates are agent-hours-equivalent; calendar
time can be longer when waiting on cloud compute / data generation.

Status legend: 🟢 shipped · 🟡 partial · ⬜ not yet built · 🟣 parked

## 5.1 Reasoning training on a tiny model ⬜

**Frontier (2024-2026):** OpenAI o1 (Sep 2024) → DeepSeek-R1 (Jan 2025)
showed that **RL on chain-of-thought traces** can teach a model to
"think" before answering, with quality scaling with inference compute.
R1's recipe:

- Cold-start SFT on a few thousand curated CoT examples
- GRPO (a PPO variant) on a math/reasoning corpus with rule-based reward
- Iterative refinement with rejection sampling

**Our 22M-scale version:**

- Take a trained base, SFT on ~5k GSM8K CoT traces (or synthesize via
  Magpie from Qwen3-1.5B as teacher)
- DPO with `chosen` = correct CoT vs `rejected` = wrong CoT pairs
  (uses existing `tinygpt dpo` infrastructure)
- Optional: implement a tiny GRPO loop (~1 week of new code)

**Expected outcome:**

- 22M is well below the threshold where CoT consistently helps —
  research suggests "emergence" of reasoning around ~1-3B params.
- The interesting result is the negative one: **measure when CoT
  starts mattering at small scale**, plot the curve, publish.
- Realistic: model produces stylistically CoT-flavoured output but
  doesn't actually solve more problems.

**Effort:** ~5-7 days end-to-end. Code is small; the real work is
curating/generating training data.

**Key papers:**
[DeepSeek-R1 (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) ·
[Quiet-STaR (Zelikman 2024)](https://arxiv.org/abs/2403.09629) ·
[STaR (Zelikman 2022)](https://arxiv.org/abs/2203.14465)

## 5.2 Test-time compute scaling ⬜

**Frontier (2024-2026):** o1's headline finding — "more thinking =
better answer". DeepMind's
[Snell et al. 2024](https://arxiv.org/abs/2408.03314) showed test-time
compute can be more cost-effective than scaling parameters. Methods:

- **Best-of-N**: sample N responses, score with a verifier, return best
- **Beam search**: maintain N partial sequences, prune by reward
- **Process reward models (PRM)**: score each reasoning step
- **MCTS** on CoT branches

**Our version:** All on existing `tinygpt sample`:

- Wire best-of-N with greedy + temperature variants
- Use TinyStories perplexity or a small reward classifier as verifier
- Plot quality-vs-inference-compute curves (the "scaling law" plot)

**Expected outcome:**

- 22M will benefit modestly from best-of-N (the standard 5-15% lift).
- Genuine artifact: a reproducible **quality-vs-FLOPs** plot at
  22M-scale, matching Snell et al.'s methodology.
- The most cleanly publishable item — small contribution, sound
  methodology.

**Effort:** 3-5 days. Mostly extending `Sample.swift` + a verifier.

**Key papers:**
[Snell et al. 2024](https://arxiv.org/abs/2408.03314) ·
[Brown et al. (large language monkeys) 2024](https://arxiv.org/abs/2407.21787)

## 5.3 Vision-language toy ⬜

**Frontier (2024-2026):** LLaVA → LLaVA-NeXT (2024) → Qwen2-VL → Gemini
2 multimodal. Standard recipe: frozen CLIP/SigLIP vision encoder →
2-layer MLP projector → into LM's embedding space; train on
image-caption pairs.

**Our version:**

- Add a ViT-Base (or tiny custom ViT) as vision encoder
- 2-layer MLP projector → tinygpt embedding space
- Train on COCO captions (~118k images, manageable on a Mac)
- Eventually: visual instruction tuning on LLaVA-Instruct subset

**Expected outcome:**

- At 22M-base + small ViT, you get coherent image captions
  ("a dog on grass") but not detailed VQA.
- This is the most ambitious of the five — biggest scope, biggest
  payoff.
- Genuine novelty: smallest published VL model from-scratch on
  consumer hardware.

**Effort:** ~2 weeks honest. Vision is genuinely new code surface.

**Key papers:**
[LLaVA (Liu 2023)](https://arxiv.org/abs/2304.08485) ·
[Qwen2-VL technical report](https://arxiv.org/abs/2409.12191) ·
[SigLIP (Zhai 2023)](https://arxiv.org/abs/2303.15343)

## 5.4 Diffusion language model micro-implementation ⬜

**Frontier (2024-2026):** Inception Labs' Mercury (Sep 2025) showed
discrete diffusion LMs can match autoregressive at similar quality
with much faster inference (parallel decoding). Earlier:
[SEDD (Lou 2023)](https://arxiv.org/abs/2310.16834),
[Plaid (Gulrajani 2023)](https://arxiv.org/abs/2305.18619),
[MDLM (Sahoo 2024)](https://arxiv.org/abs/2406.07524).

Paradigm shift: instead of next-token prediction, train to **denoise
corrupted token sequences**. At inference, start with all-`[MASK]`
and iteratively denoise.

**Our version:**

- New model class: `DiffusionLM.swift` with masked-prediction loss
- Training: corrupt random fractions of input, predict masked positions
- Inference: start with all-`[MASK]`, iterate `T` denoising steps
- Reuse the existing transformer block — only the head + loss +
  sampling change

**Expected outcome:**

- A working 22M diffusion LM. Quality probably below the autoregressive
  baseline at this scale (diffusion shines with scale), but a
  **completely novel paradigm** added to the playground.
- Educational value is huge — "how does a diffusion LM actually
  work?" with running code.

**Effort:** 1-2 weeks. Real code volume but conceptually clean.

**Key papers:**
[SEDD (Lou 2023)](https://arxiv.org/abs/2310.16834) ·
[MDLM (Sahoo 2024)](https://arxiv.org/abs/2406.07524) ·
[Mercury technical report (Inception Labs)](https://www.inceptionlabs.ai/)

## 5.5 Real sparse MoE kernels ⬜

**Frontier (2024-2026):** DeepSeek-V3 (Dec 2024) — 671B total, 37B
active. Mixtral 8x7B (Dec 2023). The headline: at the same active-params,
MoE can have 10× more "knowledge" via the sparse experts.

Our current `MoE.swift` is **dense** — we compute all experts and weight
by the router. Real MoE needs `scatter_add` or similar dispatch
primitives to actually skip the unused experts. MLX-Swift doesn't
expose these.

**Our version:**

- Write a custom Metal kernel for grouped expert dispatch
- Wire into the existing `MoE.swift`
- Measure: does it actually save FLOPs?

**Expected outcome:**

- Real per-token FLOP reduction (not just param count).
- Unblocks training larger-capacity MoE models on the Mac.
- Mostly a kernel-engineering result, not a research result.

**Effort:** 2-3 weeks. Custom Metal kernel work is genuinely hard —
write + debug + tune + verify against the dense baseline.

**Key papers:**
[DeepSeek-V3 (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437) ·
[Mixtral of Experts (arXiv 2401.04088)](https://arxiv.org/abs/2401.04088) ·
[Switch Transformer (Fedus 2021)](https://arxiv.org/abs/2101.03961)

## Infrastructure that unblocks Tier 5

These aren't research items but they're prerequisite tooling.

### Cloud save/load pipeline ⬜

CF R2 push/pull for training checkpoints (~$1.50/mo for 100 GB,
zero egress fees). Lets training-heavy plans (5.1, 5.3) run on rented
GPU instances rather than burning the Mac's thermal budget.

Design done; see [Tasks #62] in the in-conversation task list.

**Effort:** ~2-3 hours when authorized.

### Pausable training bundle ⬜

Make training friendly to interruption beyond what's already shipped
(cooperative SIGINT + `--resume`):

- Auto-throttle on GPU contention (detect concurrent MLX processes)
- Battery-discharge awareness (pause when laptop is power-starved)
- Inter-process GPU lock (sysctl or file-lock coordination)
- Time-of-day scheduler (`--train-only-between 22:00-08:00`)
- Auto-resume on boot (LaunchAgent)
- Power budget flag (`--max-watts 30`)

**Effort:** ~3-5 days. Foundation+IOKit Swift work.

## Ranking by fit + feasibility + publishability

| # | Item | Frontier-fit | Mac-scale feasibility | Risk of negative result | Publishability |
|---|---|---|---|---|---|
| **5.2 Test-time compute** | ★★★★ | ★★★★★ | low | ★★★★ (clean methodology) |
| **5.1 Reasoning training** | ★★★★★ | ★★★ (22M is below emergence) | high | ★★★★ (interesting negative) |
| **5.4 Diffusion LM** | ★★★★ | ★★★★ | medium | ★★★ (paradigm demo) |
| **5.3 Vision-language** | ★★★ | ★★★ (biggest scope) | medium | ★★★★ (smallest from-scratch VL) |
| **5.5 Sparse MoE kernel** | ★★★ | ★★★★ (we know it's doable) | low | ★★ (engineering, not research) |

## Suggested order if doing all 5

- **Phase A** (independent, parallelize): 5.2 + 5.5
- **Phase B** (uses a trained base, after Phase A): 5.1
- **Phase C** (focused projects): 5.4, then 5.3

Total wall time at parallel-agent pace: **~4-6 weeks** if you actually
want all five shipped well.
