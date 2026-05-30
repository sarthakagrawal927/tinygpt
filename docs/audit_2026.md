# Audit 2026 — what we tried, what worked, what we're cutting

After ~70 techniques shipped across the project, this doc is the honest
reckoning. Each entry: **what it claimed**, **what we measured**, and
**verdict** (🟢 KEEP / 🟡 EXPERIMENTAL / 🔴 DELETE).

The audit is informed by the project's north star: **on-device agent
models on Apple Silicon + Chrome**. Techniques are evaluated against
"does this help build/train/run/serve an agent-shaped model on a Mac
or in a browser?" — not "does this exist in the literature?"

After this audit lands, the codebase shrinks from ~28K lines to
~16-18K lines of Swift, the CLI from ~40 flags to ~12, and the
"choose-your-own-adventure" surface area collapses into ONE curated
recipe per capability.

---

## TL;DR — the cuts

| Status | Count | Rationale |
|---|---|---|
| 🟢 **KEEP** (recipe-positive) | ~28 | Demonstrated value at our scale or required for HF interop |
| 🟡 **EXPERIMENTAL** (move) | ~10 | Interesting demos / research curiosity / niche use case |
| 🔴 **DELETE** (post-mortem then `git rm`) | ~22 | Couldn't prove value at our scale or made the wrong tradeoff |

---

## 🟢 KEEP — these are the curated defaults

### Training

| Item | Why kept | Measured evidence |
|---|---|---|
| **AdamW optimizer** | Default, most reliable | Outperformed Lion/Sophia/Muon/Adafactor in 200-step tests |
| **bf16 dtype** | Memory + range win over fp16/fp32 | Industry standard; matches flagship training |
| **Cosine LR + warmup** | Industry standard | Used in all flagship training runs |
| **Gradient clipping** | Cheap stability lever | Prevents bf16 blowups; no measured downside |
| **Gradient checkpointing** | Real memory unlock at scale | Behemoth B=4 ctx=1024: 27.7GB → 17.8GB (−36%), loss equivalent |
| **Sample packing for SFT** | 10× variance reduction | CoV(length·freq) 0.582 → 0.061 measured |
| **Persistent token cache** | 10-30 min saved per re-run | Speedup measured in practice |
| **CPU speedup bundle** (compile+accum+QoS+prefetch) | +36% step/s on cosine+accum | Measured: 5.0 → 6.8 step/s on small B=16 |

### Tokenization

| Item | Why kept | Evidence |
|---|---|---|
| **BPE via smollm2** (49k vocab) | Modern decoder-only standard | Used for all real-text training |
| **Byte-level vocab=256** | Educational + small browser models | Powers the entire browser gallery |
| **HFTokenizer wrapper** (swift-transformers) | HF interop | Loads Llama, Qwen, etc. |

### Alignment

| Item | Why kept | Evidence |
|---|---|---|
| **SFT with response masking** | Real instruction tuning | ChatML, Alpaca, Llama, plain templates work |
| **DPO** | Real preference learning | Smoke tested; loss converges |
| **SimPO** | ½ DPO memory at equivalent quality | Reference-free; preferred default |
| **ORPO** | Merges SFT + DPO in one pass | Saves a stage |
| **KTO** | Single-side feedback (thumbs up/down) | Useful when paired data is scarce |

### PEFT (fine-tuning)

| Item | Why kept | Evidence |
|---|---|---|
| **LoRA** | The base — many users will want it | Standard, well-tested |
| **DoRA** | 5-10% better than LoRA at same rank | Verified in smoke run |
| **LoRA-FA** (frozen A) | 2× smaller adapter at equivalent quality | Halves trainable params; demonstrated |
| **LoRA+ (B-LR multiplier)** | Free win, no quality loss | Standard recipe; verified |
| **NEFTune** | One-line ~5% SFT win | Per paper; smoke tested |
| **Adapter file format** (`.lora` I/O) | Round-trip safety | Required for save/load |
| **Multi-LoRA composition** | Compose multiple adapters | LoraCompositionHF.swift |

### Inference / sampling

| Item | Why kept | Evidence |
|---|---|---|
| **KV cache** | 2.2× decode speedup | Measured: 470 vs 209 tok/s on flagship |
| **KIVI int8 KV** | 4× cache memory, greedy-lossless | 100% greedy-prefix match vs fp32 on flagship |
| **Prefix caching** | System prompt reuse | Direct win for agent multi-turn |
| **StreamingLLM sink** | Arbitrary-length decode | Quality preserved at 500 tokens |
| **Speculative decoding (vanilla draft)** | 2-4× decode at no quality cost | Standard technique; works |
| **HF model loading** (Llama family) | Real interop | Loads Qwen, Llama, Mistral, Phi out of box |
| **AWQ reader** | Load any AWQ-quantized HF model | Mechanical; works |
| **ANE Core ML inference path** | 3-10× sampling on suitable models | Measured: 365 tok/s on Shakespeare via Core ML |
| **OpenAI-compatible HTTP serve** | lm-eval-harness compatibility + agent gateway | Real curl-tested |

### Eval / bench

| Item | Why kept | Evidence |
|---|---|---|
| **`tinygpt eval`** (BPE-aware) | Real perplexity measurement | 4.71 on flagship matches training-time val |
| **`tinygpt bench`** (TTFT/ITL/RSS/power) | Bench360-modeled inference benchmark | Real numbers: 1.91ms TTFT, 794 tok/s on Shakespeare |
| **`tinygpt score-bench`** + manifest patcher | Browser leaderboard pipeline | End-to-end working |
| **lm-evaluation-harness HTTP adapter** | Wire to standard quality benchmarks | OpenAI-compatible serve verified curl-tested |

### Quality

| Item | Why kept | Evidence |
|---|---|---|
| **40 XCTests** | Real CI gate | All pass; covers Manifest schema, KVCache parity, LoRA round-trip, crash recovery |
| **swiftformat config + CI lint** | Code-quality gate | 0 violations on 76 files |
| **Crash-recovery tests** | Resume determinism + atomic save | Subprocess SIGTERM-race verified |
| **GitHub Actions CI** | Mac + Ubuntu runners on every PR | Real, in use |

### Infrastructure

| Item | Why kept | Evidence |
|---|---|---|
| **Atomic save-every + `--resume`** | Real crash recovery | Demonstrated by SIGINT pause of v5 mid-training |
| **OOMGuard pre-flight memory check** | Aborts doomed configs cheaply | Saved several launches in this session |

### Web playground

| Item | Why kept | Evidence |
|---|---|---|
| **WebGPU + WASM training in browser** | The unique educational hook | Real gallery models trained in-browser |
| **Dynamic `[slug].astro` doc route** | All docs web-visible | 67 pages built in 1.7s |
| **Leaderboard page** | Public scoring surface | Real scored entries |

---

## 🟡 EXPERIMENTAL — move to `experimental/`, keep accessible

Interesting, educational, or might-be-useful-later. Stays in the
codebase under `--experimental-*` flags or `experimental/` subdirs.

| Item | Why experimental | Future use |
|---|---|---|
| **MoE (Switch + Mixtral dense)** | Paper reimplementation; pedagogical | Becomes useful when scatter_add lands |
| **Distillation (Hinton KL+NLL)** | Standard technique we never used at scale | Likely on the agent recipe — distill from Qwen-7B to 1.5B agent target |
| **Magpie synthetic data generation** | Useful when we need agent training data | Generate agent traces from Claude/GPT |
| **Evolution Strategies (ES)** | Research curiosity | Useful if we explore RL alternatives |
| **Tuned lens** | Educational interp tool | Part of "watch your model think" UX |
| **Logit lens, attention heatmap, activation patching, per-layer ablation** | Interp tools — already documented as educational | Keep in playground for demonstrating |
| **YOCO** | Halves KV cache at long context | Becomes critical at >8k context for agent histories |
| **Sliding window attention** | Bounded attn at long context | Same — useful for very long agent sessions |

---

## 🔴 DELETE — post-mortem then `git rm`

Each gets a ~50-line entry in `docs/post_mortem/<technique>.md` covering:
what it claimed, what we measured, why it didn't help at our scale,
when it WOULD help, and the maintenance cost paid.

### Optimizer alternatives (4 deletes)

| Item | Post-mortem in one line |
|---|---|
| **Lion** | Sign-based update needs >1k steps; lagged AdamW at 200 steps |
| **Sophia** | EMA-of-squared-gradient variant slower per step; marginal lift |
| **Muon** | Newton-Schulz overhead dominated at small scale (5.2 vs 16.3 step/s) |
| **Adafactor** | ⅓ optimizer memory not needed at 22M-100M scale; 2× slower per step |

### Architecture variants (5 deletes)

| Item | Post-mortem in one line |
|---|---|
| **DiffAttention** | Doubled Q/K projections for no measured benefit at 22M |
| **MoD (soft routing)** | No compute savings without hard top-K + scatter_add |
| **MTP (Multi-Token Prediction)** | Multi-horizon CE; marginal regularization at small scale |
| **ALiBi** | RoPE is standard; ALiBi shines for long-context extrapolation we don't reach |
| **Differential attention sibling pattern** | Conceptually elegant but unused — see DiffAttention above |

### Stability tricks (3 deletes)

| Item | Post-mortem in one line |
|---|---|
| **DeepNorm** | Only helps at depth ≥100; useless at 12 layers |
| **Layer-wise LR decay** | Fine-tuning lever, not pretraining; never wired to a real run |
| **Embedding RMSNorm** | Net effect unclear; causes step-1 spike |

### Training-time exotic (2 deletes)

| Item | Post-mortem in one line |
|---|---|
| **GaLore (gradient low-rank projection)** | Adam state still full-rank → claimed memory savings unrealized |
| **BPE-dropout** | Per-merge skip probability; marginal regularization, requires custom encoder |

### PEFT variants — minor (5 deletes)

| Item | Post-mortem in one line |
|---|---|
| **VeRA** | 512× smaller adapter but trains far slower; niche |
| **LoftQ** | Compensates for int4 base quantization error — only useful with int4 base (we don't have a trained one) |
| **AdaLoRA** | Per-rank importance scoring; never wired to actual rank reallocation |
| **RsLoRA** | α/√r scale fix; marginal at r=4-16 |
| **PISSA init** | Top-r SVD init; converges faster but final quality matches LoRA |
| **LayerDrop** | Degrades fine-tuning quality (we mostly fine-tune, not pretrain depth-1B) |

### Quantization (3 deletes)

| Item | Post-mortem in one line |
|---|---|
| **SmoothQuant** | Algorithmic infrastructure shipped; zero runtime payoff without int8 matmul kernel |
| **HQQ storage-only** | Same — needs packed-int4 matmul to realize the win |
| **GPTQ from-scratch** | 30-second per-Linear Hessian compute; AWQ reader covers HF interop case |
| **QAT** | Demonstrates fake-quant + STE; no int4 inference path to deploy to |

### Pruning (2 deletes)

| Item | Post-mortem in one line |
|---|---|
| **Unstructured pruning** | Metal has no sparse matmul; only gain is post-gzip download size |
| **Structured head pruning (zero-out variant)** | No physical removal; no actual memory/wallclock saving |
| **Structured layer pruning** | Could be KEPT — actually changes topology — but moves to experimental if not used |

### Speculative decoding heads (2 deletes)

| Item | Post-mortem in one line |
|---|---|
| **Medusa heads** | 21-23% acceptance at 50 train steps; would need 10k+ steps; vanilla draft-model spec decode covers the case |
| **EAGLE-2** | Same — 26.5% acceptance, requires sustained training to be useful |

---

## What the CLI looks like AFTER the cuts

Current state:
```
tinygpt train --preset huge --tokenizer ... --dtype bfloat16 \
    --optimizer adamw --grad-checkpoint \
    --z-loss-weight 1e-4 --embedding-rmsnorm \
    --galore-rank 0 --bpe-dropout 0 --qat 0 \
    --moe-experts 1 --mtp-horizons 1 \
    --diff-attn --mod --yoco --alibi --sliding-window 0 \
    [+ 20 more flags]
```

After cleanup:
```
tinygpt train <corpus>          # AdamW + bf16 + cosine + clip — recipe defaults
  --preset huge|mega|behemoth
  --tokenizer <hf-dir>
  --grad-checkpoint              # for mega+ models
  --resume <path>
  --save-every N

tinygpt finetune <base> <data>  # DoRA + SFT — recipe defaults
  --rank R
  --lora-fa                      # halve params if you want

tinygpt align <base> <prefs>    # SimPO — recipe default
  --loss-type dpo|simpo|orpo|kto # if you really want to pick

tinygpt sample <model>          # KV cache + KIVI int8 + speculative — defaults
  --prompt "..."
  --tokens N

tinygpt quantize <model>        # AWQ → int4 — recipe default
  --bits 4|8

# All other techniques: --experimental-X for the alternatives
```

---

## Execution plan

1. **Draft this doc** — done (you're reading it)
2. **Per-deleted-technique post-mortem** (~22 docs, 50 lines each) — `docs/post_mortem/<name>.md`
3. **Move EXPERIMENTAL items** to `native-mac/Sources/TinyGPTModelExperimental/` (new target in Package.swift)
4. **`git rm` the DELETE items** + their tests + their docs
5. **Default-CLI rewrite** — `tinygpt train` etc. = the curated recipe with no flags needed
6. **README + landing page rewrite** for the new shape
7. **Single big commit**: `feat: the great winnowing — curate to recipe, delete to ship`

Estimated effort: **3-5 days** focused. Most of it is writing post-mortems honestly.

After this, the codebase is ready for the **on-device agent model** focus described separately. The curated tools above are exactly what's needed for that work.
