# Session handoff — pick this up cleanly

A fresh-context agent should read this first, then `docs/single_machine_roadmap.md`.

## What's running right now

**Background training (do NOT kill — user explicitly wants it running):**

- **PID 95977** — `tinygpt train --preset huge --tokenizer /tmp/smollm2 --corpus /tmp/fineweb-edu-500M.txt --dtype bfloat16 --batch 4 --accum 4 --ctx 512 --steps 5000 --save-every 100 ...`
- Output: `/tmp/huge-fineweb.log` (currently empty — still in BPE encode phase, ~30-60 min before first step)
- Checkpoint: `/tmp/huge-fineweb.tinygpt` (atomic save every 100 steps once training proper begins)
- Wrapped under `caffeinate -di` to prevent display+idle sleep

**Background process inventory at handoff time** (`pgrep -fl 'tinygpt train'`):
```
95977  tinygpt train --preset huge ...
95978  caffeinate -di tinygpt train ...
```

## Why we're on Huge not Mega

The Mega-bf16 attempt (100M params, ctx=1024, B=4×accum=4) **died at step 1** with no error message in the log. Almost certainly OOM during the first backward — 8 GB allocated, mostly compressed by macOS. The crash happened **before** save-every-1000 fired, so nothing was saved.

The current Huge run cuts the activation memory ~4× (preset is 9.6M body params, ctx=512 vs Mega's 24L/d=512/ctx=1024) and saves every 100 steps for crash-resilience. This is the de-risked retry.

## Source of truth: the roadmap doc

**`docs/single_machine_roadmap.md`** (1435 lines, 7 parts) is the master plan for what to build. Don't reinvent — read it first.

Quick TOC:
- **Part 1** — Tier 1-4 ROI ranking
- **Part 2** — orthogonal categories (optimizers, training stability, data, tokenization, interpretability, inference, browser perf, architecture, PEFT, infra)
- **Part 3** — top-10 recommended order
- **Part 4** — open-source datasets (with verified URLs + licenses)
- **Part 5** — recent research, 2024-2026, web-verified with arxiv links
- **Part 6** — **the phased roadmap** (10 phases × ~1 week each)
- **Part 7** — what we can't add right now (categorized blockers)

## What was just decided + interrupted

The user kicked off goal: *"phase one, phase two, phase three, phase four are small things that you can do right away"* — meaning execute Phases 1-4 of Part 6. Then halted the session before any code shipped.

**Tasks marked `in_progress` but with zero code written:**

| Task | Status at interrupt | Files to touch |
|---|---|---|
| #149 P1: NEFTune | reads only, no edits | `Sources/TinyGPT/SFT.swift`, `Sources/TinyGPT/DPO.swift` — add small uniform noise to embedding output during forward (alpha=5 typical) |
| #150 P1: Gradient clipping | reads only, no edits | `Sources/TinyGPTModel/Trainer.swift`, `Sources/TinyGPTModel/TrainerHF.swift` — clip total grad norm (~1.0) before optimizer step |
| #151 P1: LoRA+ | reads only, no edits | `Sources/TinyGPTModel/Lora.swift` (`LoraLinear`) — add `loraBLrMultiplier` field (default 16); thread through trainer via per-param LR group or override in optimizer call |

**Other Phase 1-4 tasks (pending, untouched):** #152 through #160. Full descriptions in `TaskList`.

**No code was modified in this session-tail**, so nothing is in a half-shipped state. Last successful build was after the documentation work — repo should `xcodebuild` cleanly.

## What shipped earlier this session (already on disk)

These ARE on disk and merged — do not redo:

**Mac CLI new commands / features:**
- `tinygpt sft` — SFT with response-only loss masking + ChatML/Alpaca/Llama/plain templates
- `tinygpt dpo` — DPO trainer (policy + frozen reference, beta param, log-sigmoid loss)
- `tinygpt train --dtype bfloat16` — bf16 training, parity-verified
- `tinygpt train --accum N` — gradient accumulation
- `tinygpt train --tokenizer <hf-dir>` — BPE training for from-scratch path
- `tinygpt train --ctx N` — context length override
- `tinygpt train --resume <path>` + `--save-every N` (atomic) + cooperative Ctrl-C + cosine LR + val loss tracking
- `tinygpt sample --quantize int4|int8` — MLX `quantize` integration
- Multi-LoRA composition for HF models (`LoraCompositionHF.swift`)
- HF KV cache (RoPE offset + GQA correct)
- HF-tokenized fine-tune path
- SwiftUI Fine-tune tab

**Browser:**
- Auto-save URL param (`?autoSave=NAME`) — fires download as soon as training completes; fixes the Sea Tales failure mode
- `train_gallery_one.mjs` reordered: save checkpoint FIRST, sample after (sample failure no longer loses model)
- `score_gallery.mjs` + `score_gallery_tasks.mjs` — Node + WASM scorers for the three launch benchmarks
- Leaderboard page at `/leaderboard.html` — 3 tabs (TinyStories PPL, Sort-6, Reverse-16), 5 scored entries
- Manifest schema extended with `submission`, `benchmarks`, `featured` fields
- Benchmark engine types: `browser/src/benchmarks/{types,registry,tinystories-ppl,sort-6,reverse-16}.ts`

**Docs:**
- `docs/single_machine_roadmap.md` (1435 lines, source of truth)
- `docs/training_phases.md` (pretrain → SFT → DPO tutorial)
- `docs/memory_tradeoffs.md` (bf16, grad accum, grad checkpointing)
- `docs/leaderboard.md` (benchmark framework)
- `python_ref/fetch_hf_corpus.py` (HF dataset streaming importer)
- README.md updated to link the new docs

**Tests:** 14/14 still passing as of the latest test run.

## Where to start next

**Smallest valuable next move (if you've got ~3 hours):** finish Phase 1 — ship NEFTune, gradient clipping, LoRA+, persistent tokenized cache, browser-side benchmark runner. ~5 small changes, all independent.

**Per-item kickoff hints:**

1. **NEFTune (#149)** — In `SFT.swift` and `DPO.swift`, after embedding lookup but before block layers, add: `embed = embed + uniformNoise(scale = alpha / sqrt(embed_dim * seq_len))` with `alpha = 5.0` typical. Off by default; enable via `--neftune-alpha 5.0` flag.

2. **Gradient clipping (#150)** — In `Trainer.step` and `Trainer.accumulatedStep`, after `gradFn(...)` returns `grads` and before `optimizer.update(...)`, clip the gradient by global L2 norm. MLX-Swift exposes `MLX.clipNorm` (check `MLX/Optimizers.swift` or use a manual `g * (max_norm / norm).clipped(0, 1)` pattern). Same for `TrainerHF`.

3. **LoRA+ (#151)** — In `LoraLinear.init`, add `loraBLrMultiplier: Float = 16.0`. To thread it through the optimizer, either (a) give B its own AdamW instance, (b) scale B's gradient by 16× in a custom step, or (c) split the model's parameter dict into two groups and pass two LRs. Option (b) is least invasive: in the trainer, walk grads, multiply any param whose path ends in `loraB` by 16 before applying. Test that loss is still decreasing.

4. **Persistent token cache (#152)** — In `Train.swift`'s BPE branch and `SFT.swift`'s example builder, before calling `tok.encode(text)`, check for a `.tokens` file with matching hash of (corpus_path, tokenizer_dir, text_hash). If present, load Int32 array directly. Else, encode + write cache. Saves the 10-30 min BPE step on every Mega run.

5. **Browser benchmark runner** — In `browser/src/main.ts` worker code, add a `runBenchmark(id)` action handler that loads `browser/src/benchmarks/<id>.ts`, invokes its `run(model)` against the currently-loaded model, returns the score. Wire a "Run benchmark" button in the gallery dialog or on the loaded-model UI.

After Phase 1, Phase 2 alignment variants (SimPO, ORPO, KTO) are all loss-function changes to the existing `DPO.swift` — they share the same trainer skeleton.

Phase 4's distillation trainer is the headliner — most worth shipping for both the educational story and the leaderboard play.

## Files that are NOT to be touched without asking

- `data/gallery/*.tinygpt` — trained gallery checkpoints
- `browser/public/gallery/*.bin` — fp16-packed published gallery
- `browser/public/gallery/manifest.json` — leaderboard manifest (re-generate via `finalize_gallery.mjs` + `score_gallery.mjs`, don't hand-edit)
- `/tmp/fineweb-edu-500M.txt` — the 2 GB pretraining corpus (took ~30 min to download)
- `/tmp/smollm2/` — downloaded HF model + tokenizer (used by all BPE paths)
- `/tmp/huge-fineweb.tinygpt` — will be the output of the currently-running training

## User constraints (from CLAUDE.md + memory)

- Ask before installing packages, large downloads, deployments
- Prefer small reviewable diffs
- Never use `--no-verify` on git hooks
- Don't edit secrets / env files / cloud configs
- The user is shipping solo, budget-constrained — Tinker etc. are ruled out
- "No quality regression" rule — perf paths need a parity gate
- "Opportunistic edge" — best perf for latest-Chrome users, graceful degradation

## What to verify before claiming any phase done

The doc's bottom checklist applies to code shipping too:
1. Code compiles (`xcodebuild -scheme tinygpt` + `xcodebuild -scheme TinyGPTApp`)
2. 14/14 unit tests pass (`xcodebuild test -scheme TinyGPT-Package`)
3. For new training paths: parity-test against the fp32 / pre-change version (loss curve within ~2% drift)
4. For new sample paths: greedy output identical to the pre-change version

Good luck. The roadmap doc is the plan; this file just gets you oriented.
