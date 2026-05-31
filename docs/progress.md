---
title: Progress — Mac + Web tracks
description: TinyGPT progress dashboard. Feature shipping timeline, perf metrics, and distance-to-targets for both the Mac (MLX-Swift) and Web (WebGPU) tracks. Refreshed as work ships.
---

# Progress — Mac + Web tracks

Last updated: **2026-05-31**  ·  Hardware: **Apple M5 Pro / 48 GB**
·  Browser baseline: M-series, Chrome WebGPU

This is the live progress dashboard. Numbers are reproducible — every
metric has a doc + raw JSON in `docs/research/data/`. Updated whenever
work ships.

## Headline metrics

| | **Mac** (MLX-Swift) | **Web** (WebGPU) |
|---|---|---|
| **Training (Huge preset)** | 42 ms/step | 720 ms/step (subgroup matmul gate failed; fallback active) |
| **Speedup vs browser** | 17.2× faster | (baseline) |
| **Inference TTFT** | 4.8–5.8 ms p99 | n/a (no inference path) |
| **Inference ITL p99** | 2.75–4.94 ms | n/a |
| **Decode tok/s** | 293–696 | n/a |
| **Largest model** | 960 M params (1.1 GB) | ~200 M cap (V8 heap) |
| **Realtime ready?** | ✅ 10× under TTFT target | n/a |

## Distance to realtime targets (Mac)

Targets from `docs/roadmap/north_star_refined.md` §realtime:

```
TTFT (warm):       < 50 ms       ░░░░░░░░░░░░░░░░░░░░ current 5.8 ms p99      ✅ 10× under
ITL p99:           < 30 ms       ░░░░░░░░░░░░░░░░░░░░ current 4.9 ms p99      ✅  6× under
Decode tok/s:      > 50 tok/s    ████████████████████ current 293 tok/s       ✅  6× over
Cold start TTFT*:  < 50 ms       ░░░░░░░░░░░░░░░░░░░░ current 24 ms (1B)      ✅  2× under
Cold launch wall:  no hard target ░░░░░░░░░░░░░░░░░░░ current 1065 ms (1B)    informational
Energy J/token:    < 0.5 J       ????????????????????? unmeasured (sudo)      pending
```

*"Cold start TTFT" = first prefill TTFT after model load (in-process). "Cold
launch wall" = process start + model load + first prefill + first token,
measured end-to-end for the ~1B mega-pilot checkpoint.

(Bars: `█` filled = at/above target  ·  `░` headroom = below target  ·  `?` = unmeasured)

## Mac inference baseline (M5 Pro)

Full numbers in `docs/research/mac_decode_baseline_m5pro.md`. Raw JSON
in `docs/research/data/`.

| Model | Params | TTFT p99 | ITL p99 | Decode tok/s | Prefill tok/s |
|---|---|---|---|---|---|
| mac-trained | 9.6 M | 5.83 ms | 3.75 ms | 564 | 26,381 |
| flagship-huge-v5 | 221 M | 4.83 ms | 4.59 ms | 385–696 | 19,913–56,815 |
| mega-pilot | ~960 M | 5.75 ms | 4.94 ms | 293 | 14,359 |

Decode tok/s scales with model size as expected (memory-bandwidth-bound).

## Mac training baseline

Run: `tinygpt bench-train --preset huge --steps 100 --warmup 20 --batch 8`

```
Mac (M5 Pro, fp32, batch=8):    42.0 ms/step  ████████████████████ 17.2× browser
Browser (M-series, fp32):       720 ms/step   █ (baseline)
```

## Capabilities shipped — Mac track

```
Wave 1  ─── agent runtime + JSON mode + HF data
        ✅ Agent runtime (multi-turn + tool dispatch + persistent KV)
        ✅ JSON-mode constrained generation (FSM token masking)
        ✅ HF Datasets / Hub integration
        ✅ GitHub data fetcher (issue → PR pairs)

Wave 2  ─── eval harness + GitHub data
        ✅ lm-evaluation-harness MLX adapter
        ✅ Mac XCTest harness + swiftformat + lint CI
        ⚠️  Tool-call eval harness (BFCL/τ-bench) — code exists, subprocess refactor needed

Wave 2.5 ── inference + training infra
        ✅ Cold-start bundle (mmap + lazy embed + async load + compile cache)
        ✅ KV cache optimization (GQA + in-place + persistent across sessions)
        ✅ Pausable training (SIGINT + atomic save + --resume)
        ✅ CF R2 cloud save/load pipeline (zero-egress, push/pull/list/setup)
        ✅ Cross-process GPU lock (~/.cache/tinygpt/gpu.lock)
        ✅ Speculative decoding heads (Medusa + EAGLE-2)
        ✅ Quantization (GPTQ reader + AWQ + SmoothQuant + QAT)
        ✅ Optimizer alternatives (Lion, Sophia, Muon, Adafactor)
        ✅ PEFT bundle (VeRA, LoftQ, AdaLoRA, RsLoRA, PISSA, LoRA-FA)
        ✅ MLXFast attention audit + tied embeddings verified
        ❌ Flash Attention Metal kernel — DROP (MLX already fused)
        ❌ Int4 packed matmul Metal kernel — DROP (MLX already hand-tuned)
        ⏸ Int8/cider W8A8 — DEFER (no win at current scales, revisit at 3B+)
        ⏸ ANE+GPU heterogeneous routing — DEFER (Apple Stateful Models API)

Wave 2.6 ── realtime + cloud escalation + tools (current)
        ✅ Cloud API client (Anthropic + OpenAI via curl)
        ✅ SSE streaming on /v1/chat/completions and /v1/completions
        ✅ SSE cancellation on client disconnect (SIGPIPE-safe)
        ✅ CloudEscalate wired into AgentLoop (--cloud-escalate flag)
        ✅ Mac decode jitter baseline measured (M5 Pro)
        ✅ Continue.dev / Ollama-compat provider adapter — /api/tags,
            /api/version, /api/show, /api/chat, /api/generate on the same
            serve socket as the OpenAI surface. NDJSON streaming. Smoke-
            tested end-to-end with the gallery model. See
            docs/continue_provider.md for Continue/Cline/Aider configs.
        ⚠️ Tool-call extractor (mini-router) — scaffold landed: ToolRouterModel +
            extractor-data / train-extractor / extract CLIs + --router agent flag.
            No router trained yet; FSM constraint-injection still a TODO.
            See docs/tool_call_extractor.md.
        ⚠️ ScreenCaptureKit + macOS Accessibility (AX) integration
            — TinyGPTScreen target shipped: AccessibilityTree.readFocused()
              works end-to-end from CLI; `tinygpt screen tree` returns AX
              JSON for the focused window once Accessibility is granted.
            — ScreenCapture.captureActiveWindow() compiles + links but
              the bare CLI process fails CGS init ("Assertion failed:
              did_initialize"), so window capture only works from a GUI
              terminal context (Terminal.app / iTerm / Ghostty that has
              Screen Recording permission). Documented in Screen.swift.
            — Vision-encoder / ViT half deliberately deferred (research-grade)
        ⬜ Vision encoder (ViT) → tinygpt decoder
        ⏸ Async tool-call dispatch — INVESTIGATED, SKIPPED. LM dominates
            by 5-100× over subprocess; flavor #1 (streaming overlap) saves
            ~10ms, flavor #2 (parallel tools) needs a specialist that emits
            them. See docs/async_tool_dispatch.md — revisit when the
            tool-call extractor + parallel-tool specialist ship.
        ✅ Cold-start TTFT measured on 1B+ models — in-process 24ms (< 50ms target);
            cold launch wall 1065ms end-to-end (process + load + first gen)

Wave 3  ─── specialists (deferred until 2.6 done)
        ⬜ Debugger specialist (GitHub issue→PR pairs)
        ⬜ Shell or SQL specialist

Wave 4  ─── polish + research
        ⬜ Multilingual specialist (Sarvam-Edge / Airavata base, NOT desi-max)
        ⚠️  MILU + IndicGenBench evals wired into harness
            (`tinygpt eval-indic` ships MILU MCQ + IndicXQuAD scoring;
             datasets must be pre-fetched via `download-dataset`;
             smoke run validates the pipeline end-to-end; full
             baseline pending real-data run — see
             docs/research/indic_evals.md)
        ⬜ Mac app demo
        ⬜ Public model card on HF Hub + writeup
```

Status legend: ✅ shipped · ❌ dropped (with reason) · ⏸ deferred ·
⬜ pending · ⚠️ partial

**Feature audit (2026-05-31)**: every CLI subcommand smoke-tested
end-to-end on M5 Pro. Datasets / quantization / fine-tuning / all
other features confirmed working. See
[feature_audit_2026_05_31.md](feature_audit_2026_05_31.md).

**Modality coverage**: text + code + structured + screen-text. See
[capability_matrix.md](capability_matrix.md) for the full matrix.

**Living backlog** (ROI-ordered): [backlog.md](backlog.md). Top items
get worked first; sort updates when we learn.

**First specialist run (2026-05-31)**: tool-caller v1 on
SmolLM2-135M-Instruct + hermes-function-calling-v1 (11k records),
2000 steps RS-LoRA rank 16, 3.5 MB adapter. Pipeline end-to-end
works; model picks up surface JSON format but doesn't yet
functionally tool-call. Major win along the way: found + fixed the
A0 LoRA-save-empty-entries bug (`f566023`). See
[specialist_v1_findings.md](specialist_v1_findings.md) for the
full writeup + ranked next-iteration options.

**Dataset inventory**: see [data_inventory.md](data_inventory.md) —
22 curated HF datasets, GitHub + BFCL + Magpie pipelines, plus
the gotchas list (gated datasets need `HF_TOKEN`, parquet shards
not yet decoded, `/tmp` reaping).

## Capabilities shipped — Web track

```
Browser playground
        ✅ Landing page + /playground route
        ✅ WebGPU training pipeline (Huge / Mega presets via cap)
        ✅ Browser BPE scorer + gallery model loader
        ✅ Doc consolidation — every doc web-visible at /docs/[slug]

WebGPU kernels (in webgpu/train.wgsl + train_sg.wgsl)
        ✅ Naive scalar matmul
        ✅ Blocked 4×4 matmul (matmul_blocked_vec4)
        ✅ Layer-norm subgroup variant (gated on gpuFeatures.subgroups)
        ✅ Cross-entropy subgroup variant
        ✅ Bias-grad subgroup variant
        ✅ FA2 forward in WGSL (browser-side flash attention)
        ❌ Subgroup matmul (matmul_sg / matmul_abt_sg) — kernel ships in
            train_sg.wgsl but the numerics gate FAILS on M5 Pro + Chromium
            (mean_rel 1415%, max_rel 132316% at K=256). Decision 19 fallback
            engaged — training stays on matmul_blocked_vec4, no regression.
            See "Browser training: lever stack" row below for verdict.

Browser learning artifacts
        ✅ docs/decision_log.md — every decision logged
        ✅ docs/research/inference_benchmarks_may_2026.md
        ✅ docs/research/quality_benchmarks_may_2026.md
        ✅ docs/research/wave_2_5_kernel_audit.md (this audit's verdicts)
        ✅ docs/research/wave_4_landscape.md (TML / Apple FM / agents / Indic)
        ✅ docs/research/mac_decode_baseline_m5pro.md (this baseline)
        ✅ docs/research/indic_evals.md (MILU + IndicGenBench wiring)
```

## Browser training: lever stack (1.0× → 17.2× equivalent)

These are the levers that brought browser → Mac parity-ish. Detailed
journey in `/roadmap`. Numbers are reproducible via the bench scripts.

```
Lever                                     Browser ms/step   Speedup
─────────────────────────────────────────────────────────────────────
Scalar baseline (no SIMD, single thread)  ~24,000 ms        1.0×
+ SIMD vectorization                      ~12,000 ms        2.0×
+ Multi-thread CPU                         ~4,500 ms        5.3×
+ WebGPU naive matmul                      ~1,200 ms       20.0×
+ Blocked 4×4 matmul (current)               720 ms        33.3×
+ Subgroup matmul (gate FAILED 2026-05-31)    720 ms        33.3× (fallback)
+ Mac MLX-Swift (target parity)              42 ms       571.4×
```

The Mac is currently **~17× the browser** at Huge preset training.

**Subgroup matmul verdict (2026-05-31)**: code-complete in d4a9de6,
gate now run on M5 Pro + Chromium WebGPU. **The kernel fails parity
catastrophically** — mean_rel 1415% (limit 0.100%), max_abs 3.47
where mean|ref|=0.17 — at the representative Huge shape M=128 K=256
N=128. Decision 19's auto-disable engaged: `matmulSgActive=false`,
`matmul()` falls back to `matmul_blocked_vec4` unchanged. No
regression risk to user-facing training. Likely causes (per d4a9de6
caveats): per-output `subgroupAdd` inside the (i,j) loop hitting
non-uniform-control-flow Apple Metal pathology, or an
`mm_sg_partial` initialization assumption (`sgSize≥32`). The
follow-up variant proposed in commit message caveat 3 (replace
shared-mem gather with `subgroupShuffle` B-broadcast) would
side-step the failing path entirely; deferred until browser focus
returns.

## Decisions logged (research-driven)

| Decision | Doc | Outcome |
|---|---|---|
| FA Metal kernel | wave_2_5_kernel_audit.md §1 | DROP — MLX already fused |
| Int4 matmul Metal kernel | wave_2_5_kernel_audit.md §3 | DROP — MLX already hand-tuned |
| Int8 W8A8 (cider) | mac_decode_baseline_m5pro.md | DEFER — no win at current model sizes |
| ANE+GPU routing | wave_2_5_kernel_audit.md §4 | DEFER — gated on Apple Stateful Models API |
| WebGPU subgroup matmul | wave_2_5_kernel_audit.md §5 | BUILT, GATE FAILED — kernel disabled, falls back to vec4 blocked (2026-05-31) |
| desi-max as Hindi base | wave_4_landscape.md §4 | WRONG — replaced with Sarvam-Edge / Airavata |
| TML "Interaction Models" framing | wave_4_landscape.md §1 | ADOPT vocabulary (foreground/background) |
| Apple FM positioning | wave_4_landscape.md §2 | tinygpt = open/hackable/multi-specialist counter-position |
| Continue.dev provider adapter | wave_4_landscape.md top-3 | NEXT product-shaped move |

## What's next

Pinned in `docs/roadmap/north_star_refined.md` §"What I'd ship next":

1. Continue.dev / Ollama-compat provider adapter (2-3 days, real users)
2. Tool-call extractor (mini-router) — 1 week
3. Fix Indic plan + wire MILU + IndicGenBench (1-2 days)
4. ScreenCaptureKit + AX (3-5 days each)
5. Train first specialist (debugger) — Wave 3

Backgrounded:
- WebGPU subgroup matmul — kernel fix (current attempt's parity gate
  fails; needs subgroupShuffle-based redesign or a uniformity audit)
- cider W8A8 (when a 3B+ specialist needs it)

## How this page updates

This doc is the canonical progress dashboard. When work ships:

1. Update the headline metrics table if the number moved
2. Toggle the capabilities checklist item (⬜ → ✅, or ⬜ → ❌ with reason)
3. Add a row to the "Decisions logged" table if a research dive closed it
4. Re-run the relevant bench command + paste the new numbers

Numbers older than the date at the top should be re-verified before
citing. The JSON files under `docs/research/data/` are the raw
provenance — keep them in git for before/after diffs.
