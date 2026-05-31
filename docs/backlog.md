---
title: ROI-ordered backlog
description: Living, ordered list of every pending tinygpt item — captured from the roadmap, research docs, conversation decisions, and the feature audit. Re-sorted as we learn. Top items get worked first.
---

# ROI-ordered backlog

**Last sort**: 2026-05-31 · **Commit**: `63f9b4b`

This is the single source of truth for "what's left." Every item came
from a doc we've written, a conversation decision, or a TODO in the
codebase. Sorted by ROI (impact ÷ cost), not by wave number.

## How this list updates

- Items finish → struck through with the resolving commit, then
  removed at the next sort
- New items discovered → added with a one-line rationale
- Learning changes ROI → re-sorted; the sort date at the top moves
- A trigger fires for a DEFERRED item → it moves into Tier A/B
- Source for every item is linked in the rationale

## Tier A — DO NEXT (high ROI, low/moderate cost)

These are the items that, when done, unlock the most downstream value.

### ~~A0. Fix the LoRA-save bug~~ — **CLOSED in `f566023`**
Diagnosis: SFT's curated DoRA-on-by-default wasn't being disabled
by PEFT-variant flags (`--rs-lora` etc), so the writer's
`as? LoraLinear` cast missed every `DoraLinear` instance →
`entries:[]` empty adapter. Fix is one line per PEFT case in
SFT.swift: also set `useDora = false`. Verified end-to-end:
3.5 MB adapter persists, reloads, runs. Latent regression-coverage
gap noted — add a save+reload XCTest (Tier C task) so this can't
slip again.

### A1. Train the first specialist end-to-end (tool-caller)
**Impact**: validates the north-star thesis. Until this happens, every
optimization is theoretical.
**Cost**: 3–5 days execution + GPU hours for training
**Depends on**: A2 + A3 (data)
**Source**: `docs/roadmap/north_star_refined.md` Wave 3 + the
"finish features" pivot we just had

### A2. Pull foundational datasets (Tier 1 + 2, ~2 GB total)
**Impact**: required precursor for A1; also unlocks Indic baselines + router training
**Cost**: ~1 hour wall time, mostly background
**Items**: xlam-function-calling-60k, hermes-function-calling-v1,
function-calling-chatml, SWE-bench_Verified, alpaca-cleaned,
orca_dpo_pairs, python_code_instructions_18k_alpaca, MetaMathQA,
ultrafeedback-binarized-preferences-cleaned, the-stack-smol
**Source**: conversation today (queued in tasks #124, #125)

### A3. Fetch GitHub issue→PR corpus for debugger training
**Impact**: training data for A1
**Cost**: ~1 day wall (rate-limited without `GITHUB_TOKEN`; faster with)
**Source**: tasks #126; `tinygpt fetch-github` CLI verified

### A4. Pull BFCL + τ-bench via extractor-data
**Impact**: training data for the mini-router; also gives a tool-calling
specialist baseline corpus
**Cost**: ~30 minutes (small datasets)
**Source**: tasks #127; CLI scaffolded in b5bbdd9

### A5. Pull Indic eval datasets (MILU + IndicGenBench-XQuAD)
**Impact**: real-data baseline for current flagship; required before any
Hindi specialist claim
**Cost**: ~30 minutes
**Source**: `docs/research/indic_evals.md`, tasks #128

### A6. Dataset inventory doc
**Impact**: discoverable record of what's pulled, what's converted, how
to re-pull
**Cost**: ~30 minutes after A2–A5 done
**Source**: tasks #129

### A7. Real-data MILU baseline on flagship-huge-v5
**Impact**: number we can cite — "current model scores X% on MILU
Hindi"; tells us if Indic specialist is worth pursuing now or later
**Cost**: ~2 hours (run + write up)
**Depends on**: A5
**Source**: `docs/research/indic_evals.md`

---

## Tier B — NEXT QUARTER (medium ROI, moderate cost)

Real value but waits on Tier A learnings.

### B1. Train a second specialist (shell or SQL)
**Impact**: validates the multi-specialist architecture (not just N=1)
**Cost**: 3–5 days
**Depends on**: A1
**Source**: `north_star_refined.md` §"Demonstration specialists"

### B2. Train + ship the mini-router on real BFCL data
**Impact**: 50–100× latency win on tool-call cases; foundational for
mixture-of-specialists
**Cost**: ~half day if A4 is done
**Depends on**: A4
**Source**: `docs/tool_call_extractor.md` (router scaffold, b5bbdd9)

### B2b. Bake-off: classifier-head router vs pure-GPT-with-FSM
**Impact**: settles whether the architectural deviation in B2 is worth
it. Pure-GPT variant uses the regular LM head + JSON-mode FSM constraint
to emit a single tool-name token — zero architectural deviation, same
model format, same training pipeline. If within 2× latency of the
classifier-head version, ship pure-GPT for purity + simpler maintenance;
if dramatically slower, the classifier deviation is justified.
**Cost**: ~half day (train a small pure-GPT router on same BFCL data,
constrain decode to tool-token set, benchmark vs B2)
**Depends on**: A4
**Source**: conversation 2026-05-31 — "is router a type of GPT?"
The classifier-head router IS a GPT trunk with a different head; both
variants are GPT-family. Measure to decide.

### B3. FSM constraint-injection from router prediction
**Impact**: when router fires, the LM CANNOT emit non-matching tool —
locks down hallucinated tool names
**Cost**: ~3 days (router exists; FSM exists; needs the pin call)
**Depends on**: B2
**Source**: `docs/tool_call_extractor.md` §"Integration with ConstrainedGen"

### B4. Tool-call eval harness — subprocess refactor for BFCL/τ-bench scoring
**Impact**: numeric eval against the canonical tool-calling benchmarks
**Cost**: ~half day
**Source**: task #91 (A5 stalled on stdout-capture deadlock)

### B5. Cloud-escalation **training signal** (`{"defer_to_cloud": true}`)
**Impact**: implicit escalation vs the explicit `escalate` tool we have
today. Specialist learns when it doesn't know.
**Cost**: ~1 week — needs paired training corpus (Claude/GPT-as-judge
bootstrap)
**Source**: `north_star_refined.md` capability table

### B6. Mac app demo
**Impact**: first user-facing product surface — show "your local
specialist responds in <50ms, escalates to Claude when needed"
**Cost**: ~1 week (SwiftUI shell + the existing agent runtime as the engine)
**Depends on**: A1 (need a model worth demoing)
**Source**: `north_star_refined.md` "What I'd ship next"

### B7. Specialist-routing model (different from tool-call extractor)
**Impact**: choose between debugger/shell/SQL specialists at the
request level; enables mixture-of-specialists
**Cost**: 1–2 weeks
**Depends on**: B1 (need ≥ 2 specialists to route between)
**Source**: `wave_4_landscape.md` §1 (TML interaction-model framing)

### B8. Multilingual specialist (Sarvam-Edge / Airavata base)
**Impact**: Indian market reach
**Cost**: 1–2 weeks (data + Indic-tokenizer + SFT)
**Depends on**: A7 (baseline first), Sarvam-Edge public release
**Source**: `wave_4_landscape.md` §4

### B9. Energy J/token measurement (needs sudo for powermetrics)
**Impact**: efficiency story for the writeup + "build for India" framing
**Cost**: ~1 day with sudo access
**Source**: `progress.md` distance-to-targets table

---

## Tier C — POLISH (low cost, low-but-positive impact)

Pick up when the next big thing is blocked.

### C1. CLI cosmetic fixes
**Items**:
- `tinygpt validate --help` should print usage instead of erroring on
  `--help` as a positional
- `tinygpt hf-inspect --help` same arg-parse quirk
- `score-bench` / `prune-structured` print partial usage on bare invocation
**Cost**: ~1 hour total
**Source**: `feature_audit_2026_05_31.md` §"Known cosmetic issues"

### C2. Roll up pre-switch CLI shims into the main switch
**Impact**: code hygiene
**Cost**: ~half day
**Items**: `score-bench`, `pruning`, `agent`, `cloud-merge`,
`hf-datasets-merge`, `github-data-merge` — all have TODOs in
TinyGPT.swift to move dispatch into the case block
**Source**: TinyGPT.swift TODO comments

### C3. DoRA on-disk adapter format
**Impact**: persistable DoRA-trained models (today DoRA is in-session only)
**Cost**: ~1 day
**Source**: `tinygpt sft --help` for `--dora`

### C4. Tool-call extractor: BPE tokenizer support
**Impact**: better accuracy on noisy queries; currently byte-level
**Cost**: ~2 days
**Source**: `docs/tool_call_extractor.md` §"Design questions"

### C5. Decode jitter under thermal load
**Impact**: realtime under sustained workload — does p99 ITL spike when
the laptop heats up?
**Cost**: ~1 day
**Source**: `progress.md` distance-to-targets

### C6. ChatML template: detect inline `system: ...` prefix and split
**Impact**: hermes-function-calling-v1 (and likely other tool-calling
datasets converted from `[{role, content}, ...]` arrays) carry the
system role as a `"system: ..."` prefix inside the `instruction`
string. The current `.chatml` template wraps it all in
`<|im_start|>user\n{instruction}<|im_end|>` — burying the system
role in the user turn. Training still works but inference is
brittle: test prompts must match the buried-system shape rather
than proper ChatML role separation, which is a real footgun.
**Cost**: ~half day. Add a `splitInlineRoles()` pre-step to the
chatml `render()` that recognises `^system:\s` / `^user:\s` / etc.
and emits separate `<|im_start|>{role}\n...<|im_end|>` blocks.
**Source**: discovered while diagnosing `toolcall-v1` — see
`docs/specialist_v1_findings.md` and `docs/data_inventory.md`
"Known gotchas" §3.

### C7. Save+reload XCTest for LoRA adapters
**Impact**: regression coverage for the A0 bug that just landed.
Currently no test exercises the SFT → save → load → sample roundtrip,
so silent-empty-adapter regressions could slip again.
**Cost**: ~2 hours. Write a tiny TinyGPTModelTests.swift case that
runs 5 SFT steps on a fake corpus, saves the LoRA, applies it back
to a fresh base, asserts logits differ vs untouched base.
**Source**: A0 closure note.

### C8. Install-path discipline (no more /tmp for build/data caches)
**Impact**: macOS reaps `/tmp` aggressively. Lost build cache, eval
harness, and 50 MB of pulled data mid-session on 2026-05-31. Should
default to `~/.cache/tinygpt/` for downloads, `~/.local/bin/tinygpt`
(or similar) for the built binary.
**Cost**: ~1 hour — defaults change + a one-liner in the README.
**Source**: discovered 2026-05-31.

---

## Tier D — DEFERRED (waiting on external trigger)

Real items but explicitly waiting. Trigger documented.

### D1. cider W8A8 adoption for Mac inference
**Trigger**: a 3B+ specialist ships (so prefill matters)
**Why deferred**: at current model sizes (≤ 1B), Mac is already 10×
under realtime targets; cider's prefill win is immaterial and decode
slightly regresses vs W8A16
**Source**: `mac_decode_baseline_m5pro.md`

### D2. ANE + GPU heterogeneous routing
**Trigger**: Apple ships the Stateful Models API (rumored late 2026)
OR the screen-watching specialist needs always-on background work
**Why deferred**: research-grade; current path uses private APIs (ANEMLL
"beta, one macOS update breaks things")
**Source**: `wave_2_5_kernel_audit.md` §4

### D3. WebGPU subgroup matmul redesign (subgroupShuffle B-broadcast)
**Trigger**: browser focus returns
**Why deferred**: current gate fails (1415% mean_rel); fallback works
unchanged; Mac is the active focus
**Source**: commit `603c0bd`, `wave_2_5_kernel_audit.md` §5

### D4. Vision encoder (ViT → tinygpt decoder)
**Trigger**: a vision-specialist demand becomes concrete (screen-reading
training data ready, or a product use case emerges)
**Why deferred**: 2 weeks research-grade work; not on the critical path
for debugger/shell/SQL specialists
**Source**: `capability_matrix.md`, `north_star_refined.md`

### D5. Audio I/O (Apple Speech.framework + AVSpeechSynthesizer)
**Trigger**: voice-mode demo becomes a priority
**Why deferred**: not in scope for any Wave 3 specialist
**Source**: `capability_matrix.md`

### D6. Async tool-call dispatch
**Trigger**: a specialist emits parallel `tool_calls: [...]` arrays,
OR tool execution dominates a meaningful fraction of agent turns
**Why deferred**: LM dominates 5–100× over subprocess at current scales
**Source**: `docs/async_tool_dispatch.md`

### D7. ScreenCaptureKit raw image — fix CGS-init from CLI
**Trigger**: vision specialist needs raw image bytes (today AX text tree
is the LM-friendly half + it works)
**Why deferred**: AX tree is sufficient for tool-calling specialists;
raw image is for future vision integration
**Source**: commit `08a7689`, `Screen.swift` known-limitation doc

### D8. Public launch (HF model card + writeup + HN post)
**Trigger**: at least one specialist beats a fair baseline on a
public benchmark
**Why deferred**: nothing to launch yet
**Source**: `north_star_refined.md` Wave 4

---

## Items intentionally NOT on this backlog

- ❌ **Flash Attention Metal kernel** — DROP (MLXFast SDPA already fused)
- ❌ **Int4 packed matmul Metal kernel** — DROP (MLX `quantized_matmul` is hand-tuned)
- ❌ **Hooking into Apple App Intents** — no public API for third-party LLMs to replace Apple's FM
- ❌ **Competing with Tinker for cloud fine-tune** — use it if needed; not a project differentiator
- ❌ **General SWE-bench leaderboard chase** — Sonnet 4.6 dominates by 5+ pt no matter the wrapper; play the local-first / on-device game instead

These are removed-with-reason, not pending.

---

## Snapshot ROI at this sort

If executed top-to-bottom from Tier A:

| Span | Items | Cumulative wall time |
|---|---|---|
| Today | A2, A4, A5 (data pulls, all small) | ~2 hours |
| This week | A3, A6, A7 + start A1 | ~3–5 days |
| Next 2 weeks | A1 (finish) + start B1 | ~7–10 days |
| 1 month | B1, B2, B3, B4 | ~3 weeks of work |
| 2 months | B5, B6 + first launch in B8 timeline | ~6 weeks |
| 3 months | Public launch (D8 trigger fires) | ~10 weeks |

## What gets reshuffled when we learn

- **A7 result high** (Hindi baseline already decent on Qwen3 vocab) →
  B8 moves up (multilingual specialist becomes Tier A)
- **A1 result low** (debugger specialist can't beat cloud) → B7
  (specialist router) drops, focus shifts to data quality / scale
- **A1 result high but slow** → D1 (cider W8A8) moves up
- **Real product user feedback** wants voice → D5 moves into Tier B
- **Apple ships Stateful Models API** → D2 trigger fires; moves to Tier B
