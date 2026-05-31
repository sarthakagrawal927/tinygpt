---
title: Tool-call extractor (mini-router)
description: Design + training recipe for tinygpt's tiny encoder model that picks which tool a user query needs, before the full LM forward pass.
---

# Tool-call extractor (mini-router)

A tiny BERT-class encoder (~30-100M params) with a classification head
over the tool catalog. Pre-step that runs **before** the full LM
forward pass: given a user query + the active tool catalog, picks
which tool the LM should be steered toward. Way cheaper than letting
the full LM hallucinate the choice — runs in <5 ms on CPU, no KV
cache needed.

A focused special case of the broader Router model (model-level
specialist selection). See `docs/roadmap/north_star_refined.md` row
"Tool-call extractor (mini-router)" for the context.

## Status

**Wave 2.6 scaffold (May 2026).** The training pipeline + model +
eval harness are wired end-to-end and the binary builds clean, but
no router has actually been trained yet — that's a compute decision
the operator makes later.

## Design

### Architecture

`ToolRouterModel` (in `native-mac/Sources/TinyGPTModel/ToolRouterModel.swift`)
reuses tinygpt's existing `TransformerBlock` stack with three changes
vs `TinyGPTModel`:

1. **No LM head.** The `lm_head` Linear is dropped entirely; no
   next-token prediction.
2. **Add a classification head.** `router_head: Linear(d_model →
   n_classes)`, with the hidden state mean-pooled over the sequence
   axis. (BERT-style `[CLS]` pooling is also supported via
   `Pooling.firstToken`, but mean is more robust for variable-length
   user queries that don't have a special classification token.)
3. **Cross-entropy loss over tool classes**, not over the vocab.

The transformer blocks themselves still use **causal** attention
(`CausalSelfAttention`). For a short classification input (≤256
tokens) with mean-pooling, the causal mask is a small loss vs a full
BERT bidirectional encoder — but the upside is enormous: every line
of training infrastructure, gradient checkpointing, attention kernel,
weight loader, and `.tinygpt` file-format round-trip works unchanged.

### Presets

| Preset  | layers | d_model | dMlp | params @ vocab=256 | params @ vocab=32k |
|---------|-------:|--------:|-----:|-------------------:|-------------------:|
| `tiny`  |     4  |    256  | 1024 |              ~5 M  |              ~22 M |
| `small` |     6  |    384  | 1536 |             ~13 M  |              ~37 M |

Defined in `ToolRouterModel.tinyPreset` / `smallPreset`. Default for
`tinygpt train-extractor` is `tiny`.

### Why a separate model?

A 1-3B specialist needs to forward through the full prompt + tool
schema (often 1-4 KB of tokens) before it can emit even the first
JSON byte. The router runs 4 encoder layers over <100 tokens and
outputs softmax over ~20-100 tool classes. That's a **50-100×
latency cut** for the most common decision the agent makes — which
tool to call.

### What we stole from Apple's `Tool` protocol

Apple's [Foundation Models framework `Tool`
protocol](https://developer.apple.com/videos/play/wwdc2025/286/)
defines a tool as `(name, description, parameters as a JSON-schema
graph)`. tinygpt's `ToolSchema` already has this shape (see
`Sources/TinyGPT/ToolSchema.swift`); the router classifies over the
`name` field. Multi-tool call graphs and argument generation are
left to the downstream LM — the router's job is just "which tool".

### What we stole from Cline

[Cline's ReAct loop](https://deepwiki.com/cline/cline) **forces a
tool call every turn** — plain assistant text is rejected and the LM
has to emit a tool call (or a sentinel `plan_mode_respond` tool).
When tinygpt's router fires with high confidence (default ≥ 0.7),
the downstream LM is supposed to be locked into the predicted tool's
JSON schema via the existing FSM in `ConstrainedGen.swift`. That
constraint-injection is the integration TODO below.

## Training data

The router's training signal is a flat JSONL:

```jsonl
{"query": "open foo.py and read it", "tool": "read_file"}
{"query": "find files containing TODO", "tool": "grep"}
```

`tinygpt extractor-data` builds this corpus from three sources:

1. **[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)** —
   Berkeley Function-Calling Leaderboard. Distributed on HF Hub as
   `gorilla-llm/Berkeley-Function-Calling-Leaderboard`. Each row has
   a user `question` + an oracle `function` call; we extract
   `(question, function.name)`.

2. **[τ-bench](https://github.com/sierra-research/tau-bench)** —
   Sierra Research's tool-use benchmark with retail + airline
   domains. Tasks expose `user_instruction` + `tools_used[]`; we
   take the first tool as the supervised label.

3. **Synthetic (`--synth`)** — for low-resource tools (≤ 5 real
   examples after BFCL + τ-bench), query CloudEscalate
   (Claude/GPT) for 30-50 plausible user queries that would invoke
   that tool. Bootstraps the long tail.

The `--tools <schema.json>` argument controls the label set. When
provided, only pairs whose tool name appears in the schema are kept;
unknown tools are dropped silently.

## Pipeline

```bash
# 1. (optional) Pull BFCL via tinygpt's HF downloader.
tinygpt download-dataset hf://datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard

# 2. (optional) Clone τ-bench (small repo, manual step).
git clone https://github.com/sierra-research/tau-bench ~/code/tau-bench

# 3. Build the training corpus.
tinygpt extractor-data \
  --bfcl ~/.cache/tinygpt/datasets/gorilla-llm/.../corpus.jsonl \
  --tau-bench ~/code/tau-bench/tau_bench/envs \
  --tools my_tools.json \
  --out router_data.jsonl

# 4. (optional) Backfill rare tools with synthetic queries.
tinygpt extractor-data \
  --tools my_tools.json --synth --synth-per-tool 40 \
  --bfcl ... --tau-bench ... \
  --out router_data.jsonl

# 5. Train.
tinygpt train-extractor router_data.jsonl \
  --preset tiny --steps 500 --batch 32 \
  --out router.tinygpt

# 6. Use.
tinygpt extract router.tinygpt --query "find files containing TODO"
# query: find files containing TODO
#   latency: 2.18 ms
#   0.8412  grep
#   0.0931  find
#   0.0421  ls

# 7. (optional) Wire into the agent.
tinygpt agent specialist.tinygpt --tools my_tools.json \
  --router router.tinygpt --router-threshold 0.7
```

## Expected accuracy ceiling

- **BFCL alone**: ~85-92% top-1 over the BFCL tool set is realistic.
  The dataset's tools are diverse + the queries are clean.
- **BFCL + τ-bench**: the τ-bench retail/airline tools have ~10
  classes each with very specific phrasings; once domain-matched,
  >95% is plausible.
- **+ synthetic for rare tools**: the long tail (custom user tools)
  is the hard part. ~80% top-1 on synthetic-only tools is the
  realistic target — the cloud model isn't a perfect oracle for what
  a real user would phrase.
- **Overall router accuracy**: 85-90% top-1 over a 20-tool catalog
  is the sweet spot. The agent loop's threshold (default 0.7) means
  uncertain queries fall through to the LM's own tool choice — so a
  90%-accurate router with 70% of its calls above threshold still
  speeds up the common case by 50-100×.

## Integration with `ConstrainedGen` (TODO)

The FSM in `ConstrainedGen.swift` accepts a JSON schema and rejects
any token that wouldn't produce a valid prefix. To wire the router:

1. After `runTurn` calls `predictWithRouter`, build a single-tool
   JSON schema by pinning `tool.name` to the predicted string and
   leaving `arguments` open per the tool's parameter spec.
2. Pass that schema to a fresh `JSONSchemaFSM` and use it to mask
   logits at each decode step inside `generateUntilJSONOrLimit`.

Today, `AgentLoop.predictWithRouter` only **records** the prediction
in the transcript — the FSM hand-off isn't wired. Doing it cleanly
requires either:
- a new `JSONSchemaFSM.pin(toolName:)` builder, or
- the caller assembling a per-prediction `JSONSchemaNode` and
  threading it into the existing decode loop.

Both are mechanical; left for the follow-up PR that comes after the
first router actually trains.

## Files

| File | Role |
|---|---|
| `native-mac/Sources/TinyGPTModel/ToolRouterModel.swift` | Model class + `ToolRouterLoader` |
| `native-mac/Sources/TinyGPT/ExtractorData.swift` | `tinygpt extractor-data` |
| `native-mac/Sources/TinyGPT/TrainExtractor.swift` | `tinygpt train-extractor` |
| `native-mac/Sources/TinyGPT/Extract.swift` | `tinygpt extract` |
| `native-mac/Sources/TinyGPT/Agent.swift` | `--router` + `--router-threshold` flags |
| `native-mac/Sources/TinyGPT/AgentLoop.swift` | `RouterHook` + `predictWithRouter` |

## References

- [BFCL leaderboard + dataset](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [τ-bench](https://github.com/sierra-research/tau-bench)
- [Apple Foundation Models `Tool` protocol (WWDC25 #286)](https://developer.apple.com/videos/play/wwdc2025/286/)
- [Cline plan/act + structured-output enforcement](https://deepwiki.com/cline/cline/3.4-plan-and-act-modes)
- `docs/roadmap/north_star_refined.md` — north star context
- `docs/research/wave_4_landscape.md` §2-3 — code-agent landscape
