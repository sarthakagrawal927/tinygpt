---
title: First specialist run — toolcall-v1 findings
description: End-to-end first specialist training on M5 Pro. What worked, what didn't, what was unblocked, and what to try next.
---

# First specialist run — `toolcall-v1` findings

**Date**: 2026-05-31  ·  Hardware: M5 Pro / 48 GB
**Goal**: Train a tool-calling specialist as the first real Wave 3
artifact + validate the full pipeline.

## TL;DR

**Pipeline works end-to-end.** **Model doesn't yet tool-call.** Found
and fixed a major LoRA-save bug along the way. Have a baseline to
iterate from.

## What ran

```
base:           SmolLM2-135M-Instruct (HF, 134.5M params, ctx=8192)
data:           NousResearch/hermes-function-calling-v1
                11,230 records · 8.4M tokens · 6.5M scored (mask=1)
adapter:        RS-LoRA rank=16 alpha=32, LoRA+ B-LR×16
                921,600 trainable params (0.68%)
training:       2000 steps · batch 1 · max-seq 1024 · lr 1e-4
                pack-mode bucket (4 length buckets)
                483.9 s wall, 4.1 step/s
output:         /tmp/specialist/toolcall-v1.lora (3.5 MB)
```

## What worked

1. **HF model load** — `tinygpt hf-load /tmp/smollm2` works clean,
   sample at 64 tok/s with coherent output
2. **Dataset download + schema sniff** — hermes JSONL detected and
   converted correctly (`format: sft, confidence 75%`)
3. **Training loop** — 2000 steps clean, no NaN/diverge, loss bouncy
   but trending low (final batch 0.007)
4. **LoRA save → reload → sample** — adapter persists, loads back
   via `sample --lora`, runs at 134 tok/s (vs 154 base)

## What didn't work

On a **novel** tool-calling prompt (a `get_weather` tool not in
training), the trained model still hallucinates the weather answer
rather than emitting `<tool_call>`. Same behavior as the base.

On a **training-distribution** prompt (an actual hermes record
re-fed), the model emits JSON-schema-like text (continuing the
schema fields with `"name":`, `"description":`, `"enum":`) instead
of the expected `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
response.

**Diagnosis**: The model picked up surface format adjacency (JSON
chatter, role tokens) but did not learn the task semantics (when to
emit tool_call). Likely causes:
- **135M is too small** for genuine generalization on this task
- **2000 steps is too few** — that's ~18% of one epoch on 11k records
- **Response format** is `<tool_call>...</tool_call>` XML wrapping
  but base model is biased toward freeform answers

## Pre-flight: bugs found + fixed during this run

### A0: LoRA save bug (FIXED)

**Symptom**: SFT/DPO/finetune saved adapter as 217 B header-only
file, `entries:[]` empty. Training ran fine but weights were never
serialized.

**Root cause**: `SFT.swift` defaults `useDora=true` (curated recipe).
`--rs-lora`/`--vera`/etc. only set `peftVariant` — they didn't turn
DoRA off. So `makeAdapterLinear()` returned `DoraLinear` instances.
Writers cast `as? LoraLinear` → nil → skip. Trainable param walker
handled both classes so the count was always right, masking the
bug.

**Fix**: PEFT-variant flags now auto-disable DoRA (one-line patch
in `SFT.swift`, commit `f566023`).

**Latent regression coverage gap**: no save+reload XCTest existed.
Worth adding to prevent recurrence.

## What to try next, ranked

### Option A — train longer + more data (cheapest)
- 10k–20k steps (50 min–1.7 hr at 4 step/s)
- Add more tool-calling datasets when available (xlam needs HF
  license accept; chatml needs parquet decoder)
- Stay on SmolLM2-135M for fast iteration

### Option B — bigger base
- SmolLM2-360M (3× params, same tokenizer)
- Needs download (~700 MB)
- Slower per step but much more capacity

### Option C — better prompt match at inference
- Re-format the test prompt to exactly match training shape
  (`system: ...` raw, not ChatML-wrapped, since hermes records
  already include the role prefix)
- Cheap; might already work and we've been mismeasuring

### Option D — task-specific eval harness
- Use the existing `tinygpt extractor-data` to build (query, expected_tool)
  pairs from hermes itself, hold out 10%, eval `tinygpt sample` outputs
  vs expected tool name
- Tighter feedback loop than eyeball-checking samples

## Recommendation

**Do C + D first** (~1 hour combined), then A or B depending on
what the eval shows. Don't grind more compute without knowing
whether the LoRA is even being applied at inference correctly.

## Reproduction

```bash
BIN=/tmp/tinygpt-sft-fix/Build/Products/Release/tinygpt

$BIN sft /tmp/smollm2 \
  --data /tmp/data/hermes.jsonl \
  --out /tmp/specialist/toolcall-v1.lora \
  --template chatml \
  --steps 2000 --batch 1 --max-seq 1024 \
  --rank 16 --alpha 32 --lr 1e-4 --rs-lora \
  --pack-mode bucket --length-bucket 4 --lora-plus-ratio 16

$BIN sample /tmp/smollm2 --lora /tmp/specialist/toolcall-v1.lora \
  --prompt "$(cat /tmp/tool-prompt.txt)" --tokens 100 --temperature 0
```

## What's now in the backlog

- A0 LoRA save bug — **CLOSED** in `f566023`
- A1 train debugger specialist — **partially started**: pipeline
  proven on tool-caller; need next-iteration plan above
- A2 dataset downloads — partial (hermes ✓, xlam gated, chatml
  needs parquet)
