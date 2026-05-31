---
title: Indic-language evals — MILU + IndicGenBench wiring
description: How tinygpt scores Indic-language ability — MILU multi-choice across 11 langs, IndicGenBench XQuAD extractive QA. The Wave 4 gate before claiming Hindi support.
---

# Indic-language evals — MILU + IndicGenBench

**Status**: CLI shipped (`tinygpt eval-indic`), datasets must be
pre-fetched, baseline numbers below are against an English byte-level
Shakespeare model — i.e. the expected ~0% baseline.
**Date**: 2026-05-31
**Wave**: 4 (multilingual specialists)
**Context**: [Wave 4 landscape §4](wave_4_landscape.md#4-multilingual--india-focus)

This is the **eval gate** for Wave 4. Before training (or claiming) any
Indic specialist, run candidate base models through these two evals to
get a real baseline. See the §4 landscape doc for *why* the previous
`desi-max` reference was wrong and what Sarvam-Edge / Airavata bring.

## What each eval measures

### MILU — Multi-task Indian Language Understanding

- AI4Bharat, NAACL 2025. Paper: [arXiv 2411.02538](https://arxiv.org/pdf/2411.02538). Repo: [github.com/AI4Bharat/MILU](https://github.com/AI4Bharat/MILU).
- **MMLU-style multiple choice** across **11 Indic languages** × **8 domains** × **41 subjects**.
- Languages: Hindi (hi), Bengali (bn), Tamil (ta), Telugu (te), Marathi (mr), Gujarati (gu), Kannada (kn), Malayalam (ml), Odia (or), Punjabi (pa), Hinglish (hi-en).
- India-centric content: regional exams (UPSC, state PSCs), festivals, local history — not translated MMLU.
- Paper baselines (5-shot, paper Table 2): GPT-4o 74.0%, Llama-3-70B 67.1%, Airavata 39.7%, random 25.0%.

### IndicGenBench

- Google Research, 2024. Paper: [arXiv 2404.16816](https://arxiv.org/pdf/2404.16816). Repo: [google-research-datasets/indic-gen-bench](https://github.com/google-research-datasets/indic-gen-bench).
- Generative tasks across **29 Indic languages**: Cross-Sum (summarization), XQuAD (extractive QA), XorQA (cross-lingual QA), FLORES (translation).
- **Wired here: IndicXQuAD only** — extractive QA, simplest scoring, smallest data footprint. Pick the shortest gold-span from the passage given the question.
- Scoring: SQuAD-style exact-match (EM) after lowercasing, punctuation strip, article strip, whitespace collapse.

## Pipeline

```
┌──────────────────────────┐
│  tinygpt eval-indic      │
│  --task milu | indic…    │
└──────────┬───────────────┘
           │ ModelLoader.load
           ▼
┌──────────────────────────┐
│  AnyModel (.tinygpt or   │
│  HF dir)                 │
└──────────┬───────────────┘
           │ per row
           ▼
┌─────────────────────────────────────────────┐
│  MILU            │  IndicGenBench (xquad)   │
│  ──────────────  │  ──────────────────────  │
│  per-option CE   │  greedy generate (≤32)   │
│  argmax LL       │  SQuAD-norm exact-match  │
└─────────────────────────────────────────────┘
```

Implementation in [`native-mac/Sources/TinyGPT/EvalIndic.swift`][src].

[src]: ../../native-mac/Sources/TinyGPT/EvalIndic.swift

## CLI

```bash
# MILU (Hindi split, 100 samples)
tinygpt eval-indic --task milu \
    --model /tmp/flagship-huge.tinygpt \
    --milu-data ~/.cache/tinygpt/datasets/ai4bharat/MILU/hi.jsonl \
    --limit 100

# IndicGenBench XQuAD (Hindi split, 50 samples)
tinygpt eval-indic --task indicgenbench --subtask xquad \
    --model /tmp/flagship-huge.tinygpt \
    --indicgen-data ~/.cache/tinygpt/datasets/google/IndicGenBench_xquad_in/hi.jsonl \
    --limit 50

# Both, aggregate report
tinygpt eval-indic --task all \
    --model /tmp/flagship-huge.tinygpt \
    --milu-data … --indicgen-data … --limit 100
```

## Data setup

Datasets are not bundled — pre-fetch with the existing dataset loader:

```bash
tinygpt download-dataset ai4bharat/MILU
tinygpt download-dataset google/IndicGenBench_xquad_in
```

Or any local JSONL with the schemas below. See the source-file
docstring for full schema details.

**MILU row**:
```json
{
  "question": "<text>",
  "option1": "...", "option2": "...", "option3": "...", "option4": "...",
  "answer": "option2",    // or "B" / 2 / literal text — all accepted
  "language": "Hindi",    // optional
  "subject": "History"    // optional
}
```

**IndicXQuAD row** (SQuAD-derivative shape):
```json
{
  "question": "<text>",
  "context": "<paragraph>",
  "answers": { "text": ["gold answer", "alt"], "answer_start": [42] },
  "language": "hi"
}
```

## Scoring details

### MILU — log-likelihood argmax

For each option we form `Question: <q>\nAnswer: <option>` and compute
cross-entropy on the **option tokens only** (masked-loss path, same as
SFT response-only scoring). Pick the option with the lowest CE. This
matches lm-eval-harness's `multiple_choice` task type.

**Why not just generate and string-match the option letter?** Because
that confounds "the model knows the answer" with "the model knows the
output format". A 27M-param byte-level model has no clue about
markdown-style "A./B./C." prompts; log-likelihood ranking is template-
neutral.

### IndicXQuAD — greedy + SQuAD EM

Greedy decode `Context: <c>\nQuestion: <q>\nAnswer:` for ≤32 new
tokens (tunable via `--max-new-tokens`). Truncate generation at the
first newline (model often hallucinates a follow-up Q). Compare against
each gold answer in `answers.text[]` after the standard SQuAD
normalization: lowercase, strip articles (a/an/the), strip punctuation,
collapse whitespace.

## Baseline numbers — Shakespeare byte-level (smoke run)

Run: `tinygpt eval-indic --task all --model
data/checkpoints/huge-shakespeare-5000-loss1.22.tinygpt --milu-data
/tmp/milu-smoke.jsonl --indicgen-data /tmp/xquad-smoke.jsonl --limit 4`

The smoke fixtures are 4 English MCQ rows + 2 English XQuAD rows
(handcrafted, see this commit's terminal log). The model is a
12-layer / 256-dim / vocab-256 byte-level Shakespeare LM trained for
5000 steps at val loss 1.22.

| Eval | Score | Sample size | Notes |
|---|---|---|---|
| MILU (smoke) | 0.00% | 4 | argmax-LL picked wrong option each time |
| IndicXQuAD (smoke) | 0.00% EM | 2 | greedy decoded "state of th…" |

This is **the documented zero baseline**: an English Shakespeare LM
has no Indic-language ability at all, AND its 256-byte tokenizer can't
even represent Devanagari/Tamil/Bengali tokens. Devanagari characters
are 3-byte UTF-8 sequences; the model has never seen those byte
trigrams. This run validates the pipeline end-to-end (model loads,
JSONL parses, MCQ option scoring picks one of N, XQuAD greedy decode +
SQuAD-EM works) without making a claim about Indic ability.

## What a real baseline run needs

1. **Pre-fetch real MILU data** (~85k questions × 11 languages):
   `tinygpt download-dataset ai4bharat/MILU` — produces JSONL per
   language at `~/.cache/tinygpt/datasets/ai4bharat/MILU/<lang>.jsonl`.
   First-time download is ~50MB; per-language shards are 2-8MB.
2. **Pre-fetch IndicGenBench XQuAD**: `tinygpt download-dataset
   google/IndicGenBench_xquad_in` — ~12MB for the Indic XQuAD shard.
3. **Pick a real base model**: flagship-huge (221M params, byte-
   level) will also score ~0 on Indic — the right baseline targets are
   Qwen-3 (or smollm2) HF-loaded, then Sarvam-Edge once it ships. See
   [Wave 4 landscape §4](wave_4_landscape.md#4-multilingual--india-focus).
4. **Run the eval** with `--limit 200` per language for the first pass
   (~30 min on a 220M model per language); scale up once the smoke
   number looks plausible.

## Known limitations (current shipping state)

1. **No few-shot prompting.** The MILU paper uses 5-shot. We do
   zero-shot. Score gap with paper numbers: 5–8 points for capable
   models, ~0 for byte-level.
2. **No batched scoring.** Each option is scored serially; XQuAD
   generates one row at a time. Acceptable for `--limit 200`;
   prohibitive for full MILU (85k × 4 forwards × 11 langs = 3.7M
   forwards).
3. **One IndicGenBench subtask only.** XQuAD is wired; Cross-Sum (free-
   form generation, needs ROUGE), XorQA (cross-lingual answer
   alignment), and FLORES (translation, BLEU + chrF) need separate
   scoring code. See §next-steps.
4. **No Hinglish / code-switching handling.** MILU's `hi-en` split is
   passed through unchanged; tokenizer-side handling of Romanized
   Hindi is up to the model's tokenizer.
5. **Tokenizer-bloat penalty is invisible.** A model with a
   Devanagari-unfriendly tokenizer (Qwen3, smollm2) will pay 2–4×
   token-bloat on Hindi prompts — this constrains how much context
   fits, but doesn't directly show up in the score. See [Wave 4
   landscape §4](wave_4_landscape.md#4-multilingual--india-focus) on
   why Sarvam's tokenizer is the right Indic choice.

## Next steps

In rough priority order:

1. **Cross-Sum (IndicGenBench)** — adds ROUGE-L scoring. The Karpathy-
   style fix-it: implement ROUGE-L in pure Swift (~30 LOC), reuse the
   greedy-generation path.
2. **Batched MCQ scoring for MILU** — score all 4 options in one
   forward by padding to max-option length. ~4× throughput; needed
   to make full-MILU runs tractable on M-series hardware.
3. **5-shot prompting** — concatenate few-shot exemplars from the
   same language/subject before the test question. The MILU repo
   ships its few-shot exemplar set; pull it via the standard data
   loader.
4. **lm-eval-harness task YAML** — write a `bench/tasks/milu_*.yaml`
   that drives MILU through the existing
   [lm-eval-harness integration](../lm_eval_integration.md). This
   would let MILU benefit from the harness's batching and few-shot
   plumbing for free, at the cost of an HTTP roundtrip per option.
5. **Re-tokenization audit** — for each candidate base model, run a
   token-bloat measurement: `tokens_per_char = encode(hindi_text).count
   / hindi_text.count`. Sarvam ≈ 0.5, Qwen3 ≈ 1.5–2.0. Document this
   per-model so the bloat penalty is explicit.

## Citation block

```bibtex
@inproceedings{milu2025,
  title={MILU: A Multi-task Indic Language Understanding Benchmark},
  author={Verma, Sshubam and others (AI4Bharat)},
  booktitle={NAACL},
  year={2025},
  url={https://arxiv.org/abs/2411.02538}
}

@article{indicgenbench2024,
  title={IndicGenBench: A Multilingual Benchmark to Evaluate
         Generation Capabilities of LLMs on Indic Languages},
  author={Singh, Harman and others (Google Research)},
  year={2024},
  url={https://arxiv.org/abs/2404.16816}
}
```

## Related

- [Wave 4 landscape — §4 Multilingual](wave_4_landscape.md#4-multilingual--india-focus) — Sarvam, Airavata, tokenizer choice
- [lm-evaluation-harness integration](../lm_eval_integration.md) — English-track eval harness this is the Indic counterpart of
- [Progress dashboard](../progress.md) — Wave 4 row tracking this work
