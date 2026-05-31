---
title: Dataset inventory — what's available, sizes, schemas
description: Reference doc for every dataset wired into tinygpt — registry entries, what they're for, how to pull them, what the records look like after conversion, and known gotchas (gated datasets, parquet decode).
---

# Dataset inventory

This is the working reference for "what dataset should I use for X."
Every entry is in the curated `list-datasets` registry; this doc
adds the practical bits (downloadable today? what's in the file?
known gotchas?) the registry doesn't cover.

**Quick map**:
- Browse: `tinygpt list-datasets [--specialist kind | --format sft|dpo|plain]`
- Pull: `tinygpt download-dataset hf://datasets/owner/name --out path.jsonl`
- Convert: `tinygpt extractor-data` (BFCL/τ-bench → router pairs)
- Default cache: `~/.cache/tinygpt/datasets/`

## Tool-calling (north-star primary)

| Dataset | Size | Schema | Status | Gotchas |
|---|---|---|---|---|
| `Salesforce/xlam-function-calling-60k` | ~80 MB | sft (query + tools + answer) | **GATED** | Needs `HF_TOKEN` + accept license at HF |
| `NousResearch/hermes-function-calling-v1` | ~50 MB JSONL | `{instruction, response}` 11,230 records, response wraps tool call in `<tool_call>…</tool_call>` | ✅ pulls clean | None |
| `Locutusque/function-calling-chatml` | ~60 MB | sft, ChatML conversations | **PARQUET** | tinygpt's converter doesn't decode parquet yet — file lands as `.parquet` shards; manual decode needed |

**Verified pull (commit `f566023`)**: hermes-function-calling-v1
schema-sniffed as `sft (confidence 75%, chat array → conversations)`,
11,230 records / 8.4M tokens / 6.5M scored tokens. **The response
format is `<tool_call>{"name": ..., "arguments": ...}</tool_call>` —
not raw JSON. Trainees must learn this XML-wrap to score on BFCL
metrics.**

## Code + debugger

| Dataset | Size | Format | Status |
|---|---|---|---|
| `princeton-nlp/SWE-bench_Verified` | ~50 MB | plain (eval set) | Open |
| `princeton-nlp/SWE-bench` | ~3 GB | sft | Open (large) |
| `bigcode/the-stack-smol` | ~250 MB | plain | Open |
| `iamtarun/python_code_instructions_18k_alpaca` | ~12 MB | sft (alpaca-style) | Open, small, ideal smoke base |
| `open-r1/codeforces-cots` | ~1.5 GB | sft | Open, reasoning trace heavy |
| `bigcode/commitpack` | ~4 TB | sft | **Subset recommended** — full set will fill any disk |

For the debugger specialist, the natural starting corpus is
**SWE-bench_Verified + python_code_instructions_18k_alpaca** (~62 MB
total), with **issue→PR pairs from `tinygpt fetch-github`** added on
top for repo-specific context. SWE-bench Verified is also the
canonical eval target.

## Math + reasoning

| Dataset | Size | Format | Notes |
|---|---|---|---|
| `meta-math/MetaMathQA` | ~200 MB | sft | Foundational math instruction set |
| `AI-MO/NuminaMath-CoT` | ~800 MB | sft, chain-of-thought | Heavier math reasoning |
| `nvidia/OpenMathReasoning` | ~1 GB | sft | Long-form reasoning traces |
| `open-thoughts/OpenThoughts-114k` | ~3 GB | sft | Reasoning trace corpus |
| `open-thoughts/OpenThoughts2-1M` | ~30 GB | sft | XL reasoning corpus — use sample |

## Instruct (general)

| Dataset | Size | Format | Status |
|---|---|---|---|
| `yahma/alpaca-cleaned` | ~25 MB | sft (alpaca) | Already cached at `~/.cache/tinygpt/datasets/yahma/` |
| `iamtarun/python_code_instructions_18k_alpaca` | ~12 MB | sft | (also in Code section) |
| `teknium/OpenHermes-2.5` | ~1.6 GB | sft | Large general-purpose SFT |
| `HuggingFaceH4/ultrachat_200k` | ~1.2 GB | sft | Multi-turn chat |

## Preference (DPO)

| Dataset | Size | Format | Notes |
|---|---|---|---|
| `argilla/ultrafeedback-binarized-preferences-cleaned` | ~200 MB | dpo | Standard DPO training corpus |
| `HuggingFaceH4/ultrafeedback_binarized` | ~250 MB | dpo | Same-family alternative |
| `Intel/orca_dpo_pairs` | ~50 MB | dpo | Smaller, faster smoke option |

## General pretrain corpora

| Dataset | Size | Format | Notes |
|---|---|---|---|
| `roneneldan/TinyStories` | ~1 GB | plain | Curriculum-style; great for small from-scratch bases |
| `HuggingFaceFW/fineweb-edu` | ~1.3 TB | plain | Use sample only |

## Indic / multilingual evals (not training data)

| Eval | Size | What it scores | Wire-up |
|---|---|---|---|
| `ai4bharat/MILU` | ~50 MB | MMLU-style MCQ, 11 Indic langs | `tinygpt eval-indic --task milu --milu-data <path>` (scaffold only — eval CLI works, run on real data is operator step) |
| `google/IndicGenBench` (XQuAD subtask) | varies | Extractive QA, 29 Indic langs | `tinygpt eval-indic --task indicgenbench --subtask xquad` |

## Special pipelines (not HF Datasets)

| Source | CLI | Output | Notes |
|---|---|---|---|
| **GitHub REST API** (issue→PR, reviews, commits) | `tinygpt fetch-github <owner/repo>` | per-record JSONL | Rate-limited 60 req/h without `GITHUB_TOKEN`; 5000 req/h with one |
| **BFCL** (Berkeley Function-Calling) | `tinygpt extractor-data --bfcl <path>` | `{query, tool}` JSONL for mini-router training | Walks the BFCL JSON dump |
| **τ-bench** | `tinygpt extractor-data` | `{query, tool}` pairs | Best-effort parser; full τ-bench ships Python files needing pre-conversion |
| **Synthetic (Magpie)** | `tinygpt magpie <chat-tuned-base>` | `{instruction, response}` JSONL | Needs a chat-tuned base; common bootstrap for low-resource tools |
| **Synthetic (cloud)** | `tinygpt extractor-data --synth` | augments small classes via Claude/GPT | Uses `CloudEscalate` — needs `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` |

## Known gotchas

1. **Gated datasets need `HF_TOKEN`**. The CLI surfaces the
   accept-license URL — copy it, click through, then
   `export HF_TOKEN=hf_xxx`. xlam-function-calling-60k is the most
   prominent example.

2. **Parquet shards aren't decoded yet**. Some datasets only ship
   as `.parquet` (e.g., `Locutusque/function-calling-chatml`). The
   `tinygpt download-dataset` CLI surfaces this with a clear error
   and the cache path. Decode manually via Python pandas /
   pyarrow until upstream support lands.

3. **JSONL vs ChatML wrap** is a real footgun. `tinygpt sft
   --template chatml` wraps **everything** in
   `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}`.
   The hermes records already prefix `"system: ..."` inline, so all
   of that ends up in the user turn at training time. Test prompts
   at inference must match this shape, NOT the proper
   `<|im_start|>system\n...\n<|im_start|>user\n...\n<|im_start|>assistant`
   you might expect. See `docs/specialist_v1_findings.md`.

4. **macOS reaps `/tmp`**. Long-lived training caches should go
   to `~/.cache/tinygpt/` or a stable project directory. `/tmp`
   gets cleaned aggressively (saw this mid-session on 2026-05-31).

5. **The 22-entry registry isn't exhaustive**. It's the curated
   slice that's been tested with `tinygpt download-dataset`'s
   schema sniffer. Other HF datasets work if you pass the field
   names manually via `--map`.

## Recommended starting bundles

| Goal | Pull bundle | Total size |
|---|---|---|
| **Tool-calling specialist (Wave 3 first run)** | hermes-function-calling-v1 | ~50 MB |
| **Add tool-calling diversity** | + Locutusque/function-calling-chatml (after parquet support), + xlam (after HF_TOKEN) | +140 MB |
| **Debugger specialist** | python_code_instructions_18k_alpaca + SWE-bench_Verified + `fetch-github` from 2-3 OSS repos | ~100 MB + repo data |
| **General SFT smoke** | alpaca-cleaned (already cached) | 25 MB |
| **DPO smoke** | Intel/orca_dpo_pairs | 50 MB |
| **Indic eval baseline** | MILU + IndicGenBench XQuAD subset | varies |

## How to extend this doc

When a new dataset becomes interesting, add a row to the right table
+ note any gotchas in the "Known gotchas" section. Keep entries
short (one-line schema, one-line status, one-line gotcha). The
canonical "what models can train on what" doc is
`docs/capability_matrix.md`.
