# The three phases of training — pretrain, SFT, DPO

A modern useful language model is the product of three distinct training
phases, each with its own dataset shape, loss function, and goal. This
guide walks the whole pipeline as it exists in TinyGPT today, with the
exact commands to reproduce each step.

Three phases, in order:

| Phase | Goal | Dataset shape | Loss | Compute share at labs |
|---|---|---|---|---:|
| **Pretrain** | Learn the structural prior of language | Continuous text | Causal next-token cross-entropy | ~50-70% |
| **SFT** (supervised fine-tune) | Follow instructions in a chat format | `{prompt, response}` pairs | Same CE, but masked to response tokens only | ~5-15% |
| **DPO** (direct preference optimization) | Prefer better responses over worse ones | `{prompt, chosen, rejected}` triplets | Log-sigmoid of policy/reference log-ratio difference | ~10-30% |

The first phase produces a base model that *can complete text*; the
second teaches it to *respond to instructions*; the third teaches it to
*prefer good responses over bad*. Lab-scale models spend ~70% of total
compute on pretraining and ~30% on SFT+DPO combined; at our scale, the
ratio inverts (pretrain is cheap-but-data-limited, post-training is the
multiplier).

---

## 1. Pretraining

### What it does

Given an enormous stream of raw text, predict the next token everywhere.
Loss is averaged over every position; gradients flow through every
token. The model learns grammar, vocabulary, world facts, and a
distribution over what humans tend to write.

### Math

```
L_pretrain = - (1 / N) * Σ_t  log P(x_{t+1} | x_1 … x_t)
```

where `x_1 … x_N` are the corpus tokens. Averaging over a single
contiguous corpus makes loss directly comparable across runs.

### What it needs

| Thing | Why | Where it lives |
|---|---|---|
| **Large text corpus** | ~5-20× more tokens than the model has parameters (Hoffmann/Chinchilla) | Streamed from HuggingFace via `python_ref/fetch_hf_corpus.py` |
| **BPE tokenizer** | Byte-level wastes ~4× the compute at the same coverage | `--tokenizer <hf-dir>` pointing at any HF model directory |
| **Long-run infrastructure** | A crash at hour 22 of 26 shouldn't lose 22 hours | Tier 0 safety nets in `tinygpt train`: resume, atomic save-every, SIGINT-flushes-final |
| **bf16 training** | 2× memory savings → 2× larger effective batch | `--dtype bfloat16` |
| **Gradient accumulation** | Effective batch larger than memory budget | `--accum N` |

### Reproduce

```bash
# 1. Stream ~500M tokens of high-quality educational web text.
source python_ref/.venv/bin/activate
python python_ref/fetch_hf_corpus.py \
    --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \
    --split train --target-tokens 500M \
    --out /tmp/fineweb-edu-500M.txt

# 2. Pretrain Mega-bf16 (76M body + 25M token embedding = ~100M total).
#    B=4 × accum=4 × ctx=1024 = effective batch 16 at ~2 GB GPU memory.
#    ~23 hours on M5 Pro / 48 GB.
cd native-mac
caffeinate -di .xcode-build/Build/Products/Debug/tinygpt train \
    --preset mega \
    --tokenizer /tmp/smollm2 \
    --corpus /tmp/fineweb-edu-500M.txt \
    --out /tmp/mega-fineweb.tinygpt \
    --dtype bfloat16 \
    --batch 4 --accum 4 --ctx 1024 \
    --steps 30500 \
    --lr-schedule cosine --warmup 1000 \
    --max-lr 6e-4 --min-lr 6e-5 \
    --val-split 0.005 --val-every 500 --save-every 1000
```

### Expected outcome at our scale

| Tokens trained on | Predicted val loss | What it looks like |
|---|---:|---|
| 5 M (Tiny demo) | 4.9 | gibberish, fragments |
| 50 M | 4.0 | real words, broken grammar |
| **500 M (this run)** | **3.0-3.5** | coherent fragments, GPT-2-124M-class |
| 1.5 B (Chinchilla floor) | 2.5 | useful base, post-trainable |
| 5 B | 2.0 | Pythia-1.4B-class base |

A "good pretrain" is anywhere from loss ~2.5 to ~3.5. Below that, the
base is becoming useful on its own; above it, post-training is doing
nearly all the work.

---

## 2. Supervised fine-tuning (SFT)

### What it does

Given a base that can complete text, teach it to follow instructions.
Same model, same forward pass, same cross-entropy loss — but the data
is `{instruction, response}` pairs templated through a chat format
(`<|im_start|>user … <|im_start|>assistant …`), and the loss is
masked to score **only the response tokens**.

### Why masking matters

Without the response-only mask, the loss includes the instruction
tokens. The gradient signal pushes the model toward predicting the
instruction back to itself — useless. With the mask, only the response
positions contribute, and the model learns "given THIS prompt, produce
THAT response."

```
L_SFT = - (1 / |R|) * Σ_{t in R}  log P(x_{t+1} | x_1 … x_t)
```

where `R` is the set of response positions. Identical math to pretrain
except for the index set.

### Templates

Three are supported by `tinygpt sft --template`:

```
chatml  (default, matches SmolLM2 / Qwen tokenizers)
  <|im_start|>user
  Capital of France?<|im_end|>
  <|im_start|>assistant
  Paris.<|im_end|>

alpaca
  ### Instruction:
  Capital of France?

  ### Response:
  Paris.

llama
  [INST] Capital of France? [/INST] Paris.
```

Use whatever template matches the tokenizer the base was trained
against. SmolLM2's tokenizer treats ChatML markers as single tokens; the
others would tokenize them as raw text.

### What datasets to use

| Dataset | Size | Style | Notes |
|---|---:|---|---|
| `databricks/databricks-dolly-15k` | 15K | hand-written instructions | High quality, small. Good first run. |
| `HuggingFaceH4/no_robots` | 10K | hand-written, diverse | Pairs well with Dolly |
| `tatsu-lab/alpaca` | 52K | GPT-generated | Broader, lower per-pair quality |
| `OpenAssistant/oasst1` | ~10K conversations | multi-turn human | Use for chat-shape SFT |

For first runs, Dolly is the canonical pick.

### Reproduce

```bash
# Tokenize Dolly into JSONL (one record per line).
python python_ref/fetch_hf_corpus.py \
    --dataset databricks/databricks-dolly-15k \
    --target-tokens 50M \
    --out /tmp/dolly.jsonl
# Hand-massage into {instruction, response} JSONL (the fetcher writes
# raw text; for SFT we want the structured form — see docs/sft_data.md
# for the one-liner).

# SFT on top of the pretrained base. Adapter is rank-4 LoRA — adapter
# file is ~MB, base stays frozen.
.xcode-build/Build/Products/Debug/tinygpt sft \
    /tmp/mega-fineweb.tinygpt \
    --data /tmp/dolly.jsonl \
    --template chatml \
    --rank 4 --alpha 8 \
    --steps 500 \
    --out /tmp/mega-sft.lora
```

### How to know it worked

Sample with and without the adapter and compare:

```
# Base only — completes text but doesn't follow instructions
tinygpt sample /tmp/mega-fineweb.tinygpt --prompt "User: What is 2+2?" --tokens 50

# With SFT adapter — responds in the expected format
tinygpt sample /tmp/mega-fineweb.tinygpt --lora /tmp/mega-sft.lora \
    --prompt "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n" \
    --tokens 50
```

The masked-tokens count printed by `tinygpt sft` tells you how much
signal you actually trained on — for Dolly that's ~1.5 M response
tokens, vs ~3 M total prompt+response tokens. Half the data is
"context for the loss, not scored."

---

## 3. Direct preference optimization (DPO)

### What it does

Given a base + an SFT adapter, take it one step further: train the
model to PREFER one response over another. The data is
`{prompt, chosen, rejected}` triplets — humans or a stronger model
ranked the two responses.

### Math

Define the implicit reward function:
```
r_θ(y | x) = log π_θ(y | x) - log π_ref(y | x)
```

where `π_θ` is the policy (the model we're training) and `π_ref` is a
frozen reference (a copy of the base before DPO). Then the DPO loss:

```
L_DPO = - E_{(x, y_w, y_l)} [ log σ ( β · (r_θ(y_w | x) - r_θ(y_l | x)) ) ]
```

Expanding the reward:

```
L_DPO = - log σ ( β · ( logπ_pol(chosen)   - logπ_pol(rejected)
                      - logπ_ref(chosen)   + logπ_ref(rejected) ) )
```

At step 0, `π_θ = π_ref` (policy starts as a copy of reference), so the
log-ratios cancel and the inner expression is 0; the loss is
`-log σ(0) = log 2 ≈ 0.693`. **That's the canonical sanity check —
the first DPO step should print loss ≈ 0.69.**

`β` is the temperature: lower keeps the policy close to the reference
(safer, more conservative); higher sharpens preferences (more
aggressive, more risk of drift). 0.1 is a typical default.

### Why a reference model?

The reference is a regularizer. Without it, the model would maximize
chosen-vs-rejected by any means including catastrophic
shifts in the output distribution. The KL constraint to the reference
keeps the policy in a meaningful neighborhood of the base.

Memory cost: ~2× the base size (policy + reference both held in
memory). At bf16 on a 100M Mega, that's ~400 MB.

### What datasets to use

| Dataset | Size | Source of preference | Notes |
|---|---:|---|---|
| `HuggingFaceH4/ultrafeedback_binarized` | 60K pairs | GPT-4 judgments | Strong default. |
| `argilla/dpo-mix-7k` | 7K | mixed sources, cleaned | Smaller, higher per-example quality |
| `anthropic/hh-rlhf` | ~170K | human labels | Slow but human-grade |

### Reproduce

```bash
# Once we tokenize UltraFeedback into the JSONL shape DPO expects.
.xcode-build/Build/Products/Debug/tinygpt dpo \
    /tmp/mega-fineweb.tinygpt \
    --data /tmp/ultrafeedback.jsonl \
    --template chatml \
    --rank 4 --alpha 8 \
    --beta 0.1 \
    --steps 500 \
    --lr 5e-5 \
    --out /tmp/mega-dpo.lora
```

`tinygpt dpo` accepts either the flat `{prompt, chosen, rejected}`
shape or the HF chat-array shape — see `PreferenceReader` for details.

### How to know it worked

DPO loss alone is hard to interpret directly. The useful signal is
**preference accuracy**: at evaluation, sample two responses from the
policy and the reference for the same held-out prompt, run them through
a stronger judge model, and report what fraction of the time the policy
beats the reference. That's an upcoming `tinygpt dpo-eval` command;
for now, eyeball samples.

---

## End-to-end pipeline

The three phases compose into one workflow:

```bash
# Phase 1 — pretrain on FineWeb-edu 500M (~23 hr).
caffeinate -di tinygpt train --preset mega --tokenizer /tmp/smollm2 \
    --corpus /tmp/fineweb-edu-500M.txt --out /tmp/mega.tinygpt \
    --dtype bfloat16 --batch 4 --accum 4 --ctx 1024 \
    --steps 30500 --lr-schedule cosine --warmup 1000 \
    --save-every 1000 --val-split 0.005 --val-every 500

# Phase 2 — SFT on Dolly (~30 min).
tinygpt sft /tmp/mega.tinygpt --data /tmp/dolly.jsonl \
    --template chatml --steps 500 --out /tmp/mega-sft.lora

# Phase 3 — DPO on UltraFeedback (~30 min).
tinygpt dpo /tmp/mega.tinygpt --data /tmp/ultrafeedback.jsonl \
    --template chatml --beta 0.1 --steps 500 \
    --out /tmp/mega-dpo.lora

# Sample with the full stack — base + SFT + DPO adapters.
tinygpt sample /tmp/mega.tinygpt \
    --lora /tmp/mega-sft.lora --lora-weight 1.0 \
    --lora /tmp/mega-dpo.lora --lora-weight 1.0 \
    --prompt "<|im_start|>user\nExplain DPO simply.<|im_end|>\n<|im_start|>assistant\n"
```

A weekend's worth of compute on one M5 Pro produces a 100M-param
instruction-following model that scores ~2.5-3.5 on TinyStories PPL
and follows simple conversational prompts in the ChatML format. Not
GPT-quality — but a working artifact end-to-end.

---

## Background reading

- **Pretraining scaling laws**: Hoffmann et al., 2022 (Chinchilla);
  Kaplan et al., 2020 (Kaplan scaling laws).
- **SFT response-only loss**: standard practice since GPT-3
  fine-tuning. The mechanic of masking to the response is described
  cleanly in the [Alpaca paper](https://arxiv.org/abs/2303.18223)
  appendix.
- **DPO**: Rafailov et al., 2023 ("Direct Preference Optimization:
  Your Language Model is Secretly a Reward Model"), NeurIPS 2023.
  The original paper; the closed-form derivation in §4 is the math
  we implement.
- **Why this order**: the [LIMA paper](https://arxiv.org/abs/2305.11206)
  argues most "alignment" is shallow — pretraining does the heavy
  lifting, SFT teaches format, DPO polishes. Our pipeline structure
  matches that thesis.
