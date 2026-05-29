# Phase 1-10 validation + end-to-end workflows

This doc captures the validation runs and workflows that exercise
the features shipped in Phases 1-10. The goal isn't peer-review
rigour — it's enough evidence that each surface RUNS end-to-end on
real data, so a future user knows the pieces actually compose.

---

## 1. End-to-end training with the new architecture knobs

Run a Huge model on FineWeb-Edu with `--diff-attn` and `--mod`
enabled, plus the standard stability stack (`--grad-clip 1.0`,
cosine LR, val split, atomic save).

```sh
caffeinate -di tinygpt train \
    --preset huge \
    --tokenizer /tmp/smollm2 \
    --corpus /tmp/fineweb-edu-500M.txt \
    --dtype bfloat16 --batch 4 --accum 4 --ctx 512 \
    --steps 500 \
    --diff-attn --mod \
    --lr-schedule cosine --warmup 50 --max-lr 6e-4 --min-lr 6e-5 \
    --save-every 100 --val-split 0.005 --val-every 100 \
    --sample-every 999999 \
    --out /tmp/validation-huge.tinygpt
```

Expected runtime (M-series, bf16):
- BPE tokenize: 20-30 min on 2 GB corpus (cached on subsequent runs)
- Train: ~500 steps × ~0.5 s/step = 4-5 min

Success criteria:
- Training completes without OOM, NaN, or kernel errors
- Final loss decreases relative to step 0
- Checkpoint loads + samples successfully via `tinygpt sample`

The combination of DiffAttn (2× attention projections + λ) and MoD
(sigmoid gate per token per block) puts both new architectural
surfaces under load simultaneously. If either had a wiring bug, this
run would crash or NaN.

## 2. MoE end-to-end (the Phase 5 deliverable)

Now that MoE save/load works, this is the workflow that bookends
Phase 5: train a tiny MoE on BPE-tokenized data, save it, reload,
sample.

```sh
# Step 1 — train a tiny MoE
tinygpt train \
    --preset tiny \
    --tokenizer /tmp/smollm2 \
    --corpus /tmp/smoke-corpus.txt \
    --moe-experts 4 --moe-topk 2 \
    --steps 200 \
    --out /tmp/moe-tiny.tinygpt

# Step 2 — sample from the saved MoE
tinygpt sample /tmp/moe-tiny.tinygpt \
    --prompt "Once upon a time" --tokens 80 --temperature 0.8
```

Success criteria:
- Step 1 trains without router collapse (loss decreases)
- Step 2 loads the MoE blocks correctly and produces coherent text
- `tinygpt inspect /tmp/moe-tiny.tinygpt` shows `moe.router.weight`
  and `moe.experts.0..3.{fc_in,fc_out}.weight` entries per layer

## 3. MoE distillation pipeline

The original Phase 5 headline was "distill from a big teacher into
our smaller MoE". The mechanics:

```sh
# Step 1 — initialise a small MoE student with the teacher's tokenizer.
tinygpt train \
    --preset tiny \
    --tokenizer /tmp/smollm2 \
    --corpus /tmp/smoke-corpus.txt \
    --moe-experts 4 --moe-topk 2 \
    --steps 100 \
    --out /tmp/moe-student-init.tinygpt

# Step 2 — distill from SmolLM2 (HF, dense) into the MoE student.
tinygpt distill /tmp/moe-student-init.tinygpt \
    --teacher /tmp/smollm2 \
    --corpus /tmp/smoke-corpus.txt \
    --tokenizer /tmp/smollm2 \
    --steps 200 --temperature 4 --alpha 0.7 \
    --out /tmp/moe-distilled.tinygpt
```

The student is a from-scratch MoE; the teacher is an HF dense model
(SmolLM2). Both share the SmolLM2 tokenizer so the cross-entropy
on softmax distributions is well-defined.

Note: distilling INTO an HF MoE teacher (Mixtral, DeepSeek) is the
next step but is blocked on the HF MoE safetensors loader — see
`docs/phase_9_10_status.md`.

## 4. Interpretability — tuned-lens pipeline

Trained probes give better-calibrated per-layer predictions than the
raw final-LN + LM-head lens.

```sh
# Train the probes — base frozen, only the lens probes update.
tinygpt tuned-lens /tmp/validation-huge.tinygpt \
    --corpus /tmp/smoke-corpus.txt \
    --steps 300 --lr 1e-3 \
    --out /tmp/huge.lenses
```

Browser side: open the playground, load a model, click the 🎯 button
next to "Logit lens", select the `.lenses` file. The worker parses it
and uses the trained probes on the next "Logit lens" click — the
ASCII table shows per-layer predictions that are SHARPER than the
raw lens (the trained probes are layer-calibrated; the raw lens is
not).

## 5. Activation patching + ablation in the browser

After loading any gallery model:
- **"Ablate & sample"** — pick a layer index + a target (attn, mlp,
  or whole layer) → that component is zeroed at every position during
  generation. Reveals how load-bearing the block is.
- **Patch button** (via worker `patch` message; UI in next iteration)
  — zero out one (layer, position) pair in the residual stream.
  Pinpoints whether THAT token's representation at THAT depth was
  load-bearing.

## 6. LASER + HQQ — post-hoc weight surgery

Post-training operations on a finished `.tinygpt` file:

```sh
# Drop the bottom 30% of singular components from the late layers'
# MLP outputs. Sometimes improves downstream accuracy by removing
# the "noise tail" that the higher components had to fight.
tinygpt laser /tmp/validation-huge.tinygpt \
    --target mlp.fc_out --layers 8-11 \
    --rank-fraction 0.7 \
    --out /tmp/huge-lasered.tinygpt

# Quantize-then-dequantise via HQQ's IRLS solver. Stores the
# REQUANTISED weights as dense fp32 (the inference-time memory win
# would require a packed-int4 matmul kernel).
tinygpt hqq /tmp/validation-huge.tinygpt \
    --bits 4 --group-size 64 --p 0.7 \
    --layers 0-11 \
    --out /tmp/huge-hqq.tinygpt
```

Both operate at the .tinygpt file level — load, modify the weight
tensors, write a new file. The rest of the toolchain (sample, eval,
finetune) treats the output identically to the input.

## 7. ES — gradient-free training

A separate trainer for non-differentiable rewards or as an
educational counterpoint to AdamW:

```sh
tinygpt es /tmp/validation-huge.tinygpt \
    --corpus /tmp/smoke-corpus.txt \
    --steps 50 --population 40 --sigma 0.02 --lr 0.01 \
    --out /tmp/huge-es.tinygpt
```

Per step: K=40 forward passes (no backward). Slower per step than
SGD but fully parallelisable and works on rewards that aren't
differentiable.

## 8. Magpie — synthetic SFT data

Bootstrap an SFT dataset from any chat-format base:

```sh
tinygpt magpie /path/to/chat-tuned-model \
    --count 500 --template chatml --temperature 0.9 \
    --out /tmp/magpie-sft.jsonl
```

Each line: `{"instruction": "...", "response": "..."}`. Pipe this
into `tinygpt sft` for fine-tuning, applying user-side quality
filters (length ≥ 20, no repeating loops) first.

---

## What this validation does NOT cover

- **Leaderboard scoring** of the validation artifact. The benchmark
  scorers (`browser/score_gallery.mjs`) target byte-level models in
  the gallery format; scoring a BPE-tokenized Huge would need a
  parallel scoring path.
- **Long-context behaviour** — YOCO's KV-cache memory win only
  shows up at ctx ≥ 1024 with autoregressive decode. The 500-step
  validation runs at ctx=512.
- **Quality A/B** between configurations. We're validating that
  things RUN, not that the new features improve quality on a
  specific task. Doing the A/Bs is the next round of experiments.

This is a "smoke" report — strong evidence the pieces compose, weak
evidence on RELATIVE performance. Pulling the latter takes its own
focused experiment + the leaderboard scorers extending to BPE
models.
