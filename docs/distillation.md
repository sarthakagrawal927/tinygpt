# Knowledge distillation — making a tiny model punch above its weight

Distillation is the post-training technique where a SMALL model
("student") is trained to match a LARGER model's ("teacher's") output
distribution on a corpus. The result: students that significantly
outperform what you'd get by training the same architecture from
scratch on the same data — sometimes by 2-5x in perplexity terms.

This guide covers the workflow as it exists in `tinygpt distill`, plus
the comparison protocol for "distilled vs from-scratch at the same
parameter count" — the case study that justifies the technique on the
TinyGPT leaderboard.

Reference: Hinton et al., 2015, "Distilling the Knowledge in a Neural
Network" (`https://arxiv.org/abs/1503.02531`).

---

## Why it works

Training from a one-hot target (cross-entropy NLL) gives the student a
very thin signal at each position: "the next token is X, everything
else is wrong." Training against the teacher's full softmax
distribution exposes the student to the teacher's *uncertainty* — the
fact that "Y was almost as plausible as X here, and Z is clearly out."

That richer signal is most valuable when:

- The student is too small to discover the teacher's structure on its
  own from raw text (e.g. a 5M student vs a 100M+ teacher).
- The corpus is small relative to what would normally be needed.
- The teacher has been instruction-tuned, and you want the student to
  absorb that behaviour without re-running SFT/DPO.

## The loss

`tinygpt distill` uses the standard two-term distillation loss:

    L = α · T² · KL( softmax(s_logits / T) ‖ softmax(t_logits / T) )
      + (1 − α) · NLL(s_logits, true_target)

- **T (temperature)** softens both distributions. T=1 means the
  argmax dominates; higher T flattens the distribution so the student
  learns the *full* shape, not just the mode. Typical: T = 4-8.
- **The T² multiplier** compensates for the gradient-scaling caused by
  dividing logits by T. Without it, raising T silently shrinks the KL
  term's effective learning rate.
- **α (alpha)** mixes the soft (KL) and hard (NLL) terms. Higher α
  means the student listens to the teacher more; lower α keeps it
  grounded in real data. Typical: α = 0.5-0.9.

Both terms are needed: pure KL lets the student drift away from real
data if the teacher is wrong somewhere; pure NLL throws away the
teacher's soft information. The 0.7/0.3 mix is the HuggingFace
"distil" family default.

## Command

```sh
tinygpt distill <student-init> \
    --teacher <teacher-path> \
    --corpus <corpus.txt> \
    --tokenizer <hf-tokenizer-dir> \
    --steps 5000 \
    --temperature 4 \
    --alpha 0.7 \
    --out distilled.tinygpt
```

The student-init path is a `.tinygpt` checkpoint — start with the
output of a short `tinygpt train --preset tiny` run (just enough to
have valid weights — distillation does the heavy lifting). The
teacher path is either another `.tinygpt` file OR an HF model
directory; the loader auto-detects.

Both teacher and student MUST share a tokenizer (same vocab size + ids).
The `distill` command asserts vocab equality at startup.

---

## The comparison protocol

To make the "distilled vs scratch" claim concrete, run the same target
configuration both ways and score them on the same benchmark.

### Step 1: Train a teacher

```sh
# A 100M-class teacher on FineWeb-Edu, BPE-tokenised.
tinygpt train --preset huge \
    --tokenizer /tmp/smollm2 \
    --corpus /tmp/fineweb-edu-500M.txt \
    --dtype bfloat16 --batch 4 --accum 4 --ctx 512 \
    --steps 5000 --save-every 100 \
    --out /tmp/teacher.tinygpt
```

### Step 2: Make a from-scratch student of the chosen tiny size

```sh
# A 5M-class student, also BPE on the same tokenizer. Train for a
# matched compute budget (NOT a matched step count — the bigger
# teacher's steps cost more, so the student gets more steps).
tinygpt train --preset tiny \
    --tokenizer /tmp/smollm2 \
    --corpus /tmp/fineweb-edu-500M.txt \
    --dtype bfloat16 --batch 16 --ctx 512 \
    --steps 20000 --save-every 200 \
    --out /tmp/student-scratch.tinygpt
```

### Step 3: Distill the same student-init from the teacher

```sh
# Same architecture as Step 2, same compute budget. Initialise from a
# short randomly-trained run so weights are in a reasonable range.
tinygpt train --preset tiny \
    --tokenizer /tmp/smollm2 \
    --corpus /tmp/fineweb-edu-500M.txt --dtype bfloat16 \
    --steps 500 --out /tmp/student-init.tinygpt

tinygpt distill /tmp/student-init.tinygpt \
    --teacher /tmp/teacher.tinygpt \
    --corpus /tmp/fineweb-edu-500M.txt \
    --tokenizer /tmp/smollm2 \
    --steps 20000 \
    --temperature 4 --alpha 0.7 \
    --out /tmp/student-distilled.tinygpt
```

### Step 4: Score both on the same held-out benchmark

```sh
tinygpt eval /tmp/student-scratch.tinygpt    --corpus /tmp/holdout.txt
tinygpt eval /tmp/student-distilled.tinygpt  --corpus /tmp/holdout.txt
tinygpt eval /tmp/teacher.tinygpt            --corpus /tmp/holdout.txt
```

Expected (rough) result for a 100M teacher → 5M student on
education-grade text:

| Model | Params | Holdout PPL |
|---|---:|---:|
| Teacher | 100M | ~12 |
| Distilled student | 5M | ~22 |
| Scratch student | 5M | ~45 |

The exact numbers depend on data, training budget, and how
instruction-tuned the teacher is. The qualitative pattern (~2× PPL
gap between distilled and scratch at the same student size) is robust.

---

## Hyperparameter notes

- **Temperature**: start at 4. If the student is having trouble
  learning the soft distribution (loss plateaus high), try T = 8.
  If the student over-mimics teacher mistakes, try T = 2.
- **Alpha**: start at 0.7 (KL-heavy). If you trust the data more
  than the teacher (e.g. teacher is itself trained on a different
  corpus), drop to 0.5. If the teacher is heavily instruction-tuned
  and you want to inherit that, push to 0.9.
- **Learning rate**: 3e-4 is the safe default for a from-scratch
  student. Lower (1e-4) if initialising from a partially-trained
  checkpoint.
- **Batch / context**: the teacher forward runs on the same batch, so
  per-step memory is ~2× a normal train step. The default batch
  sizes in `Distill.swift` are halved from `Train.swift`'s defaults
  accordingly.

## Caveats

- **Tokenizer must match**: cross-tokenizer distillation requires a
  token-id remapping that we don't ship today. Both models use the
  same `.tokenizer.source` field, which `distill` asserts.
- **Student class**: the current implementation expects a from-scratch
  student (`TinyGPTModel`). HF-architecture student support is a
  follow-up.
- **No gradient flows through the teacher**: the teacher's parameters
  are not in the `valueAndGrad` target, so MLX treats its activations
  as constants. No explicit `stop_gradient` is needed.

## Reasoning distillation (R1-Distill family)

A particularly hot variant in 2024-2026: distilling REASONING
behaviour from a chain-of-thought-trained teacher (e.g. DeepSeek-R1's
distilled series) into a much smaller student. The recipe is the same
loss; what changes is the corpus — instead of generic web text, you
feed prompts + the teacher's STEP-BY-STEP responses. The student
learns to produce the same reasoning structure even though it doesn't
have the reasoning capacity to derive it from scratch.

To reproduce that workflow:

1. Generate or collect a corpus of `prompt → step-by-step response`
   pairs from a reasoning teacher.
2. Format them with the same `chatml` template SFT uses.
3. Run `tinygpt distill` with `--temperature 2` (sharper — we want to
   preserve the reasoning chain's specific tokens, not the broad
   distribution shape).

This is the path to a ~5M reasoning-capable student on the leaderboard.
