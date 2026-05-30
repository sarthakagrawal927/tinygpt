# Pruning — magnitude masks and structured surgery

Pruning is the post-training technique where you remove parts of a
trained model and accept a small quality hit in exchange for a smaller
model. There are two flavours here, with very different payoff
profiles:

- **Unstructured** (`tinygpt prune-unstructured`) — zero out
  individual weights below a magnitude threshold. The model shape is
  unchanged; matmuls still operate on the original `[out, in]` slabs.
  On Metal this means **no wallclock speedup at inference** (there's
  no sparse-matmul kernel). The win shows up at **distribution time**
  — once you gzip the file, the long runs of zeros collapse.
- **Structured** (`tinygpt prune-structured`) — remove whole attention
  heads or whole transformer layers. Layer pruning is genuinely
  topology-changing: the output model has fewer blocks, fewer
  parameters, fewer FLOPs, **real wallclock + memory win**. Head
  pruning is shape-preserving (zero-out style, Michel et al. 2019)
  because the existing `CausalSelfAttention` module assumes
  `dModel = nHeads × headDim` end-to-end — see "Caveats" below for
  the asymmetric-attention work needed to make head pruning
  physically reduce the projection matrices.

References:
- Han et al., 2015, "Learning both Weights and Connections for Efficient Neural Networks"
- Frankle & Carbin, 2019, "The Lottery Ticket Hypothesis"
- Michel et al., 2019, "Are Sixteen Heads Really Better than One?"
- Gromov et al., 2024, "The Unreasonable Ineffectiveness of the Deeper Layers"

---

## Unstructured pruning (magnitude)

The simplest recipe: for every Linear weight tensor, find the
`sparsity` quantile of absolute values, zero everything below it,
keep everything above.

```bash
tinygpt prune-unstructured browser/public/gallery/shakespeare.bin \
    --sparsity 0.5 --out shakespeare-p50.tinygpt
```

The output `.tinygpt` carries:
- The same shapes as the input.
- A `sparsityMasks` JSON object in the header — one bit-packed (or
  RLE if shorter) mask per pruned tensor. Decoders that don't know
  about the field silently ignore it.
- A `pruningInfo` record naming the recipe (`unstructured`,
  `sparsity: 0.5`, `iterations: 1`).

### File size — measured on the gallery Shakespeare model

The gallery distribution layout is `fp16, no optimizer state` — one
contiguous fp16 weight buffer indexed by `floatOffset`. Pruning
doesn't change that buffer's size: zeros take the same two bytes as
non-zeros.

| variant                       | raw `.bin` | gzipped |
| ----------------------------- | ---------- | ------- |
| shakespeare.bin (baseline)    | 19.25 MB   | 17.78 MB |
| 50% pruned, mask in header    | 20.87 MB   | 12.29 MB |
| 50% pruned, `--no-mask`       | 19.25 MB   | 11.08 MB |

Observations:
- **The raw bytes don't shrink.** A zero fp16 is still two bytes. The
  mask costs about 8% of file size on a 50%-pruned model (one bit per
  weight, plus a per-tensor base64 wrapper).
- **gzip cuts the file by ~30-40%.** That's the actual distribution
  benefit. The pruned model gzip-compresses much better than the
  baseline because long runs of zeros are perfectly predictable.
- **`--no-mask` is the best size if you don't need the mask later.**
  At inference time the mask is informational — the weights are
  already zeroed, so the forward pass produces the same logits with
  or without it. Keep the mask only if you intend to re-apply it
  after further fine-tuning (where the zeroed weights would otherwise
  drift back to non-zero).

### Iterative Magnitude Pruning (IMP)

```bash
tinygpt prune-unstructured shakespeare.bin \
    --sparsity 0.3 --iterations 3 \
    --corpus shakespeare.txt --ft-steps 100 \
    --out shakespeare-p65.tinygpt
```

Each iteration prunes `sparsity` of the still-non-zero weights, then
fine-tunes for `--ft-steps` steps with the mask kept fixed (zeroed
weights stay zero; the optimizer can't undo a prune). After N rounds
the cumulative sparsity is `1 − (1 − sparsity)^N`:

| iterations | sparsity per round | total |
| ---------- | ------------------ | ----- |
| 1          | 0.5                | 50.0% |
| 2          | 0.3                | 51.0% |
| 3          | 0.3                | 65.7% |
| 4          | 0.25               | 68.4% |

IMP usually recovers most of the loss-delta of one-shot pruning at
the same final sparsity, at the cost of running fine-tuning between
rounds. The recipe is from Frankle & Carbin (2019) and is the
standard for "high-sparsity but still good" pruning.

**Smoke result** (Shakespeare, 51% total sparsity over 2 rounds with
20 fine-tune steps per round on the original training corpus):
loss 1.27 → 1.65 vs the dense baseline. Most of that delta is the
loss of model capacity — 100 fine-tune steps is too few to recover;
500-1000 steps per round is the production recipe.

### What gets pruned

By default, every 2-D Linear weight tensor (`q/k/v/o_proj`,
`fc_in/fc_out`). Embeddings are skipped because pruning them silently
kills rare tokens. Pass `--include-embeddings` to override.

LayerNorm gains, biases, and 1-D tensors are never pruned —
pruning them doesn't help (they're tiny) and breaks the model
much faster than weight-magnitude pruning the same fraction.

### Why no wallclock speedup

Metal has no built-in sparse-matmul kernel. Even at 90% sparsity, the
projection matmuls have to walk every row × column pair to multiply
zero by activation. The forward path is identical to the dense
model's. This is an honest limitation of the platform — on CUDA you
could plug in a 2:4 structured-sparse matmul (Ampere+) for a real
~1.6x speedup at sparsity 50%, but that's not portable.

---

## Structured: head pruning

```bash
tinygpt prune-structured shakespeare.bin \
    --heads-to-drop 4 --out shakespeare-h4.tinygpt
```

Drops the K weakest heads PER LAYER (Michel et al. 2019's standard
convention). With `nHeads=8 --heads-to-drop=4`, every transformer
block loses its 4 lowest-importance heads.

**Head importance** = Frobenius norm of each head's Q/K/V/O slabs:

    score(h) = ‖Q_h‖_F + ‖K_h‖_F + ‖V_h‖_F + ‖O_h‖_F

A head whose projections have collapsed to (near-)zero is
contributing little to the residual stream — safe to drop. This is
weaker than gradient-based saliency (`‖∂L/∂h · h‖`) but doesn't
require a forward+backward pass over a calibration corpus.

The CLI walks every layer, computes `nHeads` scores, picks the
bottom K, and zeros the corresponding rows/columns of Q, K, V (and
the columns of O). For GQA models (`nKvHeads < nHeads`), a KV head
is only zeroed when ALL grouped query heads have been dropped.

**Smoke result** on Shakespeare (huge preset, 12L × 8H):

| flag                       | loss  | perplexity | sample quality |
| -------------------------- | ----- | ---------- | -------------- |
| baseline                   | 1.27  | 3.56       | grammar emerges |
| `--heads-to-drop 1`        | 1.33  | 3.77       | grammar emerges |
| `--heads-to-drop 4`        | 2.81  | 16.6       | English-like but garbled |

Dropping 4 of 8 heads PER LAYER (= 50% of total head capacity) is
significant. The model still produces English-shaped text, but
grammar is broken. Fine-tuning after the drop typically recovers
most of the quality (Michel et al. report this in their paper); the
CLI doesn't ship that yet — see "Future work".

### Caveats (head pruning)

**Shape-preserving.** Dropping heads does NOT shrink the projection
matrices. `q_proj`, `k_proj`, `v_proj`, `o_proj` are still
`[dModel, dModel]`. The dropped heads' rows/columns are filled with
zeros but the matmul still runs over them.

Why? `CausalSelfAttention.callAsFunction` reshapes each projection
to `[B, T, nHeads, headDim]` with `headDim = dModel / nHeads` and
`nHeads * headDim == dModel`. To physically remove a head's columns,
we'd need an asymmetric attention module where the
projections' output dim is independent of the residual-stream dim:

```swift
public final class AsymmetricCausalSelfAttention: Module {
    let attnInnerDim: Int   // = (nHeads_after_pruning) * headDim
    // q,k,v: dModel → attnInnerDim
    // o:     attnInnerDim → dModel
}
```

This is one self-contained module; the rest of the model stays the
same (residual stream is still dModel everywhere). The
implementation is ~200 LOC and is the natural next step. Until then,
head pruning gives memory savings only (the zeroed columns
gzip-compress) and no wallclock savings.

---

## Structured: layer pruning

```bash
tinygpt prune-structured shakespeare.bin \
    --layers-to-drop 2 --calibration shakespeare.txt \
    --out shakespeare-l2.tinygpt
```

Physically removes M transformer blocks. The output has
`nLayers - M` layers and is genuinely smaller (fewer parameters,
fewer FLOPs).

**Layer importance = block angular distance** (Gromov et al., 2024):
for each layer L, the cosine angle between the residual stream
entering L and exiting L. A layer whose output is nearly identical to
its input (angle ≈ 0) is contributing little — safe to drop.

The CLI runs a small calibration forward (`--calib-batches 4
--calib-batch 4` by default), captures per-layer hidden states, and
scores each layer by `angularDistance(prev, current)`. The K lowest
scores are dropped; the surviving blocks are re-numbered contiguously
in the output manifest. `header.config.layers` is updated so the
loader builds the smaller model.

**Smoke result** on Shakespeare:

```
layer importance (angular distance — lower = drop me):
  block  0  0.4299
  block  1  0.1531
  block  2  0.0486 ← DROP
  block  3  0.0363 ← DROP
  block  4  0.0542
  block  5  0.0693
  block  6  0.1172
  block  7  0.1295
  block  8  0.1259
  block  9  0.1514
  block 10  0.1297
  block 11  0.1683
```

Block 0 has the highest distance (embedding-to-hidden conversion is
the biggest jump). Blocks 2 and 3 do almost nothing on this small
model — they get dropped.

| flag                  | params | loss | perplexity |
| --------------------- | ------ | ---- | ---------- |
| baseline              | 9.6 M  | 1.27 | 3.56       |
| `--layers-to-drop 2`  | 8.0 M  | 2.01 | 7.47       |

The 16% parameter reduction comes with a substantial loss delta. A
real production workflow would fine-tune the result for a few
thousand steps to recover; "drop layers + fine-tune" is the
distillation-free path to a smaller model.

### Calibration text

The `--calibration` argument is a UTF-8 text file used to compute the
hidden-state vectors at each block boundary. It should be
representative of the deployment distribution (a held-out split of
the training data is fine). Without `--calibration`, the CLI falls
back to dropping the mid-most layers — Gromov et al. observed that
mid-layers are usually the safest drops, but the heuristic is
weaker than calibration-based scoring.

A few KB of calibration text is enough — we average per-position
angular distance across batches and only need to differentiate
similar-magnitude layers from each other.

---

## File-level metadata

Both subcommands write a `pruningInfo` block into the header for
inspectability. Example after a 50%-sparsity prune:

```json
{
  "config": { ... unchanged ... },
  "manifest": [ ... unchanged shapes ... ],
  "pruningInfo": {
    "kind": "unstructured",
    "sparsity": 0.5002,
    "iterations": 1
  },
  "sparsityMasks": {
    "blocks.0.attn.q_proj.weight": "<base64 of bit-packed mask>",
    "blocks.0.mlp.fc_in.weight":   "<base64 of bit-packed mask>",
    ...
  }
}
```

`tinygpt inspect` doesn't surface these yet (this surface is new);
the JSON is human-readable if you pull the header out yourself.

---

## Caveats and known limitations

1. **No Metal sparse matmul.** Unstructured pruning gives ZERO
   wallclock speedup at inference. The win is distribution size
   (post-gzip) and as a regulariser if you fine-tune after pruning.

2. **Head pruning is shape-preserving.** The CLI ships the
   Michel-zero-out variant. Physical head removal needs an
   asymmetric-attention module; see "Caveats (head pruning)" above
   for the design sketch. Memory savings only (via gzip), no
   wallclock savings.

3. **Layer pruning IS topology-changing — and the win is real.**
   Output has fewer blocks, runs that much faster on the GPU. Quality
   loss is bigger than head pruning (each layer is a chunkier unit of
   capacity), but a few thousand fine-tune steps usually recover most
   of it.

4. **IMP only works on from-scratch (.tinygpt) models, not HF model
   directories.** The fine-tune step uses `TinyGPTModel` directly;
   plugging into `TinyGPTModelHF` would require a parallel mask-
   application path on the HF parameter tree (suffix names differ:
   `embed_tokens.weight` vs `token_embedding.weight`, etc.). The one-
   shot path works on any `.tinygpt`; HF dirs would need an HF-format
   writer first.

5. **Adam optimizer state is dropped.** Both pruning paths write
   their output via the standard `TinyGPTFile` writer, which sets the
   Adam moments to zeros. Resuming training from a pruned checkpoint
   incurs the usual ~100-step optimizer warm-up.

6. **MoE / differential attention / YOCO blocks are NOT specially
   handled.** The unstructured path picks up any 2-D Linear weight
   it finds, which catches the MoE expert MLPs and diff-attn
   projections correctly but might not be what you want for
   exotic architectures. The structured path assumes the standard
   `attn.q_proj / k_proj / v_proj / o_proj` names — MoE/diff-attn/
   YOCO models work for layer pruning (layer indices are
   architecture-agnostic) but head pruning only touches the standard
   attention path.

---

## Future work

- **Asymmetric attention module** so head pruning actually shrinks
  the projection matrices (real wallclock win, ~30 minutes of work
  plus testing).
- **Fine-tune after head/layer pruning** with the standard
  `tinygpt finetune` plumbing. Probably 200-500 LOC to wire through.
- **Block-sparse 2:4 or 4:8 patterns** so the sparsity actually
  accelerates on a future Metal sparse-matmul kernel. Not portable
  yet.
- **HF model support** for IMP. Mirror image of LoraInjection vs
  LoraInjectionHF — parallel paths for the two model variants.
- **Surface `pruningInfo` in `tinygpt inspect`.** Currently you have
  to look at the raw JSON.
