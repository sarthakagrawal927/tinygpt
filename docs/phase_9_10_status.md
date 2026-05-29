# Phase 9 + 10 — status and follow-up design

This doc closes out the remaining Phase 9 (quantization) and Phase 10
(architecture menu) items. For each: what's shipped today, and for the
items not yet shipped, what's needed to land them.

---

## Phase 9 — quantization

| Item | Status | Notes |
|---|---|---|
| DoRA | ✅ shipped | `--dora` flag on sft + dpo. Adapter file format extension is queued. |
| LASER selective rank reduction | ✅ shipped | `tinygpt laser` command. File-level SVD truncation. |
| QLoRA (int4 base + LoRA) | 📋 designed | See below. Blocker: MLX's quantized arrays don't yet fwd-prop gradients through to the underlying float matrices. |
| AWQ safetensors reader | 📋 designed | Adds a `--awq-quantized` path to `HFModelLoader`. Mechanical work. |
| HQQ (half-quadratic quantization) | 📋 designed | Implementing the convex-opt step in Swift is feasible; the inference-time win needs a Metal kernel that consumes the HQQ format. |

### QLoRA — what's needed

Concept: load the BASE model in int4 (e.g. via existing `--quantize int4`
or AWQ), then attach a normal LoRA on top. Training only updates the
LoRA — gradient flows through the int4 base as a constant.

Two pieces are missing:

1. **Gradient passes through quantized weights.** Today,
   `MLXNN.quantize(model:...)` swaps Linear for QuantizedLinear, which
   is purely an inference module — its weight isn't a regular
   `@ParameterInfo` MLXArray that autograd accepts. Until MLX-Swift
   either makes quantized weights gradient-transparent (treating them
   as no-grad constants in the trace) OR exposes a "frozen quantized
   constant" type that gradient can flow PAST, we can't run backward
   through a quantized base.

   Workaround idea: do the quantization MANUALLY in user code — keep
   the base as a regular fp32/bf16 `Linear` whose `weight` is held
   constant via `freeze()`, but apply a fake-quant function in the
   forward (cast → round → cast back). Loses the memory win but
   preserves the gradient flow. Useful pedagogically; not the real
   QLoRA story.

2. **Persistent quantized base loading.** If we want QLoRA on an
   AWQ-quantized HF model, the AWQ reader below is the prerequisite.

### AWQ reader

AWQ (Lin et al., 2023) safetensors files store weights as
`qweight` (int32-packed 4-bit), `qzeros`, and `scales` per output
channel. Reading is mechanical:

```swift
// inside HFModelLoader.makeMLXArray when dtype == "I32" and name
// ends in ".qweight", and a sibling "scales" + "qzeros" exist:
let unpacked = unpackAwqInt4(qweight, scales, qzeros)
return MLXArray(unpacked, originalShape)
```

The conversion produces a dense fp16/fp32 representation that the
existing forward path can use unchanged. The pure-AWQ runtime
(matmul against packed int4 directly) would need a kernel.

### HQQ

HQQ (Badri & Shaji, 2023) uses convex optimization to find better
quantization scales than the naive min-max approach. The algorithm:

1. Group weights into blocks of size G (e.g. 64).
2. For each block, solve a small convex problem:
   minimise `‖W - dequant(quantize(W; scale, zero))‖₂` over (scale, zero).
3. Store (quantized weights, scale, zero) per block.

The optimisation is fast (closed-form per block). The inference-time
win requires a Metal kernel that does grouped int4 matmul against
the block layout — same kernel-engineering bar as the sparse MoE
dispatch. The quantization step itself is Swift-side and feasible.

---

## Phase 10 — architecture menu

| Item | Status | Notes |
|---|---|---|
| Sliding window attention | ✅ shipped | `--sliding-window N` flag, persisted in header. |
| ALiBi position bias | ✅ shipped | `--alibi` flag, per-head geometric slopes. |
| Differential attention | 📋 designed | Two SDPAs subtracted; needs 4 attention projections per head. |
| YOCO cross-layer KV sharing | 📋 designed | First half of layers compute KV; second half cross-attends. Halves the KV cache. |
| Mixture of Depths | 📋 designed | Per-token, per-layer router skips uninformative layers. |

### Differential attention (Ye et al., 2024)

Each attention head computes TWO independent softmax attention maps
and subtracts them, weighted by a learnable scalar λ:

```
A = softmax(Q1 K1ᵀ / √d) − λ · softmax(Q2 K2ᵀ / √d)
out = A · V
```

The subtraction cancels attention "noise" across the two heads,
typically improving long-context reasoning and reducing hallucinations.

**Shipping cost**:
- Per-block: 4 attention projections instead of 2 (Q1, K1, Q2, K2, V, O).
- Per-head: 2× λ scalars (`λ_q`, `λ_k`) with the paper's reparam
  `λ = λ_init − exp(λ_q · λ_k)`.
- Manifest entries doubled for attention.
- Per-step compute roughly 1.5× the standard attention path.

The wire-up is bounded — biggest churn is the manifest expansion
across the .tinygpt format. Single-file change to TransformerBlock +
manifest entries + checkpoint loaders.

### YOCO — "You Only Cache Once"

Lin et al., 2024. The model is split in two halves. The first half
computes K, V normally. The second half does CROSS-ATTENTION onto the
last K, V produced by the first half — no new K, V are computed for
those layers. KV cache memory drops by ~2× at long context.

**Shipping cost**:
- A new block type for the "cross-attention" layers (no K, V proj, no
  KV save).
- Caching glue between the two halves at decode time.
- Manifest schema change to encode "this layer uses YOCO cross-attn".

Single-file changes; estimated ~150-200 lines including tests.

### Mixture of Depths (Raposo et al., 2024)

A small router per layer decides which top-K tokens get processed
through that layer; the rest skip via residual. Compute scales to
fewer tokens per layer at the same expressivity.

**Shipping cost**:
- Per layer: a 1-output Linear router.
- Per-step: a topK-by-router-prob to pick the active subset.
- Same scatter/gather pattern as sparse MoE — **blocked on the same
  MLX-Swift `scatter_add` gap** documented in `docs/moe.md`. A dense-
  with-masking implementation works (multiply by router-mask, residual
  takes the no-op path) but gives no compute saving.

The architecture's training story (router + load-balance loss) is
identical to MoE's; the saving requires real sparse dispatch.

---

## Phase 8 — interpretability remainder

| Item | Status | Notes |
|---|---|---|
| Logit lens | ✅ shipped | Button in browser playground. |
| Attention heatmap | ✅ shipped | Existing "Watch the model think" panel. |
| Per-layer ablation | ✅ shipped | New "Ablate & sample" button. |
| Activation patching | 📋 designed | Donor-recipient swap; see below. |
| Tuned lens | 📋 designed | Trained linear probe per layer; see below. |

### Activation patching (Meng et al., 2022)

Run forward TWICE:

1. **Donor run** on prompt A — save the hidden state at (layer L,
   position P) — call it `h_donor`.
2. **Recipient run** on prompt B — at (layer L, position P),
   REPLACE the hidden state with `h_donor`, then continue forward
   normally.

The recipient's output then reveals "what would the model say if the
representation at this position had been the donor's?". A causal
intervention that pinpoints WHERE a piece of information lives in
the residual stream.

**Shipping cost**: extends the existing ablation mechanism with a
"donor cache" (per-layer, per-position MLXArray). Forward checks if a
position has a saved donor activation and uses it instead. The UI
needs two prompt boxes + a layer/position picker — moderate work.

### Tuned lens

Belrose et al., 2023. Train a small `Linear(d_model → d_model)` per
layer that projects the residual stream into a "better lens" space
before the final LN + LM head. Lower noise than the standard logit
lens because it's calibrated per layer rather than reusing the
final-LN parameters.

**Shipping cost**:
- Add `tunedLensHeads: [Linear]?` to TinyGPTModel.
- A training procedure that freezes the base model and trains only
  the per-layer projections via cross-entropy on each layer's
  projected logits.
- Use the trained heads in `logitLens` instead of the raw final-LN
  reuse.

The training is a side-pass over the same data — cheap relative to
main training. The architectural addition is one Linear per layer.

---

## Cross-cutting blockers

These items appear across multiple phases and share a root cause:

1. **MLX-Swift doesn't expose `mlx_checkpoint`** — blocks gradient
   checkpointing (Phase 6). The C primitive exists; the Swift
   wrapper doesn't. Workarounds in `docs/memory_tradeoffs.md`.
2. **MLX-Swift doesn't expose `scatter_add`** — blocks sparse MoE
   compute and MoD compute savings (Phase 5, Phase 10). Workarounds
   in `docs/moe.md` and above.
3. **Cmlx is internal to MLX-Swift** — neither of the above
   primitives can be bridged from outside the package without
   forking MLX-Swift. The right resolution is upstream PRs.

These are real engineering tasks, not session-sized work. Each
unblocks several roadmap items simultaneously — landing them is
the highest-leverage move for the next phase of work.
