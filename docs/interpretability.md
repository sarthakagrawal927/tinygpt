# Interpretability tools — what is the model thinking?

Two interpretability surfaces ship in the browser playground:

1. **Attention heatmap** — for any prompt, show the per-head
   attention weights from the LAST transformer block. The "Watch the
   model think" panel that already exists.
2. **Logit lens** (Nostalgebraist 2020) — for any prompt, show what
   each layer "would predict" if its hidden state were projected
   straight through the final layernorm + LM head. A window into when
   specific knowledge first appears in the residual stream.

Both are WebGPU-only — they need access to intermediate tensors that
the WASM build doesn't currently expose.

---

## Attention heatmap

Already wired. After training or loading a gallery model, generate
once and the "Watch the model think" card appears. The heatmap shows,
for every token position in the prompt and every attention head in
the last block, which earlier tokens the head looked at.

Implementation: `gpu_model.ts:GpuModel.inspect`. One forward over the
prompt, save the last block's `[B, H, T, T]` attention matrix,
download to CPU, return.

## Logit lens

**New as of Phase 8.** A second button next to "Run benchmark" in the
Sample card: **Logit lens**. Click it, the worker runs forward over
the current prompt and returns one row per layer × one column per
input position; each cell is the top-1 byte the model "would output"
if it stopped at that depth.

What to look for:
- **Early layers** usually predict the most common byte (space or
  newline) regardless of context. The residual stream is still mostly
  positional + low-order ngrams.
- **Mid layers** start producing context-aware predictions (correct
  next byte for short n-grams).
- **Last layer** matches the actual next-byte prediction.
- **Where a prediction first becomes correct** is the depth at which
  the relevant feature crystallises. For ROMEO-style prompts that's
  often layer 4-6 on a 12-layer Huge model.

### Algorithm

For each layer L:

1. Run the standard forward through layers 0..L.
2. Apply the final layernorm (with its real `γ, β`) to the layer-L
   output.
3. Project through the tied LM head (`tokEmb.asLinear`).
4. Softmax over vocab → top-K per position.

The lens uses the FINAL layernorm's parameters even at intermediate
depths. That's the standard "logit lens" interpretation; alternative
"tuned lens" variants train layer-specific projection heads, which
this implementation does NOT ship.

### Cost

One extra layernorm + lm-head matmul per layer per inspection. For
the Huge config (12 layers, vocab=256, d=256), that's 12 extra ops
per lens run — a ~2× slowdown vs. a normal forward. Cheap enough to
run interactively for any prompt in the playground.

### Code map

- `webgpu/gpu_model.ts:GpuModel.logitLens` — the forward + per-layer
  head projection.
- `browser/src/worker.ts:doLens` — message handler that softmaxes
  and picks top-K per position.
- `browser/src/main.ts:renderLens` — the ASCII-table render of the
  result.
- `browser/src/types.ts:LensResult` — the result type definition.

## What's NOT yet shipped

- **Tuned lens.** The current lens reuses the final-layernorm
  parameters at every depth — a noisy approximation for mid layers.
  A tuned lens would train a small `Linear` per layer to better
  project that layer's residual stream into the LM head's space.
- **Per-layer ablation tool** (roadmap Phase 8, third item). The
  ability to zero out an attention head or MLP block and watch the
  prediction shift. Mechanism is clear (set the relevant tensor to
  zero between forwards) — the UI for it is the work.
- **Cross-layer attribution.** Tracing a specific output token's
  attribution back through layers (logit attribution / integrated
  gradients). Higher-leverage but substantially more complex.
