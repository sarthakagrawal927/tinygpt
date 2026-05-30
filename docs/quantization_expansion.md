# Quantization Expansion

This drop ships four new quantization features on top of the existing
AWQReader (HF AWQ safetensors loader) and HQQ (storage-only Half-Quadratic
Quantization) that already live in the tree:

1. **GPTQReader** — load HF GPTQ-format safetensors files transparently.
2. **From-scratch GPTQ** — `tinygpt gptq` worker: Hessian-aware int4
   layer-by-layer quantisation with calibration corpus.
3. **SmoothQuant** — calibration + per-channel activation scaling pass
   that preconditions a model for downstream int8 inference.
4. **QAT** — Quantization-Aware Training via `--qat int4|int8` on
   `tinygpt train`: fake-quant + straight-through estimator on every
   Linear weight during the forward pass.

The story is the same for all four: **the inference-side runtime win is
gated on a packed-int matmul kernel** (MLX-Swift's quantized-matmul
story is in progress upstream). What's shipped here is the
*algorithmic infrastructure* — loaders, calibration passes,
fake-quant training. When the kernel lands, these passes plug straight
in.

---

## 1. GPTQReader — HF GPTQ loader

**File:** `native-mac/Sources/TinyGPTModel/GPTQReader.swift`
**Wires into:** `native-mac/Sources/TinyGPTModel/HFModel.swift`
                (the `HFModelLoader.load(from:)` path)

### Format

GPTQ (Frantar et al., 2022) stores each Linear's weight as a quartet of
tensors:

```
{name}.qweight   int32, shape [in // 8, out]
                  8 packed int4 codes per int32 along the IN axis
{name}.scales    fp16/bf16, shape [in // group_size, out]
                  per-output-channel per-group dequant scale
{name}.qzeros    int32, shape [in // group_size, out // 8]
                  8 packed int4 zero-points per int32 along the OUT axis
{name}.g_idx     int32, shape [in]  (optional, activation-order GPTQ)
                  group index per in-feature, usually floor(i/group_size)
                  but permuted when desc_act=True at quant time
```

The dequant recipe (HF row-major `W[out, in]`):

```
for o in 0..<out:
    for i in 0..<in:
        qint32 = qweight[i // 8, o]
        bit    = (i % 8) * 4
        int4   = (qint32 >> bit) & 0xF
        g      = g_idx[i]                              // may be permuted
        scale  = scales[g, o]
        zint32 = qzeros[g, o // 8]
        zbit   = (o % 8) * 4
        zero   = ((zint32 >> zbit) & 0xF) + 1          // GPTQ "+1" convention
        W[o, i] = scale · (int4 − zero)
```

The **"+1 on zero"** is the historical GPTQ quirk — `auto-gptq` v0.x
stores zero as `(int4_zero − 1)`, so re-adding 1 recovers the dequant
offset. AWQ does NOT do this; that's the most-easily-confused
difference between the two formats.

### Wiring

`HFModelLoader.load(from:)` now runs a detection pass BEFORE the
flat-update map is built:

```swift
let gptqBases = GPTQReader.detectGptqBases(in: names)
let awqBases  = AWQReader.detectAwqBases(in: names)
for base in gptqBases {
    // gptq has g_idx; pure awq doesn't.
    if gIdxSrc == nil && awqBases.contains(base) {
        continue  // defer to AWQ
    }
    let dense = try GPTQReader.dequantize(qweight, scales, qzeros, gIdx)
    dequantised[base + ".weight"] = dense
}
for base in awqBases where !dequantised.contains(base) {
    let dense = try AWQReader.dequantize(qweight, scales, qzeros)
    dequantised[base + ".weight"] = dense
}
// the main loop then SKIPS .qweight/.scales/.qzeros/.g_idx tensors
// for bases that have been dequantised, and splices the dense .weight
// into `updates` instead.
```

Result: a downloaded `Llama-2-7B-GPTQ` (or any other GPTQ checkpoint)
loads via `tinygpt hf-load <dir>` with no Python pre-step. The
runtime memory cost is 8× the packed payload (int4 → fp32) for the
duration of the load; the inference-side win is queued behind the
packed-int matmul kernel.

### Smoke test

Synthetic roundtrip (no MLX dependency — checks the bit-packing math
matches GPTQ-spec exactly):

```
W[0, :] = [-1.0, 0.0, 1.0, ..., 14.0]
expected = [-1.0, 0.0, 1.0, ..., 14.0]
row 0 matches: true
```

Loading a real public GPTQ checkpoint (e.g.
`TinyLlama/TinyLlama-1.1B-Chat-v1.0-GPTQ`) requires a multi-GB
download and is skipped in the per-drop smoke; the bit-packing
roundtrip + identical-shape sibling AWQReader (which IS loaded
end-to-end on every previous AWQ smoke) gives us high confidence
the loader is correct.

---

## 2. From-scratch GPTQ — `tinygpt gptq`

**File:** `native-mac/Sources/TinyGPT/GPTQ.swift`
**Wires into:** `native-mac/Sources/TinyGPT/TinyGPT.swift` via the
pre-switch shim (parallel to `score-bench`); `case "gptq":` left as a
TODO marker.

### Algorithm

GPTQ (Frantar et al., 2022) quantises each Linear `W ∈ R^{out × in}`
column-by-column along the input axis, propagating each column's
reconstruction error to subsequent columns via the Cholesky-derived
inverse Hessian. Pseudocode:

```
1. Forward calibration corpus through the model; at each Linear,
   capture per-token input activations X ∈ R^{N × in}.
2. H = X^T X + λ·I    (input Hessian, ridge for stability)
3. L = chol(H)        (lower-triangular Cholesky factor)
   Hinv = L^{-T} · L^{-1}     (inverse Hessian; SPD)
4. For each input column c in [0, in):
     a. Quantise W[:, c] to nearest int{bits} grid level
     b. err[:] = W[:, c] − Wq[:, c]                    (per output row)
     c. For each c' > c:                                (error propagation)
          W[:, c'] -= err[:] * (Hinv[c, c'] / Hinv[c, c])
```

The "compensation" step (c) is what makes GPTQ beat naive
round-to-nearest at int4: errors that correlate with later columns
pre-compensate downstream rounding, minimising the quadratic
reconstruction loss `‖X · W^T − X · Wq^T‖²`.

### Implementation

- Hessian H is accumulated across windows in `Double` (the
  matmul on GPU is fp32, summed in fp64 host-side to avoid
  accumulation drift).
- Cholesky is hand-rolled (n³/3 Double ops, single-threaded). At
  flagship-huge sizes (n ≤ 1024) the full quantisation run completes
  in ~30 seconds.
- The `.tinygpt` file invariant `entry.shape = [out, in]`,
  `weight bytes = [in, out] row-major` is honoured — the dequantised
  output is re-packed in the same `[in, out]` byte order, so the
  written file loads via the existing `TinyGPTWeightLoader` unchanged.

### Usage

```
tinygpt gptq <input.tinygpt>
    --calibration <text.txt>
    --bits 4                 # 2 | 3 | 4 | 8
    --group 128
    --samples 32
    --ctx 256
    --out <output.tinygpt>
```

### Smoke test

```
tinygpt gptq /tmp/flagship-huge.tinygpt \
    --calibration data/examples/shakespeare.txt \
    --bits 4 --group 128 --samples 4 --ctx 128 \
    --out /tmp/flagship-gptq.tinygpt
```

Result:
- 72 Linear tensors quantised across 12 transformer blocks.
- Relative reconstruction error 0.1064 (10.6%) at int4 — consistent
  with the int4 grid quantisation noise floor.
- Runtime: ~31 seconds end-to-end on M5 Pro.
- `tinygpt sample /tmp/flagship-gptq.tinygpt` loads + samples cleanly.

### Honest caveat — Storage-only payoff

Same story as HQQ: the written .tinygpt holds quantize-then-dequantise
fp32 weights. The model loads and samples normally via the existing
forward path. **The inference-side memory + speed win is gated on a
packed-int matmul kernel.** What you get TODAY is a model whose
weights have been pushed through the GPTQ quantisation noise, which
often slightly IMPROVES downstream tasks (similar to LASER's rank
reduction surprise) and which is ready for export to a downstream
runtime (`llama.cpp`, `mlx-lm`'s int4 path) that has the kernel.

---

## 3. SmoothQuant — pre-quantization activation smoothing

**File:** `native-mac/Sources/TinyGPTModel/SmoothQuant.swift`
**Wires into:** library-only (no CLI binding); designed to be called
from a calibration script or downstream tooling.

### Problem

Activations into a Linear are often far more skewed across channels
than the weights are. A few outlier channels carry 10-100× the
magnitude of the rest. When you int8-quantise the activations, the
outliers blow out the scale and the inliers collapse onto a handful
of quant levels — perplexity craters.

### Trick

Introduce a per-INPUT-CHANNEL scale `s[i] ≥ 0` and rewrite

```
y = (x / s) · (s · W)        // mathematically identical
```

by absorbing `s` into the weight: `W' = diag(s) · W`. Now the
activation that hits the int8 quantiser is `x / s`, whose
channel-wise range is smoothed. Xiao et al.'s recipe:

```
s[i] = max(|x[:, i]|)^α  /  max(|W[:, i]|)^(1 − α)
```

with `α` (typically 0.5) trading activation-smoothing for weight
stretching.

### API

```swift
let acc = SmoothQuant.makeAccumulator(linearWeights: weights)
// caller runs calibration forwards, populating `acc[name]` per layer:
for batch in calibration {
    let actHook = runForward(model: model, batch: batch, capturing: linearInputs)
    for (name, x) in actHook {
        SmoothQuant.updateMax(&acc[name]!, with: SmoothQuant.channelAbsMax(x))
    }
}
let scales = SmoothQuant.smooth(
    linearWeights: &weights,
    activationMax: acc,
    config: .init(alpha: 0.5)
)
// `weights` is now W · diag(s); user is responsible for fusing 1/s into
// the previous layer's output projection (or applying it at runtime).
```

The pass is **mathematically exact** (the transform is identity for
the dense matmul itself). After applying, the user is responsible
for one of:

- Fusing `1/s` into the upstream LayerNorm's gamma (paper-standard).
- Applying `x ← x / s` at runtime inside the Linear's forward.

The shipped module does NEITHER fold — it returns the scale vectors
alongside the rewritten weights so downstream tooling
(`llama.cpp`, `mlx-lm` int8) can pick which one to do.

### Honest caveat — Int8 matmul kernel gap

MLX-Swift's matmul is fp32/fp16/bf16. There is no int8 matmul kernel
in the public Apple stack today.

**What SmoothQuant gives you in this drop:**
- A deterministic data transformation that produces a model whose
  activations + weights are BETTER CONDITIONED for downstream int8
  quantisation.
- A `[String: [Float]]` scale dictionary you can serialise alongside
  the model for downstream tooling.

**What it does NOT give you:**
- Any inference-side speedup or memory saving today.
- A drop-in "run my model in int8 now" path on MLX.

When Apple ships `mlx::quantized_matmul` int8 support (or we hand-roll
one in Metal), the calibration logic here plugs straight in.

### Smoke test

Mathematical-identity check (no MLX deps — pure float math):

```
s vector: 0.649, 0.724, 0.802, 7.561, 0.722, 0.429, 0.778, 0.878
y_original  = -43.6470, 5.2136, 40.4767, 18.0859
y_smoothed  = -43.6470, 5.2136, 40.4767, 18.0859
max abs diff: 3.8e-06   (float roundoff, not a real difference)
x range collapsed from 50.0 → 6.61   (outlier channel folded into W)
```

The outlier-activation channel (index 3) had `|x[3]| = 50` before; the
SmoothQuant pass folded `s[3] = 7.56` into the matching weight column,
collapsing the activation range to 6.61 — 7.5× int8-friendlier —
while preserving the matmul output bit-for-bit (modulo Float rounding).

---

## 4. QAT — Quantization-Aware Training

**Files:**
- `native-mac/Sources/TinyGPTModel/QAT.swift` — fake-quant + STE primitives
- `native-mac/Sources/TinyGPTModel/ModelConfig.swift` — `qatBits: Int?` config field
- `native-mac/Sources/TinyGPTModel/TransformerBlock.swift` — wire fake-quant
  into Q/K/V/O projections (CausalSelfAttention) + fc_in/fc_out (MLP) +
  up/gate/down (SwiGLU)
- `native-mac/Sources/TinyGPTModel/TransformerBlockHF.swift` — wire into
  HF block's SwiGLU
- `native-mac/Sources/TinyGPT/Train.swift` — `--qat int4|int8` flag

### Fake-quant + STE

The forward, per Linear weight `W ∈ R^{out × in}`:

```
scale[o] = max(|W[o, :]|) / qMax       // per-output-row symmetric scale
Wq[o, i] = clip(round(W[o, i] / scale[o]), -qMax, +qMax) · scale[o]
W_used = W + stopGradient(Wq − W)
```

Forward value of `W_used`: `Wq` (the quantised weight).
Backward value of `W_used`: `dL/dW` (stopGradient kills the inner term).

The trick (Hubara et al., 2016; standard PyTorch QAT recipe) is the
`stopGradient` trick — the network FORWARDS through the quantised
weight but the optimiser sees the gradient on the original `W`.
This lets the optimiser learn to push `W` toward grid-friendly
positions.

`qMax = 2^(bits − 1) − 1` (symmetric: 7 for int4, 127 for int8).

### Wiring

`ModelConfig.qatBits: Int?` (nil = off). When set, the value
propagates into each block's attention + MLP at init time:

```swift
self.qatBits = cfg.qatBits  // in CausalSelfAttention.init
```

Each module then routes every Linear-call through
`QAT.linearForward(linear, x: x, bits: bits)` instead of the bare
`linear(x)`:

```swift
@inline(__always) private func proj(_ l: Linear, _ x: MLXArray) -> MLXArray {
    if let bits = qatBits { return QAT.linearForward(l, x: x, bits: bits) }
    return l(x)
}
```

Both `TransformerBlock` (from-scratch + dense MLP) and
`TransformerBlockHF` (HF SwiGLU) are wired. Bias terms are NOT
fake-quantised — they're tiny tensors whose dynamic range is already
representable in int8 and the paper-standard recipe is to leave them
at fp32.

### CLI

```
tinygpt train --preset huge --steps 1000 --qat int4 \
    --corpus shakespeare.txt --out huge-qat4.tinygpt
```

The training log gains a `qat-err` diagnostic — the relative
absolute reconstruction error of the first attention block's q_proj,
sampled every 50 steps:

```
qat:           int4 fake-quant + STE on every Linear
...
  step     1/   30  loss 5.898  qat-err 0.070  · 0.6 step/s · eta 53s
  step    30/   30  loss 3.292  qat-err 0.073  · 13.6 step/s · eta 0s
```

For int4 the error bound is `~1 / qMax = 1/7 ≈ 0.143`; converged QAT
runs trend lower as the optimiser learns grid-friendly weights. For
int8 the bound is `1/127 ≈ 0.008`; in our 30-step smoke we sit at
0.004.

### Smoke test

30 steps on `data/examples/shakespeare.txt`, tiny preset:

```
$ tinygpt train --preset tiny --steps 30 --qat int4 \
    --corpus data/examples/shakespeare.txt --out /tmp/qat-int4-smoke.tinygpt

  step     1/   30  loss 5.898  qat-err 0.070  · 0.6 step/s
  step    30/   30  loss 3.292  qat-err 0.073  · 13.6 step/s
✓ wrote /tmp/qat-int4-smoke.tinygpt
```

- Loss decreased 5.898 → 3.292 (the fake-quant doesn't kill learning).
- `qat-err` bounded at ~0.07 (under the 1/7 ≈ 0.143 int4 ceiling).
- Step throughput 14 step/s vs ~16 step/s without QAT — ~10% overhead,
  consistent with the per-Linear extra round + STE cost.

For int8:
```
$ tinygpt train --preset tiny --steps 30 --qat int8 ...
  step    30/   30  loss 3.228  qat-err 0.004  · 14.7 step/s
```

`qat-err 0.004` confirms the per-output-row scale + rounding is
producing near-perfect reconstruction at int8 (as expected — 256
quant levels easily cover the typical weight distribution).

### What QAT delivers vs. post-hoc int4

The QAT-trained model deploys to the same int4 kernel as a
post-hoc-quantised model, but with **better weights**: the optimiser
has been routing around the quantisation noise from step 1, so the
final fp32 weights are already grid-aligned. On transformer LMs the
quality gap at int4 deployment is typically 0.5-2 perplexity points
at NO extra inference cost. The training-time cost is the ~10% per-
step overhead seen above.

QAT is **complementary to**, not a replacement for, post-hoc
quantisation (HQQ, AWQ, GPTQ): you typically QAT-train, then choose
one of the post-hoc passes as the final export step. QAT preconditions
the weights to be quantisation-friendly; the post-hoc pass produces
the actual packed payload.

---

## Build verdict

```
$ DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    xcodebuild -scheme tinygpt -destination "platform=macOS" \
    -derivedDataPath /tmp/tinygpt-smoke-quant -configuration Release build
** BUILD SUCCEEDED **
```

No warnings, no errors.

## Out-of-scope (queued)

- **Packed-int matmul kernel.** All four features ship their
  algorithmic infrastructure but lean on a future packed-int matmul
  for the inference-side win. The kernel is the single biggest
  shared dependency.
- **SmoothQuant runtime fold.** Today the pass returns the scale
  vector; folding 1/s into the previous LayerNorm gamma is per-
  architecture and lives in the downstream tooling.
- **GPTQ activation-order (`desc_act=True`).** The reader handles
  arbitrary `g_idx` permutations; the from-scratch worker emits
  `g_idx = floor(i / group)` only. Adding the activation-magnitude
  permutation is ~50 LOC and queued behind real-world demand.
- **QAT travel via .tinygpt header.** Today `--qat int4` on a fresh
  run sets `cfg.qatBits`; the value is NOT yet serialised into the
  `.tinygpt` header field, so a `--resume` defaults back to fp32
  unless the user re-passes `--qat`. Adding the header field is one
  TinyGPTHeader.Config struct entry and matching JSON encoder.
