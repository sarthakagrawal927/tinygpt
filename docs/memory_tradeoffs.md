# Memory tradeoffs — bf16, gradient accumulation, gradient checkpointing

What fits on a 48 GB Mac for training is dominated by four memory
costs: weights, optimizer state, gradients, and per-step activations.
This guide explains each in concrete numbers and the levers we have
shipped (bf16, gradient accumulation) plus the lever we haven't yet
(gradient checkpointing).

## The four memory costs

For a model with `P` parameters at fp32:

| What | Size | Notes |
|---|---|---|
| **Weights** | 4P bytes | The trainable matrices themselves. |
| **Optimizer state** (AdamW) | 8P bytes | Two moving averages (m, v) per param. |
| **Gradients** | 4P bytes | One per param, freshly computed every backward. |
| **Activations** | B · T · C · L · ~10 bytes | All intermediate tensors needed for backward. The biggest variable. |

Where B is batch, T is context length, C is `d_model`, L is layer count.
The "~10" is a rough constant for the number of distinct activations
saved per layer per position; modern transformer layers have ~10
(input, ln1 output, q, k, v, attn output, ln2 output, mlp up, mlp gate,
mlp out — roughly).

### Concrete: Mega @ fp32, B=4, ctx=1024

| Cost | Size |
|---|---|
| Weights | 4 × 100M = 400 MB |
| AdamW state | 8 × 100M = 800 MB |
| Gradients | 4 × 100M = 400 MB |
| Activations | 4 × 1024 × 512 × 24 × 10 × 4 = ~2 GB |
| **Total** | **~3.6 GB** |

Plus the corpus (2 GB tokenized) and various overhead → **~6 GB
working set**. Comfortably fits in 48 GB.

### Concrete: Titan-1.3B @ fp32, B=2, ctx=1024

| Cost | Size |
|---|---|
| Weights | 5.2 GB |
| AdamW state | 10.4 GB |
| Gradients | 5.2 GB |
| Activations | ~5 GB |
| **Total** | **~26 GB** |

Tight on 48 GB but fits. **Doubling the batch wouldn't.**

---

## Lever 1: bf16 training (`--dtype bfloat16`)

### What it does

bf16 stores every floating-point tensor in 2 bytes instead of 4. Halves
the weight, gradient, optimizer-state, and activation memory at once.
~2× more headroom for batch size or context.

### Why bf16, not fp16

| Format | Mantissa | Exponent | Range | Training-stable? |
|---|---:|---:|---|---|
| fp32 | 23 | 8 | ~1e-38 to ~1e38 | yes |
| **bf16** | 7 | 8 | ~1e-38 to ~1e38 | yes (same range as fp32) |
| fp16 | 10 | 5 | ~6e-5 to ~65504 | no — gradients underflow without loss scaling |

bf16 is the modern training format because it has fp32's range with
fp16's bitwidth. No loss scaling, no master weights, no scaffolding —
just train.

### The catch

bf16 has only 7 bits of mantissa (vs fp32's 23). Optimizer m/v moments
accumulate gradient signal over many steps; with only 7 bits of
precision, the small running-average updates can quantize to zero. For
short runs (thousands of steps) this is fine; for very long runs
(hundreds of thousands of steps) you may want fp32 master weights or
fp32 optimizer state. We do not yet keep optimizer state in fp32 — a
known limitation for the Titan-class training horizon.

### Reproduce

```bash
tinygpt train --preset mega --dtype bfloat16 ...
```

Verified parity: a 100-step bf16 run on alice.txt lands within 0.04 nats
of an fp32 run (1.6% drift) — within typical batch sampling noise. See
`docs/precision.md` for the parity-test methodology.

---

## Lever 2: Gradient accumulation (`--accum N`)

### What it does

Run N micro-batches through the model BEFORE applying an optimizer
update. Sum the gradients across the micro-batches, divide by N, then
step. Effective batch size = `--batch × --accum`, with the memory cost
of just `--batch`.

### Why it matters

Memory is dominated by activations, which scale with `B × T`. If you
want effective batch 16 at ctx=1024 but only have memory for batch 4,
gradient accumulation lets you train at the same compute-effective
batch size with ¼ the activation cost.

The catch: each micro-batch is a full forward + backward, so wall time
scales linearly with `--accum`. You spend the SAME tokens-per-step but
take N× longer to compute them. The tradeoff is memory for time —
useful when memory is the binding constraint, not when wall time is.

### Parity

Verified equivalent to a single big batch (up to batch sampling noise):

| Config | Final loss |
|---|---:|
| B=16, single step | 2.856 |
| B=4, accum=4 | 2.800 |
| B=2, accum=8 | 2.887 |

All within 0.087 nats of each other after 50 steps on alice.txt. The
math is correct.

### Reproduce

```bash
tinygpt train --preset mega --batch 4 --accum 4 ...
# effective batch 16, memory cost of batch 4
```

Compile is disabled when `--accum > 1` (the operation graph changes
shape with N micro-batches; per-step retracing erases the compile win
anyway).

---

## Lever 3: Gradient checkpointing (not yet shipped)

### What it would do

Don't save activations during forward. Re-compute them during backward
from saved layer-input checkpoints. Trades ~30% extra compute for ~√L
activation memory reduction (where L is layer count).

### Why it'd unlock the next tier

For Behemoth (404M params, 32 layers) at fp32 with full activations,
the math doesn't fit B=4 × ctx=1024 on 48 GB. With gradient
checkpointing it would. Same for Titan at ctx=2048.

### Why it's not shipped yet

MLX-Swift doesn't expose checkpointing as a first-class feature; we'd
need to either write a custom forward path that drops + recomputes
manually, or wait for the MLX team to add it. Tracked but not in the
critical path for our scale today (Mega fits comfortably without).

---

## Combined recipe — what we use for Mega-on-FineWeb

```bash
tinygpt train --preset mega \
    --dtype bfloat16 \    # 2× memory savings → 2× batch headroom
    --batch 4 \           # micro-batch memory budget
    --accum 4 \           # effective batch 16 — Chinchilla-ish for stable training
    --ctx 1024 \          # long enough for paragraph-level dependencies
    --tokenizer /tmp/smollm2 \
    --corpus /tmp/fineweb-edu-500M.txt \
    ...
```

Working set at this config: ~8-10 GB total (model + optimizer +
gradients + activations + tokenized corpus). On a 48 GB Mac there's
roughly 4× the headroom needed; you could lift to ctx=2048 or B=8 and
still fit.

---

## Gradient checkpointing — blocked on MLX-Swift exposure

Gradient checkpointing trades compute for memory by discarding
activations during forward and recomputing them during backward.
At per-layer granularity, activation memory drops from `O(L)` to
`O(sqrt(L))` at a ~30% wall-clock cost — the classic lever for
fitting much bigger models in the same RAM.

The underlying op exists in MLX (`mx.checkpoint` in Python; the C API
exposes `mlx_checkpoint`), but **MLX-Swift has not yet wrapped it as
a public function**. The C primitive is present in
`mlx-swift/Source/Cmlx/mlx-c/mlx/c/transforms.h`:

```c
int mlx_checkpoint(mlx_closure* res, const mlx_closure fun);
```

To ship gradient checkpointing today, MLX-Swift would need to expose
something like:

```swift
public func checkpoint(
    _ fn: @escaping ([MLXArray]) -> [MLXArray]
) -> ([MLXArray]) -> [MLXArray]
```

Bridging it from outside the package isn't clean because `Cmlx` is
internal to MLX-Swift. The status of this is tracked upstream in the
MLX-Swift repo.

### What this means in practice today

If you need to fit a model that doesn't quite fit in 48 GB:
1. First lever — `--dtype bfloat16`. Halves weights + gradients
   + activations.
2. Second lever — `--accum N`. Same effective batch with B/N
   per-step activation memory.
3. Third lever — `--ctx 512` or `--ctx 256`. Quadratic savings in
   attention memory.

These three together get the user to roughly the same memory
footprint that gradient checkpointing would unlock for a single
modestly larger model. They're not free — bf16 has slight accuracy
implications, accumulation slows per-step throughput, shorter context
trains the model on shorter windows — but they ship today.

---

## Cross-reference

- `docs/precision.md` — the fp32 vs fp16 vs bf16 numerics study
- `docs/training_phases.md` — how these levers compose into the full
  pretrain → SFT → DPO pipeline
- `native-mac/ARCHITECTURE.md` — where in the code each lever lives
- Mac CLI source for the levers:
  - bf16: `Sources/TinyGPT/Train.swift` (search "bf16 / fp16 training")
  - gradient accumulation: `Sources/TinyGPTModel/Trainer.swift`
    (`accumulatedStep` method)
