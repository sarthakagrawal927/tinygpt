# Gradient checkpointing — worked example

This doc captures the measurements from wiring custom gradient
checkpointing into the Mac training path (Tier 1.7 of the
single-machine roadmap). The numbers below were collected on an M5
Pro with 48 GB unified memory, MLX-Swift 0.25, fp32 training, Metal
backend.

## TL;DR

`tinygpt train --grad-checkpoint` wraps every `TransformerBlock`
forward in an MLX `CustomFunction` whose VJP re-runs the block
forward at backward time. The block's intermediate activations are
not retained across the outer backward, so per-block activation
memory drops at the cost of one extra forward per block at backward
time.

The wins land at the scales where activations actually dominate
memory (large models or long contexts at non-trivial batch sizes):

| Config | B | ctx | Peak no-ckpt | Peak with ckpt | Δ memory | step/s no-ckpt | step/s with ckpt | Δ speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny (4L · d=128) | 8 | 128 | 161 MB | 189 MB | **+17%** | 142.7 | 126.3 | -11% |
| huge (12L · d=256) | 8 | 256 | 1611 MB | 1438 MB | -11% | 18.6 | 16.4 | -12% |
| huge (12L · d=256) | 4 | 1024 | 3938 MB | 3223 MB | -18% | 4.8 | 3.9 | -19% |
| mega (24L · d=512) | 2 | 1024 | 5800 MB | 4300 MB | **-26%** | 2.1 | 2.0 | -5% |
| behemoth (32L · d=1024) | 1 | 1024 | 11230 MB | 8848 MB | -21% | 1.3 | 1.2 | -8% |
| behemoth (32L · d=1024) | 2 | 1024 | 16747 MB | 11749 MB | **-30%** | 0.8 | 0.7 | -12% |
| behemoth (32L · d=1024) | 3 | 1024 | 22019 MB | 14777 MB | **-33%** | 0.5 | 0.5 | 0% |
| behemoth (32L · d=1024) | 4 | 1024 | 27679 MB | 17816 MB | **-36%** | 0.4 | 0.3 | -25% |

Two patterns are visible:

1. **Tiny models are a net loss.** When the block forward is small
   enough that the CustomFunction trace + recompute overhead dominates
   the savings, checkpointing _adds_ memory (the recompute trace
   itself has to live somewhere). The flag is opt-in for exactly this
   reason — it's tuned for models too big to train without it.

2. **At behemoth scale the savings are substantial** — going from
   28 GB peak down to 18 GB peak (a 10 GB headroom recovery) at
   B=4 ctx=1024 trades for a 25% step-time hit. That's roughly the
   "30% extra compute, ~√L activation memory" trade the literature
   advertises (Chen et al. 2016, "Training Deep Nets with Sublinear
   Memory Cost"), though MLX's recompute overhead is a little less
   than the theoretical ~33% in practice — the kernels for our
   transformer blocks are well-batched, and the CustomFunction
   path inherits MLX's lazy-eval batching.

## How it works in code

MLX-Swift (as of 0.25) doesn't expose a first-class `mlx.checkpoint`
primitive. The mechanism we use instead:

```swift
// Pseudocode of GradCheckpoint.wrap
let cf = CustomFunction {
    Forward { inputs in
        let x = inputs[0]
        let params = unflatten(Array(inputs.dropFirst()), keys)
        block.update(parameters: params)
        return [block.rawForward(x)]
    }
    VJP { primals, cotangents in
        // re-execute forward inside vjp() so gradients exist
        let (_, grads) = MLX.vjp(sameForward, primals: primals,
                                 cotangents: cotangents)
        return grads
    }
}
return cf([x] + flatParams)
```

The block's trainable parameters are threaded through the
`CustomFunction` as declared inputs, mirroring how `valueAndGrad`
injects them as primal inputs to the gradient transform. The block's
@ModuleInfo slots are re-slotted with the input tracers via
`block.update(parameters:)` before each invocation — safe under
MLX-Swift's tracing model because each training step retraces (or
replays a compiled trace) with fresh tracers.

See `native-mac/Sources/TinyGPTModel/GradCheckpoint.swift` for the
real implementation and inline comments.

## What's NOT done

* **No selective layer checkpointing.** Every block is wrapped when
  `--grad-checkpoint` is on. The classical recipe is to checkpoint
  every √L blocks (giving √L peak activation memory with one extra
  forward per checkpointed block); we always checkpoint all blocks,
  giving the simpler "one extra forward per block" trade. Adding the
  √L stride would be a config knob + a one-line change in the model
  loop.

* **Compile path interaction.** The wrapper is compatible with
  `MLX.compile`. The compiled trace simply includes the
  CustomFunction's Forward + VJP closures — no special handling
  needed. We verified end-to-end with compile-on training (the
  numbers above all use compile).

* **MoE / MoD / differential-attention / YOCO paths.** The wrapper
  runs `block.callAsFunction` via `rawForward`, which threads all
  those flags through. None of those modes were specifically tested
  with `--grad-checkpoint` in this pass — they're expected to work
  but could need additional debugging if MoE's auxiliary-loss
  side channel interacts with the recompute pass strangely.

## Reproducing the numbers

The test corpus is the first 5 MB of `/tmp/fineweb-edu-500M.txt`
(any UTF-8 text works; activations don't depend on token statistics
beyond shape). Each measurement is from a 2-5 step run — long enough
to amortise compile-trace cost but short enough to iterate quickly.

```bash
# Baseline (no checkpoint)
tinygpt train --preset behemoth --steps 3 --corpus /tmp/corpus.txt --batch 3

# With checkpoint
tinygpt train --preset behemoth --steps 3 --corpus /tmp/corpus.txt --batch 3 --grad-checkpoint
```

The "memory:" line at end of training reports peak GPU memory,
active memory, and cache memory as measured by `MLX.Memory.peakMemory`
(reset at training start so the number reflects training only, not
model construction).

## Honest caveats

* **MLX-Swift doesn't have a first-class checkpoint primitive.** What
  we shipped is a workaround built on `CustomFunction` + `vjp`. It
  works (the loss curves match), but the line count and complexity
  is meaningfully higher than the PyTorch one-liner
  (`torch.utils.checkpoint.checkpoint(block, x)`). If upstream MLX
  ever adds `mlx.checkpoint`, we should swap to that — it'll likely
  shave another ~5-10% off the recompute trace overhead by avoiding
  the per-block `block.update(parameters:)` round-trip.

* **The activation accounting in OOMGuard doesn't yet know about
  `--grad-checkpoint`.** The pre-flight memory estimate prints the
  un-checkpointed activation projection; the runtime peak is the
  honest measurement. Updating the estimator is a one-line change
  but wasn't in scope for this pass.

* **Resume + change-of-flag is allowed but mildly fragile.** The
  flag is persisted in the `.tinygpt` header, so a `--resume` keeps
  the same memory profile by default. Passing `--grad-checkpoint`
  alongside `--resume` upgrades a non-checkpointed checkpoint to
  checkpointed training (the OR in `Train.swift`). Downgrading
  isn't currently exposed — you'd have to manually edit the header
  or just not pass the flag and rely on the persisted value.
