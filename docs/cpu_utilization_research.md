# CPU utilization research — what we're leaving on the table

Status: research + recommendations. No code shipped. Numbers are estimates
unless explicitly called out as measured.

The flagship `huge` training run (12L · d=256 · ctx=512, ~22M params, bf16,
B=4 × accum=4, cosine LR) is currently observing ~0.07-0.1 step/s on an
M5 Pro (48 GB). Earlier `huge`-preset numbers in `docs/perf_research.md`
recorded **47 ms/step ≈ 21 step/s** (no accumulation, no LR schedule,
compile on). Even after correcting for accumulation (4× microbatches per
step) and the slightly larger configs, we'd expect 1.5-3 step/s, not 0.1.
Something on the CPU side, the optimizer side, or the GPU contention side
is eating ~20× throughput. This doc surveys where the CPU could pull more
weight, both to recover lost throughput and to push past the current
ceiling.

---

## 1. Apple Silicon CPU architecture — what's available to us

The M-series chips Apple ships expose three distinct compute domains, all
sharing the same unified memory:

- **Performance cores (P-cores)** — the high-clock, wide-OoO cores. M5 Pro
  ships 6 P-cores @ ~4.0-4.5 GHz with NEON SIMD and full out-of-order
  execution. These are the cores the Darwin scheduler hands `.userInitiated`
  and `.userInteractive` work to.
- **Efficiency cores (E-cores)** — lower-clock, narrower cores (4 on M5
  Pro). They run at ~2 GHz, draw a fraction of the power, and ideal for
  background work (logging, checkpoint I/O, batch prefetch). QoS levels
  `.utility` and `.background` land here.
- **AMX coprocessor** — the undocumented Apple Matrix Coprocessor, one
  per CPU cluster (so 2 AMX units on M5 Pro per [Asahi Linux docs][asahi]
  and [corsix/amx][corsix]). AMX is **not** a peer of the CPU cores; it's
  driven by special opcodes interleaved with regular ARM instructions, and
  it **shares the CPU's L2 cache**. Apple does not expose AMX directly —
  the only sanctioned route is through Accelerate (BLAS, BNNS, vDSP,
  LAPACK, vImage). AMX achieves roughly **2× NEON throughput** on dense
  matmuls per [Zhou's MIT thesis][zhou-thesis].
- **Neural Engine (ANE)** — separate from CPU/GPU; not relevant for the
  training-loop question (it's inference-only via Core ML).
- **GPU** — where MLX dispatches by default. On M5 specifically, Apple
  added per-GPU-core **Neural Accelerators** (matrix HW) per [Apple's
  MLX/M5 announcement][apple-mlx-m5], so the GPU itself now has tensor
  cores. That changes the CPU/GPU tradeoff again: the GPU is the right
  target for matmuls more than ever.

**Unified memory** means an `MLXArray` allocated on one device is
directly addressable from the other — no PCIe-style copy. That makes
"hybrid CPU+GPU" patterns cheap: we can stage data on the CPU and the GPU
sees it in the same `MLXArray` without a `memcpy`.

### When is the CPU actually faster than the GPU on Apple Silicon?

Per MLX's design notes ([compile docs][mlx-compile-docs] and the M5/MLX
post from Apple), the GPU wins on:

- Large matmuls (the bulk of forward/backward).
- Anything where launch overhead is amortized over enough work.

The CPU is preferred when:

- The op isn't implemented on Metal yet (MLX's `MLXLinalg.svd` is the
  example we already hit — `GaLore.swift:116` and `PeftVariants.swift:111`
  both pass `stream: .cpu` to work around the missing GPU SVD).
- Tiny ops where Metal launch overhead dominates (rare for us — most of
  our small ops happen inside a compiled graph).
- Anything that benefits from CPU cache locality and produces a small
  output. AMX-backed BLAS can be competitive at small sizes (≤ ~256-dim
  GEMMs) because the launch cost is one instruction, not a Metal
  dispatch.

---

## 2. What's actually CPU-bottlenecked in tinygpt's loop

I walked the live code paths in `native-mac/Sources/TinyGPT/Train.swift`
and `TinyGPTModel/Trainer.swift`. Here are the CPU candidates, ranked by
likely impact on the in-flight huge run.

### 2a. Batch sampling (CPU array indexing → MLXArray construction)

`ByteCorpus.sampleBatchRaw` (Trainer.swift:32) builds `[Int32]` arrays of
size B·T = 4·512 = 2048 ints per micro-batch, then `MLXArray(inputs, [B,
T])` materialises them. With accum=4 we do this 4× per optimiser step.

This loop:
```swift
for i in 0..<B {
    let start = Int.random(in: 0..<(bytes.count - T - 1))
    for j in 0..<T {
        inputs[i * T + j] = Int32(bytes[start + j])
        targets[i * T + j] = Int32(bytes[start + j + 1])
    }
}
```
is single-threaded and uses Swift's default `SystemRandomNumberGenerator`
under `Int.random`. On a 440 M-token corpus (well-cached after first
pass) this is **~0.5-2 ms per micro-batch** — call it ~5 ms per step.
**Not the dominant cost** at 47 ms/step, but at 10 s/step it's still
~0.05% — well below the threshold for caring.

**Streaming BPE-dropout (`StreamingTokenizedCorpus.sampleBatchRaw`,
Trainer.swift:141) is a different story.** It re-tokenises the chunk
through `swift-transformers`'s priority-queue BPE on every batch draw.
The comment at line 106 says "~5-15× slower batch construction." For B=4
× T=512 × accum=4 that's 4·512·4 = 8192 tokens encoded per step. If swift-transformers
tokenizes at ~1 MB/s single-threaded (a conservative estimate based on
the priority-queue PR referenced in [swift-transformers releases][st-releases]),
that's **~25-100 ms per step**. **Worth caring about** if BPE-dropout is
on (it is **off** in the live huge run per the launch flags, so this
doesn't explain the regression).

### 2b. MLX dispatch / Python-style host overhead

Even with `compile=on`, each step still does host-side work: closure
invocation, MLXArray construction from the `[Int32]`, the `eval(loss,
model, optimizer)` call that walks the parameter tree, and the
`.item(Float.self)` host-readback that synchronises with the GPU. Each of
these is on the order of 100 µs to 1 ms. At 21 step/s (47 ms/step) this
is in the noise. At 0.1 step/s it's still in the noise — the GPU itself
must be running 10× slower per step.

### 2c. Compile=OFF when cosine LR is on — **likely a real cost**

`Train.swift:458`:
```swift
let canCompile = !useSchedule && accumSteps == 1 && !galoreActive
```
With `--lr-schedule cosine` the entire compiled path is disabled. The
in-flight huge run uses cosine, so it pays interpreted-MLX cost on every
step. Two compounding factors:

1. **No graph fusion** — each `MLX.add`/`MLX.matmul`/`RMSNorm.forward` is
   a separate kernel launch. For a 12-layer model with ~30 fused-able op
   pairs per layer, that's hundreds of Metal launches per forward where a
   compiled graph would issue tens. Typical fusion speedup on
   transformer-class workloads is **1.5-3×** ([mlx compile docs][mlx-compile-docs],
   the "writing fast MLX" gist [linked by awni][awni-fast-mlx]).
2. **Grad accumulation also disables compile** — the comment at
   Trainer.swift:384 confirms this. Even with constant LR, B × accum > 1
   forces interpreted mode. The huge run uses accum=4, so even if we
   moved off cosine, compile would still be off.

This is one of the strongest hypotheses for the v3→v4 regression: if v3
benchmarks were `huge` with `compile=on` and v4 is `huge` with
`compile=off` (cosine + accum), the ~3× compile loss alone could explain
a big chunk.

### 2d. Optimizer step host overhead

Adafactor (`optimizerKind = .adafactor`) is reported in
`docs/optimizers.md` lines 96-102 as **7.8 step/s at huge** vs AdamW's
**16.3** — about **2× slower per step**. That's documented and not a
mystery. If v4 switched from AdamW to Adafactor (to fit Mega/Behemoth in
memory), that compounds with the compile-off cost.

### 2e. Per-step `eval(loss, model, optimizer)` + `.item(Float.self)`

`Trainer.swift:374-376`:
```swift
eval(loss, model, optimizer)
stepCount += 1
return loss.item(Float.self)
```
`eval` forces the lazy graph; `.item()` blocks until GPU finishes. This
is necessary — without it the graph would grow unboundedly. But during
the block, **the CPU is idle**. That idleness is **the opportunity for
batch prefetch** (see §3a).

---

## 3. Concrete CPU-utilization improvements

Listed roughly in order of expected impact for the in-flight huge run.

### 3a. Multi-step batch prefetch (real, not just an `actor`)

`Trainer.swift:203` declares `BatchPrefetcher` as an `actor`:
```swift
public actor BatchPrefetcher {
    public func next() -> ([Int32], [Int32]) {
        corpus.sampleBatchRaw(batchSize: batchSize, contextLength: contextLength)
    }
}
```
But `Train.swift:518` admits it's not wired into the loop:
> "We've dropped the explicit prefetch pipeline because MLXArray
> construction blocks anyway; the saved overlap was small."

That's **only true when MLXArray construction goes to the GPU and forces
sync.** With unified memory, MLXArray construction from a Swift `[Int32]`
is mostly a pointer copy into MLX's allocator and **does not block on
the GPU**. The "small saved overlap" claim is suspect at 0.1 step/s — at
10 s per step we have 10 s of GPU work to hide CPU sampling behind, so
even a 1 ms sampler is hidden for free, but with accum=4 the **next**
step's sampling can start as soon as the **previous** step's last
microbatch is dispatched. Microbatches don't block each other on
construction.

**Concrete pattern that would work**:
```swift
// Pseudocode — not for this doc to ship
Task(priority: .userInitiated) {
    while training {
        let batch = await corpus.sampleBatchRaw(...)
        await queue.put(batch)   // bounded queue, depth 2-4
    }
}
```
A bounded-channel async producer with depth 2 lets the CPU stay one step
ahead. Use `Task` with `.userInitiated` so the cooperative pool schedules
it on a P-core ([SwiftLee on actor scheduling][swiftlee-actors]).

**Impact estimate**: 5-15% step/s recovery in the BPE-dropout case
(where sampling is ~50 ms). Negligible (<2%) for the byte/cached-token
path. Effort: ~2 hours.

### 3b. Re-enable compile under cosine LR (LR-as-MLXArray-input)

Today the compiled closure captures the LR as a captured Swift `Float`
inside the optimizer object — when the schedule rewrites
`trainer.optimizer.learningRate`, the compiled graph can't see the new
value (per [mlx-swift Transforms+Compile][mlx-compile-source], "if a
captured Float changes between calls, the compiled function would retain
its closure over the original value, not reflect the change").

The MLX docs explicitly recommend threading scalars **as inputs** rather
than captured constants ([mlx-compile-docs]):
> "instead of capturing learning_rate as a constant ... pass them as
> function inputs"

For us that means:
1. Make the optimizer's `learningRate` an `MLXArray` of shape `[]`
   (scalar tensor) instead of a Swift `Float`.
2. Pass it as an additional `Updatable` to `compile(inputs: [m,
   optimizer], outputs: [m, optimizer])` so the graph sees the in-place
   write.
3. The cosine schedule writes into that scalar tensor each step (cheap;
   the GPU sees it via unified memory).

After that, `canCompile = !galoreActive && accumSteps == 1` — cosine no
longer kills compile.

**Impact estimate**: **1.5-3× step/s on cosine runs**, per the MLX
compile-fusion literature. This is probably **the highest-ROI single
change** in this list. Effort: ~3-4 hours — needs `LearningRateMutable`
protocol changes and audit of every optimizer that conforms (AdamW,
Lion, Sophia, Muon, Adafactor adapter).

### 3c. Compile-friendly accumulation

The same logic applies to gradient accumulation: instead of building a
new `micros: [(MLXArray, MLXArray)]` list each step (Train.swift:538),
compile a fixed-shape `[N, B, T]` accumulator step where N=accum is a
trace-time constant. The accumulated step then becomes a single compiled
function rather than N interpreted forward+backward calls.

**Impact estimate**: +30-50% step/s on any accum>1 run. Effort: ~4-6
hours — the gradFn returns nested `ModuleParameters`; summing them
inside a compiled trace needs `mapValues(grads) { a, b in a + b }` to be
graph-pure (it is).

### 3d. Rust-FFI BPE tokenizer (only if BPE-dropout is on)

The `swift-transformers` BPE tokenizer is single-threaded per its source
(the priority-queue PR speeds up the *inner* merge loop but doesn't add
parallelism). For BPE-dropout training, where the same text is
re-tokenised on every batch, we'd ideally parallelise across the B rows
of the microbatch.

Two options:

1. **Use `huggingface/tokenizers` (Rust) via C ABI.** The Rust crate
   exposes `encode_batch()` that uses rayon under the hood ([tokenizers
   crate][hf-tokenizers]). We'd build it as a `staticlib`, link via
   Swift Package Manager's system-library target, and call through a
   thin C header. The Rust crate is ~50× faster than the swift-transformers
   priority-queue path for batch encode on multi-core CPUs.
2. **Roll our own with rayon-style work-stealing.** Possible but
   reinvents what HF already ships.

**Impact estimate**: For BPE-dropout runs only — 5-10× tokenization
throughput, which translates to maybe **10-25% step/s** on a
streaming-corpus run. **Not** in the critical path for the current
huge-fineweb run (BPE-dropout is off). Effort: ~1-2 days including the
Rust toolchain integration.

### 3e. Pre-allocated CPU buffers (avoid per-step `[Int32]` allocation)

`sampleBatchRaw` allocates a fresh `[Int32]` of size B·T each call —
2048 × 4 bytes = 8 KB per call. At 4 calls per step that's 32 KB malloc
churn per step. Trivial in absolute terms but visible in a Time Profiler
trace; reuse a pre-allocated buffer that lives on the prefetcher actor.
**Impact estimate**: <1%. Effort: ~30 min. Not worth doing alone, but a
freebie once we touch the prefetcher anyway.

### 3f. QoS tuning on the training thread

The CLI's training loop currently runs on whatever thread `main()`
dispatched to — which is the main thread for the SwiftUI
`TrainController.swift` path, and an `await`-driven Task for the CLI
path. Neither path explicitly sets QoS, so Swift's defaults apply:
`.utility` for unannotated `Task {}`. That can land the training thread
on an **E-core**, where the orchestration latency (Metal dispatch,
gradient-tree walks) is **~2× slower** than on a P-core (per the
M5 P-core / E-core clock-frequency gap).

**Fix**: wrap CLI invocations in `Task(priority: .userInitiated)` or use
`DispatchQueue.global(qos: .userInteractive).async`. Move logging,
manifest writes, and checkpoint I/O to `.utility` so they land on the
E-cores and don't steal P-core cycles from the train loop.

**Impact estimate**: **5-20% step/s** if we're actually getting stuck on
an E-core for the orchestration thread. Hard to tell without
`powermetrics` confirming. Effort: ~30 minutes.

### 3g. Verify SVD-on-CPU-stream is actually using AMX

We already pass `stream: .cpu` for SVD in `GaLore.swift:116` and
`PeftVariants.swift:111`. The good news: MLX's CPU backend calls into
LAPACK, and LAPACK on macOS routes through Accelerate, which **does** use
AMX silently ([corsix/amx][corsix], [eclectic light "Why apps need
Accelerate"][accel-eclectic]). So we get AMX "for free" on those calls.
There's no further win here unless we add more CPU-stream ops.

**Action**: not a change — just verify with `xctrace` that the SVD calls
show up under `libBLAS.dylib` (Accelerate) and not under a fallback
codepath.

---

## 4. What can't be meaningfully CPU-accelerated

Being honest about the ceiling:

- **The forward/backward matmuls are 99% of compute.** A `huge` training
  step does ~12 × (4 × matmul per attention) + ~12 × (2 × matmul per
  MLP) ≈ 70 matmuls per microbatch, plus the embedding lookup and
  unembedding. These are large enough (d=256, T=512) that GPU
  Metal-shader throughput exceeds AMX. Even if MLX-Swift exposed AMX
  directly (it doesn't), we'd lose on the matmul itself by ~10× moving
  off the GPU.

- **MLX-Swift doesn't expose AMX directly.** AMX is accessible only
  through Accelerate routines, and MLX's CPU backend calls those for
  BLAS/LAPACK ops only. There's no `mlx.amx.matmul()`. We depend on
  Accelerate's routing.

- **Single-threaded Swift code on the orchestration thread.** Even with
  P-core scheduling, MLX-Swift's host-side closure invocation is
  fundamentally serial: we can't parallelise the train-step closure
  itself, only the data pipeline feeding it.

- **`item(Float.self)` is mandatory.** We need to read the loss back to
  the CPU for logging; that's a forced GPU→CPU sync. The sync cost is
  microseconds, but it does serialize the train loop.

---

## 5. Diagnostic tools to verify hypotheses

Before shipping anything, we should *measure*. The MIT-thesis profiling
playbook ([Zhou 2025][zhou-thesis]) and Apple's own [WWDC25 MLX video][wwdc25-mlx]
both recommend the following stack:

### 5a. `xctrace` Time Profiler (CPU sampling)

```
xctrace record --template "Time Profiler" \
    --launch -- ./tinygpt train --preset huge --steps 100 --corpus /tmp/x.txt \
    --output /tmp/train.trace
```
Then open in Instruments. Look for:

- Time spent in `swift_release` / `swift_retain` — high ARC pressure means
  we're allocating in the hot path (likely candidate: per-step
  `ModuleParameters` reconstruction).
- Time in `Tokenizers.BPE.tokenize` — confirms or refutes the BPE-dropout
  hypothesis.
- Time in `MLX.Core.eval` — the GPU sync wait; should be the dominant
  bucket if we're GPU-bound.
- Time in `libBLAS.dylib` / `libLAPACK.dylib` — confirms AMX is being
  hit on SVD/CPU-stream paths.

### 5b. `powermetrics` (per-core utilization, P vs E)

```
sudo powermetrics --samplers cpu_power -i 500 -n 60
```
Watch the per-core "active residency" column ([osxdaily howto][osx-powermetrics]).
If we see the training thread bouncing between E-cores, that confirms
the §3f QoS hypothesis. If a single P-core is pegged at 100% and others
are idle, that's the serial-orchestration ceiling.

### 5c. `MLX.synchronize()` to separate dispatch from compute

Wrap each step with explicit synchronize boundaries:
```
let t1 = Date()
let (x, y) = sample()
MLX.synchronize()  // hypothetical
let t2 = Date()
let loss = trainer.step(...)
MLX.synchronize()
let t3 = Date()
```
The `t2 - t1` bucket is pure CPU dispatch + sampling. The `t3 - t2`
bucket is GPU compute + sync. If `t2 - t1` is 50 ms and `t3 - t2` is 50
ms, the CPU is the bottleneck and §3a/b/c help. If `t2 - t1` is 1 ms and
`t3 - t2` is 10 s, the GPU is the bottleneck and the problem is
contention or model architecture, not CPU.

### 5d. `asitop` for a TUI overview

[asitop][asitop] gives a real-time CPU+GPU+ANE+memory dashboard. Good
for spotting GPU contention from concurrent agent processes (the
"3 agents hogging GPU" hypothesis from the task description).

---

## 6. Prioritized next steps

In order of impact-per-effort:

| # | Change | Effort | Expected step/s lift |
|---|---|---|---|
| 1 | **Re-enable compile under cosine LR** (3b: LR-as-MLXArray-input) | ~4h | 1.5-3× |
| 2 | **Compile-friendly accumulation** (3c: fixed-N traced accum loop) | ~5h | +30-50% on accum>1 |
| 3 | **QoS tuning + verify with powermetrics** (3f) | ~1h | +5-20% |
| 4 | **Re-enable BatchPrefetcher pipeline** (3a: bounded async queue) | ~2h | +5-15% (mostly with BPE) |
| 5 | **Rust-FFI tokenizers** (3d) | ~1-2 days | +10-25% on BPE-dropout only |

The combination of #1 + #2 alone should plausibly restore the huge-run
throughput from 0.1 step/s to 1-2 step/s, assuming no GPU contention.
Past that, #3 and #4 are quick wins that take us further if measurement
confirms. #5 is only worth it if we commit to BPE-dropout as a
first-class training mode.

---

## 7. The v3→v4 regression theory

The task description notes v4 is at ~0.07 step/s vs v3's ~3.2 step/s on
"the same hardware/preset." Several effects compound; here are the
hypotheses ranked by my prior on which is real:

1. **`compile=off` because of cosine LR + grad accumulation** (§2c).
   This single change can cost **2-3×** compared to a compile-on run.
   The v3 47-ms/step number in `docs/perf_research.md` was measured
   with no schedule and no accum (the `Bench.swift` defaults). High
   confidence — backed by the explicit `canCompile` gating logic.

2. **Adafactor vs AdamW** (§2d). If v4 switched optimizer to Adafactor
   to leave more memory for activations, that's **~2× per step**,
   documented in `docs/optimizers.md:174` ("AdamW (5.2 vs 16.3
   step/s)"). High confidence if optimizer changed.

3. **z-loss + embedding RMSNorm** add small constant-factor work:
   - z-loss is a `logsumexp(logits).square().mean()` — one extra
     reduction over `[B, T, V]` logits. Cost is similar to one cross-entropy
     pass — perhaps **+5-10% per step**.
   - Embedding RMSNorm is one extra RMSNorm over the embedded `[B, T,
     d]` tensor — **+1-3% per step**.
   Each is small; together maybe **+10-15%**.

4. **GPU contention from concurrent agents.** The task description
   mentions 3 agents running smoke tests during v4's start. MLX-Swift
   shares a single Metal device; concurrent processes serialize on the
   command queue. This can be **arbitrarily bad** — if 3 other
   processes are each dispatching ~equal work, you get ~25% of the
   GPU. **Easy to verify** with `asitop` or `powermetrics --samplers
   gpu_power`. Medium-high confidence.

5. **Memory pressure / swap.** 48 GB unified with multiple agents
   running can hit swap. A bf16 huge model at ~22M params + B=4·accum=4
   activations is well under 48 GB, but if other agents are also
   loading 100M-class models, the system could be paging. **Symptom**:
   step-rate is bimodal — some steps fast, some 10× slow. Worth
   checking with `vm_stat 5`.

6. **Long-context cost.** The v3 huge bench was ctx=512 too (per the
   `huge` preset definition), so this isn't a new cost vector unless
   ctx was bumped. Low confidence.

7. **BPE-dropout streaming.** Off in the current run per launch flags,
   so doesn't explain v4. Low confidence.

**My best guess**: it's the multiplicative stacking of (1)+(2)+(4).
`compile=off` × Adafactor × ~25% GPU share ≈ **2 × 2 × 4 = 16×** slower
than the 3.2 step/s baseline → ~0.2 step/s, close to the observed 0.07-0.1.
The remaining gap is z-loss + embedding RMSNorm + general overhead.

The way to confirm without changing the live run: launch a parallel
quick `bench` with the **same flags** on the same hardware (`tinygpt
bench --preset huge --steps 50`) once the other agents finish. If that
bench also reports ~0.1 step/s, it's the flag stack. If it reports
~3 step/s, it's GPU contention.

---

## Sources

- [Asahi Linux: Apple Silicon Accelerators][asahi]
- [corsix/amx: Apple AMX Instruction Set][corsix]
- [Zhou 2025: Performance Analysis of the Apple AMX Matrix Accelerator (MIT thesis)][zhou-thesis]
- [Apple ML Research: Exploring LLMs with MLX and the M5 Neural Accelerators][apple-mlx-m5]
- [MLX documentation: Compilation][mlx-compile-docs]
- [mlx-swift: Transforms+Compile.swift][mlx-compile-source]
- [Awni Hannun: Writing Fast MLX][awni-fast-mlx]
- [WWDC25 Session 315: Get started with MLX for Apple silicon][wwdc25-mlx]
- [Apple Developer: Accelerate Overview][accel-overview]
- [Eclectic Light: Why apps need Accelerate][accel-eclectic]
- [huggingface/tokenizers (Rust crate)][hf-tokenizers]
- [swift-transformers releases (BPE priority queue PR)][st-releases]
- [SwiftLee: Thread dispatching and Actors][swiftlee-actors]
- [osxdaily: per-core CPU usage via powermetrics][osx-powermetrics]
- [asitop: perf monitoring CLI for Apple Silicon][asitop]

[asahi]: https://asahilinux.org/docs/hw/soc/accelerators/
[corsix]: https://github.com/corsix/amx
[zhou-thesis]: https://commit.csail.mit.edu/papers/2025/Jonathan_Zhou_SB_Thesis.pdf
[apple-mlx-m5]: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
[mlx-compile-docs]: https://ml-explore.github.io/mlx/build/html/usage/compile.html
[mlx-compile-source]: https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Transforms+Compile.swift
[awni-fast-mlx]: https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50
[wwdc25-mlx]: https://developer.apple.com/videos/play/wwdc2025/315/
[accel-overview]: https://developer.apple.com/accelerate/
[accel-eclectic]: https://eclecticlight.co/2023/12/16/why-apps-need-to-accelerate/
[hf-tokenizers]: https://github.com/huggingface/tokenizers
[st-releases]: https://github.com/huggingface/swift-transformers/releases
[swiftlee-actors]: https://www.avanderlee.com/concurrency/thread-dispatching-actor-execution/
[osx-powermetrics]: https://osxdaily.com/2024/07/05/how-to-see-individual-core-cpu-usage-on-mac-with-powermetrics/
[asitop]: https://github.com/tlkh/asitop
