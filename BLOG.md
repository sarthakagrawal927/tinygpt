# I built a GPT-2 in the browser, then made it 9.7× faster

I started [TinyGPT](https://github.com/sarthakagrawal927/tinygpt) as a
teaching project. I wanted to understand the modern LLM stack at a size where
nothing stays a black box — every backward pass derived by hand, every kernel
parity-checked, no autograd engine hiding the maths on the C++ and GPU side.
The plan: write the PyTorch reference, port it to C++/WASM, port it again to
WebGPU, ship a playground, write up what I learned, move on.

I did not plan to spend a month optimizing matmul kernels.

But once the playground worked end-to-end, the obvious next question was: how
fast can a GPT-2-shaped model actually train inside a Chrome tab, without
lying about the numbers? That turned into a perf project. This post is what
happened — what worked, what didn't, and what the negative results taught me.
All measurements are on the same Apple M-series laptop. The full log lives
in [`browser/devlog.html`](browser/devlog.html); the public roadmap is in
[`browser/roadmap.html`](browser/roadmap.html).

## The architecture in 60 seconds

The same model exists at three levels, built in that order.

`python_ref/` is the PyTorch reference — the clearest version, the oracle
when anything else disagrees. `wasm/` is the same maths in C++, with every
backward pass derived by hand and compiled to WebAssembly via Emscripten.
`webgpu/` is the whole training loop in WGSL — forward, backward, AdamW, all
24 kernels — each one finite-difference checked against the reference. All
three read and write the same `.tinygpt` binary file format, so a model
trained in any path continues training in any other.

One detail mattered more than I expected: Memory64. V8 caps each tab's
WebAssembly heap near 4 GB using 32-bit pointers — roughly 250M fp32
parameters plus their AdamW optimizer state. Compiling with `-sMEMORY64=1
-sWASM_BIGINT` produces a separate `tinygpt64.{js,wasm}` module that uses
64-bit pointers; the runtime feature-detects and picks the right one. With
that, I could allocate a **473M-parameter** model — same C++ source, same
training-step machinery — that hard-OOMs on the 32-bit module. Allocation
takes 3.7 s, one training step takes 82.2 s on CPU. Slow, but it allocates,
which is the part that matters: the model-size ceiling moved out of the way,
and Memory64 became the gate for the "Behemoth" preset in the playground.

## The kernel sweep

About 80% of training time is matmul. So 80% of the speed work was matmul.
The starting point: a naive WebGPU matmul — one thread per output cell,
every input read from global memory every iteration. Even that was already
about 7× faster than WASM SIMD on the Small preset, because GPUs are GPUs.
But on the 2048³ matmul that dominates the Mega and Behemoth presets, naive
was leaving 5× on the table.

Two optimizations actually shipped. Here they are, with real numbers from
the playground's benchmark button.

**Step 1 — workgroup-shared tiling (16×16).** The textbook Goto/VandeGeijn
pattern. Each workgroup of 16×16 threads cooperatively loads a 16×16 block
of A and a 16×16 block of B into `var<workgroup>` shared memory; each thread
then does 16 multiply-accumulates from shared. 16 global reads become 1
global + 16 shared reads — and on big matmuls, that's where the GPU starts
looking like a GPU. Clean ~2.5× over naive across every realistic size.

**Step 2 — 4×4 register blocking on top of tiling.** Each of the 256 threads
in the workgroup now computes a *4×4 block* of output values held in
registers, and the workgroup outputs a 64×64 tile. Because of the
outer-product structure, each shared-memory load gets reused 4× across the
thread's register accumulator. Arithmetic intensity per shared-memory load
climbs from roughly 1 multiply-add to 16 — well past the point where the
kernel flips from bandwidth-bound to compute-bound.

The measured kernel sweep at four sizes:

| matmul size | naive ms | tiled ms | blocked4 ms | vs naive |
| ----------- | -------- | -------- | ----------- | -------- |
| 256³        | 0.87     | 0.72     | 0.45        | 1.93×    |
| 512³        | 1.74     | 0.86     | 0.64        | 2.72×    |
| 1024³       | 6.43     | 2.85     | 1.80        | 3.58×    |
| 2048³       | 47.24    | 17.23    | 9.12        | 5.18×    |

The speedup grows with matrix size, because bigger problems amortize the
workgroup-shared loading more effectively across the 4×4 register reuse.
At 2048³ — the realistic shape for Mega/Behemoth — the blocked kernel runs
5.18× faster than naive and 1.89× faster than merely tiled.

Wiring it into `train.wgsl` was a drop-in replacement (same bind-group
layout as the naive kernel). The end-to-end parity test
(`tests/test_webgpu_train.mjs`) confirmed it produces equivalent training:

```
WASM SIMD       6.8 s · loss 2.9385
WebGPU+block    0.7 s · loss 2.9719   →  9.7× wall-clock, 1.1% loss drift
```

9.7× end-to-end speedup. 1.1% loss drift after 50 training steps — pure
float-reorder noise from the different accumulation order in the GPU
kernels.

## The three things that didn't work

This is the part of the project that taught me the most, and the part most
posts about perf work skip. Three optimizations I expected to win, all
backed by reasonable intuitions, all rejected by the bench.

**f16-packed storage — the compound assumption was wrong.** Idea: store
weights as two f16 per u32 via `pack2x16float`, accumulate in f32, halve
global bandwidth. Standalone benchmark at 2048³: **1.7× faster than naive**
WebGPU matmul. Looked great. Then I compared against the right baseline —
the already-tiled kernel — and stacked f16 on top of it. The combined
`tiled+f16` ran *slower* than plain tiled at 2048³: **17.78 ms vs 16.90 ms**.
Once tiling has amortized global reads, the kernel is compute-bound on
shared-memory ops; halving global bandwidth has nowhere left to help. The
1.7× win was real but not additive — it was the same underlying mechanism
as tiling, captured worse. **Lesson:** always bench against the best
baseline, not the naive one.

**8×8 register blocking — the register-budget cliff.** Natural next step
from 4×4: scale the per-thread output block to 8×8, with a 128×128
workgroup tile. Hypothesis: 4× the arithmetic intensity per shared-memory
load, so another ~1.5× on top of blocked4. **Lost at every size.** At
1024³, blocked4 was 1.78 ms vs blocked8's 1.96 ms (0.91×); at 2048³,
10.15 ms vs 11.52 ms (0.88×). Most likely cause: 64 floats per thread for
the accumulator exceeds the per-thread register budget on Apple GPUs,
forcing register spill into local memory and tanking effective compute.
Lower workgroup occupancy (16 KB shared per workgroup vs 4 KB) compounded
it. Kept in the codebase as a documented negative result.
**Lesson:** more aggressive is not the same as faster.

**vec4 broke once, then root-caused.** Same blocked4 algorithm, but
issuing 128-bit memory transactions for A and B via `vec4<f32>`.
Standalone bench at 2048³: **1.37× faster than scalar blocked4** — the
best single-kernel measurement in the project. I wired it into
`train.wgsl`, ran the end-to-end parity test, and watched loss diverge to
**88.67 vs WASM's 2.94** — about 30× off.

The standalone bench had used square shapes with WebGPU's `layout: "auto"`,
which inferred read-only-storage bindings to match the WGSL
`var<storage, read>` declaration. Production uses an explicit pipeline
layout declaring `buffer: { type: "storage" }` — read-write. When WGSL
access mode and bind-group-layout type disagree, Chromium/Apple silently
returns wrong data instead of erroring out at validation. The fix was one
line per binding: declare all six as `var<storage, read_write>` in
`train_vec4.wgsl` — the kernel only reads from g0/g1; the decoration just
has to match. With the fix in, end-to-end parity passes at 1.6% drift.
**Lesson — really two of them:** standalone benchmarks miss bugs that
only show up in real training, AND "validation passed" is not the same as
"the data is right." The end-to-end parity test — 50 steps WASM vs 50
steps WebGPU on the same seed, asserting loss drift below 5% — is now the
bar that every kernel integration has to clear.

## Pair-programming with AI as a meta-observation

Most of this work happened in conversation with Claude (which is also the
agent reading and editing this codebase under the convention I'm using). A
few things stood out, worth writing down.

The AI's first answer is often the most aggressive one. Initial proposals
on every kernel were variations of "let's do 8×8 blocking, that's 4× the
reuse" and "f16 just stacks on top of tiled." Both sounded right; both
were wrong; the bench said so. What worked was *always-bench*: take the
suggestion, write the kernel, measure it against the right baseline,
accept or reject based on numbers not narrative.

The most useful part of the conversation, by a wide margin, was the
negative results. Documenting *why* f16-on-top-of-tiled doesn't compound,
*why* 8×8 loses to 4×4, *why* vec4 breaks under a bind-group-layout
mismatch — those are now honest entries on the roadmap. The next person
trying this (human or AI) won't waste a day re-discovering the same dead
ends. Treating negative-result documentation as a first-class deliverable,
equal in weight to the shipped wins, changed how productive the loop felt.

## What's next

Most of the easy wins are done. What's left:

- **Full Flash Attention 2 in WGSL.** Workgroup-cooperative attention with
  tiling and backward recomputation. The biggest remaining lever at
  ctx ≥ 256, where attention's share of step time climbs past 40%.
- **Pre-trained model gallery** — R2-hosted, manifest-driven, so visitors
  can load and continue-train from real checkpoints. Deferred until the
  speed work is fully shipped, so the gallery's implicit promise stays
  honest.
- **Native macOS app** — MLX-Swift + SwiftUI, same `.tinygpt` file format
  both ways, lifts the ceiling into the 7B–30B range on Apple Silicon.

Code, devlog, and roadmap: [github.com/sarthakagrawal927/tinygpt](https://github.com/sarthakagrawal927/tinygpt).
Playground: [tinygpt.sarthakagrawal.dev](https://tinygpt.sarthakagrawal.dev).

— Sarthak Agrawal
