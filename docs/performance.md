# Performance notes

How fast TinyGPT trains, what has been done to speed it up, and what is left.
All numbers are from an Apple M5 Pro laptop.

## Measuring it

Two benchmarks, so a change can be measured instead of guessed at:

- `python_ref/bench.py` — native training (PyTorch on CUDA / MPS / CPU).
- `tests/bench_wasm.mjs` — the compiled WASM module, which is the browser's
  actual training path, timed from Node.

## The browser path (WebAssembly)

Browser training runs in C++ compiled to WebAssembly, on one thread. Measured
with `bench_wasm.mjs`:

| Build                          | standard (0.37M) | capable (0.48M) |
| ------------------------------ | ---------------: | --------------: |
| baseline (scalar)              |      304 ms/step |     632 ms/step |
| + backward-scratch reuse       |      305 ms/step |     ~640 ms/step |
| + WASM SIMD (`-msimd128`)      |  **191 ms/step** | **391 ms/step** |
| net speed-up                   |        **1.6×**  |       **1.6×**  |

What worked, and what didn't:

- **Allocation reuse — no measurable gain.** Caching the backward pass's scratch
  buffers on the model (instead of allocating ~12 vectors per step) changed
  nothing measurable. `malloc` was not the bottleneck; compute is. The change is
  kept — not allocating in a hot path is still correct hygiene — but it is not a
  speed-up and is not claimed as one.
- **WASM SIMD — ~1.6×.** `-msimd128` lets LLVM autovectorize the matmul,
  layernorm, and attention inner loops (four float32 lanes at a time). WASM SIMD
  is supported in every current browser and costs nothing at deploy time. The
  SIMD build is verified by `tests/smoke_wasm_node.mjs` — it still trains
  correctly (loss falls, greedy generation reproduces the corpus).

Still on the table for the WASM path:

- **Hand-written SIMD intrinsics.** Autovectorization reached 1.6×; explicit
  `wasm_f32x4` intrinsics in the matmul hot loop could push toward 2.5–3×.
- **Threads.** Multi-core training via `SharedArrayBuffer` needs the COOP/COEP
  cross-origin-isolation headers, which a plain GitHub Pages host cannot set.
  Deferred — it is coupled to where the site is hosted.

## The native path, for contrast

`bench.py` on the same laptop trains a 2.7M model at ~10 ms/step on the GPU
(MPS). The browser does a smaller 0.37M model at 191 ms/step. Native is roughly
two orders of magnitude faster per parameter-step. That gap is why anything
past a demo-sized model should be trained locally — and why WebGPU is the real
browser lever.

## Rust?

Considered and set aside. C++ and Rust both compile through LLVM to effectively
the same WebAssembly — the source language is not the bottleneck. Rewriting the
kernels in Rust would be a large change for no speed-up. SIMD and threads are
equally reachable from the current C++/Emscripten setup.

## WebGPU training

The whole training loop now also runs on the GPU. It was built in six verified
stages (`webgpu/`):

1. GPU tensors + matmul forward/backward
2. layernorm, GELU, the elementwise ops
3. causal multi-head attention, forward and backward
4. embeddings, cross-entropy, AdamW
5. `gpu_model.ts` — the orchestrator: a full forward + backward + AdamW loop,
   every tensor resident on the GPU
6. wired into the app as a backend toggle (WASM / WebGPU)

**Correctness** is solid: 24 kernel parity checks against plain-JS references,
the project's overfit gate run on the GPU (loss 5.55 → 0.002 in 150 steps), and
a headless-browser e2e that trains on the WebGPU backend.

### Optimizations done

- **Buffer pool** (`webgpu/tensor.ts`) — per-step activation/gradient buffers
  are returned to a pool and reused; after step 1 a run allocates no buffers.
- **One submit per step** (`webgpu/ops.ts`) — a whole step's dispatches record
  into a single command encoder and submit once, instead of one submit per
  kernel.

Both are real reductions in CPU-side overhead and both keep every parity check
and the overfit gate green.

### Why there is no speed number here — and it matters

WebGPU's speed **cannot be measured in this project's test setup.** The headless
Chromium that runs the e2e exposes a WebGPU adapter whose architecture is
`swiftshader` — Google's *software* renderer. It is a CPU implementation of the
WebGPU API; it never touches a real GPU.

So any headless "WebGPU vs WASM" number is *software-emulated WebGPU vs
SIMD-vectorized WASM* — and WASM wins that, which says nothing about real
hardware. (An earlier version of this file quoted such numbers as a verdict;
that was wrong, and is the reason the buffer-pool and batching optimizations
showed no change — SwiftShader's bottleneck is its own software compute, not
buffer allocation or submit count.)

**To measure the real thing:** open the app in a normal browser on a machine
with a real GPU, pick the WebGPU backend, and read the tokens/sec in the
playground. That is the only valid measurement, and it is not something the
headless CI can do. On a real GPU the matmul-heavy work parallelizes hard;
whether end-to-end training beats WASM depends on how much the small
elementwise kernels' dispatch overhead costs. That number is genuinely unknown
until run on hardware — this doc will not guess it.
