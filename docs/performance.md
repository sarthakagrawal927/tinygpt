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

**Speed is not there yet, and this is the honest part.** Measured in headless
Chromium, the same config trained on each backend:

| Model | WASM (SIMD) | WebGPU      | WebGPU vs WASM |
| ----- | ----------- | ----------- | -------------- |
| 0.07M | 24,858 tok/s | 11,005 tok/s | 0.45× (slower) |
| 0.36M |  4,769 tok/s |  2,691 tok/s | 0.58× (slower) |

WebGPU training is currently **~2× slower** than the WASM path, not faster. The
implementation is naive about memory: every step allocates fresh GPU buffers for
every activation and gradient and frees them at the end of the step (the `keep`
/ `freeScratch` pattern in `gpu_model.ts`), and it downloads the loss every
step, which forces a full GPU sync. That per-step buffer churn dominates — the
kernels are fast, the orchestration is what's slow.

One encouraging sign: WebGPU's relative speed *improves* with model size
(0.45× → 0.58×), because larger tensors mean more compute per dispatch to
amortize the fixed overhead against. The crossover — where the GPU's
parallelism finally wins — is past these sizes.

The optimization that unlocks the real speed-up: a **buffer pool** — allocate
the activation/gradient buffers once for a given batch shape and reuse them
across steps (exactly what `wasm/src/model.cpp` already does on the CPU side),
and stop syncing every step. That is the clear next piece of work; the kernels
and the orchestration are correct and in place for it.
