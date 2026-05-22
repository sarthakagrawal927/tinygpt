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

## WebGPU — the real ceiling

Today WebGPU is used only for the standalone matmul benchmark; training runs
entirely in WASM. Moving training onto the GPU is the change that would make a
1M+ parameter model fast in the browser.

The work, in the order `docs/browser_notes.md` lays out:

1. matmul (done — `webgpu/matmul.wgsl`)
2. linear backward
3. attention scores → softmax → value aggregation
4. layernorm
5. AdamW

The thing that actually makes it fast is not porting the kernels but keeping
the tensors resident in GPU buffers between ops — round-tripping every
intermediate through JavaScript would erase the gain. On real GPU hardware the
matmul-heavy parts run 10×+ faster than the WASM kernels; end-to-end the gain is
smaller (the small elementwise ops have fixed dispatch overhead) but still
large for the model sizes here.
