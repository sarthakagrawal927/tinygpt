# TinyGPT

A GPT-2-shaped transformer, written from scratch and trained in your browser.
Python reference, hand-written C++/WASM, hand-written WebGPU — the same model
at three levels, with every gradient pinned down by a test.

Live playground: **[tinygpt.sarthakagrawal.dev](https://tinygpt.sarthakagrawal.dev)**
· Devlog: [browser/devlog.html](browser/devlog.html)
· Roadmap: [browser/roadmap.html](browser/roadmap.html)

![TinyGPT playground](browser/public/og-image.png)

## Why this exists

It started as a teaching project — the goal was to build the whole modern LLM
stack at a size where nothing stays a black box. Every backward pass is derived
by hand, every kernel is parity-checked against a reference, no autograd engine
is involved on the C++/WebGPU side.

Somewhere along the way it became a performance project. The interesting work
stopped being "does the maths come out right" and started being "how fast can
this model train inside a browser tab, without lying about the numbers." Most
of what's in [`browser/devlog.html`](browser/devlog.html) is that second half.

## Key measured numbers

All on the same Apple M-series laptop, same model, same seed, same data.
Reproducible from the playground's bench button or `tests/test_webgpu_train.mjs`.

- **9.7× end-to-end speedup** — WASM SIMD takes 6.8 s/step, WebGPU with the
  blocked-4×4 matmul kernel takes 0.7 s/step. Loss drift between the two
  backends after 50 steps: 1.1% (pure float-reorder noise from different
  GPU accumulation order).
- **5.18× kernel speedup at 2048³ matmul** — the size that dominates the
  Mega/Behemoth presets. Naive WebGPU matmul: 47.24 ms. Workgroup-tiled:
  17.23 ms. Tiled + 4×4 register blocking: 9.12 ms.
- **473M-parameter model allocated in a tab** — `-sMEMORY64=1 -sWASM_BIGINT`
  lifts the 4 GB V8 heap ceiling. The same allocation hard-OOMs the 32-bit
  module; on the 64-bit one it allocates cleanly in 3.7 s and takes one
  training step in 82.2 s with `loss 5.78` (the correct initial loss for
  random init).

The full speed-evolution table — scalar → SIMD → threads → WebGPU naive →
WebGPU blocked — lives on the [roadmap](browser/roadmap.html). Each measured
bar is anchored to a number you can reproduce in the playground.

## Architecture in three sentences

`python_ref/` is the PyTorch reference — the clearest version, used as the
oracle when anything else disagrees. `wasm/` is the same maths in C++ with
every backward pass derived by hand, compiled to WebAssembly with Emscripten
(SIMD + pthreads, plus a Memory64 build that lifts the heap ceiling).
`webgpu/` is the whole training loop in WGSL — forward, backward, and AdamW
— every kernel finite-difference checked and parity-tested against the WASM
reference. All three read and write the same `.tinygpt` binary file format,
so a model trained in one path continues training in another.

## What's interesting under the hood

The long-form is in [`browser/devlog.html`](browser/devlog.html). Short version:

- **Memory64 in WebAssembly** lifts the per-tab heap cap from ~4 GB to tens
  of GB. Build script, runtime feature-detect, and a "Behemoth" preset that
  exercises it.
- **A 4×4-register blocked matmul kernel** in WGSL. Workgroup-shared tiling
  (Goto/VandeGeijn 16×16) plus a 4×4 output block per thread held in
  registers, so each shared-memory load gets reused 4× across the
  accumulator. The point where the kernel stops being bandwidth-bound and
  starts being compute-bound.
- **End-to-end parity testing as the only honest bar.** Standalone matmul
  benchmarks lie — they hide bugs that only show up in non-square production
  shapes. The `tests/test_webgpu_train.mjs` driver runs 50 training steps
  under WASM and 50 under WebGPU on the same seed, then asserts loss drift
  is below 5%. Every integration goes through that gate before it counts.

## Negative results — the most valuable lessons came from things that didn't work

This is the part of the project I'd most want a reviewer to look at, because
it's the part most blog posts skip.

- **f16-packed storage on top of tiled matmul** — standalone, packing weights
  as two f16 per u32 was 1.7× faster than naive WebGPU matmul. Stacked on
  top of the tiled kernel, the combined version ran *slower* than plain
  tiled at 2048³: 17.78 ms vs 16.90 ms. Once tiling has amortized the
  global-memory reads, the kernel is compute-bound on shared-memory ops and
  halving global bandwidth has nowhere left to help. **Lesson:** always
  bench an optimization against the *best* baseline, not the naive one.
- **8×8 register blocking** — the natural next step from 4×4, with 4× the
  arithmetic intensity per shared-memory load. Lost at every benchmarked
  size — 0.91× at 1024³, 0.88× at 2048³. Most likely cause: 64 floats per
  thread for the accumulator exceeds the per-thread register budget on
  Apple GPUs, forcing register spill and dropping workgroup occupancy.
  **Lesson:** more aggressive is not always faster.
- **vec4 global loads — broke once, then root-caused.** Wins by 1.37×
  standalone at 2048³, the best single-kernel speedup measured in the
  project. First integration attempt diverged loss to 88.67 vs WASM's 2.94
  — 30× off. Took the end-to-end parity test to catch it; the standalone
  square-shape bench passed cleanly. **Root cause:** the WGSL kernel
  declared `var<storage, read>` for the input buffers, but the shared
  bind-group layout in `ops.ts` declares them as `buffer: { type: "storage" }`
  (read-write). When WGSL access mode doesn't match the layout type,
  Chromium/Apple silently returns wrong data instead of erroring. Fixed by
  declaring all six bindings as `read_write` in `train_vec4.wgsl` — the
  kernel only reads from g0/g1 anyway, the decoration just has to match.
  Now passes parity at 1.6% drift. **Lesson:** standalone benchmarks miss
  bugs that only show up in real training, and "the validation passed" is
  not the same as "the data is right."

The first two are kept in the repo as documented negative results.
The vec4 fix is shipped.

## Tech used

- [PyTorch](https://pytorch.org/) — the reference path
- [Emscripten](https://emscripten.org/) — C++ → WebAssembly (SIMD + pthreads + Memory64)
- [WebGPU](https://www.w3.org/TR/webgpu/) + [WGSL](https://www.w3.org/TR/WGSL/) — the GPU training loop
- [Vite](https://vitejs.dev/) + TypeScript — the playground UI
- [Cloudflare Pages](https://pages.cloudflare.com/) — hosting

## Try it

Open **[tinygpt.sarthakagrawal.dev](https://tinygpt.sarthakagrawal.dev)** and
click Start. It detects your machine, suggests a model size, shows a live
training-time estimate, and saves checkpoints to OPFS so a run survives a
refresh. The WebGPU backend kicks in automatically on Chrome/Edge 113+ and
Safari 18+.

## What's next

- **Pre-trained model gallery** — Cloudflare R2-hosted, manifest-driven; let
  visitors load and continue-train from real checkpoints instead of just the
  one shipped demo. Deferred until the speed work is fully shipped, so the
  gallery's implicit "you can train these too" promise is honest.
- **Full Flash Attention 2** in WGSL — workgroup-cooperative attention with
  tiling and backward recomputation. The biggest remaining lever at ctx ≥ 256.
- **Native macOS app** — MLX-Swift + SwiftUI, same `.tinygpt` file format both
  ways, lifts the model-size ceiling into the 7B–30B range on Apple Silicon.
  See [`docs/shared_vs_native.md`](docs/shared_vs_native.md) for the boundary.

## Repo layout

```
tinygpt/
  python_ref/   PyTorch reference: model, train, sample, LoRA, bench
  wasm/         C++ kernels + a full C++ model, compiled to WebAssembly
  webgpu/       WGSL kernels (forward, backward, AdamW) + JS glue
  browser/      The web app: UI, training Web Worker, tokenizer, storage
  configs/      Model / training / LoRA settings as JSON
  data/         Dataset builder + example corpora
  docs/         The learning guide and the per-phase specs
  tests/        Correctness tests — finite-diff, overfit, end-to-end parity
  native-mac/   (Planned) MLX-Swift macOS app
```

## Build it locally

```
# Python reference
python -m venv python_ref/.venv && source python_ref/.venv/bin/activate
pip install -r python_ref/requirements.txt
python tests/test_phase1.py
python python_ref/train.py --overfit

# Browser app
bash wasm/build_wasm.sh          # needs Emscripten SDK
cd browser && npm install && npm run dev
```

The C++ kernels can also be checked without Emscripten — `bash wasm/build_native.sh`
builds and tests them with a normal compiler. Full deploy notes:
[`docs/deploy.md`](docs/deploy.md).

## Docs

- [`docs/status.md`](docs/status.md) — where the project stands; a review map
- [`docs/learn.md`](docs/learn.md) — guided learning path through the repo
- [`docs/performance.md`](docs/performance.md) — the SIMD and WebGPU performance work
- [`docs/model_guide.md`](docs/model_guide.md) — the model, from scratch
- [`docs/lora_guide.md`](docs/lora_guide.md) — LoRA fine-tuning
- [`docs/shared_vs_native.md`](docs/shared_vs_native.md) — browser vs. native boundary
- [`docs/feature_ideas.md`](docs/feature_ideas.md) — the future-ideas backlog

## License

MIT — see [`LICENSE`](LICENSE). Author: Sarthak Agrawal ([@sarthakagrawal927](https://github.com/sarthakagrawal927)).
