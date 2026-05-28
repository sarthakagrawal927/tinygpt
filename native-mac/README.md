# TinyGPT — native macOS

Native macOS implementation of TinyGPT, sibling to the browser playground.
Same architecture, same `.tinygpt` file format, runs on Metal via MLX-Swift.

## Status

| Milestone | State | Deliverable |
|---|---|---|
| File format I/O | ✅ ships | `TinyGPTIO` library + `tinygpt inspect/validate` CLI, 12 round-trip tests |
| Model port | ✅ ships | `TinyGPTModel` library with full transformer (MLX-Swift) |
| Weight loader | ✅ ships | `TinyGPTWeightLoader.load()` — browser `.tinygpt` → MLX-Swift model |
| Training loop | ✅ ships | `Trainer` class with compiled train step + AdamW |
| Benchmark CLI | ✅ ships | `tinygpt bench` — measures real GPU throughput vs WebGPU baseline |
| Sample CLI | ✅ ships | `tinygpt sample` — load checkpoint + generate Shakespeare-quality text at ~130 tok/s |
| SwiftUI app | ✅ ships | `TinyGPTApp` — single window, Sample + Train tabs, gallery sidebar, live loss chart |
| Notarized DMG | ⏳ blocked | needs Apple Developer account |

## Build

Requires Xcode (not just Command Line Tools) — MLX-Swift bundles Metal
shaders that only the Xcode build system compiles via SPM.

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

# Tests (file-format suite; runs on CPU stream):
swift test

# CLI executable (must go through xcodebuild for Metal):
xcodebuild -scheme tinygpt -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .xcode-build build

# Run the CLI:
.xcode-build/Build/Products/Debug/tinygpt inspect path/to/model.tinygpt
.xcode-build/Build/Products/Debug/tinygpt bench --preset mega --batch 8
.xcode-build/Build/Products/Debug/tinygpt sample path/to/model.tinygpt --prompt "ROMEO:"

# SwiftUI app — single window, gallery sidebar, Sample + Train tabs.
# Auto-discovers gallery files in ../browser/public/gallery/ at launch.
xcodebuild -scheme TinyGPTApp -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .xcode-build build
.xcode-build/Build/Products/Debug/TinyGPTApp
```

## CLI

```
tinygpt inspect <path>            print manifest + metadata
tinygpt validate <path>           round-trip check (read → encode → byte compare)
tinygpt bench [flags]             training-throughput benchmark vs WebGPU baseline
tinygpt train [flags]             train from scratch and save a checkpoint
tinygpt sample <path> [flags]     load checkpoint, generate text
```

Round-trip example — train on Mac, save, sample back:

```sh
tinygpt train --preset tiny --steps 500 --corpus shakespeare.txt --out my.tinygpt
tinygpt sample my.tinygpt --prompt "ROMEO:" --tokens 100
```

## Measured perf (M5 Pro, 48 GB unified memory)

### Training

| Preset | Params | Time/step | Step/s | Speedup vs WebGPU |
|---|---|---|---:|---:|
| Tiny (4L · d=128) | 842K | 11 ms | 90 | ~9× |
| Huge (12L · d=256 · ctx=256, B=8) | 9.6M | 47 ms | 21 | **15.5×** |
| Mega (24L · d=512 · ctx=512, B=8) | 76M | 372 ms | 2.7 | **24.2×** (browser can't run Mega at all) |

### Sampling

| Preset | Tokens/sec | Notes |
|---|---:|---|
| Tiny | 438 | M5 Pro GPU well under-utilised at this size |
| Huge (gallery Shakespeare model) | 131 | Real Shakespeare quality: matches the browser's output |

**Honest reading of the numbers:**

- The "browser baseline" is 720 ms/step for Huge (measured on the same
  M-series via `train_gallery_one.mjs`) and ~9000 ms/step extrapolated
  for Mega (the browser can't actually run Mega — V8 heap + WebGPU
  buffer caps).
- **At Huge size, the Mac is bandwidth-bound to ~10×, not compute-bound.**
  The model is small enough that kernel-launch overhead dominates.
- **At Mega size and above, MLX-Swift hits its ~25-30× MLX-baseline
  number** — consistent with published Apple Silicon benchmarks.
- Pushing past 30× currently requires either custom Metal kernels
  (weeks of work), bigger models that saturate the GPU (Behemoth-class),
  or M5-specific kernel patterns that MLX hasn't shipped yet.
- For **sampling/inference** specifically, the ANE-routed path can
  plausibly hit 100-500× — that's the next perf push.

The browser ceiling is ~250M params (V8 heap caps out). On this 48 GB
M5 Pro the trainable ceiling is **~3-13 B params** depending on
precision/optimizer state. That's the structural unlock — not just
faster, but able to train completely different model classes.

## The bug worth knowing about

Browser-trained `.tinygpt` files store Linear weights in WASM's `[in, out]`
layout (because the C++ matmul is `y = x @ W`). The manifest claims PyTorch's
`[out, in]` shape, but the BYTES are in WASM order. Reading them at the
manifest shape silently reinterprets bytes incorrectly: forward passes run
without error but produce loss ≈ 3.7 (vs trained ≈ 1.2). Fix: `WeightLoader`
reads each Linear weight at the WASM shape `[in, out]` then transposes to
`[out, in]` — see `TinyGPTWeightLoader.load` and `isLinearWeightName`.

A similar fix may eventually be needed in `python_ref/load_tinygpt.py` —
the comment there says "no transposes needed" but for the WASM-trained
gallery files, that's not quite true (square attention projections work
either way, but asymmetric MLP weights don't). Worth raising upstream.

## Known limitations

1. **`swift build` doesn't compile MLX's Metal library.** Use Xcode or
   `xcodebuild`. The `tinygpt` executable from `swift build` works for
   pure-Foundation subcommands (`inspect`, `validate`) but crashes on
   any MLX operation with "Failed to load the default metallib."
2. **fp16 training works but doesn't speedup as expected.** The dtype
   cast applies, but kernel paths for some MLX ops may not have
   distinct fp16 implementations on M5 Pro yet (MLX 0.31 predates the
   M5 chip ramp). Revisit with newer MLX-Swift.
3. **No SwiftUI shell yet.** The CLI subcommands work end-to-end; the
   visual app is the next milestone.

## What's next (in priority order)

1. **Numerics parity**: get `sample` producing Shakespeare-quality text.
   Test: load gallery model, generate from "ROMEO:", visually verify
   output matches browser-side output.
2. **bf16 path**: bf16 preserves fp32 dynamic range — less risky than
   fp16, may unlock better M5 utilisation.
3. **ANE-routed sampling**: route the forward-only path through
   `mx.compile` with `device=.ane` to chase the 100-500× sampling speedup.
4. **SwiftUI shell**: Setup ⟷ Watch UI, live loss chart, sample panel.
   The CLI does the work; this is the visible artifact.
5. **MLX 0.32+**: when it ships with M5 Neural Accelerator support,
   re-benchmark — the 30→100× jump should be free.

## Module layout

```
Sources/
├── TinyGPTIO/                       Pure Foundation; no MLX dependency
│   ├── Manifest.swift               JSON header schema (config, manifest, etc.)
│   └── TinyGPTFile.swift            Binary reader/writer (fp32 + fp16 layouts)
├── TinyGPTModel/                    MLX-Swift; the model + training
│   ├── ModelConfig.swift            Architecture config (Huge / Mega / custom)
│   ├── TransformerBlock.swift       CausalSelfAttention (mx.fast.attn) + MLP + Block
│   ├── TinyGPTModel.swift           Top-level TinyGPT (embedding + blocks + head)
│   ├── WeightLoader.swift           Load .tinygpt → MLX-Swift module
│   └── Trainer.swift                AdamW + compiled train step
└── TinyGPT/                         Executable
    ├── TinyGPT.swift                CLI entry + subcommand dispatch
    ├── Bench.swift                  `tinygpt bench` — throughput benchmark
    └── Sample.swift                 `tinygpt sample` — load + generate

Tests/
├── TinyGPTIOTests/                  12 file-format round-trip tests (passing)
└── TinyGPTModelTests/               2 compile-only tests (Xcode for full MLX tests)
```
