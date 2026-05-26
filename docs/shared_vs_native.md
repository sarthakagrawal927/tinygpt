# Shared vs. native — how the browser and macOS paths relate

TinyGPT will ship as two binaries: a browser playground (Chromium + M-series
targeted) and a native macOS app (M-series only). They are **separate
implementations** of the same model, optimized for different ceilings. They
deliberately share what's cheap to share (data + UX + words) and don't share
what's expensive to share (kernels + runtime).

The browser is the focus right now. The native app is on the roadmap; this
document exists so that when we start it, the boundary is already drawn.

---

## Shared (must stay in lockstep)

### 1. `.tinygpt` file format (v2)

One binary format for trained models. Both paths read and write it.

- Header + manifest (JSON): config, vocab, optimizer state shape, loss history.
- Body: named tensors in a flat blob.
- Tensor names match `python_ref/model.py` exactly — that's our schema source
  of truth.
- `python_ref/load_tinygpt.py` is the ground-truth round-trip checker —
  anything the browser or Mac app writes must load there.

A `.tinygpt` file written in the browser must train-continue cleanly in the
Mac app, and vice versa. This is the only hard cross-path contract.

### 2. UI patterns + copy

The macOS app uses SwiftUI, not HTML, so no literal code reuse — but the
**design language is shared verbatim**:

- The Setup ⟷ Watch two-screen split.
- Preset names and their parameter ranges (Tiny → Small → Medium → Massive
  → Behemoth).
- Field labels, helper text, the run verdict wording, the pre-flight
  warnings.
- The loss chart's hover-inspect tooltip.
- The favicon / app icon — the loss-curve sigil works at both scales.

Anything user-visible that exists on both sides reads identically. New copy
ships in both places or in neither.

### 3. Docs + brand

- `README.md` is the single landing page for the whole project — both binaries
  link to it.
- `docs/` holds plain-Markdown explainers (`model_guide.md`, `learn.md`,
  `evaluation.md`, etc.) that apply to either path.
- The blog post + HN submission talk about the project as a whole, not as
  two products.

### 4. Python reference

`python_ref/` stays authoritative. Any disagreement in numerics between the
browser, the Mac app, and `python_ref/` is resolved by trusting
`python_ref/`. It exists to keep both ports honest.

---

## Distinct (do not try to share)

### Kernels + runtime

| Surface          | Browser                            | Native macOS                       |
| ---------------- | ---------------------------------- | ---------------------------------- |
| CPU kernels      | C++ → WASM (SIMD + pthreads)       | MLX-Swift (Apple Accelerate)       |
| GPU kernels      | WGSL via WebGPU                    | Metal via MLX-Swift (uses MPS)     |
| Memory model     | Memory64 WASM heap, ~16 GB cap     | Unified memory, machine RAM cap    |
| Threading        | Web Workers + SharedArrayBuffer    | Grand Central Dispatch + MLX async |
| Attention        | Custom WGSL Flash Attention 2      | `mx.fast.scaled_dot_product_attn`  |
| Mixed precision  | WebGPU `shader-f16` extension      | MLX fp16 / bf16 paths              |
| Quantization     | Custom int4 sample-only path       | MLX quantize utilities             |
| File I/O         | `File` / `Blob` / OPFS             | `Foundation` + native disk         |

Trying to share kernel code across these stacks is a dead end — the abstraction
overhead eats the perf benefit on both sides. Each path uses the most
native-feeling tools for its platform.

### Build + distribution

| Surface          | Browser                            | Native macOS                       |
| ---------------- | ---------------------------------- | ---------------------------------- |
| Build            | Emscripten + Vite                  | Swift Package Manager              |
| Distribution     | Cloudflare Pages (a URL)           | Direct DMG download (notarized)    |
| Update           | Refresh the tab                    | Sparkle auto-update                |
| Telemetry        | Opt-in PostHog                     | Opt-in only (TBD)                  |

### What each path is for

- **Browser**: the on-ramp. Zero install. People see a real model train on
  their machine in seconds. Hardware ceiling: ~1B params (Memory64 + Behemoth
  preset) — enough to *prove the thing works*, not enough to train something
  you'd quote in a paper.
- **Native macOS**: the depth. People who hit the browser ceiling and want to
  train a 7B-on-M-Max model. Same UI, same file format, much higher ceiling.

The split is intentional. The browser is "show me," the native app is "now
let me actually do it."

---

## Working rule

When adding a feature, the first question is: which path does this live on?

- **Either path alone**: ship it there only. Don't pre-build the other side.
- **Both paths**: ship the file-format/UX/docs piece *first* (shared) so the
  contract is in place, then implement each side.
- **Shared piece**: changes to `.tinygpt` format, preset names/ranges, or
  user-visible copy go through both paths or neither. Don't drift these.
