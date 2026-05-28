# Native macOS app — build plan

The browser playground is the on-ramp. The Mac app is the depth: same
architecture, same `.tinygpt` file format, ~20-30× the training throughput.
This doc translates roadmap lever 20 into a concrete week-by-week build
plan you can act on.

Boundary is already drawn in `docs/shared_vs_native.md` — read that first.
This doc picks up where it leaves off: what to actually build, in what
order, and what each milestone proves.

## Why MLX-Swift, not Python MLX

MLX (Python) is faster to prototype but ships a Python interpreter and
a `~150 MB` MLX dylib stack with every app. MLX-Swift compiles into the
app binary, links MLX statically through SwiftPM, and produces a single
notarizable .app under ~50 MB. The Swift API is a thin wrapper over the
same C++ MLX core — same kernels, same perf, no interpreter tax. The
distribution + UX wins are decisive for a public app.

## MVP scope (what ships in v0.1)

The MVP is "the browser playground, but native." Same UI ideas, same
model, same file format — Metal where WebGPU was.

In:

- SwiftUI shell with the Setup ⟷ Watch two-screen split
- Preset picker (Tiny / Small / Medium / Massive / Behemoth)
  with the same parameter ranges as the browser
- MLX-Swift training loop driving a 12L / d=256 / ctx=256 transformer
- Loss chart with the same hover-inspect behaviour as the browser
- Sample generation with temperature + token count
- `.tinygpt` load and save — interop-tested against the browser
- "Continue training" path (load checkpoint → train N more steps)
- Pre-trained gallery (same 4 corpora as browser, identical file format)

Out (deferred to v0.2+):

- LoRA fine-tuning (lever 19 — lands here, but not MVP)
- Quantized inference (lever 19)
- ANE-accelerated sampling
- Larger presets (Mega / Behemoth-Pro) that exploit the higher ceiling
- Sparkle auto-update
- Opt-in telemetry

Out forever:

- Cross-platform — this is M-series Mac only. Linux / Windows / Intel
  are explicit non-goals. The win is Apple unified memory + Metal +
  MLX; chasing portability erases it.

## File / module layout

```
mac/
├── Package.swift                    # SwiftPM, depends on MLX-Swift
├── Sources/
│   ├── TinyGPTApp/                  # SwiftUI entry point
│   │   ├── TinyGPTApp.swift         # @main, scene
│   │   ├── ContentView.swift        # the two-screen split
│   │   ├── SetupView.swift          # preset picker + start
│   │   ├── WatchView.swift          # loss chart + samples
│   │   └── GalleryView.swift        # pre-trained model loader
│   ├── TinyGPTModel/                # the model — pure MLX-Swift
│   │   ├── Model.swift              # Transformer, parity with python_ref/model.py
│   │   ├── Attention.swift          # multi-head + RoPE
│   │   ├── MLP.swift                # SwiGLU MLP block
│   │   ├── Tokenizer.swift          # byte-level + small BPE
│   │   ├── Optimizer.swift          # AdamW
│   │   └── TrainStep.swift          # forward + backward + AdamW step
│   ├── TinyGPTIO/                   # the file format
│   │   ├── TinyGPTFile.swift        # .tinygpt reader/writer
│   │   ├── Manifest.swift           # JSON header schema
│   │   └── Tensors.swift            # MLXArray ↔ flat-blob conversion
│   └── TinyGPTUI/                   # shared design primitives
│       ├── Theme.swift              # accent / panel / line colors
│       ├── LossChart.swift          # SwiftUI canvas mirror of charts.ts
│       └── Pills.swift              # capability pills (Metal, ANE, etc.)
├── Resources/
│   ├── gallery/                     # bundled .tinygpt files
│   └── AppIcon.appiconset/
├── Tests/
│   ├── ParityTests/                 # browser ↔ Mac round-trip checks
│   │   ├── FileFormatTests.swift    # write → read → bit-identical
│   │   └── NumericsTests.swift      # forward pass matches python_ref
│   └── ModelTests/
│       └── TrainStepTests.swift     # one AdamW step loss-descent
└── README.md
```

Keep `mac/` at the repo root, sibling to `browser/`. Don't merge into
`browser/` — Swift Package Manager wants clean ownership of its directory
tree.

## Milestone sequence

Each milestone is a few days of work and produces a runnable artifact.
Don't skip ahead; the value of TinyGPT-the-product was always
"every milestone is demoable."

### M1 — file format parity (1-2 days)

Goal: a Swift CLI tool that reads a browser-written `.tinygpt` file and
prints the same manifest the browser would print.

What's in:

- `TinyGPTIO` module fully working
- Swift round-trip test: write a file from a hand-built tensor dict,
  reload it, check bitwise equality
- Cross-path test (runs on CI): browser writes Shakespeare gallery model
  → Swift loads it → all tensor shapes and dtypes match the manifest

What's out: any actual model — just file format.

Why this first: the file format is the only hard cross-path contract.
If it's broken, every later milestone is built on sand. Verifying it
in isolation, before any model code, keeps the contract honest.

### M2 — forward-pass numerics parity (3-4 days)

Goal: a Swift CLI that loads a `.tinygpt` file, runs a forward pass on
a single token, and produces logits identical to `python_ref/model.py`
within tight numerics (< 1e-4 max-abs on float32 path).

What's in:

- `TinyGPTModel` module (no training yet — just forward)
- RoPE, multi-head attention, SwiGLU MLP, layernorm, output projection
- Numerics test against `python_ref` reference outputs

What's out: backward pass, training loop, UI.

Why: training is much harder to debug than inference. Get forward right
first; the gradient code can then be checked against autograd one layer
at a time.

### M3 — training loop end-to-end (3-4 days)

Goal: a Swift CLI that takes a corpus + config and trains a Tiny model
for 500 steps, producing a `.tinygpt` checkpoint the browser can load
and continue training from.

What's in:

- Backward pass + AdamW (MLX-Swift's autograd handles the backward;
  AdamW is `mx.optimizers.AdamW` off-the-shelf)
- Batch sampling from a UTF-8 corpus
- Loss-descent test: 500-step run on Shakespeare → final loss within
  3% of the browser's 500-step run on the same corpus
- Train-continue test: browser-trained 1000-step checkpoint loads into
  Swift, trains 1000 more steps, no NaN / no loss explosion

What's out: UI, gallery, samples.

Why this is the milestone that unlocks everything else: once training
works, the rest is plumbing. This is the highest-risk piece — give it
breathing room.

### M4 — SwiftUI shell (2-3 days)

Goal: the app opens, the Setup screen shows, you can pick a preset and
start training, the Watch screen shows the loss chart updating live.

What's in:

- `TinyGPTApp`, `ContentView`, `SetupView`, `WatchView`
- `LossChart` — SwiftUI Canvas with the same x/y semantics as the
  browser chart (incremental + hover-inspect)
- Wire the training loop from M3 to a SwiftUI `@Observable` that
  publishes loss/step
- Sample generation button → output panel with token streaming

What's out: gallery dialog, file open/save UI, LoRA, quantization.

Why: this is the demo. Once it exists, the app is real even if it has
zero non-MVP features.

### M5 — file open / save / gallery (1-2 days)

Goal: you can save a trained checkpoint to disk, reopen it, continue
training. Gallery dialog with the four bundled corpora.

What's in:

- `NSOpenPanel` / `NSSavePanel` for `.tinygpt`
- Bundled gallery models in `Resources/gallery/` (same .bin files as
  the browser uses, just under their original `.tinygpt` extension —
  no CF Pages cache constraint here)
- `GalleryView` mirroring the browser's `#galleryDialog`

Why: ship M5 and the MVP is done. Cut a v0.1 DMG.

### M6 — packaging + distribution (1-2 days)

Goal: a notarized DMG you can drop on a public landing page next to the
browser playground.

What's in:

- Code-sign with a Developer ID Application cert (paid Apple Dev
  account required — gate this milestone on that)
- Notarize via `notarytool` and staple the ticket
- Build a simple landing page (or add to `tinygpt.sarthakagrawal.dev`)
  with the DMG download
- Sparkle setup for auto-update (gate on Apple Dev approval)

Why: this is when "Mac app" stops being a repo subdirectory and starts
being a product.

### v0.2 features (after MVP ships)

In rough priority:

1. **LoRA fine-tuning** (lever 19) — "load a gallery model, paste your
   own text, watch it adapt." This is the headline post-MVP feature.
   MLX-Swift supports LoRA primitively, so most of the work is UI.
2. **Quantized inference** (lever 19) — load int4 / int8 checkpoints
   for sample-only paths. Lets the bundled gallery shrink ~4×.
3. **ANE-accelerated sampling** — MLX-Swift exposes a `useANE` flag
   on selected ops; profile sampling with and without.
4. **Larger presets** — Mega (24L, d=512) and Behemoth-Pro (24L, d=768).
   Browser can't run these; the Mac app's parameter ceiling makes them
   demoable.

## Decisions to make before starting

These are the choices that will be hard to reverse — surface them now so
they're not bikesheds mid-project.

### 1. SwiftUI minimum macOS version

Lean toward macOS 14 (Sonoma, 2023). Anything later restricts the audience
without unlocking much; anything earlier means skipping recent SwiftUI
features (`@Observable`, modern Charts). Picking 14 covers 90%+ of M-series
Mac users.

### 2. Single window vs. document-based

Single window. The browser playground is single-window; matching that
keeps the UX patterns parallel and avoids the document-architecture
overhead (auto-save, version history, etc.) for a project that has
exactly one file type.

### 3. Hand-tuned Metal vs. pure MLX-Swift

Pure MLX-Swift for v0.1. Custom Metal kernels are a v0.3+ optimization —
"we hit the MLX-Swift ceiling, here's the next 1.3×." MLX-Swift itself is
already ~25× the browser; that's the win we ship with.

### 4. Tokenizer parity

Byte-level for MVP (matches browser default). BPE comes later when the
Mac app starts training models too big for byte-level to be practical
(roughly Mega and above). The tokenizer module is small; switching is
not a one-way door.

### 5. Telemetry

None for v0.1. The browser uses opt-in PostHog; the Mac app's first
release should be telemetry-free to keep the trust bar high. Add later
if there's a concrete question only telemetry can answer.

## Open risks

- **Apple Developer account is gated by a paid subscription** ($99/yr).
  Without it, no notarization → no distribution outside dev machines.
  Decide this before starting M6.
- **MLX-Swift is still pre-1.0.** API churn is real; pin a specific
  release in `Package.swift` and budget for one breaking upgrade during
  the build.
- **Numerics parity is the silent risk.** RoPE, layernorm, and the
  attention mask edge cases between MLX-Swift and the browser are where
  divergence will hide. M2 (forward parity) is the milestone that catches
  this — don't skimp on it.
- **You don't yet have a Mac dev environment for SwiftUI.** If Xcode +
  iOS-style debugging is unfamiliar, allocate one extra day for ramp-up
  before M1.

## Estimated total

MVP (M1-M6): ~2-3 weeks of focused work.
v0.2 (LoRA + quant): another ~1-2 weeks.

Don't try to ship MVP + v0.2 together. The browser launch taught the
lesson: small artifacts ship, big ones rot.
