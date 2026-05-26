# TinyGPT — native macOS app

Status: bootstrapping. Goal: lift the browser-tab ceiling from ~250M params to
7B+ params on Apple Silicon, with a UI that matches the playground's feel.

The browser version stays the on-ramp; this app is for the people who outgrew
it. Both read and write the same `.tinygpt` v2 format so models migrate
freely between them.

---

## Why this exists

The browser playground is wall-bumping on three constraints:

1. **V8 per-tab heap cap (~4 GB)** — anything past ~250M params in fp32 won't
   allocate.
2. **WebGPU buffer-size cap (~2 GB single allocation)** — weight matrices
   eventually outgrow it.
3. **Browser-sandbox overhead** — WebGPU through the browser is ~30-50%
   slower than direct Metal on the same hardware.

Native macOS removes all three. On a 16 GB M-series machine the realistic
ceiling jumps from 250M to ~1B params; on a 64 GB Max it goes to 7B (fp16);
on a 128 GB Ultra to 13-30B. The browser's "loss curve falling" demo becomes
"actually grammatical English generated locally" at this scale.

---

## Stack

- **MLX-Swift** (`https://github.com/ml-explore/mlx-swift`) — Apple's ML
  framework for Apple Silicon. Unified memory (one buffer, GPU+CPU see it),
  optimised attention/layernorm/AdamW, autodiff. The Python `python_ref/`
  reference ports almost line-for-line.
- **SwiftUI** — UI shell mirroring the browser playground's Setup ⟷ Watch
  two-screen flow.
- **Swift Package Manager** — build/dependency management. No Xcode project
  ceremony unless we eventually need entitlements (sandboxing, notarization).

### Why not the alternatives

- **Electron + the existing WebGPU code**: loses ~all the perf benefit (still
  in a browser engine). Quick to ship but no real ceiling-break.
- **Tauri + Rust + ndarray/burn**: cross-platform, but loses the
  M-series-optimised kernels MLX provides for free. Worth revisiting for a
  Windows/Linux port later.
- **Pure SwiftUI + raw Metal**: rewrite the entire kernel set. Months. MLX
  gives us the kernels at native quality immediately.

---

## Roadmap

### Phase 1 — MVP MLX-Swift port (~1 week)

- [ ] Add the MLX-Swift package as a dependency
- [ ] Port `python_ref/model.py` to MLX-Swift (`Sources/TinyGPTCore/Model.swift`)
- [ ] Port `python_ref/train.py` train loop (`Sources/TinyGPTCore/Trainer.swift`)
- [ ] Wire the existing tests as XCTest cases against the MLX implementation
  - Overfit on `the quick brown fox` — loss must collapse to <0.5
  - Checkpoint round-trip reproduces output bytes
- [ ] Basic SwiftUI shell: corpus textarea + preset picker + Start button
- [ ] Stream loss values into a native Swift `Chart` view
- [ ] Sample generation from the trained model

**Exit criterion:** train the same Small preset (~360k params) end-to-end and
generate text. Loss trajectory matches the browser within ~5%.

### Phase 2 — Feature parity with the browser (~1-2 weeks)

- [ ] All 7 size presets including Massive (25M) and a new HUGE preset (~100M)
- [ ] Sample mid-training (matches the live-inference trick from the browser)
- [ ] Loss chart with the same hover-inspect tooltip
- [ ] Run verdict + pre-flight warnings ported
- [ ] `.tinygpt` v2 file format read/write — must produce files identical to
  the browser's output for the same training inputs
- [ ] Dataset loaders: TinyShakespeare from disk, HF datasets via URL
- [ ] LoRA fine-tuning (port `python_ref/lora.py`)

**Exit criterion:** a `.tinygpt` file produced by the browser loads and
continues training in the Mac app, and vice versa. Same Generate output for
the same seed.

### Phase 3 — Native advantages (~2-3 weeks)

- [ ] Background training daemon — close the window without stopping the run.
  Save checkpoints periodically; resume on relaunch.
- [ ] Multi-model session management — list of in-flight + completed runs in
  a sidebar.
- [ ] Memory-resident large models — load a 7B model once, sample many times
  without reloading.
- [ ] FP16 / BF16 training paths via MLX (free with the framework).
- [ ] Flash Attention via MLX's optimised attention op.
- [ ] HUGE preset (~100M params) and MEGA preset (~500M-1B) actually trained
  on real corpora.
- [ ] LoRA fine-tuning of HuggingFace base models (load from local cache).

**Exit criterion:** train a 100M-param model to readable English on
TinyShakespeare-class data, in under 30 minutes on M-series Pro.

### Phase 4 — Polish + ship (~1 week)

- [ ] Code signing + notarization
- [ ] Sparkle for auto-update
- [ ] First-run welcome flow
- [ ] App icon (extend the favicon's loss-curve sigil)
- [ ] Mac App Store assets (optional — direct download is fine for v1)
- [ ] "Download Mac app" CTA on the browser playground footer

---

## Concrete tasks for day 1

1. `swift package init --type executable` in `native-mac/`
2. Add MLX-Swift to `Package.swift` dependencies
3. Stub `Sources/TinyGPTCore/Model.swift` with the model dataclass
4. Stub `Sources/TinyGPTApp/App.swift` with a "Hello, MLX" SwiftUI window
5. Build with `swift build` — make sure the toolchain is set up

The first PR is just "MLX-Swift compiles + a window appears." Then we port
the model in pieces.

---

## Open questions

- **Distribution**: direct DMG download. Simpler than MAS, no sandboxing
  pain. Notarized + signed so it just opens.
- **Telemetry**: opt-in only. Measured tokens/sec across GPU types feeds the
  /roadmap benchmark table — useful data, low ethical cost when opt-in.
- **Cross-platform later?** Windows/Linux ports via Tauri+Rust. Defer.

## Pricing: free.

This is a portfolio piece, not a product. Free + open source is the right
fit for the goals: be hireable, be cited, be seen. Charging would only
narrow the audience and add friction for nothing.

---

## What this unlocks (for the project's stated goal)

The goal is **hireability + visibility**, not revenue. That makes the calculus
clearer:

1. **One coherent technical story**: "I built a real transformer from scratch
   — hand-derived backprop in C++/WASM, WebGPU port, native Mac app with
   MLX, all reading the same file format." That's a story that lands with
   serious engineering hiring managers.
2. **Visible breadth**: browser + native + ML + UI taste. Each piece is solid
   on its own; together they're rare.
3. **A demo people screenshot**: the Mac app training a 1B+ model with the
   loss curve falling live. Very tweetable; very HN-friendly.
4. **A blog post writes itself**: "What I learned building TinyGPT —
   hand-deriving every backward pass." 1500 words, plenty of code snippets,
   plenty of "I got X wrong, here's how I noticed."
5. **A talk possibility**: the project is fertile for "build it from scratch"
   conference talks (Strange Loop, !!Con, papers-we-love adjacent).

### Phase 5 — Launch (after Phase 4 ships)

- [ ] Polish the README — single-page, screenshots, the project's whole
  story top to bottom.
- [ ] Blog post: "I built a transformer from scratch — twice" (browser +
  native). Honest about what failed (tiled matmul wash, swiftshader
  fiasco). Honest about what worked (multi-thread WASM 2×, WebGPU 7×).
- [ ] HN submission: lead with the playground URL; pin the blog post in the
  thread; be in the comments for the first 2 hours.
- [ ] Twitter/X thread mirroring the post.
- [ ] Outreach to one or two relevant newsletters (TLDR, Pragmatic Engineer)
  — only if the HN post lands well.
