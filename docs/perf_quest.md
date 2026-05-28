# The browser-frontier performance quest

A working document for the in-flight push to give users on the latest Chrome
every GPU acceleration the platform exposes — without ever degrading model
quality. Started during the HN-launch prep thread, paused mid-flight after
the foundation landed.

## The product principle

> "The best possible performance for users on the latest browser, graceful
> degradation for everyone else, and we inform them about which path is
> active." — user direction

Concretely, every fast path in this push obeys three rules:

1. **Feature-detected at startup.** Use it only if the browser exposes it.
2. **Verified against the f32 reference before activating.** If the path
   doesn't match the slow-but-correct path's loss curve within 1% over a
   500-step Shakespeare run, it's disabled for the session. Silently. The
   user gets the slower correct path, never broken output.
3. **Surfaced in the capability pills** (`+f16`, `+subgroups`, `+coop-matrix`,
   `+WebNN`) so the user can see which acceleration is running, and click
   each pill for an explainer.

The non-negotiable: **no quality regression, ever.** Speed only counts if
it preserves loss.

## The four levers

| Lever | Status | Detection | Realistic gain | Risk |
|---|---|---|---|---|
| **Storage-f16 matmul** (existing `matmul_tiled_f16.wgsl`) | scaffolded; not wired into prod path yet | always available (uses `pack2x16float`) | ~1.5×, all users | medium — needs weight pre-packing, train.wgsl bind-layout adaptation |
| **shader-f16 full compute** | not started | `device.features.has("shader-f16")` — Chrome 121+ stable | additional ~1.3-1.5× | medium — precision drift; needs gate per #94 |
| **Cooperative matrix** (`enable chromium_experimental_subgroup_matrix`) | probe-compile detection landed | probe-compile a trial shader | ~3× NVIDIA · ~1.3× Apple | high — experimental WGSL extension, evolving API, sparse docs |
| **WebNN inference (sampling only)** | scaffolded; probe lands | `navigator.ml` + `createContext({deviceType: "gpu"\|"npu"})` | ~3-5× on Apple/Windows | high — new code path; training stays WebGPU |

Each lever is independently shippable — failure of one doesn't block another.

## What landed in commit `28f2533`

- `webgpu/tensor.ts` — `createGpuContext` opportunistically requests
  `subgroups` + `shader-f16` + `timestamp-query`; pulls adapter `info`
  (vendor / architecture / device); probes cooperative-matrix via a trial
  shader compile inside the worker. New `GpuCapabilities` interface.
  Separate `probeWebNN()` for the navigator.ml side.
- `browser/src/runtime_detect.ts` — `Capabilities` extended with
  `gpuFeatures: GpuSubFeatures` and `webnnPresent`. Cheap adapter-features
  probe runs at startup (no device creation); deep cooperative-matrix probe
  later from the worker.
- `browser/src/main.ts` + `index.astro` — `+f16` / `+subgroups` / `+WebNN`
  pills in `#caps` when each is active. `+coop-matrix` slot updated
  post-init via `window.__tgUpdateGpuAccelPills` (the worker→main bridge).
  Dismissible "Power user?" yellow nudge for Chromium users without the
  `chrome://flags#enable-unsafe-webgpu` flag.
- `browser/src/explainers.ts` — four new pill explainers: `shaderF16`,
  `subgroups`, `coopMatrix`, `webnn`. Each links to spec / docs.

## What's queued (in execution order)

1. **#90 — Wire storage-f16 matmul into the production training path.**
   The existing kernel `matmul_tiled_f16.wgsl` is a standalone benchmark
   runner; the production path is `train.wgsl` with a six-storage-buffer
   bind layout. Work: mirror the `train_vec4.wgsl` approach — write
   `train_f16.wgsl` with the same bind layout but `array<u32>` inputs
   (packed f16), add `matmul_blocked_f16` entries to `ops.ts`, pre-pack
   weight `GpuTensor`s to f16 at upload. Verify against f32 over 500
   Shakespeare steps; if loss diverges, disable.
2. **#91 — `shader-f16` full-compute matmul.** New WGSL with `enable f16;`
   and `f16` accumulators. Gated on `capabilities.shaderF16`. Same
   numerics gate.
3. **#92 — Cooperative-matrix kernel.** WGSL using
   `enable chromium_experimental_subgroup_matrix` and `subgroup_matrix_multiply`.
   Probe-compile detection already lives in `tensor.ts`; once it succeeds,
   wire the path + the worker→main bridge so `+coop-matrix` appears in the
   pill cluster. Most uncertain in scope — debugging without docs.
4. **#93 — WebNN inference for sampling.** Forward-only path using
   `MLGraphBuilder`. Training stays on WebGPU. Routes to CoreML/DirectML/ANE
   under the hood. Falls back to WebGPU sampling silently if WebNN context
   creation fails.
5. **#94 — The numerics gate itself.** A `verifyAcceleratorPath()` helper
   that runs the 500-step Shakespeare check on each enabled path at first
   use; caches the verdict; falls back automatically. Document measured
   deltas in `docs/precision.md` (to be created).
6. **#85 — Retrain the gallery** (3 Huge models) sequentially under
   `caffeinate -i` on the now-faster path. ~2.5 hr expected at the new
   throughput.
7. **#87 — E2E + deploy.** Add gallery checks to `e2e_chrome.mjs`,
   verify on local then live, push.

## Gallery v1 state

- `browser/public/gallery/shakespeare.tinygpt` (18.4 MB fp16) — landed.
- `browser/public/gallery/manifest.json` — landed with one entry. Will be
  rewritten by `browser/finalize_gallery.mjs` once the 3 new models exist.
- `browser/train_gallery_one.mjs` — Playwright trainer per corpus,
  validated against TinyStories + Recipes (both successfully started, then
  hit the wall cap due to parallel-on-one-GPU contention; see lesson below).
- `browser/finalize_gallery.mjs` — fp16-packs canonical checkpoints and
  assembles the unified manifest. Verified end-to-end on Shakespeare.
- Corpora pre-fetched and on disk at `/tmp/tinygpt-corpora/`
  (`tinystories.txt`, `code.txt`, `recipes.txt`, ~1.1 MB each).

The gallery dialog UI ships now — manifest-driven, ready to show the other
three models the moment they're trained.

## The lesson behind why this took longer than expected

The parallel-on-one-GPU training experiment (3 Playwright Chromium instances
training concurrently against the same M-series GPU) was a strategic
mistake. The throughput data:

- TinyStories solo: ~100 steps/min at start
- All three parallel: ~17-30 steps/min each (≈3-5× slowdown)
- 15-minute gaps showed up in every log when the system slept

After 2 hours: 40-46% of each run completed, **none downloaded**, work
discarded. The right move was sequential under `caffeinate -i`. Three
sequential Huge runs at ~75 min each = 3.75 hr total; that beats the
"parallel + sleep + GPU contention + zero output" actual cost.

The takeaway baked into the queue: **#85 explicitly says sequential +
caffeinate.**

## Why this sequencing is right

The user's launch direction is "lock down everything before HN, no waves."
That gates publishing on every lever working. Sequencing within that:

- The **foundation** (capability detection + UI) is independent of the
  kernels and ships first. Done.
- The **storage-f16 path** is the lowest-risk lever and benefits every user
  (no flag needed). Do it next so the rest of the kernel work has a faster
  baseline to build on.
- **shader-f16** stacks on top of storage-f16 cheaply once the bind-layout
  refactor is done.
- **Cooperative matrix** and **WebNN** are the high-risk / high-wow levers.
  Do them last so the launch isn't gated on debugging an experimental
  WGSL extension.
- **Numerics gate (#94)** runs alongside each lever — it's not the last
  step, it's the always-on invariant.
- **Retrain (#85)** waits for the fast path because that's where the wall
  time pays off — and because it's the visible artifact of "we made
  training faster."
- **Deploy (#87)** is last and is mostly mechanical once the rest is green.

## Cross-references

- `browser/src/pages/roadmap.astro` lever 21 documents the same frontier
  with public-facing framing (what users see).
- `docs/lessons.md` carries the parallel-training-on-one-GPU lesson (added
  as part of session retrospective work).
- `docs/decision_log.md` will get a new entry pinning the "no quality
  regression" rule and the parallel-training pivot.
- The capability detection layer lives in `webgpu/tensor.ts`; capability UI
  + nudge in `browser/src/main.ts`; pill explainers in
  `browser/src/explainers.ts`.

This doc is the picking-up-where-we-left-off page for whoever resumes the
quest. Keep it terse; update it as levers ship.
