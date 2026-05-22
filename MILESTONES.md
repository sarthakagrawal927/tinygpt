# Milestones

Project milestone tracker for tinygpt. Each milestone links to the phase and
docs that specify it; the underlying curriculum is in `docs/learning_roadmap.md`.
Interactive-feature backlog: `docs/feature_ideas.md`.

- [x] **1. PyTorch TinyGPT baseline** — the ~0.8M byte-level model runs a correct
  forward pass and matches expected shapes at every layer.
  Phase 1 · `python_ref/model.py` · `docs/model_guide.md`
  _Done: 842,496 params; shape, loss-sanity (5.56 ≈ ln 256) and gradient-check tests pass._
- [x] **2. Training from scratch** — AdamW training loop drives loss down,
  overfits a 1–10 KB file, and sampling works.
  Phase 1 · `python_ref/train.py`, `sample.py` · `docs/model_guide.md`
  _Done: tiny-overfit drives loss 5.53 → 0.017; `train.py`/`sample.py` verified end-to-end._
- [x] **3. LoRA fine-tuning** — frozen base + low-rank adapter trains, saves, and
  reloads; output differs from the base model.
  Phase 3 · `python_ref/lora.py` · `docs/lora_guide.md`
  _Done: rank-4 adapter (8,192 params, 0.96% of total) trains (loss 4.71 → 2.25),
  saves adapter-only + reloads; base-vs-LoRA output differs. Verified on the 0.8M
  base — a 5–15M base (roadmap step 4) is still future work._
- [x] **4. Evaluation suite** — required correctness tests plus the
  base / few-shot / LoRA / LoRA+retrieval comparison matrix.
  Phase 9 · `tests/` · `docs/evaluation.md`
  _Done: `tests/test_phase1.py` (8/8) + `tests/test_lora.py` (6/6) cover the
  required correctness tests; `python_ref/evaluate.py` produces the four-way
  comparison matrix and the memorization check._
- [x] **5. Browser WASM port** — C++ kernels compiled with Emscripten; training
  runs in a Web Worker without freezing the UI.
  Phase 4 · `browser/`, `wasm/` · `docs/browser_notes.md`
  _Done: five C++ kernels + a full C++ TinyGPT (`wasm/src/`), all hand-written
  backward, verified natively (kernels 18/18 finite-diff; model overfits 5.56 →
  0.03). Compiled to WASM and driven by a Web Worker; headless-browser e2e trains
  to completion (loss 5.5 → 0.017) with the UI thread free and zero errors._
- [x] **6. WebGPU acceleration prototype** — one WGSL kernel (matmul) correct
  against WASM and measurably faster.
  Phase 5 · `webgpu/` · `docs/browser_notes.md`
  _Done: `webgpu/matmul.wgsl` compute kernel + `kernels.ts` glue; the in-app
  benchmark checks parity vs the WASM matmul (bit-exact) and reports the
  speed-up. Headless-browser e2e: parity OK, ~1.9× faster on a 384² matmul._
- [x] **7. Checkpointing** — save/resume of weights + optimizer state in Python,
  then OPFS/IndexedDB in the browser; survives a page refresh.
  Phase 1 & 4 · `checkpoints/` · `browser/src/storage.ts`
  _Done: Python save/resume (`python_ref/checkpoint.py`); the WASM model
  serialises weights + AdamW moments + step (`tg_export/import_state`); the
  browser persists that blob to OPFS. Headless e2e: the trained model and its
  chart are restored after a page refresh and still generate._
- [x] **8. Metrics dashboard** — live train/val loss, tokens/sec, and active
  backend rendered from `TrainingProgress`.
  Phase 4 · `browser/src/charts.ts`
  _Done: the browser app renders a live train/val loss chart plus step,
  tokens/sec and backend, all driven by `TrainingProgress` from the Worker._
- [x] **9. Clear write-up** — learning notes explaining every component and what
  each experiment showed.
  Phase 9 · `docs/`
  _Done: `docs/notes.md` — a component-by-component tour with the concrete
  result that verified each one._
- [x] **10. Public repo with experiments** — flip this repo to public with an
  experiments log once the milestones above are stable.
  _Done: all milestones merged to `main`; `docs/notes.md` is the detailed
  experiments write-up; the repo is public._

## Progress

**10 / 10 complete.** Every milestone is done, verified end-to-end, and merged
to `main`. The detailed write-up is in `docs/notes.md`.
