# Milestones

Project milestone tracker for tinygpt. Everything is unchecked — the repo is
currently a scaffold (see `README.md`). Each milestone links to the phase and
docs that specify it; the underlying curriculum is in `docs/learning_roadmap.md`.

- [ ] **1. PyTorch TinyGPT baseline** — the ~0.8M byte-level model runs a correct
  forward pass and matches expected shapes at every layer.
  Phase 1 · `python_ref/model.py` · `docs/model_guide.md`
- [ ] **2. Training from scratch** — AdamW training loop drives loss down,
  overfits a 1–10 KB file, and sampling works.
  Phase 1 · `python_ref/train.py`, `sample.py` · `docs/model_guide.md`
- [ ] **3. LoRA fine-tuning** — frozen base + low-rank adapter trains, saves, and
  reloads; output differs from the base model.
  Phase 3 · `python_ref/lora.py` · `docs/lora_guide.md`
- [ ] **4. Evaluation suite** — required correctness tests plus the
  base / few-shot / LoRA / LoRA+retrieval comparison matrix.
  Phase 9 · `tests/` · `docs/evaluation.md`
- [ ] **5. Browser WASM port** — C++ kernels compiled with Emscripten; training
  runs in a Web Worker without freezing the UI.
  Phase 4 · `browser/`, `wasm/` · `docs/browser_notes.md`
- [ ] **6. WebGPU acceleration prototype** — one WGSL kernel (matmul) correct
  against WASM and measurably faster.
  Phase 5 · `webgpu/` · `docs/browser_notes.md`
- [ ] **7. Checkpointing** — save/resume of weights + optimizer state in Python,
  then OPFS/IndexedDB in the browser; survives a page refresh.
  Phase 1 & 4 · `checkpoints/` · `browser/src/storage.ts`
- [ ] **8. Metrics dashboard** — live train/val loss, tokens/sec, and active
  backend rendered from `TrainingProgress`.
  Phase 4 · `browser/src/charts.ts`
- [ ] **9. Clear write-up** — learning notes explaining every component and what
  each experiment showed.
  Phase 9 · `docs/`
- [ ] **10. Public repo with experiments** — flip this repo to public with an
  experiments log once the milestones above are stable. (Currently private.)

## Progress

**0 / 10 complete** — scaffold stage. Next: milestones 1–2 (the Phase 1 Python
reference in `python_ref/`).
