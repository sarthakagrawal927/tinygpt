# tinygpt

A learning project: a **browser-capable TinyGPT** system that can (1) train a tiny
GPT from scratch and (2) adapt a small base model with **LoRA** — built as a correct
Python reference first, then ported to WASM, then accelerated with WebGPU.

This is a learning sandbox, not a deployed fleet product. The goal is **correctness
and understanding**, not impressive output.

## Two browser targets

| Mode               | Purpose                              | Model                                  |
| ------------------ | ------------------------------------ | -------------------------------------- |
| Train from scratch | Learn how GPT training works         | 0.5M–3M param byte-level TinyGPT       |
| LoRA fine-tune     | Learn personalization/style adapting | 5M–15M frozen base + tiny LoRA adapter |

## Build order (do not reorder)

```
Phase 1: Python / PyTorch reference   → python_ref/
Phase 2: TinyGPT from scratch (TS/C++/Rust)
Phase 3: LoRA on the tiny base model  → python_ref/lora.py
Phase 4: Browser WASM implementation  → browser/ + wasm/
Phase 5: WebGPU acceleration          → webgpu/
```

**Never start in the browser.** Build a correct reference, then port. WebGPU is
browser/platform-dependent and HTTPS-only, so the browser build needs a WASM fallback.

## Status

Scaffold only. Every code file under `python_ref/`, `browser/`, `wasm/`, and
`webgpu/` is a **documented stub** — a header describing its role, interface, and
the doc section that specifies it. No runnable code yet.

Recommended next step: implement `python_ref/` (Phase 1) following `docs/model_guide.md`.

Progress tracker: [`MILESTONES.md`](MILESTONES.md) — 10 project milestones, 0/10 done.

## Layout

```
tinygpt/
  configs/        Exact model / training / LoRA specs as JSON
  python_ref/     Phase 1–3 — PyTorch reference (model, dataset, train, sample, lora)
  browser/        Phase 4 — UI, Web Worker, tokenizer, storage, runtime detection
  wasm/           Phase 4 — C++ tensor ops compiled to WebAssembly via Emscripten
  webgpu/         Phase 5 — WGSL compute kernels + JS glue
  data/           Dataset builder + example corpora
  checkpoints/    Saved weights / adapters (gitignored)
  docs/           The full implementation + learning guide
  tests/          Required correctness tests (see tests/README.md)
```

## Docs

- `docs/model_guide.md` — building the TinyGPT model from scratch (Phase 1–2)
- `docs/lora_guide.md` — LoRA fine-tuning (Phase 3)
- `docs/learning_roadmap.md` — the 9-phase + 12-week learning curriculum
- `docs/browser_notes.md` — WASM, Web Workers, OPFS, WebGPU specifics (Phase 4–5)
- `docs/evaluation.md` — required tests, evaluation matrix, memorization checks

The learning curriculum is also mirrored into the `swe-interview-prep` fleet project
(`docs/TINYGPT_LEARNING_PATH.md`) as 19 `ml-*` FSRS-tracked concepts.

## Prerequisites (per phase)

- Phase 1–3: Python 3.10+, PyTorch, NumPy (`python_ref/requirements.txt`)
- Phase 4: Node.js, a bundler (Vite), Emscripten SDK
- Phase 5: a WebGPU-capable browser (Chrome/Edge 113+, feature-detected)
