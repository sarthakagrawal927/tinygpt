# tinygpt

A learning project: a **browser-capable TinyGPT** system that can (1) train a tiny
GPT from scratch and (2) adapt a small base model with **LoRA** — built as a correct
Python reference first, then ported to WASM, then accelerated with WebGPU.

This is a learning sandbox, not a deployed fleet product. The goal is **correctness
and understanding**, not impressive output.

## New to AI? Start here → [`docs/learn.md`](docs/learn.md)

If you are a software engineer with little or no machine-learning background,
**[`docs/learn.md`](docs/learn.md)** is a guided path through this *entire* repo.
For each concept it links the best free explainer, then the exact file that
implements it here, then the test that proves it works — in order, until none of
it is a black box. Read that first; the rest of this README is the reference map.

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

**All nine implementation milestones are done and verified.** Phases 1–5 — the
Python reference, LoRA, the WASM backend, the browser training app, and the
WebGPU kernel — all work end-to-end:

- `python_ref/` — runnable PyTorch reference (`model`, `dataset`, `train`,
  `sample`, `checkpoint`, `lora`, `evaluate`).
- `wasm/src/` — five C++ kernels + a full C++ TinyGPT, all hand-written
  forward+backward, compiled to WebAssembly with Emscripten.
- `webgpu/` — a WGSL matmul compute kernel, bit-exact vs WASM and ~1.9× faster.
- `browser/` — a Vite app that trains a byte-level GPT from scratch in a Web
  Worker, with a live loss chart, OPFS checkpointing, and a WebGPU benchmark;
  the UI never freezes.

Verified: `tests/` 14/14 Python + 18/18 C++ kernel checks + the C++ model
overfit/checkpoint gates; the compiled module trains from Node; a
headless-browser e2e trains to completion (loss 5.5 → 0.017), runs the WebGPU
benchmark, and confirms the model survives a page refresh.

The detailed write-up — every component, every design decision, and the
concrete result that verified each one — is in [`docs/notes.md`](docs/notes.md).

Progress tracker: [`MILESTONES.md`](MILESTONES.md) — **10/10 milestones done**.
Interactive-feature backlog: [`docs/feature_ideas.md`](docs/feature_ideas.md).

## Running the Python reference

```bash
python -m venv python_ref/.venv && source python_ref/.venv/bin/activate
pip install -r python_ref/requirements.txt

# Phase 1 — train a tiny model from scratch
python tests/test_phase1.py                                   # correctness gate
python python_ref/train.py --data data/examples/tiny-corpus.txt --out checkpoints/base
python python_ref/sample.py --checkpoint checkpoints/base --prompt "A small model "
python python_ref/train.py --overfit                          # built-in smoke run

# Phase 3 — LoRA fine-tune a frozen base onto a different corpus
python tests/test_lora.py                                     # LoRA correctness
python python_ref/lora.py --base checkpoints/base --data data/examples/tiny-corpus-2.txt \
    --out checkpoints/adapter
python python_ref/lora.py --base checkpoints/base --adapter checkpoints/adapter \
    --compare --prompt "The "

# Milestone 4 — the base / few-shot / LoRA / LoRA+retrieval comparison matrix
python python_ref/evaluate.py --base checkpoints/base --adapter checkpoints/adapter

# Phase 4 — verify the WASM C++ kernels + model natively (needs only clang/g++)
bash wasm/build_native.sh
```

## Running the browser app (Phase 4)

```bash
bash wasm/build_wasm.sh          # compile kernels+model to WASM (needs Emscripten)
cd browser && npm install
npm run dev                      # open the printed localhost URL, then "Start training"
```

`node tests/smoke_wasm_node.mjs` verifies the compiled WASM module from Node;
`npm run e2e` (in `browser/`, after `npm run build` + `npm run preview`) drives
the whole app in a headless browser.

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

- `docs/learn.md` — **the guided learning path — start here if you're new to AI**
- `docs/model_guide.md` — building the TinyGPT model from scratch (Phase 1–2)
- `docs/lora_guide.md` — LoRA fine-tuning (Phase 3)
- `docs/learning_roadmap.md` — the 9-phase + 12-week learning curriculum
- `docs/browser_notes.md` — WASM, Web Workers, OPFS, WebGPU specifics (Phase 4–5)
- `docs/evaluation.md` — required tests, evaluation matrix, memorization checks
- `docs/feature_ideas.md` — interactive-learning feature backlog (from 5k+ star repos)
- `docs/notes.md` — learning write-up: every component and what each experiment showed

The learning curriculum is also mirrored into the `swe-interview-prep` fleet project
(`docs/TINYGPT_LEARNING_PATH.md`) as 19 `ml-*` FSRS-tracked concepts.

## Prerequisites (per phase)

- Phase 1–3: Python 3.10+, PyTorch, NumPy (`python_ref/requirements.txt`)
- Phase 4: Node.js, a bundler (Vite), Emscripten SDK
- Phase 5: a WebGPU-capable browser (Chrome/Edge 113+, feature-detected)

## License

MIT — see [`LICENSE`](LICENSE). Fork it, read it, learn from it.
