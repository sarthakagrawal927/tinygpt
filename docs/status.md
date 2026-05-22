# Project status

A review-oriented snapshot of where TinyGPT stands. The detailed docs are linked
at the bottom; this page is the map.

TinyGPT is a small GPT you can read end to end — it trains from scratch,
fine-tunes with LoRA, and runs in the browser. The original ten-milestone
project is finished; several things were built on top of it afterward. Repo is
public; everything is on `main`.

## The original project — done

The ten milestones in [`MILESTONES.md`](../MILESTONES.md) are complete and
verified: the PyTorch reference, training, LoRA, the WASM backend, the browser
app, WebGPU matmul, checkpointing, the metrics dashboard, the write-up, and the
public repo. The component-by-component account with the numbers is in
[`notes.md`](notes.md).

## Built since, in this stretch of work

| Area | What | State |
| --- | --- | --- |
| Learning | [`learn.md`](learn.md) — a guided path through the whole repo for a SWE new to AI; links the best external explainer, then the file here, then the test | done |
| Data | Hugging Face dataset loading — a picker in the app + a `dataset_builder.py hf` command, via the public datasets-server API | done |
| UX | Machine detection — the app probes the browser/CPU and recommends a model size; a live training-time ETA | done |
| Perf | WASM SIMD — `-msimd128`, a measured **1.6×** | done |
| Perf | WebGPU training — the full forward+backward+AdamW on the GPU, six staged + verified parts | done; see caveat |
| Perf | Buffer pool + one-submit-per-step for the WebGPU path | done |

## What's verified, and how

| Suite | Covers | Result |
| --- | --- | --- |
| `tests/test_phase1.py` | model, training, sampling | 8/8 |
| `tests/test_lora.py` | LoRA | 6/6 |
| `wasm/build_native.sh` | C++ kernels (finite-diff) + the C++ model overfit gate | pass |
| `tests/smoke_wasm_node.mjs` | the compiled WASM module trains | pass |
| `browser/npm run webgpu-test` | 24 WebGPU kernel parity checks + the GPU overfit gate | pass |
| `browser/npm run e2e` | the whole app in a headless browser | pass |

Everything that can be checked by a machine, is — that was the method throughout.

## Open — worth your attention

- **The real WebGPU training speed is unmeasured.** The headless test
  environment only has a *software* WebGPU adapter (`swiftshader`), so it cannot
  measure GPU performance. An earlier claim that WebGPU training was "~2× slower"
  was measuring software emulation and has been withdrawn. To get the real
  number: run the app in a normal browser on a machine with a real GPU, pick the
  WebGPU backend, and read tokens/sec. Details in [`performance.md`](performance.md).
- **In-browser training is slow regardless of backend** — single-threaded WASM;
  a ~1M model is the practical ceiling. Bigger models belong on the local Python
  trainer (`python_ref/bench.py` measures your machine; `configs/model.small.json`
  is a ready ~10.8M config).
- **The model is tiny by design.** ~0.8M parameters on a few KB of text will
  never produce good prose. Going further would mean pretraining a 5–15M base —
  noted, not done.

## The change trail to review

Seventeen pull requests, all merged to `main`:

- **#1–#2** — the Python reference, LoRA, the WASM backend, the browser app, the
  write-up.
- **#3–#4** — the `learn.md` guided path; the MIT license.
- **#5–#8** — machine detection + model recommendation, `bench.py` + the live
  ETA + the local-training path, Hugging Face data loading, honest in-browser
  timings + browser-compat info + a rewritten README.
- **#9** — WASM SIMD (1.6×).
- **#10–#15** — WebGPU training, built in six verified stages (GPU tensors +
  matmul, layernorm/GELU/elementwise, attention, embeddings/cross-entropy/AdamW,
  the orchestrator, the app integration).
- **#16–#17** — the WebGPU speed write-up, then the buffer pool + the correction
  once the software-adapter issue was found.

## Where the docs are

- [`learn.md`](learn.md) — start here to understand the repo
- [`notes.md`](notes.md) — what each component does and what each experiment showed
- [`performance.md`](performance.md) — the SIMD and WebGPU performance work
- [`model_guide.md`](model_guide.md), [`lora_guide.md`](lora_guide.md),
  [`browser_notes.md`](browser_notes.md), [`evaluation.md`](evaluation.md) —
  per-phase detail
- [`feature_ideas.md`](feature_ideas.md) — interactive-learning backlog
