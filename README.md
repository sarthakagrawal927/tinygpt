# tinygpt

A GPT small enough to read in an afternoon. About 0.8M parameters — byte-level,
no dependencies you can't inspect — written so every part can be understood and
is backed by a test.

It does three things: trains a transformer from scratch, fine-tunes one with
LoRA, and runs both in the browser. None of it produces good text; a model this
size can't. The point was to build the whole modern LLM stack at a size where
nothing stays a black box.

It started as a documented scaffold and is now finished — the ten build
milestones in [`MILESTONES.md`](MILESTONES.md) are all done.

## New to this? Read docs/learn.md

If you can write code but have never built a neural network,
[`docs/learn.md`](docs/learn.md) is a guided path through the whole repo: for
each idea it links a good explainer, points at the file here that implements it,
and names the test that proves it. The rest of this README is just the map.

## What's in it

The same model exists at three levels, built in that order:

- `python_ref/` — the PyTorch reference: model, training loop, sampler, LoRA, an
  evaluation harness, and `bench.py` to measure training speed. Read this first;
  it's the clearest.
- `wasm/` — the same maths in C++, every backward pass derived and written by
  hand (there is no autograd engine), compiled to WebAssembly with Emscripten.
- `webgpu/` — one matrix-multiply compute shader in WGSL, checked against the
  WASM version.

On top of that, `browser/` is a small web app. It trains a GPT in a Web Worker
so the page never freezes, draws the loss as it goes, samples from the model,
and saves checkpoints to OPFS so a run survives a refresh. It also detects your
machine and suggests a model size, estimates training time live, can pull a
dataset from Hugging Face, and benchmarks the WebGPU matmul against WASM.

Everything is gated by tests — finite-difference gradient checks on the kernels,
an overfit check on every model, a headless-browser end-to-end run. That was the
method throughout: no layer was trusted until a test pinned it down.

## Build order

It mattered, and the project followed it strictly: PyTorch reference first, then
the C++/WASM port, then WebGPU. A bug in a WASM kernel and a bug in browser glue
look identical from the UI, so the maths was made correct twice — in PyTorch,
then in native C++ — before anything ran in a browser.

## Running it

The Python reference:

```
python -m venv python_ref/.venv && source python_ref/.venv/bin/activate
pip install -r python_ref/requirements.txt
python tests/test_phase1.py                 # the correctness gate
python python_ref/train.py --overfit        # watch a loss curve fall
python python_ref/train.py --data data/examples/tiny-corpus.txt --out checkpoints/base
python python_ref/sample.py --checkpoint checkpoints/base --prompt "A small model "
```

The browser app:

```
bash wasm/build_wasm.sh          # compile the C++ to WebAssembly (needs Emscripten)
cd browser && npm install && npm run dev
```

Open the printed URL and click Start. The C++ kernels can also be checked
without Emscripten — `bash wasm/build_native.sh` builds and tests them with a
normal compiler.

## How big a model can you train

In the browser, small. Training is single-threaded WebAssembly. Measured on an
M5 Pro laptop: a 0.36M model is about 0.4s per step, a 1.3M model about 2.6s — so
a real run of anything past ~0.5M takes ten minutes or more. The app detects
your machine, suggests a size, and shows a live time estimate once training
starts; trust the estimate. Around 1.5M parameters is the practical ceiling
in-browser.

Locally it's a different story. On the same laptop the Python trainer does a
10M model at ~24s per 1,000 steps, a 25M model at ~47s — fine for real
iteration. Run `python python_ref/bench.py` to measure your own machine.
`configs/model.small.json` is a ready ~10.8M config:

```
python python_ref/train.py --model-config configs/model.small.json --data your-text.txt
```

## Training data

There's no bundled dataset. The files in `data/examples/` are short original
prose written for this project; for anything real you supply your own text
(`--data`, or the browser textarea). The browser app and
`data/dataset_builder.py` can also pull an open dataset from the Hugging Face
Hub — see [`data/README.md`](data/README.md).

## Layout

```
tinygpt/
  configs/        model / training / LoRA settings, as JSON
  python_ref/     the PyTorch reference (model, dataset, train, sample, lora, bench)
  wasm/           C++ kernels + a full C++ model, compiled to WebAssembly
  webgpu/         a WGSL matmul compute shader + its JS glue
  browser/        the web app: UI, training Web Worker, tokenizer, storage
  data/           the dataset builder + example corpora
  checkpoints/    saved weights / adapters (gitignored)
  docs/           the learning guide and the per-phase specs
  tests/          the correctness tests (see tests/README.md)
```

## Docs

- [`docs/learn.md`](docs/learn.md) — the guided learning path; start here
- [`docs/notes.md`](docs/notes.md) — what was built and what each experiment showed
- [`docs/model_guide.md`](docs/model_guide.md) — the model, from scratch
- [`docs/lora_guide.md`](docs/lora_guide.md) — LoRA fine-tuning
- [`docs/browser_notes.md`](docs/browser_notes.md) — WASM, Web Workers, OPFS, WebGPU
- [`docs/evaluation.md`](docs/evaluation.md) — the tests and the evaluation matrix
- [`docs/learning_roadmap.md`](docs/learning_roadmap.md) — the phase-by-phase curriculum

## Prerequisites

- Python reference: Python 3.10+, PyTorch, NumPy (`python_ref/requirements.txt`)
- Browser app: Node.js, and the Emscripten SDK to compile the WASM module
- The WebGPU benchmark: a WebGPU-capable browser (Chrome/Edge 113+, Safari 18+)

## License

MIT — see [`LICENSE`](LICENSE).
