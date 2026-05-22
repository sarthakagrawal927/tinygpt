# tests/ — required correctness tests

Do not skip these. They are the difference between a model that works and a
model that looks like it works.

Runnable suites (plain `python`, no pytest needed — pytest also works):

```
python tests/test_phase1.py        # Phase 1: tokenizer, shapes, loss, overfit, gradcheck
python tests/test_lora.py          # Phase 3: adapter step-0, frozen grads, roundtrip
bash   wasm/build_native.sh        # Phase 4: C++ kernels (finite-diff) + model overfit gate
node   tests/smoke_wasm_node.mjs   # Phase 4: the compiled WASM module trains, from Node
cd browser && npm run e2e          # Phase 4: headless-browser end-to-end (build+preview first)
```

| Test                | Purpose                                      |
| ------------------- | -------------------------------------------- |
| Tokenizer roundtrip | bytes → text → bytes is lossless             |
| Shape tests         | every layer returns the expected shape       |
| Loss sanity         | a random model's loss is near `ln(256) ≈ 5.54` |
| Tiny overfit        | the model overfits 1–10 KB of repeated text  |
| Gradient check      | finite-difference check on a tiny layer      |
| PyTorch parity      | ported forward output matches the PyTorch reference |
| Checkpoint reload   | same loss after save + reload                |
| Sampling fixed seed | generation is deterministic for a fixed seed |
| Browser refresh     | a run resumes correctly after a page reload  |

## The one that matters most

**Can it overfit a tiny repeated dataset?**

If a tiny model cannot drive loss down on 1–10 KB of repeated text, something
is wrong in the model, the backprop, or the data pipeline. Fix that before
scaling anything — scaling a broken model just wastes time.

## LoRA-specific

| Test               | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| Adapter step-0     | with `B = 0`, base + LoRA output == base output      |
| Frozen-base grads  | gradients still flow THROUGH frozen layers to LoRA   |
| Memorization       | prefix → continuation does not reproduce training text verbatim |
| Baseline beat      | LoRA actually beats few-shot prompting               |

## Backend parity (Phase 4–5)

| Test               | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| WASM vs PyTorch    | WASM forward matches the Python reference            |
| WebGPU vs WASM     | each WebGPU kernel matches WASM within tolerance     |

See `../docs/evaluation.md` for the full evaluation matrix.
