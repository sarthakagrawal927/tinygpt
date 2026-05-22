# Learning notes — what was built and what each experiment showed

This is the milestone-9 write-up: a plain-language tour of every component of
TinyGPT and the concrete result that proved each one works. The numbers below
are from actual runs in this repo — reproduce them with the commands in
`README.md`.

The guiding rule throughout: **never trust a component until a test pins it
down.** Every layer of the stack has a check that fails loudly if the layer is
wrong.

---

## 1. The model — a 0.8M byte-level GPT

`python_ref/model.py` builds the smallest thing that is still a real GPT:

- **Byte-level tokenizer.** Every byte is a token, so `vocab_size = 256`. No
  BPE, no merge tables — one less thing that can be subtly wrong.
- **Token + position embeddings.** A token id looks up a `d_model`-vector;
  the position adds a second vector so the model knows *where* a token sits.
- **Pre-LayerNorm transformer blocks.** Each block is
  `x = x + attn(norm(x))` then `x = x + mlp(norm(x))`. Normalizing *before* the
  sublayer (rather than after) keeps the residual stream clean and is markedly
  easier to train.
- **Causal self-attention.** `softmax(QKᵀ / √d) · V`, with a mask that forbids
  a position from attending to the future. This is the only place tokens
  exchange information.
- **MLP.** `Linear → GELU → Linear`, widening to `4·d_model` and back. This is
  where most parameters and most per-token "thinking" live.
- **Tied output head.** The logits are `x · Eᵀ`, reusing the token-embedding
  matrix `E`. Fewer parameters, and it usually helps small models.

**What the experiments showed**

| Check | Result | Why it matters |
| --- | --- | --- |
| Parameter count | **842,496** | Matches the ~0.8M target in the config |
| Random-model loss | **5.56** vs `ln(256) = 5.545` | An untrained model is exactly as good as guessing uniformly — the loss has no bug |
| Layer shapes | all pass | Every tensor is the shape the math expects |
| Gradient check | `gradcheck` passes | The backward pass (PyTorch autograd here) matches finite differences |

A subtle bug surfaced here and is worth recording: an early loss-sanity test
fed the model *same-position* targets. Because residual connections + tied
embeddings leave the input token's own embedding in the output, the untrained
model could "cheat" and score **below** `ln(256)`. Real training uses *shifted*
next-token targets, which are genuinely unpredictable on random data — and then
the loss sits exactly at `ln(256)`. Lesson: a sanity test can lie if the task
it poses is easier than the real one.

---

## 2. Training — and the shape of a loss curve

`python_ref/train.py` is a standard loop: sample a batch, forward, cross-entropy
loss, backward, clip gradients, AdamW step. Weight decay is applied only to
matrices and embeddings, never to biases or LayerNorm gains — decaying a bias
toward zero regularizes nothing.

**What the experiments showed**

- **Tiny overfit** (the single most important test): on a few KB of repeated
  text the loss fell from **5.53 → 0.015**. If a tiny model *cannot* overfit a
  tiny dataset, the model, the backward pass, or the data pipeline is broken —
  and scaling a broken model only wastes time.
- **A real run** on `data/examples/tiny-corpus.txt` (~3 KB): train loss fell
  `5.58 → ~1.0`, while **validation loss bottomed at ~2.50 and then rose**.
  That divergence is not a failure — it is overfitting, drawn live. A 0.8M
  model has more than enough capacity to memorize 3 KB, so after a point it
  improves on text it has seen at the expense of text it has not.
- **Checkpoint reload** reproduced the loss bit-for-bit, and **resuming**
  continued the loss curve smoothly — because the AdamW moments (`m`, `v`) are
  saved too. Without them, resume restarts the moments from zero and the curve
  visibly kinks.

---

## 3. LoRA — adapting a frozen model cheaply

`python_ref/lora.py` replaces a linear layer `y = xW` with
`y = xW + (α/r)·xAB`, where `W` is frozen and only the small matrices `A`
(`d_in×r`) and `B` (`r×d_out`) train. `B` starts at zeros, so at step 0 the
adapter contributes nothing and the model is *exactly* the base model.

The subtle part is the backward pass: freezing `W` does **not** mean stopping
gradients through the layer. `xW` stays in the graph, so gradient still flows to
`x` and reaches LoRA adapters in lower layers. Detaching frozen layers is a
classic bug that silently starves the lower adapters.

**What the experiments showed**

- The rank-4 adapter on the 0.8M model is **8,192 trainable parameters —
  0.96%** of the total. That ratio is the whole point of LoRA.
- Fine-tuning on a different-style corpus drove the adapter loss **4.71 → 2.25**.
- **Step-0 identity**: with `B = 0`, base+LoRA output is bit-identical to the
  base — the test confirms it.
- **Frozen-base gradient flow**: after a backward pass the base weights have
  no gradient, while the *first* block's adapter does — proving gradient
  traversed the frozen upper blocks to get there.
- Base vs base+LoRA outputs visibly differ, and adapter save/reload round-trips
  exactly.

LoRA teaches **style**, not facts. On a 0.8M model the "style" is crude, but
the mechanism is correct and that is what milestone 3 asked for.

---

## 4. The WASM port — re-deriving every backward by hand

Phase 4 re-implements the compute in C++ (`wasm/src/`) with **no autograd**.
Each kernel — `matmul`, `layernorm`, `attention`, `adamw` — carries its own
hand-written forward *and* backward. `model.cpp` assembles them into a full
TinyGPT (embeddings, GELU MLP, tied head, cross-entropy).

Writing backward passes by hand is error-prone, so every one is checked by a
**finite-difference gradient check**: perturb each input by ±h, measure the
change in a scalar loss, and compare to the analytic gradient.

**What the experiments showed**

- **18/18 kernel checks pass.** One lesson came for free: the attention weight
  gradients first appeared to fail at 4–7% error. The kernels were correct —
  the *test metric* was wrong. A per-element relative error with a fragile
  denominator inflates noise on the small, near-cancelling gradients the
  softmax produces. Switching to the standard metric (error normalized by the
  buffer's largest gradient) dropped the error to 6e-4. A test can be wrong
  even when the code is right.
- **The C++ model overfits**: loss `5.56 → 0.03` on repeated text — the same
  gate as Phase 1, now proving the entire hand-written backward chain.
- The compiled WASM module trains identically when driven from Node, so the
  JS↔WASM boundary carries the data correctly.

---

## 5. WebGPU — the same matmul, on the GPU

`webgpu/matmul.wgsl` is a compute shader: one GPU invocation computes one output
element of `C = A·B`. `webgpu/kernels.ts` sets up the device, buffers, and
pipeline, and benchmarks the result against the WASM matmul.

**What the experiments showed**

- **Bit-exact parity** with the WASM kernel (max error `0.0`). Both accumulate
  the `k` sum in the same order with the same float32 rounding, so they produce
  *identical* results — the strongest possible "correct against WASM".
- **~1.9× faster** on a 384×384 matmul, even in a headless browser. On real
  GPU hardware the gap is far larger; the point of the milestone is "correct
  first, then measurably faster", and both halves hold.

---

## 6. The browser app — training without freezing the UI

`browser/` is a Vite + TypeScript app. The crucial architecture decision: the
**main thread only does UI**. All model math runs in a Web Worker
(`worker.ts`), which calls the WASM module. The training loop runs in small
chunks and yields between them, so pause/stop messages are handled promptly and
the page never stutters.

**What the experiments showed**

- A headless-browser end-to-end test trains a model to completion *in the
  Worker*, loss **5.5 → 0.017**, then samples from it — with **zero console or
  page errors**.
- **Checkpointing**: the WASM model serializes its weights + AdamW moments +
  step; the browser writes that blob to OPFS. After a page refresh the trained
  model and its loss chart are restored and still generate — milestone 7's
  "survives a refresh", demonstrated.

---

## 7. Evaluation — comparing honestly

`python_ref/evaluate.py` produces the four conditions `docs/evaluation.md`
requires for every held-out prompt: **base**, **base + few-shot prompt**,
**base + LoRA**, **base + LoRA + retrieval** (a tiny trigram retriever stands in
for a vector store). It also runs the **memorization check** — feed a training
prefix back in and measure the longest verbatim continuation.

**What the experiments showed**

- All four conditions render side by side and clearly differ. The retrieval
  condition visibly drags the output toward the retrieved passage.
- Memorization: the longest verbatim continuation was **8 characters** — well
  under the copying-risk threshold. The tiny model learned patterns, not a
  photographic copy of the corpus.

The honest caveat: a 0.8M byte-level model on a few KB of text produces rough
prose in every condition. The deliverable is the **harness** — a reproducible,
apples-to-apples comparison — not the quality of the text.

---

## 8. What this project is and is not

It **is** a correct, end-to-end, byte-by-byte implementation of the modern LLM
stack: a GPT, trained with AdamW, adapted with LoRA, ported to hand-written
WASM kernels, accelerated with a WebGPU kernel, and wrapped in a browser app
that trains without freezing — every layer gated by a test.

It is **not** a capable language model. 0.8M parameters on kilobytes of text
cannot reason or be reliably factual; LoRA here moves tone and vocabulary, not
truth. That limit is by design — the goal was **understanding the machinery**,
and the machinery is now understood, line by line.

### If this were taken further

- Pretrain a 5–15M base on megabytes of text so LoRA adaptation produces
  legible style, not just legible *mechanism*.
- Port more kernels to WebGPU (attention, the optimizer) and add tiled,
  shared-memory matmul.
- Add the interactive teaching surfaces sketched in `docs/feature_ideas.md`:
  an attention visualizer, a tokenizer playground, a semantic-zoom diagram.
