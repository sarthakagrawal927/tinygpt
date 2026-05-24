# TinyGPT — a detailed write-up

How a documented scaffold became a verified, end-to-end implementation of the
modern LLM stack: a GPT trained from scratch, adapted with LoRA, ported to
hand-written WebAssembly kernels, accelerated with a WebGPU shader, and wrapped
in a browser app that trains without freezing the UI.

Every number below comes from an actual run in this repo. Reproduce them with
the commands in [§12](#12-how-to-reproduce-everything).

---

## 1. The point of the project

TinyGPT is a **learning project**, not a product. The goal was never impressive
text — a 0.8M-parameter model on a few kilobytes of data cannot produce that.
The goal was to understand the whole stack *by building every layer of it* and
**proving each layer correct before moving on**.

One rule shaped everything:

> **Never trust a component until a test pins it down.** If a layer can be
> wrong, there is a check that fails loudly when it is.

That rule is why this write-up can quote exact numbers for every claim.

### The build order

The phases were built strictly bottom-up, and never reordered:

```
Phase 1   PyTorch reference        python_ref/
Phase 2   from-scratch kernels     wasm/src/ (C++)
Phase 3   LoRA on the tiny base    python_ref/lora.py
Phase 4   browser WASM app         browser/ + wasm/
Phase 5   WebGPU acceleration      webgpu/
```

The discipline here is deliberate: **never start in the browser.** A bug in a
WASM kernel and a bug in the browser glue look identical from the UI. By the
time anything ran in a browser, the maths had already been verified twice — once
in PyTorch, once in native C++.

---

## 2. The model

`python_ref/model.py` builds the smallest thing that is still a real GPT. The
exact spec lives in `configs/model.byte-tinygpt-v0.json`:

| Field | Value | Meaning |
| --- | --- | --- |
| `vocab_size` | 256 | one token per byte — no BPE |
| `context_length` | 128 | tokens of history the model can see |
| `n_layers` | 4 | transformer blocks |
| `n_heads` | 4 | attention heads (`head_dim = 32`) |
| `d_model` | 128 | residual-stream width |
| `d_mlp` | 512 | MLP hidden width (`4 × d_model`) |
| `tie_embeddings` | true | output head reuses the token embedding |

**Total: 842,496 parameters** — matched against the `~0.8M` target by a test.

### Why byte-level

Every byte is a token, so the vocabulary is fixed at 256. There is no tokenizer
to train, no merge table, no out-of-vocabulary handling — one entire class of
subtle bugs simply does not exist. The cost is that the model must spell every
word one byte at a time, but for a *learning* project that trade is correct.

### The forward pass, component by component

```
token ids ─┐
           ├─ token_embedding[id] + position_embedding[pos]   → x  [B,T,128]
           │
           ├─ 4 × TransformerBlock:
           │     x = x + attention(layernorm(x))
           │     x = x + mlp(layernorm(x))
           │
           ├─ final layernorm
           └─ logits = x · token_embedding.T                  → [B,T,256]
                                                              → cross-entropy
```

- **Embeddings.** A token id indexes a `[256,128]` table; the position indexes a
  `[128,128]` table. The two vectors are summed so the model knows both *what*
  and *where*.

- **Pre-LayerNorm blocks.** Normalisation happens *before* each sublayer, not
  after. This keeps the residual stream — the `x` that is added back each time —
  un-normalised and clean, which makes gradients flow straight through many
  layers. Post-LayerNorm is famously harder to train; pre-LN was the right call
  and needed no warm-up tricks.

- **Causal self-attention.** The only place tokens exchange information:

  ```
  q,k,v = q_proj(x), k_proj(x), v_proj(x)      each [B,4,T,32]
  scores = q @ kᵀ / √32
  scores = causal_mask(scores)                 future positions → −∞
  attn   = softmax(scores)
  out    = o_proj(attn @ v)
  ```

  The four projections are kept as **separate named modules** (`q_proj`,
  `k_proj`, `v_proj`, `o_proj`) rather than one fused QKV matrix. A fused matrix
  is marginally faster, but LoRA later needs to target `q_proj`/`v_proj` *by
  name* — so clarity won over the micro-optimisation. Same parameter count
  either way.

- **MLP.** `Linear(128→512) → GELU → Linear(512→128)`. This is where most
  parameters live and where each position does its per-token "thinking"
  independently of the others.

- **Tied output head.** Logits are `x · Eᵀ`, reusing the token-embedding matrix
  `E`. No separate output matrix — fewer parameters, and tying usually helps
  small models.

### Initialisation

GPT-2-style: weights `N(0, 0.02)`, biases zero, LayerNorm gains one. The two
projections that write *into* the residual stream — `o_proj` and the MLP's
`fc_out` — get a scaled init, `std = 0.02 / √(2·n_layers)`, so the residual
stream does not blow up as depth accumulates.

### What the experiments showed

| Check | Result |
| --- | --- |
| Parameter count | **842,496** — matches the ~0.8M target |
| Random-model loss | **5.56**, vs `ln(256) = 5.5452` |
| Every layer's output shape | as expected |
| Gradient check (`torch.autograd.gradcheck`) | passes |

The random-model loss is the first real sanity check. An untrained model should
be exactly as good as guessing uniformly over 256 bytes — and the cross-entropy
of a uniform guess is `ln(256)`. Landing at 5.56 means the loss function, the
softmax, and the forward pass have no bug.

---

## 3. Training

`python_ref/train.py` is a textbook loop: sample a batch, forward, cross-entropy
loss, backward, clip gradients to norm 1.0, AdamW step. The hyperparameters
(`configs/training.json`): batch 16, lr 3e-4, betas (0.9, 0.95), weight decay
0.1.

One detail worth calling out: **weight decay is applied only to matrices and
embeddings, never to biases or LayerNorm gains.** Decaying a bias toward zero
regularises nothing — it just adds noise to a parameter whose whole job is to
shift activations. The optimiser is built with two parameter groups for exactly
this reason.

### What a loss curve actually looks like

- **The tiny-overfit test — the single most important one.** On a few KB of
  repeated text the loss fell from **5.53 → 0.015**. The logic: if a tiny model
  *cannot* drive loss to near-zero on a tiny dataset, then the model, the
  backward pass, or the data pipeline is broken. Scaling a broken model only
  burns time. This test is the gate every later phase repeats.

- **A real run** on `data/examples/tiny-corpus.txt` (~3 KB): training loss fell
  `5.58 → ~1.0`, while **validation loss bottomed near 2.50 and then rose**.
  That divergence is not a bug — it is overfitting, observed live. A 0.8M model
  has far more capacity than 3 KB of text needs, so past a point it improves on
  text it has seen at the expense of text it has not. Watching the two curves
  split is the clearest possible lesson in what "overfitting" means.

- **Checkpoint reload** reproduced the loss bit-for-bit. **Resuming** continued
  the curve smoothly — because the AdamW moment estimates (`m`, `v`) are saved
  alongside the weights. Without them, resume restarts the moments from zero and
  the loss curve visibly kinks at the resume point.

---

## 4. LoRA — adapting a frozen model cheaply

`python_ref/lora.py` implements Low-Rank Adaptation. A linear layer `y = xW` is
replaced with:

```
y = xW + (α/r) · x A B
```

`W` is **frozen**. Only `A` (`d_in × r`) and `B` (`r × d_out`) train — and with
`r = 4`, that is a tiny fraction of the original layer. `B` is initialised to
**zeros**, so at step 0 the adapter contributes exactly nothing and the model is
*bit-identical* to the base model. `A` is initialised small but non-zero so it
has a gradient to follow once `B` moves off zero.

### The subtle part: gradient flow through frozen layers

Freezing `W` does **not** mean stopping gradients through the layer. `xW` stays
in the autograd graph, so the gradient still flows back to `x` — and therefore
reaches LoRA adapters in *lower* layers. A frequent beginner bug is to
`.detach()` frozen layers "for efficiency", which silently starves every
adapter below them. The test below exists specifically to catch that.

### What the experiments showed

| Check | Result |
| --- | --- |
| Adapter size | **8,192 params — 0.96%** of the 850,688 total |
| Fine-tune loss (different-style corpus) | **4.71 → 2.25** |
| Step-0 identity (`B = 0`) | base + LoRA output is **bit-identical** to base |
| Frozen-base gradient flow | base weights get **no** grad; the *first* block's adapter does |
| Adapter save / reload | round-trips exactly |
| Base vs base+LoRA output | visibly different |

The "frozen-base gradient flow" test is the clever one. LoRA is injected into
*every* block. After one backward pass, the **first** block's adapter must have
a non-zero gradient — and the only way gradient reaches it is by travelling back
through all the frozen base weights of the blocks above. If those were detached,
the first adapter would get nothing. It gets a gradient: the path is intact.

LoRA teaches **style**, not facts. On a 0.8M base the "style" is crude, but the
mechanism — frozen base, trainable low-rank adapter, adapter-only checkpoints —
is exactly right.

---

## 5. The WASM port — re-deriving every backward by hand

Phase 4 re-implements the compute in C++ (`wasm/src/`) with **no autograd
engine**. This is the heart of the project. Each kernel carries its own
hand-written forward *and* backward:

| Kernel | Forward | Backward |
| --- | --- | --- |
| `matmul` | `C = A·B` | `dA = dC·Bᵀ`, `dB = Aᵀ·dC` |
| `layernorm` | normalise over the last dim | `dx`, `dγ`, `dβ` |
| `attention` | causal multi-head attention | through softmax, mask, all 4 projections |
| `adamw` | the optimiser step | — (plus grad-norm/clip helpers) |

Then `model.cpp` assembles them into a full TinyGPT — embeddings, the GELU MLP,
the tied head, cross-entropy — again with every backward derived by hand.

### Verifying a hand-written backward

There is no autograd to fall back on, so each backward is checked by a
**finite-difference gradient check**: perturb each input element by `±h`,
measure the change in a scalar loss, and compare the numerical gradient to the
analytic one. A wrong backward formula shows up as a large relative error long
before it could quietly corrupt a training run.

**Results: 18/18 kernel checks pass.** And one of those checks taught a lesson
worth more than the test itself —

> The attention weight gradients (`dWq`, `dWk`) first appeared to fail at 4–7%
> error. The instinct is to suspect the kernel. But `dx`, `dWv`, and `dWo` all
> passed — and `dWq` uses the *same* `linear_backward` routine as the passing
> `dWv`. The kernel was correct. The **test metric** was wrong: a per-element
> relative error with a fragile `+1e-3` denominator inflates finite-difference
> noise on the small, near-cancelling gradients the softmax produces. Switching
> to the standard metric — error normalised by the buffer's largest gradient
> component — dropped the reported error to 6e-4.
>
> **A test can be wrong even when the code is right.** When a test fails, check
> the test, not just the code.

### The C++ model

`tests/test_wasm_model.cpp` runs the same overfit gate as Phase 1, now against
the entire hand-written backward chain: loss **5.56 → 0.03** on repeated text.
It also does a **checkpoint round-trip** — export the full state, load it into a
fresh model, and confirm the fresh model produces the *identical* greedy
continuation. It does.

---

## 6. WebGPU — the same matmul, on the GPU

`webgpu/matmul.wgsl` is a compute shader: one GPU invocation computes one output
element of `C = A·B`, with 16×16 workgroups tiling the output. `webgpu/kernels.ts`
sets up the device, buffers, and pipeline, and benchmarks the result against the
WASM matmul.

### What the experiments showed

- **Bit-exact parity** with the WASM kernel — max absolute error `0.0`. Both
  accumulate the `k`-sum in the same order with the same float32 rounding, so
  they produce *identical* results. That is the strongest possible reading of
  "correct against WASM".
- **~1.4–1.9× faster** on a 384×384 matmul, even in a headless browser (likely
  a software adapter). On real GPU hardware the gap is far larger. The milestone
  asked for "correct first, then measurably faster" — both halves hold.

The shader is the *naive* version — one element per invocation, no shared-memory
tiling. That was deliberate: correct and readable first. Tiling is the obvious
next optimisation and is noted as future work.

---

## 7. The browser app

`browser/` is a Vite + TypeScript app. The architecture decision that matters:

```
Main thread (main.ts)    UI only — controls, capability panel, loss chart
        │  postMessage (corpus + config)  ↑  TrainingProgress / checkpoint
Web Worker (worker.ts)   the entire training loop
        │  calls
WASM module              the kernels + the C++ TinyGPT
```

**The main thread never does model maths.** Training runs in a Web Worker — a
genuinely separate thread — so the page cannot freeze no matter how heavy the
run. The worker trains in small chunks and yields between them, so `pause` /
`stop` messages from the UI are handled promptly.

The message protocol (`types.ts`) is a small tagged union in each direction —
`train`/`pause`/`stop`/`sample`/`restore` to the worker, `progress`/`sample`/
`checkpoint`/`done`/`error` back. The loss chart (`charts.ts`) is a
dependency-free canvas renderer fed entirely by `TrainingProgress` messages —
that live chart, with tokens/sec and the active backend, *is* milestone 8.

### What the experiments showed

A headless-browser end-to-end test (`browser/e2e_browser.mjs`, driven by
Playwright) trains a model to completion **inside the Worker**, loss
**5.5 → 0.017**, then samples from it — with **zero console or page errors**. It
also runs the WebGPU benchmark and the checkpoint test in the same pass.

---

## 8. Checkpointing — surviving a page refresh

The browser holds the model inside the WASM heap. A page refresh destroys it. To
make a run survive:

1. The WASM C ABI gained `tg_state_bytes` / `tg_export_state` / `tg_import_state`,
   which serialise the **full trainable state** — every weight, both AdamW
   moments, and the step count.
2. When training finishes, the worker exports that blob and posts it to the main
   thread (as a transferable `ArrayBuffer`).
3. `storage.ts` writes it to **OPFS** (the Origin-Private File System), along
   with a small JSON snapshot of the config and loss history.
4. On the next page load, `main.ts` reads both back, rebuilds the model in the
   worker, and restores the chart.

The e2e test proves it: after `page.reload()`, the trained model and its loss
chart are restored, and the restored model still generates text. Saving the
moments (not just the weights) is what would make a *resumed* run continue its
loss curve smoothly rather than kink.

---

## 9. Evaluation — comparing honestly

`python_ref/evaluate.py` produces, for every held-out prompt, the four
conditions `docs/evaluation.md` requires:

| Condition | What it isolates |
| --- | --- |
| **A. Base** | the model with no help |
| **B. Base + few-shot** | style supplied by in-context examples |
| **C. Base + LoRA** | style baked into an adapter |
| **D. Base + LoRA + retrieval** | adapter for style, retrieval for context |

A tiny character-trigram retriever stands in for a real vector store — enough to
show that retrieval drags the output toward a relevant passage. The harness also
runs the **memorization check**: feed a training prefix back in and measure the
longest verbatim continuation.

**Result:** the longest verbatim continuation was **8 characters** — far below
the copying-risk threshold. The model learned patterns, not a photographic copy.

The honest caveat: a 0.8M byte-level model on a few KB of text produces rough
prose in *every* condition. The deliverable is the **harness** — a reproducible,
apples-to-apples comparison and a memorization metric — not the prose.

---

## 10. The bugs, and what they taught

Five bugs were worth recording. None were in the "interesting" maths — they were
all in the seams.

1. **The loss-sanity test that cheated.** An early test fed the model
   *same-position* targets. Residual connections + tied embeddings leave the
   input token's own embedding in the output, so the untrained model scored
   *below* `ln(256)`. *Lesson: a sanity test can lie if its task is easier than
   the real one.*

2. **The grad-check metric, not the kernel.** Covered in [§5](#5-the-wasm-port--re-deriving-every-backward-by-hand).
   *Lesson: when a test fails, suspect the test too.*

3. **A swapped Playwright argument.** `waitForFunction(fn, options)` silently
   treated the options object as the page-function *argument*, so a 180s timeout
   defaulted to 30s. *Lesson: read the signature; a "timeout" can be ignored
   without erroring.*

4. **`_malloc` not exported.** Emscripten 5.x no longer exports the allocator by
   default — it had to be named explicitly in `EXPORTED_FUNCTIONS`. *Lesson:
   toolchain defaults drift between major versions.*

5. **TypeScript 5.7 generic typed arrays.** `Float32Array` became generic over
   its buffer type, and `Uint8Array<ArrayBufferLike>` is not assignable where
   `<ArrayBuffer>` is required. *Lesson: a language can tighten types under you.*

The content-filter block on a pasted public-domain corpus also forced a useful
habit: example corpora here are original prose, and a real dataset should be
*downloaded* by a script rather than embedded.

---

## 11. Results at a glance

| Phase | What was verified | Result |
| --- | --- | --- |
| 1 | model param count | 842,496 |
| 1 | random-model loss | 5.56 ≈ ln(256) |
| 1 | tiny-overfit gate | 5.53 → 0.015 |
| 1 | gradient check, checkpoint reload, deterministic sampling | pass |
| 3 | LoRA adapter size | 8,192 params (0.96%) |
| 3 | LoRA fine-tune loss | 4.71 → 2.25 |
| 3 | step-0 identity, frozen-grad flow, adapter round-trip | pass |
| 4 | C++ kernel finite-diff checks | 18 / 18 |
| 4 | C++ model overfit gate | 5.56 → 0.03 |
| 4 | C++ checkpoint round-trip | pass |
| 4 | compiled WASM module, trained from Node | 5.56 → 0.016 |
| 4 | headless-browser e2e (train in Worker) | 5.5 → 0.017, 0 errors |
| 6 | WebGPU matmul parity vs WASM | bit-exact (error 0.0) |
| 6 | WebGPU matmul speed-up | ~1.4–1.9× |
| 7 | model survives a page refresh | restored + still generates |
| 9 | memorization check | 8 chars verbatim — ok |

**Test suites:** Python 14/14 · C++ kernels 18/18 · C++ model gates · Node WASM
smoke · headless-browser e2e — all green.

---

## 12. How to reproduce everything

```bash
# Python reference (Phases 1, 3) — needs PyTorch
python -m venv python_ref/.venv && source python_ref/.venv/bin/activate
pip install -r python_ref/requirements.txt
python tests/test_phase1.py
python tests/test_lora.py
python python_ref/train.py --data data/examples/tiny-corpus.txt --out checkpoints/base
python python_ref/lora.py  --base checkpoints/base --data data/examples/tiny-corpus-2.txt \
    --out checkpoints/adapter
python python_ref/evaluate.py --base checkpoints/base --adapter checkpoints/adapter

# WASM kernels + model (Phase 4) — needs only clang/g++
bash wasm/build_native.sh

# Browser app (Phases 4-5) — needs Emscripten + Node
bash wasm/build_wasm.sh
cd browser && npm install && npm run dev      # then "Start training"
node ../tests/smoke_wasm_node.mjs             # verify the compiled module
npm run build && npm run preview & npm run e2e # full headless end-to-end
```

---

## 13. What this is — and is not

It **is** a correct, end-to-end, byte-by-byte implementation of the modern LLM
stack: a GPT, trained with AdamW, adapted with LoRA, ported to hand-written WASM
kernels, accelerated with a WebGPU kernel, and wrapped in a browser app that
trains without freezing — every layer gated by a test.

It is **not** a capable language model, and was never meant to be. 0.8M
parameters on kilobytes of text cannot reason or be reliably factual; LoRA here
moves tone and vocabulary, not truth. That ceiling is by design — the goal was
**understanding the machinery**, and the machinery is now understood line by
line.

### If this were taken further

- Pretrain a **5–15M base** on megabytes of text, so LoRA adaptation produces
  legible *style* and not merely legible *mechanism*.
- Port more kernels to WebGPU — attention, the optimiser — and add a tiled,
  shared-memory matmul.
- Build the interactive teaching surfaces sketched in `docs/feature_ideas.md`:
  an attention visualiser, a tokenizer playground, a semantic-zoom diagram of
  the forward pass.

The scaffold is now a working system. Every box in `MILESTONES.md` that can be
ticked by code, is.

---

## 14. Related projects

- **[mlc-ai/web-llm](https://github.com/mlc-ai/web-llm)** — runs already-trained
  LLMs (Llama, Phi, Mistral, Qwen, Gemma) in the browser via WebGPU, exposing an
  OpenAI-compatible API. It is the *other half* of the in-browser LLM story:
  TinyGPT is how a transformer is **built and trained** from the maths up;
  WebLLM is how a real, large pretrained model is **served** in a tab. They are
  not stackable — WebLLM consumes MLC-compiled 4-bit artifacts, TinyGPT writes
  its own f32 weights, and a LoRA adapter trained here cannot be applied to a
  Llama checkpoint there. They are complementary in spirit: read this repo to
  understand *what is happening inside the model*, read WebLLM to see *how the
  big ones run in a browser*.
