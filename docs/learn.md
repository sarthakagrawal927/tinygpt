# Learn TinyGPT — a guided path for software engineers new to AI

You can write code. You have never built or trained a neural network. This guide
takes you, in order, through **everything in this repository** until none of it
is a black box.

It does not re-teach what others teach better. For each concept it gives you:

> **Learn it** — a link to the best free explainer (watch / read it).
> **In the repo** — the exact file and function where TinyGPT implements it.
> **See it work** — the command that runs that code and proves it.

The repo is a real, working GPT — trained from scratch, fine-tuned with LoRA,
ported to hand-written WebAssembly, accelerated with WebGPU, and wrapped in a
browser app. By the end you will understand all of it, both the AI and the
systems engineering.

**Time:** ~12–20 hours if you watch every linked video. You do not have to do it
in one sitting — the parts are self-contained.

---

## Contents

- [0. Orientation](#0-orientation--the-repo-in-10-minutes)
- [1. The math you actually need](#1-the-math-you-actually-need)
- [2. Language modeling](#2-language-modeling--what-the-model-is-doing)
- [3. The transformer](#3-the-transformer)
- [4. LoRA — cheap fine-tuning](#4-lora--cheap-fine-tuning)
- [5. Running it in the browser](#5-running-it-in-the-browser-the-systems-half)
- [6. See the whole thing run](#6-see-the-whole-thing-run)
- [Going deeper](#going-deeper)

---

## 0. Orientation — the repo in 10 minutes

**What TinyGPT is.** A ~0.8M-parameter GPT — the same architecture as ChatGPT,
about 200,000× smaller. It is intentionally tiny so that every part is readable
and every part is *tested*. It will never say anything clever; that is not the
point. The point is that you can understand a complete one.

**The guiding rule of the codebase:** never trust a component until a test pins
it down. Every layer has a check that fails loudly if it is wrong — which is
also why this guide can always end a section with "see it work".

**The repo map:**

| Directory | What's in it |
| --- | --- |
| `configs/` | The exact model / training / LoRA settings, as JSON |
| `python_ref/` | The reference implementation in PyTorch — **read this first** |
| `wasm/` | The same maths re-written in C++, compiled to WebAssembly |
| `webgpu/` | A matrix-multiply kernel for the GPU |
| `browser/` | The web app that trains a model in your browser |
| `tests/` | The correctness tests for every layer above |
| `docs/` | This guide, plus a deeper spec for each phase |

**Read order for the code:** `python_ref/` first (it is the clearest), then
`wasm/`, then `webgpu/` and `browser/`. The whole project was *built* in that
order on purpose — a correct reference first, then ports of it.

**Set up your environment** (you will need these as you go):

```bash
# Python reference — needs Python 3.10+
python -m venv python_ref/.venv && source python_ref/.venv/bin/activate
pip install -r python_ref/requirements.txt

# C++ / WASM parts — needs a C/C++ compiler (clang or gcc); Emscripten later
# Browser app — needs Node.js
```

The companion doc [`notes.md`](notes.md) is the *retrospective* — what was built
and what each experiment showed. Skim it now for the shape of the project; come
back to it at the end when it will all make sense.

---

## 1. The math you actually need

You do **not** need a maths degree. You need to be comfortable with five ideas.
Treat this as vocabulary, not a course.

### 1.1 Vectors, matrices, and matrix multiplication

A neural network is mostly **arrays of numbers multiplied together**. A "vector"
is a 1-D array; a "matrix" is a 2-D array; "matmul" is the one operation that
combines them.

- **Learn it:** [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
  (chapters 1–4 are enough: vectors, linear combinations, matrix multiplication).
- **In the repo:** the single most important operation. See it bare in
  `wasm/src/matmul.cpp` — `matmul_forward` is a 3-line triple loop. That *is*
  matrix multiplication, no abstraction.
- **See it work:** `bash wasm/build_native.sh` — the first test, "matmul
  forward", checks that loop against a hand-computed reference.

### 1.2 The dot product

Multiply two vectors element-wise, sum the result — one number that measures how
*aligned* two vectors are. Attention (Part 3) is built entirely on this.

- **Learn it:** covered in the 3B1B series above (chapter 9), or just read
  `matmul_forward` — a matmul *is* a grid of dot products.

### 1.3 softmax

Turns a list of arbitrary numbers ("scores") into a list of probabilities that
sum to 1. The model produces scores; softmax makes them a probability
distribution over the 256 possible next bytes.

- **Learn it:** it is small — `softmax(x)[i] = exp(x[i]) / Σ exp(x[j])`. The
  3B1B GPT video below visualises it.
- **In the repo:** `python_ref/model.py` — `F.softmax` inside
  `CausalSelfAttention`; and the hand-written version in
  `wasm/src/attention.cpp` (the `std::exp` loop), which also shows the
  *numerical-stability* trick of subtracting the max first.

### 1.4 Loss, and cross-entropy

"Loss" is a single number measuring how wrong the model is — training is just
making it smaller. For predicting one of 256 bytes, the loss is **cross-entropy**:
it is low when the model put high probability on the correct byte.

- **Learn it:** the key intuition — a model that guesses uniformly over 256
  options has loss `ln(256) ≈ 5.54`. That number recurs everywhere in this repo.
- **In the repo:** the `F.cross_entropy` call in `TinyGPT.forward`
  (`python_ref/model.py`), and the from-scratch `cross_entropy` in
  `wasm/src/model.cpp`.
- **See it work:** `python tests/test_phase1.py` — the "loss sanity" test
  confirms an untrained model's loss is `≈ 5.54`.

### 1.5 Gradients, backpropagation, gradient descent

This is the heart of *learning*. A **gradient** tells you which direction to
nudge each number to make the loss smaller. **Backpropagation** is the algorithm
that computes every gradient efficiently. **Gradient descent** is the loop: nudge,
re-measure, repeat. **AdamW** is the specific, well-tuned nudging rule used here.

- **Learn it:** [3Blue1Brown — Backpropagation](https://www.3blue1brown.com/lessons/backpropagation),
  then — highly recommended — [Karpathy, "The spelled-out intro to neural
  networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0).
  Karpathy builds a tiny autograd engine by hand; it makes everything below click.
- **In the repo:** PyTorch computes gradients automatically in `python_ref/`.
  But in `wasm/src/` there is **no autograd** — every backward pass is written by
  hand. `wasm/src/matmul.cpp` `matmul_backward` is the gradient of a matmul; the
  formula in the comment is exactly what backprop derives. AdamW itself:
  `wasm/src/adamw.cpp`.
- **See it work:** `wasm/build_native.sh` runs a **finite-difference gradient
  check** on every hand-written backward — it nudges each input and confirms the
  measured slope matches the analytic gradient. That is backprop, verified.

> ✓ **You've got this when** you can say, in your own words: what a loss is, what
> a gradient is, and what one training step does (measure loss → compute
> gradients → nudge every weight a little).

---

## 2. Language modeling — what the model is doing

### 2.1 The one job: predict the next token

A GPT is **not** answering questions or reasoning. It does exactly one thing:
given some text, predict the next token. Everything else (chat, code) is that
one trick, repeated.

- **Learn it:** [3Blue1Brown — "But what is a GPT?"](https://www.3blue1brown.com/lessons/gpt)
  — the single best 25 minutes for the big picture. Watch this now.

### 2.2 Tokenization — text becomes numbers

A model only sees numbers. A **tokenizer** maps text to integers. Real GPTs use
"BPE"; TinyGPT uses the simplest possible scheme — **one byte = one token**, so
the vocabulary is exactly 256. No tokenizer to train, no edge cases.

- **In the repo:** `python_ref/dataset.py` — `encode` / `decode` (4 lines each);
  the browser's copy is `browser/src/tokenizer.ts`.
- **See it work:** `python tests/test_phase1.py` — the "tokenizer roundtrip"
  test confirms `decode(encode(text)) == text`.

### 2.3 Context window and the causal rule

The model sees a fixed window of recent tokens (here, 128). When predicting
position *t*, it may look only at positions *≤ t* — never the future. That
"causal" rule is what makes left-to-right generation possible.

- **In the repo:** the `causal_mask` buffer in `python_ref/model.py`
  `CausalSelfAttention`.

### 2.4 Sampling — turning predictions into text

The model outputs a probability for every possible next byte. To generate, you
**sample** one, append it, and repeat. `temperature` controls boldness (low =
safe and repetitive, high = wild); `top_k` restricts the choice to the *k* most
likely bytes.

- **In the repo:** `python_ref/sample.py`, and the `generate` method in
  `python_ref/model.py`.
- **See it work:** after training a model (Part 6),
  `python python_ref/sample.py --checkpoint checkpoints/base --prompt "The "`.

> ✓ **You've got this when** you can explain why "a language model predicts the
> next token" and "a language model reasons" are very different claims.

---

## 3. The transformer

This is the architecture itself. Do this part with two tabs open: the reading
below, and `python_ref/model.py` — it is ~230 lines and you should read all of
it here.

**First, the big picture.** Read [Jay Alammar — The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/), and keep
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/) open
in a tab — it is a live GPT-2 you can poke, and it makes the next four ideas
concrete.

### 3.1 Embeddings — a number becomes a vector

A token id (0–255) indexes into a learned table, producing a `d_model`-length
vector. A second table does the same for the token's *position*. The two are
added: now the model knows *what* the token is and *where* it sits.

- **In the repo:** `token_embedding` and `position_embedding` in
  `python_ref/model.py` `TinyGPT`.

### 3.2 Self-attention — tokens look at each other

The one place tokens exchange information. Each token asks a question (a "query"
vector), every token offers a key; the dot product of query·key scores how
relevant each other token is; softmax turns the scores into weights; the token
pulls in a weighted blend of every token's "value" vector.

- **Learn it:** [3Blue1Brown — "Attention in transformers"](https://www.3blue1brown.com/lessons/attention).
  Then play with Transformer Explainer.
- **In the repo:** `CausalSelfAttention` in `python_ref/model.py`. The maths is
  also spelled out in [`model_guide.md`](model_guide.md) §5. The *from-scratch*
  version — with the backward pass written by hand — is `wasm/src/attention.cpp`.
- **See it work:** `python tests/test_phase1.py` — the "gradient check" test
  runs `torch.autograd.gradcheck` on the attention module.

### 3.3 The supporting cast

Read these straight from `python_ref/model.py` — each is a few lines:

- **Multi-head attention** — run several small attentions in parallel so
  different heads can specialise. (`n_heads` in the config.)
- **LayerNorm** — rescales a vector to a steady mean/variance so deep stacks
  stay numerically stable. (`nn.LayerNorm`; hand-written in
  `wasm/src/layernorm.cpp`.)
- **Residual connections** — `x = x + sublayer(x)`. The `+ x` gives gradients a
  clean path back through many layers. (The `x = x + ...` lines in
  `TransformerBlock`.)
- **MLP + GELU** — after attention mixes tokens, a small 2-layer network
  processes each position on its own. GELU is its activation function. (`MLP` in
  `model.py`; `gelu_forward` in `wasm/src/model.cpp`.)

### 3.4 The whole model

A `TransformerBlock` is `attention` + `MLP`, each wrapped in LayerNorm and a
residual. Stack four of them, add a final LayerNorm, and project back to 256
scores with the **tied** output head (it reuses the embedding table). That is
`TinyGPT`.

- **In the repo:** `TransformerBlock` and `TinyGPT` in `python_ref/model.py`;
  the design rationale is in [`model_guide.md`](model_guide.md).
- **See it work:** `python tests/test_phase1.py` — the "param count" (842,496)
  and "layer shapes" tests confirm the assembled model is wired correctly.

### 3.5 Training it

Training is the gradient-descent loop from Part 1, wrapped around this model: get
a batch of text, predict next tokens, measure cross-entropy loss, backpropagate,
AdamW step, repeat.

- **In the repo:** `python_ref/train.py`. The deeper spec is in
  [`model_guide.md`](model_guide.md) §7–8.
- **See it work:** the most important test in the repo —
  `python tests/test_phase1.py`, the "tiny overfit" test: a tiny model is made
  to drive its loss from `5.5` to near `0` on a small text. If a model cannot do
  that, something is broken — so this test gates everything.

> ✓ **You've got this when** you can trace one token's vector through
> `python_ref/model.py`: embedding → 4 blocks → final norm → 256 scores.

---

## 4. LoRA — cheap fine-tuning

Once a model is trained, **LoRA** adapts it to a new style without retraining the
whole thing. It freezes the original weights and trains two tiny matrices
alongside each chosen layer. Here the adapter is *0.96%* of the model's size.

- **Learn it:** the idea, plainly: a big weight matrix `W` is frozen; you learn a
  small low-rank correction `A·B` and use `W + A·B`. The original paper is
  [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (read the
  abstract and Figure 1).
- **In the repo:** `python_ref/lora.py` — `LoRALinear` is the whole idea in one
  class. The spec with the gradient maths is [`lora_guide.md`](lora_guide.md).
- **The subtlety worth understanding:** freezing a layer does **not** stop
  gradients flowing *through* it to adapters below — see the long comment in
  `lora.py` and the `test_frozen_base_grads` test.
- **See it work:** `python tests/test_lora.py` — confirms the step-0 adapter is a
  no-op, gradients reach the lowest adapter, and a saved adapter reloads exactly.

> ✓ **You've got this when** you can explain why LoRA trains <1% of the
> parameters and still changes the model's behaviour.

---

## 5. Running it in the browser — the systems half

This is the half almost no tutorial covers, and it is what makes this repo
unusual. A browser cannot run PyTorch. To train a model in a browser tab you
have to rebuild the compute yourself and respect the browser's constraints.

### 5.1 WebAssembly — running C++ in the browser

**WebAssembly (WASM)** is a fast, low-level bytecode browsers can run. You write
C/C++ and compile it to WASM with **Emscripten**. TinyGPT's kernels are written
in plain C++ for exactly this.

- **Learn it:** [MDN — WebAssembly concepts](https://developer.mozilla.org/en-US/docs/WebAssembly/Concepts),
  then skim [emscripten.org](https://emscripten.org/).
- **In the repo:** `wasm/src/` — `matmul`, `layernorm`, `attention`, `adamw`,
  and `model.cpp` (a full GPT in C++). [`browser_notes.md`](browser_notes.md)
  explains the build. There is **no autograd** here — every backward pass is
  derived and written by hand, which is the best possible way to truly learn
  backprop.
- **See it work:** `bash wasm/build_native.sh` builds and tests the C++ with
  your normal compiler (no Emscripten needed) — 18 kernel checks + a full-model
  overfit test.

### 5.2 Verifying a backward pass you wrote yourself

If you hand-write a gradient, how do you know it is right? You **nudge each input
by a tiny amount and measure the slope of the loss** — the "finite-difference"
gradient check. If your formula and the measured slope disagree, your formula is
wrong.

- **In the repo:** `grad_check` in `tests/test_wasm_kernels.cpp`. Read it — it is
  the technique that makes hand-written backprop trustworthy.

### 5.3 Web Workers — not freezing the page

JavaScript runs on one thread. If you train a model on it, the whole tab
freezes. A **Web Worker** is a separate thread; TinyGPT runs all training there
and the UI thread only draws.

- **Learn it:** [MDN — Using Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers).
- **In the repo:** `browser/src/worker.ts` (the training loop, off-thread) and
  `browser/src/main.ts` (the UI, which only sends/receives messages).

### 5.4 Sharing memory across the JS ↔ WASM boundary

WASM has its own block of memory. JavaScript reaches into it through typed-array
"views" and raw pointers. This is the seam where browser ML actually lives.

- **In the repo:** `browser/src/backend.ts` — `_malloc`, `HEAPU8`/`HEAPF32`, and
  copying arrays in and out. Checkpointing uses the same idea to save the model
  to **OPFS** (browser-local storage): `browser/src/storage.ts`,
  [MDN — OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system).

### 5.5 WebGPU — using the GPU

**WebGPU** is the browser API for running compute on the GPU. TinyGPT uses it for
the heaviest operation, matrix multiply, written as a **compute shader** in WGSL.

- **Learn it:** [WebGPU Fundamentals](https://webgpufundamentals.org/) — the
  "compute" articles.
- **In the repo:** `webgpu/matmul.wgsl` (the shader — one GPU thread per output
  number) and `webgpu/kernels.ts` (the JavaScript that drives it). The repo's
  benchmark checks it produces the *same* result as the WASM matmul, faster.

> ✓ **You've got this when** you can explain why training must run in a Worker,
> and what a "finite-difference gradient check" proves.

---

## 6. See the whole thing run

You have read it. Now watch every layer execute. If all of this passes, you have
understood a complete, working version of the modern LLM stack.

```bash
# --- the AI half (Parts 1-4) -----------------------------------------
source python_ref/.venv/bin/activate
python tests/test_phase1.py          # model, training, sampling — 8 tests
python tests/test_lora.py            # LoRA fine-tuning — 6 tests
python python_ref/train.py --overfit # watch a real loss curve fall

# --- the systems half (Part 5) ---------------------------------------
bash wasm/build_native.sh            # the C++ kernels + model, all verified

# --- the browser app -------------------------------------------------
bash wasm/build_wasm.sh              # compile C++ -> WebAssembly (needs Emscripten)
cd browser && npm install && npm run dev
# open the printed URL, click "Start training" — a GPT trains in your tab
```

For the full story of what each test proved — with the numbers — read
[`notes.md`](notes.md). For a per-phase deep spec, see the other files in
`docs/`. The phase/week breakdown is in [`learning_roadmap.md`](learning_roadmap.md).

---

## Going deeper

When you want to go beyond TinyGPT — to models that actually produce good text:

- **Andrej Karpathy — [Let's build GPT, from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)**
  — the same journey as Parts 1–3, in video, in more depth.
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** — the slightly bigger,
  production-shaped cousin of `python_ref/`. TinyGPT's structure mirrors it.
- **[Lil'Log — The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)**
  — every variation on the architecture, rigorously.

You started as a software engineer who had never built a neural network. If you
have followed this path, you can now read — and have tested — every line of a
real GPT, and the browser engineering that runs it. That was the whole goal.
