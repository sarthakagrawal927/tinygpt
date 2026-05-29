# Model guide — building TinyGPT from scratch

Phase 1–2. Build a tiny GPT-style causal language model. First goal is
**correctness**, not impressive output.

Exact numbers live in `configs/model.byte-tinygpt-v0.json` and
`configs/training.json` — this doc explains them.

---

## 1. What you are building

A tiny GPT-style causal language model:

```
input tokens
  → token embeddings
  → position embeddings
  → transformer blocks
  → final layernorm
  → logits over vocabulary
  → next-token prediction loss
```

For v0, use a **byte-level tokenizer**: `vocab_size = 256`. Every byte is a
token. This avoids all BPE / tokenizer complexity.

---

## 2. MVP model spec

```json
{
  "model_name": "byte-tinygpt-v0",
  "vocab_size": 256,
  "context_length": 128,
  "n_layers": 4,
  "n_heads": 4,
  "d_model": 128,
  "d_mlp": 512,
  "dropout": 0.0,
  "tie_embeddings": true,
  "dtype": "float32"
}
```

Expected size of the reference config above: roughly **0.8M parameters**.
Intentionally small. The browser playground exposes a preset table from
360k (Small) to ~470M (Behemoth via Memory64), backed by the same
architecture.

**Why float32 everywhere?** All training (Python, WASM, WebGPU) uses float32
for numeric stability — gradients on tiny models are unforgiving and lower
precision multiplied the loss-drift budget faster than it bought speed.
f16 lives in the project as an *inference-only* path, gated behind the
end-to-end parity tests (see the "f16-packed storage" entry in the README's
"Negative results" section for what didn't pan out and why).

---

## 3. Data requirements

Plain text only. See `data/README.md` for good/bad sources and dataset sizes.
Byte-level: 1 byte ≈ 1 token, so a 1 MB file ≈ 1 million tokens.

| Stage        | Size        | Purpose                     |
| ------------ | ----------- | --------------------------- |
| Smoke test   | 1–10 KB     | Check loss decreases        |
| Overfit test | 10–100 KB   | Prove gradients are correct |
| Demo dataset | 500 KB–5 MB | Realistic browser demo      |
| Stress test  | 10–100 MB   | Later only                  |

---

## 4. Dataset pipeline

```
raw text → UTF-8 bytes → integer token array → train/val split
         → random batch sampler → (x, y) pairs
```

```
tokens = [72, 101, 108, 108, 111, ...]
x = tokens[i : i + context_length]
y = tokens[i + 1 : i + context_length + 1]
```

Split 90% train / 10% val. Write a dataset manifest — the **hash** is what
makes checkpoint resume reproducible:

```json
{
  "dataset_id": "sha256_of_raw_bytes",
  "name": "my_blog_posts.txt",
  "raw_bytes": 1249301,
  "token_count": 1249301,
  "tokenizer": "byte-v1",
  "train_split": 0.9,
  "val_split": 0.1,
  "seed": 42
}
```

---

## 5. Architecture details

### Embeddings

```
token_embedding:    [vocab_size, d_model]
position_embedding: [context_length, d_model]
x = token_embedding[token_ids] + position_embedding[position_ids]
```

### Transformer block — use pre-LayerNorm

```
x = x + attention(layernorm(x))
x = x + mlp(layernorm(x))
```

Pre-LayerNorm is easier to train than post-LayerNorm.

### Causal self-attention

```
q = x @ Wq;  k = x @ Wk;  v = x @ Wv
scores = q @ k.T / sqrt(head_dim)
scores = causal_mask(scores)
attn   = softmax(scores)
out    = attn @ v
out    = out @ Wo
```

Shapes (B batch, T seq, C d_model, H heads, head_dim = C / H):

```
B = 16   T = 128   C = 128   H = 4   head_dim = 32
```

### MLP

```
Linear(d_model → 4 * d_model)  →  GELU  →  Linear(4 * d_model → d_model)
```

For `d_model = 128`: `128 → 512 → 128`.

### Output head — tied embeddings

```
x = final_layernorm(x)
logits = x @ token_embedding.T
output_projection_weight = token_embedding_weight
```

Tied embeddings reduce parameter count and usually improve tiny models.

---

## 6. Loss function

Next-token cross-entropy. For a 256-byte vocab:

```
initial_loss ≈ ln(256) ≈ 5.54
```

| Condition             | Expected                                         |
| --------------------- | ------------------------------------------------ |
| Random model          | loss near 5.54                                   |
| Repeated tiny dataset | loss falls fast                                  |
| Loss does not fall    | bug in model / backprop / data                   |
| Loss becomes NaN      | learning rate, softmax, grad explosion, bad init |

---

## 7. Training config

```json
{
  "batch_size": 16,
  "learning_rate": 0.0003,
  "optimizer": "adamw",
  "betas": [0.9, 0.95],
  "eps": 1e-8,
  "weight_decay": 0.1,
  "grad_clip": 1.0,
  "max_steps": 10000,
  "eval_interval": 100,
  "sample_interval": 500,
  "checkpoint_interval": 500,
  "seed": 42
}
```

- Loss unstable → lower LR `0.0003 → 0.0001`.
- Loss too slow on tiny data → raise LR `0.0003 → 0.001`, but only after
  verifying gradients.

---

## 8. Training loop

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model.forward(x)
    loss = cross_entropy(logits, y)

    model.zero_grad()
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()

    if step % eval_interval == 0:        val_loss = evaluate()
    if step % sample_interval == 0:      sample_text = generate(prompt)
    if step % checkpoint_interval == 0:  save_checkpoint()
```

In the browser this becomes: Web Worker → get batch → WASM/WebGPU forward →
backward → optimizer step → post progress to UI. See `browser_notes.md`.

---

## 9. Implementation order

### Step 1 — Python / PyTorch reference (do this first)

Deliverables: `model.py`, `dataset.py`, `train.py`, `sample.py`.
Goal: train the reference 0.8M-param config on 100 KB of text; loss decreases; sampling works;
checkpoint reloads. Use Karpathy's nanoGPT as a structural reference — not
something to copy blindly.

### Step 2 — tiny model from scratch

Reimplement in TypeScript / C++ / Rust. For browser learning: a TypeScript
reference plus a C++/Rust WASM backend. Do **not** write a general autograd
engine — you only need backprop for: Linear, Embedding, LayerNorm, GELU,
Softmax, Attention, CrossEntropy, AdamW.

### Steps 3–6 — WASM, Web Worker, checkpointing, WebGPU

See `browser_notes.md`.

---

## 10. Required tests

The full list and rationale is in `../tests/README.md`. The most important one:

> **Can it overfit a tiny repeated dataset?**
> If not, scaling is pointless.

---

## References

- nanoGPT — minimal GPT training/finetuning repo: https://github.com/karpathy/nanoGPT
- build-nanogpt — step-by-step construction: https://github.com/karpathy/build-nanogpt
