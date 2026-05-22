# data/ — datasets

## Use plain text only

Good first sources: your own notes/blog posts, public-domain text, a small
technical blog, a small codebase, short stories, README files.

Avoid initially: PDFs, HTML scraping, large websites, mixed-author corpora,
private messages, social scraping, large multilingual datasets.

## Dataset sizes (from-scratch training)

| Stage        | Size        | Purpose                     |
| ------------ | ----------- | --------------------------- |
| Smoke test   | 1–10 KB     | Check loss decreases        |
| Overfit test | 10–100 KB   | Prove gradients are correct |
| Demo dataset | 500 KB–5 MB | Realistic browser demo      |
| Stress test  | 10–100 MB   | Later only                  |

Byte-level: 1 byte ≈ 1 token, so a 1 MB text file ≈ 1 million tokens.

## LoRA datasets (structured examples)

For LoRA, do not dump raw text — build task-style examples and write JSONL:

```json
{"task":"continuation","title":"...","prefix":"...","completion":"..."}
{"task":"rewrite","draft":"...","completion":"..."}
{"task":"title","excerpt":"...","completion":"..."}
```

Aim for 300–1,000 clean examples from one consistent author. Quality beats
quantity. See `../docs/lora_guide.md` and `../docs/evaluation.md`.

## Files

- `dataset_builder.py` — turns raw text into token arrays / JSONL + manifests
- `examples/` — small sample corpora live here

`.gitignore` keeps bulk corpora out of git; `examples/` is kept.

## What the model is actually trained on

There is **no external or scraped dataset**. The model trains on whatever plain
text you give it. The two corpora in `examples/` are short **original prose
written for this project**:

- `tiny-corpus.txt` (~3 KB) — explanatory prose about how a small model learns;
  the default for from-scratch training.
- `tiny-corpus-2.txt` (~2 KB) — terse, distinctly different prose (woodworking
  maxims), used as the *different style* for the LoRA fine-tuning demo.

The browser app ships its own short original corpus as the textarea default, and
you can paste any text in there. They are deliberately tiny — a few KB is all
the overfit and LoRA demos need.

For a real run, supply your own plain text via `--data your.txt` (Python) or the
browser textarea: your own notes, public-domain books, a small codebase. See the
source guidance at the top of this file for what makes good vs. bad training
text.

## Loading an open dataset from Hugging Face

You don't have to supply the text yourself — both the browser app and the Python
builder can pull an open dataset from the Hugging Face Hub through its public
datasets-server API (no API key, CORS-enabled, so it works from a static site):

- **Browser** — the playground's corpus card has a dataset picker (TinyStories,
  Simple English Wikipedia, WikiText, English quotes). Pick one, click *Load*.
- **Python** — `dataset_builder.py hf` writes a text file for `train.py --data`:

  ```
  python data/dataset_builder.py hf roneneldan/TinyStories --rows 2000
  python python_ref/train.py --data data/examples/roneneldan_TinyStories.txt
  ```

`roneneldan/TinyStories` — simple, short stories — is the best fit for a tiny
model. Each dataset keeps its own license; the browser app shows it.
