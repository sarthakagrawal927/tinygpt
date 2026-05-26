# Feature ideas — interactive learning content

A backlog of concrete features for the browser TinyGPT, drawn from the famous
educational GPT projects worth learning from. Source repos are filtered to
**5,000+ GitHub stars**; blogs to well-known authors only. This doc is a menu —
pick from it per milestone; it does not change the build order in `README.md`.

## Reference projects

### Repos (5k+ stars)

| Repo | Stars | What to take from it |
| --- | --- | --- |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | ~58k | Architecture + training-loop shape; the closest cousin to `python_ref/`. Char-level Shakespeare as the default browser task. |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | ~54k | Full pipeline framing: pretrain → fine-tune → chat. A single "depth dial" UX. A capstone chat UI. |
| [karpathy/minGPT](https://github.com/karpathy/minGPT) | ~24k | Toy algorithmic tasks (sort/copy/reverse) — converge in seconds, exactly verifiable. |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | ~30k | "PyTorch reference and C reference produce identical outputs" testing discipline → WASM-vs-Python parity. |
| [karpathy/llama2.c](https://github.com/karpathy/llama2.c) | ~20k | Single-file, dependency-free `run.c` — the model to follow for the Emscripten/WASM build. |
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | ~16k | A "Lesson 0: autograd" — clickable computation graph, gradients flowing backward. |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | ~95k | Curriculum shape: one self-contained, runnable lesson per component. |
| [poloclub/transformer-explainer](https://github.com/poloclub/transformer-explainer) | ~8k | **Closest peer.** Live in-browser GPT-2, temperature slider, next-token bar chart, semantic zoom. |
| [jessevig/bertviz](https://github.com/jessevig/bertviz) | ~8k | Three attention views: head (lines), model (layers×heads grid), neuron (Q·K breakdown). |
| [bbycroft/llm-viz](https://github.com/bbycroft/llm-viz) | ~5k | Guided animated data-flow walkthrough — one token's vector through every matmul. |

### Blogs (famous authors)

- **Jay Alammar** — *The Illustrated Transformer / GPT-2*: a consistent visual vocabulary (color-coded Q/K/V, token vectors as bars). Reuse it across every module.
- **3Blue1Brown** — *GPT* & *Attention* chapters: embedding-space intuition; attention as tokens updating each other.
- **The Annotated Transformer** (Harvard NLP): interleave runnable code with prose — the model for the Milestone 9 write-up.
- **Lil'Log** (Lilian Weng) — *Attention? Attention!*: rigor and diagram reference for the written notes.
- **distill.pub** (Chris Olah): the north-star interaction model — reactive diagrams that live-update on input.
- **Jay Mody** — *GPT in 60 Lines of NumPy*: the "whole forward pass fits on one screen" hook.

> Note: the user's "fishGPT" reference has no notable (5k+) educational repo —
> GitHub "FishGPT" projects are fish-image classifiers, unrelated. Treat it as a
> stand-in for the Karpathy tiny-GPT family above.

## Ideas grouped by milestone / difficulty

### Easy — early modules (Milestones 1–2)

1. **Tokenizer playground** — type text, see colored byte/token spans + IDs update live; later toggle char-level vs BPE.
2. **Lesson 0: autograd** — a clickable computation graph; hover a node, watch gradients flow backward (micrograd).
3. **Toy-task training demo** — train on "sort 3 symbols" or "reverse a string"; converges in seconds, exactly checkable (minGPT).
4. **One-screen forward pass** — render `python_ref/model.py`'s forward as a single annotated, scrollable listing.
5. **Next-token bar chart + temperature slider** — type text, watch the output distribution shift (Transformer Explainer).

### Medium — core visualizations (Milestones 4 & 8)

6. **Three-mode attention viewer** — head view (attention lines), model view (layers×heads grid), neuron view (Q·K) (bertviz).
7. **Semantic zoom** — a block diagram that expands into actual matrix math on click (Transformer Explainer).
8. **Animated data-flow walkthrough** — follow one token vector through embedding → attention → MLP → unembedding, with synced narration (llm-viz, 3Blue1Brown).
9. **"All numbers visible" grid** — a spreadsheet-style view of every intermediate value in one transformer layer.
10. **Live metrics dashboard** — train/val loss, tokens/sec, active backend; a single "model depth" dial that retrains (Milestone 8, `browser/src/charts.ts`).
11. **Embedding-space explorer** — 2D/3D projection; drag a vector, see nearest tokens (3Blue1Brown).
12. **Code panel with execution highlighting** — model code beside the visualization, highlighting the executing line.

### Advanced — capstone modules (Milestones 3, 5–7, 9)

13. **In-browser training in a Web Worker** — char-level corpus, overfit live without freezing the UI (Milestone 5).
14. **WASM/WebGPU build** — follow llama2.c's single-file design; add llm.c-style WASM-vs-Python identical-output tests (Milestones 4–6).
15. **LoRA fine-tuning demo** — freeze the base, train a low-rank adapter, A/B base vs fine-tuned output (Milestone 3).
16. **Capstone chat UI** — a minimal chat box so the learner can talk to the model they trained (nanochat).
17. **Paper-annotation write-up** — interleave `python_ref/` code with prose for the final learning notes (Milestone 9).
18. **Pre-trained model gallery (R2-hosted)** — replace the single demo banner with a grid of "Try a model" cards, each a different pre-trained checkpoint. Manifest-driven (`browser/public/models.json`), so URLs can point at local `/demo.tinygpt`, a Cloudflare R2 bucket, or HuggingFace. Cloudflare R2 is the natural backing store: free egress to a CF Pages domain on the same account, S3-compatible upload. **Why this isn't shipped yet:** speed work is the priority — the in-browser kernels need to land before "look at all these working models" carries weight, because every gallery model implicitly promises "you can train one of these yourself" and currently that takes too long on big presets. Revisit once Flash Attention 2 + the in-pipeline blocked matmul make Mega/Behemoth realistic in-browser. Sketch of the entry shape: `{ id, label, blurb, url, bytes, tag }`. Cards show final-loss + corpus + size; click → `loadModelFromFile(fetch(url))`.
