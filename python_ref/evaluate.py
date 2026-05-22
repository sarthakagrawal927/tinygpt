"""
evaluate.py — the evaluation comparison matrix (Phase 9 / milestone 4).

docs/evaluation.md requires four comparisons for every held-out prompt:

    A. Base model only            — generic
    B. Base + few-shot prompt     — style from in-context examples
    C. Base + LoRA adapter        — style baked into an adapter
    D. Base + LoRA + retrieval    — adapter for style, retrieval for context

plus the memorization check: feed a training prefix back in and measure how
much of the continuation is reproduced verbatim.

The deliverable is the *harness* — a reproducible apples-to-apples comparison.
Output quality is limited by the tiny 0.8M model; what matters is that the four
conditions are produced identically and the memorization risk is measured.

Usage:
    python python_ref/evaluate.py --base checkpoints/base --adapter checkpoints/adapter
    python python_ref/evaluate.py --base checkpoints/base --adapter checkpoints/adapter \\
        --corpus data/examples/tiny-corpus-2.txt

Guide: docs/evaluation.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lora import apply_adapter, load_base
from model import TinyGPT
from sample import generate

REPO = Path(__file__).resolve().parent.parent

DEFAULT_PROMPTS = [
    "The ",
    "A model ",
    "When the ",
    "Work ",
]


def trigrams(text: str) -> set[str]:
    return {text[i : i + 3] for i in range(len(text) - 2)}


class Retriever:
    """A tiny character-trigram retriever — stands in for a real vector store.

    The corpus is cut into overlapping chunks; a query returns the chunk with
    the most shared trigrams. LoRA teaches *style*; retrieval supplies *context*.
    """

    def __init__(self, corpus: str, chunk_size: int = 160):
        step = max(1, chunk_size // 2)
        self.chunks = [
            corpus[i : i + chunk_size] for i in range(0, max(1, len(corpus) - step), step)
        ]
        self.chunk_grams = [trigrams(c) for c in self.chunks]

    def retrieve(self, query: str) -> str:
        q = trigrams(query)
        if not q or not self.chunks:
            return ""
        scores = [len(q & g) for g in self.chunk_grams]
        return self.chunks[max(range(len(scores)), key=scores.__getitem__)]


def longest_verbatim_match(generated: str, corpus: str) -> int:
    """Length of the longest substring of `generated` that also occurs in `corpus`."""
    best = 0
    for i in range(len(generated)):
        j = i + best + 1
        while j <= len(generated) and generated[i:j] in corpus:
            best = j - i
            j += 1
    return best


def memorization_check(model: TinyGPT, corpus: str, device, n_probes: int = 3) -> dict:
    """Feed training prefixes back in; measure verbatim reproduction."""
    matches = []
    step = max(1, len(corpus) // (n_probes + 1))
    for p in range(n_probes):
        start = p * step
        prefix = corpus[start : start + 40]
        if len(prefix) < 40:
            break
        out = generate(model, prefix, max_new_tokens=80, temperature=0.2,
                       top_k=20, seed=p, device=device)
        continuation = out[len(prefix):]
        matches.append(longest_verbatim_match(continuation, corpus))
    worst = max(matches) if matches else 0
    return {
        "longest_verbatim_chars": worst,
        # >40 verbatim chars on a tiny corpus is a copying red flag.
        "verdict": "copying risk" if worst > 40 else "ok",
    }


def run_matrix(base: TinyGPT, lora: TinyGPT, retriever: Retriever,
               few_shot: str, prompts: list[str], device) -> None:
    def gen(model: TinyGPT, text: str) -> str:
        return generate(model, text, max_new_tokens=90, temperature=0.8,
                        top_k=40, seed=0, device=device)

    for prompt in prompts:
        print(f"\n=== prompt: {prompt!r} ===")
        print(f"  A base          : {gen(base, prompt)!r}")
        print(f"  B few-shot      : {gen(base, few_shot + prompt)!r}")
        print(f"  C LoRA          : {gen(lora, prompt)!r}")
        ctx = retriever.retrieve(prompt)
        d_out = gen(lora, ctx + "\n" + prompt)
        print(f"  D LoRA+retrieval: {d_out!r}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LoRA evaluation matrix (milestone 4).")
    p.add_argument("--base", required=True, help="base model checkpoint directory")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory")
    p.add_argument("--corpus", default=str(REPO / "data" / "examples" / "tiny-corpus-2.txt"),
                   help="the style/context corpus the adapter was trained on")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    corpus = Path(args.corpus).read_text()
    device = torch.device(args.device)

    base, _ = load_base(args.base, device)
    base.eval()
    lora, _ = load_base(args.base, device)
    apply_adapter(lora, args.adapter, device)
    lora.to(device).eval()

    retriever = Retriever(corpus)
    # Few-shot context: two short excerpts as in-context demonstrations.
    few_shot = retriever.chunks[0][:120] + "\n" + retriever.chunks[len(retriever.chunks) // 2][:120] + "\n"

    print("Evaluation matrix — base / few-shot / LoRA / LoRA+retrieval")
    print(f"base={args.base}  adapter={args.adapter}  corpus={Path(args.corpus).name}")
    run_matrix(base, lora, retriever, few_shot, DEFAULT_PROMPTS, device)

    print("\n=== memorization check (LoRA model) ===")
    mem = memorization_check(lora, corpus, device)
    print(f"  longest verbatim continuation: {mem['longest_verbatim_chars']} chars"
          f"  -> {mem['verdict']}")
    print("\nNote: a 0.8M byte model on a few KB of text produces rough text — "
          "the harness, not the prose, is the deliverable.")


if __name__ == "__main__":
    main()
