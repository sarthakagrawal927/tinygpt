"""
sample.py — text generation / sampling from a trained TinyGPT (Phase 1).

Autoregressive decoding:
    prompt bytes -> token ids
    repeat: logits = model(tokens[-context_length:])
            next  = sample(logits[-1], temperature, top_k)
            append next
    decode tokens -> text

Sampling controls:
    temperature   scales logits before softmax (lower = greedier; 0 = argmax)
    top_k         restrict sampling to the k most likely tokens

Determinism: with a fixed --seed, generation is reproducible (the RNG is an
explicit torch.Generator, not global state).

Usage:
    python python_ref/sample.py --checkpoint checkpoints/run --prompt "the "
    python python_ref/sample.py --checkpoint checkpoints/run --temperature 0 --tokens 200

Guide: docs/model_guide.md  Tests: tests/README.md ("Sampling fixed seed")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from dataset import decode, encode
from model import ModelConfig, TinyGPT

REPO = Path(__file__).resolve().parent.parent


def load_model(ckpt_dir: str | Path, device: str | torch.device = "cpu") -> TinyGPT:
    """Rebuild a TinyGPT from a checkpoint directory and load its weights."""
    ckpt = load_checkpoint(ckpt_dir, map_location=device)
    model = TinyGPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def generate(
    model: TinyGPT,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> str:
    """Generate a continuation of `prompt`. Returns prompt + completion as text."""
    # An empty prompt still needs a seed token; byte 10 ('\n') is a safe start.
    ids = encode(prompt).tolist() or [10]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = model.generate(
        idx, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, generator=generator)
    return decode(out[0].tolist())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sample from a trained TinyGPT.")
    p.add_argument("--checkpoint", default=str(REPO / "checkpoints" / "run"))
    p.add_argument("--prompt", default="")
    p.add_argument("--tokens", type=int, default=200, help="max new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy/argmax")
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    model = load_model(args.checkpoint, device=args.device)
    text = generate(
        model, args.prompt, max_new_tokens=args.tokens,
        temperature=args.temperature, top_k=args.top_k or None,
        seed=args.seed, device=args.device)
    print(text)


if __name__ == "__main__":
    main()
