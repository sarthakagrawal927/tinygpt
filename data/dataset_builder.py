"""
dataset_builder.py — turn raw text into training-ready data (Phase 1 / Phase 6).

Two outputs:

1. From-scratch training (Phase 1):
   raw text -> UTF-8 bytes -> token array (.bin) + dataset manifest (sha256 hash).

2. LoRA fine-tuning (Phase 6):
   raw author text -> structured task examples -> JSONL, deduped + train/val split.

CLI:
    python data/dataset_builder.py tokens  <input.txt> [--out-dir data/examples]
    python data/dataset_builder.py jsonl   <examples.jsonl> [--out-dir data/examples]

Guide: docs/model_guide.md ("Dataset pipeline"), docs/lora_guide.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# python_ref/ is a sibling of data/ — make its tokenizer importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python_ref"))
from dataset import DatasetManifest, dataset_id  # noqa: E402


def build_token_array(input_path: str | Path, out_dir: str | Path) -> Path:
    """Phase 1: raw text file -> <name>.bin (uint8 tokens) + <name>.manifest.json.

    Returns the path to the written manifest.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = input_path.read_bytes()
    tokens = np.frombuffer(raw, dtype=np.uint8)

    stem = input_path.stem
    bin_path = out_dir / f"{stem}.bin"
    manifest_path = out_dir / f"{stem}.manifest.json"

    bin_path.write_bytes(tokens.tobytes())
    manifest = DatasetManifest(
        dataset_id=dataset_id(raw),
        name=input_path.name,
        raw_bytes=len(raw),
        token_count=len(tokens),
    )
    manifest.write(manifest_path)

    print(f"tokens  -> {bin_path}  ({len(tokens):,} tokens)")
    print(f"manifest-> {manifest_path}  (id {manifest.dataset_id[:12]}…)")
    return manifest_path


def build_jsonl(input_path: str | Path, out_dir: str | Path, val_split: float = 0.1) -> Path:
    """Phase 6: clean a JSONL of LoRA task examples — dedup, train/val split, hash.

    Each input line is a JSON object with a "task" field (continuation / rewrite /
    title / qa). Exact-duplicate lines are dropped; the rest are shuffled
    deterministically and split into <stem>.train.jsonl / <stem>.val.jsonl.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    examples: list[dict] = []
    for line in input_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        key = json.dumps(obj, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        examples.append(obj)

    # Deterministic shuffle keyed on the file's content hash.
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    rng.shuffle(examples)

    n_val = max(1, int(len(examples) * val_split)) if len(examples) > 1 else 0
    val, train = examples[:n_val], examples[n_val:]

    stem = input_path.stem
    train_path = out_dir / f"{stem}.train.jsonl"
    val_path = out_dir / f"{stem}.val.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)

    print(f"jsonl   -> {train_path} ({len(train)})  {val_path} ({len(val)})")
    print(f"         deduped {len(seen)} unique of total, id {digest[:12]}…")
    return train_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build TinyGPT training data.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tokens = sub.add_parser("tokens", help="Phase 1: text file -> token array + manifest")
    p_tokens.add_argument("input")
    p_tokens.add_argument("--out-dir", default="data/examples")

    p_jsonl = sub.add_parser("jsonl", help="Phase 6: clean LoRA examples JSONL")
    p_jsonl.add_argument("input")
    p_jsonl.add_argument("--out-dir", default="data/examples")
    p_jsonl.add_argument("--val-split", type=float, default=0.1)

    args = parser.parse_args(argv)
    if args.cmd == "tokens":
        build_token_array(args.input, args.out_dir)
    elif args.cmd == "jsonl":
        build_jsonl(args.input, args.out_dir, args.val_split)


if __name__ == "__main__":
    main()
