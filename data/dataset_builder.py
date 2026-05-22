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
    python data/dataset_builder.py hf      <hf-dataset-id> [--config .. --rows ..]

The `hf` command pulls plain text from a Hugging Face dataset via the public
datasets-server HTTP API — no API key and no `datasets` dependency — so you can
train on an open dataset instead of supplying your own file.

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


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def build_from_hf(dataset: str, config: str, split: str, text_column: str,
                  rows: int, out_dir: str | Path) -> Path:
    """Pull text rows from a Hugging Face dataset into a plain-text file.

    Uses the public datasets-server HTTP API — no API key, no `datasets`
    dependency. The written file is ready for `train.py --data`.
    """
    import urllib.parse
    import urllib.request

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collected: list[str] = []
    offset, page = 0, 100
    while offset < rows:
        n = min(page, rows - offset)
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split,
             "offset": offset, "length": n})
        with urllib.request.urlopen(f"{HF_ROWS_URL}?{query}", timeout=30) as resp:
            payload = json.loads(resp.read())
        batch = payload.get("rows", [])
        if not batch:
            break
        for item in batch:
            value = item.get("row", {}).get(text_column)
            if isinstance(value, str) and value.strip():
                collected.append(value.strip())
        offset += len(batch)
        if len(batch) < n:
            break

    if not collected:
        raise SystemExit(f"no '{text_column}' text rows returned for {dataset}")

    text = "\n\n".join(collected)
    out_path = out_dir / f"{dataset.replace('/', '_')}.txt"
    out_path.write_text(text)
    print(f"hf      -> {out_path}  ({len(text):,} chars from {len(collected)} rows)")
    print(f"         train: python python_ref/train.py --data {out_path}")
    return out_path


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

    p_hf = sub.add_parser("hf", help="pull a Hugging Face dataset into a text file")
    p_hf.add_argument("dataset", help="HF dataset id, e.g. roneneldan/TinyStories")
    p_hf.add_argument("--config", default="default")
    p_hf.add_argument("--split", default="train")
    p_hf.add_argument("--text-column", default="text")
    p_hf.add_argument("--rows", type=int, default=2000, help="max rows to pull")
    p_hf.add_argument("--out-dir", default="data/examples")

    args = parser.parse_args(argv)
    if args.cmd == "tokens":
        build_token_array(args.input, args.out_dir)
    elif args.cmd == "jsonl":
        build_jsonl(args.input, args.out_dir, args.val_split)
    elif args.cmd == "hf":
        build_from_hf(args.dataset, args.config, args.split, args.text_column,
                      args.rows, args.out_dir)


if __name__ == "__main__":
    main()
