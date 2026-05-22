"""
dataset.py — byte-level dataset pipeline (Phase 1).

Pipeline:
    raw text -> UTF-8 bytes -> integer token array (1 byte = 1 token, vocab 0..255)
             -> train/val split (90/10) -> random batch sampler -> (x, y) pairs

Batch construction (context_length C):
    x = tokens[i : i + C]
    y = tokens[i + 1 : i + 1 + C]

The dataset manifest records a sha256 of the raw bytes; that hash is what makes
checkpoint resume reproducible (a resumed run must see the same data).

Guide: docs/model_guide.md  ("Data requirements", "Dataset pipeline")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

TOKENIZER_ID = "byte-v1"


# --------------------------------------------------------------------------
# Byte tokenizer — every byte of the UTF-8 encoding is one token (vocab 256).
# --------------------------------------------------------------------------
def encode(text: str) -> np.ndarray:
    """UTF-8 text -> uint8 token array."""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def decode(tokens: np.ndarray | list[int]) -> str:
    """Token array -> text. errors='replace' so partial multi-byte tails never crash."""
    return bytes(bytearray(int(t) for t in tokens)).decode("utf-8", errors="replace")


def dataset_id(raw_bytes: bytes) -> str:
    """sha256 of the raw file bytes — the stable identity of a dataset."""
    return hashlib.sha256(raw_bytes).hexdigest()


# --------------------------------------------------------------------------
# Manifest — written alongside a token array; see docs/model_guide.md.
# --------------------------------------------------------------------------
@dataclass
class DatasetManifest:
    dataset_id: str
    name: str
    raw_bytes: int
    token_count: int
    tokenizer: str = TOKENIZER_ID
    train_split: float = 0.9
    val_split: float = 0.1
    seed: int = 42

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> "DatasetManifest":
        return cls(**json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------
# Dataset — holds the token array and serves random (x, y) batches.
# --------------------------------------------------------------------------
class ByteDataset:
    """A byte-tokenized corpus split into a train and a val region.

    The split is contiguous (first 90% train, last 10% val) so val text is never
    seen during training — important for the overfit / generalization tests.
    """

    def __init__(self, tokens: np.ndarray, manifest: DatasetManifest):
        self.manifest = manifest
        tokens = np.asarray(tokens, dtype=np.uint8)
        n_train = int(len(tokens) * manifest.train_split)
        self.train = tokens[:n_train]
        self.val = tokens[n_train:]
        if len(self.val) == 0:  # tiny smoke-test corpora: reuse train as val
            self.val = self.train

    @classmethod
    def from_text(cls, text: str, name: str = "inline", **manifest_kwargs) -> "ByteDataset":
        raw = text.encode("utf-8")
        tokens = encode(text)
        manifest = DatasetManifest(
            dataset_id=dataset_id(raw),
            name=name,
            raw_bytes=len(raw),
            token_count=len(tokens),
            **manifest_kwargs,
        )
        return cls(tokens, manifest)

    @classmethod
    def from_file(cls, path: str | Path, **manifest_kwargs) -> "ByteDataset":
        path = Path(path)
        raw = path.read_bytes()
        tokens = np.frombuffer(raw, dtype=np.uint8).copy()
        manifest = DatasetManifest(
            dataset_id=dataset_id(raw),
            name=path.name,
            raw_bytes=len(raw),
            token_count=len(tokens),
            **manifest_kwargs,
        )
        return cls(tokens, manifest)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        context_length: int,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (x, y) windows. Returns int64 tensors [B, C]."""
        data = self.train if split == "train" else self.val
        max_start = len(data) - context_length - 1
        assert max_start >= 1, (
            f"{split} split has {len(data)} tokens — need > context_length+1 "
            f"({context_length + 1}); use a larger corpus or smaller context"
        )
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)

        # uint8 -> int64 because Embedding indices and cross_entropy targets need int64.
        data_t = torch.from_numpy(data.astype(np.int64))
        x = torch.stack([data_t[s : s + context_length] for s in starts])
        y = torch.stack([data_t[s + 1 : s + 1 + context_length] for s in starts])
        return x.to(device), y.to(device)


if __name__ == "__main__":
    # Smoke check: tokenizer roundtrip + a batch shape.
    sample = "Hello, TinyGPT! " * 64
    ds = ByteDataset.from_text(sample, name="smoke")
    assert decode(encode(sample)) == sample, "tokenizer roundtrip failed"
    x, y = ds.get_batch("train", batch_size=4, context_length=32)
    print(f"dataset_id={ds.manifest.dataset_id[:12]}…  tokens={ds.manifest.token_count}")
    print(f"batch x={tuple(x.shape)} y={tuple(y.shape)} dtype={x.dtype}")
