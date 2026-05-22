"""
checkpoint.py — save / resume training state (Phase 1).

A checkpoint is a directory. Saving the optimizer state (Adam's m/v moments) is
what makes a run *truly* resumable — without it, resume restarts the moments
from zero and the loss curve visibly kinks.

Layout (see checkpoints/README.md):
    checkpoint/
      checkpoint.pt          model + optimizer state, step, RNG states
      model_config.json      copy of configs/model.*.json
      training_config.json   copy of configs/training.json
      dataset_manifest.json  dataset identity (hash) the run was trained on
      trainer_state.json     step, best val loss, tokens seen, wall time
      loss_history.json      [{step, train_loss, val_loss}, ...]

The Phase 4 browser port swaps checkpoint.pt for raw weights.f32 / adam_m.f32 /
adam_v.f32 in OPFS; the JSON sidecars carry over unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _to_dict(obj: Any) -> Any:
    return asdict(obj) if is_dataclass(obj) and not isinstance(obj, type) else obj


def save_checkpoint(
    out_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: Any,
    training_config: Any,
    manifest: Any,
    step: int,
    loss_history: list[dict],
    best_val_loss: float,
    tokens_seen: int,
    wall_time: float,
) -> Path:
    """Write a full, resumable checkpoint directory. Returns the directory path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "torch_rng_state": torch.get_rng_state(),
        },
        out_dir / "checkpoint.pt",
    )

    _write_json(out_dir / "model_config.json", _to_dict(model_config))
    _write_json(out_dir / "training_config.json", _to_dict(training_config))
    _write_json(out_dir / "dataset_manifest.json", _to_dict(manifest))
    _write_json(out_dir / "loss_history.json", loss_history)
    _write_json(
        out_dir / "trainer_state.json",
        {
            "step": step,
            "best_val_loss": best_val_loss,
            "tokens_seen": tokens_seen,
            "wall_time_sec": round(wall_time, 2),
        },
    )
    return out_dir


def load_checkpoint(
    ckpt_dir: str | Path, map_location: str | torch.device = "cpu"
) -> dict:
    """Load a checkpoint directory into a plain dict (state dicts + sidecars)."""
    ckpt_dir = Path(ckpt_dir)
    blob = torch.load(ckpt_dir / "checkpoint.pt", map_location=map_location, weights_only=False)
    return {
        "model": blob["model"],
        "optimizer": blob["optimizer"],
        "step": blob["step"],
        "torch_rng_state": blob.get("torch_rng_state"),
        "model_config": _read_json(ckpt_dir / "model_config.json"),
        "training_config": _read_json(ckpt_dir / "training_config.json"),
        "manifest": _read_json(ckpt_dir / "dataset_manifest.json"),
        "loss_history": _read_json(ckpt_dir / "loss_history.json", default=[]),
        "trainer_state": _read_json(ckpt_dir / "trainer_state.json", default={}),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())
