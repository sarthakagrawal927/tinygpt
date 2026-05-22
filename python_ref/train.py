"""
train.py — training loop for TinyGPT (Phase 1).

    for step in range(max_steps):
        x, y   = get_batch("train")
        logits, loss = model(x, y)
        optimizer.zero_grad(); loss.backward()
        clip_grad_norm(model.parameters(), grad_clip); optimizer.step()
        eval / sample / checkpoint on their intervals

Optimizer: AdamW, betas (0.9, 0.95), eps 1e-8, weight_decay 0.1. Weight decay is
applied only to >=2D tensors (matrices/embeddings), not to biases or LayerNorm
gains — decaying a bias toward zero has no useful regularizing effect.

Debugging expectations:
    random model        -> loss near ln(256) ~= 5.54
    repeated tiny data  -> loss falls fast
    loss does not fall  -> bug in model / backprop / data
    loss becomes NaN    -> learning rate, softmax, grad explosion, or bad init

Usage:
    python python_ref/train.py --data data/examples/<file>.txt
    python python_ref/train.py --data <file>.txt --resume checkpoints/run1
    python python_ref/train.py --overfit            # tiny built-in smoke corpus

Spec:  configs/training.json   Guide: docs/model_guide.md  ("Training loop")
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from checkpoint import load_checkpoint, save_checkpoint
from dataset import ByteDataset, decode
from model import ModelConfig, TinyGPT

REPO = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
    """Mirrors configs/training.json — the spec is the source of truth."""

    batch_size: int = 16
    learning_rate: float = 3e-4
    optimizer: str = "adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_steps: int = 10000
    eval_interval: int = 100
    sample_interval: int = 500
    checkpoint_interval: int = 500
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        raw = json.loads(Path(path).read_text())
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        cfg = cls(**{k: v for k, v in raw.items() if k in fields})
        cfg.betas = tuple(cfg.betas)  # JSON lists -> tuple
        return cfg


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with decay applied only to >=2D parameters (see module docstring)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=cfg.betas, eps=cfg.eps)


@torch.no_grad()
def evaluate(model: TinyGPT, data: ByteDataset, cfg: TrainConfig, device: torch.device,
             n_batches: int = 20) -> dict[str, float]:
    """Average loss over a few fixed-count batches per split."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(n_batches)
        for i in range(n_batches):
            x, y = data.get_batch(split, cfg.batch_size, model.cfg.context_length, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


OVERFIT_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
) * 64


def train(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    model_cfg = ModelConfig.from_json(args.model_config)
    train_cfg = TrainConfig.from_json(args.config)
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps

    torch.manual_seed(train_cfg.seed)

    # ---- data ----------------------------------------------------------
    if args.overfit:
        data = ByteDataset.from_text(OVERFIT_CORPUS, name="overfit-smoke", seed=train_cfg.seed)
    elif args.data:
        data = ByteDataset.from_file(args.data, seed=train_cfg.seed)
    else:
        raise SystemExit("provide --data <file.txt> or --overfit")
    print(f"dataset: {data.manifest.name}  {data.manifest.token_count:,} tokens  "
          f"(train {len(data.train):,} / val {len(data.val):,})  id {data.manifest.dataset_id[:12]}…")

    # ---- model + optimizer --------------------------------------------
    model = TinyGPT(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg)
    print(f"model:   {model_cfg.model_name}  {model.num_params():,} params  device={device}")

    # ---- resume --------------------------------------------------------
    start_step, loss_history, best_val = 0, [], math.inf
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        loss_history = ckpt["loss_history"]
        best_val = ckpt["trainer_state"].get("best_val_loss", math.inf)
        if ckpt["torch_rng_state"] is not None:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        print(f"resumed from {args.resume} at step {start_step}")

    out_dir = Path(args.out)
    ctx = model_cfg.context_length
    model.train()
    t0 = time.time()
    tokens_seen = start_step * train_cfg.batch_size * ctx

    for step in range(start_step, train_cfg.max_steps + 1):
        # --- eval / sample / checkpoint hooks (also fire on step 0) ------
        if step % train_cfg.eval_interval == 0:
            losses = evaluate(model, data, train_cfg, device)
            loss_history.append({"step": step, "train_loss": losses["train"],
                                 "val_loss": losses["val"]})
            best_val = min(best_val, losses["val"])
            dt = time.time() - t0
            tok_s = tokens_seen / dt if dt > 0 else 0.0
            print(f"step {step:>6}  train {losses['train']:.4f}  val {losses['val']:.4f}  "
                  f"{tok_s:,.0f} tok/s")

        if step > start_step and step % train_cfg.sample_interval == 0:
            sample_ids = model.generate(
                torch.tensor([[ord("t")]], device=device), max_new_tokens=80,
                temperature=0.8, top_k=40)
            print(f"  sample: {decode(sample_ids[0].tolist())!r}")
            model.train()

        if step > start_step and step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(
                out_dir, model=model, optimizer=optimizer, model_config=model_cfg,
                training_config=train_cfg, manifest=data.manifest, step=step,
                loss_history=loss_history, best_val_loss=best_val,
                tokens_seen=tokens_seen, wall_time=time.time() - t0)

        if step == train_cfg.max_steps:
            break

        # --- one optimization step --------------------------------------
        x, y = data.get_batch("train", train_cfg.batch_size, ctx, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        tokens_seen += train_cfg.batch_size * ctx

        if not math.isfinite(loss.item()):
            raise SystemExit(f"loss became {loss.item()} at step {step} — see "
                             "docs/model_guide.md §6 (lower LR / check init).")

    save_checkpoint(
        out_dir, model=model, optimizer=optimizer, model_config=model_cfg,
        training_config=train_cfg, manifest=data.manifest, step=train_cfg.max_steps,
        loss_history=loss_history, best_val_loss=best_val,
        tokens_seen=tokens_seen, wall_time=time.time() - t0)
    print(f"done. best val loss {best_val:.4f}. checkpoint -> {out_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train TinyGPT (Phase 1).")
    p.add_argument("--data", help="path to a plain-text training corpus")
    p.add_argument("--overfit", action="store_true",
                   help="train on a tiny built-in corpus (the overfit smoke test)")
    p.add_argument("--model-config", default=str(REPO / "configs" / "model.byte-tinygpt-v0.json"))
    p.add_argument("--config", default=str(REPO / "configs" / "training.json"))
    p.add_argument("--out", default=str(REPO / "checkpoints" / "run"))
    p.add_argument("--resume", help="checkpoint directory to resume from")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--max-steps", type=int, help="override configs/training.json max_steps")
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
