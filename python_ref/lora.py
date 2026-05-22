"""
lora.py — LoRA fine-tuning reference (Phase 3).

LoRA adapts an already-trained, FROZEN base model by training small low-rank
matrices inside selected linear layers:

    y = base(x) + (alpha / rank) * dropout(x) @ A @ B

    A  [d_in, rank]   trainable, small random init
    B  [rank, d_out]  trainable, initialised to ZEROS  -> step 0 == base model
    base.weight       frozen (requires_grad=False)

The subtle correctness point (docs/lora_guide.md §6): freezing W does NOT stop
gradients flowing through the layer. base(x) stays in the autograd graph, so
dx = dy @ W.T + scale * dy @ B.T @ A.T still reaches earlier layers — LoRA in
lower blocks only learns because gradient passes THROUGH the frozen weights
above it. A common bug is `.detach()`-ing frozen layers and killing that path.

Result: one base model, many small swappable adapters.

Usage:
    # 1. train a base model first (Phase 1)
    python python_ref/train.py --data data/examples/tiny-corpus.txt --out checkpoints/base
    # 2. LoRA fine-tune onto a different corpus
    python python_ref/lora.py --base checkpoints/base \\
        --data data/examples/tiny-corpus-2.txt --out checkpoints/adapter
    # 3. compare base vs base+adapter
    python python_ref/lora.py --base checkpoints/base --adapter checkpoints/adapter \\
        --compare --prompt "The "

Spec:  configs/lora.json   Guide: docs/lora_guide.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from checkpoint import load_checkpoint
from dataset import ByteDataset, decode, encode
from model import ModelConfig, TinyGPT

REPO = Path(__file__).resolve().parent.parent
LORA_PARAM_SUFFIXES = ("lora_A", "lora_B")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    """One named preset from configs/lora.json (e.g. 'first_run')."""

    rank: int = 4
    alpha: int = 8
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 1e-4
    batch_size: int = 4
    context_length: int = 256
    steps: int = 500

    @classmethod
    def from_json(cls, path: str | Path, preset: str = "first_run") -> "LoRAConfig":
        raw = json.loads(Path(path).read_text())
        if preset not in raw:
            raise SystemExit(f"preset {preset!r} not in {path}; have {list(raw)}")
        block = raw[preset]
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        cfg = cls(**{k: v for k, v in block.items() if k in fields})
        cfg.target_modules = tuple(cfg.target_modules)
        return cfg

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


# --------------------------------------------------------------------------
# LoRA layer
# --------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """A frozen nn.Linear wrapped with a trainable low-rank update.

    Drop-in replacement for the nn.Linear it wraps: same in/out features, same
    call signature. The base weight is frozen; only lora_A / lora_B train.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert rank > 0, "LoRA rank must be positive"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)  # freeze W (and bias) — see module docstring

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        # A: small random so it can learn;  B: zeros so step-0 output == base.
        self.lora_A = nn.Parameter(torch.empty(base.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, base.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (self.lora_dropout(x) @ self.lora_A) @ self.lora_B
        return self.base(x) + self.scale * update


# --------------------------------------------------------------------------
# Injection / freezing / parameter selection
# --------------------------------------------------------------------------
def inject_lora(
    model: nn.Module, target_modules, rank: int, alpha: float, dropout: float = 0.0
) -> list[str]:
    """Replace every nn.Linear whose attribute name is in `target_modules` with a
    LoRALinear. Returns the dotted paths of the replaced modules."""
    targets = set(target_modules)
    injected: list[str] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                injected.append(f"{parent_name}.{child_name}".lstrip("."))
    if not injected:
        raise SystemExit(f"no modules matched target_modules={sorted(targets)}")
    return injected


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze everything except the LoRA adapter parameters."""
    for name, p in model.named_parameters():
        p.requires_grad_(name.endswith(LORA_PARAM_SUFFIXES))


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if n.endswith(LORA_PARAM_SUFFIXES)]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the adapter tensors — never the frozen base (see checkpoints/README)."""
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if n.endswith(LORA_PARAM_SUFFIXES)
    }


def count_params(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# --------------------------------------------------------------------------
# Base model + adapter checkpoint I/O
# --------------------------------------------------------------------------
def load_base(ckpt_dir: str | Path, device: str | torch.device = "cpu") -> tuple[TinyGPT, str]:
    """Load a frozen Phase 1 base model. Returns (model, sha256-of-checkpoint)."""
    ckpt = load_checkpoint(ckpt_dir, map_location=device)
    model = TinyGPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    digest = hashlib.sha256((Path(ckpt_dir) / "checkpoint.pt").read_bytes()).hexdigest()
    return model, digest


def save_adapter(
    out_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lora_cfg: LoRAConfig,
    base_dir: str,
    base_sha: str,
    manifest,
    step: int,
    loss_history: list[dict],
) -> Path:
    """Write an adapter-only checkpoint (see checkpoints/README.md §'LoRA adapter')."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_sd = lora_state_dict(model)

    torch.save(
        {"lora": adapter_sd, "optimizer": optimizer.state_dict(), "step": step},
        out_dir / "adapter.pt",
    )
    meta = {
        "base_model": {"checkpoint": str(base_dir), "sha256": base_sha},
        "adapter": {
            "type": "lora",
            "rank": lora_cfg.rank,
            "alpha": lora_cfg.alpha,
            "dropout": lora_cfg.dropout,
            "target_modules": list(lora_cfg.target_modules),
            "params": sum(t.numel() for t in adapter_sd.values()),
        },
        "training": {
            "learning_rate": lora_cfg.learning_rate,
            "batch_size": lora_cfg.batch_size,
            "context_length": lora_cfg.context_length,
            "step": step,
        },
        "dataset": {
            "dataset_id": manifest.dataset_id,
            "name": manifest.name,
            "token_count": manifest.token_count,
        },
        "loss_history": loss_history,
    }
    (out_dir / "adapter_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def apply_adapter(model: nn.Module, adapter_dir: str | Path,
                  device: str | torch.device = "cpu") -> dict:
    """Inject + load an adapter onto a base model, using adapter_meta.json for the
    rank/alpha/targets. Returns the metadata dict."""
    adapter_dir = Path(adapter_dir)
    meta = json.loads((adapter_dir / "adapter_meta.json").read_text())
    a = meta["adapter"]
    inject_lora(model, a["target_modules"], a["rank"], a["alpha"], a["dropout"])
    blob = torch.load(adapter_dir / "adapter.pt", map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(blob["lora"], strict=False)
    assert not unexpected, f"unexpected adapter keys: {unexpected}"
    return meta


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@torch.no_grad()
def _eval_loss(model, data, ctx, batch_size, device, n_batches=20) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(n_batches)
        for i in range(n_batches):
            x, y = data.get_batch(split, batch_size, ctx, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_lora(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    lora_cfg = LoRAConfig.from_json(args.config, args.preset)

    base, base_sha = load_base(args.base, device)
    # The LoRA context cannot exceed the base model's position embeddings.
    ctx = min(lora_cfg.context_length, base.cfg.context_length)
    if ctx != lora_cfg.context_length:
        print(f"note: clamping context {lora_cfg.context_length} -> {ctx} "
              f"(base model context_length)")

    injected = inject_lora(base, lora_cfg.target_modules, lora_cfg.rank,
                           lora_cfg.alpha, lora_cfg.dropout)
    base.to(device)
    mark_only_lora_trainable(base)
    trainable, total = count_params(base)
    print(f"base:    {args.base}  sha {base_sha[:12]}…")
    print(f"adapter: rank {lora_cfg.rank} alpha {lora_cfg.alpha} "
          f"scale {lora_cfg.scale:g}  ->  {len(injected)} modules: {injected}")
    print(f"params:  {trainable:,} trainable / {total:,} total  "
          f"({100 * trainable / total:.2f}%)")

    data = ByteDataset.from_file(args.data)
    print(f"dataset: {data.manifest.name}  {data.manifest.token_count:,} tokens")

    torch.manual_seed(args.seed)
    optimizer = torch.optim.AdamW(lora_parameters(base), lr=lora_cfg.learning_rate,
                                  betas=(0.9, 0.95))
    steps = args.steps if args.steps is not None else lora_cfg.steps
    eval_every = max(1, steps // 10)

    base.train()
    loss_history: list[dict] = []
    t0 = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            losses = _eval_loss(base, data, ctx, lora_cfg.batch_size, device)
            loss_history.append({"step": step, **{f"{k}_loss": v for k, v in losses.items()}})
            print(f"step {step:>5}  train {losses['train']:.4f}  val {losses['val']:.4f}")
        if step == steps:
            break

        x, y = data.get_batch("train", lora_cfg.batch_size, ctx, device)
        _, loss = base(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(base), 1.0)
        optimizer.step()

    save_adapter(args.out, model=base, optimizer=optimizer, lora_cfg=lora_cfg,
                 base_dir=args.base, base_sha=base_sha, manifest=data.manifest,
                 step=steps, loss_history=loss_history)
    print(f"done in {time.time() - t0:.1f}s. adapter -> {args.out}")


# --------------------------------------------------------------------------
# Compare base vs base+adapter
# --------------------------------------------------------------------------
def compare(args: argparse.Namespace) -> None:
    """Generate from the base model and the base+adapter for the same prompt."""
    device = torch.device(args.device)
    ids = encode(args.prompt).tolist() or [10]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    def gen(model: TinyGPT) -> str:
        g = torch.Generator().manual_seed(args.seed)
        out = model.generate(idx, max_new_tokens=args.tokens,
                             temperature=args.temperature,
                             top_k=args.top_k or None, generator=g)
        return decode(out[0].tolist())

    base, _ = load_base(args.base, device)
    print(f"[base]      {gen(base)!r}")

    base_lora, _ = load_base(args.base, device)
    meta = apply_adapter(base_lora, args.adapter, device)
    base_lora.to(device)
    print(f"[base+LoRA] {gen(base_lora)!r}")
    a = meta["adapter"]
    print(f"\nadapter: rank {a['rank']} alpha {a['alpha']} "
          f"targets {a['target_modules']}  {a['params']:,} params")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for TinyGPT (Phase 3).")
    p.add_argument("--base", required=True, help="base model checkpoint directory")
    p.add_argument("--data", help="text corpus to fine-tune on")
    p.add_argument("--out", default=str(REPO / "checkpoints" / "adapter"))
    p.add_argument("--config", default=str(REPO / "configs" / "lora.json"))
    p.add_argument("--preset", default="first_run", help="preset in configs/lora.json")
    p.add_argument("--steps", type=int, help="override the preset's step count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    # compare mode
    p.add_argument("--compare", action="store_true", help="compare base vs base+adapter")
    p.add_argument("--adapter", help="adapter directory (for --compare)")
    p.add_argument("--prompt", default="")
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args(argv)

    if args.compare:
        if not args.adapter:
            raise SystemExit("--compare needs --adapter <dir>")
        compare(args)
    else:
        if not args.data:
            raise SystemExit("training needs --data <corpus.txt>")
        train_lora(args)


if __name__ == "__main__":
    main()
