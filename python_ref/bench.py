"""
bench.py — measure training throughput on this machine (Phase 1 utility).

Times forward + backward + optimizer step for a range of model sizes on the
best available device (CUDA, Apple MPS, or CPU). Use it to see how large a
model is practical to train locally *before* committing to a real run — it is
the native-training counterpart of the browser's machine detector
(`browser/src/runtime_detect.ts`).

Usage:
    python python_ref/bench.py
    python python_ref/bench.py --device cpu --batch 16 --steps 10

Reading the output: "ms/step" is wall time per training step; "1k steps" /
"5k steps" extrapolate that to a short and a medium run. A model is comfortable
to "test things out" with if a few-thousand-step run finishes in a few minutes.
"""

from __future__ import annotations

import argparse
import time

import torch

from model import ModelConfig, TinyGPT
from train import pick_device

# (label, d_model, n_layers) — head_dim is fixed at 64, so n_heads = d_model/64.
SIZES = [
    ("tiny", 192, 6),
    ("small", 384, 6),
    ("medium", 512, 8),
    ("large", 768, 10),
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench_one(d_model: int, n_layers: int, device: torch.device,
              batch: int, ctx: int, steps: int) -> tuple[int, float]:
    """Return (parameter count, seconds per training step)."""
    cfg = ModelConfig(n_layers=n_layers, n_heads=d_model // 64, d_model=d_model,
                      d_mlp=4 * d_model, context_length=ctx)
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, 256, (batch, ctx), device=device)

    for _ in range(3):  # warm up (first steps pay compile/allocation costs)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, x)
        loss.backward()
        opt.step()
    _sync(device)

    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, x)
        loss.backward()
        opt.step()
    _sync(device)
    return model.num_params(), (time.time() - t0) / steps


def fmt_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Benchmark TinyGPT training throughput.")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--steps", type=int, default=20, help="timed steps per model")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    print(f"device={device}  batch={args.batch}  ctx={args.ctx}  "
          f"({args.steps} timed steps per model)\n")
    print(f"{'model':8} {'params':>9} {'ms/step':>9} {'tok/s':>10} "
          f"{'1k steps':>9} {'5k steps':>9}")
    print("-" * 60)
    for label, d_model, n_layers in SIZES:
        params, dt = bench_one(d_model, n_layers, device, args.batch, args.ctx,
                               args.steps)
        print(f"{label:8} {params / 1e6:7.1f}M {dt * 1000:8.1f} "
              f"{args.batch * args.ctx / dt:10,.0f} "
              f"{fmt_time(dt * 1000):>9} {fmt_time(dt * 5000):>9}")
    print("\nRule of thumb: a model is comfortable for quick iteration if a "
          "few-thousand-step\nrun finishes in a few minutes. Memory is rarely "
          "the limit; speed is.")


if __name__ == "__main__":
    main()
