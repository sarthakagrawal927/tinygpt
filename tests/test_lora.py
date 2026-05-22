"""
test_lora.py — required correctness tests for LoRA fine-tuning (Phase 3).

Covers the LoRA-specific table in tests/README.md:
    Adapter step-0     with B = 0, base + LoRA output == base output
    Frozen-base grads  gradients still flow THROUGH frozen layers to LoRA
plus: trainable-param reduction, adapter save/load roundtrip, and that a trained
adapter actually changes the model's output.

Run directly:   python tests/test_lora.py
Or via pytest:  pytest tests/test_lora.py
"""

from __future__ import annotations

import sys
import tempfile

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python_ref"))

import torch  # noqa: E402

from dataset import ByteDataset  # noqa: E402
from lora import (  # noqa: E402
    LoRAConfig,
    LoRALinear,
    apply_adapter,
    count_params,
    inject_lora,
    lora_parameters,
    mark_only_lora_trainable,
    save_adapter,
)
from model import ModelConfig, TinyGPT  # noqa: E402

TEST_CFG = ModelConfig(
    vocab_size=256, context_length=64, n_layers=3, n_heads=2, d_model=64, d_mlp=256
)


def _fresh_model() -> TinyGPT:
    torch.manual_seed(0)
    return TinyGPT(TEST_CFG)


def _batch():
    torch.manual_seed(1)
    return torch.randint(0, TEST_CFG.vocab_size, (4, TEST_CFG.context_length))


def test_injection_targets() -> None:
    """inject_lora replaces exactly the named modules — one per block."""
    model = _fresh_model()
    injected = inject_lora(model, ["q_proj", "v_proj"], rank=4, alpha=8)
    assert len(injected) == 2 * TEST_CFG.n_layers, injected
    assert all(isinstance(m, LoRALinear)
               for m in model.modules() if isinstance(m, LoRALinear))
    # untargeted projections stay plain nn.Linear
    assert not isinstance(model.blocks[0].attn.k_proj, LoRALinear)
    print(f"ok  injection targets ({len(injected)} modules)")


def test_adapter_step_zero() -> None:
    """B initialised to zeros => base + LoRA output is identical to the base."""
    idx = _batch()
    model = _fresh_model()
    model.eval()
    with torch.no_grad():
        base_logits, _ = model(idx)

    inject_lora(model, ["q_proj", "v_proj", "o_proj"], rank=4, alpha=8)
    model.eval()
    with torch.no_grad():
        lora_logits, _ = model(idx)

    assert torch.equal(base_logits, lora_logits), "step-0 adapter changed the output"
    print("ok  adapter step-0 (base + LoRA == base)")


def test_trainable_param_reduction() -> None:
    """Only adapter params train; they are a small fraction of the total."""
    model = _fresh_model()
    inject_lora(model, ["q_proj", "v_proj"], rank=4, alpha=8)
    mark_only_lora_trainable(model)
    trainable, total = count_params(model)

    for name, p in model.named_parameters():
        is_lora = name.endswith(("lora_A", "lora_B"))
        assert p.requires_grad == is_lora, f"{name}: requires_grad != is-LoRA"
    assert trainable < total * 0.05, f"{trainable}/{total} — adapter not small"
    print(f"ok  trainable-param reduction ({trainable:,}/{total:,} = "
          f"{100 * trainable / total:.2f}%)")


def test_frozen_base_grads() -> None:
    """Gradients flow THROUGH the frozen base weights to the LoRA params.

    The frozen base weights must get NO grad, while the *first* block's adapter
    must — which can only happen if gradient traversed the frozen upper blocks.
    (lora_A.grad is zero at step 0 because dA is proportional to B = 0; lora_B is
    the one that carries signal on the first step.)"""
    model = _fresh_model()
    inject_lora(model, ["q_proj", "v_proj"], rank=4, alpha=8)
    mark_only_lora_trainable(model)

    idx = _batch()
    _, loss = model(idx, idx)
    loss.backward()

    # frozen base weights: requires_grad False and no grad accumulated
    for name, p in model.named_parameters():
        if ".base.weight" in name:
            assert not p.requires_grad and p.grad is None, f"{name} not frozen"

    first = model.blocks[0].attn.v_proj
    assert first.lora_B.grad is not None and first.lora_B.grad.norm() > 0, (
        "first block's adapter got no gradient — path through frozen layers broken")
    assert first.lora_A.grad is not None, "lora_A should still be in the graph"
    print("ok  frozen-base grad flow")


def test_adapter_roundtrip() -> None:
    """Saved adapter + reload reproduces the exact same output."""
    idx = _batch()
    cfg = LoRAConfig(rank=4, alpha=8, dropout=0.0, target_modules=("q_proj", "v_proj"))

    model = _fresh_model()
    inject_lora(model, cfg.target_modules, cfg.rank, cfg.alpha)
    mark_only_lora_trainable(model)
    # perturb B so the adapter actually changes the output (else step-0 identity)
    with torch.no_grad():
        for p in lora_parameters(model):
            p.add_(torch.randn_like(p) * 0.1)
    model.eval()
    with torch.no_grad():
        before, _ = model(idx)

    manifest = ByteDataset.from_text("roundtrip corpus " * 40, name="rt").manifest
    opt = torch.optim.AdamW(lora_parameters(model), lr=1e-4)
    with tempfile.TemporaryDirectory() as tmp:
        save_adapter(tmp, model=model, optimizer=opt, lora_cfg=cfg,
                     base_dir="checkpoints/fake", base_sha="deadbeef",
                     manifest=manifest, step=10, loss_history=[])
        reloaded = _fresh_model()
        apply_adapter(reloaded, tmp)
        reloaded.eval()
        with torch.no_grad():
            after, _ = reloaded(idx)

    assert torch.equal(before, after), "adapter output changed across save/reload"
    print("ok  adapter save/reload roundtrip")


def test_lora_changes_output() -> None:
    """After a few training steps the adapter measurably changes the output."""
    idx = _batch()
    model = _fresh_model()
    model.eval()
    with torch.no_grad():
        base_logits, _ = model(idx)

    inject_lora(model, ["q_proj", "v_proj"], rank=4, alpha=8)
    mark_only_lora_trainable(model)
    data = ByteDataset.from_text("the lazy adapter learns a little. " * 60, name="lc")
    opt = torch.optim.AdamW(lora_parameters(model), lr=1e-3)

    model.train()
    for _ in range(60):
        x, y = data.get_batch("train", 4, TEST_CFG.context_length)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        lora_logits, _ = model(idx)
    delta = (lora_logits - base_logits).abs().max().item()
    assert delta > 1e-3, f"adapter barely changed output (max delta {delta:.2e})"
    print(f"ok  trained adapter changes output (max delta {delta:.3f})")


ALL_TESTS = [
    test_injection_targets,
    test_adapter_step_zero,
    test_trainable_param_reduction,
    test_frozen_base_grads,
    test_adapter_roundtrip,
    test_lora_changes_output,
]


def main() -> int:
    failed = 0
    for test in ALL_TESTS:
        try:
            test()
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {test.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}: {type(e).__name__}: {e}")
    total = len(ALL_TESTS)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
