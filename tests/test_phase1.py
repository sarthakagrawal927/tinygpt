"""
test_phase1.py — required correctness tests for the Phase 1 Python reference.

Covers the checks from tests/README.md that apply to the PyTorch reference:
tokenizer roundtrip, layer shapes, loss sanity, tiny overfit, gradient check,
checkpoint reload, and deterministic sampling.

Run directly (no pytest needed):
    python tests/test_phase1.py
Or under pytest (every test_* function is also a pytest case):
    pytest tests/test_phase1.py

The one that matters most is test_tiny_overfit — if a tiny model cannot drive
loss down on a few KB of repeated text, the model/backprop/data is broken.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python_ref"))

import torch  # noqa: E402

from checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from dataset import ByteDataset, decode, encode  # noqa: E402
from model import ModelConfig, TinyGPT  # noqa: E402
from train import TrainConfig, build_optimizer  # noqa: E402

# A small, fast config for tests — proves correctness without the full 0.8M run.
TEST_CFG = ModelConfig(
    vocab_size=256, context_length=64, n_layers=2, n_heads=2, d_model=64, d_mlp=256
)


def test_tokenizer_roundtrip() -> None:
    """bytes -> text -> bytes is lossless, including non-ASCII UTF-8."""
    for text in ["hello world", "café — naïve — 日本語 🐟", "\n\t mixed \x00 bytes"]:
        assert decode(encode(text)) == text, f"roundtrip failed for {text!r}"
    print("ok  tokenizer roundtrip")


def test_layer_shapes() -> None:
    """Every layer returns the documented shape; loss is a scalar."""
    cfg = TEST_CFG
    model = TinyGPT(cfg)
    B, T = 3, cfg.context_length
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    emb = model.token_embedding(idx)
    assert emb.shape == (B, T, cfg.d_model), emb.shape
    logits, loss = model(idx, idx)
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape
    assert loss.ndim == 0, "loss must be a scalar"
    print("ok  layer shapes")


def test_param_count() -> None:
    """The shipped config sits near the documented ~0.8M parameters."""
    model = TinyGPT(ModelConfig.from_json(REPO / "configs" / "model.byte-tinygpt-v0.json"))
    n = model.num_params()
    assert 0.7e6 < n < 1.0e6, f"expected ~0.8M params, got {n:,}"
    print(f"ok  param count ({n:,})")


def test_loss_sanity() -> None:
    """A random (untrained) model's loss sits near ln(vocab_size).

    Targets are independent of the inputs. Using same-position inputs as targets
    would let the model 'cheat': residual connections + tied embeddings leave the
    input token's embedding in the output, so logits favor it and loss drops
    below ln(vocab). Real batches use shifted next-token targets, which on random
    data the model genuinely cannot predict — hence loss == ln(vocab)."""
    cfg = TEST_CFG
    torch.manual_seed(0)
    model = TinyGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (8, cfg.context_length))
    targets = torch.randint(0, cfg.vocab_size, (8, cfg.context_length))
    _, loss = model(idx, targets)
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 0.5, f"loss {loss.item():.3f} far from ln(vocab) {expected:.3f}"
    print(f"ok  loss sanity ({loss.item():.3f} ~ {expected:.3f})")


def test_tiny_overfit() -> None:
    """THE test that matters: the model drives loss far down on repeated text."""
    cfg = TEST_CFG
    torch.manual_seed(42)
    corpus = "the quick brown fox jumps over the lazy dog. " * 50
    data = ByteDataset.from_text(corpus, name="overfit-test")
    model = TinyGPT(cfg)
    opt = build_optimizer(model, TrainConfig(learning_rate=1e-3))

    _, first = model(*data.get_batch("train", 8, cfg.context_length))
    for _ in range(300):
        x, y = data.get_batch("train", 8, cfg.context_length)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    assert loss.item() < first.item() * 0.3, (
        f"loss did not fall enough: {first.item():.3f} -> {loss.item():.3f}")
    assert loss.item() < 1.0, f"final loss {loss.item():.3f} should be well under 1.0"
    print(f"ok  tiny overfit ({first.item():.3f} -> {loss.item():.3f})")


def test_gradient_check() -> None:
    """Finite-difference vs autograd on the attention module (float64, the
    standard torch.autograd.gradcheck). Catches a wrong backward formula."""
    torch.manual_seed(0)
    from model import CausalSelfAttention

    cfg = ModelConfig(context_length=8, n_heads=2, d_model=16, d_mlp=64)
    attn = CausalSelfAttention(cfg).double()
    x = torch.randn(1, 8, cfg.d_model, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(attn, (x,), atol=1e-4), "gradcheck failed"
    print("ok  gradient check")


def test_checkpoint_reload() -> None:
    """Loss is identical (bit-for-bit) after save + reload."""
    cfg = TEST_CFG
    torch.manual_seed(7)
    model = TinyGPT(cfg)
    opt = build_optimizer(model, TrainConfig())
    data = ByteDataset.from_text("checkpoint reload test corpus. " * 40, name="ckpt")
    x, y = data.get_batch("train", 4, cfg.context_length)

    model.eval()
    with torch.no_grad():
        _, loss_before = model(x, y)

    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(
            tmp, model=model, optimizer=opt, model_config=cfg,
            training_config=TrainConfig(), manifest=data.manifest, step=123,
            loss_history=[{"step": 0, "train_loss": 5.5, "val_loss": 5.5}],
            best_val_loss=5.5, tokens_seen=999, wall_time=1.0)
        ckpt = load_checkpoint(tmp)
        reloaded = TinyGPT(cfg)
        reloaded.load_state_dict(ckpt["model"])
        reloaded.eval()
        with torch.no_grad():
            _, loss_after = reloaded(x, y)
        assert ckpt["step"] == 123

    assert torch.equal(loss_before, loss_after), (
        f"loss changed across reload: {loss_before.item()} != {loss_after.item()}")
    print("ok  checkpoint reload")


def test_sampling_fixed_seed() -> None:
    """Generation is reproducible with a fixed seed, and varies without one."""
    cfg = TEST_CFG
    torch.manual_seed(0)
    model = TinyGPT(cfg)
    prompt = torch.tensor([[ord("a")]])

    def gen(seed: int) -> list[int]:
        g = torch.Generator().manual_seed(seed)
        return model.generate(prompt, 30, temperature=0.9, top_k=20, generator=g)[0].tolist()

    assert gen(123) == gen(123), "same seed must produce identical output"
    assert gen(123) != gen(456), "different seeds should differ"

    # Greedy decoding (temperature 0) is deterministic with no RNG at all.
    greedy = lambda: model.generate(prompt, 20, temperature=0.0)[0].tolist()
    assert greedy() == greedy(), "greedy decoding must be deterministic"
    print("ok  sampling fixed seed")


ALL_TESTS = [
    test_tokenizer_roundtrip,
    test_layer_shapes,
    test_param_count,
    test_loss_sanity,
    test_tiny_overfit,
    test_gradient_check,
    test_checkpoint_reload,
    test_sampling_fixed_seed,
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
