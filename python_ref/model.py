"""
model.py — byte-level TinyGPT model (Phase 1-2).

The ~0.8M-parameter causal language model from docs/model_guide.md and
configs/model.byte-tinygpt-v0.json.

Forward pass:
    token ids
      -> token embedding + position embedding
      -> N pre-LayerNorm transformer blocks
      -> final LayerNorm
      -> logits = x @ token_embedding.T        (tied embeddings)
      -> next-token cross-entropy loss

Sanity: a random model's loss should sit near ln(256) ~= 5.54.

Spec:  configs/model.byte-tinygpt-v0.json
Guide: docs/model_guide.md  ("Architecture details", "Output head", "Loss function")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Mirrors configs/model.byte-tinygpt-v0.json — the spec is the source of truth."""

    model_name: str = "byte-tinygpt-v0"
    vocab_size: int = 256
    context_length: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_mlp: int = 512
    dropout: float = 0.0
    tie_embeddings: bool = True
    dtype: str = "float32"

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        raw = json.loads(Path(path).read_text())
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in fields})

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must divide evenly into n_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

        q,k,v = q_proj(x), k_proj(x), v_proj(x)   -> split into n_heads
        scores = q @ k.T / sqrt(head_dim)
        scores = causal_mask(scores)        (future positions -> -inf)
        attn   = softmax(scores)
        out    = o_proj(attn @ v)

    The four projections are kept as separate named Linear modules (rather than a
    fused qkv) so LoRA can target q_proj / v_proj / o_proj by name (see lora.py).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular causal mask, registered so it moves with .to(device).
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.context_length, cfg.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # [B, T, C] -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        out = attn @ v  # [B, n_heads, T, head_dim]

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(out))


class MLP(nn.Module):
    """Position-wise feed-forward: Linear -> GELU -> Linear."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.fc_out = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block:  x = x + attn(ln1(x));  x = x + mlp(ln2(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Byte-level causal language model with tied input/output embeddings."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_final = nn.LayerNorm(cfg.d_model)
        # Tied output head: logits = x @ token_embedding.weight.T (no separate params).
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled init for the residual-path output projections.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.tie_embeddings:
            return F.linear(x, self.token_embedding.weight)
        return self.lm_head(x)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: [B, T] int64 token ids. Returns (logits, loss-or-None)."""
        B, T = idx.shape
        assert T <= self.cfg.context_length, f"sequence length {T} exceeds context {self.cfg.context_length}"

        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self._head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count. Tied head adds no params, so it is never double-counted."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.position_embedding.weight.numel()
        return n

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive decoding. idx: [B, T] prompt. See sample.py for the CLI."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # last position only

            if temperature <= 0.0:  # greedy / argmax decoding
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)

            idx = torch.cat([idx, next_id], dim=1)
        return idx


def build_model(config_path: str | Path) -> TinyGPT:
    """Convenience constructor used by train.py / sample.py / tests."""
    return TinyGPT(ModelConfig.from_json(config_path))


if __name__ == "__main__":
    # Smoke check: param count and a random-model forward pass.
    here = Path(__file__).resolve().parent.parent
    cfg = ModelConfig.from_json(here / "configs" / "model.byte-tinygpt-v0.json")
    model = TinyGPT(cfg)
    print(f"{cfg.model_name}: {model.num_params():,} parameters")

    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    logits, loss = model(x, x)
    print(f"logits {tuple(logits.shape)}  loss {loss.item():.4f}  (expect ~{math.log(cfg.vocab_size):.2f})")
