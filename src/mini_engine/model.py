"""Qwen3 decoder in plain PyTorch, operating on a packed batch of new tokens.

Parameter names match Hugging Face's `Qwen3ForCausalLM`, so an HF state dict (from
safetensors or from an in-memory HF model) loads with `load_state_dict` unchanged.
Numerics mirror HF's implementation op for op (RMSNorm in fp32, RoPE tables in fp32),
which is what makes token-exact greedy parity achievable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mini_engine.attention import AttentionMetadata, paged_attention


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, hf: Any) -> ModelConfig:
        if getattr(hf, "model_type", "qwen3") != "qwen3":
            raise ValueError(f"only qwen3 is supported, got {hf.model_type}")
        rope = getattr(hf, "rope_parameters", None) or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("only default RoPE is supported")
        if getattr(hf, "use_sliding_window", False):
            raise ValueError("sliding-window attention is not supported")
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_layers=hf.num_hidden_layers,
            num_heads=hf.num_attention_heads,
            num_kv_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=float(rope.get("rope_theta", getattr(hf, "rope_theta", 10000.0))),
            max_position_embeddings=hf.max_position_embeddings,
            tie_word_embeddings=bool(hf.tie_word_embeddings),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin of shape [T, head_dim] for the given absolute positions."""
        freqs = positions[:, None].float() * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, heads, head_dim]. Rotate-half convention, as in HF Llama/Qwen."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        # Qwen3-specific: RMSNorm over each head's q and k before RoPE.
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        t = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(t, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(t, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        out = paged_attention(q, k, v, kv_cache[0], kv_cache[1], meta)
        y: torch.Tensor = self.o_proj(out.reshape(t, -1))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return y


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), rope, kv_cache, meta)
        y: torch.Tensor = x + self.mlp(self.post_attention_layernorm(x))
        return y


class Qwen3Body(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.model = Qwen3Body(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        meta: AttentionMetadata,
        logits_index: torch.Tensor,
    ) -> torch.Tensor:
        """input_ids/positions: [T]. Returns logits [len(logits_index), vocab] for the
        selected token rows only, so prefill never materialises [T, vocab]."""
        x = self.model.embed_tokens(input_ids)
        rope = self.model.rotary_emb(positions, x.dtype)
        for layer, kv in zip(self.model.layers, kv_caches, strict=True):
            x = layer(x, rope, kv, meta)
        x = self.model.norm(x[logits_index])
        logits: torch.Tensor = self.lm_head(x)
        return logits

    def load_hf_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        state_dict = dict(state_dict)
        if self.config.tie_word_embeddings:
            state_dict.pop("lm_head.weight", None)
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        self.load_state_dict(state_dict, strict=True)
