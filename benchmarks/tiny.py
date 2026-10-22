"""A tiny random Qwen3 for the CPU smoke test of bench.py (same shape as the test model)."""

from __future__ import annotations

from typing import Any

import torch


def tiny_hf_model() -> Any:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        initializer_range=0.2,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=0,
        attn_implementation="sdpa",
    )
    model = Qwen3ForCausalLM(config).eval()
    model.generation_config.eos_token_id = None
    return model
