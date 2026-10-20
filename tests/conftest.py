"""Shared fixtures. The tiny model is a randomly initialised Qwen3 built from a config,
so CI needs no download; the real-checkpoint test is opt-in (see test_parity.py)."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

TINY_VOCAB = 512


def make_tiny_hf_model(seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=TINY_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # exercise GQA
        head_dim=16,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        # Precaution: a wider init than the default 0.02 makes logits peaked, so a
        # near-tie argmax cannot be flipped by fp32 rounding between implementations.
        initializer_range=0.2,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=0,
        attn_implementation="sdpa",
    )
    model = Qwen3ForCausalLM(config).eval()
    model.generation_config.eos_token_id = None
    return model


@pytest.fixture(scope="session")
def tiny_hf_model() -> Qwen3ForCausalLM:
    return make_tiny_hf_model()


def random_prompts(n: int, lo: int, hi: int, seed: int = 1) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(lo, hi + 1, (n,), generator=g).tolist()
    return [torch.randint(1, TINY_VOCAB, (k,), generator=g).tolist() for k in lens]
