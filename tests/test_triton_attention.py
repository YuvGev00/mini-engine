"""Triton paged decode kernel vs. the PyTorch grouped path (CUDA + Triton only)."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from mini_engine.attention import AttentionMetadata, paged_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="needs CUDA and Triton",
)

H, KVH, D, BS, NUM_BLOCKS = 16, 8, 128, 16, 256


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_kernel_matches_grouped(dtype: torch.dtype) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    dev = torch.device("cuda")
    k_cache = torch.randn(KVH, NUM_BLOCKS, BS, D, device=dev, dtype=dtype, generator=g)
    v_cache = torch.randn(KVH, NUM_BLOCKS, BS, D, device=dev, dtype=dtype, generator=g)
    # Context lengths: a single token, block-aligned, unaligned, and longer than one tile.
    ctx_lens = [1, 16, 17, 33, 100, 255]
    perm = torch.randperm(NUM_BLOCKS).tolist()
    tables = [[perm.pop() for _ in range(-(-n // BS))] for n in ctx_lens]
    q = torch.randn(len(ctx_lens), H, D, device=dev, dtype=dtype, generator=g)
    k = torch.randn(len(ctx_lens), KVH, D, device=dev, dtype=dtype, generator=g)
    v = torch.randn(len(ctx_lens), KVH, D, device=dev, dtype=dtype, generator=g)

    outs = {}
    for backend in ("grouped", "triton"):
        meta = AttentionMetadata.build([1] * len(ctx_lens), ctx_lens, tables, BS, dev, backend=backend)
        outs[backend] = paged_attention(q, k, v, k_cache.clone(), v_cache.clone(), meta)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(outs["triton"], outs["grouped"], rtol=tol, atol=tol)
