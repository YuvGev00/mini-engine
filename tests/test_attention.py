"""Paged attention vs. a dense per-sequence causal reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mini_engine.attention import AttentionMetadata, paged_attention

H, KVH, D, BS, NUM_BLOCKS = 4, 2, 8, 4, 32


@pytest.mark.parametrize("backend", ["sdpa", "grouped"])
def test_mixed_prefill_and_decode_matches_dense(backend: str) -> None:
    g = torch.Generator().manual_seed(0)
    k_cache = torch.zeros(NUM_BLOCKS, BS, KVH, D)
    v_cache = torch.zeros(NUM_BLOCKS, BS, KVH, D)
    # (cached tokens, new tokens): fresh prefill, decode, chunked prefill continuation.
    shapes = [(0, 7), (9, 1), (8, 5)]
    # Scattered, non-contiguous block tables on purpose.
    perm = torch.randperm(NUM_BLOCKS, generator=g).tolist()
    tables, full_k, full_v = [], [], []
    for cached, new in shapes:
        n_blocks = -(-(cached + new) // BS)
        tables.append([perm.pop() for _ in range(n_blocks)])
        full_k.append(torch.randn(cached + new, KVH, D, generator=g))
        full_v.append(torch.randn(cached + new, KVH, D, generator=g))
        for pos in range(cached):  # pretend earlier steps wrote the cached part
            b = tables[-1][pos // BS]
            k_cache[b, pos % BS] = full_k[-1][pos]
            v_cache[b, pos % BS] = full_v[-1][pos]

    qs = [torch.randn(new, H, D, generator=g) for _, new in shapes]
    meta = AttentionMetadata.build(
        query_lens=[n for _, n in shapes],
        context_lens=[c + n for c, n in shapes],
        block_tables=tables,
        block_size=BS,
        device=torch.device("cpu"),
        backend=backend,
    )
    out = paged_attention(
        torch.cat(qs),
        torch.cat([k[c:] for k, (c, _) in zip(full_k, shapes, strict=True)]),
        torch.cat([v[c:] for v, (c, _) in zip(full_v, shapes, strict=True)]),
        k_cache,
        v_cache,
        meta,
    )

    expected = []
    for q, k, v, (cached, new) in zip(qs, full_k, full_v, shapes, strict=True):
        k_rep = k.repeat_interleave(H // KVH, dim=1).transpose(0, 1)
        v_rep = v.repeat_interleave(H // KVH, dim=1).transpose(0, 1)
        mask = torch.ones(new, cached + new, dtype=torch.bool).tril(diagonal=cached)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k_rep, v_rep, attn_mask=mask)
        expected.append(o.transpose(0, 1))
    torch.testing.assert_close(out, torch.cat(expected), rtol=1e-5, atol=1e-5)

    # The new tokens' K/V were written to the right slots.
    c, n = shapes[2]
    for pos in range(c, c + n):
        torch.testing.assert_close(k_cache[tables[2][pos // BS], pos % BS], full_k[2][pos])
