"""Paged decode attention as a Triton kernel: reads K/V straight from the block cache.

The plain-PyTorch path gathers every sequence's context into a padded [kvh, B, K, d]
tensor, then reads it again for the two matmuls. On a decode step that gather is pure
memory traffic (one read and one write of the whole context, per layer, per step) and
it dominated the step time. This kernel walks each sequence's block table instead, so
the context is read once and nothing is materialized.

One program handles one (sequence, KV head) pair and all query heads of that group. It
loops over the context in tiles of TILE tokens (whole blocks), with an online softmax
in fp32, so memory use does not grow with context length. Decode only: one query token
per sequence. Prefill and speculative-verify steps keep the PyTorch path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(  # type: ignore[no-untyped-def]  # Triton pointers/constexprs
    q_ptr,  # [B, H, D]
    k_cache_ptr,  # [KVH, num_blocks, BLOCK_SIZE, D]
    v_cache_ptr,
    block_tables_ptr,  # [B, max_blocks]
    context_lens_ptr,  # [B]
    out_ptr,  # [B, H, D]
    scale,
    stride_qb,
    stride_qh,
    stride_ch,
    stride_cb,
    stride_cs,
    stride_tb,
    stride_ob,
    stride_oh,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    ctx_len = tl.load(context_lens_ptr + seq)

    offs_g = tl.arange(0, GROUP_PAD)
    offs_t = tl.arange(0, TILE)
    offs_d = tl.arange(0, HEAD_DIM)
    heads = kv_head * GROUP + offs_g  # query head i uses KV head i // GROUP
    g_valid = offs_g < GROUP

    q_ptrs = q_ptr + seq * stride_qb + heads[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=g_valid[:, None], other=0.0).to(tl.float32) * scale  # [G, D]

    m = tl.full([GROUP_PAD], float("-inf"), tl.float32)
    denom = tl.zeros([GROUP_PAD], tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], tl.float32)
    head_base = kv_head.to(tl.int64) * stride_ch

    for start in range(0, ctx_len, TILE):
        pos = start + offs_t
        valid = pos < ctx_len
        block = tl.load(block_tables_ptr + seq * stride_tb + pos // BLOCK_SIZE, mask=valid, other=0)
        rows = head_base + block.to(tl.int64) * stride_cb + (pos % BLOCK_SIZE) * stride_cs  # [TILE]
        kv_offs = rows[:, None] + offs_d[None, :]
        k = tl.load(k_cache_ptr + kv_offs, mask=valid[:, None], other=0.0).to(tl.float32)  # [TILE, D]
        v = tl.load(v_cache_ptr + kv_offs, mask=valid[:, None], other=0.0).to(tl.float32)

        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2)  # [G, TILE]
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        denom = denom * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        m = m_new

    out = acc / denom[:, None]
    out_ptrs = out_ptr + seq * stride_ob + heads[:, None] * stride_oh + offs_d[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=g_valid[:, None])


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """q [B, H, D] (one token per sequence), caches [KVH, num_blocks, block_size, D],
    block_tables [B, max_blocks], context_lens [B] (including the new token, whose K/V
    must already be in the cache). Returns [B, H, D]."""
    b, h, d = q.shape
    kvh, _, block_size, _ = k_cache.shape
    group = h // kvh
    q = q.contiguous()
    out = torch.empty_like(q)
    tile = max(block_size, 32)
    _paged_decode_kernel[(b, kvh)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out,
        d**-0.5,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        block_tables.stride(0),
        out.stride(0),
        out.stride(1),
        GROUP=group,
        GROUP_PAD=triton.next_power_of_2(group),
        BLOCK_SIZE=block_size,
        TILE=tile,
        HEAD_DIM=d,
    )
    return out
