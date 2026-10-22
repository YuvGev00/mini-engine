"""Paged decode attention as a Triton kernel: reads K/V straight from the block cache.

The plain-PyTorch path gathers every sequence's context into a padded [kvh, B, K, d]
tensor, then reads it again for the two matmuls. On a decode step that gather is pure
memory traffic (one read and one write of the whole context, per layer, per step) and
it dominated the step time. This kernel walks each sequence's block table instead, so
the context is read once and nothing is materialized.

One program handles one (sequence, KV head) pair and all query heads of that group. It
loops over the context in tiles of TILE tokens (whole blocks), with an online softmax
in fp32, so memory use does not grow with context length. Decode only: one query token
per sequence. `paged_multi_query_attention` is the same idea for a few query tokens per
sequence (speculative verify), using tl.dot on [query tokens x group heads] rows.
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


@triton.jit
def _paged_multi_query_kernel(  # type: ignore[no-untyped-def]  # Triton pointers/constexprs
    q_ptr,  # [B, Q, H, D]
    k_cache_ptr,  # [KVH, num_blocks, BLOCK_SIZE, D]
    v_cache_ptr,
    block_tables_ptr,  # [B, max_blocks]
    context_lens_ptr,  # [B]
    query_lens_ptr,  # [B]
    out_ptr,  # [B, Q, H, D]
    scale,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_ch,
    stride_cb,
    stride_cs,
    stride_tb,
    stride_ob,
    stride_oq,
    stride_oh,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DOT_PRECISION: tl.constexpr,  # "ieee" keeps fp32 exact; bf16/fp16 ignore it
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    ctx_len = tl.load(context_lens_ptr + seq)
    q_len = tl.load(query_lens_ptr + seq)

    # Row r holds query token r // GROUP_PAD for query head kv_head * GROUP + r % GROUP_PAD.
    offs_r = tl.arange(0, ROWS)
    offs_t = tl.arange(0, TILE)
    offs_d = tl.arange(0, HEAD_DIM)
    qi = offs_r // GROUP_PAD
    heads = kv_head * GROUP + offs_r % GROUP_PAD
    row_valid = (qi < q_len) & (offs_r % GROUP_PAD < GROUP)
    q_pos = ctx_len - q_len + qi  # absolute position of each row's query token

    q_ptrs = q_ptr + seq * stride_qb + qi[:, None] * stride_qq + heads[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0)  # [ROWS, D], input dtype

    m = tl.full([ROWS], float("-inf"), tl.float32)
    denom = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, HEAD_DIM], tl.float32)
    head_base = kv_head.to(tl.int64) * stride_ch

    for start in range(0, ctx_len, TILE):
        pos = start + offs_t
        valid = pos < ctx_len
        block = tl.load(block_tables_ptr + seq * stride_tb + pos // BLOCK_SIZE, mask=valid, other=0)
        rows = head_base + block.to(tl.int64) * stride_cb + (pos % BLOCK_SIZE) * stride_cs
        kv_offs = rows[:, None] + offs_d[None, :]
        k = tl.load(k_cache_ptr + kv_offs, mask=valid[:, None], other=0.0)  # [TILE, D]
        v = tl.load(v_cache_ptr + kv_offs, mask=valid[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION) * scale  # [ROWS, TILE], fp32
        # Causal: a query sees keys up to its own position. Padding rows see the whole
        # context so their softmax stays finite; they are never stored.
        see = valid[None, :] & ((pos[None, :] <= q_pos[:, None]) | ~row_valid[:, None])
        s = tl.where(see, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        denom = denom * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=DOT_PRECISION)
        m = m_new

    out = acc / denom[:, None]
    out_ptrs = (
        out_ptr + seq * stride_ob + qi[:, None] * stride_oq + heads[:, None] * stride_oh + offs_d[None, :]
    )
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=row_valid[:, None])


def paged_multi_query_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    query_lens: torch.Tensor,
) -> torch.Tensor:
    """A few query tokens per sequence (speculative verify, short chunks): q [B, Q, H, D],
    padded; sequence b's real queries are the first query_lens[b], at the last positions
    of its context_lens[b] tokens (whose K/V must already be in the cache). Causal within
    the new tokens. Returns [B, Q, H, D]; padded query rows are left unwritten."""
    b, q_max, h, d = q.shape
    kvh, _, block_size, _ = k_cache.shape
    group = h // kvh
    group_pad = triton.next_power_of_2(group)
    q = q.contiguous()
    out = torch.empty_like(q)
    _paged_multi_query_kernel[(b, kvh)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        query_lens,
        out,
        d**-0.5,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        block_tables.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        GROUP=group,
        GROUP_PAD=group_pad,
        ROWS=max(16, triton.next_power_of_2(q_max) * group_pad),  # tl.dot needs >= 16 rows
        BLOCK_SIZE=block_size,
        TILE=max(block_size, 32),
        HEAD_DIM=d,
        DOT_PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
    )
    return out
