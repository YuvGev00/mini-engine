"""Paged attention in plain PyTorch: scatter new K/V into the block cache, gather each
sequence's context through its block table, then one masked SDPA call for the batch.

Layouts
  q:              [T, num_heads, head_dim]       T = total new tokens in the step
  k, v:           [T, num_kv_heads, head_dim]
  k_cache/v_cache [num_blocks, block_size, num_kv_heads, head_dim]   (one per layer)

The batch is padded to [B, max_q] queries and [B, max_blocks * block_size] keys. That
wastes compute on padding (a fused kernel would not), but every step is a single SDPA
call, and the same code serves prefill, chunked prefill and decode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor  # [T] int64, cache slot each new token's K/V is written to
    block_tables: torch.Tensor  # [B, max_blocks] int64, padded with 0 (masked out)
    query_index: torch.Tensor  # [T] int64, position of each new token in the [B * max_q] padded layout
    mask: torch.Tensor  # [B, 1, max_q, max_k] bool, True = may attend
    batch_size: int
    max_q: int

    @classmethod
    def build(
        cls,
        query_lens: list[int],
        context_lens: list[int],
        block_tables: list[list[int]],
        block_size: int,
        device: torch.device,
    ) -> AttentionMetadata:
        """`context_lens[b]` counts cached plus new tokens of sequence b, so its new
        tokens sit at absolute positions [context_len - query_len, context_len)."""
        batch_size, max_q = len(query_lens), max(query_lens)
        max_blocks = max(len(t) for t in block_tables)

        slots: list[int] = []
        query_index: list[int] = []
        for b, (q_len, ctx_len, table) in enumerate(zip(query_lens, context_lens, block_tables, strict=True)):
            for pos in range(ctx_len - q_len, ctx_len):
                slots.append(table[pos // block_size] * block_size + pos % block_size)
            query_index.extend(range(b * max_q, b * max_q + q_len))
        padded_tables = [t + [0] * (max_blocks - len(t)) for t in block_tables]

        # Query i of sequence b is at absolute position ctx_len - q_len + i and may see
        # keys at positions <= that. Rows for padded queries (i >= q_len) still see at
        # least one key, so SDPA never produces NaN; their outputs are discarded.
        q_lens = torch.tensor(query_lens, device=device)
        ctx_lens = torch.tensor(context_lens, device=device)
        q_pos = (ctx_lens - q_lens)[:, None] + torch.arange(max_q, device=device)[None, :]
        k_pos = torch.arange(max_blocks * block_size, device=device)
        mask = k_pos[None, None, :] <= q_pos[:, :, None]

        return cls(
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
            block_tables=torch.tensor(padded_tables, dtype=torch.int64, device=device),
            query_index=torch.tensor(query_index, dtype=torch.int64, device=device),
            mask=mask[:, None, :, :],
            batch_size=batch_size,
            max_q=max_q,
        )


def paged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: AttentionMetadata,
) -> torch.Tensor:
    """Returns attention output of shape [T, num_heads, head_dim]."""
    _, _, num_kv_heads, head_dim = k_cache.shape
    num_heads = q.shape[1]
    b, max_q = meta.batch_size, meta.max_q

    # 1. Write the new tokens' K/V into their slots.
    k_cache.view(-1, num_kv_heads, head_dim)[meta.slot_mapping] = k
    v_cache.view(-1, num_kv_heads, head_dim)[meta.slot_mapping] = v

    # 2. Gather each sequence's full context: [B, max_blocks, bs, kvh, d] -> [B, kvh, K, d].
    keys = k_cache[meta.block_tables].flatten(1, 2).transpose(1, 2)
    values = v_cache[meta.block_tables].flatten(1, 2).transpose(1, 2)

    # 3. Scatter queries into the padded layout: [T, h, d] -> [B, h, max_q, d].
    q_pad = q.new_zeros(b * max_q, num_heads, head_dim)
    q_pad[meta.query_index] = q
    q_pad = q_pad.view(b, max_q, num_heads, head_dim).transpose(1, 2)

    out = F.scaled_dot_product_attention(q_pad, keys, values, attn_mask=meta.mask, enable_gqa=True)

    # 4. Back to the packed token layout.
    out = out.transpose(1, 2).reshape(b * max_q, num_heads, head_dim)
    return out[meta.query_index]
