"""Paged attention in plain PyTorch: scatter new K/V into the block cache, gather each
sequence's context through its block table, then one masked attention call for the batch.

Layouts
  q:              [T, num_heads, head_dim]       T = total new tokens in the step
  k, v:           [T, num_kv_heads, head_dim]
  k_cache/v_cache [num_kv_heads, num_blocks, block_size, head_dim]   (one per layer)

The cache is head-major so that a single index op over the flattened slot axis returns
the batch's context as [num_kv_heads, B, K, head_dim], already contiguous in the order the
batched matmuls want. With a slot-major cache the gather needs a second full copy to
transpose heads in front, and on CUDA that copy cost more than the gather itself.

The batch is padded to [B, max_q] queries and [B, max_blocks * block_size] keys. That
wastes compute on padding (a fused kernel would not), but every step is a single SDPA
call, and the same code serves prefill, chunked prefill and decode.

Two backends compute the same attention:
  "sdpa"     one `F.scaled_dot_product_attention` call with `enable_gqa`. With a boolean
             mask plus GQA, PyTorch has no fused kernel and falls back to its math path,
             which expands K/V to every query head (repeat_interleave) and is slow on CUDA.
  "grouped"  folds each KV head's query group into the query-length axis, so one batched
             matmul per side serves all heads of a group with no K/V expansion. Softmax runs
             in fp32. This is the default on CUDA; "sdpa" stays the reference elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor  # [T] int64, cache slot each new token's K/V is written to
    key_slots: torch.Tensor  # [B, max_blocks * block_size] int64, cache slot of every key (padding masked)
    query_index: torch.Tensor  # [T] int64, position of each new token in the [B * max_q] padded layout
    mask: torch.Tensor  # [B, 1, max_q, max_k] bool, True = may attend
    batch_size: int
    max_q: int
    backend: str = "sdpa"

    @classmethod
    def build(
        cls,
        query_lens: list[int],
        context_lens: list[int],
        block_tables: list[list[int]],
        block_size: int,
        device: torch.device,
        backend: str = "sdpa",
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

        tables = torch.tensor(padded_tables, dtype=torch.int64, device=device)
        key_slots = (tables[:, :, None] * block_size + torch.arange(block_size, device=device)).flatten(1)
        return cls(
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
            key_slots=key_slots,
            query_index=torch.tensor(query_index, dtype=torch.int64, device=device),
            mask=mask[:, None, :, :],
            batch_size=batch_size,
            max_q=max_q,
            backend=backend,
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
    num_kv_heads, _, _, head_dim = k_cache.shape
    num_heads = q.shape[1]
    b, max_q = meta.batch_size, meta.max_q

    # 1. Write the new tokens' K/V into their slots.
    k_slots = k_cache.view(num_kv_heads, -1, head_dim)
    v_slots = v_cache.view(num_kv_heads, -1, head_dim)
    k_slots[:, meta.slot_mapping] = k.transpose(0, 1)
    v_slots[:, meta.slot_mapping] = v.transpose(0, 1)

    # 2. Gather each sequence's full context in one op: [kvh, B, K, d].
    keys = k_slots[:, meta.key_slots]
    values = v_slots[:, meta.key_slots]

    # 3. Scatter queries into the padded layout: [T, h, d] -> [B, max_q, h, d].
    q_pad = q.new_zeros(b * max_q, num_heads, head_dim)
    q_pad[meta.query_index] = q
    q_pad = q_pad.view(b, max_q, num_heads, head_dim)

    if meta.backend == "grouped":
        out = grouped_attention(q_pad, keys, values, meta.mask)  # [B, max_q, h, d]
    else:
        out = F.scaled_dot_product_attention(
            q_pad.transpose(1, 2),
            keys.transpose(0, 1),
            values.transpose(0, 1),
            attn_mask=meta.mask,
            enable_gqa=True,
        ).transpose(1, 2)

    # 4. Back to the packed token layout.
    return out.reshape(b * max_q, num_heads, head_dim)[meta.query_index]


def grouped_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """GQA attention without expanding K/V: q [B, Q, h, d], k/v [kvh, B, K, d], mask
    [B, 1, Q, K] (True = may attend). Returns [B, Q, h, d]. Query head i uses KV head
    i // (h // kvh), as in HF, so the heads of one group are contiguous in h."""
    b, q_len, h, d = q.shape
    kvh, _, k_len, _ = k.shape
    g = h // kvh
    # Fold each group's heads into the query axis: [kvh, B, g*Q, d]. Only q is reshuffled;
    # K/V stay as gathered, and (kvh, B) is a plain batch dimension for the matmuls.
    qg = q.view(b, q_len, kvh, g, d).permute(2, 0, 3, 1, 4).reshape(kvh, b, g * q_len, d)
    scores = torch.matmul(qg, k.transpose(-1, -2)).float() * (d**-0.5)  # [kvh, B, g*Q, K]
    group_mask = mask.expand(b, g, q_len, k_len).reshape(b, g * q_len, k_len)
    scores = scores.masked_fill(~group_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.matmul(probs, v)  # [kvh, B, g*Q, d]
    return out.view(kvh, b, g, q_len, d).permute(1, 3, 0, 2, 4).reshape(b, q_len, h, d)
