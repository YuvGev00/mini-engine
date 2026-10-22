"""CUDA graphs for decode and speculative-verify steps: capture the forward pass once per
batch-size bucket, then replay it with one launch instead of ~1,600 small kernel launches.

With the Triton kernels the GPU finished its work in about half of each step; the rest
was the CPU launching kernels one by one. A graph records the whole launch sequence
against fixed input buffers, so a step becomes: copy this step's token ids, positions,
slots, block tables and lengths into those buffers, then replay.

One `StepGraphs` serves steps with up to `query_len` new tokens per sequence, laid out
padded: sequence b's tokens sit at rows [b * query_len, b * query_len + n_b). The engine
keeps one with query_len 1 (plain decode) and, with speculation on, one with query_len
k + 1 (each sequence: its last real token plus up to k drafts). Unused rows, and whole
padding sequences up to the bucket size, write their K/V to a spare cache block that no
sequence owns, attend only to that slot, and are never read back.
"""

from __future__ import annotations

from collections.abc import Collection

import torch

from mini_engine.attention import AttentionMetadata
from mini_engine.model import Qwen3ForCausalLM
from mini_engine.scheduler import ScheduledSeq


def batch_buckets(max_batch: int) -> list[int]:
    small = [b for b in (1, 2, 4, 8) if b < max_batch]
    return small + list(range(16, max_batch, 16)) + [max_batch]


class StepGraphs:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        max_batch: int,
        query_len: int,
        max_blocks_per_seq: int,
        block_size: int,
        spare_block: int,
    ):
        self.query_len = query_len
        self.block_size = block_size
        self.spare_block = spare_block
        dev = kv_caches[0][0].device
        n, rows = max_batch, max_batch * query_len
        # Graphs record raw addresses: every tensor a graph reads must stay alive on self,
        # or the caching allocator hands its memory to someone else after capture.
        self.input_ids = torch.zeros(rows, dtype=torch.int64, device=dev)
        self.positions = torch.zeros(rows, dtype=torch.int64, device=dev)
        self.slot_mapping = torch.full((rows,), spare_block * block_size, dtype=torch.int64, device=dev)
        self.block_tables = torch.full((n, max_blocks_per_seq), spare_block, dtype=torch.int64, device=dev)
        self.context_lens = torch.ones(n, dtype=torch.int64, device=dev)
        self.query_lens = torch.ones(n, dtype=torch.int64, device=dev)
        self.row_index = torch.arange(rows, device=dev)  # identity: tokens are already padded
        self.empty = torch.empty(0, device=dev)

        self.buckets = batch_buckets(max_batch)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.logits: dict[int, torch.Tensor] = {}
        pool = None
        with torch.inference_mode():
            for b in reversed(self.buckets):  # largest first, so smaller graphs fit in its memory pool
                r = b * query_len
                meta = AttentionMetadata(
                    slot_mapping=self.slot_mapping[:r],
                    key_slots=self.empty,
                    block_tables=self.block_tables[:b],
                    context_lens=self.context_lens[:b],
                    query_lens=self.query_lens[:b],
                    query_index=self.row_index[:r],
                    mask=self.empty,
                    batch_size=b,
                    max_q=query_len,
                    backend="triton",
                )
                args = (self.input_ids[:r], self.positions[:r], kv_caches, meta, self.row_index[:r])
                model(*args)  # warm-up outside capture: compiles the Triton kernels, sets up cuBLAS
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self.logits[b] = model(*args)
                pool = graph.pool()
                self.graphs[b] = graph
        torch.cuda.synchronize()

    @property
    def max_batch(self) -> int:
        return self.buckets[-1]

    def run(self, batch: list[ScheduledSeq], verify: Collection[int] = ()) -> torch.Tensor:
        """Logits in the order `LLMEngine.step` reads them: every computed position of a
        sequence in `verify` (seq ids with drafts), the last position of any other."""
        n, q = len(batch), self.query_len
        b = next(size for size in self.buckets if size >= n)
        bs, spare_slot = self.block_size, self.spare_block * self.block_size
        ids: list[int] = []
        pos: list[int] = []
        slots: list[int] = []
        ctx: list[int] = []
        qlens: list[int] = []
        rows: list[int] = []
        width = max(len(item.seq.block_table) for item in batch)
        tables = []
        for i, item in enumerate(batch):
            start, count, table = item.seq.num_computed_tokens, item.num_tokens, item.seq.block_table
            ids += item.seq.token_ids[start : start + count] + [0] * (q - count)
            pos += list(range(start, start + count)) + [0] * (q - count)
            slots += [table[p // bs] * bs + p % bs for p in range(start, start + count)]
            slots += [spare_slot] * (q - count)
            ctx.append(start + count)
            qlens.append(count)
            if item.seq.seq_id in verify:
                rows += range(i * q, i * q + count)
            else:
                rows.append(i * q + count - 1)
            tables.append(table + [self.spare_block] * (width - len(table)))
        pad = b - n
        ids += [0] * (pad * q)
        pos += [0] * (pad * q)
        slots += [spare_slot] * (pad * q)
        ctx += [1] * pad
        qlens += [1] * pad
        tables += [[self.spare_block] * width] * pad

        r = b * q
        tokens = torch.tensor([ids, pos, slots], dtype=torch.int64)
        self.input_ids[:r].copy_(tokens[0])
        self.positions[:r].copy_(tokens[1])
        self.slot_mapping[:r].copy_(tokens[2])
        seqs = torch.tensor([ctx, qlens], dtype=torch.int64)
        self.context_lens[:b].copy_(seqs[0])
        self.query_lens[:b].copy_(seqs[1])
        self.block_tables[:b, :width].copy_(torch.tensor(tables, dtype=torch.int64))
        self.graphs[b].replay()
        return self.logits[b][rows]
