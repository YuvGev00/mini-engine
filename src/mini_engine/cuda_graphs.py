"""CUDA graphs for decode steps: capture the forward pass once per batch-size bucket,
then replay it with one launch instead of ~1,600 small kernel launches.

With the Triton decode kernel the GPU finished its work in about half of each decode
step; the rest was the CPU launching kernels one by one. A graph records the whole
launch sequence against fixed input buffers, so a step becomes: copy this step's
token ids, positions, slots, block tables and context lengths into those buffers,
then replay.

Only pure decode batches use graphs (one new token per sequence, every sequence
sampling, no speculative drafts). The batch is padded up to the next bucket; padded
rows write their K/V to a spare cache block that no sequence owns and attend only to
that slot, and their logits are dropped.
"""

from __future__ import annotations

import torch

from mini_engine.attention import AttentionMetadata
from mini_engine.model import Qwen3ForCausalLM
from mini_engine.scheduler import ScheduledSeq


def batch_buckets(max_batch: int) -> list[int]:
    small = [b for b in (1, 2, 4, 8) if b < max_batch]
    return small + list(range(16, max_batch, 16)) + [max_batch]


class DecodeGraphs:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        max_batch: int,
        max_blocks_per_seq: int,
        block_size: int,
        spare_block: int,
    ):
        self.block_size = block_size
        self.spare_block = spare_block
        dev = kv_caches[0][0].device
        n = max_batch
        self.input_ids = torch.zeros(n, dtype=torch.int64, device=dev)
        self.positions = torch.zeros(n, dtype=torch.int64, device=dev)
        self.slot_mapping = torch.full((n,), spare_block * block_size, dtype=torch.int64, device=dev)
        self.block_tables = torch.full((n, max_blocks_per_seq), spare_block, dtype=torch.int64, device=dev)
        self.context_lens = torch.ones(n, dtype=torch.int64, device=dev)
        logits_index = torch.arange(n, device=dev)
        empty = torch.empty(0, device=dev)

        self.buckets = batch_buckets(max_batch)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.logits: dict[int, torch.Tensor] = {}
        pool = None
        with torch.inference_mode():
            for b in reversed(self.buckets):  # largest first, so smaller graphs fit in its memory pool
                meta = AttentionMetadata(
                    slot_mapping=self.slot_mapping[:b],
                    key_slots=empty,
                    block_tables=self.block_tables[:b],
                    context_lens=self.context_lens[:b],
                    query_index=empty,
                    mask=empty,
                    batch_size=b,
                    max_q=1,
                    backend="triton",
                )
                args = (self.input_ids[:b], self.positions[:b], kv_caches, meta, logits_index[:b])
                model(*args)  # warm-up outside capture: compiles the Triton kernel, sets up cuBLAS
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

    def run(self, batch: list[ScheduledSeq]) -> torch.Tensor:
        """Logits [len(batch), vocab] for a pure decode batch, in batch order."""
        n = len(batch)
        b = next(size for size in self.buckets if size >= n)
        bs = self.block_size
        ids, pos, slots, ctx = [], [], [], []
        width = max(len(item.seq.block_table) for item in batch)
        tables = []
        for item in batch:
            p = item.seq.num_computed_tokens
            table = item.seq.block_table
            ids.append(item.seq.token_ids[p])
            pos.append(p)
            slots.append(table[p // bs] * bs + p % bs)
            ctx.append(p + 1)
            tables.append(table + [self.spare_block] * (width - len(table)))
        pad = b - n
        slots += [self.spare_block * bs] * pad
        ctx += [1] * pad
        tables += [[self.spare_block] * width] * pad

        cpu = torch.tensor([ids + [0] * pad, pos + [0] * pad, slots, ctx], dtype=torch.int64)
        self.input_ids[:b].copy_(cpu[0])
        self.positions[:b].copy_(cpu[1])
        self.slot_mapping[:b].copy_(cpu[2])
        self.context_lens[:b].copy_(cpu[3])
        self.block_tables[:b, :width].copy_(torch.tensor(tables, dtype=torch.int64))
        self.graphs[b].replay()
        return self.logits[b][:n]
