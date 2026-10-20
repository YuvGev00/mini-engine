"""Continuous-batching scheduler with chunked prefill and recompute preemption.

Every step builds one batch under a token budget. Each scheduled sequence contributes
`n` tokens starting at its `num_computed_tokens`: n == 1 for a decode, n > 1 for a
(possibly partial) prefill. Prefills and decodes share a batch, because the attention
path handles any mix of query lengths.

Policy (FCFS):
  1. Running sequences first, in admission order. If one cannot get the block it
     needs, the most recently admitted running sequence is preempted (its blocks are
     freed and it is re-queued at the front of `waiting`) until space appears.
  2. Then waiting sequences are admitted in order, reusing prefix-cache hits, until
     the token budget, the sequence cap or the KV cache runs out. Admission stops at
     the first sequence that does not fit, so no later request jumps the queue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from mini_engine.block_manager import BlockManager
from mini_engine.sequence import Sequence, SequenceStatus


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_tokens: int

    @property
    def samples(self) -> bool:
        """Whether this step reaches the end of the sequence, so its logits are used."""
        return self.seq.num_computed_tokens + self.num_tokens == len(self.seq)


class Scheduler:
    def __init__(self, config: SchedulerConfig, block_manager: BlockManager, eos_token_ids: set[int]):
        self.config = config
        self.bm = block_manager
        self.eos_token_ids = eos_token_ids
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0
        self.num_cached_tokens = 0  # tokens served by the prefix cache, summed over admissions

    def add(self, seq: Sequence) -> None:
        max_len = len(seq) + seq.params.max_tokens
        if max_len > self.config.max_model_len:
            raise ValueError(f"prompt + max_tokens = {max_len} exceeds max_model_len")
        if self.bm.blocks_needed(max_len) > self.bm.num_blocks:
            raise ValueError(f"request needs more than the {self.bm.num_blocks} KV blocks available")
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> list[ScheduledSeq]:
        budget = self.config.max_num_batched_tokens
        batch: list[ScheduledSeq] = []
        preempted = False

        # 1. Running sequences. Victims are taken from the tail of `running`, which this
        #    loop has not reached yet, so a victim is never already in `batch`.
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_uncomputed_tokens, budget)
            while not self.bm.try_allocate(seq, seq.num_computed_tokens + n):
                victim = self.running[-1]
                self._preempt(victim)
                preempted = True
                if victim is seq:
                    break
            else:
                batch.append(ScheduledSeq(seq, n))
                budget -= n
                i += 1

        # 2. Admit waiting sequences. Skip this after a preemption: memory is tight, and
        #    admitting would only re-trigger it next step.
        while not preempted and self.waiting and budget > 0 and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            cached = self.bm.match_prefix(seq.token_ids)
            num_cached = len(cached) * self.bm.block_size
            n = min(len(seq) - num_cached, budget)
            if not self.bm.try_allocate(seq, num_cached + n, cached):
                break
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            seq.num_computed_tokens = num_cached
            seq.num_cached_tokens = num_cached
            self.num_cached_tokens += num_cached
            self.running.append(seq)
            batch.append(ScheduledSeq(seq, n))
            budget -= n

        return batch

    def postprocess(self, batch: list[ScheduledSeq], next_tokens: list[int | None]) -> list[Sequence]:
        """Advance sequences after a model step. `next_tokens[i]` is the sampled token
        for `batch[i]`, or None if that entry was a non-final prefill chunk."""
        finished: list[Sequence] = []
        for item, token in zip(batch, next_tokens, strict=True):
            seq = item.seq
            seq.num_computed_tokens += item.num_tokens
            self.bm.register_computed_blocks(seq)
            if token is None:
                continue
            seq.token_ids.append(token)
            if self._is_done(seq, token):
                seq.status = SequenceStatus.FINISHED
                self.bm.free(seq)
                self.running.remove(seq)
                finished.append(seq)
        return finished

    def _is_done(self, seq: Sequence, token: int) -> bool:
        if not seq.params.ignore_eos and token in self.eos_token_ids:
            return True
        return seq.num_output_tokens >= seq.params.max_tokens

    def _preempt(self, seq: Sequence) -> None:
        """Recompute-style preemption: drop the KV cache, keep the tokens. On
        re-admission the prefix cache usually still holds most of its blocks."""
        self.running.remove(seq)
        self.bm.free(seq)
        seq.status = SequenceStatus.WAITING
        seq.num_computed_tokens = 0
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)
