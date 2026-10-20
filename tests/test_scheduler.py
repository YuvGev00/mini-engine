"""Scheduler tests with a fake model: `step()` samples token 0 for every finished row."""

from __future__ import annotations

import pytest

from mini_engine.block_manager import BlockManager
from mini_engine.scheduler import ScheduledSeq, Scheduler, SchedulerConfig
from mini_engine.sequence import SamplingParams, Sequence, SequenceStatus

BS = 4
EOS = 2


def make(num_blocks: int = 64, **cfg: int) -> Scheduler:
    return Scheduler(SchedulerConfig(**cfg), BlockManager(num_blocks, BS), {EOS})


def step(s: Scheduler, token: int = 0) -> tuple[list[ScheduledSeq], list[Sequence]]:
    batch = s.schedule()
    tokens: list[int | None] = [token if item.samples else None for item in batch]
    return batch, s.postprocess(batch, tokens)


def add(s: Scheduler, n_prompt: int, max_tokens: int = 4, first: int = 10) -> Sequence:
    seq = Sequence(list(range(first, first + n_prompt)), SamplingParams(max_tokens=max_tokens))
    s.add(seq)
    return seq


def test_prefill_then_decode_in_one_mixed_batch() -> None:
    s = make()
    a = add(s, 6)
    batch, _ = step(s)
    assert [(i.seq, i.num_tokens) for i in batch] == [(a, 6)]
    b = add(s, 5, first=100)
    batch, _ = step(s)
    assert [(i.seq, i.num_tokens) for i in batch] == [(a, 1), (b, 5)]  # decode + prefill


def test_runs_to_completion_and_frees_blocks() -> None:
    s = make()
    seqs = [add(s, 5, max_tokens=3, first=10 * k) for k in range(1, 4)]
    finished: list[Sequence] = []
    while s.has_unfinished():
        finished += step(s)[1]
    assert set(finished) == set(seqs)
    assert all(q.is_finished and q.num_output_tokens == 3 for q in seqs)
    assert all(rc == 0 for rc in s.bm.ref_counts)


def test_eos_stops_unless_ignored() -> None:
    s = make()
    a = add(s, 3, max_tokens=10)
    b = Sequence([50, 51], SamplingParams(max_tokens=3, ignore_eos=True))
    s.add(b)
    _, done = step(s, token=EOS)
    assert done == [a] and a.output_token_ids == [EOS]
    while s.has_unfinished():
        step(s, token=EOS)
    assert b.output_token_ids == [EOS] * 3


def test_max_num_seqs_caps_admission() -> None:
    s = make(max_num_seqs=2)
    for k in range(3):
        add(s, 3, first=10 * k + 10)
    batch, _ = step(s)
    assert len(batch) == 2 and len(s.waiting) == 1


def test_token_budget_chunks_long_prefill() -> None:
    s = make(max_num_batched_tokens=8)
    a = add(s, 20)
    b = add(s, 3, first=100)
    sizes = []
    while a.num_output_tokens == 0:
        batch, _ = step(s)
        sizes.append([(i.seq is a, i.num_tokens) for i in batch])
    assert sizes[0] == [(True, 8)]  # b waits behind a: FCFS, no queue jumping
    assert sizes[1] == [(True, 8)]
    assert sizes[2] == [(True, 4), (False, 3)]  # a's last chunk samples; b gets the leftover budget
    assert b.status is SequenceStatus.RUNNING


def test_preempts_newest_and_readmits_first() -> None:
    # 4 blocks of 4 = 16 slots. Two 7-token prompts fit in 2 blocks each through the
    # first decode (8 tokens); the next decode needs a 3rd block each -> one must yield.
    s = make(num_blocks=4)
    a = add(s, 7, max_tokens=6)
    b = add(s, 7, max_tokens=6, first=100)
    step(s)  # prefill both
    step(s)  # decode both: 8 tokens each, 2 full blocks each
    batch, _ = step(s)  # a needs block 3 -> b (newest) preempted
    assert [i.seq for i in batch] == [a]
    assert b.status is SequenceStatus.WAITING and b.num_computed_tokens == 0
    assert s.waiting[0] is b and s.num_preemptions == 1
    while s.has_unfinished():
        step(s)
    assert a.num_output_tokens == b.num_output_tokens == 6
    assert all(rc == 0 for rc in s.bm.ref_counts)


def test_preemption_keeps_generated_tokens_and_hits_prefix_cache() -> None:
    s = make(num_blocks=4)
    add(s, 7, max_tokens=6)
    b = add(s, 7, max_tokens=6, first=100)
    for _ in range(3):
        step(s)  # b preempted after generating 2 tokens; its 2 full blocks were hashed
    assert b.output_token_ids == [0, 0]
    while b.status is SequenceStatus.WAITING:
        step(s)
    # On re-admission b's 9 tokens need recomputing, but a only needed one more block
    # and (blocks being freed tail-first) evicted b's 2nd block, so b's 1st still hits.
    assert b.num_cached_tokens == BS
    while s.has_unfinished():
        step(s)
    assert b.num_output_tokens == 6


def test_prefix_hit_skips_cached_tokens() -> None:
    s = make()
    a = add(s, 9)
    step(s)  # a's two full blocks are now hashed
    b = add(s, 11)  # same first 9 tokens, longer
    batch, _ = step(s)
    entry = next(i for i in batch if i.seq is b)
    assert b.num_cached_tokens == 8 and entry.num_tokens == 3
    assert b.block_table[:2] == a.block_table[:2]


def test_rejects_requests_that_can_never_fit() -> None:
    s = make(num_blocks=2, max_model_len=64)
    with pytest.raises(ValueError, match="KV blocks"):
        add(s, 8, max_tokens=4)
    with pytest.raises(ValueError, match="max_model_len"):
        make(max_model_len=10).add(Sequence([1] * 8, SamplingParams(max_tokens=4)))
