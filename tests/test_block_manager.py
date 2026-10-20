from __future__ import annotations

import pytest

from mini_engine.block_manager import ROOT_HASH, BlockManager, hash_block
from mini_engine.sequence import Sequence

BS = 4


def computed(bm: BlockManager, tokens: list[int]) -> Sequence:
    """Admit a sequence through the prefix cache and mark its prompt computed."""
    seq = Sequence(tokens)
    cached = bm.match_prefix(seq.token_ids)
    assert bm.try_allocate(seq, len(seq), cached)
    seq.num_computed_tokens = len(seq)
    bm.register_computed_blocks(seq)
    return seq


def check_invariants(bm: BlockManager, live: list[Sequence]) -> None:
    refs = [0] * bm.num_blocks
    for s in live:
        for b in s.block_table:
            refs[b] += 1
    assert refs == bm.ref_counts
    assert set(bm.free_pool) == {b for b in range(bm.num_blocks) if refs[b] == 0}
    for h, b in bm.hash_to_block.items():
        assert bm.block_hash[b] == h


def test_hash_is_chained_and_deterministic() -> None:
    a = hash_block(ROOT_HASH, [1, 2, 3, 4])
    assert a == hash_block(ROOT_HASH, [1, 2, 3, 4])
    assert hash_block(a, [5, 6, 7, 8]) != hash_block(ROOT_HASH, [5, 6, 7, 8])


def test_allocate_and_free_roundtrip() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    seq = Sequence(list(range(10)))
    assert bm.try_allocate(seq, 10)
    assert len(seq.block_table) == 3 and bm.num_free_blocks == 5
    assert bm.try_allocate(seq, 12)  # still fits in block 3
    assert len(seq.block_table) == 3
    assert bm.try_allocate(seq, 13)  # needs a 4th
    assert len(seq.block_table) == 4
    check_invariants(bm, [seq])
    bm.free(seq)
    assert seq.block_table == [] and bm.num_free_blocks == 8
    check_invariants(bm, [])


def test_allocation_is_all_or_nothing() -> None:
    bm = BlockManager(num_blocks=2, block_size=BS)
    seq = Sequence([1])
    assert not bm.try_allocate(seq, 9)  # needs 3 blocks
    assert seq.block_table == [] and bm.num_free_blocks == 2


def test_prefix_hit_shares_blocks_and_counts_refs() -> None:
    bm = BlockManager(num_blocks=16, block_size=BS)
    a = computed(bm, list(range(1, 11)))  # 2 full blocks + partial
    b = computed(bm, [*list(range(1, 9)), 99, 98, 97])
    assert b.block_table[:2] == a.block_table[:2]
    assert b.block_table[2] != a.block_table[2]
    assert [bm.ref_counts[x] for x in a.block_table] == [2, 2, 1]
    check_invariants(bm, [a, b])

    bm.free(a)  # shared blocks survive with ref 1
    assert [bm.ref_counts[x] for x in b.block_table] == [1, 1, 1]
    check_invariants(bm, [b])


def test_full_prompt_hit_still_computes_last_token() -> None:
    bm = BlockManager(num_blocks=16, block_size=BS)
    computed(bm, list(range(8)))  # exactly 2 full blocks
    assert len(bm.match_prefix(list(range(8)))) == 1  # only tokens [0, 7) are eligible
    assert len(bm.match_prefix(list(range(9)))) == 2


def test_freed_blocks_stay_cached_until_evicted() -> None:
    bm = BlockManager(num_blocks=4, block_size=BS)
    a = computed(bm, list(range(9)))  # blocks: full, full, partial
    bm.free(a)
    assert len(bm.match_prefix(list(range(9)))) == 2  # still hit after free
    b = computed(bm, list(range(9)))  # revives the 2 cached blocks
    assert bm.ref_counts[b.block_table[0]] == 1
    assert bm.num_evictions == 0
    check_invariants(bm, [b])


def test_eviction_prefers_unhashed_then_lru() -> None:
    bm = BlockManager(num_blocks=6, block_size=BS)
    old = computed(bm, [1] * 5)  # 1 hashed full block + 1 partial
    new = computed(bm, [2] * 5)
    bm.free(old)
    bm.free(new)
    # Free pool order: unhashed partials first, then old's full block, then new's.
    hog = Sequence([3])
    assert bm.try_allocate(hog, 4 * BS)  # 4 fresh blocks: 2 never-used + 2 partials
    assert bm.num_evictions == 0
    assert len(bm.match_prefix([1] * 5)) == 1 and len(bm.match_prefix([2] * 5)) == 1
    assert bm.try_allocate(hog, 5 * BS)  # must evict: the LRU one is old's
    assert bm.num_evictions == 1
    assert bm.match_prefix([1] * 5) == [] and len(bm.match_prefix([2] * 5)) == 1
    check_invariants(bm, [hog])


def test_reviving_cached_block_counts_against_capacity() -> None:
    bm = BlockManager(num_blocks=3, block_size=BS)
    a = computed(bm, list(range(9)))  # 3 blocks, 2 hashed
    bm.free(a)
    seq = Sequence(list(range(9)) + [50] * 4)  # hits 2, needs 2 more -> 4 > 3
    cached = bm.match_prefix(seq.token_ids)
    assert len(cached) == 2
    assert not bm.try_allocate(seq, len(seq), cached)
    assert seq.block_table == [] and bm.num_free_blocks == 3
    check_invariants(bm, [])


def test_duplicate_content_is_not_double_registered() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    a, b = Sequence([7] * 5), Sequence([7] * 5)
    for s in (a, b):  # both admitted before either is computed: no sharing possible
        assert bm.try_allocate(s, len(s))
    for s in (a, b):
        s.num_computed_tokens = len(s)
        bm.register_computed_blocks(s)
    assert bm.hash_to_block[a.block_hashes[0]] == a.block_table[0]
    assert bm.block_hash[b.block_table[0]] is None
    bm.free(a)
    bm.free(b)
    check_invariants(bm, [])


def test_decode_tokens_get_hashed_as_blocks_fill() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    seq = computed(bm, [1, 2, 3])
    assert seq.block_hashes == []
    seq.token_ids.append(4)
    seq.num_computed_tokens = 4
    bm.register_computed_blocks(seq)
    assert len(seq.block_hashes) == 1
    assert bm.match_prefix([1, 2, 3, 4, 5]) == [seq.block_table[0]]


def test_prefix_caching_disabled() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS, enable_prefix_caching=False)
    computed(bm, list(range(9)))
    assert bm.match_prefix(list(range(9))) == [] and bm.hash_to_block == {}


def test_reset_prefix_cache() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    bm.free(computed(bm, list(range(9))))
    bm.reset_prefix_cache()
    assert bm.match_prefix(list(range(9))) == []
    assert all(h is None for h in bm.block_hash)


def test_double_free_is_detected() -> None:
    bm = BlockManager(num_blocks=4, block_size=BS)
    seq = Sequence([1])
    bm.try_allocate(seq, 1)
    table = list(seq.block_table)
    bm.free(seq)
    seq.block_table = table
    with pytest.raises(AssertionError):
        bm.free(seq)
