"""Paged KV-cache bookkeeping: block allocator, block tables and prefix cache.

The KV cache is `num_blocks` fixed-size blocks of `block_size` token slots. A sequence
owns an ordered list of block ids (its block table); token `i` lives in slot
`block_table[i // block_size] * block_size + i % block_size`.

Prefix caching: once a block is *full and computed*, it gets a chained hash
`H(parent_hash, its token ids)`, so equal hashes imply equal whole prefixes. A new
sequence whose leading full blocks hash to known blocks reuses them (ref_count += 1)
instead of recomputing. Blocks whose ref_count drops to 0 are not wiped: they stay in
the free pool, still hashed, and are only evicted (in LRU order) when a fresh block is
needed and nothing unhashed is left to hand out.

Nothing here touches tensors; this module is pure Python and unit-testable alone.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence as Seq

from mini_engine.sequence import Sequence


def hash_block(parent: bytes, token_ids: Seq[int]) -> bytes:
    """Chained, process-independent block hash (Python's `hash()` is per-process salted
    for bytes/str, and we want hashes that can later be compared across traces)."""
    h = hashlib.blake2b(parent, digest_size=16)
    for t in token_ids:
        h.update(t.to_bytes(4, "little", signed=False))
    return h.digest()


ROOT_HASH = b""


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.ref_counts = [0] * num_blocks
        self.block_hash: list[bytes | None] = [None] * num_blocks
        self.hash_to_block: dict[bytes, int] = {}
        # Blocks with ref_count == 0, in eviction order (front = evicted first).
        # Unhashed blocks are kept at the front since they hold nothing reusable.
        self.free_pool: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.num_evictions = 0

    # ---------------------------------------------------------------- queries

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_pool)

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def match_prefix(self, token_ids: Seq[int]) -> list[int]:
        """Longest run of cached full blocks that prefix `token_ids`.

        Only the first `len - 1` tokens are eligible: the model must compute at least
        one token per sequence to produce logits for the next one.
        """
        if not self.enable_prefix_caching:
            return []
        hits: list[int] = []
        parent = ROOT_HASH
        bs = self.block_size
        for i in range((len(token_ids) - 1) // bs):
            parent = hash_block(parent, token_ids[i * bs : (i + 1) * bs])
            block = self.hash_to_block.get(parent)
            if block is None:
                break
            hits.append(block)
        return hits

    # -------------------------------------------------------------- mutations

    def try_allocate(self, seq: Sequence, num_tokens: int, cached_blocks: Seq[int] = ()) -> bool:
        """Grow `seq.block_table` to hold `num_tokens` tokens. All-or-nothing.

        `cached_blocks` (from `match_prefix`, only for a sequence with an empty block
        table) are shared rather than allocated. Returns False and changes nothing if
        there is not enough free space.
        """
        if cached_blocks and seq.block_table:
            raise ValueError("cached blocks can only seed an empty block table")
        num_new = self.blocks_needed(num_tokens) - len(seq.block_table) - len(cached_blocks)
        # Reviving a cached block that sits in the free pool consumes free capacity too.
        num_revived = sum(1 for b in cached_blocks if self.ref_counts[b] == 0)
        if max(num_new, 0) + num_revived > self.num_free_blocks:
            return False

        for b in cached_blocks:
            if self.ref_counts[b] == 0:
                del self.free_pool[b]
            self.ref_counts[b] += 1
            seq.block_table.append(b)
        for _ in range(num_new):
            seq.block_table.append(self._take_fresh_block())
        return True

    def free(self, seq: Sequence) -> None:
        """Drop the sequence's references. Freed in reverse so that, among blocks freed
        together, the tail of a sequence is evicted before its (more shareable) head."""
        for b in reversed(seq.block_table):
            self.ref_counts[b] -= 1
            assert self.ref_counts[b] >= 0, "double free"
            if self.ref_counts[b] == 0:
                self.free_pool[b] = None
                if self.block_hash[b] is None:
                    self.free_pool.move_to_end(b, last=False)
        seq.block_table.clear()
        seq.block_hashes.clear()

    def register_computed_blocks(self, seq: Sequence) -> None:
        """Hash every block of `seq` that has become full and computed since last call."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        for i in range(len(seq.block_hashes), seq.num_computed_tokens // bs):
            parent = seq.block_hashes[-1] if seq.block_hashes else ROOT_HASH
            h = hash_block(parent, seq.token_ids[i * bs : (i + 1) * bs])
            seq.block_hashes.append(h)
            block = seq.block_table[i]
            # If another block already holds this content (two identical prompts admitted
            # in the same step), keep the existing entry; ours stays an unhashed private copy.
            if self.block_hash[block] is None and h not in self.hash_to_block:
                self.block_hash[block] = h
                self.hash_to_block[h] = block

    def reset_prefix_cache(self) -> None:
        for h, b in self.hash_to_block.items():
            assert self.block_hash[b] == h
            self.block_hash[b] = None
        self.hash_to_block.clear()

    # ---------------------------------------------------------------- private

    def _take_fresh_block(self) -> int:
        block, _ = self.free_pool.popitem(last=False)
        h = self.block_hash[block]
        if h is not None:  # evict: its content is about to be overwritten
            del self.hash_to_block[h]
            self.block_hash[block] = None
            self.num_evictions += 1
        self.ref_counts[block] = 1
        return block
