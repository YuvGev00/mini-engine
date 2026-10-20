"""Request state: one `Sequence` per generation request."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0  # 0.0 means greedy
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()


_seq_counter = itertools.count()


@dataclass(eq=False)
class Sequence:
    """All token ids (prompt + generated) plus the bookkeeping the scheduler needs.

    Invariant: tokens `[0, num_computed_tokens)` have their K/V in the cache, in the
    blocks listed by `block_table`. The scheduler only ever asks the model to compute
    tokens `[num_computed_tokens, len(token_ids))`, so prefill, chunked prefill, decode
    and recompute-after-preemption are all the same operation.
    """

    token_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    seq_id: int = field(default_factory=lambda: next(_seq_counter))
    status: SequenceStatus = SequenceStatus.WAITING
    num_computed_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    # Chained hashes of the full blocks of `token_ids` that were computed so far.
    block_hashes: list[bytes] = field(default_factory=list)
    num_prompt_tokens: int = field(init=False)
    num_cached_tokens: int = 0  # tokens served by the prefix cache at the last admission
    num_preemptions: int = 0

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("prompt must contain at least one token")
        self.token_ids = list(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return len(self.token_ids) - self.num_computed_tokens

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED
