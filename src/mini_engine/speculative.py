"""N-gram speculative decoding (a.k.a. prompt lookup decoding).

Proposal: find the most recent earlier occurrence of the sequence's last `n` tokens
(longest `n` first, from `ngram_max` down to `ngram_min`) and propose the `k` tokens
that followed it. This costs no extra model and works well whenever the output repeats
spans of the input: code edits, extraction, summaries that quote, agent tool output.

Verification (in `LLMEngine.step`): the drafts are appended to the sequence and the
target model computes them in the same forward pass as the real last token, like a
prefill chunk. With greedy decoding the longest prefix of drafts that matches the
model's own argmax is accepted, plus one token from the model, so the output is
identical to plain decoding; only the number of forward passes changes.
"""

from __future__ import annotations

from collections.abc import Sequence as Seq
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeculativeConfig:
    num_speculative_tokens: int = 4  # k: drafts proposed per step
    ngram_max: int = 3
    ngram_min: int = 1

    def __post_init__(self) -> None:
        if self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        if not 1 <= self.ngram_min <= self.ngram_max:
            raise ValueError("need 1 <= ngram_min <= ngram_max")


def propose_ngram(token_ids: Seq[int], k: int, ngram_max: int, ngram_min: int) -> list[int]:
    """Up to `k` draft tokens continuing `token_ids`, or [] if no earlier n-gram matches."""
    length = len(token_ids)
    for n in range(min(ngram_max, length - 1), ngram_min - 1, -1):
        suffix = list(token_ids[length - n :])
        # Most recent earlier occurrence; it must be followed by at least one token.
        for start in range(length - n - 1, -1, -1):
            if list(token_ids[start : start + n]) == suffix:
                follow = token_ids[start + n : start + n + k]
                return list(follow)
    return []


def verify_greedy(drafts: Seq[int], predicted: Seq[int]) -> tuple[list[int], int]:
    """Accept drafts against the target's greedy predictions.

    `predicted[j]` is the target's argmax for the token right after position j of the
    computed chunk (j = 0 is the real last token). Returns the new tokens (accepted
    drafts, then the target's own next token if its logits were computed) and `a`, the
    number of drafts accepted.
    """
    a = 0
    while a < len(drafts) and a < len(predicted) and drafts[a] == predicted[a]:
        a += 1
    new_tokens = list(drafts[:a])
    if a < len(predicted):
        new_tokens.append(predicted[a])
    return new_tokens, a
