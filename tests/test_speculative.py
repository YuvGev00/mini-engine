"""N-gram speculative decoding: greedy output must be token-identical to plain decoding.

Repetitive prompts make the tiny random model's greedy output loop, so n-gram drafts
get accepted and the verify / partial-accept / rollback paths are actually exercised;
each parity test also asserts that some drafts were accepted.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import random_prompts
from test_parity import hf_greedy

from mini_engine.engine import EngineConfig, LLMEngine
from mini_engine.sequence import SamplingParams
from mini_engine.speculative import SpeculativeConfig, propose_ngram, verify_greedy

MAX_NEW = 24


def engine(model: Any, spec: bool, **overrides: Any) -> LLMEngine:
    cfg: dict[str, Any] = {"block_size": 4, "num_blocks": 256, **overrides}
    if spec:
        cfg["speculative"] = SpeculativeConfig(num_speculative_tokens=4)
    return LLMEngine.from_hf_model(model, EngineConfig(**cfg))


@pytest.fixture(scope="module")
def prompts() -> list[list[int]]:
    repetitive = [[5, 6, 7, 8] * 6, [9, 10, 11] * 5 + [12], [3] * 10, [20, 21, 20, 21, 22] * 3]
    return repetitive + random_prompts(4, lo=2, hi=30, seed=7)


@pytest.fixture(scope="module")
def reference(tiny_hf_model: Any, prompts: list[list[int]]) -> list[list[int]]:
    return hf_greedy(tiny_hf_model, prompts, max_new=MAX_NEW)


# ------------------------------------------------------------------ proposer / verifier


def test_propose_prefers_longest_most_recent_ngram() -> None:
    ids = [1, 2, 3, 9, 1, 2, 3, 7, 8, 5, 2, 3]
    # suffix (2, 3) last occurred at index 5, followed by 7, 8, 5
    assert propose_ngram(ids, k=3, ngram_max=3, ngram_min=1) == [7, 8, 5]
    # the 3-gram (5, 2, 3) never occurred before, so the proposer backs off to n=2
    assert propose_ngram([4, 4, 4], k=2, ngram_max=3, ngram_min=1) == [4]
    assert propose_ngram([1, 2, 3], k=4, ngram_max=3, ngram_min=1) == []
    assert propose_ngram([1, 2, 1], k=4, ngram_max=3, ngram_min=1) == [2, 1]


def test_verify_accepts_matching_prefix_plus_one() -> None:
    assert verify_greedy([7, 8, 9], [7, 8, 1, 4]) == ([7, 8, 1], 2)
    assert verify_greedy([7, 8], [7, 8, 3]) == ([7, 8, 3], 2)  # all accepted, bonus token
    assert verify_greedy([7, 8], [5, 8, 3]) == ([5], 0)  # first draft wrong
    # chunk cut short by the token budget: no logits after the last computed draft
    assert verify_greedy([7, 8, 9], [7, 8]) == ([7, 8], 2)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SpeculativeConfig(num_speculative_tokens=0)
    with pytest.raises(ValueError):
        SpeculativeConfig(ngram_min=3, ngram_max=2)


# ------------------------------------------------------------------------- parity


def _run(eng: LLMEngine, prompts: list[list[int]], max_new: int = MAX_NEW) -> list[list[int]]:
    return eng.generate(prompts, SamplingParams(max_tokens=max_new))


def test_matches_hf_unbatched(
    tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]]
) -> None:
    eng = engine(tiny_hf_model, spec=True)
    assert [_run(eng, [p])[0] for p in prompts] == reference
    assert eng.stats.accepted_tokens > 0


def test_matches_hf_batched_with_fewer_steps(
    tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]]
) -> None:
    plain, spec = engine(tiny_hf_model, spec=False), engine(tiny_hf_model, spec=True)
    assert _run(plain, prompts) == reference
    assert _run(spec, prompts) == reference
    assert spec.stats.accepted_tokens > 0
    assert spec.stats.generated_tokens == plain.stats.generated_tokens
    # A batch runs as many steps as its slowest member, so count steps per sequence:
    # run alone, every accepted draft saves that sequence a forward pass.
    plain_steps = spec_steps = 0
    for p in prompts:
        e_plain, e_spec = engine(tiny_hf_model, spec=False), engine(tiny_hf_model, spec=True)
        assert _run(e_spec, [p]) == _run(e_plain, [p])
        plain_steps += e_plain.stats.steps
        spec_steps += e_spec.stats.steps
        assert e_plain.stats.steps - e_spec.stats.steps <= e_spec.stats.accepted_tokens
    assert spec_steps < plain_steps


def test_matches_hf_with_tiny_token_budget(
    tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]]
) -> None:
    # A 3-token budget cuts verification chunks short (1 real + up to 4 drafts).
    eng = engine(tiny_hf_model, spec=True, max_num_batched_tokens=3)
    assert _run(eng, prompts) == reference
    assert eng.stats.accepted_tokens > 0


def test_matches_hf_under_preemption(
    tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]]
) -> None:
    eng = engine(tiny_hf_model, spec=True, num_blocks=26)
    assert _run(eng, prompts) == reference
    assert eng.scheduler.num_preemptions > 0
    assert eng.stats.accepted_tokens > 0


def test_eos_and_max_tokens_cut_accepted_drafts(tiny_hf_model: Any, prompts: list[list[int]]) -> None:
    plain, spec = engine(tiny_hf_model, spec=False), engine(tiny_hf_model, spec=True)
    unbounded = _run(plain, prompts)
    eos = unbounded[0][5]  # a token the first (looping) prompt produces mid-output
    for e in (plain, spec):
        e.scheduler.eos_token_ids = {eos}
    expected = _run(plain, prompts, max_new=11)
    assert _run(spec, prompts, max_new=11) == expected
    assert all(len(o) <= 11 for o in expected)
    assert any(o and o[-1] == eos and len(o) < 11 for o in expected)


def test_sampled_requests_are_not_speculated(tiny_hf_model: Any, prompts: list[list[int]]) -> None:
    eng = engine(tiny_hf_model, spec=True)
    outs = eng.generate(prompts, SamplingParams(max_tokens=8, temperature=0.8))
    assert all(len(o) == 8 for o in outs)
    assert eng.stats.draft_tokens == 0


def test_blocks_are_released(tiny_hf_model: Any, prompts: list[list[int]]) -> None:
    eng = engine(tiny_hf_model, spec=True, num_blocks=64)
    _run(eng, prompts)
    assert eng.block_manager.num_free_blocks == 64
