"""The "grouped" attention backend (the CUDA default) must keep exact greedy parity with HF.

CI has no GPU, so these run the grouped path on CPU in fp32: unbatched, batched, under
preemption, with chunked prefill and with speculative decoding.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import random_prompts
from test_parity import MAX_NEW, hf_greedy

from mini_engine.engine import EngineConfig, LLMEngine
from mini_engine.sequence import SamplingParams
from mini_engine.speculative import SpeculativeConfig


@pytest.fixture(scope="module")
def prompts() -> list[list[int]]:
    return random_prompts(8, lo=1, hi=45)  # same prompts as test_parity.py


@pytest.fixture(scope="module")
def reference(tiny_hf_model: Any, prompts: list[list[int]]) -> list[list[int]]:
    return hf_greedy(tiny_hf_model, prompts)


def engine(model: Any, **overrides: Any) -> LLMEngine:
    cfg = {"block_size": 4, "num_blocks": 256, "attention_backend": "grouped", **overrides}
    return LLMEngine.from_hf_model(model, EngineConfig(**cfg))


def test_backend_resolution(tiny_hf_model: Any) -> None:
    auto = LLMEngine.from_hf_model(tiny_hf_model, EngineConfig(block_size=4, num_blocks=16))
    assert auto.attention_backend == "sdpa"  # CPU default stays the SDPA reference
    with pytest.raises(ValueError):
        LLMEngine.from_hf_model(tiny_hf_model, EngineConfig(attention_backend="flash"))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"num_blocks": 20},  # forces preemption
        {"max_num_batched_tokens": 7},  # chunked prefill
        {"speculative": SpeculativeConfig(num_speculative_tokens=4)},
    ],
    ids=["batched", "preemption", "chunked", "speculative"],
)
def test_grouped_matches_hf(
    tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]], overrides: dict[str, Any]
) -> None:
    eng = engine(tiny_hf_model, **overrides)
    assert eng.generate(prompts, SamplingParams(max_tokens=MAX_NEW)) == reference
    if "num_blocks" in overrides:
        assert eng.scheduler.num_preemptions > 0


def test_grouped_unbatched(tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]]) -> None:
    eng = engine(tiny_hf_model, enable_prefix_caching=False)
    assert [eng.generate([p], SamplingParams(max_tokens=MAX_NEW))[0] for p in prompts] == reference
