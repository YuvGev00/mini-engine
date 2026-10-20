"""Greedy token parity with Hugging Face `generate`.

The reference is HF run one prompt at a time (no padding), so HF's own left-padding
numerics are not part of the comparison. The engine is then run unbatched, batched,
with prefix-cache hits, under preemption and with chunked prefill; all must produce
identical tokens.

The real-checkpoint test downloads nothing in CI: it runs only when
MINI_ENGINE_REAL_MODEL is set (and uses ~1.2 GB of cached Qwen3-0.6B weights).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
from conftest import random_prompts

from mini_engine.engine import EngineConfig, LLMEngine
from mini_engine.sequence import SamplingParams

MAX_NEW = 16


def hf_greedy(model: Any, prompts: list[list[int]], max_new: int = MAX_NEW) -> list[list[int]]:
    eos = set(_ids(model.generation_config.eos_token_id))
    outs = []
    for p in prompts:
        ids = torch.tensor([p])
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )[0, len(p) :].tolist()
        for i, t in enumerate(out):  # HF pads after EOS; the engine stops at it
            if t in eos:
                out = out[: i + 1]
                break
        outs.append(out)
    return outs


def _ids(x: Any) -> list[int]:
    return [] if x is None else [x] if isinstance(x, int) else list(x)


@pytest.fixture(scope="module")
def prompts() -> list[list[int]]:
    return random_prompts(8, lo=1, hi=45)


@pytest.fixture(scope="module")
def reference(tiny_hf_model: Any, prompts: list[list[int]]) -> list[list[int]]:
    return hf_greedy(tiny_hf_model, prompts)


def engine(model: Any, **overrides: Any) -> LLMEngine:
    return LLMEngine.from_hf_model(model, EngineConfig(**{"block_size": 4, "num_blocks": 256, **overrides}))


def test_logits_match_hf(tiny_hf_model: Any, prompts: list[list[int]]) -> None:
    eng = engine(tiny_hf_model)
    p = prompts[0]
    seq = eng.add_request(p)
    batch = eng.scheduler.schedule()
    input_ids, positions, meta, _ = eng._prepare(batch)
    with torch.inference_mode():
        ours = eng.model(input_ids, positions, eng.kv_caches, meta, torch.arange(len(p)))
        theirs = tiny_hf_model(torch.tensor([p])).logits[0]
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-4)
    assert seq.num_computed_tokens == 0


def test_unbatched(tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]]) -> None:
    eng = engine(tiny_hf_model, enable_prefix_caching=False)
    got = [eng.generate([p], SamplingParams(max_tokens=MAX_NEW))[0] for p in prompts]
    assert got == reference


def test_batched(tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]]) -> None:
    eng = engine(tiny_hf_model)
    assert eng.generate(prompts, SamplingParams(max_tokens=MAX_NEW)) == reference
    assert max(eng.stats.step_batch_sizes) == len(prompts)


def test_prefix_cache_hits_keep_parity(tiny_hf_model: Any) -> None:
    system = random_prompts(1, 23, 23, seed=7)[0]  # shared "system prompt", 5+ blocks
    tails = random_prompts(5, 1, 12, seed=8)
    prompts = [system + t for t in tails] + [system]
    reference = hf_greedy(tiny_hf_model, prompts)

    eng = engine(tiny_hf_model)
    first = eng.generate(prompts[:1], SamplingParams(max_tokens=MAX_NEW))
    rest = eng.generate(prompts[1:], SamplingParams(max_tokens=MAX_NEW))
    assert first + rest == reference
    # Every later prompt reuses the 5 full blocks of the shared prefix (the bare
    # `system` prompt gets 5 too: its 23rd token is the one recomputed).
    assert eng.scheduler.num_cached_tokens == 5 * 4 * 5


def test_preemption_keeps_parity(
    tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]]
) -> None:
    eng = engine(tiny_hf_model, num_blocks=20)  # 80 slots for ~8 x 40 tokens
    assert eng.generate(prompts, SamplingParams(max_tokens=MAX_NEW)) == reference
    assert eng.scheduler.num_preemptions > 0


def test_chunked_prefill_keeps_parity(
    tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]]
) -> None:
    eng = engine(tiny_hf_model, max_num_batched_tokens=7)
    assert eng.generate(prompts, SamplingParams(max_tokens=MAX_NEW)) == reference


def test_engine_stops_at_eos(
    tiny_hf_model: Any, prompts: list[list[int]], reference: list[list[int]]
) -> None:
    eos = reference[0][3]
    eng = LLMEngine.from_hf_model(tiny_hf_model, EngineConfig(block_size=4, num_blocks=64))
    eng.scheduler.eos_token_ids = {eos}
    out = eng.generate([prompts[0]], SamplingParams(max_tokens=MAX_NEW))[0]
    assert out == reference[0][: reference[0].index(eos) + 1]


@pytest.mark.skipif(not os.environ.get("MINI_ENGINE_REAL_MODEL"), reason="set MINI_ENGINE_REAL_MODEL=1")
def test_qwen3_0_6b_parity() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(name)
    texts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Paged attention stores the KV cache in",
        "The capital of France is a city known for",  # shares a prefix with the first
        "1, 2, 3, 4,",
    ]
    prompts = [tok(t).input_ids for t in texts]
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    reference = hf_greedy(hf, prompts, max_new=24)
    eng = LLMEngine.from_hf_model(hf, EngineConfig(block_size=4, num_blocks=128))
    del hf
    params = SamplingParams(max_tokens=24)
    assert eng.generate(prompts[:1], params) + eng.generate(prompts[1:], params) == reference
    assert eng.scheduler.num_cached_tokens > 0
