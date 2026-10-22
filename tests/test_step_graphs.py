"""The CUDA-graph step path, run on CPU: graph capture and replay are faked (replay reruns
the forward on the captured buffers) and the Triton kernels are replaced by a dense
reference. What this pins down is the bookkeeping: padded token layout, spare-slot
padding, buffer reuse across steps, and which logits rows the engine reads back."""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any

import pytest
import torch

from mini_engine import EngineConfig, LLMEngine, SamplingParams
from mini_engine.attention import AttentionMetadata
from mini_engine.speculative import SpeculativeConfig


def reference_multi_query(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    query_lens: torch.Tensor,
) -> torch.Tensor:
    kvh, _, bs, d = k_cache.shape
    out = torch.zeros_like(q)
    for i in range(q.shape[0]):
        n, n_q = int(context_lens[i]), int(query_lens[i])
        slots = (block_tables[i, : -(-n // bs), None] * bs + torch.arange(bs)).flatten()[:n]
        k = k_cache.view(kvh, -1, d)[:, slots].repeat_interleave(q.shape[2] // kvh, 0)
        v = v_cache.view(kvh, -1, d)[:, slots].repeat_interleave(q.shape[2] // kvh, 0)
        for j in range(n_q):
            seen = n - n_q + j + 1
            scores = torch.einsum("hd,hkd->hk", q[i, j], k[:, :seen]) * d**-0.5
            out[i, j] = torch.einsum("hk,hkd->hd", torch.softmax(scores, -1), v[:, :seen])
    return out


def reference_decode(q: torch.Tensor, *cache_args: torch.Tensor) -> torch.Tensor:
    k_cache, v_cache, block_tables, context_lens = cache_args
    ones = torch.ones_like(context_lens)
    return reference_multi_query(q[:, None], k_cache, v_cache, block_tables, context_lens, ones)[:, 0]


class FakeGraph:
    def __init__(self) -> None:
        self.replay_fn: Any = None

    def replay(self) -> None:
        self.replay_fn()

    def pool(self) -> None:
        return None


@pytest.fixture
def fake_cuda_graphs(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    kernels = types.SimpleNamespace(
        paged_decode_attention=reference_decode, paged_multi_query_attention=reference_multi_query
    )
    monkeypatch.setitem(sys.modules, "mini_engine.triton_attention", kernels)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, pool=None: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    import mini_engine.cuda_graphs as cg

    replays = {"decode": 0, "verify": 0}
    init = cg.StepGraphs.__init__

    def init_with_replay(self: cg.StepGraphs, model: Any, kv_caches: Any, **kwargs: Any) -> None:
        init(self, model, kv_caches, **kwargs)
        q = self.query_len
        for b, graph in self.graphs.items():
            r = b * q
            meta = AttentionMetadata(
                slot_mapping=self.slot_mapping[:r],
                key_slots=self.empty,
                block_tables=self.block_tables[:b],
                context_lens=self.context_lens[:b],
                query_lens=self.query_lens[:b],
                query_index=self.row_index[:r],
                mask=self.empty,
                batch_size=b,
                max_q=q,
                backend="triton",
            )

            def replay(b: int = b, r: int = r, meta: AttentionMetadata = meta, q: int = q) -> None:
                replays["verify" if q > 1 else "decode"] += 1
                out = model(self.input_ids[:r], self.positions[:r], kv_caches, meta, self.row_index[:r])
                self.logits[b].copy_(out)

            graph.replay_fn = replay  # type: ignore[attr-defined]

    monkeypatch.setattr(cg.StepGraphs, "__init__", init_with_replay)
    return replays


def run(model: Any, *, graphs: bool, spec: bool) -> list[list[list[int]]]:
    # Repetitive prompts, so n-gram drafts are proposed and partly accepted.
    prompts = [([5, 6, 7, 8, 9] * 9)[: 8 + 7 * i] for i in range(5)]
    prompts += [[(7 * i + j) % 500 + 1 for j in range(9)] for i in range(2)]
    cfg = EngineConfig(
        block_size=4,
        num_blocks=256,
        max_num_seqs=5,  # buckets 1, 2, 4, 5: batches of 3 run padded
        max_model_len=256,
        attention_backend="triton" if graphs else "sdpa",
        cuda_graphs=graphs,
        speculative=SpeculativeConfig(num_speculative_tokens=4) if spec else None,
    )
    eng = LLMEngine.from_hf_model(model, cfg)
    assert (eng.decode_graphs is not None) == graphs
    assert (eng.verify_graphs is not None) == (graphs and spec)
    params = SamplingParams(max_tokens=40, ignore_eos=True)
    return [eng.generate(prompts[:3], params), eng.generate(prompts, params)]


def test_graph_steps_match_eager(tiny_hf_model: Any, fake_cuda_graphs: dict[str, int]) -> None:
    eager = run(tiny_hf_model, graphs=False, spec=False)
    assert run(tiny_hf_model, graphs=True, spec=False) == eager
    assert run(tiny_hf_model, graphs=True, spec=True) == eager
    assert fake_cuda_graphs["decode"] > 0 and fake_cuda_graphs["verify"] > 0
