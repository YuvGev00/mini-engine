"""GPU-only checks (CUDA + Triton): the paged decode kernel against the PyTorch grouped
path, and CUDA-graph decode against eager decode, end to end."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import torch

from mini_engine import EngineConfig, LLMEngine, SamplingParams
from mini_engine.attention import AttentionMetadata, paged_attention
from mini_engine.speculative import SpeculativeConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="needs CUDA and Triton",
)

H, KVH, D, BS, NUM_BLOCKS = 16, 8, 128, 16, 256


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_kernel_matches_grouped(dtype: torch.dtype) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    dev = torch.device("cuda")
    k_cache = torch.randn(KVH, NUM_BLOCKS, BS, D, device=dev, dtype=dtype, generator=g)
    v_cache = torch.randn(KVH, NUM_BLOCKS, BS, D, device=dev, dtype=dtype, generator=g)
    # Context lengths: a single token, block-aligned, unaligned, and longer than one tile.
    ctx_lens = [1, 16, 17, 33, 100, 255]
    perm = torch.randperm(NUM_BLOCKS).tolist()
    tables = [[perm.pop() for _ in range(-(-n // BS))] for n in ctx_lens]
    q = torch.randn(len(ctx_lens), H, D, device=dev, dtype=dtype, generator=g)
    k = torch.randn(len(ctx_lens), KVH, D, device=dev, dtype=dtype, generator=g)
    v = torch.randn(len(ctx_lens), KVH, D, device=dev, dtype=dtype, generator=g)

    outs = {}
    for backend in ("grouped", "triton"):
        meta = AttentionMetadata.build([1] * len(ctx_lens), ctx_lens, tables, BS, dev, backend=backend)
        outs[backend] = paged_attention(q, k, v, k_cache.clone(), v_cache.clone(), meta)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(outs["triton"], outs["grouped"], rtol=tol, atol=tol)


def test_cuda_graph_decode_matches_eager(tiny_hf_model: Any) -> None:
    """max_num_seqs=5 gives buckets 1, 2, 4, 5, so batches of 3 run padded."""
    prompts = [[(7 * i + j) % 500 + 1 for j in range(5 + 9 * i)] for i in range(5)]
    params = SamplingParams(max_tokens=40, ignore_eos=True)
    outs = {}
    for name, backend, graphs in (("eager", "grouped", False), ("graphs", "triton", True)):
        cfg = EngineConfig(
            device="cuda",
            block_size=4,
            num_blocks=256,
            max_num_seqs=5,
            max_model_len=256,
            attention_backend=backend,
            cuda_graphs=graphs,
        )
        eng = LLMEngine.from_hf_model(tiny_hf_model, cfg)
        assert (eng.decode_graphs is not None) == graphs
        outs[name] = [eng.generate(prompts[:3], params), eng.generate(prompts, params)]
    assert outs["graphs"] == outs["eager"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_multi_query_kernel_matches_grouped(dtype: torch.dtype) -> None:
    """Verify-shaped steps: 1-5 new tokens per sequence on top of cached contexts."""
    g = torch.Generator(device="cuda").manual_seed(1)
    dev = torch.device("cuda")
    k_cache = torch.randn(KVH, NUM_BLOCKS, BS, D, device=dev, dtype=dtype, generator=g)
    v_cache = torch.randn(KVH, NUM_BLOCKS, BS, D, device=dev, dtype=dtype, generator=g)
    q_lens = [1, 5, 3, 5, 2, 4]
    ctx_lens = [1, 5, 18, 33, 100, 255]
    perm = torch.randperm(NUM_BLOCKS).tolist()
    tables = [[perm.pop() for _ in range(-(-n // BS))] for n in ctx_lens]
    t = sum(q_lens)
    q = torch.randn(t, H, D, device=dev, dtype=dtype, generator=g)
    k = torch.randn(t, KVH, D, device=dev, dtype=dtype, generator=g)
    v = torch.randn(t, KVH, D, device=dev, dtype=dtype, generator=g)

    outs = {}
    for backend in ("grouped", "triton"):
        meta = AttentionMetadata.build(q_lens, ctx_lens, tables, BS, dev, backend=backend)
        outs[backend] = paged_attention(q, k, v, k_cache.clone(), v_cache.clone(), meta)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(outs["triton"], outs["grouped"], rtol=tol, atol=tol)


def test_cuda_graph_speculative_matches_eager(tiny_hf_model: Any) -> None:
    prompts = [([5, 6, 7, 8, 9] * 9)[: 8 + 7 * i] for i in range(5)]
    params = SamplingParams(max_tokens=40, ignore_eos=True)
    outs = {}
    for name, backend, graphs in (("eager", "grouped", False), ("graphs", "triton", True)):
        cfg = EngineConfig(
            device="cuda",
            block_size=4,
            num_blocks=256,
            max_num_seqs=5,
            max_model_len=256,
            attention_backend=backend,
            cuda_graphs=graphs,
            speculative=SpeculativeConfig(4),
        )
        eng = LLMEngine.from_hf_model(tiny_hf_model, cfg)
        assert (eng.verify_graphs is not None) == graphs
        outs[name] = [eng.generate(prompts[:3], params), eng.generate(prompts, params)]
        assert eng.stats.accepted_tokens > 0
    assert outs["graphs"] == outs["eager"]
