# mini-engine

A small LLM inference engine written from scratch in PyTorch: a paged KV cache,
continuous batching with preemption, chunked prefill, prefix caching and n-gram
speculative decoding, plus a CUDA fast path (a Triton paged-decode kernel and CUDA graphs).
It runs
Qwen3 (tested on Qwen3-0.6B), and its greedy output is **token-identical to Hugging Face
`generate`**, both one request at a time and batched.

The core is about 1,500 lines in `src/mini_engine/`, small enough to read in one sitting.

```
sequence.py       request state: tokens, num_computed_tokens, block table
block_manager.py  block allocator, ref counts, chained block hashes, LRU eviction (pure Python)
scheduler.py      continuous batching under a token budget; FCFS; recompute preemption
attention.py      paged attention: scatter K/V into blocks, gather via block tables, one masked call
triton_attention.py  CUDA kernels (decode, multi-query verify): read K/V from the block cache in place
cuda_graphs.py    decode / verify forward captured once per batch-size bucket, then replayed
model.py          Qwen3 decoder over a packed batch (HF-compatible parameter names)
engine.py         schedule -> prepare tensors -> forward -> sample -> postprocess
speculative.py    n-gram (prompt lookup) drafts and greedy verification
```

## Quick start

```bash
uv sync
uv run pytest -q                                   # CPU, tiny random Qwen3, under a second
MINI_ENGINE_REAL_MODEL=1 uv run pytest -q -k qwen3 # parity against real Qwen3-0.6B (~1.2 GB download)
uv run python examples/generate.py --device mps    # or --device cpu
```

```python
from mini_engine import EngineConfig, LLMEngine, SamplingParams

engine = LLMEngine.from_pretrained("Qwen/Qwen3-0.6B", EngineConfig(device="cpu"))
outputs = engine.generate([[...token ids...], [...]], SamplingParams(max_tokens=64))
```

## How one step works

1. **Schedule.** Running sequences go first. Each one asks for the next
   `num_tokens - num_computed_tokens` tokens (one for a decode, many for a prefill),
   capped by the token budget. If the KV cache has no room, the newest running sequence
   is preempted: its blocks are freed and it is re-queued with its tokens kept. Then
   waiting sequences are admitted, and full blocks that are already cached are reused.
2. **Prepare.** Every scheduled token is packed into one flat batch with its position and
   the cache slot its K/V will be written to.
3. **Forward.** Each layer writes the new K/V into the block cache, gathers each sequence's
   context through its block table, and runs one masked attention call over the padded
   batch. Prefill and decode share this path, so they can sit in the same batch. On CUDA,
   decode and speculative-verify steps take a fast path instead: Triton kernels walk the
   block tables directly, and the whole forward is replayed from a captured CUDA graph.
4. **Postprocess.** `num_computed_tokens` advances, newly full blocks get hashed into the
   prefix cache, sampled tokens are appended, and finished sequences free their blocks.

## Speculative decoding

With `EngineConfig(speculative=SpeculativeConfig(num_speculative_tokens=4))`, each greedy
decode step proposes up to k draft tokens by finding the latest earlier occurrence of the
sequence's last n tokens (prompt lookup; no draft model). The drafts are appended as
uncomputed tokens, so the scheduler runs them like a short prefill chunk next to the real
last token. The engine reads logits at every computed position, keeps the drafts that match
the model's own argmax plus one model token, and moves `num_computed_tokens` back over the
rejected drafts, whose K/V is simply overwritten later. Greedy output is therefore identical
to plain decoding, in fewer forward passes. Sampled requests are not speculated.

## Benchmarks (NVIDIA L4, Qwen3-0.6B, bf16)

`benchmarks/bench.py` runs the same token-id workloads through Hugging Face `generate`,
mini-engine and vLLM 0.32.0, each in its own process, greedy, on a Colab L4
(`benchmarks/colab_bench.ipynb`; raw results in [`benchmarks/results/`](benchmarks/results/)).
Output tokens per second, higher is better:

| Workload | HF generate | mini-engine v0.1 | mini-engine now | vLLM |
|---|---|---|---|---|
| Throughput: 64 requests × (256 in, 128 out) | 1,292 | 294 | **1,982** | 3,256 |
| Shared prefix: 64 × (512 shared + 32 in, 64 out), cache warm | 484 | 167 | **1,691** | 3,168 |
| Copy task, 1 request (speculation off → on) | 26 | 21 → 100 | 114 → **484** | 171 → 135 |
| Copy task, 8 requests (speculation off → on) | 205 | 166 → 750 | 809 → **3,075** | 1,228 → 4,029 |

- **6.7× faster than v0.1 on batched decoding, and now 1.5× faster than HF `generate`**
  (61% of vLLM). On the decode-only profile (`benchmarks/profile_engine.py`, 64 sequences)
  a step went from 183 ms to 19 ms (349 → 3,364 tokens/s), in four measured steps:

  | Change | ms/step | What the profiler showed |
  |---|---|---|
  | v0.1: gather + SDPA | 183 | SDPA with a bool mask and GQA falls back to its math path and copies K/V per head |
  | grouped GQA matmuls | 60 | half the GPU time now copies the gathered context into a transposed layout |
  | head-major KV cache | 46 | the gather itself is at memory bandwidth: 44% of GPU time |
  | Triton paged-decode kernel | 38 | the GPU is busy only ~18 ms; the CPU launches ~1,600 kernels per step |
  | CUDA graphs for decode | **19** | step time ≈ GPU work |

- **Speculative decoding on the fast path:** 4.3× over plain decoding with one request and
  3.8× with eight (every draft accepted on the copy task). A verify step carries up to k+1
  tokens per sequence, so it gets its own Triton kernel (a multi-query version of the decode
  kernel, using tensor-core `tl.dot`) and its own set of CUDA graphs. Without them, verify
  steps ran eagerly at ~45 ms and speculation was *slower* than graph-replayed plain decoding
  (106 vs 115 tok/s). With one request, mini-engine with speculation (484 tok/s) beats vLLM
  with or without it (171 / 135) on this task; with eight, vLLM is still ahead (4,029).
- **Prefix caching:** 4.6× on the shared-prefix workload (32,256 prompt tokens served from
  the cache), against 2.21× for vLLM. Part of that is the cold run being slow: prefill still
  uses the padded PyTorch path.
- **Correctness on the GPU:** greedy output is token-identical to HF on 8/8 chat prompts in
  fp32, with and without speculation, with the Triton kernel and CUDA graphs on. In bf16,
  4-5/8 match, because different kernels
  legitimately flip near-tie argmaxes. Details, and a comparison bug that a diagnosis script
  caught, are in [`benchmarks/results/gpu_parity.md`](benchmarks/results/gpu_parity.md).

## Tests

| file | what it pins down |
|---|---|
| `test_block_manager.py` | alloc/free, all-or-nothing allocation, ref-count sharing, the "always recompute one token" rule, LRU eviction order, revived-block capacity, duplicate content |
| `test_scheduler.py` | mixed prefill+decode batches, token-budget chunking, FCFS without queue jumping, preempting the newest, prefix-cache hit on re-admission, EOS |
| `test_attention.py` | paged attention vs a dense causal reference on scattered block tables (prefill, decode and chunk continuation in one batch) |
| `test_speculative.py` | n-gram proposer and greedy verifier; output identical to HF unbatched, batched, with a 3-token budget (verification cut short), under preemption and with EOS / max_tokens cutting accepted drafts; fewer steps per sequence; sampled requests not speculated; blocks released |
| `test_grouped_backend.py` | the grouped-GQA attention backend keeps greedy parity with HF (batched, chunked, preempted, speculative) |
| `test_step_graphs.py` | the CUDA-graph step path on CPU (capture faked, kernels replaced by a dense reference): decode and speculative-verify graphs give the same tokens as eager decoding, with padded batches and reused buffers |
| `test_gpu.py` | CUDA only: the Triton decode and multi-query kernels against the PyTorch path (fp32, bf16, unaligned and multi-tile contexts); CUDA-graph decode and speculative decoding token-identical to eager, including padded batch buckets |
| `test_parity.py` | logits match HF to 1e-4; greedy tokens match HF unbatched, batched, with prefix hits, under preemption and with 7-token chunked prefill; real Qwen3-0.6B (opt-in) |

CI runs ruff, `mypy --strict` and the tests on CPU-only torch. It downloads no model. The
GPU tests run on a Colab L4 (`benchmarks/colab_profile.ipynb`).

## Credit

The design follows [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
(and through it, vLLM): the block-manager/scheduler/model-runner split, hash-based prefix
caching and recompute preemption. The code was written independently. See
[DECISIONS.md](DECISIONS.md) for where this design differs and why.

## Roadmap

Done: M4 (the correct, tested core), M5 (n-gram speculative decoding, the GPU benchmark
against HF and vLLM, and GPU parity) and the GPU fast path (Triton kernels and CUDA graphs
for decode and speculative verify).
Next:
- a paged prefill kernel (prefill still pads and gathers);
- draft-model speculative decoding (e.g. Qwen3-0.6B drafting for Qwen3-4B);
- an OpenAI-compatible HTTP server with streaming;
- a prefix-cache retention study on real agent traces (cf. vLLM RFC #37003).
