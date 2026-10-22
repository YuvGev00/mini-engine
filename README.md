# mini-engine

A small LLM inference engine written from scratch in PyTorch: a paged KV cache,
continuous batching with preemption, chunked prefill, prefix caching and n-gram
speculative decoding. It runs
Qwen3 (tested on Qwen3-0.6B), and its greedy output is **token-identical to Hugging Face
`generate`**, both one request at a time and batched.

The core is about 860 lines in `src/mini_engine/`, small enough to read in one sitting.

```
sequence.py       request state: tokens, num_computed_tokens, block table
block_manager.py  block allocator, ref counts, chained block hashes, LRU eviction (pure Python)
scheduler.py      continuous batching under a token budget; FCFS; recompute preemption
attention.py      paged attention: scatter K/V into blocks, gather via block tables, one SDPA call
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
   context through its block table, and runs one masked SDPA call over the padded batch.
   Prefill and decode share this path, so they can sit in the same batch.
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

| Workload | HF generate | mini-engine | vLLM |
|---|---|---|---|
| Throughput: 64 requests × (256 in, 128 out) | 1,292 | 294 | 3,256 |
| Shared prefix: 64 × (512 shared + 32 in, 64 out), cache warm | 484 | 167 | 3,168 |
| Copy task, 1 request (speculation off → on) | 26 | 21 → **100** | 171 → 135 |
| Copy task, 8 requests (speculation off → on) | 205 | 166 → **750** | 1,228 → 4,029 |

- **Speculative decoding works as designed:** 4.7× faster on the copy task (one request) and
  4.5× with 8 requests, with every draft accepted. On copy-heavy work, mini-engine with
  speculation outruns HF `generate` 3.7×. On ordinary chat prompts acceptance is 12-13%
  (`results/gpu_parity.md`), so the gain there is small.
- **Prefix caching:** 1.40× on the shared-prefix workload (32,256 prompt tokens served from
  the cache), against 2.21× for vLLM.
- **Raw throughput is the cost of the design.** mini-engine is 4.4× slower than HF and
  11× slower than vLLM on plain batched decoding. Each step gathers every sequence's whole
  context out of the block cache and runs a masked, padded SDPA call, with Python
  bookkeeping per step and no custom kernels or CUDA graphs ([DECISIONS #1](DECISIONS.md)).
  That choice keeps one code path that runs on a Mac and is easy to verify. Closing the gap
  needs a paged attention kernel (FlashInfer or a Triton decode kernel) and CUDA graphs.
- **Correctness on the GPU:** greedy output is token-identical to HF on 8/8 chat prompts in
  fp32, with and without speculation. In bf16, 5/8 match, because different kernels
  legitimately flip near-tie argmaxes. Details, and a comparison bug that a diagnosis script
  caught, are in [`benchmarks/results/gpu_parity.md`](benchmarks/results/gpu_parity.md).

## Tests

| file | what it pins down |
|---|---|
| `test_block_manager.py` | alloc/free, all-or-nothing allocation, ref-count sharing, the "always recompute one token" rule, LRU eviction order, revived-block capacity, duplicate content |
| `test_scheduler.py` | mixed prefill+decode batches, token-budget chunking, FCFS without queue jumping, preempting the newest, prefix-cache hit on re-admission, EOS |
| `test_attention.py` | paged attention vs a dense causal reference on scattered block tables (prefill, decode and chunk continuation in one batch) |
| `test_speculative.py` | n-gram proposer and greedy verifier; output identical to HF unbatched, batched, with a 3-token budget (verification cut short), under preemption and with EOS / max_tokens cutting accepted drafts; fewer steps per sequence; sampled requests not speculated; blocks released |
| `test_parity.py` | logits match HF to 1e-4; greedy tokens match HF unbatched, batched, with prefix hits, under preemption and with 7-token chunked prefill; real Qwen3-0.6B (opt-in) |

CI runs ruff, `mypy --strict` and the tests on CPU-only torch. It downloads no model.

## Credit

The design follows [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
(and through it, vLLM): the block-manager/scheduler/model-runner split, hash-based prefix
caching and recompute preemption. The code was written independently. See
[DECISIONS.md](DECISIONS.md) for where this design differs and why.

## Roadmap

Done: M4 (the correct, tested core) and M5 (n-gram speculative decoding, the GPU benchmark
against HF and vLLM, and GPU parity). Next:
- a paged decode-attention kernel and CUDA graphs, to close the throughput gap;
- draft-model speculative decoding (e.g. Qwen3-0.6B drafting for Qwen3-4B);
- an OpenAI-compatible HTTP server with streaming;
- a prefix-cache retention study on real agent traces (cf. vLLM RFC #37003).
