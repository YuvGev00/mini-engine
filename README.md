# mini-engine

A small LLM inference engine written from scratch in PyTorch: a paged KV cache,
continuous batching with preemption, chunked prefill and prefix caching. It runs
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

## Tests

| file | what it pins down |
|---|---|
| `test_block_manager.py` | alloc/free, all-or-nothing allocation, ref-count sharing, the "always recompute one token" rule, LRU eviction order, revived-block capacity, duplicate content |
| `test_scheduler.py` | mixed prefill+decode batches, token-budget chunking, FCFS without queue jumping, preempting the newest, prefix-cache hit on re-admission, EOS |
| `test_attention.py` | paged attention vs a dense causal reference on scattered block tables (prefill, decode and chunk continuation in one batch) |
| `test_parity.py` | logits match HF to 1e-4; greedy tokens match HF unbatched, batched, with prefix hits, under preemption and with 7-token chunked prefill; real Qwen3-0.6B (opt-in) |

CI runs ruff, `mypy --strict` and the tests on CPU-only torch. It downloads no model.

## Credit

The design follows [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
(and through it, vLLM): the block-manager/scheduler/model-runner split, hash-based prefix
caching and recompute preemption. The code was written independently. See
[DECISIONS.md](DECISIONS.md) for where this design differs and why.

## Roadmap

This is milestone M4, the correct and tested core. Next:
- speculative decoding (draft model and n-gram), with exact-match acceptance tests;
- an OpenAI-compatible HTTP server with streaming;
- a benchmark harness on Modal L4 GPUs: SLO and goodput curves against vLLM;
- a prefix-cache retention study on real agent traces (cf. vLLM RFC #37003).
