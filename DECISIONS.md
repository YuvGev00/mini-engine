# Design decisions

The reference design is [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). Where
mini-engine differs, it is because of one of two constraints: it has to run on an M2 Mac
(MPS/CPU, so no flash-attn, Triton or CUDA graphs), and every line has to be explainable
in an interview. Speed is milestone M5's job.

## 1. Attention: gather + one padded SDPA call, no custom kernels
nano-vllm uses flash-attn varlen kernels for prefill, a paged decode kernel and a Triton
kernel to store K/V. None of these exist on MPS. mini-engine instead (a) scatters the new
K/V with plain index assignment, (b) gathers each sequence's blocks with
`k_cache[block_tables]`, and (c) runs one `scaled_dot_product_attention` call over a batch
padded to `[B, max_q, max_ctx]` with a boolean causal-plus-length mask.
- *Cost:* the gather copies the whole context every step, and padding wastes FLOPs. For
  decode, attention is O(B * max_ctx) memory traffic, not O(sum ctx).
- *Benefit:* a single code path for prefill, decode and chunk continuation, which is what
  makes mixed batches (#2) and chunked prefill (#3) almost free. It is also trivially
  correct to compare against a dense reference (`test_attention.py`).
- Padded query rows always see key 0, so SDPA never returns NaN; those rows are dropped.

## 2. One unified step instead of separate prefill and decode steps
nano-vllm schedules either a prefill batch or a decode batch in each step. Here a
sequence is just "tokens `[num_computed, len)` still to compute", so prefill, decode and
recompute after preemption are the same operation, and they share a batch. This is
closer to vLLM v1's scheduler. It also means a long prefill no longer stalls every
running decode for a step.

## 3. Chunked prefill
It falls out of #2: a prefill larger than the remaining token budget runs as several
chunks, and logits are only computed (and sampled) on the chunk that reaches the end of
the sequence. It is tested down to a 7-token budget.

## 4. Prefix cache
- **Chained hash** `blake2b(parent_hash ‖ token_ids)`, 16 bytes, from the standard library.
  Python's `hash()` is not used, so hashes are stable across processes, which the trace
  study needs. (nano-vllm uses xxhash.)
- **Hash once the block is computed, not when it is allocated.** Blocks get registered in
  `postprocess` after their K/V exists. Generated tokens are hashed too, as their blocks
  fill, which is what multi-turn and agent traces need.
- **Always compute at least one token.** `match_prefix` only looks at `len - 1` tokens,
  because a request needs logits for its last position. This is the same rule as vLLM.
- **Free ≠ evicted.** A block whose ref count reaches 0 keeps its hash and sits in an
  `OrderedDict` LRU pool. Unhashed blocks go to the front, since they are worthless.
  Blocks are freed tail-first, so a sequence's head (the most shareable part) is evicted
  last. Eviction happens only when a fresh block is taken. `num_evictions` is counted for
  the retention study.
- **Duplicates:** if two identical prompts are admitted in the same step, neither can hit
  the cache. The second one's block stays an unhashed private copy, never double-registered.
- **Capacity accounting:** reviving a cached block from the free pool uses up free
  capacity. `try_allocate` counts it, and it is all-or-nothing.

## 5. Preemption: recompute, newest first, no admission in the same step
When a running sequence needs a block and none is free, the most recently admitted
running sequence is preempted. Its blocks are freed, but its tokens (including generated
ones) are kept, and it goes to the front of `waiting`. On re-admission the prefix cache
usually still holds most of its blocks (`test_preemption_keeps_generated_tokens_...`).
Admission is skipped in any step that preempted, to avoid thrashing. Swapping to CPU is
not implemented: recompute is simpler, and the prefix cache makes it cheap.

## 6. Model code mirrors HF numerics op for op
RMSNorm runs in fp32 and is cast back before the weight multiply, and RoPE tables are
built in fp32. These are the same choices as `transformers`' Qwen3. q/k/v and gate/up are
**not** fused into packed linears, unlike nano-vllm. Fusing them is faster, but separate
projections keep HF parameter names, so `load_state_dict(strict=True)` works with no
remapping. No tensor parallelism, no CUDA graphs, no `torch.compile`.

## 7. How parity is tested
- The reference is HF `generate` run **one prompt at a time**, so HF's own left-padding
  numerics are kept out of the comparison. Every engine mode must match it exactly.
- The CI model is a random 2-layer Qwen3 with GQA and tied embeddings. Its init std is
  widened to 0.2 so that logits are peaked, and a near-tie argmax cannot be flipped by
  fp32 rounding differences between two correct implementations. This is a precaution:
  a 16-prompt check with the default 0.02 also matched exactly.
- Parity is checked in fp32. In bf16, different reduction orders legitimately flip
  near-tie tokens, so bf16 needs a tolerance-based test (e.g. a logprob gap), not
  token equality.
- Real Qwen3-0.6B parity (5 prompts × 24 tokens, including a prefix hit) passes
  locally; it is opt-in because CI should not download 1.2 GB.

## 8. Explicit `num_blocks`
nano-vllm sizes the KV cache by profiling free GPU memory. Here `num_blocks` is a config
value (`kv_cache_bytes_per_token()` helps pick it). That is easier to reason about in
tests, and memory profiling on MPS is unreliable anyway. M5 adds profiling for CUDA.

## Not in scope (yet)
Top-p/top-k sampling, stop strings, streaming, CUDA graphs, tensor parallelism,
speculative decoding, an HTTP server. See the README roadmap.
