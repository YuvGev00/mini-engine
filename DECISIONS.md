# Design decisions

The reference design is [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). Where
mini-engine differs, it is because of one of two constraints: it has to run on an M2 Mac
(MPS/CPU, so no flash-attn, Triton or CUDA graphs), and every line has to be explainable
in an interview. The CUDA fast path (#12) is an addition on top: every CPU path still runs,
and the GPU path is tested against it.

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
- *Later:* on CUDA, SDPA with a boolean mask and GQA has no fused kernel and falls back to
  its math path, which copies K/V once per query head. The "grouped" backend folds each KV
  head's query group into the query axis and uses two batched matmuls instead (3× faster
  per decode step). The KV cache is stored head-major (`[kvh, blocks, block_size, d]`) so
  the gather lands directly in the layout those matmuls want, with no transpose copy.

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

## 9. Speculative decoding: n-gram drafts through the normal scheduler
- **Drafts are just uncomputed tokens.** They are appended to `token_ids` before `schedule()`,
  so the scheduler allocates blocks for them and the attention path computes them like a
  prefill chunk. The only special cases are in the engine: logits are read at every computed
  position, and drafts are stripped from sequences that were not scheduled (preempted or out
  of budget) before anything else sees them.
- **Rollback is a counter.** After verification, `num_computed_tokens` is set to the real
  last token plus the accepted drafts that were computed; rejected drafts' K/V stays in the
  cache and is overwritten later. Blocks are only hashed up to `num_computed_tokens`, so the
  prefix cache never registers a rejected draft.
- **A cut-short chunk still works.** If the token budget computes only part of the drafts,
  the accepted prefix is limited to what was computed, and the step yields no bonus token.
- **Greedy only.** Sampled requests skip speculation: exact speculative *sampling* needs
  rejection sampling against draft probabilities, which an n-gram proposer does not have.
- **n-gram first, draft model later.** Prompt lookup needs no second model or second KV cache,
  and it is the case where speculation helps most (copying, extraction, code edits).

## 10. Benchmarks: separate processes, same token ids, staged prefix
- Each system runs in its own process: vLLM reserves most GPU memory, and torch state
  must not leak between systems.
- Prompts are token ids from a fixed seed, so all three systems see identical inputs.
- **Prefix workload is staged** (1 request, then 63) for every system. Admitted together,
  identical prefixes cannot hit mini-engine's cache (§4), and a server with a cached
  system prompt is the realistic case. A CPU smoke test showed 0 cached tokens before
  this change.
- **Throughput runs force output length** (`ignore_eos` / HF `min_new_tokens`) and make no
  correctness claim: the two options differ at EOS (§11).

## 11. GPU parity is checked with natural EOS stopping
The first GPU parity check forced 48 tokens on both sides and matched HF on 3/8 prompts in
fp32. `benchmarks/gpu_divergence.py` showed this was the comparison, not the engine: HF
`sdpa` vs `eager` agree 8/8, and at each divergence HF's own full forward ranks the engine's
token first by 3-12 logits. HF `min_new_tokens` *bans* EOS, while `ignore_eos` lets the engine
choose EOS and continue, so the runs split exactly where the model ends its answer. With natural
EOS stopping, parity is 8/8 in fp32 with and without speculation (`benchmarks/results/gpu_parity.md`).

## 12. CUDA fast path: Triton kernels and CUDA graphs for decode and verify steps
Profiling the decode step on an L4 (`benchmarks/profile_engine.py`) showed two walls in turn.
First, memory: gathering every context into a padded tensor already ran at memory bandwidth,
then the matmuls read it again. A Triton kernel (`triton_attention.py`) now walks the block
table itself, one program per (sequence, KV head), with an online softmax, so each step
reads the cache once and writes nothing. Second, launches: the GPU then finished its work in
half the step while the CPU issued ~1,600 kernel launches. The decode forward is captured as
a CUDA graph per batch-size bucket (1, 2, 4, 8, then every 16) at engine start and replayed.
- *Scope:* steps where every sequence has at most 8 new tokens use the kernels: one token
  (decode) goes to the decode kernel, more (speculative verify, short chunks) to a
  multi-query kernel that puts [query tokens x group heads] on the rows of a `tl.dot`, with
  a causal mask inside the new tokens. Two graph sets are captured: query length 1, and
  k + 1 when speculation is on, with tokens laid out padded so one graph fits any mix of
  draft lengths. Longer prefill chunks stay on the eager grouped path. Before the verify
  graphs existed, speculation was slower than graph-replayed plain decoding, which is the
  reason the second set exists.
- *Padding:* a batch is padded up to its bucket. Padded rows and tokens write K/V to one spare cache
  block that the block manager never hands out, attend only to that slot, and are dropped.
- *A bug worth remembering:* a graph records raw addresses. The first version kept one input
  (`logits_index`) in a local variable; after capture its memory was reused and replays read
  garbage indices (a device-side assert). Every tensor a graph reads now lives on the object.
- *Tests:* both kernels are compared with the PyTorch path in fp32 and bf16, and graph
  decoding (plain and speculative) with eager decoding token for token (`tests/test_gpu.py`,
  skipped without CUDA). `tests/test_step_graphs.py` runs the same graph bookkeeping on CPU
  in CI, with capture faked and the kernels replaced by a dense reference. End to end,
  `gpu_parity.py` still gives 8/8 against HF in fp32.

## Not in scope (yet)
Top-p/top-k sampling, stop strings, streaming, tensor parallelism,
draft-model speculative decoding, an HTTP server. See the README roadmap.
