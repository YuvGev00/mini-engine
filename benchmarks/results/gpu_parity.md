# Greedy parity on the GPU (NVIDIA L4, Qwen3-0.6B, 8 chat prompts, up to 48 new tokens)

Output of `python benchmarks/gpu_parity.py` (vLLM 0.32.0 runtime: torch 2.13.0+cu130, transformers 5.19.0).
Both sides stop at EOS; "first divergence" is the index of the first differing token (None = identical).

| dtype | mini-engine mode | identical to HF | first divergence per sequence | n-gram acceptance |
|---|---|---|---|---|
| float32 | plain | **8/8** | all None | - |
| float32 | speculative (k=4) | **8/8** | all None | 0.12 |
| bfloat16 | plain | 5/8 | 12, –, –, –, –, 23, –, 16 | - |
| bfloat16 | speculative (k=4) | 4/8 | 12, 2, –, –, –, 15, –, 5 | 0.13 |

Speculative vs plain mini-engine: 8/8 identical in float32, 5/8 in bfloat16.

In bfloat16 a different kernel or reduction order (and verifying 5 tokens in one pass instead of one
at a time) legitimately flips near-tie argmaxes, so token equality is only asserted in float32.

A first version of this check compared forced-length runs and showed 3/8. `gpu_divergence.py` traced
it to the comparison itself: HF's `min_new_tokens` bans EOS, while the engine's `ignore_eos` may pick
EOS and continue, so the runs split exactly where the model ends its answer (HF's own full forward pass
ranks the engine's token first there, by 3-12 logits). HF `sdpa` and `eager` agree 8/8 with each other.
