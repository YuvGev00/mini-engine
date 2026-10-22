"""Greedy parity of mini-engine against HF on the real model, on the GPU.

fp32: mini-engine (batched, with and without speculation) must match HF token for token.
bf16: different kernels legitimately flip near-tie argmaxes, so this reports where each
sequence first diverges instead of asserting equality.

Both sides stop naturally at EOS. (Forcing a length differs between the two: HF's
`min_new_tokens` *bans* EOS, while `ignore_eos` lets the engine pick EOS and keep going, so
a forced-length comparison diverges exactly where the model wants to end its answer.)

    python benchmarks/gpu_parity.py
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_engine import EngineConfig, LLMEngine, SamplingParams
from mini_engine.speculative import SpeculativeConfig

MODEL = "Qwen/Qwen3-0.6B"
QUESTIONS = [
    "Explain what a KV cache is in two sentences.",
    "Write a haiku about the sea.",
    "List three prime numbers and explain why they are prime.",
    "What is the capital of France, and why is it famous?",
    "Translate 'good morning, how are you?' into Spanish.",
    "Describe continuous batching for LLM serving.",
    "Repeat exactly: the quick brown fox jumps over the lazy dog.",
    "Give a one-line summary of the theory of relativity.",
]
NEW = 48


def hf_greedy(model, prompts: list[list[int]]) -> list[list[int]]:  # type: ignore[no-untyped-def]
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    outs = []
    for p in prompts:
        ids = torch.tensor([p], device="cuda")
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=NEW,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        toks = out[0, len(p) :].tolist()
        end = next((i + 1 for i, t in enumerate(toks) if t in eos_ids), len(toks))
        outs.append(toks[:end])  # HF pads after EOS; the engine stops at it
    return outs


def first_divergence(a: list[int], b: list[int]) -> int | None:
    diff = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    if diff is None and len(a) != len(b):
        return min(len(a), len(b))
    return diff


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for q in QUESTIONS:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
        prompts.append(list(enc["input_ids"] if hasattr(enc, "keys") else enc))
    params = SamplingParams(max_tokens=NEW)

    for dtype in (torch.float32, torch.bfloat16):
        hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to("cuda").eval()
        with torch.inference_mode():
            ref = hf_greedy(hf, prompts)
        del hf
        results = {}
        for spec in (False, True):
            cfg = EngineConfig(
                device="cuda",
                dtype=dtype,
                num_blocks=512,
                speculative=SpeculativeConfig(num_speculative_tokens=4) if spec else None,
            )
            eng = LLMEngine.from_pretrained(MODEL, cfg)
            results[spec] = eng.generate(prompts, params)
            acc = eng.stats.acceptance_rate
            del eng
            torch.cuda.empty_cache()
            same = sum(a == b for a, b in zip(results[spec], ref, strict=True))
            div = [first_divergence(a, b) for a, b in zip(results[spec], ref, strict=True)]
            name = str(dtype).removeprefix("torch.")
            print(
                f"PARITY {name} speculative={spec}: {same}/{len(ref)} sequences identical to HF; "
                f"first divergence per sequence {div}; acceptance {acc:.2f}"
            )
        spec_same = sum(a == b for a, b in zip(results[True], results[False], strict=True))
        name = str(dtype).removeprefix("torch.")
        print(f"PARITY {name}: speculative vs plain identical {spec_same}/{len(prompts)}")


if __name__ == "__main__":
    main()
