"""Generate with Qwen3-0.6B and show prefix-cache reuse. Short on purpose.

uv run python examples/generate.py [--device mps]
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer

from mini_engine import EngineConfig, LLMEngine, SamplingParams

MODEL = "Qwen/Qwen3-0.6B"
# A shared prefix longer than a few 16-token blocks, as in a real system prompt.
SYSTEM = (
    "You are a terse assistant for engineers who build LLM serving systems. "
    "You answer in exactly one sentence, without lists, code or caveats, and you never "
    "repeat the question back to the user.\n\nQuestion: "
)
QUESTIONS = ["What is a KV cache?", "Why batch requests?", "What is paged attention?"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    engine = LLMEngine.from_pretrained(MODEL, EngineConfig(device=args.device, dtype=dtype, num_blocks=128))
    params = SamplingParams(max_tokens=args.max_tokens)
    prompts = [tok(SYSTEM + q + "\nAnswer:").input_ids for q in QUESTIONS]

    start = time.perf_counter()
    # The first request warms the prefix cache; the rest are batched and reuse it.
    outputs = engine.generate(prompts[:1], params) + engine.generate(prompts[1:], params)
    elapsed = time.perf_counter() - start

    for q, out in zip(QUESTIONS, outputs, strict=True):
        print(f"Q: {q}\nA: {tok.decode(out, skip_special_tokens=True).strip()}\n")
    s = engine.stats
    print(
        f"{s.generated_tokens} tokens in {elapsed:.1f}s over {s.steps} steps | "
        f"prompt tokens {s.prompt_tokens}, served from prefix cache {engine.scheduler.num_cached_tokens}"
    )


if __name__ == "__main__":
    main()
