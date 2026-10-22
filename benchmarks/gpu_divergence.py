"""Is a GPU fp32 divergence from HF a bug or a near-tie? Two checks on the real model:

1. HF against itself: `sdpa` vs `eager` attention, both fp32 on the GPU. If two correct HF
   paths also diverge, kernel-level rounding is enough to flip greedy choices.
2. At each point where mini-engine first differs from HF, the gap between HF's top-2
   logits there. A near-tie (tiny gap) is rounding; a large gap would be a real bug.

    python benchmarks/gpu_divergence.py
"""

from __future__ import annotations

import torch
from gpu_parity import NEW, QUESTIONS, first_divergence, hf_greedy
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_engine import EngineConfig, LLMEngine, SamplingParams

MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for q in QUESTIONS:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
        prompts.append(list(enc["input_ids"] if hasattr(enc, "keys") else enc))

    refs = {}
    for impl in ("sdpa", "eager"):
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation=impl)
        model = model.to("cuda").eval()
        with torch.inference_mode():
            refs[impl] = hf_greedy(model, prompts)
        if impl == "sdpa":
            hf = model
        else:
            del model
    div = [first_divergence(a, b) for a, b in zip(refs["sdpa"], refs["eager"], strict=True)]
    same = sum(d is None for d in div)
    print(f"DIVERGE HF sdpa vs HF eager fp32: {same}/{len(prompts)} identical, first divergence {div}")

    eng = LLMEngine.from_pretrained(MODEL, EngineConfig(device="cuda", dtype=torch.float32, num_blocks=512))
    ours = eng.generate(prompts, SamplingParams(max_tokens=NEW, ignore_eos=True))
    for i, (p, a, b) in enumerate(zip(prompts, ours, refs["sdpa"], strict=True)):
        d = first_divergence(a, b)
        if d is None:
            print(f"DIVERGE seq {i}: identical")
            continue
        ctx = torch.tensor([p + b[:d]], device="cuda")
        with torch.inference_mode():
            logits = hf(ctx).logits[0, -1].float()
        top2 = torch.topk(logits, 2)
        gap = (top2.values[0] - top2.values[1]).item()
        ours_tok, hf_tok = a[d], b[d]
        ours_gap = (logits[hf_tok] - logits[ours_tok]).item()
        rank = int((logits > logits[ours_tok]).sum())
        print(
            f"DIVERGE seq {i}: at token {d}, HF top-2 logit gap {gap:.5f}; "
            f"mini picked HF's rank-{rank} token, {ours_gap:.5f} below HF's choice"
        )


if __name__ == "__main__":
    main()
