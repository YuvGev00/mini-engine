"""Benchmark mini-engine against Hugging Face `generate` and vLLM on the same workloads.

Each system runs in its own process (vLLM reserves most GPU memory), on identical token-id
prompts built from a fixed seed, with greedy decoding:

  throughput  64 requests, 256-token prompts, 128 output tokens each (ignore EOS)
  prefix      64 requests sharing a 512-token prefix plus a 32-token suffix, 64 output tokens,
              with the prefix cache on and off. As in a server whose system prompt is already
              cached, 1 request runs first and the other 63 follow (timed together, every system)
  copy        "repeat this text" prompts, where n-gram speculative decoding should help:
              1 request and 8 requests, speculation off and on

    python benchmarks/bench.py --system mini --out results/mini.json
    python benchmarks/bench.py --system hf   --out results/hf.json
    python benchmarks/bench.py --system vllm --out results/vllm.json
    python benchmarks/bench.py --system mini --model tiny --device cpu   # local smoke test

Only output tokens count toward tokens/s. Wall time includes prefill and scheduling.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

MODEL = "Qwen/Qwen3-0.6B"
SEED = 0
COPY_TEXT = (
    "A paged KV cache splits the key and value tensors of every request into fixed-size blocks, "
    "so memory is allocated on demand instead of reserved for the longest possible output. "
    "Each request keeps a block table that maps its token positions to physical blocks, much "
    "like a page table maps virtual addresses to frames in an operating system. Because blocks "
    "are small and uniform, fragmentation stays low and many more requests fit on the same GPU. "
    "Blocks whose contents are identical, such as a shared system prompt, can be referenced by "
    "several requests at once, which is the basis of prefix caching. Continuous batching then "
    "admits new requests and retires finished ones at every step, so short requests never wait "
    "for long ones and the accelerator stays busy."
)


# ---------------------------------------------------------------------------- workloads


def random_ids(rng: random.Random, n: int, vocab: int) -> list[int]:
    return [rng.randrange(100, vocab) for _ in range(n)]


def workloads(tokenizer: Any, vocab: int, small: bool) -> dict[str, dict[str, Any]]:
    rng = random.Random(SEED)
    n_req, p_len, out = (8, 32, 16) if small else (64, 256, 128)
    throughput = [random_ids(rng, p_len, vocab) for _ in range(n_req)]
    prefix_len, suffix_len, prefix_out = (32, 8, 8) if small else (512, 32, 64)
    shared = random_ids(rng, prefix_len, vocab)
    prefix = [shared + random_ids(rng, suffix_len, vocab) for _ in range(n_req)]
    if tokenizer is None:  # tiny random model: any repetitive prompt exercises the code path
        copy_prompt = [5, 6, 7, 8, 9] * 8
        copy_out = 16
    else:
        messages = [
            {"role": "user", "content": f"Repeat the following text exactly, word for word:\n\n{COPY_TEXT}"}
        ]
        copy_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
        copy_out = len(tokenizer(COPY_TEXT).input_ids)
    return {
        "throughput": {"prompts": throughput, "max_tokens": out},
        "prefix": {"prompts": prefix, "max_tokens": prefix_out},
        "copy_1": {"prompts": [copy_prompt], "max_tokens": copy_out},
        "copy_8": {"prompts": [copy_prompt] * 8, "max_tokens": copy_out},
    }


# ------------------------------------------------------------------------------ systems


def staged(
    name: str, prompts: list[list[int]], run: Callable[[list[list[int]]], list[list[int]]]
) -> list[list[int]]:
    """The prefix workload sends one request first, so its prefix is computed (and cached,
    where the system has a cache) before the rest arrive. Other workloads run in one call."""
    if name == "prefix":
        return run(prompts[:1]) + run(prompts[1:])
    return run(prompts)


def timed(fn: Callable[[], list[list[int]]], sync: Callable[[], None]) -> tuple[list[list[int]], float]:
    sync()
    start = time.perf_counter()
    outs = fn()
    sync()
    return outs, time.perf_counter() - start


def run_mini(
    args: argparse.Namespace, wl: dict[str, dict[str, Any]], sync: Callable[[], None]
) -> list[dict[str, Any]]:
    from mini_engine import EngineConfig, LLMEngine, SamplingParams
    from mini_engine.speculative import SpeculativeConfig

    def make(prefix_cache: bool, spec: bool) -> Any:
        cfg = EngineConfig(
            device=args.device,
            dtype=args.torch_dtype,
            block_size=16,
            num_blocks=args.num_blocks,
            max_num_seqs=64,
            max_num_batched_tokens=8192,
            max_model_len=2048,
            enable_prefix_caching=prefix_cache,
            speculative=SpeculativeConfig(num_speculative_tokens=4) if spec else None,
        )
        if args.model == "tiny":
            from tiny import tiny_hf_model

            return LLMEngine.from_hf_model(tiny_hf_model(), cfg)
        return LLMEngine.from_pretrained(args.model, cfg)

    rows = []
    cases = [
        ("throughput", True, False),
        ("prefix", False, False),
        ("prefix", True, False),
        ("copy_1", True, False),
        ("copy_1", True, True),
        ("copy_8", True, False),
        ("copy_8", True, True),
    ]
    warm = make(True, False)
    warm.generate(wl["throughput"]["prompts"][:2], SamplingParams(max_tokens=4, ignore_eos=True))
    for name, prefix_cache, spec in cases:
        eng = make(prefix_cache, spec)
        params = SamplingParams(max_tokens=wl[name]["max_tokens"], ignore_eos=True)
        outs, wall = timed(
            lambda e=eng, p=params, n=name: staged(n, wl[n]["prompts"], lambda ps: e.generate(ps, p)), sync
        )
        s = eng.stats
        rows.append(
            row("mini", name, outs, wall, prefix_cache=prefix_cache, speculative=spec)
            | {
                "steps": s.steps,
                "cached_prompt_tokens": eng.scheduler.num_cached_tokens,
                "acceptance_rate": round(s.acceptance_rate, 3),
                "outputs_head": [o[:32] for o in outs[:4]],
            }
        )
        del eng
    return rows


def run_hf(
    args: argparse.Namespace, wl: dict[str, dict[str, Any]], sync: Callable[[], None]
) -> list[dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    if args.model == "tiny":
        from tiny import tiny_hf_model

        model = tiny_hf_model().to(args.device, args.torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=args.torch_dtype).to(args.device)
    model.eval()

    def generate(prompts: list[list[int]], max_new: int) -> list[list[int]]:
        width = max(len(p) for p in prompts)
        pad = 0
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=args.device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=args.device)
        with torch.inference_mode():
            out = model.generate(
                ids,
                attention_mask=mask,
                max_new_tokens=max_new,
                min_new_tokens=max_new,  # same lengths as ignore_eos elsewhere
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=pad,
            )
        return [o[width:].tolist() for o in out]

    generate(wl["throughput"]["prompts"][:2], 4)  # warm-up
    rows = []
    for name in ("throughput", "prefix", "copy_1", "copy_8"):
        outs, wall = timed(
            lambda n=name: staged(n, wl[n]["prompts"], lambda ps: generate(ps, wl[n]["max_tokens"])), sync
        )
        rows.append(row("hf", name, outs, wall) | {"outputs_head": [o[:32] for o in outs[:4]]})
    return rows


def run_vllm(
    args: argparse.Namespace, wl: dict[str, dict[str, Any]], sync: Callable[[], None]
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = []
    dtype = "float16" if args.torch_dtype == torch.float16 else "bfloat16"

    def bench(llm: Any, name: str, **tags: Any) -> None:
        sp = SamplingParams(temperature=0.0, max_tokens=wl[name]["max_tokens"], ignore_eos=True)

        def run(ps: list[list[int]]) -> list[list[int]]:
            reqs = [TokensPrompt(prompt_token_ids=p) for p in ps]
            return [list(r.outputs[0].token_ids) for r in llm.generate(reqs, sp, use_tqdm=False)]

        res, wall = timed(lambda: staged(name, wl[name]["prompts"], run), sync)
        outs = [list(o) for o in res]
        rows.append(row("vllm", name, outs, wall, **tags) | {"outputs_head": [o[:32] for o in outs[:4]]})

    for prefix_cache in (True, False):
        llm = LLM(
            model=args.model,
            dtype=dtype,
            enable_prefix_caching=prefix_cache,
            gpu_memory_utilization=0.8,
            seed=SEED,
        )
        llm.generate(
            [TokensPrompt(prompt_token_ids=wl["throughput"]["prompts"][0])], SamplingParams(max_tokens=4)
        )
        if prefix_cache:
            bench(llm, "throughput", prefix_cache=True)
            bench(llm, "copy_1", prefix_cache=True, speculative=False)
            bench(llm, "copy_8", prefix_cache=True, speculative=False)
        bench(llm, "prefix", prefix_cache=prefix_cache)
        del llm
        _free_cuda()
    try:  # vLLM's own n-gram speculation, if this version accepts the config
        spec = {
            "method": "ngram",
            "num_speculative_tokens": 4,
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 1,
        }
        llm = LLM(
            model=args.model, dtype=dtype, gpu_memory_utilization=0.8, seed=SEED, speculative_config=spec
        )
        bench(llm, "copy_1", prefix_cache=True, speculative=True)
        bench(llm, "copy_8", prefix_cache=True, speculative=True)
    except Exception as e:  # record and keep the other results
        rows.append({"system": "vllm", "workload": "copy_speculative", "error": repr(e)[:300]})
    return rows


def _free_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def row(system: str, workload: str, outs: list[list[int]], wall: float, **tags: Any) -> dict[str, Any]:
    n_out = sum(len(o) for o in outs)
    return {
        "system": system,
        "workload": workload,
        **tags,
        "requests": len(outs),
        "output_tokens": n_out,
        "wall_s": round(wall, 3),
        "output_tok_per_s": round(n_out / wall, 1),
    }


# --------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--system", choices=["mini", "hf", "vllm"], required=True)
    parser.add_argument("--model", default=MODEL, help="HF model id, or 'tiny' for a CPU smoke test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--num-blocks", type=int, default=2048, help="mini-engine KV blocks (16 tokens each)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cuda = args.device == "cuda"
    if args.dtype == "auto":
        bf16 = cuda and torch.cuda.is_bf16_supported()
        args.dtype = "bfloat16" if bf16 else ("float16" if cuda else "float32")
    args.torch_dtype = getattr(torch, args.dtype)

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    tokenizer = None
    vocab = 512
    if args.model != "tiny":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        vocab = len(tokenizer) - 200  # stay clear of special tokens at the end of the vocab
    wl = workloads(tokenizer, vocab, small=args.model == "tiny")
    runner = {"mini": run_mini, "hf": run_hf, "vllm": run_vllm}[args.system]
    rows = runner(args, wl, sync)

    env = {
        "gpu": torch.cuda.get_device_name() if cuda else platform.processor(),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "model": args.model,
    }
    if args.system == "vllm":
        import vllm

        env["vllm"] = vllm.__version__
    result = {"env": env, "rows": rows}
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    for r in rows:
        tag = ",".join(f"{k}={r[k]}" for k in ("prefix_cache", "speculative") if k in r)
        print(f"{r['system']:5s} {r['workload']:11s} {tag:35s} {r.get('output_tok_per_s', r.get('error'))}")


if __name__ == "__main__":
    main()
