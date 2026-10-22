"""Where does a decode step spend its time? Phase timers plus a torch.profiler op table.

Runs the throughput workload (64 requests, 256-token prompts) until every request is
decoding, then times decode steps phase by phase (schedule / prepare / forward / sample +
postprocess, with a CUDA sync around each) and profiles a few of them with torch.profiler.

    python benchmarks/profile_engine.py                       # Qwen3-0.6B on CUDA, bf16
    python benchmarks/profile_engine.py --model tiny --device cpu   # smoke test
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from typing import Any

import torch

from mini_engine import EngineConfig, LLMEngine, SamplingParams


def build_engine(args: argparse.Namespace) -> LLMEngine:
    cfg = EngineConfig(device=args.device, dtype=args.dtype, num_blocks=args.num_blocks, max_num_seqs=64)
    if args.model == "tiny":
        from tiny import tiny_hf_model

        return LLMEngine.from_hf_model(tiny_hf_model(), cfg)
    return LLMEngine.from_pretrained(args.model, cfg)


def timed_step(eng: LLMEngine, sync: Any, totals: dict[str, float]) -> None:
    """One engine step with the same logic as LLMEngine.step (no speculation), timed by phase."""

    def mark(name: str, start: float) -> float:
        sync()
        now = time.perf_counter()
        totals[name] += now - start
        return now

    t = time.perf_counter()
    with torch.inference_mode():
        batch = eng.scheduler.schedule()
        t = mark("schedule", t)
        graphs = eng.decode_graphs
        if graphs and len(batch) <= graphs.max_batch and all(i.num_tokens == 1 and i.samples for i in batch):
            logits = graphs.run(batch)
            t = mark("model forward (CUDA graph, incl. H2D)", t)
        else:
            input_ids, positions, meta, logits_index = eng._prepare(batch)
            t = mark("prepare (metadata + H2D)", t)
            logits = eng.model(input_ids, positions, eng.kv_caches, meta, logits_index)
            t = mark("model forward", t)
        sampled = eng._sample(logits, [i for i in batch if i.samples]).tolist()
        it = iter(sampled)
        eng.scheduler.postprocess(batch, [next(it) if i.samples else None for i in batch])
        mark("sample + postprocess", t)


def main() -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    cuda = args.device == "cuda"
    args.dtype = torch.bfloat16 if cuda else torch.float32
    vocab = 512 if args.model == "tiny" else 150_000

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    eng = build_engine(args)
    rng = random.Random(0)
    n, p_len = (8, 32) if args.model == "tiny" else (64, 256)
    for _ in range(n):
        prompt = [rng.randrange(100, vocab) for _ in range(p_len)]
        max_tokens = 200  # enough for warm-up + timed + profiled decode steps
        eng.add_request(prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=True))
    while any(s.num_uncomputed_tokens > 1 for s in eng.scheduler.running) or eng.scheduler.waiting:
        eng.step()  # prefill
    for _ in range(3):
        eng.step()  # warm-up decode

    totals: dict[str, float] = defaultdict(float)
    for _ in range(args.steps):
        timed_step(eng, sync, totals)
    wall = sum(totals.values())
    batch = len(eng.scheduler.running)
    per_step = 1000 * wall / args.steps
    print(f"PROFILE decode, batch {batch}: {per_step:.1f} ms/step, {batch * args.steps / wall:.0f} tok/s")
    for name, sec in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"PROFILE   {name:28s} {1000 * sec / args.steps:7.2f} ms/step  {100 * sec / wall:5.1f}%")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as prof:
        for _ in range(5):
            eng.step()
        sync()
    sort = "cuda_time_total" if cuda else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort, row_limit=18, max_name_column_width=48)
    for line in table.splitlines():
        print("OPS " + line)
    if cuda:
        gpu = torch.autograd.DeviceType.CUDA
        n_kernels = sum(e.count for e in prof.key_averages() if e.device_type == gpu)
        print(f"PROFILE CUDA kernel launches per step: {n_kernels / 5:.0f}")


if __name__ == "__main__":
    main()
