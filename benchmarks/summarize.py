"""Turn benchmarks/results/*.json into the README's Markdown tables.

python benchmarks/summarize.py benchmarks/results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load(folder: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    envs: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text())
        envs[path.stem] = data["env"]
        rows.extend(data["rows"])
    return envs, rows


def find(rows: list[dict[str, Any]], system: str, workload: str, **tags: Any) -> dict[str, Any] | None:
    for r in rows:
        if (
            r["system"] == system
            and r["workload"] == workload
            and all(r.get(k) == v for k, v in tags.items())
        ):
            return r
    return None


def tps(r: dict[str, Any] | None) -> str:
    return "-" if not r or "output_tok_per_s" not in r else f"{r['output_tok_per_s']:,.0f}"


def ratio(a: dict[str, Any] | None, b: dict[str, Any] | None) -> str:
    if not a or not b or "output_tok_per_s" not in a or "output_tok_per_s" not in b:
        return "-"
    return f"{a['output_tok_per_s'] / b['output_tok_per_s']:.2f}x"


def main(folder: Path) -> None:
    envs, rows = load(folder)
    env = next(iter(envs.values()))
    print(f"GPU: {env['gpu']} · model: {env['model']} · dtype: {env['dtype']} · torch {env['torch']}")
    if "vllm" in envs:
        print(f"vLLM {envs['vllm'].get('vllm')}")
    print()

    print("| Workload | HF generate | mini-engine | vLLM | mini / HF | mini / vLLM |")
    print("|---|---|---|---|---|---|")
    for name, label in (
        ("throughput", "64 x (256 in, 128 out)"),
        ("prefix", "64 x (512 shared + 32 in, 64 out)"),
    ):
        hf = find(rows, "hf", name)
        mini = find(rows, "mini", name, prefix_cache=True)
        vl = find(rows, "vllm", name, prefix_cache=True)
        print(f"| {label} | {tps(hf)} | {tps(mini)} | {tps(vl)} | {ratio(mini, hf)} | {ratio(mini, vl)} |")
    print("\nOutput tokens per second (higher is better), greedy, same token-id prompts.\n")

    off, on = (
        find(rows, "mini", "prefix", prefix_cache=False),
        find(rows, "mini", "prefix", prefix_cache=True),
    )
    voff, von = (
        find(rows, "vllm", "prefix", prefix_cache=False),
        find(rows, "vllm", "prefix", prefix_cache=True),
    )
    print("| Prefix cache | off | on | speed-up | prompt tokens served from cache |")
    print("|---|---|---|---|---|")
    cached = on.get("cached_prompt_tokens", "-") if on else "-"
    print(f"| mini-engine | {tps(off)} | {tps(on)} | {ratio(on, off)} | {cached} |")
    print(f"| vLLM | {tps(voff)} | {tps(von)} | {ratio(von, voff)} | - |")
    print()

    print("| Speculative (n-gram, k=4) on a copy task | off | on | speed-up | acceptance |")
    print("|---|---|---|---|---|")
    for name, label in (("copy_1", "1 request"), ("copy_8", "8 requests")):
        m_off = find(rows, "mini", name, speculative=False)
        m_on = find(rows, "mini", name, speculative=True)
        acc = m_on.get("acceptance_rate", "-") if m_on else "-"
        print(f"| mini-engine, {label} | {tps(m_off)} | {tps(m_on)} | {ratio(m_on, m_off)} | {acc} |")
        v_off, v_on = find(rows, "vllm", name, speculative=False), find(rows, "vllm", name, speculative=True)
        print(f"| vLLM, {label} | {tps(v_off)} | {tps(v_on)} | {ratio(v_on, v_off)} | - |")
    print()

    print("Correctness (natural EOS stopping) is checked by gpu_parity.py: see results/gpu_parity.md.")
    for r in rows:
        if "error" in r:
            print(f"\nnote: {r['system']} {r['workload']}: {r['error']}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results"))
