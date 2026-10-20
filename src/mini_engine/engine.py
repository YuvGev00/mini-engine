"""The engine loop: schedule -> build tensors -> forward -> sample -> postprocess."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from mini_engine.attention import AttentionMetadata
from mini_engine.block_manager import BlockManager
from mini_engine.model import ModelConfig, Qwen3ForCausalLM
from mini_engine.scheduler import ScheduledSeq, Scheduler, SchedulerConfig
from mini_engine.sequence import SamplingParams, Sequence


@dataclass(frozen=True)
class EngineConfig:
    block_size: int = 16
    num_blocks: int = 256
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096
    enable_prefix_caching: bool = True
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 0


@dataclass
class EngineStats:
    steps: int = 0
    prompt_tokens: int = 0  # cached hits are in `scheduler.num_cached_tokens`
    computed_tokens: int = 0  # tokens actually run through the model
    generated_tokens: int = 0
    step_batch_sizes: list[int] = field(default_factory=list)


def kv_cache_bytes_per_token(cfg: ModelConfig, dtype: torch.dtype) -> int:
    elem = torch.empty((), dtype=dtype).element_size()
    return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * elem


class LLMEngine:
    def __init__(self, model: Qwen3ForCausalLM, config: EngineConfig, eos_token_ids: Iterable[int] = ()):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(device=self.device, dtype=config.dtype).eval()
        mcfg = model.config
        self.block_manager = BlockManager(config.num_blocks, config.block_size, config.enable_prefix_caching)
        self.scheduler = Scheduler(
            SchedulerConfig(config.max_num_seqs, config.max_num_batched_tokens, config.max_model_len),
            self.block_manager,
            set(eos_token_ids),
        )
        shape = (config.num_blocks, config.block_size, mcfg.num_kv_heads, mcfg.head_dim)
        self.kv_caches = [
            (
                torch.zeros(shape, dtype=config.dtype, device=self.device),
                torch.zeros(shape, dtype=config.dtype, device=self.device),
            )
            for _ in range(mcfg.num_layers)
        ]
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self.stats = EngineStats()

    # ----------------------------------------------------------------- loading

    @classmethod
    def from_pretrained(cls, model_id_or_path: str, config: EngineConfig | None = None) -> LLMEngine:
        """Load a Qwen3 checkpoint from a local directory or the HF Hub (safetensors)."""
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, GenerationConfig

        path = Path(model_id_or_path)
        if not path.is_dir():
            path = Path(
                snapshot_download(model_id_or_path, allow_patterns=["*.json", "*.safetensors", "*.txt"])
            )
        hf_config = AutoConfig.from_pretrained(path)
        model = Qwen3ForCausalLM(ModelConfig.from_hf(hf_config))
        state: dict[str, torch.Tensor] = {}
        for f in sorted(path.glob("*.safetensors")):
            state.update(load_file(f))
        model.load_hf_state_dict(state)
        try:
            eos: Any = GenerationConfig.from_pretrained(path).eos_token_id
        except OSError:
            eos = hf_config.eos_token_id
        return cls(model, config or EngineConfig(), _as_id_list(eos))

    @classmethod
    def from_hf_model(cls, hf_model: Any, config: EngineConfig | None = None) -> LLMEngine:
        """Build from an in-memory `transformers` model (used by the tests)."""
        model = Qwen3ForCausalLM(ModelConfig.from_hf(hf_model.config))
        model.load_hf_state_dict(hf_model.state_dict())
        return cls(model, config or EngineConfig(), _as_id_list(hf_model.generation_config.eos_token_id))

    # -------------------------------------------------------------------- API

    def add_request(self, prompt_token_ids: list[int], params: SamplingParams | None = None) -> Sequence:
        seq = Sequence(prompt_token_ids, params or SamplingParams())
        self.scheduler.add(seq)
        self.stats.prompt_tokens += len(seq)
        return seq

    def generate(self, prompts: list[list[int]], params: SamplingParams | None = None) -> list[list[int]]:
        """Run all prompts to completion as one continuously batched workload."""
        seqs = [self.add_request(p, params) for p in prompts]
        while self.scheduler.has_unfinished():
            self.step()
        return [s.output_token_ids for s in seqs]

    @torch.inference_mode()
    def step(self) -> list[Sequence]:
        batch = self.scheduler.schedule()
        if not batch:
            raise RuntimeError("scheduler produced an empty batch with work pending")
        input_ids, positions, meta, logits_index = self._prepare(batch)
        logits = self.model(input_ids, positions, self.kv_caches, meta, logits_index)
        sampled = self._sample(logits, [item for item in batch if item.samples]).tolist()

        next_tokens: list[int | None] = []
        it = iter(sampled)
        for item in batch:
            next_tokens.append(next(it) if item.samples else None)
        self._update_stats(batch)
        return self.scheduler.postprocess(batch, next_tokens)

    # ---------------------------------------------------------------- private

    def _prepare(
        self, batch: list[ScheduledSeq]
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionMetadata, torch.Tensor]:
        input_ids: list[int] = []
        positions: list[int] = []
        logits_index: list[int] = []
        for item in batch:
            start = item.seq.num_computed_tokens
            input_ids.extend(item.seq.token_ids[start : start + item.num_tokens])
            positions.extend(range(start, start + item.num_tokens))
            if item.samples:
                logits_index.append(len(input_ids) - 1)
        meta = AttentionMetadata.build(
            query_lens=[item.num_tokens for item in batch],
            context_lens=[item.seq.num_computed_tokens + item.num_tokens for item in batch],
            block_tables=[item.seq.block_table for item in batch],
            block_size=self.config.block_size,
            device=self.device,
        )

        def as_tensor(xs: list[int]) -> torch.Tensor:
            return torch.tensor(xs, dtype=torch.int64, device=self.device)

        return as_tensor(input_ids), as_tensor(positions), meta, as_tensor(logits_index)

    def _sample(self, logits: torch.Tensor, items: list[ScheduledSeq]) -> torch.Tensor:
        if not items:
            return logits.new_empty(0, dtype=torch.int64)
        temps = torch.tensor([i.seq.params.temperature for i in items], device=self.device)
        greedy = logits.argmax(dim=-1)
        if bool((temps == 0).all()):
            return greedy
        probs = torch.softmax(logits.float() / temps.clamp(min=1e-5)[:, None], dim=-1)
        sampled = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
        return torch.where(temps == 0, greedy, sampled)

    def _update_stats(self, batch: list[ScheduledSeq]) -> None:
        s = self.stats
        s.steps += 1
        s.step_batch_sizes.append(len(batch))
        for item in batch:
            s.computed_tokens += item.num_tokens
            s.generated_tokens += int(item.samples)


def _as_id_list(eos: Any) -> list[int]:
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return [int(e) for e in eos]
