"""The engine loop: schedule -> build tensors -> forward -> sample -> postprocess."""

from __future__ import annotations

import importlib.util
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
from mini_engine.speculative import SpeculativeConfig, propose_ngram, verify_greedy


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
    speculative: SpeculativeConfig | None = None  # n-gram speculative decoding (greedy requests only)
    attention_backend: str = (
        "auto"  # "sdpa", "grouped", "triton", or "auto" (triton/grouped on CUDA, else sdpa)
    )


@dataclass
class EngineStats:
    steps: int = 0
    prompt_tokens: int = 0  # cached hits are in `scheduler.num_cached_tokens`
    computed_tokens: int = 0  # tokens actually run through the model
    generated_tokens: int = 0
    step_batch_sizes: list[int] = field(default_factory=list)
    draft_tokens: int = 0  # speculative drafts sent to the model
    accepted_tokens: int = 0  # drafts that matched the model's own greedy choice

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.draft_tokens if self.draft_tokens else 0.0


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
        shape = (mcfg.num_kv_heads, config.num_blocks, config.block_size, mcfg.head_dim)
        self.kv_caches = [
            (
                torch.zeros(shape, dtype=config.dtype, device=self.device),
                torch.zeros(shape, dtype=config.dtype, device=self.device),
            )
            for _ in range(mcfg.num_layers)
        ]
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        backend = config.attention_backend
        if backend == "auto":
            if self.device.type != "cuda":
                backend = "sdpa"
            else:
                backend = "triton" if importlib.util.find_spec("triton") else "grouped"
        if backend not in ("sdpa", "grouped", "triton"):
            raise ValueError(f"unknown attention_backend {config.attention_backend!r}")
        self.attention_backend = backend
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
        drafts = self._append_drafts()
        batch = self.scheduler.schedule()
        if not batch:
            raise RuntimeError("scheduler produced an empty batch with work pending")
        scheduled = {item.seq.seq_id for item in batch}
        for seq_id, (seq, draft) in list(drafts.items()):
            if seq_id not in scheduled:  # preempted or out of budget: drop its drafts
                del seq.token_ids[-len(draft) :]
                del drafts[seq_id]

        input_ids, positions, meta, logits_index = self._prepare(batch, drafts)
        logits = self.model(input_ids, positions, self.kv_caches, meta, logits_index)

        plain = [item for item in batch if item.seq.seq_id not in drafts]
        rows = iter(range(logits.shape[0]))
        plain_rows: list[int] = []
        spec_rows: dict[int, list[int]] = {}
        for item in batch:
            if item.seq.seq_id in drafts:
                spec_rows[item.seq.seq_id] = [next(rows) for _ in range(item.num_tokens)]
            elif item.samples:
                plain_rows.append(next(rows))
        sampled = self._sample(logits[plain_rows], [item for item in plain if item.samples]).tolist()
        it = iter(sampled)
        next_tokens = [next(it) if item.samples else None for item in plain]
        self._update_stats(batch)
        finished = self.scheduler.postprocess(plain, next_tokens)

        for seq_id, row_ids in spec_rows.items():
            item = next(i for i in batch if i.seq.seq_id == seq_id)
            if self._finish_speculative(item, drafts[seq_id][1], logits[row_ids]):
                finished.append(item.seq)
        return finished

    # ---------------------------------------------------------------- private

    def _prepare(
        self, batch: list[ScheduledSeq], drafts: dict[int, tuple[Sequence, list[int]]] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionMetadata, torch.Tensor]:
        input_ids: list[int] = []
        positions: list[int] = []
        logits_index: list[int] = []
        for item in batch:
            start = item.seq.num_computed_tokens
            input_ids.extend(item.seq.token_ids[start : start + item.num_tokens])
            positions.extend(range(start, start + item.num_tokens))
            if drafts and item.seq.seq_id in drafts:  # verify: logits at every computed position
                logits_index.extend(range(len(input_ids) - item.num_tokens, len(input_ids)))
            elif item.samples:
                logits_index.append(len(input_ids) - 1)
        meta = AttentionMetadata.build(
            query_lens=[item.num_tokens for item in batch],
            context_lens=[item.seq.num_computed_tokens + item.num_tokens for item in batch],
            block_tables=[item.seq.block_table for item in batch],
            block_size=self.config.block_size,
            device=self.device,
            backend=self.attention_backend,
        )

        def as_tensor(xs: list[int]) -> torch.Tensor:
            return torch.tensor(xs, dtype=torch.int64, device=self.device)

        return as_tensor(input_ids), as_tensor(positions), meta, as_tensor(logits_index)

    def _append_drafts(self) -> dict[int, tuple[Sequence, list[int]]]:
        """Propose n-gram drafts for greedy decoding sequences and append them, so the
        scheduler treats them as tokens still to compute (like a short prefill chunk)."""
        spec = self.config.speculative
        drafts: dict[int, tuple[Sequence, list[int]]] = {}
        if spec is None:
            return drafts
        for seq in self.scheduler.running:
            if seq.num_uncomputed_tokens != 1 or seq.params.temperature != 0:
                continue
            # Never let the drafts push the sequence past its own token limits.
            room = min(
                seq.num_prompt_tokens + seq.params.max_tokens - len(seq) - 1,
                self.config.max_model_len - len(seq) - 1,
            )
            k = min(spec.num_speculative_tokens, room)
            if k < 1:
                continue
            draft = propose_ngram(seq.token_ids, k, spec.ngram_max, spec.ngram_min)
            if draft:
                seq.token_ids.extend(draft)
                drafts[seq.seq_id] = (seq, draft)
        return drafts

    def _finish_speculative(self, item: ScheduledSeq, draft: list[int], logits: torch.Tensor) -> bool:
        """Accept the drafts that match the model's greedy choices. Row j of `logits` is
        the prediction after computed position j; position 0 is the real last token."""
        seq = item.seq
        del seq.token_ids[-len(draft) :]
        num_real = len(seq)
        new_tokens, accepted = verify_greedy(draft, logits.argmax(dim=-1).tolist())
        self.stats.draft_tokens += min(len(draft), item.num_tokens)
        self.stats.accepted_tokens += accepted
        self.stats.generated_tokens += len(new_tokens) - 1  # _update_stats counted one
        # K/V is valid for the real last token plus the accepted drafts that were computed.
        num_valid = num_real - 1 + min(accepted + 1, item.num_tokens)
        return self.scheduler.accept_speculative(seq, num_valid, new_tokens)

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
            s.generated_tokens += int(item.samples)  # speculative steps add the rest later


def _as_id_list(eos: Any) -> list[int]:
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return [int(e) for e in eos]
