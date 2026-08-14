"""LungReco model with CoST, causal xLSTM memory and masked reasoning."""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from timm.models.registry import register_model

from model.visual_backbone import VisionTransformer, _cfg, visual_backbone


PHASE_NAMES = (
    "Fissure Dissection",
    "Vein Dissection",
    "Artery Dissection",
    "Bronchus Dissection",
    "Lymph Node Dissection",
    "Lung Segment Resection",
    "Other Operations",
)


class ConcurrentSpatialTemporalEncoding(nn.Module):
    """Concurrent temporal attention and cross-scale spatial aggregation.

    The native 4/8/16 target-frame hierarchy is retained. Coarser
    temporal ranges use coarser spatial grids, as specified in Sec. 4.2.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        temporal_ranges: tuple[int, ...] = (4, 8, 16),
        spatial_grids: tuple[int, ...] = (14, 7, 4),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(temporal_ranges) != len(spatial_grids):
            raise ValueError("temporal_ranges and spatial_grids must have equal length")
        self.temporal_ranges = temporal_ranges
        self.spatial_grids = spatial_grids
        self.temporal_attn = nn.ModuleList(
            [nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
             for _ in temporal_ranges]
        )
        self.temporal_norm = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in temporal_ranges]
        )
        self.spatial_attn = nn.MultiheadAttention(
            dim, num_heads, dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)

    @staticmethod
    def _spatial_pool(tokens: torch.Tensor, grid: int) -> torch.Tensor:
        # tokens: B,T,K,C; the patch grid is square.
        batch, time, patches, channels = tokens.shape
        side = int(patches**0.5)
        if side * side != patches:
            raise ValueError(f"Expected a square patch grid, got K={patches}")
        images = rearrange(tokens, "b t (h w) c -> (b t) c h w", h=side, w=side)
        images = F.adaptive_avg_pool2d(images, (grid, grid))
        return rearrange(images, "(b t) c h w -> b t (h w) c", b=batch, t=time)

    def encode_scales(self, tokens: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        batch, full_time, _, channels = tokens.shape
        levels: list[torch.Tensor] = []
        fine_sequence = None
        for level, (time_range, grid) in enumerate(
            zip(self.temporal_ranges, self.spatial_grids)
        ):
            length = min(time_range, full_time)
            pooled = self._spatial_pool(tokens[:, -length:], grid)
            patches = pooled.shape[2]
            temporal = rearrange(pooled, "b t k c -> (b k) t c")
            normalized = self.temporal_norm[level](temporal)
            temporal, _ = self.temporal_attn[level](normalized, normalized, normalized)
            temporal = temporal + rearrange(pooled, "b t k c -> (b k) t c")
            temporal = rearrange(
                temporal, "(b k) t c -> b t k c", b=batch, k=patches
            )
            levels.append(temporal)
            if level == len(self.temporal_ranges) - 1:
                fine_sequence = temporal.mean(dim=2)
        assert fine_sequence is not None
        return levels, fine_sequence

    def fuse_target(
        self,
        levels: Iterable[torch.Tensor],
        text_token: torch.Tensor | None = None,
        text_scale: float = 1.0,
    ) -> torch.Tensor:
        levels = list(levels)
        # Query with the finest-spatial target tokens.  The text token is optional
        # so the history/text-fusion ablation can use purely visual CoST context.
        query = levels[0][:, -1]
        context_parts = [level[:, -1] for level in levels]
        if text_token is not None:
            context_parts.append(text_token[:, None])
        context = torch.cat(context_parts, dim=1)
        query_norm = self.spatial_norm(query)
        context_norm = self.spatial_norm(context)
        # LayerNorm removes any scale applied before CoST. Scale the normalized
        # history token here so it remains a bounded auxiliary cue.
        if text_token is not None:
            context_norm = torch.cat(
                (context_norm[:, :-1], context_norm[:, -1:] * float(text_scale)),
                dim=1,
            )
        fused, _ = self.spatial_attn(query_norm, context_norm, context_norm)
        return self.output_norm((query + fused).mean(dim=1))

    def forward(
        self, tokens: torch.Tensor, text_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        levels, sequence = self.encode_scales(tokens)
        return self.fuse_target(levels, text_token), sequence


class FrozenLlavaTextEncoder(nn.Module):
    """Frozen LLaVA/Vicuna-7B sentence encoder used by MCR.

    `model_path=none` enables a deterministic embedding-only smoke-test mode;
    reproduction runs must provide the LLaVA v1.5 Vicuna-7B checkpoint.
    """

    def __init__(
        self,
        model_path: str,
        output_dim: int = 768,
        prompt_cache_size: int = 0,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.output_dim = output_dim
        self.tokenizer = None
        self.language_model = None
        self.max_text_length = 512
        self.prompt_cache_size = max(0, int(prompt_cache_size))
        self._prompt_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.prompt_cache_hits = 0
        self.prompt_cache_misses = 0
        if model_path.lower() in {"", "none", "dummy"}:
            self.smoke_embedding = nn.Embedding(8192, output_dim)
            self.projection = nn.Identity()
            return

        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
        from transformers.utils.hub import cached_file

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Original LLaVA-1.5 checkpoints store the Vicuna language weights as
        # `model.*` plus a small `model.mm_projector.*`.  Loading the Llama
        # component directly avoids allocating the unused CLIP vision tower.
        config_path = cached_file(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config_values = json.load(handle)
        config_values.pop("model_type", None)
        config_values.pop("architectures", None)
        text_config = LlamaConfig(**config_values)
        self.max_text_length = min(text_config.max_position_embeddings, 4096)
        # If an unusually fragmented case exceeds Vicuna's context, retain
        # the most recent phases and the final prediction question.
        self.tokenizer.truncation_side = "left"
        llama = LlamaForCausalLM.from_pretrained(
            model_path,
            config=text_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        # Keep the frozen 7B encoder outside the registered module tree.  It is
        # reloaded from `model_path`, so duplicating 13 GB into every training
        # checkpoint would be wasteful and can exhaust the experiment disk.
        object.__setattr__(self, "language_model", llama.model)
        hidden_size = text_config.hidden_size
        for parameter in self.language_model.parameters():
            parameter.requires_grad_(False)
        self.language_model.eval()
        self.projection = nn.Linear(hidden_size, output_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.language_model is not None:
            self.language_model.eval()
        return self

    def _encode_frozen_prompts(
        self, prompts: list[str], device: torch.device
    ) -> torch.Tensor:
        """Run only the frozen Vicuna body and return pooled hidden states."""
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = self.language_model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                return_dict=True,
            )
            hidden = output.last_hidden_state
            last_index = encoded.attention_mask.sum(dim=1) - 1
            return hidden[
                torch.arange(hidden.shape[0], device=device), last_index
            ].detach()

    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        if self.language_model is None:
            # Stable token hashing is deliberately limited to smoke tests.
            ids = torch.tensor(
                [[sum(bytearray(prompt.encode("utf-8"))) % 8192] for prompt in prompts],
                device=device,
            )
            return self.smoke_embedding(ids).squeeze(1)
        parameter = next(self.language_model.parameters())
        if parameter.device != device:
            self.language_model.to(device)
        if self.prompt_cache_size == 0:
            pooled = self._encode_frozen_prompts(prompts, device)
            return self.projection(pooled.to(self.projection.weight.dtype))

        pooled_rows: list[torch.Tensor | None] = [None] * len(prompts)
        missing: OrderedDict[str, list[int]] = OrderedDict()
        for index, prompt in enumerate(prompts):
            if prompt in self._prompt_cache:
                pooled = self._prompt_cache.pop(prompt)
                if pooled.device != device:
                    pooled = pooled.to(device)
                self._prompt_cache[prompt] = pooled
                pooled_rows[index] = pooled
                self.prompt_cache_hits += 1
            else:
                missing.setdefault(prompt, []).append(index)
                self.prompt_cache_misses += 1

        if missing:
            missing_prompts = list(missing)
            encoded_missing = self._encode_frozen_prompts(missing_prompts, device)
            for prompt, pooled in zip(missing_prompts, encoded_missing):
                pooled = pooled.detach()
                self._prompt_cache[prompt] = pooled
                for index in missing[prompt]:
                    pooled_rows[index] = pooled
            while len(self._prompt_cache) > self.prompt_cache_size:
                self._prompt_cache.popitem(last=False)

        if any(row is None for row in pooled_rows):
            raise RuntimeError("Prompt cache failed to populate every batch row")
        pooled = torch.stack([row for row in pooled_rows if row is not None])
        pooled = pooled.to(self.projection.weight.dtype)
        return self.projection(pooled)


class MaskedCausalReasoning(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_xlstm_blocks: int = 4,
        context_length: int = 16,
        mask_ratio: float = 0.4,
        gaussian_sigma: float = 3.0,
        smooth_feature_memory: bool = False,
        phase_history_mask_probability: float = 0.0,
        phase_history_replace_ratio: float = 0.0,
        llava_model_path: str = "liuhaotian/llava-v1.5-7b",
        llava_prompt_cache_size: int = 0,
    ) -> None:
        super().__init__()
        from xlstm import (
            mLSTMBlockConfig,
            mLSTMLayerConfig,
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
        )

        config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(proj_factor=2.0, num_heads=4)
            ),
            context_length=context_length,
            num_blocks=num_xlstm_blocks,
            embedding_dim=dim,
        )
        self.memory_model = xLSTMBlockStack(config)
        self.mask_ratio = mask_ratio
        self.gaussian_sigma = float(gaussian_sigma)
        self.smooth_feature_memory = bool(smooth_feature_memory)
        self.phase_history_mask_probability = float(
            phase_history_mask_probability
        )
        self.phase_history_replace_ratio = float(phase_history_replace_ratio)
        if self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        if not 0.0 <= self.phase_history_mask_probability <= 1.0:
            raise ValueError("phase_history_mask_probability must be in [0, 1]")
        if not 0.0 <= self.phase_history_replace_ratio <= 1.0:
            raise ValueError("phase_history_replace_ratio must be in [0, 1]")
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.text_encoder = FrozenLlavaTextEncoder(
            llava_model_path, dim, prompt_cache_size=llava_prompt_cache_size
        )

    @property
    def xlstm_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.memory_model.parameters())

    def random_frame_mask(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = memory.shape
        mask = torch.zeros(batch, length, dtype=torch.bool, device=memory.device)
        if self.training and self.mask_ratio > 0:
            mask = torch.rand(batch, length, device=memory.device) < self.mask_ratio
        masked = torch.where(mask[:, :, None], self.mask_token, memory)
        return masked, mask

    def gaussian_smooth_memory(self, memory: torch.Tensor) -> torch.Tensor:
        """Gaussian smoothing over the causal xLSTM feature-memory sequence."""
        if not self.smooth_feature_memory or memory.shape[1] <= 1:
            return memory
        length = memory.shape[1]
        radius = min(length - 1, max(1, int(3.0 * self.gaussian_sigma + 0.5)))
        offsets = torch.arange(
            -radius, radius + 1, device=memory.device, dtype=torch.float32
        )
        kernel = torch.exp(-0.5 * (offsets / self.gaussian_sigma).square())
        kernel = kernel.view(1, 1, -1)
        values = memory.float().transpose(1, 2).reshape(-1, 1, length)
        numerator = F.conv1d(values, kernel, padding=radius)
        denominator = F.conv1d(
            torch.ones(1, 1, length, device=memory.device),
            kernel,
            padding=radius,
        )
        smoothed = numerator / denominator.clamp_min(1e-12)
        return smoothed.reshape(memory.shape[0], memory.shape[2], length).transpose(1, 2).to(memory.dtype)

    def mask_phase_history(self, history: list) -> list:
        """Paper RFM: independently mask causal frame-history tokens."""
        history = list(history)
        if not self.training or not history:
            return history
        for index in range(len(history)):
            if torch.rand(()) >= self.phase_history_mask_probability:
                continue
            entry = history[index]
            time_seconds = entry[1] if isinstance(entry, (tuple, list)) else None
            phase_id = entry[0] if isinstance(entry, (tuple, list)) else entry
            if phase_id is not None and torch.rand(()) < self.phase_history_replace_ratio:
                phase_id = int(phase_id)
                # Histories exclude Other Operations. Draw uniformly from the
                # other five non-OO classes so replacement is always wrong.
                offset = int(torch.randint(1, len(PHASE_NAMES) - 1, ()).item())
                history[index] = ((phase_id + offset) % (len(PHASE_NAMES) - 1), time_seconds)
            else:
                history[index] = (None, time_seconds)
        return history

    @staticmethod
    def _prompt(
        recent_phases: Iterable = (), current_time_seconds: int | None = None
    ) -> str:
        entries = []
        for entry in recent_phases:
            if isinstance(entry, (tuple, list)):
                phase_id, time_seconds = entry
            else:
                phase_id, time_seconds = entry, None
            if phase_id is not None:
                phase_id = int(phase_id)
                if phase_id < 0 or phase_id >= len(PHASE_NAMES):
                    raise ValueError("Recent phase history contains an invalid phase id")
                phase_name = PHASE_NAMES[phase_id]
            else:
                phase_name = "[MASKED PHASE]"
            entries.append((phase_name, time_seconds))
        if not entries:
            question_time = (
                f" at time {int(current_time_seconds)} s"
                if current_time_seconds is not None
                else ""
            )
            return (
                "We are performing a VATS surgical workflow. Based on the causal "
                f"visual memory, what's the most probable next phase{question_time}?"
            )

        # Merge consecutive one-second frame tokens into the phase intervals
        # used by the paper prompt template. Masked runs remain explicit.
        runs = []
        for phase_name, start_time in entries:
            start_time = None if start_time is None else int(start_time)
            if (
                runs
                and runs[-1][0] == phase_name
                and start_time is not None
                and runs[-1][2] is not None
                and start_time == runs[-1][2] + 1
            ):
                runs[-1][2] = start_time
            else:
                runs.append([phase_name, start_time, start_time])

        intervals = []
        for phase_name, start_time, run_end_time in runs:
            if start_time is None or int(start_time) < 0:
                intervals.append(f"We are performing {phase_name}.")
                continue
            start_time = int(start_time)
            end_time = start_time if run_end_time is None else int(run_end_time)
            intervals.append(
                f"We are performing {phase_name} from time {start_time} s "
                f"to time {end_time} s."
            )

        question_time = (
            f" at time {int(current_time_seconds)} s"
            if current_time_seconds is not None
            else ""
        )
        return " ".join(intervals) + f" What's the most probable next phase{question_time}?"

    def forward(
        self,
        visual_sequence: torch.Tensor,
        recent_phase_histories: list[list] | None = None,
        current_time_seconds: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        memory = self.memory_model(visual_sequence)
        memory = self.gaussian_smooth_memory(memory)
        masked_memory, mask = self.random_frame_mask(memory)
        reasoned = self.memory_model(masked_memory)
        if recent_phase_histories is None:
            recent_phase_histories = [[] for _ in range(visual_sequence.shape[0])]
        if len(recent_phase_histories) != visual_sequence.shape[0]:
            raise ValueError("Phase histories must match the visual batch size")
        if current_time_seconds is None:
            current_time_seconds = [None] * visual_sequence.shape[0]
        if len(current_time_seconds) != visual_sequence.shape[0]:
            raise ValueError("Current times must match the visual batch size")
        masked_phase_histories = [
            self.mask_phase_history(history) for history in recent_phase_histories
        ]
        prompts = [
            self._prompt(history, time_seconds)
            for history, time_seconds in zip(
                masked_phase_histories, current_time_seconds
            )
        ]
        text_token = self.text_encoder(prompts, visual_sequence.device)
        return text_token, {
            "memory": reasoned,
            "mask": mask,
            "masked_phase_histories": masked_phase_histories,
            "prompts": prompts,
        }

class LungReco(VisionTransformer):
    def __init__(
        self,
        *args,
        llava_model_path: str = "liuhaotian/llava-v1.5-7b",
        llava_prompt_cache_size: int = 0,
        mcr_mask_ratio: float = 0.4,
        gaussian_sigma: float = 1.0,
        gaussian_history_length: int = 64,
        transition_history_length: int = 3,
        history_text_scale: float = 0.1,
        feature_memory_gaussian_smoothing: bool = False,
        smooth_output_probabilities: bool = True,
        phase_history_mask_probability: float = 0.0,
        phase_history_replace_ratio: float = 0.0,
        use_text_cost_fusion: bool = False,
        include_text_final_fusion: bool = False,
        cost_temporal_ranges: tuple[int, ...] = (4, 8, 16),
        cost_spatial_grids: tuple[int, ...] = (14, 7, 4),
        use_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        dim = self.embed_dim
        self.cost = ConcurrentSpatialTemporalEncoding(
            dim=dim,
            num_heads=12,
            temporal_ranges=cost_temporal_ranges,
            spatial_grids=cost_spatial_grids,
        )
        self.mcr = MaskedCausalReasoning(
            dim=dim,
            num_xlstm_blocks=4,
            context_length=self.time_embed.shape[1],
            mask_ratio=mcr_mask_ratio,
            gaussian_sigma=gaussian_sigma,
            smooth_feature_memory=feature_memory_gaussian_smoothing,
            phase_history_mask_probability=phase_history_mask_probability,
            phase_history_replace_ratio=phase_history_replace_ratio,
            llava_model_path=llava_model_path,
            llava_prompt_cache_size=llava_prompt_cache_size,
        )
        self.gaussian_sigma = float(gaussian_sigma)
        self.gaussian_history_length = int(gaussian_history_length)
        self.transition_history_length = int(transition_history_length)
        self.history_text_scale = float(history_text_scale)
        self.feature_memory_gaussian_smoothing = bool(
            feature_memory_gaussian_smoothing
        )
        self.smooth_output_probabilities = bool(smooth_output_probabilities)
        self.use_text_cost_fusion = bool(use_text_cost_fusion)
        self.include_text_final_fusion = bool(include_text_final_fusion)
        if self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        if self.gaussian_history_length <= 0:
            raise ValueError("gaussian_history_length must be positive")
        if self.transition_history_length <= 0:
            raise ValueError("transition_history_length must be positive")
        if not 0.0 <= self.history_text_scale <= 1.0:
            raise ValueError("history_text_scale must be in [0, 1]")
        self.use_checkpoint = use_checkpoint
        fusion_inputs = 3 if self.include_text_final_fusion else 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * fusion_inputs),
            nn.Linear(dim * fusion_inputs, dim),
            nn.GELU(),
            nn.Dropout(0.5),
        )
        self.head = nn.Linear(dim, self.num_classes)
        self._stream_memory: dict[str, dict] = defaultdict(dict)

    def no_weight_decay(self):
        return super().no_weight_decay() | {"mcr.mask_token"}

    def forward_visual_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve the final token sequence produced by the visual encoder.
        x = self.patch_embed(x)
        batch, time, patches, _ = x.shape
        x = rearrange(x, "b t k c -> (b t) k c")
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = self.pos_drop(torch.cat((cls_tokens, x), dim=1) + self.pos_embed)
        cls_tokens = x[:batch, 0].unsqueeze(1)
        x = rearrange(x[:, 1:], "(b t) k c -> (b k) t c", b=batch)
        x = self.time_drop(x + self.time_embed)
        x = rearrange(x, "(b k) t c -> b (k t) c", b=batch)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(
                    lambda values, layer=block: layer(values, batch, time, patches),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, batch, time, patches)
        x = self.norm(x)[:, 1:]
        return rearrange(x, "b (k t) c -> b t k c", t=time, k=patches)

    def forward_features(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        recent_phase_histories: list[list] | None = None,
        current_time_seconds: list[int] | None = None,
    ):
        visual_tokens = self.forward_visual_tokens(x)
        levels, visual_sequence = self.cost.encode_scales(visual_tokens)
        # MCR predicts time t strictly from M_{t-1}; the target-frame local
        # visual tokens are fused only after causal reasoning.
        causal_sequence = torch.cat(
            (visual_sequence[:, :1], visual_sequence[:, :-1]), dim=1
        )
        text_token, aux = self.mcr(
            causal_sequence,
            recent_phase_histories,
            current_time_seconds=current_time_seconds,
        )
        # Keep the text-guided CoST path opt-in so checkpoints from the
        # first-fusion experiment can be evaluated without changing the
        # visual-only training default.
        local = self.cost.fuse_target(
            levels,
            text_token if self.use_text_cost_fusion else None,
            text_scale=self.history_text_scale,
        )
        memory = aux["memory"][:, -1]
        fusion_parts = (local, memory, text_token) if self.include_text_final_fusion else (local, memory)
        fused = self.fusion(torch.cat(fusion_parts, dim=-1))
        return (fused, aux) if return_aux else fused

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        recent_phase_histories: list[list] | None = None,
        current_time_seconds: list[int] | None = None,
        video_ids: list[str] | None = None,
        update_history: bool = False,
    ):
        if video_ids is not None:
            video_ids = self._validate_video_ids(video_ids, x.shape[0])
            if recent_phase_histories is not None:
                raise ValueError("Pass either video_ids or explicit phase histories, not both")
            recent_phase_histories = self._recent_phase_histories(video_ids)
        elif update_history:
            raise ValueError("update_history requires video_ids")

        features, aux = self.forward_features(
            x,
            return_aux=True,
            recent_phase_histories=recent_phase_histories,
            current_time_seconds=current_time_seconds,
        )
        logits = self.head(self.fc_dropout(features))
        if update_history:
            aux.update(self._update_stream_state(logits, video_ids))
        if return_aux:
            aux["logits"] = logits
            return logits, aux
        return logits

    def reset_stream_state(self, video_id: str | None = None) -> None:
        if video_id is None:
            self._stream_memory.clear()
        else:
            self._stream_memory.pop(video_id, None)

    def _validate_video_ids(
        self, video_ids: list[str], batch_size: int
    ) -> list[str]:
        video_ids = [str(video_id) for video_id in video_ids]
        if len(video_ids) != batch_size:
            raise ValueError("video_ids must match the input batch")
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("A batch accepts at most one frame per video")
        return video_ids

    def _recent_phase_histories(self, video_ids: list[str]) -> list[list[int]]:
        return [
            list(self._stream_memory[video_id].get("recent_phases", ()))
            for video_id in video_ids
        ]

    @torch.no_grad()
    def _causal_gaussian(self, probabilities: list[torch.Tensor]) -> torch.Tensor:
        """One-sided Gaussian smoothing over past/current class probabilities."""
        values = torch.stack(probabilities[-self.gaussian_history_length :], dim=0)
        count = values.shape[0]
        ages = torch.arange(
            count - 1, -1, -1, device=values.device, dtype=values.dtype
        )
        weights = torch.exp(-0.5 * (ages / self.gaussian_sigma).square())
        weights = weights / weights.sum()
        return (values * weights[:, None]).sum(dim=0)

    @torch.no_grad()
    def _update_stream_state(
        self,
        raw_logits: torch.Tensor,
        video_ids: list[str],
        current_time_seconds: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        raw_probabilities = raw_logits.float().softmax(dim=-1)
        smoothed_rows = []
        smoothed_predictions = []
        history_length = self.transition_history_length
        if current_time_seconds is None:
            current_time_seconds = [None] * len(video_ids)
        for index, video_id in enumerate(video_ids):
            state = self._stream_memory[video_id]
            probability_history = list(state.get("probabilities", ()))
            probability_history.append(raw_probabilities[index].detach())
            probability_history = probability_history[-self.gaussian_history_length :]
            smoothed = (
                self._causal_gaussian(probability_history)
                if self.smooth_output_probabilities
                else raw_probabilities[index]
            )
            prediction = int(smoothed.argmax(dim=-1).item())

            recent_phases = list(state.get("recent_phases", ()))
            # Store every preceding prediction, including repeated phases and
            # Other Operations. This is a strictly causal 32-frame window.
            recent_phases.append((prediction, current_time_seconds[index]))
            recent_phases = recent_phases[-history_length:]
            self._stream_memory[video_id] = {
                "probabilities": probability_history,
                "recent_phases": recent_phases,
            }
            smoothed_rows.append(smoothed)
            smoothed_predictions.append(prediction)
        return {
            "raw_probabilities": raw_probabilities,
            "smoothed_probabilities": torch.stack(smoothed_rows, dim=0),
            "smoothed_predictions": torch.tensor(
                smoothed_predictions, device=raw_logits.device, dtype=torch.long
            ),
        }

    @torch.no_grad()
    def predict_online(
        self,
        x: torch.Tensor,
        video_ids: list[str],
        current_time_seconds: list[int] | None = None,
    ) -> dict:
        """Predict one chronological frame per video and update causal state.

        Different videos may be batched together.  Two frames from the same
        video in one call are rejected because the later frame would otherwise
        miss the earlier prediction.
        """
        video_ids = self._validate_video_ids(video_ids, x.shape[0])
        histories = self._recent_phase_histories(video_ids)
        raw_logits = self.forward(
            x,
            recent_phase_histories=histories,
            current_time_seconds=current_time_seconds,
        )
        result = self._update_stream_state(
            raw_logits, video_ids, current_time_seconds=current_time_seconds
        )
        result["raw_logits"] = raw_logits
        return result

    @torch.no_grad()
    def predict_online_with_histories(
        self,
        x: torch.Tensor,
        video_ids: list[str],
        recent_phase_histories: list[list],
        current_time_seconds: list[int] | None = None,
    ) -> dict:
        """Use externally cached causal histories while retaining output smoothing."""
        video_ids = self._validate_video_ids(video_ids, x.shape[0])
        if len(recent_phase_histories) != x.shape[0]:
            raise ValueError("Phase histories must match the visual batch size")
        raw_logits = self.forward(
            x,
            recent_phase_histories=recent_phase_histories,
            current_time_seconds=current_time_seconds,
        )
        result = self._update_stream_state(
            raw_logits, video_ids, current_time_seconds=current_time_seconds
        )
        result["raw_logits"] = raw_logits
        return result

    @torch.no_grad()
    def forward_stream(self, x: torch.Tensor, video_ids: list[str]) -> torch.Tensor:
        """Backward-compatible alias returning log-probabilities after smoothing."""
        result = self.predict_online(x, video_ids)
        return result["smoothed_probabilities"].clamp_min(1e-12).log()


@register_model
def lungreco(pretrained: bool = False, pretrain_path: str | None = None, **kwargs):
    model = LungReco(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        target_frames = kwargs.get("all_frames", 16)
        backbone = visual_backbone(
            pretrained=True,
            pretrain_path=pretrain_path,
            num_classes=kwargs.get("num_classes", 7),
            # The released K400 TimeSformer checkpoint contains 8-frame
            # temporal embeddings; interpolate them explicitly for 16 frames.
            all_frames=8,
        )
        state = backbone.state_dict()
        if state["time_embed"].shape[1] != target_frames:
            state["time_embed"] = F.interpolate(
                state["time_embed"].transpose(1, 2),
                size=target_frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        incompatible = model.load_state_dict(state, strict=False)
        print(
            "Loaded visual encoder into LungReco; "
            f"new_keys={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        del backbone
    print(f"xLSTM parameters: {model.mcr.xlstm_parameter_count:,}")
    return model
