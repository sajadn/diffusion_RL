# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GRPO on sampled levels of the recorded diffusion decoding trajectory."""

from typing import Any, Literal, Self

import torch
from pydantic import Field, model_validator

from just_grpo.diffusion.block_layout import BlockDiffusionLayout
from just_grpo.diffusion.config import DiffusionExperimentConfig, BlockDiffusionConfig
from just_grpo.diffusion.denoising_schedule import DenoisingSchedule
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class TraceScheduleConfig(BlockDiffusionConfig, extra="forbid"):
    num_level_samples: int = Field(gt=0)
    seed_base: int
    sampled_level_reduction: Literal["sum", "mean"]


class TraceGRPOConfig(DiffusionExperimentConfig):
    schedule: TraceScheduleConfig

    @model_validator(mode="after")
    def validate_trace(self) -> Self:
        if self.runtime not in ("megatron", "reference_vllm"):
            raise ValueError("Trace requires Megatron or reference vLLM generation")
        if self.sampling.temperature <= 0:
            raise ValueError("Training temperature must be positive")
        for sampling in (self.sampling, self.validation_sampling):
            if self.runtime == "megatron" and sampling.selection_policy not in (
                "leftmost",
                "confidence_threshold",
            ):
                raise ValueError(
                    "Megatron generation supports leftmost or confidence_threshold selection"
                )
            if sampling.block_size != self.schedule.block_size:
                raise ValueError("Trace sampling and training block sizes must match")
            if not sampling.returns_reveal_steps or not sampling.emit_full_blocks:
                raise ValueError(
                    "Trace requires recorded reveal steps and full terminal blocks"
                )
            if sampling.returns_entropy:
                raise ValueError("Trace does not consume token entropies")
            if (
                sampling.selection_policy == "leftmost"
                and sampling.block_size % sampling.max_steps
            ):
                raise ValueError(
                    "Leftmost sampling requires a divisible reveal schedule"
                )
        return self


def sample_trace_levels(
    reveal_levels: torch.Tensor,
    *,
    num_level_samples: int,
    seeds: torch.Tensor,
    reduction: Literal["sum", "mean"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample distinct per-row levels; retain reference depth/k or 1/k weights."""
    depths = (reveal_levels.max(dim=1).values + 1).clamp_min(0)
    levels = depths[:, None].expand(-1, num_level_samples).clone()
    weights = torch.zeros(depths.shape, device=depths.device, dtype=torch.float32)
    for row, (depth, seed) in enumerate(zip(depths.tolist(), seeds.tolist())):
        if depth == 0:
            continue
        count = min(num_level_samples, depth)
        generator = torch.Generator().manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
        levels[row, :count] = torch.randperm(depth, generator=generator)[:count].to(
            levels.device
        )
        weights[row] = 1 / count if reduction == "mean" else depth / count
    return levels, weights


class TraceGRPO:
    """Prepare one reproducible set of level draws before any policy dispatch."""

    def __init__(self, config: TraceGRPOConfig, *, stop_token_ids: list[int]) -> None:
        self.config = config
        self.stop_token_ids = stop_token_ids

    def prepare_training_data(
        self,
        data: BatchedDataDict[Any],
        message_logs: list[list[dict]],
        step: int,
    ) -> None:
        ids = data["input_ids"]
        steps = torch.full_like(ids, -1)
        for row, messages in enumerate(message_logs):
            offset = 0
            for message in messages:
                length = len(message["token_ids"])
                if "reveal_steps" in message:
                    recorded = message["reveal_steps"]
                    if len(recorded) != length:
                        raise ValueError(
                            "Reveal steps must align with generated tokens"
                        )
                    steps[row, offset : offset + length] = recorded.to(steps.device)
                offset += length
        response = data["token_mask"].bool()
        response &= (
            torch.arange(ids.shape[1], device=ids.device)[None]
            < data["input_lengths"][:, None]
        )
        stops = (
            torch.isin(ids, torch.tensor(self.stop_token_ids, device=ids.device))
            & response
        )
        after_stop = (stops.long().cumsum(1) - stops.long()) > 0
        loss_mask = response & ~after_stop
        included = data["sample_mask"].bool()[:, None]
        if ((steps < 0) & loss_mask & included).any():
            raise ValueError(
                "Trace requires a recorded reveal step for every scored response token"
            )
        levels = torch.full_like(steps, -1)
        for row in range(ids.shape[0]):
            committed = response[row] & (steps[row] >= 0)
            distinct = torch.unique(steps[row, committed], sorted=True)
            levels[row, committed] = torch.searchsorted(distinct, steps[row, committed])
        config = self.config.schedule
        seeds = (
            config.seed_base
            + step * 1_000_003
            + torch.arange(ids.shape[0], device=ids.device)
        )
        selected, weights = sample_trace_levels(
            levels,
            num_level_samples=config.num_level_samples,
            seeds=seeds,
            reduction=config.sampled_level_reduction,
        )
        data["trace_reveal_levels"] = levels
        data["trace_sampled_levels"] = selected
        data["trace_loss_weights"] = weights
        data["trace_loss_mask"] = loss_mask
        # Sparse scores must never be diagnosed as zero-logprob full responses.
        data["logprob_token_mask"] = loss_mask & (
            levels[:, :, None] == selected[:, None, :]
        ).any(-1)


class TraceGRPOSchedule(DenoisingSchedule):
    """Render recorded contexts with a fixed, DP-uniform number of sampled views."""

    def __init__(
        self,
        data: BatchedDataDict[Any],
        config: TraceScheduleConfig,
        *,
        pad_token_id: int,
        padded_width: int | None = None,
        weighted_loss: bool,
    ) -> None:
        for key in (
            "trace_reveal_levels",
            "trace_sampled_levels",
            "trace_loss_weights",
            "trace_loss_mask",
        ):
            if key not in data:
                raise ValueError(
                    "Trace batches must be prepared before policy/reference/training dispatch"
                )
        self.layout = BlockDiffusionLayout(
            data,
            block_size=config.block_size,
            pad_token_id=pad_token_id,
            padded_width=padded_width,
        )
        padding = self.layout.base["input_ids"].shape[1] - data["input_ids"].shape[1]
        self.levels = torch.nn.functional.pad(
            data["trace_reveal_levels"], (0, padding), value=-1
        )
        self.loss_mask = (
            torch.nn.functional.pad(data["trace_loss_mask"], (0, padding))
            & self.layout.response_mask
        )
        self.sampled = data["trace_sampled_levels"]
        if self.sampled.shape != (data.size, config.num_level_samples):
            raise ValueError("Sampled levels must match num_level_samples")
        self.weights = (
            data["trace_loss_weights"]
            if weighted_loss
            else torch.ones_like(data["trace_loss_weights"])
        )
        super().__init__(num_steps=config.num_level_samples, num_samples=data.size)

    def generate_single_trajectory(self, time_step: int) -> BatchedDataDict[Any]:
        if not 0 <= time_step < self.num_steps:
            raise IndexError(time_step)
        chosen = self.sampled[:, time_step, None]
        result = BatchedDataDict(self.layout.base)
        result["masked_indices"] = self.layout.canvas_mask & (
            (self.levels < 0) | (self.levels >= chosen)
        )
        result["token_mask"] = (
            self.loss_mask & (self.levels == chosen)
        ).float() * self.weights[:, None]
        return result


def validate_trace_experiment(config: Any) -> None:
    """Reject controller features that require full-response policy scores."""
    if config.grpo.seq_logprob_error_threshold is not None:
        raise ValueError(
            "Sampled Trace scores cannot apply whole-response seq_logprob_error_threshold"
        )
    if (config.get("data_plane") or {}).get("enabled"):
        raise ValueError("Trace batch preparation requires the driver rollout path")
    if (
        config.policy.worker_extension_cls_fqn
        != "just_grpo.algorithms.trace_grpo_policy_worker.TraceGRPOPolicyWorker"
    ):
        raise ValueError("Trace requires TraceGRPOPolicyWorker")
