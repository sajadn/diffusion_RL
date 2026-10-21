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
from block_diffusion.block_layout import BlockDiffusionLayout
from block_diffusion.config import BlockDiffusionConfig, DiffusionExperimentConfig
from block_diffusion.denoising_schedule import DenoisingSchedule
from block_diffusion.replay import recorded_reveal_states
from pydantic import Field, model_validator

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
        levels, loss_mask = recorded_reveal_states(
            data, message_logs, stop_token_ids=self.stop_token_ids
        )
        ids = data["input_ids"]
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


class TraceGRPOSchedule(DenoisingSchedule):
    """Replay prepared levels, loss masks, and weights for TraceGRPO or ForkGRPO.

    Selected levels can be per response [N,K] or per response block [N,K,B].
    Block weights have shape [N,B]; response weights have shape [N].
    The prepared view count survives DP slicing, including empty shards.
    """

    def __init__(
        self,
        data: BatchedDataDict[Any],
        config: BlockDiffusionConfig,
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
        positions = self.layout.base["original_positions"]
        self.levels = data["trace_reveal_levels"].gather(1, positions)
        self.levels = self.levels.masked_fill(~self.layout.response_mask, -1)
        self.loss_mask = (
            data["trace_loss_mask"].gather(1, positions) & self.layout.response_mask
        )
        self.sampled = data["trace_sampled_levels"]
        if self.sampled.ndim not in (2, 3) or self.sampled.shape[0] != data.size:
            raise ValueError("Selected levels must have shape [N,K] or [N,K,B]")
        self.block_indices = None
        if self.sampled.ndim == 3:
            num_blocks = self.sampled.shape[2]
            if (
                num_blocks * config.block_size
                < int(self.layout.base["noisy_valid_lengths"].max())
                or num_blocks == 0
            ):
                raise ValueError("Selected block levels must cover the response canvas")
            # Fixed-width worker padding can exceed the driver's response canvas.
            self.block_indices = (
                torch.arange(positions.shape[1], device=positions.device)
                .div(config.block_size, rounding_mode="floor")
                .clamp_max(num_blocks - 1)
                .expand(data.size, -1)
            )
        if (
            isinstance(config, TraceScheduleConfig)
            and self.sampled.shape[1] != config.num_level_samples
        ):
            raise ValueError("Sampled levels must match num_level_samples")
        self.weights = (
            data["trace_loss_weights"]
            if weighted_loss
            else torch.ones_like(data["trace_loss_weights"])
        )
        expected_weights = (
            (data.size,)
            if self.sampled.ndim == 2
            else (data.size, self.sampled.shape[2])
        )
        if self.weights.shape != expected_weights:
            raise ValueError(
                "Loss weights must match the selected response/block levels"
            )
        self.weights = (
            self.weights[:, None]
            if self.block_indices is None
            else self.weights.gather(1, self.block_indices)
        )
        super().__init__(num_steps=self.sampled.shape[1], num_samples=data.size)

    def generate_single_trajectory(self, time_step: int) -> BatchedDataDict[Any]:
        if not 0 <= time_step < self.num_steps:
            raise IndexError(time_step)
        chosen = (
            self.sampled[:, time_step, None]
            if self.block_indices is None
            else self.sampled[:, time_step].gather(1, self.block_indices)
        )
        result = BatchedDataDict(self.layout.base)
        result["masked_indices"] = self.layout.canvas_mask & (
            (self.levels < 0) | (self.levels >= chosen)
        )
        result["token_mask"] = (
            self.loss_mask & (self.levels == chosen)
        ).float() * self.weights
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
        != "trace_grpo.policy_worker.TraceGRPOPolicyWorker"
    ):
        raise ValueError("Trace requires TraceGRPOPolicyWorker")
