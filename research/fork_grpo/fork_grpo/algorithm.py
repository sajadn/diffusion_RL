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
"""Sample Fork-dLLM fallback decisions and replay their recorded masked states."""

from typing import Any, Self

import torch
from block_diffusion.block_layout import BlockDiffusionLayout
from block_diffusion.config import BlockDiffusionConfig, DiffusionExperimentConfig
from block_diffusion.replay import recorded_reveal_states
from just_grpo.algorithms.llada_schedule import prepare_llada_training_data
from pydantic import Field, model_validator

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class ForkScheduleConfig(BlockDiffusionConfig, extra="forbid"):
    """Number of replay views and seed for uniform per-block fallback sampling."""

    num_level_samples: int = Field(gt=0)
    seed_base: int


class ForkGRPOConfig(DiffusionExperimentConfig):
    schedule: ForkScheduleConfig
    generation_batch_size: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def validate_fork(self) -> Self:
        if self.runtime not in {"megatron", "reference_vllm", "megatron_llada"}:
            raise ValueError("ForkGRPO requires Megatron or reference vLLM generation")
        selection_policy = (
            "fork"
            if self.runtime in {"megatron", "megatron_llada"}
            else "confidence_threshold_leftmost"
        )
        if self.sampling.temperature <= 0:
            raise ValueError("Training temperature must be positive")
        for is_validation, sampling in (
            (False, self.sampling),
            (True, self.validation_sampling),
        ):
            if (
                self.runtime == "megatron_llada"
                and is_validation
                and sampling.selection_policy == "low_confidence"
            ):
                if (
                    self.schedule.block_size % sampling.block_size
                    or sampling.max_steps != sampling.block_size
                    or sampling.returns_entropy
                ):
                    raise ValueError(
                        "LLaDA baseline validation needs complete blocks, one token "
                        "per step, and no entropy metadata"
                    )
                continue
            if sampling.selection_policy != selection_policy:
                raise ValueError(f"ForkGRPO requires {selection_policy} sampling")
            if sampling.block_size != self.schedule.block_size:
                raise ValueError("Fork sampling and training block sizes must match")
            if (
                self.runtime == "megatron_llada"
                and sampling.max_steps != sampling.block_size
            ):
                raise ValueError(
                    "LLaDA Fork needs one possible fallback per canvas token"
                )
            if sampling.max_steps < sampling.block_size:
                raise ValueError("Fork decoding requires max_steps >= block_size")
            if not sampling.returns_reveal_steps or not sampling.emit_full_blocks:
                raise ValueError(
                    "ForkGRPO requires reveal steps and full terminal blocks"
                )
            if sampling.returns_entropy:
                raise ValueError("ForkGRPO uses fallback flags, not token entropy")
        return self


class ForkGRPO:
    """Retain rewards for every rollout and select only recorded fallback tokens."""

    def __init__(self, config: ForkGRPOConfig, *, stop_token_ids: list[int]) -> None:
        self.config = config
        self.stop_token_ids = stop_token_ids

    def prepare_training_data(
        self, data: BatchedDataDict[Any], message_logs: list[list[dict]], step: int
    ) -> None:
        if self.config.runtime == "megatron_llada":
            prepare_llada_training_data(
                data, message_logs, step, canvas_length=self.config.schedule.block_size
            )
        levels, loss_mask = recorded_reveal_states(
            data, message_logs, stop_token_ids=self.stop_token_ids
        )
        fallback = torch.zeros_like(loss_mask)
        for row, messages in enumerate(message_logs):
            offset = 0
            for message in messages:
                length = len(message["token_ids"])
                if "reveal_steps" in message:
                    if "fallback_mask" not in message:
                        raise ValueError("ForkGRPO requires explicit fallback metadata")
                    flags = message["fallback_mask"]
                    if len(flags) != length:
                        raise ValueError(
                            "Fallback mask must align with generated tokens"
                        )
                    fallback[row, offset : offset + length] = flags.to(fallback.device)
                offset += length
        if (fallback & (levels < 0)).any():
            raise ValueError("Fallback tokens must have recorded reveal steps")
        fallback &= loss_mask
        config = self.config.schedule
        if self.config.runtime == "megatron_llada":
            fallback &= data["sample_mask"].bool()[:, None]
            counts = fallback.sum(1)
            views = min(config.num_level_samples, max(1, int(counts.max())))
            sampled = torch.full(
                (data.size, views),
                self.config.sampling.max_steps,
                device=levels.device,
                dtype=torch.long,
            )
            weights = torch.zeros(data.size, device=levels.device)
            for row in range(data.size):
                values = levels[row, fallback[row]]
                if values.unique().numel() != values.numel():
                    raise ValueError(
                        "Full-canvas Fork needs one fallback per global level"
                    )
                count = min(views, values.numel())
                if not count:
                    continue
                seed = config.seed_base + step * 1_000_003 + row
                generator = torch.Generator().manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
                selected = torch.randperm(values.numel(), generator=generator)[:count]
                sampled[row, :count] = values[selected.to(values.device)]
                weights[row] = 1 / count
            data["trace_reveal_levels"] = levels
            data["trace_loss_mask"] = fallback
            data["trace_sampled_levels"] = sampled
            data["trace_loss_weights"] = weights
            print(
                f"Fork LLaDA replay: views={views}, cap={config.num_level_samples}, "
                f"fallback_mean={counts.float().mean().item():.2f}, "
                f"fallback_max={int(counts.max())}",
                flush=True,
            )
            return
        layout = BlockDiffusionLayout(
            data, block_size=config.block_size, pad_token_id=config.mask_token_id
        )
        positions = layout.base["original_positions"]
        block_levels = levels.gather(1, positions)
        block_fallback = fallback.gather(1, positions) & layout.response_mask
        num_blocks = positions.shape[1] // config.block_size
        sampled = torch.full(
            (data.size, config.num_level_samples, num_blocks),
            self.config.sampling.max_steps,
            device=levels.device,
            dtype=torch.long,
        )
        weights = torch.zeros(
            (data.size, num_blocks), device=levels.device, dtype=torch.float32
        )
        totals = block_fallback.sum(1).clamp_min(1)
        for row in range(data.size):
            seed = config.seed_base + step * 1_000_003 + row
            generator = torch.Generator().manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
            for block in range(num_blocks):
                span = slice(block * config.block_size, (block + 1) * config.block_size)
                values = block_levels[row, span][block_fallback[row, span]]
                count = values.numel()
                if count == 0:
                    continue
                if values.unique().numel() != count:
                    raise ValueError("Fork requires one fallback token per block level")
                selected_count = min(config.num_level_samples, count)
                selected = torch.randperm(count, generator=generator)[:selected_count]
                sampled[row, :selected_count, block] = values[
                    selected.to(values.device)
                ]
                # Inverse inclusion probability preserves the full fallback mean.
                weights[row, block] = count / selected_count / totals[row]
        data["trace_reveal_levels"] = levels
        data["trace_loss_mask"] = fallback
        data["trace_sampled_levels"] = sampled
        data["trace_loss_weights"] = weights
