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
"""Configuration types and validation; experiment defaults live in YAML."""

import re
from typing import Self

from block_diffusion.config import (
    BlockDiffusionConfig,
    DiffusionExperimentConfig,
)
from block_diffusion.config import (
    DiffusionSamplingParams as DiffusionSamplingParams,
)
from block_diffusion.validation import validate_experiment
from omegaconf import DictConfig, OmegaConf
from pydantic import Field, model_validator


class ScheduleConfig(BlockDiffusionConfig, extra="allow"):
    reveal_tokens_per_step: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_width(self) -> Self:
        if self.reveal_tokens_per_step > self.block_size:
            raise ValueError("reveal_tokens_per_step cannot exceed block_size")
        return self


class JustGRPOConfig(DiffusionExperimentConfig):
    """Diffusion-specific extension to the standard NeMo-RL configuration."""

    generation_batch_size: int = Field(default=4, gt=0)
    training_token_fraction: float = Field(gt=0, le=1)
    schedule: ScheduleConfig
    validation_modes: dict[str, DiffusionSamplingParams] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_training_distribution(self) -> Self:
        s, p = self.schedule, self.sampling
        if p.temperature <= 0:
            raise ValueError("Training temperature must be positive")
        if p.block_size != s.block_size or p.selection_policy != "leftmost":
            raise ValueError("Training requires matching blocks and leftmost sampling")
        if s.block_size % s.reveal_tokens_per_step:
            raise ValueError(
                "Reference vLLM needs block_size divisible by reveal width"
            )
        if p.max_steps != s.block_size // s.reveal_tokens_per_step:
            raise ValueError("Generation steps must match the training reveal schedule")
        if self.training_token_fraction < 1 and s.reveal_tokens_per_step != 1:
            raise ValueError("Fast partial views require one token per reveal")
        if self.validation_modes and self.runtime != "megatron_llada":
            raise ValueError("Named LLaDA validation modes require megatron_llada")
        if any(
            not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in self.validation_modes
        ):
            raise ValueError("Validation mode names must be filename-safe identifiers")
        for params in (p, self.validation_sampling, *self.validation_modes.values()):
            if (
                params.returns_entropy
                or params.returns_reveal_steps
                or params.emit_full_blocks
            ):
                raise ValueError(
                    "Generation does not yet transport entropy/reveal channels"
                )
        if self.runtime not in (
            "reference_vllm",
            "megatron",
            "megatron_llada",
        ):
            raise ValueError(
                "Upstream generation cannot yet replay leftmost reveals; use megatron or reference_vllm"
            )
        if self.runtime == "megatron_llada":
            if s.reveal_tokens_per_step != 1 or self.training_token_fraction != 1:
                raise ValueError("LLaDA supports exact one-token JustGRPO replay only")
            for v in (self.validation_sampling, *self.validation_modes.values()):
                if (
                    v.selection_policy not in (
                        "low_confidence",
                        "confidence_threshold",
                        "fork",
                    )
                    or not 1 <= v.max_steps <= v.block_size
                    or v.block_size % v.max_steps
                ):
                    raise ValueError(
                        "LLaDA evaluation requires confidence or Fork unmasking with at most one pass per canvas token"
                    )
                if (
                    v.max_steps != v.block_size
                    and v.selection_policy in ("confidence_threshold", "fork")
                ):
                    raise ValueError(
                        "Fixed parallel LLaDA evaluation requires low_confidence"
                    )
                if s.block_size % v.block_size:
                    raise ValueError(
                        "LLaDA canvas must contain complete validation blocks"
                    )
        return self


def validate_config(config: DictConfig) -> JustGRPOConfig:
    """Parse JustGRPO settings and validate the shared experiment configuration."""
    diffusion = JustGRPOConfig.model_validate(
        OmegaConf.to_container(config.just_grpo, resolve=True)
    )
    validate_experiment(config, diffusion)
    return diffusion
