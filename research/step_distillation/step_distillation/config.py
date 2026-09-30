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
"""Configuration and guards for positive-rollout context distillation."""

from typing import Literal, Self

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, model_validator

from just_grpo.algorithms.trace_grpo import TraceGRPOConfig
from just_grpo.config import validate_config as validate_diffusion_config


class StepDistillationConfig(BaseModel, extra="forbid"):
    """Frozen next-state teacher KD on positive student rollouts."""

    target_source: Literal["less_masked_kd"] = "less_masked_kd"
    positive_only: bool = True
    min_reward: float = Field(default=0.0, allow_inf_nan=False)


class StepDistillationExperimentConfig(TraceGRPOConfig):
    step_distillation: StepDistillationConfig

    @model_validator(mode="after")
    def validate_distillation(self) -> Self:
        if (
            self.schedule.num_level_samples != 1
            or self.schedule.sampled_level_reduction != "mean"
        ):
            raise ValueError(
                "Step distillation requires one sampled level with mean reduction"
            )
        return self


def validate_config(config: DictConfig) -> StepDistillationExperimentConfig:
    """Validate KD restrictions before configuring shared diffusion infrastructure."""
    diffusion = StepDistillationExperimentConfig.model_validate(
        {
            **OmegaConf.to_container(config.trace_grpo, resolve=True),
            "step_distillation": OmegaConf.to_container(
                config.step_distillation, resolve=True
            ),
        }
    )
    g, p = config.grpo, config.policy
    if (
        g.get("freeze_generation")
        or g.async_grpo.enabled
        or diffusion.runtime != "reference_vllm"
    ):
        raise ValueError(
            "Context KD requires live synchronous reference-vLLM student rollouts"
        )
    if not p.generation.colocated.enabled:
        raise ValueError("Context KD requires colocated generation")
    if diffusion.sampling.selection_policy != "entropy":
        raise ValueError("Context KD requires fixed-count entropy sampling")
    if (
        not diffusion.step_distillation.positive_only
        or g.num_generations_per_prompt < 2
    ):
        raise ValueError(
            "Context KD requires positive-only filtering and multiple rollouts per prompt"
        )
    if (
        diffusion.sampling.max_steps < 2
        or diffusion.schedule.block_size % diffusion.sampling.max_steps
    ):
        raise ValueError(
            "Context KD requires at least two fixed-count student steps per block"
        )
    if p.generation.vllm_cfg.precision != "bfloat16":
        raise ValueError("Context KD currently requires BF16")
    if config.policy.megatron_cfg.tensor_model_parallel_size != 1:
        raise ValueError(
            "Step distillation currently requires TP=1 for full-vocabulary KD"
        )
    if (
        config.loss_fn.reference_policy_kl_penalty != 0
        or not config.grpo.skip_reference_policy_logprobs_calculation
        or config.loss_fn.use_kl_in_reward
    ):
        raise ValueError("Step distillation requires reference KL disabled")
    validate_diffusion_config(
        config,
        diffusion=diffusion,
        trace_worker_cls_fqn="step_distillation.policy_worker.ContextDistillationPolicyWorker",
        allow_checkpointing=True,
    )
    return diffusion
