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
"""Shared diffusion sampling configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class DiffusionSamplingParams(BaseModel, extra="allow"):
    temperature: float = Field(ge=0)
    block_size: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    selection_policy: Literal[
        "leftmost", "confidence_threshold", "random", "entropy", "low_confidence"
    ]
    threshold: float = Field(gt=0, lt=1)
    returns_reveal_steps: bool
    returns_entropy: bool
    emit_full_blocks: bool | None = None


class BlockDiffusionConfig(BaseModel):
    mask_token_id: int
    block_size: int = Field(gt=0)


class DiffusionExperimentConfig(BaseModel, extra="forbid"):
    """Backend and decoding settings shared by diffusion GRPO algorithms."""

    schedule: BlockDiffusionConfig
    runtime: Literal["reference_vllm", "megatron", "upstream_vllm", "upstream_sglang"]
    generation_python: str | None
    max_generation_kl: float = Field(ge=0)
    distributed: bool
    sampling: DiffusionSamplingParams
    validation_sampling: DiffusionSamplingParams
