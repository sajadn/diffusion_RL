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
"""Wire the distillation algorithm into the shared rollout/training driver."""

from typing import TYPE_CHECKING

from omegaconf import DictConfig
from transformers import AutoConfig

from just_grpo.train import run as run_diffusion
from step_distillation.config import validate_config
from step_distillation.context_distillation import (
    ContextDistillation,
    ContextDistillationLoss,
)

if TYPE_CHECKING:
    from nemo_rl.models.policy.lm_policy import Policy


def run(config: DictConfig) -> None:
    """Train a student against a frozen teacher on positive recorded rollouts."""
    diffusion = validate_config(config)

    def configure_training(
        policy: "Policy", stop_token_ids: list[int]
    ) -> tuple[ContextDistillation, ContextDistillationLoss]:
        model_config = AutoConfig.from_pretrained(
            config.policy.model_name, trust_remote_code=True
        )
        return (
            ContextDistillation(
                diffusion, stop_token_ids=stop_token_ids, policy=policy
            ),
            ContextDistillationLoss(
                vocab_size=model_config.vocab_size,
                temperature=diffusion.sampling.temperature,
            ),
        )

    run_diffusion(
        config, validate=validate_config, configure_training=configure_training
    )
