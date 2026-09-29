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
"""Block JustGRPO schedule selection for the shared diffusion policy worker."""

import math
from typing import Any

import ray
from omegaconf import OmegaConf

from just_grpo.algorithms.block_just_grpo import (
    BlockJustGRPOSchedule,
    select_low_confidence_tokens,
)
from just_grpo.config import JustGRPOConfig
from just_grpo.diffusion.denoising_schedule import SchedulePurpose
from just_grpo.diffusion.megatron_diffusion_policy import (
    MegatronDiffusionPolicyWorkerImpl,
)
from nemo_rl.data.interfaces import TokenizerType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker


class BlockJustGRPOPolicyWorkerImpl(MegatronDiffusionPolicyWorkerImpl):
    """Supply JustGRPO reveal schedules and Fast token selection."""

    def __init__(
        self, config: PolicyConfig, tokenizer: TokenizerType, **kwargs: Any
    ) -> None:
        research_config = OmegaConf.create(config.pop("diffusion_config"))
        self.just_grpo = JustGRPOConfig.model_validate(
            OmegaConf.to_container(research_config.just_grpo, resolve=True)
        )
        super().__init__(
            config,
            tokenizer,
            mask_token_id=self.just_grpo.schedule.mask_token_id,
            block_size=self.just_grpo.schedule.block_size,
            sampling=self.just_grpo.sampling,
            generation_seed=research_config.grpo.seed,
            max_generation_kl=self.just_grpo.max_generation_kl,
            **kwargs,
        )

    def _build_schedule(
        self, data: BatchedDataDict[Any], *, purpose: SchedulePurpose
    ) -> BlockJustGRPOSchedule:
        fraction = (
            1.0 if purpose == "policy" else self.just_grpo.training_token_fraction
        )
        data["training_token_mask"] = select_low_confidence_tokens(
            data,
            fraction=fraction,
            block_size=self.just_grpo.schedule.block_size,
        )
        block = self.just_grpo.schedule.block_size
        return BlockJustGRPOSchedule(
            data,
            self.just_grpo.schedule,
            pad_token_id=self.tokenizer.pad_token_id,
            padded_width=self.cfg["max_total_sequence_length"],
            selected_positions_per_block=math.ceil(fraction * block)
            if fraction < 1
            else None,
        )


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class BlockJustGRPOPolicyWorker(BlockJustGRPOPolicyWorkerImpl):
    pass
