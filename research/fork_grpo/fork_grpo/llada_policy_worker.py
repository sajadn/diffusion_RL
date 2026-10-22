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
"""Fork replay on the existing full-canvas LLaDA worker."""

from typing import Any

import ray
from block_diffusion.config import DiffusionSamplingParams
from block_diffusion.denoising_schedule import SchedulePurpose
from block_diffusion.full_canvas import FullCanvasLayout
from just_grpo.algorithms.llada_policy_worker import LLaDAJustGRPOPolicyWorkerImpl
from omegaconf import OmegaConf
from trace_grpo.algorithm import TraceGRPOSchedule

from fork_grpo.algorithm import ForkGRPOConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker


class LLaDAForkGRPOPolicyWorkerImpl(LLaDAJustGRPOPolicyWorkerImpl):
    """Share model construction, generation, loss processing, and streaming training."""

    def _load_diffusion_config(self, research: Any) -> ForkGRPOConfig:
        return ForkGRPOConfig.model_validate(
            OmegaConf.to_container(research.fork_grpo, resolve=True)
        )

    def _generation_block_length(self, sampling: DiffusionSamplingParams) -> int:
        return sampling.block_size

    def _build_schedule(
        self, data: BatchedDataDict[Any], *, purpose: SchedulePurpose
    ) -> TraceGRPOSchedule:
        return TraceGRPOSchedule(
            data,
            self.diffusion.schedule,
            pad_token_id=self.tokenizer.pad_token_id,
            weighted_loss=purpose == "train",
            layout=FullCanvasLayout(
                data, canvas_length=self.diffusion.schedule.block_size
            ),
        )


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class LLaDAForkGRPOPolicyWorker(LLaDAForkGRPOPolicyWorkerImpl):  # pragma: no cover
    pass
