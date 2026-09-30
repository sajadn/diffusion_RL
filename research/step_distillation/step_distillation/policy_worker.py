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
"""Frozen teacher scoring and KD schedule for the shared Megatron worker."""

from typing import Any

import ray
import torch

from step_distillation.context_distillation import (
    LessMaskedKDSchedule,
    score_teacher_context,
)
from nemo_rl.models.megatron.train import model_forward
from just_grpo.algorithms.trace_grpo import TraceGRPOSchedule
from just_grpo.diffusion.denoising_schedule import SchedulePurpose
from just_grpo.algorithms.trace_grpo_policy_worker import TraceGRPOPolicyWorkerImpl
from nemo_rl.data.interfaces import TokenizerType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker


class ContextDistillationPolicyWorkerImpl(TraceGRPOPolicyWorkerImpl):
    def __init__(
        self, config: PolicyConfig, tokenizer: TokenizerType, **kwargs: Any
    ) -> None:
        kwargs["init_reference_model"] = True
        super().__init__(config, tokenizer, **kwargs)

    def _build_schedule(
        self, data: BatchedDataDict[Any], *, purpose: SchedulePurpose
    ) -> TraceGRPOSchedule:
        return LessMaskedKDSchedule(
            data,
            self.trace_grpo.schedule,
            pad_token_id=self.tokenizer.pad_token_id,
            padded_width=self.cfg["max_total_sequence_length"],
            weighted_loss=purpose == "train",
        )

    @torch.no_grad()
    def get_frozen_teacher_targets(
        self, *, data: BatchedDataDict[Any]
    ) -> BatchedDataDict[Any]:
        """Use the original checkpoint reference; restore student weights on return."""
        data.to("cuda")

        def forward(
            input_ids: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            return model_forward(
                self.model,
                BatchedDataDict(),
                input_ids,
                position_ids,
                attention_mask=None,
                defer_fp32_logits=True,
            )[..., : self.generation_vocab_size]

        with self.use_reference_model():
            result = score_teacher_context(
                forward,
                data,
                schedule_config=self.trace_grpo.schedule,
                pad_token_id=self.tokenizer.pad_token_id,
                padded_width=self.cfg["max_total_sequence_length"],
                vocab_size=self.generation_vocab_size,
            )
        return result.to("cpu")


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class ContextDistillationPolicyWorker(ContextDistillationPolicyWorkerImpl):
    pass
