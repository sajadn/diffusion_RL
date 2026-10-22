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
"""Validate ForkGRPO and shared diffusion runtime settings."""

from block_diffusion.validation import validate_experiment
from omegaconf import DictConfig, OmegaConf

from fork_grpo.algorithm import ForkGRPOConfig


def validate_config(config: DictConfig) -> ForkGRPOConfig:
    diffusion = ForkGRPOConfig.model_validate(
        OmegaConf.to_container(config.fork_grpo, resolve=True)
    )
    if config.grpo.seq_logprob_error_threshold is not None:
        raise ValueError("Fallback-only scores cannot apply whole-response filtering")
    if (config.get("data_plane") or {}).get("enabled"):
        raise ValueError("ForkGRPO requires driver rollout batch preparation")
    if config.policy.worker_extension_cls_fqn != (
        "fork_grpo.llada_policy_worker.LLaDAForkGRPOPolicyWorker"
        if diffusion.runtime == "megatron_llada"
        else "fork_grpo.policy_worker.ForkGRPOPolicyWorker"
    ):
        raise ValueError("ForkGRPO requires ForkGRPOPolicyWorker")
    if config.policy.get("diffusion_aux_loss") is not None:
        raise ValueError("The ForkGRPO baseline trains only fallback tokens")
    validate_experiment(config, diffusion)
    return diffusion
