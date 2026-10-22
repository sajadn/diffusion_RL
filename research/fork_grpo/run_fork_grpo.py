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
"""Run ForkGRPO through the shared diffusion GRPO driver."""

import argparse
from pathlib import Path

from block_diffusion import training
from block_diffusion.training import PrepareTrainingData
from fork_grpo.algorithm import ForkGRPO
from fork_grpo.config import validate_config
from omegaconf import DictConfig, OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def run(config: DictConfig) -> None:
    """Validate Fork settings and supply its rollout batch preparation callback."""
    diffusion = validate_config(config)

    def prepare_training_data_factory(stop_token_ids: list[int]) -> PrepareTrainingData:
        return ForkGRPO(diffusion, stop_token_ids=stop_token_ids).prepare_training_data

    if diffusion.runtime == "megatron_llada":
        # LLaDA uses the live Megatron model and a bidirectional full canvas.
        from just_grpo.generation.llada_generation import validate_llada
        from just_grpo.generation.llada_megatron import LLaDAMegatronGeneration

        training.run(
            config,
            diffusion=diffusion,
            generation_factory=LLaDAMegatronGeneration,
            validation_fn=validate_llada,
            prepare_training_data_factory=prepare_training_data_factory,
        )
        return
    training.run(
        config,
        diffusion=diffusion,
        prepare_training_data_factory=prepare_training_data_factory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent
        / "configs/recipes/fork_grpo-deepscaler-3b-4n8g-megatron-vllm-async-long.yaml",
    )
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = parse_hydra_overrides(load_config(args.config), overrides)
    OmegaConf.resolve(config)
    run(config)


if __name__ == "__main__":
    main()
