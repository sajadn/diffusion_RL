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
"""Run Block JustGRPO through the shared diffusion driver."""

import argparse
from functools import partial
from pathlib import Path

from block_diffusion import training
from just_grpo.algorithms.llada_schedule import make_llada_training_data_preparer
from just_grpo.config import validate_config
from omegaconf import DictConfig, OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def run(config: DictConfig, *, evaluation_only: bool = False) -> None:
    """Validate JustGRPO settings and start the shared training loop."""
    diffusion = validate_config(config)
    if config.data.default.env_name == "justgrpo_code":
        # Code grading brings optional reference dependencies only for this task.
        from just_grpo.environments.code_reference import register_code_reference

        register_code_reference()
    if diffusion.runtime == "megatron_llada":
        # Load the optional LLaDA adapter only for its shared-worker runtime.
        from just_grpo.generation.llada_generation import validate_llada
        from just_grpo.generation.llada_megatron import LLaDAMegatronGeneration

        training.run(
            config,
            diffusion=diffusion,
            evaluation_only=evaluation_only,
            generation_factory=LLaDAMegatronGeneration,
            validation_fn=validate_llada,
            prepare_training_data_factory=partial(
                make_llada_training_data_preparer,
                canvas_length=diffusion.schedule.block_size,
            ),
        )
    else:
        training.run(config, diffusion=diffusion, evaluation_only=evaluation_only)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent
        / "configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Validate once without training"
    )
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = parse_hydra_overrides(load_config(args.config), overrides)
    OmegaConf.resolve(config)
    run(config, evaluation_only=args.eval_only)


if __name__ == "__main__":
    main()
