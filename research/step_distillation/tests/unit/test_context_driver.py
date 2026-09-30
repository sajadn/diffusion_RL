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
"""Dedicated KD driver uses the native controller with its own worker and loss."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import transformers
from omegaconf import OmegaConf

from just_grpo import train as shared_train
from just_grpo.environments import sudoku
from nemo_rl.algorithms import utils as algorithm_utils
from nemo_rl.distributed import ray_actor_environment_registry as registry
from nemo_rl.distributed import virtual_cluster
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from step_distillation import train
from step_distillation.context_distillation import (
    ContextDistillation,
    ContextDistillationLoss,
)


def test_dedicated_driver_installs_kd_callback_loss_and_worker(monkeypatch, tmp_path):
    register_omegaconf_resolvers()
    config = load_config(
        Path(__file__).parents[3]
        / "just_grpo/configs/recipes/trace_grpo-sudoku6x6-4n8g-megatron-vllm-long.yaml"
    )
    config.policy.worker_extension_cls_fqn = (
        "step_distillation.policy_worker.ContextDistillationPolicyWorker"
    )
    config.step_distillation = {"target_source": "less_masked_kd"}
    config.trace_grpo.schedule.sampled_level_reduction = "mean"
    config.trace_grpo.sampling.selection_policy = "entropy"
    config.trace_grpo.sampling.max_steps = 4
    config.grpo.skip_reference_policy_logprobs_calculation = True
    config.loss_fn.reference_policy_kl_penalty = 0.0
    config.loss_fn.use_importance_sampling_correction = False
    config.loss_fn.truncated_importance_sampling_type = None
    config.loss_fn.truncated_importance_sampling_ratio = None
    config.checkpointing.enabled = True
    config.logger.log_dir = str(tmp_path)
    policy, generation = Mock(), Mock()
    logger, checkpointer = Mock(wandb_logger=None), MagicMock()

    def master_config(**raw):
        return SimpleNamespace(policy=raw["policy"], grpo=OmegaConf.create(raw["grpo"]))

    def setup(master, tokenizer, train_dataset, val_dataset):
        assert (
            master.policy["worker_extension_cls_fqn"]
            == config.policy.worker_extension_cls_fqn
        )
        assert "step_distillation" in master.policy["diffusion_config"]
        assert (
            "step_distillation" not in master.policy["diffusion_config"]["trace_grpo"]
        )
        return (
            policy,
            generation,
            None,
            None,
            [],
            [],
            Mock(),
            logger,
            checkpointer,
            Mock(),
            master,
            None,
            None,
        )

    def trainer(*args, **kwargs):
        assert args[0] is policy
        assert args[1] is generation
        assert isinstance(args[5], ContextDistillationLoss)
        callback = kwargs["prepare_training_data_fn"]
        assert isinstance(callback.__self__, ContextDistillation)
        assert callback.__self__.policy is policy
        assert kwargs == {"shift_labels": False, "prepare_training_data_fn": callback}

    grpo = ModuleType("nemo_rl.algorithms.grpo")
    grpo.MasterConfig = master_config
    grpo.setup = Mock(side_effect=setup)
    grpo.grpo_train = Mock(side_effect=trainer)
    grpo.async_grpo_train = Mock()
    monkeypatch.setitem(sys.modules, grpo.__name__, grpo)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(vocab_size=128),
    )
    monkeypatch.setattr(
        algorithm_utils,
        "get_tokenizer",
        lambda _: SimpleNamespace(pad_token_id=0, eos_token_id=2),
    )
    monkeypatch.setattr(sudoku, "SudokuResponseDataset", Mock(return_value=[]))
    monkeypatch.setattr(sudoku, "SudokuEnvironment", Mock())
    monkeypatch.setattr(registry, "ACTOR_ENVIRONMENT_REGISTRY", {})
    monkeypatch.setattr(registry, "get_actor_python_env", lambda _: sys.executable)
    monkeypatch.setattr(virtual_cluster, "init_ray", Mock())
    monkeypatch.setattr(shared_train.ray, "shutdown", Mock())
    train.run(config)
    grpo.setup.assert_called_once()
    grpo.grpo_train.assert_called_once()
    grpo.async_grpo_train.assert_not_called()
    generation.shutdown.assert_called_once()
    logger.finish.assert_called_once()
    shared_train.ray.shutdown.assert_called_once()
