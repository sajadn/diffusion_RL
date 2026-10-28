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
"""DeepScaleR AR configuration and its multi-mode controller wiring."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from block_diffusion import training
from block_diffusion.generation.validation import (
    build_validation_config,
    validation_variants,
)
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

PROJECT = Path(__file__).resolve().parents[2]
RECIPE = (
    PROJECT
    / "configs/recipes/ar_grpo-deepscaler-3b-4n8g-megatron-vllm-async-dualval-long.yaml"
)


def load():
    register_omegaconf_resolvers()
    return load_config(RECIPE)


def test_deepscaler_recipe_has_causal_rollouts_and_two_validation_modes():
    config = load()
    assert config.data.train.dataset_name == "DeepScaler"
    assert "split_validation_size" not in config.data.train
    assert config.data.validation.dataset_name == "AIME2024"
    assert config.data.validation.repeat == 16
    assert config.grpo.max_val_samples % config.grpo.val_batch_size == 0
    assert config.policy.megatron_cfg.optimizer.lr == 1e-6
    assert config.policy.megatron_cfg.activation_checkpointing is False
    assert config.policy.megatron_cfg.context_parallel_size == 1
    assert config.grpo.async_grpo.enabled
    assert config.loss_fn.use_importance_sampling_correction
    rollout = OmegaConf.to_container(config.policy.generation, resolve=True)
    assert rollout["vllm_kwargs"]["diffusion_config"] is None
    assert rollout["vllm_kwargs"]["hf_overrides"]["architectures"] == [
        "NemotronLabsDiffusionForCausalLM"
    ]
    variants = validation_variants(rollout)
    assert set(variants) == {"ar", "diffusion_conf09"}
    for name, variant in variants.items():
        generation = build_validation_config(rollout, name, variant)
        diffusion = generation["vllm_kwargs"]["diffusion_config"]
        extra = 0 if diffusion is None else diffusion["canvas_length"]
        assert (
            config.data.max_input_seq_length + generation["max_new_tokens"] + extra
            <= generation["vllm_cfg"]["max_model_len"]
        )
    train_gpus = (
        config.cluster.num_nodes - rollout["colocated"]["resources"]["num_nodes"]
    ) * config.cluster.gpus_per_node
    assert train_gpus == 16
    assert config.policy.train_global_batch_size == 1024
    assert (
        config.policy.train_global_batch_size
        % (train_gpus * config.policy.train_micro_batch_size)
        == 0
    )


@pytest.mark.parametrize("algorithm", ["ar", "just", "trace", "just_eval"])
@pytest.mark.parametrize("asynchronous", [True, False])
@pytest.mark.parametrize("fail_training", [False, True])
def test_shared_driver_wires_dual_validation_and_training(
    monkeypatch, tmp_path, algorithm, asynchronous, fail_training
):
    spec = importlib.util.spec_from_file_location(
        "ar_recipe_driver", PROJECT.parent / "ar_grpo/run_ar_grpo.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    config = load()
    if algorithm == "ar":
        run = lambda cfg: driver.run(cfg, generation_python="/vllm/python")
    else:
        if algorithm in ("just", "just_eval"):
            from run_just_grpo import run

            config = load_config(
                PROJECT
                / "configs/recipes/just_grpo-deepscaler-3b-10n8g-megatron-vllm-fast-async-dualval-long.yaml"
            )
            config.just_grpo.generation_python = "/vllm/python"
        else:
            from run_trace_grpo import run

            config = load_config(
                PROJECT.parent
                / "trace/configs/recipes/trace_grpo-deepscaler-3b-4n8g-megatron-vllm-entropy-temp06-async-dualval-long.yaml"
            )
            config.trace_grpo.generation_python = "/vllm/python"
        if not asynchronous:
            config.cluster.num_nodes = 4
    if algorithm == "just_eval":
        entrypoint_run = run
        run = lambda cfg: entrypoint_run(cfg, evaluation_only=True)
    config.logger.log_dir = str(tmp_path)
    config.grpo.async_grpo.enabled = asynchronous
    config.policy.generation.colocated.enabled = not asynchronous
    config.policy.generation.vllm_cfg.async_engine = asynchronous
    config.policy.generation.worker_extension_cls_fqn = (
        "block_diffusion.generation.reference_vllm.ReferenceVllmAsyncWorker"
        if asynchronous
        else "block_diffusion.generation.reference_vllm.ReferenceVllmWorker"
    )
    validation = Mock(return_value=({"accuracy": 0.8}, {"total_validation_time": 1.0}))
    if algorithm == "just_eval" and fail_training:
        validation.side_effect = RuntimeError("training failed")
    monkeypatch.setattr(training, "MultiModeValidation", Mock(return_value=validation))
    policy, generation, tokenizer = Mock(), Mock(), Mock()
    train_cluster, inference_cluster = Mock(), Mock()
    train_data, val_data = [Mock()], [Mock()]
    environments, val_environments = {"DeepScaler": Mock()}, {"AIME2024": Mock()}
    logger, checkpointer = Mock(wandb_logger=None), MagicMock()
    grpo = ModuleType("nemo_rl.algorithms.grpo")
    grpo.MasterConfig = lambda **raw: SimpleNamespace(
        **{**raw, "grpo": OmegaConf.create(raw["grpo"])}
    )

    def setup(master, *args, **kwargs):
        return (
            policy,
            generation,
            None,
            (train_cluster, inference_cluster),
            train_data,
            val_data,
            Mock(),
            logger,
            checkpointer,
            Mock(),
            master,
            None,
            None,
        )

    grpo.setup = Mock(side_effect=setup)
    grpo.grpo_train, grpo.async_grpo_train = Mock(), Mock()
    trainer = grpo.async_grpo_train if asynchronous else grpo.grpo_train
    if fail_training:
        trainer.side_effect = RuntimeError("training failed")
    monkeypatch.setitem(sys.modules, grpo.__name__, grpo)
    utils = ModuleType("nemo_rl.algorithms.utils")
    utils.get_tokenizer = Mock(return_value=tokenizer)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    data = ModuleType("nemo_rl.data.utils")
    data.setup_response_data = Mock(
        return_value=(train_data, val_data, environments, val_environments)
    )
    monkeypatch.setitem(sys.modules, data.__name__, data)
    from nemo_rl.distributed import (
        ray_actor_environment_registry as registry,
    )
    from nemo_rl.distributed import (
        virtual_cluster,
    )
    from nemo_rl.environments import utils as env_utils
    from nemo_rl.models import generation as generation_module

    monkeypatch.setattr(registry, "ACTOR_ENVIRONMENT_REGISTRY", {})
    monkeypatch.setattr(registry, "get_actor_python_env", lambda _: sys.executable)
    monkeypatch.setattr(virtual_cluster, "init_ray", Mock())
    monkeypatch.setattr(training.ray, "shutdown", Mock())
    cleanup = Mock()
    monkeypatch.setattr(env_utils, "shutdown_environments", cleanup)
    monkeypatch.setattr(
        generation_module, "configure_generation_config", lambda config, _: config
    )
    if fail_training:
        with pytest.raises(RuntimeError, match="training failed"):
            run(config)
    else:
        run(config)
    assert (
        data.setup_response_data.call_args.args[1]["train"]["dataset_name"]
        == "DeepScaler"
    )
    assert (
        grpo.setup.call_args.kwargs["vllm_generation_factory"]
        == validation.create_generation
    )
    assert grpo.setup.call_args.kwargs["additional_colocated_worker_groups"] == 2
    validation.initialize.assert_called_once_with(
        policy,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
        refit_buffer_size_gb=1.0,
    )
    if algorithm == "just_eval":
        trainer.assert_not_called()
        validation.assert_called_once()
        assert validation.call_args.args[:4] == (
            generation,
            val_data,
            tokenizer,
            val_environments,
        )
        generation.prepare_for_generation.assert_called_once()
        generation.finish_generation.assert_called_once()
    else:
        assert trainer.call_args.kwargs["shift_labels"] is (algorithm == "ar")
        assert ("diffusion_config" in trainer.call_args.args[11].policy) is (
            algorithm != "ar"
        )
        if algorithm == "trace":
            assert (
                trainer.call_args.kwargs[
                    "prepare_training_data_fn"
                ].__self__.__class__.__name__
                == "TraceGRPO"
            )
        else:
            assert "prepare_training_data_fn" not in trainer.call_args.kwargs
        assert trainer.call_args.kwargs["validation_fn"] is validation
        if asynchronous:
            assert trainer.call_args.kwargs["max_trajectory_age_steps"] == 1
    validation.shutdown.assert_called_once()
    generation.shutdown.assert_called_once()
    cleanup.assert_called_once_with(environments, val_environments)
    logger.finish.assert_called_once()
    training.ray.shutdown.assert_called_once()


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"policy": {"generation": {"vllm_kwargs": {"diffusion_config": {}}}}},
            "causal rollout",
        ),
        (
            {"policy": {"worker_extension_cls_fqn": "other.Worker"}},
            "causal policy worker",
        ),
        (
            {"policy": {"generation": {"vllm_val_dllm_variants": None}}},
            "validation variants",
        ),
        (
            {"policy": {"generation": {"colocated": {"enabled": True}}}},
            "dedicated generation GPUs",
        ),
    ],
)
def test_ar_rejects_incompatible_modes_before_starting_training(
    monkeypatch, overrides, message
):
    spec = importlib.util.spec_from_file_location(
        "ar_recipe_driver", PROJECT.parent / "ar_grpo/run_ar_grpo.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    start = Mock()
    monkeypatch.setattr(training, "run", start)
    config = OmegaConf.merge(load(), overrides)
    with pytest.raises(ValueError, match=message):
        driver.run(config, generation_python="/vllm/python")
    start.assert_not_called()


def test_ar_cli_uses_native_overrides_and_generation_environment(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "ar_recipe_driver", PROJECT.parent / "ar_grpo/run_ar_grpo.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    start = Mock()
    monkeypatch.setattr(training, "run", start)
    monkeypatch.setenv("NRL_VLLM_PY_EXECUTABLE", "/vllm/python")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ar_grpo.py",
            "--config",
            str(RECIPE),
            "policy.model_name=/test/checkpoint",
            f"logger.log_dir={tmp_path}",
        ],
    )
    driver.main()
    config = start.call_args.args[0]
    assert config.policy.model_name == "/test/checkpoint"
    assert config.policy.tokenizer.name == "/test/checkpoint"
    assert config.logger.log_dir == str(tmp_path)
    assert start.call_args.kwargs == {"generation_python": "/vllm/python"}


@pytest.mark.parametrize(
    "worker",
    [
        "ar_grpo.policy.ARModeForMultiModeMegatronPolicy",
        "ar_grpo.positive_diffusion_policy.PositiveDiffusionARPolicy",
    ],
)
def test_ar_cli_dispatches_single_controller_with_actor_environments(
    monkeypatch: pytest.MonkeyPatch, worker: str
) -> None:
    spec = importlib.util.spec_from_file_location(
        "ar_recipe_driver", PROJECT.parent / "ar_grpo/run_ar_grpo.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    registry = ModuleType("nemo_rl.distributed.ray_actor_environment_registry")
    registry.ACTOR_ENVIRONMENT_REGISTRY = {}
    registry.get_actor_python_env = Mock(return_value="/policy/python")
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setenv("NRL_VLLM_PY_EXECUTABLE", "/vllm/python")
    monkeypatch.setenv("NRL_GYM_PY_EXECUTABLE", "/gym/python")
    start = Mock()
    monkeypatch.setattr(driver.runpy, "run_path", start)
    legacy = Mock()
    monkeypatch.setattr(training, "run", legacy)
    repo = PROJECT.parents[1]
    args = [
        "run_ar_grpo.py",
        "--config",
        str(
            repo
            / "examples/configs/ar_grpo_superv3_exp544_step100_helpful_128k_ready_first_tp2_cp8_300steps.yaml"
        ),
        "grpo.max_num_steps=9",
        f"policy.worker_extension_cls_fqn={worker}",
    ]
    monkeypatch.setattr(sys, "argv", args.copy())
    driver.main()
    start.assert_called_once_with(
        str(repo / "examples/run_grpo_single_controller.py"), run_name="__main__"
    )
    assert sys.argv == args
    assert registry.ACTOR_ENVIRONMENT_REGISTRY == {
        worker: "/policy/python",
        "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": "/vllm/python",
        "nemo_rl.environments.nemo_gym.NemoGym": "/gym/python",
    }
    registry.get_actor_python_env.assert_called_once_with(
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    )
    legacy.assert_not_called()


def test_moved_ar_cli_keeps_the_default_legacy_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "ar_recipe_driver", PROJECT.parent / "ar_grpo/run_ar_grpo.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    start = Mock()
    monkeypatch.setattr(training, "run", start)
    monkeypatch.setenv("NRL_VLLM_PY_EXECUTABLE", "/vllm/python")
    monkeypatch.setattr(sys, "argv", ["run_ar_grpo.py"])
    driver.main()
    assert start.call_count == 1
    config = start.call_args.args[0]
    assert config.data.train.dataset_name == "DeepScaler"
    assert config.grpo.async_grpo.enabled
