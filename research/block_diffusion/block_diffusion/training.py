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
"""Run AR, JustGRPO, and Trace through NeMo-RL's upstream GRPO controller."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import ray
from omegaconf import DictConfig, OmegaConf

from block_diffusion.config import DiffusionExperimentConfig
from block_diffusion.generation.validation import (
    MultiModeValidation,
    validation_variants,
)
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

PrepareTrainingData = Callable[
    [BatchedDataDict[Any], list[LLMMessageLogType], int], None
]


def run(
    config: DictConfig,
    *,
    diffusion: DiffusionExperimentConfig | None = None,
    evaluation_only: bool = False,
    generation_python: str | None = None,
    generation_factory: Callable[..., Any] | None = None,
    validation_fn: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None = None,
    validation_fn_factory: Callable[..., Callable[..., Any]] | None = None,
    prepare_training_data_factory: Callable[[list[int]], PrepareTrainingData]
    | None = None,
) -> None:
    """Configure research adapters, then delegate setup and training upstream.

    Args:
        config: Validated upstream and research experiment settings.
        evaluation_only: Run validation once without entering the training loop.
        diffusion: Parsed diffusion algorithm configuration; None selects causal AR.
        generation_factory: Optional dedicated adapter or shared Megatron adapter.
        validation_fn: Optional validator using the controller's validation timing.
        validation_fn_factory: Build a validator from the initialized policy and config.
        generation_python: Optional Python executable for the rollout and validation engines.
        prepare_training_data_factory: Optional algorithm callback factory, called
            with the tokenizer-resolved stop token IDs before training starts.
    """
    if validation_fn is not None and validation_fn_factory is not None:
        raise ValueError("Choose a validation function or a factory, not both")
    # The full controller imports optional training and tracking dependencies.
    from block_diffusion.environments.sudoku import (
        SudokuEnvironment,
        SudokuResponseDataset,
    )
    from block_diffusion.generation.megatron_generation import (
        MegatronDiffusionGeneration,
    )
    from nemo_rl.algorithms.grpo import (
        MasterConfig,
        async_grpo_train,
        grpo_train,
        setup,
    )
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_response_data
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
        get_actor_python_env,
    )
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.environments.utils import shutdown_environments
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.weight_sync.factory import create_weight_synchronizer

    output = Path(config.logger.log_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml")
    raw = OmegaConf.to_container(config, resolve=True)
    master = MasterConfig(**raw)
    worker = master.policy["worker_extension_cls_fqn"]
    ACTOR_ENVIRONMENT_REGISTRY[worker] = get_actor_python_env(
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    )
    # Copy before attaching the full research config to avoid a recursive dict.
    if diffusion is not None:
        master.policy = {**master.policy, "diffusion_config": raw}
    tokenizer = get_tokenizer(master.policy["tokenizer"])
    master.policy["generation"] = configure_generation_config(
        master.policy["generation"], tokenizer
    )
    master.policy["megatron_cfg"]["train_iters"] = master.grpo.max_num_steps
    if diffusion is None or diffusion.runtime == "reference_vllm":
        generation_worker = master.policy["generation"]["worker_extension_cls_fqn"]
        ACTOR_ENVIRONMENT_REGISTRY[generation_worker] = (
            generation_python
            or (diffusion.generation_python if diffusion is not None else None)
            or get_actor_python_env(
                "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
                if master.grpo.async_grpo.enabled
                else "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
            )
        )
    init_ray()
    if config.data.default.env_name == "sudoku6x6":
        train_data = SudokuResponseDataset(config, tokenizer)
        val_data = SudokuResponseDataset(config, tokenizer, validation=True)
        env = SudokuEnvironment.remote()
        environments = {config.data.default.env_name: env}
        val_environments = environments
    else:
        train_data, val_data, environments, val_environments = setup_response_data(
            tokenizer, master.data, master.env
        )
    variants = validation_variants(master.policy["generation"])
    validation = (
        MultiModeValidation(
            variants, colocated=master.policy["generation"]["colocated"]["enabled"]
        )
        if variants
        else None
    )
    if validation is not None and (
        validation_fn is not None or validation_fn_factory is not None
    ):
        raise ValueError("Custom validation cannot be combined with vLLM variants")
    setup_kwargs = (
        {
            "vllm_generation_factory": validation.create_generation,
            "additional_colocated_worker_groups": len(variants),
        }
        if validation is not None
        else {}
    )
    if generation_factory is not None and (
        diffusion is None or diffusion.runtime not in ("megatron", "megatron_llada")
    ):
        if validation is not None:
            raise ValueError(
                "Custom generation factories cannot use vLLM validation variants"
            )
        setup_kwargs["vllm_generation_factory"] = generation_factory
    logger = generation = None
    try:
        (
            policy,
            generation,
            _,
            clusters,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            state,
            master,
            _,
            _,
        ) = setup(master, tokenizer, train_data, val_data, **setup_kwargs)
        if validation is not None:
            validation.initialize(
                policy,
                train_cluster=clusters[0],
                inference_cluster=clusters[1],
                refit_buffer_size_gb=master.policy.get("refit_buffer_size_gb"),
            )
        if diffusion is not None and diffusion.runtime in (
            "megatron",
            "megatron_llada",
        ):
            # Colocated setup has not started an engine (HTTP serving is disabled).
            generation.shutdown()
            shared_generation_factory = (
                generation_factory or MegatronDiffusionGeneration
            )
            generation = shared_generation_factory(policy=policy, config=master)
            generation.weight_synchronizer = create_weight_synchronizer(
                policy=policy,
                generation=generation,
                generation_backend="megatron",
                colocated=True,
            )
            generation.weight_synchronizer.init_communicator()
        if validation_fn_factory is not None:
            validation_fn = validation_fn_factory(policy=policy, config=master)
        if evaluation_only:
            validator = validation if validation is not None else validation_fn
            if validator is None:
                raise ValueError("Evaluation-only execution requires a validator")
            generation.prepare_for_generation()
            try:
                metrics, timings = validator(
                    generation,
                    val_dataloader,
                    tokenizer,
                    val_environments,
                    step=0,
                    master_config=master,
                    logger=logger,
                )
                logger.log_metrics(metrics, 0, prefix="validation")
                logger.log_metrics(timings, 0, prefix="timing/validation")
            finally:
                generation.finish_generation()
            return
        trainer = async_grpo_train if master.grpo.async_grpo.enabled else grpo_train
        trainer_kwargs = (
            {
                "max_trajectory_age_steps": master.grpo.async_grpo.max_trajectory_age_steps
            }
            if master.grpo.async_grpo.enabled
            else {}
        )
        if validation is not None:
            trainer_kwargs["validation_fn"] = validation
        elif validation_fn is not None:
            trainer_kwargs["validation_fn"] = validation_fn
        if logger.wandb_logger is not None:
            (output / "wandb_url.txt").write_text(logger.wandb_logger.run.url + "\n")
        if prepare_training_data_factory is not None:
            trainer_kwargs["prepare_training_data_fn"] = prepare_training_data_factory(
                master.policy["generation"]["stop_token_ids"]
            )
        with checkpointer:
            trainer(
                policy,
                generation,
                dataloader,
                val_dataloader,
                tokenizer,
                loss_fn,
                environments,
                val_environments,
                logger,
                checkpointer,
                state,
                master,
                shift_labels=diffusion is None,
                **trainer_kwargs,
            )
    finally:
        shutdown_environments(environments, val_environments)
        if validation is not None:
            validation.shutdown()
        if generation is not None:
            generation.shutdown()
        if logger is not None:
            logger.finish()
        ray.shutdown()
