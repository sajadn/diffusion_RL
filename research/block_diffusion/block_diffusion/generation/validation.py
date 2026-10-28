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
"""Named AR/diffusion validation passes on current Megatron policy weights."""

import copy
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from block_diffusion.generation.confidence_sweep import (
    log_sweep,
    sweep_modes,
    sweep_points,
)

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
    from nemo_rl.models.generation.interfaces import GenerationInterface
    from nemo_rl.models.policy.lm_policy import Policy


class ValidationEngineConfig(BaseModel, extra="forbid"):
    # Every engine profiles its own memory budget; do not inherit this silently.
    gpu_memory_utilization: float = Field(gt=0, lt=1)


class ValidationModelConfig(BaseModel, extra="forbid"):
    # These dictionaries are passed to the reference fork's external schemas.
    hf_overrides: dict[str, Any]
    diffusion_config: dict[str, Any] | None

    @model_validator(mode="after")
    def validate_mode(self) -> "ValidationModelConfig":
        architecture = (
            "NemotronLabsDiffusionForCausalLM"
            if self.diffusion_config is None
            else "NemotronLabsDiffusionModel"
        )
        llada = self.hf_overrides.get("architectures") == ["LLaDAModelLM"]
        if llada and self.diffusion_config is None:
            raise ValueError("LLaDA validation requires diffusion decoding")
        if not llada and self.hf_overrides.get("architectures") != [architecture]:
            raise ValueError(
                f"Validation mode requires architectures: [{architecture}]"
            )
        if self.diffusion_config is not None:
            required = {
                "canvas_length",
                "max_denoising_steps",
                "temperature",
                "selection_policy",
                "confidence_threshold",
            }
            if not required <= self.diffusion_config.keys():
                raise ValueError(
                    "Pin all validation diffusion decoding knobs explicitly"
                )
            if not llada and self.diffusion_config[
                "canvas_length"
            ] != self.hf_overrides.get("block_size"):
                raise ValueError(
                    "Validation canvas_length must match hf_overrides.block_size"
                )
        return self


class ValidationVariant(BaseModel, extra="forbid"):
    temperature: float = Field(ge=0, allow_inf_nan=False)
    vllm_cfg: ValidationEngineConfig
    vllm_kwargs: ValidationModelConfig

    @model_validator(mode="after")
    def validate_temperature(self) -> "ValidationVariant":
        diffusion = self.vllm_kwargs.diffusion_config
        if diffusion is not None and diffusion["temperature"] != self.temperature:
            raise ValueError("Validation engine and diffusion temperatures must agree")
        return self


def validation_variants(config: dict[str, Any]) -> dict[str, ValidationVariant]:
    """Parse the same named engine-override format used in diffusion_RL."""
    variants = config.get("vllm_val_dllm_variants")
    if variants is None:
        return {}
    if not variants:
        raise ValueError("vllm_val_dllm_variants must be nonempty or null")
    parsed = {}
    for name, overrides in variants.items():
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError(
                "Validation names must contain only letters, digits, '_' or '-'"
            )
        parsed[name] = ValidationVariant.model_validate(overrides)
    return parsed


def build_validation_config(
    rollout: dict[str, Any], name: str, variant: ValidationVariant
) -> dict[str, Any]:
    """Copy rollout engine settings, pin decoding, and isolate the refit socket."""
    merged = OmegaConf.to_container(
        OmegaConf.merge(rollout, variant.model_dump()), resolve=True
    )
    merged["vllm_val_dllm_variants"] = None
    merged["refit_namespace"] = f"vllm_val_{name}"
    merged["val_temperature"] = variant.temperature
    # Validation never supplies training trajectories to either algorithm.
    diffusion = merged["vllm_kwargs"]["diffusion_config"]
    if diffusion is not None:
        diffusion.update(
            return_entropy=False, return_reveal_steps=False, emit_full_blocks=False
        )
    return merged


class MultiModeValidation:
    """Own validation-only engines; the GRPO controller owns validation timing.

    Engines are constructed before rollout/policy workers, then sleep until
    their pass. Only one generation group holds GPU memory at a time. The
    first configured variant owns unsuffixed metrics and sample artifacts.
    """

    def __init__(
        self, variants: dict[str, ValidationVariant], *, colocated: bool
    ) -> None:
        self.variants = variants
        self.colocated = colocated
        self.generations: dict[str, GenerationInterface] = {}
        self.policy: Policy | None = None

    def create_generation(
        self, *, cluster: "RayVirtualCluster", config: dict[str, Any]
    ) -> "GenerationInterface":
        # Avoid loading the optional serving stack when validating YAML on CPU.
        from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration

        for name, variant in self.variants.items():
            group = VllmGeneration(
                cluster=cluster,
                config=build_validation_config(config, name, variant),
                name_prefix=f"vllm_val_{name}",
            )
            self.generations[name] = group
            if not group.sleep():
                raise RuntimeError(f"Could not sleep validation engine {name}")
        return VllmGeneration(cluster=cluster, config=config)

    def initialize(
        self,
        policy: "Policy",
        *,
        train_cluster: "RayVirtualCluster",
        inference_cluster: "RayVirtualCluster",
        refit_buffer_size_gb: float | int | None = None,
    ) -> None:
        # Weight synchronization depends on the full training runtime.
        from nemo_rl.weight_sync.factory import create_weight_synchronizer

        self.policy = policy
        for group in self.generations.values():
            group.weight_synchronizer = create_weight_synchronizer(
                policy=policy,
                generation=group,
                generation_backend="vllm",
                colocated=self.colocated,
                train_cluster=train_cluster,
                inference_cluster=inference_cluster,
                refit_buffer_size_gb=refit_buffer_size_gb,
            )
            group.weight_synchronizer.init_communicator()

    def __call__(
        self,
        policy_generation: "GenerationInterface",
        val_dataloader: Any,
        tokenizer: Any,
        val_task_to_env: Any,
        step: int,
        master_config: "MasterConfig",
        logger: Any = None,
        processor: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Reuse the controller's scoring, logging and rollout dispatch unchanged.
        from nemo_rl.algorithms.grpo import refit_policy_generation, validate

        if self.policy is None or not self.generations:
            raise RuntimeError("Validation engines must be initialized before training")
        if val_dataloader is None:
            raise ValueError("Multi-mode validation requires a validation dataloader")
        loader_state = copy.deepcopy(val_dataloader.state_dict())
        metrics, timings = {}, {}
        if not policy_generation.sleep():
            raise RuntimeError("Could not sleep rollout engines before validation")
        try:
            for index, (name, group) in enumerate(self.generations.items()):
                val_dataloader.load_state_dict(copy.deepcopy(loader_state))
                # A preceding refit offloads policy weights. Restore them before
                # streaming to the next group even when no optimizer step ran.
                if self.colocated:
                    self.policy.prepare_for_lp_inference()
                try:
                    if not self.colocated and not group.wake_up():
                        raise RuntimeError(f"Could not wake validation engine {name}")
                    refit_policy_generation(
                        self.policy, group, colocated_inference=self.colocated
                    )
                    config = master_config.model_copy(deep=True)
                    config.policy["generation"] = group.cfg
                    print(f"Validation decode: {name}", flush=True)
                    group.snapshot_step_metrics()
                    values, elapsed = validate(
                        group,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=step,
                        master_config=config,
                        logger=logger if index == 0 else None,
                        processor=processor,
                    )
                    values.update(
                        {
                            key: value
                            for key, value in group.get_step_metrics().items()
                            if key.startswith("vllm/denoising_")
                            or key.startswith("vllm/diffusion_")
                            or key == "vllm/tokens_per_denoising_nfe"
                        }
                    )
                    metrics.update(
                        {f"{key}/{name}": value for key, value in values.items()}
                    )
                    timings.update(
                        {f"{key}/{name}": value for key, value in elapsed.items()}
                    )
                    if index == 0:
                        metrics.update(values)
                    for key, value in elapsed.items():
                        timings[key] = timings.get(key, 0) + value
                finally:
                    if not group.sleep():
                        raise RuntimeError(f"Could not sleep validation engine {name}")
        finally:
            val_dataloader.load_state_dict(loader_state)
            if self.colocated:
                self.policy.offload_after_refit()
            if not policy_generation.wake_up():
                raise RuntimeError("Could not restore rollout engines after validation")
        if all(
            v.vllm_kwargs.hf_overrides.get("architectures") == ["LLaDAModelLM"]
            for v in self.variants.values()
        ):
            modes = sweep_modes({k: v.model_dump() for k, v in self.variants.items()})
            if modes:
                log_sweep(
                    sweep_points(metrics, modes),
                    step=step,
                    output=Path(master_config.logger["log_dir"]),
                    logger=logger,
                    nfe_description="NFE counts full-canvas denoising model forward passes.",
                )
        return metrics, timings

    def shutdown(self) -> None:
        """Release all validation engine groups, including partially built sets."""
        for group in self.generations.values():
            group.shutdown()
        self.generations.clear()
