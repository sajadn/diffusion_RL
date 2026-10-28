# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation profiles and decoding statistics for Megatron LLaDA generation."""

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from block_diffusion.generation.confidence_sweep import SweepPoint, log_sweep


if TYPE_CHECKING:
    from just_grpo.generation.llada_megatron import LLaDAMegatronGeneration
    from nemo_rl.algorithms.grpo import MasterConfig


class _ValidationModeLogger:
    """Keep the native validation step while separating per-mode sample files."""

    def __init__(self, logger: Any, mode: str) -> None:
        self.logger = logger
        self.mode = mode

    def log_batched_dict_as_jsonl(self, data: Any, filename: str) -> None:
        path = Path(filename)
        self.logger.log_batched_dict_as_jsonl(
            data, f"{path.stem}_{self.mode}{path.suffix}"
        )


def validate_llada(
    policy_generation: "LLaDAMegatronGeneration",
    val_dataloader: Any,
    tokenizer: Any,
    val_task_to_env: Any,
    step: int,
    master_config: "MasterConfig",
    logger: Any = None,
    processor: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate named decoders on identical prompts, retaining primary metrics."""
    # Native validation imports optional training dependencies.
    from nemo_rl.algorithms.grpo import validate

    modes = getattr(master_config, "just_grpo", {}).get("validation_modes", {})
    if modes:
        if val_dataloader is None:
            raise ValueError("Multi-mode validation requires a validation dataloader")
        loader_state = copy.deepcopy(val_dataloader.state_dict())
        profiles = list(modes)
    else:
        profiles = [None]
    metrics, timings = {}, {}
    try:
        for index, name in enumerate(profiles):
            if modes:
                val_dataloader.load_state_dict(copy.deepcopy(loader_state))
            context = (
                policy_generation.validation_sampling(mode=name)
                if name is not None
                else policy_generation.validation_sampling()
            )
            with context:
                values, elapsed = validate(
                    policy_generation,
                    val_dataloader,
                    tokenizer,
                    val_task_to_env,
                    step=step,
                    master_config=master_config,
                    logger=(
                        _ValidationModeLogger(logger, name)
                        if index > 0 and logger is not None
                        else logger
                    ),
                    processor=processor,
                )

            counts = policy_generation.forward_counts
            if not counts:
                raise RuntimeError("LLaDA validation produced no decoding statistics")
            values["num_samples"] = len(counts)
            values["mean_forward_passes"] = sum(counts) / len(counts)
            values["max_forward_passes"] = max(counts)
            values["min_forward_passes"] = min(counts)
            # The whole canvas is filled, including tokens after the first EOS.
            if hasattr(master_config, "policy"):
                canvas = master_config.policy["generation"]["max_new_tokens"]
                values["tokens_per_forward"] = canvas / values["mean_forward_passes"]
            output = Path(master_config.logger["log_dir"])
            suffix = f"_{name}" if index else ""
            (output / f"decode_stats_step{step}{suffix}.json").write_text(
                json.dumps({"forward_passes": counts, "metrics": values}, indent=2)
            )
            if name is not None:
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
        if modes:
            val_dataloader.load_state_dict(loader_state)
    if modes:
        # Plot only complete comparisons, after restoring the training decoder/data.
        points = [
            SweepPoint(
                params["threshold"]
                if params["selection_policy"] == "confidence_threshold"
                else None,
                metrics[f"mean_forward_passes/{name}"],
                metrics[f"accuracy/{name}"],
                metrics[f"num_samples/{name}"],
                name,
            )
            for name, params in modes.items()
        ]
        log_sweep(
            points,
            step=step,
            output=output,
            logger=logger,
            nfe_description="NFE counts full-canvas model forward passes.",
        )
    return metrics, timings


def make_llada_policy_validator(*, policy: Any, config: Any) -> Any:
    """Validate on the live policy while the colocated rollout engine sleeps."""
    from just_grpo.generation.llada_megatron import LLaDAMegatronGeneration

    adapter = LLaDAMegatronGeneration(policy=policy, config=config)

    def validate_policy(generation: Any, *args: Any, **kwargs: Any) -> Any:
        # The native controller enters validation with the rollout engine awake.
        # Restore that lifecycle on exit, including when validation raises.
        if not generation.finish_generation():
            raise RuntimeError("Failed to sleep the rollout engine before validation")
        try:
            policy.prepare_for_lp_inference()
            return validate_llada(adapter, *args, **kwargs)
        finally:
            policy.offload_after_refit()
            if not generation.prepare_for_generation():
                raise RuntimeError(
                    "Failed to restore the rollout engine after validation"
                )

    return validate_policy
