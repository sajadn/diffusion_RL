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
"""Configuration gates for opt-in corrected-confidence Trace experiments."""

import os
from typing import Any


def is_corrected_confidence_trace(policy_config: dict[str, Any]) -> bool:
    """Whether the active estimator explicitly enables the Trace correction."""
    estimator = policy_config.get("logprob_estimation") or {}
    return estimator.get("type") == "trace_grpo" and bool(
        estimator.get("confidence_transition_correction")
    )


def get_confidence_experiment_config(
    policy_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the experiment only for an explicitly corrected Trace estimator."""
    if not is_corrected_confidence_trace(policy_config):
        return None
    return policy_config["logprob_estimation"].get("confidence_experiment") or None


def clear_confidence_recording(generation_config: dict[str, Any] | None) -> None:
    """Remove internal capture flags from an inactive/validation generation config."""
    if generation_config is None:
        return
    decode = (generation_config.get("vllm_kwargs") or {}).get("diffusion_config")
    if decode is not None:
        decode.pop("record_confidence_experiment", None)
        decode.pop("record_confidence_audit", None)


def configure_confidence_recording(
    policy_config: dict[str, Any], *, audit_requested: bool
) -> None:
    """Derive sampler capture flags before generation workers are constructed.

    Directory environment variables supply paths, not experiment activation.
    Do not add custom vLLM fields to inactive configurations. Remove inherited
    capture flags, including validation overrides, when deriving the new state.
    """
    generation = policy_config.get("generation") or {}
    clear_confidence_recording(generation)
    overrides = generation.get("vllm_val_dllm_overrides")
    if overrides:
        clear_confidence_recording(overrides)
    for variant in (generation.get("vllm_val_dllm_variants") or {}).values():
        clear_confidence_recording(variant)

    if not is_corrected_confidence_trace(policy_config):
        return
    if generation.get("backend") != "vllm":
        return
    experiment = get_confidence_experiment_config(policy_config)
    if experiment and not os.environ.get("NRL_CONFIDENCE_EXPERIMENT_DIR"):
        raise ValueError(
            "An active confidence experiment requires NRL_CONFIDENCE_EXPERIMENT_DIR; "
            "launch it through tools/nemotron_diffusion/run_confidence_experiment.py."
        )
    decode = generation["vllm_kwargs"]["diffusion_config"]
    if experiment:
        decode["record_confidence_experiment"] = True
    if audit_requested:
        decode["record_confidence_audit"] = True


def start_confidence_round(policy_config: dict[str, Any], step: int) -> None:
    """Start recording only when the active Trace config requests an experiment."""
    if not get_confidence_experiment_config(policy_config):
        return
    if not os.environ.get("NRL_CONFIDENCE_EXPERIMENT_DIR"):
        raise ValueError(
            "An active confidence experiment requires NRL_CONFIDENCE_EXPERIMENT_DIR"
        )
    from confidence_experiment import start_round

    start_round(step)
