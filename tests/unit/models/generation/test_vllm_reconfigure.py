# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Regression tests for switching diffusion policies on a live sampler."""

import math
from types import SimpleNamespace

import pytest


@pytest.mark.vllm
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_entropy_validation_policy_round_trip(temperature):
    from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension

    worker = VllmInternalWorkerExtension()
    sampler = SimpleNamespace(
        leftmost=False,
        random_mode=False,
        entropy_mode=True,
        threshold_mode=False,
        temperature=temperature,
        log_threshold=math.log(0.9),
        diffusion_states=SimpleNamespace(max_denoising_steps=8),
    )
    worker.model_runner = SimpleNamespace(sampler=sampler)
    previous = worker.reconfigure_dllm(
        {
            "selection_policy": "confidence_threshold",
            "max_denoising_steps": 16,
            "temperature": 1.0,
        }
    )
    assert previous == {
        "selection_policy": "entropy",
        "max_denoising_steps": 8,
        "temperature": temperature,
    }
    assert sampler.temperature == 1.0
    assert sampler.threshold_mode and not sampler.entropy_mode
    assert sampler.diffusion_states.max_denoising_steps == 16

    worker.reconfigure_dllm(previous)
    assert sampler.entropy_mode and not sampler.threshold_mode
    assert sampler.diffusion_states.max_denoising_steps == 8
    assert sampler.temperature == temperature
    for policy, flag in (
        ("low_confidence", None),
        ("random", "random_mode"),
        ("entropy", "entropy_mode"),
        ("leftmost", "leftmost"),
        ("confidence_threshold", "threshold_mode"),
    ):
        worker.reconfigure_dllm({"selection_policy": policy})
        assert worker._read_selection_policy(sampler) == policy
        assert sum(
            getattr(sampler, name)
            for name in ("leftmost", "random_mode", "entropy_mode", "threshold_mode")
        ) == int(flag is not None)
