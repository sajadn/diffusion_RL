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
"""Check NFE normalization and plotted coordinates against measured inputs."""

import pytest

from block_diffusion.generation.confidence_sweep import (
    make_figure,
    sweep_modes,
    sweep_points,
)


def measurements():
    values = {}
    for name, count, total, accuracy in [
        ("c9", 4, 48, 0.75),
        ("c8", 4, 32, 0.5),
        ("c7", 4, 20, 0.25),
    ]:
        values.update(
            {
                f"num_samples/{name}": count,
                f"vllm/denoising_nfe/{name}": total,
                f"accuracy/{name}": accuracy,
            }
        )
    return values


def test_coordinates_use_response_mean_and_percent_accuracy():
    points = sweep_points(measurements(), {"c9": 0.9, "c8": 0.8, "c7": 0.7})
    figure = make_figure({"test": points}, step=2)
    line = figure.axes[0].lines[0]
    assert list(line.get_xdata()) == [5, 8, 12]
    assert list(line.get_ydata()) == [25, 50, 75]
    assert [t.get_text() for t in figure.axes[0].texts] == ["0.7", "0.8", "0.9"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_samples/c7", 0),
        ("num_samples/c7", 3),
        ("vllm/denoising_nfe/c8", float("nan")),
        ("accuracy/c9", 1.5),
    ],
)
def test_invalid_or_mismatched_samples_are_rejected(key, value):
    values = measurements()
    values[key] = value
    with pytest.raises(ValueError):
        sweep_points(values, {"c9": 0.9, "c8": 0.8, "c7": 0.7})


def test_single_token_variant_is_a_fourth_measured_point():
    raw = {
        name: {
            "vllm_kwargs": {
                "diffusion_config": {
                    "selection_policy": "low_confidence"
                    if threshold is None
                    else "confidence_threshold",
                    "confidence_threshold": threshold,
                    "canvas_length": 32,
                    "max_denoising_steps": 32,
                }
            }
        }
        for name, threshold in [
            ("diffusion_conf09", 0.9),
            ("diffusion_conf08", 0.8),
            ("diffusion_conf07", 0.7),
            ("diffusion_single_token", None),
        ]
    }
    modes = sweep_modes(raw)
    assert modes == {
        "diffusion_conf09": 0.9,
        "diffusion_conf08": 0.8,
        "diffusion_conf07": 0.7,
        "diffusion_single_token": None,
    }
    values = {}
    for mode, nfe, acc in zip(modes, (12, 8, 5, 20), (0.75, 0.5, 0.25, 1.0)):
        values.update(
            {
                f"num_samples/{mode}": 4,
                f"vllm/denoising_nfe/{mode}": 4 * nfe,
                f"accuracy/{mode}": acc,
            }
        )
    points = sweep_points(values, modes)
    figure = make_figure({"test": points}, step=2)
    assert list(figure.axes[0].lines[0].get_xdata()) == [5, 8, 12, 20]
    assert list(figure.axes[0].lines[0].get_ydata()) == [25, 50, 75, 100]
    assert figure.axes[0].texts[-1].get_text() == "1 token/forward"
    assert points[-1].threshold is None
    assert points[-1].variant == "diffusion_single_token"
    # A schedule revealing multiple tokens must never be labelled one token.
    diffusion = raw["diffusion_single_token"]["vllm_kwargs"]["diffusion_config"]
    diffusion["max_denoising_steps"] = 8
    assert "diffusion_single_token" not in sweep_modes(raw)
