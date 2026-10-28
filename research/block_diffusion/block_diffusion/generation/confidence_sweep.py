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
"""Measured accuracy versus mean request-denoising NFE for confidence sweeps."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweepPoint:
    threshold: float | None
    mean_nfe: float
    accuracy: float
    num_samples: int
    variant: str


def sweep_modes(variants: Mapping[str, dict]) -> dict[str, float | None]:
    """Select threshold sweeps and fixed one-token confidence decoding."""
    modes = {}
    for name, variant in variants.items():
        dc = variant["vllm_kwargs"]["diffusion_config"]
        if dc is None:
            continue
        if dc["selection_policy"] == "confidence_threshold":
            modes[name] = dc["confidence_threshold"]
        elif (
            dc["selection_policy"] == "low_confidence"
            and dc["canvas_length"] == dc["max_denoising_steps"]
        ):
            # The even transfer schedule reveals ceil(masks / remaining_steps).
            # Equal block length and step budget therefore reveal one token.
            modes[name] = None
    return modes


def sweep_points(
    metrics: Mapping[str, float], modes: Mapping[str, float | None]
) -> list[SweepPoint]:
    points = []
    for mode, threshold in modes.items():
        count = metrics[f"num_samples/{mode}"]
        total_nfe = metrics[f"vllm/denoising_nfe/{mode}"]
        accuracy = metrics[f"accuracy/{mode}"]
        if not all(math.isfinite(v) for v in (count, total_nfe, accuracy)) or (
            threshold is not None and not math.isfinite(threshold)
        ):
            raise ValueError(f"Non-finite sweep measurements for {mode}")
        if (
            count <= 0
            or int(count) != count
            or total_nfe <= 0
            or not 0 <= accuracy <= 1
        ):
            raise ValueError(f"Invalid sweep measurements for {mode}")
        points.append(
            SweepPoint(threshold, total_nfe / count, accuracy, int(count), mode)
        )
    if len({p.num_samples for p in points}) > 1:
        raise ValueError("Decoding variants must evaluate the same number of responses")
    return points


def make_figure(
    series: Mapping[str, Sequence[SweepPoint]],
    *,
    step: int,
    nfe_description: str = "NFE excludes prefill and block KV commits.",
):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.2, 4.8), layout="constrained")
    ax = figure.subplots()
    for label, points in series.items():
        ordered = sorted(points, key=lambda p: p.mean_nfe)
        (line,) = ax.plot(
            [p.mean_nfe for p in ordered],
            [100 * p.accuracy for p in ordered],
            marker="o",
            linewidth=1.6,
            markersize=6,
            label=label,
        )
        for index, p in enumerate(ordered):
            endpoint = index == len(ordered) - 1
            ax.annotate(
                "1 token/forward" if p.threshold is None else f"{p.threshold:.1f}",
                (p.mean_nfe, 100 * p.accuracy),
                xytext=(
                    -5 if endpoint or index % 2 else 5,
                    7 if endpoint or index % 2 else -14,
                ),
                ha="right" if endpoint or index % 2 else "left",
                textcoords="offset points",
                fontsize=9,
                color=line.get_color(),
            )
    ax.set(
        xlabel="Mean denoising NFE per response",
        ylabel="Accuracy (%)",
        title=f"Validation accuracy vs. NFE · step {step}",
    )
    ax.set_ylim(-1, 101)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    if len(series) > 1:
        ax.legend(frameon=False)
    figure.text(
        0.5,
        -0.025,
        f"Labels: confidence threshold or 1 token/forward. {nfe_description}",
        ha="center",
        fontsize=8,
    )
    return figure


def log_sweep(
    points: Sequence[SweepPoint],
    *,
    step: int,
    output: Path,
    logger: Any,
    nfe_description: str,
) -> None:
    """Save a measured comparison and publish it at the trainer's global step."""
    figure = make_figure(
        {"Current policy": points}, step=step, nfe_description=nfe_description
    )
    figure.savefig(
        output / f"accuracy_vs_nfe_step{step}.png", dpi=160, bbox_inches="tight"
    )
    if logger is not None:
        logger.log_table(
            [
                "training_step",
                "variant",
                "confidence_threshold",
                "mean_denoising_nfe",
                "accuracy",
                "num_samples",
            ],
            [
                [step, p.variant, p.threshold, p.mean_nfe, p.accuracy, p.num_samples]
                for p in points
            ],
            step,
            "validation/accuracy_vs_nfe_data",
        )
        logger.log_plot(figure, step, "validation/accuracy_vs_nfe")
