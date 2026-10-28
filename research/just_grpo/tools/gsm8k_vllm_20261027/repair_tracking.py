# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild training-step W&B history from finalized GSM8K run artifacts."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot  # noqa: F401 - required by W&B figure conversion
import wandb
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.util import tensor_util

from block_diffusion.generation.confidence_sweep import SweepPoint, make_figure

ROOT = Path(
    "/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/llada_gsm8k_vllm_20261027"
)
MODES = {
    "fixed1_t1": None,
    "threshold09_t1": 0.9,
    "threshold08_t1": 0.8,
    "threshold07_t1": 0.7,
}
COLUMNS = [
    "training_step",
    "variant",
    "confidence_threshold",
    "mean_denoising_nfe",
    "accuracy",
    "num_samples",
]


def prepare(variant):
    run_dir = ROOT / variant
    output = ROOT / "tracking_repair" / variant
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = [
        (int(p.parent.name[5:]), p)
        for p in (run_dir / "checkpoints").glob("step_*/training_info.json")
    ]
    _, latest = max(checkpoints)
    last_step = json.loads(latest.read_text())["total_steps"]
    rows = {}
    for file in sorted((run_dir / "tensorboard").glob("events.out.tfevents.*")):
        for event in LegacyEventFileLoader(str(file)).Load():
            if event.step > last_step:
                continue
            for value in event.summary.value:
                if not (value.HasField("simple_value") or value.HasField("tensor")):
                    continue
                if value.tag.startswith(
                    ("train/", "validation/", "timing/train/", "timing/validation/")
                ):
                    scalar = (
                        value.simple_value
                        if value.HasField("simple_value")
                        else float(tensor_util.make_ndarray(value.tensor))
                    )
                    rows.setdefault(event.step, {})[value.tag] = scalar
    assert all("train/gen_kl_error" in rows[step] for step in range(1, last_step + 1))
    points_by_step = {}
    for file in sorted(run_dir.glob("decode_stats_step*.json")):
        stem = file.stem.removeprefix("decode_stats_step")
        if not stem.isdigit() or int(stem) > last_step:
            continue
        step = int(stem)
        points = []
        for index, (mode, threshold) in enumerate(MODES.items()):
            suffix = "" if index == 0 else "_" + mode
            stats = json.loads(
                (run_dir / f"decode_stats_step{step}{suffix}.json").read_text()
            )
            points.append(
                SweepPoint(
                    threshold,
                    stats["metrics"]["mean_forward_passes"],
                    stats["metrics"]["accuracy"],
                    len(stats["forward_passes"]),
                    mode,
                )
            )
        figure = make_figure(
            {"Current policy": points},
            step=step,
            nfe_description="NFE counts full-canvas model forward passes.",
        )
        figure.savefig(
            output / f"accuracy_vs_nfe_step{step}.png", dpi=160, bbox_inches="tight"
        )
        points_by_step[step] = points
    assert points_by_step and rows
    receipt = {
        "variant": variant,
        "checkpoint_step": last_step,
        "scalar_steps": sorted(rows),
        "plot_steps": sorted(points_by_step),
        "old_run": (run_dir / "wandb_url.txt").read_text().strip(),
    }
    (output / "prepared.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return rows, points_by_step, receipt, output


def publish(variant, rows, points, receipt, output):
    receipt_file = output / "published.json"
    if receipt_file.exists():
        raise RuntimeError(f"Already published: {receipt_file}")
    os.environ["WANDB_API_KEY"] = (
        Path("/home/snorouzi/wandb_api_key.txt").read_text().strip()
    )
    old = wandb.Api().run(receipt["old_run"].removeprefix("https://wandb.ai/"))
    config = dict(old.config)
    config["just_grpo"]["max_generation_kl"] = float("inf")
    config["logger"]["wandb"]["training_step_history"] = True
    config["history_repaired_from"] = receipt["old_run"]
    previous = output / "publishing.json"
    completed_steps = set()
    if previous.exists():
        prior = json.loads(previous.read_text())
        assert prior["old_run"] == receipt["old_run"]
        run_id = prior["new_run"].rsplit("/", 1)[-1]
        resumed = wandb.Api().run(prior["new_run"].removeprefix("https://wandb.ai/"))
        completed_steps = {
            int(row["nemo_rl/step"])
            for row in resumed.scan_history(keys=["_step", "nemo_rl/step"])
        }
    else:
        run_id = wandb.util.generate_id()
    run = wandb.init(
        entity=old.entity,
        project=old.project,
        id=run_id,
        name=old.name + "_stepfix",
        resume="must" if previous.exists() else "never",
        config=config,
        dir=str(output),
        tags=["training-step-history", "kl-limit-disabled"],
    )
    receipt["new_run"] = run.url
    (output / "publishing.json").write_text(json.dumps(receipt, indent=2) + "\n")
    run.define_metric("nemo_rl/step", hidden=True)
    for name in sorted({key for row in rows.values() for key in row}):
        run.define_metric(name, step_metric="nemo_rl/step")
    for name in ("validation/accuracy_vs_nfe", "validation/accuracy_vs_nfe_data"):
        run.define_metric(name, step_metric="nemo_rl/step")
    for step, metrics in sorted(rows.items()):
        if step in completed_steps:
            continue
        payload = {**metrics, "nemo_rl/step": step}
        if step in points:
            payload["validation/accuracy_vs_nfe"] = make_figure(
                {"Current policy": points[step]},
                step=step,
                nfe_description="NFE counts full-canvas model forward passes.",
            )
            payload["validation/accuracy_vs_nfe_data"] = wandb.Table(
                columns=COLUMNS,
                data=[
                    [
                        step,
                        p.variant,
                        p.threshold,
                        p.mean_nfe,
                        p.accuracy,
                        p.num_samples,
                    ]
                    for p in points[step]
                ],
            )
        run.log(payload, step=step)
    run.summary["history_repaired_from"] = receipt["old_run"]
    run.summary["checkpoint_step_at_repair"] = receipt["checkpoint_step"]
    run.finish()
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    for variant in ("plain", "diffaux"):
        rows, points, receipt, output = prepare(variant)
        print(json.dumps(receipt), flush=True)
        if args.publish:
            publish(variant, rows, points, receipt, output)
