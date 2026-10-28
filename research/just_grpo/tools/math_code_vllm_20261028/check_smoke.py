"""Require successful training, small generation KL, and complete validation sweeps."""

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("math", "code"), required=True)
    parser.add_argument("--kl-limit", type=float, required=True)
    args = parser.parse_args()
    accounting = subprocess.check_output(
        [
            "sacct",
            "-j",
            args.job_id,
            "--noheader",
            "--parsable2",
            "--format=JobID,State,ExitCode",
        ],
        text=True,
    )
    record = next(
        line.split("|")
        for line in accounting.splitlines()
        if line.startswith(args.job_id + "|")
    )
    if record[1:3] != ["COMPLETED", "0:0"]:
        raise RuntimeError(f"Smoke job did not complete successfully: {record}")
    config = OmegaConf.load(args.output / "smoke-config.yaml")
    assert config.just_grpo.max_generation_kl == args.kl_limit
    histories = []
    for directory in sorted(
        {p.parent for p in args.output.rglob("events.out.tfevents.*")}
    ):
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
        for tag in events.Tags()["scalars"]:
            if re.search(
                r"(?:^|[/_])gen(?:eration)?_kl_error$", tag.lower().replace(" ", "_")
            ):
                histories.append((tag, events.Scalars(tag)))
    if len(histories) != 1:
        raise RuntimeError(
            f"Expected one training KL history, found {[tag for tag, _ in histories]}"
        )
    tag, history = histories[0]
    kl = {event.step: event.value for event in history}
    if len(kl) != 2 or any(
        not math.isfinite(v) or abs(v) >= args.kl_limit for v in kl.values()
    ):
        raise RuntimeError(
            f"Smoke requires two small, finite generation KL measurements: {kl}"
        )
    step = max(kl)
    benchmarks = ["humaneval", "mbpp"] if args.task == "code" else [""]
    samples = {}
    for benchmark in benchmarks:
        output = args.output / benchmark
        for mode in ("", "threshold09_t0", "threshold08_t0", "threshold07_t0"):
            suffix = f"_{mode}" if mode else ""
            stats = json.loads(
                (output / f"decode_stats_step{step}{suffix}.json").read_text()
            )
            counts = stats["forward_passes"]
            if len(counts) != 16 or stats["metrics"]["num_samples"] != 16:
                raise RuntimeError(f"Incomplete validation for {benchmark}/{mode}")
            if any(n <= 0 or n > 256 for n in counts):
                raise RuntimeError(f"Invalid NFE counts for {benchmark}/{mode}")
            if not mode and counts != [256] * 16:
                raise RuntimeError("Fixed-one validation must use 256 forward passes")
        assert (output / f"accuracy_vs_nfe_step{step}.png").is_file()
        samples[benchmark or "math500"] = 16
    report = {
        "passed": True,
        "job_id": args.job_id,
        "kl_limit": args.kl_limit,
        "training_kl_tag": tag,
        "training_kl": kl,
        "max_training_kl": max(kl.values()),
        "validation_samples_per_mode": samples,
        "final_step": step,
    }
    (args.output / "smoke_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
