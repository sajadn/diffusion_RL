"""The campaign may spend GPU allocations only after numerical smoke checks pass."""

import json
import runpy
import sys
from pathlib import Path

import pytest
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


@pytest.mark.parametrize(
    "case",
    [
        "math",
        "code",
        "failed_job",
        "large_kl",
        "nan_kl",
        "one_update",
        "missing_mbpp",
        "bad_nfe",
    ],
)
def test_smoke_promotion_gate(tmp_path, monkeypatch, case):
    task = "math" if case == "math" else "code"
    status = "FAILED|1:0" if case == "failed_job" else "COMPLETED|0:0"
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: f"123|{status}|\n")
    (tmp_path / "smoke-config.yaml").write_text(
        "just_grpo:\n  max_generation_kl: 0.001\n"
    )
    writer = EventFileWriter(str(tmp_path / "tensorboard"))
    for step in (1,) if case == "one_update" else (1, 2):
        value = (
            0.002
            if case == "large_kl"
            else float("nan")
            if case == "nan_kl"
            else 0.0002
        )
        writer.add_event(
            Event(
                step=step,
                summary=Summary(
                    value=[Summary.Value(tag="train/gen_kl_error", simple_value=value)]
                ),
            )
        )
    writer.close()
    for benchmark in [""] if task == "math" else ["humaneval", "mbpp"]:
        if case == "missing_mbpp" and benchmark == "mbpp":
            continue
        output = tmp_path / benchmark
        output.mkdir(exist_ok=True)
        for mode in ("", "threshold09_t0", "threshold08_t0", "threshold07_t0"):
            suffix = f"_{mode}" if mode else ""
            count = 0 if case == "bad_nfe" else 256 if not mode else 100
            (output / f"decode_stats_step2{suffix}.json").write_text(
                json.dumps(
                    {"forward_passes": [count] * 16, "metrics": {"num_samples": 16}}
                )
            )
        (output / "accuracy_vs_nfe_step2.png").touch()
    script = Path(__file__).parents[2] / "tools/math_code_vllm_20261028/check_smoke.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--job-id",
            "123",
            "--output",
            str(tmp_path),
            "--task",
            task,
            "--kl-limit",
            "0.001",
        ],
    )
    if case in ("math", "code"):
        runpy.run_path(str(script), run_name="__main__")
        assert json.loads((tmp_path / "smoke_check.json").read_text())["passed"]
    else:
        with pytest.raises((RuntimeError, FileNotFoundError)):
            runpy.run_path(str(script), run_name="__main__")
        assert not (tmp_path / "smoke_check.json").exists()
