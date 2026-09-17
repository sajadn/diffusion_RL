#!/usr/bin/env python3
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
"""Submit, generate, and score the four supported DFW checkpoint evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from eval_datasets import load_benchmark_samples, subsample
from eval_grpo_checkpoint_validation import select_validation_shard
from tools.benchmark_eval.scoring import load_records

BENCHMARKS = ("ifbench", "livecodebench", "scicode", "aa_lcr")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def dataset_environment(config: dict) -> dict[str, str]:
    data = Path(config["runtime"]["data_root"])
    return {
        "IFBENCH_DATA_DIR": str(data / "ifbench"),
        "LIVECODEBENCH_DATA_DIR": str(data / "livecodebench"),
        "AA_LCR_DATA_DIR": str(data / "aa_lcr"),
    }


def sample_ids(records: list[dict], benchmark: str) -> list[str]:
    key = "problem_id" if benchmark == "scicode" else "source_id"
    ids = [str(record[key]) for record in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate IDs for {benchmark}")
    return ids


def selected_samples(config: dict, benchmark: str, settings: dict) -> list[dict]:
    if benchmark == "scicode":
        path = Path(config["runtime"]["data_root"]) / settings["dataset"]
        samples = load_records(path, "problem_id")
        return subsample(samples, settings["num_samples"], settings["seed"])
    os.environ.update(dataset_environment(config))
    return load_benchmark_samples(benchmark, settings["num_samples"], settings["seed"])


def validate_settings(settings: dict) -> None:
    for key in (
        "shards",
        "concurrent",
        "block_size",
        "max_steps",
        "context_length",
        "max_new_tokens",
    ):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["num_samples"] != -1 and settings["num_samples"] < 1:
        raise ValueError("num_samples must be -1 (all) or positive")
    if settings["algorithm"] not in ("FastDiffuser", "AR"):
        raise ValueError("This vLLM launcher supports FastDiffuser and AR")
    if settings["selection_policy"] != "confidence":
        raise ValueError("This launcher supports the confidence selection policy")
    if settings["top_k"] != -1 or not 0 < settings["top_p"] <= 1:
        raise ValueError("This suite requires top_k=-1 and 0 < top_p <= 1")
    if settings["temperature"] < 0 or not 0 < settings["memory_fraction"] < 1:
        raise ValueError("Invalid temperature or memory_fraction")
    if not 0 <= settings["thinking_budget"] < settings["max_new_tokens"]:
        raise ValueError("thinking_budget must be >= 0 and < max_new_tokens")
    if settings["max_new_tokens"] >= settings["context_length"]:
        raise ValueError("max_new_tokens must be smaller than context_length")
    if settings["algorithm"] != "AR" and (
        settings["top_p"] != 1 or settings["top_k"] != -1
    ):
        raise ValueError("Diffusion requests require top_p=1 and top_k=-1")
    if "judge" in settings and settings["judge"] != "heuristic":
        raise ValueError(
            "Suite scoring uses the explicitly labeled heuristic AA-LCR metric; see README for LLM judging"
        )


def make_manifest(args: argparse.Namespace) -> dict:
    import yaml

    config = yaml.safe_load(args.config.read_text())
    if args.account == "coreai_dlalgo_modelopt":
        raise ValueError("The modelopt account must not be used")
    if args.max_parallel is not None:
        config["cluster"]["max_parallel"] = args.max_parallel
    if config["cluster"]["max_parallel"] < 1:
        raise ValueError("max_parallel must be positive")
    for path in (args.model, args.tokenizer or args.model):
        if not path.is_dir():
            raise FileNotFoundError(path)
    runtime = config["runtime"]
    for key in (
        "python",
        "scoring_python",
        "vllm_python",
        "container",
        "scicode_container",
    ):
        if not Path(runtime[key]).is_file():
            raise FileNotFoundError(runtime[key])
    tasks, files = [], {str(args.config.resolve()): digest(args.config)}
    benchmarks = args.benchmarks or list(BENCHMARKS)
    if len(set(benchmarks)) != len(benchmarks):
        raise ValueError("Benchmark names must be unique")
    for benchmark in benchmarks:
        settings = config["common"] | config["benchmarks"][benchmark]
        for key in (
            "num_samples",
            "shards",
            "temperature",
            "algorithm",
            "max_new_tokens",
            "thinking_budget",
        ):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        validate_settings(settings)
        fixed_datasets = {
            "ifbench": "ifbench/test.jsonl",
            "livecodebench": "livecodebench/prompts_v6.jsonl",
            "aa_lcr": "aa_lcr/prompts.jsonl",
        }
        if (
            benchmark in fixed_datasets
            and settings["dataset"] != fixed_datasets[benchmark]
        ):
            raise ValueError(
                f"{benchmark} expects {fixed_datasets[benchmark]} under runtime.data_root"
            )
        samples = selected_samples(config, benchmark, settings)
        dataset_path = Path(runtime["data_root"]) / settings["dataset"]
        files[str(dataset_path)] = digest(dataset_path)
        if benchmark == "livecodebench":
            tests = Path(runtime["data_root"]) / settings["tests"]
            files[str(tests)] = digest(tests)
        if benchmark == "scicode":
            if not (dataset_path.parent / "test_data.h5").is_file():
                raise FileNotFoundError(dataset_path.parent / "test_data.h5")
        # Use one global validation batch so ceil-based shards have no gaps/overlap.
        for rank in range(settings["shards"]):
            shard = select_validation_shard(
                samples, len(samples), settings["shards"], rank
            )
            if not shard:
                raise ValueError(
                    f"Empty {benchmark} shard {rank}; reduce --shards for a smoke run"
                )
            tasks.append(
                {
                    "benchmark": benchmark,
                    "rank": rank,
                    "settings": settings,
                    "sample_count": len(samples),
                    "expected_ids": sample_ids([row for _, row in shard], benchmark),
                }
            )
    sources = [
        ROOT / "examples/eval_grpo_checkpoint_validation.py",
        ROOT / "examples/eval_datasets.py",
        ROOT / runtime["prompt_file"],
    ]
    for directory in (
        "benchmark_eval",
        "ifbench",
        "livecodebench",
        "aa_lcr",
        "scicode",
    ):
        sources.extend(
            path
            for path in (ROOT / "tools" / directory).iterdir()
            if path.suffix in (".py", ".sh")
        )
    for path in sources:
        files[str(path)] = digest(path)
    template_files = [
        path
        for name in ("chat_template.jinja", "tokenizer_config.json", "config.json")
        if (path := (args.tokenizer or args.model) / name).is_file()
    ]
    for path in template_files:
        files[str(path.resolve())] = digest(path)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": str(ROOT),
        "model": str(args.model.resolve()),
        "tokenizer": str((args.tokenizer or args.model).resolve()),
        "account": args.account,
        "outdir": str(args.outdir.resolve()),
        "config": config,
        "tasks": tasks,
        "sha256": files,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }


def verify_files(manifest: dict) -> None:
    for filename, expected in manifest["sha256"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(
                f"Input/code changed after submission: {filename}; create a new run"
            )


def command_string(argv: list) -> str:
    return shlex.join([str(value) for value in argv])


def submit(args: argparse.Namespace) -> None:
    manifest = make_manifest(args)
    outdir = Path(manifest["outdir"])
    if outdir.exists():
        raise FileExistsError(f"Use a fresh output directory: {outdir}")
    config, cluster = manifest["config"], manifest["config"]["cluster"]
    manifest_path = outdir / "manifest.json"
    invoke = [config["runtime"]["python"], __file__]
    common = [
        "sbatch",
        "--parsable",
        "--account",
        args.account,
        "--time",
        cluster["time"],
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task",
        str(cluster["cpus"]),
        "--mem",
        cluster["memory"],
        "--export=ALL",
    ]
    generation = common + [
        "--partition",
        cluster["partition"],
        "--gpus=1",
        "--array",
        f"0-{len(manifest['tasks']) - 1}%{cluster['max_parallel']}",
        "--job-name",
        f"eval-{outdir.name}",
        "--output",
        str(outdir / "generate-%A_%a.log"),
        "--wrap",
        command_string(invoke + ["worker", "--manifest", manifest_path]),
    ]
    score = common + [
        "--partition",
        cluster["scoring_partition"],
        "--job-name",
        f"score-{outdir.name}",
        "--output",
        str(outdir / "score-%j.log"),
        "--wrap",
        command_string(invoke + ["collect", "--manifest", manifest_path]),
    ]
    print(
        json.dumps(
            [
                {k: task[k] for k in ("benchmark", "rank", "expected_ids")}
                for task in manifest["tasks"]
            ],
            indent=2,
        )
    )
    print(command_string(generation), flush=True)
    if args.dry_run:
        print(
            command_string(score + ["--dependency=afterany:<generation-job-id>"]),
            flush=True,
        )
        return
    outdir.mkdir(parents=True)
    write_json(manifest_path, manifest)
    # Validate both resource requests before starting either job.
    subprocess.run(generation + ["--test-only"], check=True)
    subprocess.run(score + ["--test-only"], check=True)
    generation_id = subprocess.check_output(generation, text=True).strip().split(";")[0]
    write_json(outdir / "jobs.json", {"generation": generation_id})
    score += [f"--dependency=afterany:{generation_id}"]
    print(command_string(score), flush=True)
    score_id = subprocess.check_output(score, text=True).strip().split(";")[0]
    write_json(outdir / "jobs.json", {"generation": generation_id, "scoring": score_id})
    print(
        f"Submitted generation={generation_id}, scoring={score_id}; results: {outdir}"
    )


def container_command(manifest: dict, action: str, benchmark: str = "") -> list[str]:
    config = manifest["config"]
    runtime = config["runtime"]
    image = (
        runtime["scicode_container"] if benchmark == "scicode" else runtime["container"]
    )
    return [
        "srun",
        "--container-image",
        image,
        "--container-mounts",
        config["cluster"]["mounts"],
        "--container-workdir",
        str(ROOT),
        "--no-container-mount-home",
        runtime["python"],
        __file__,
        action,
        "--manifest",
        str(Path(manifest["outdir"]) / "manifest.json"),
        "--inside",
    ]


def allocate_ports() -> tuple[int, int]:
    # Bind both simultaneously, then release immediately before launching our children.
    with socket.socket() as server, socket.socket() as sandbox:
        server.bind(("127.0.0.1", 0))
        sandbox.bind(("127.0.0.1", 0))
        return server.getsockname()[1], sandbox.getsockname()[1]


def generation_command(
    manifest: dict, task: dict, outdir: Path, port: int
) -> list[str]:
    settings, runtime = task["settings"], manifest["config"]["runtime"]
    values = {
        "model-path": manifest["model"],
        "tokenizer-path": manifest["tokenizer"],
        "outdir": outdir,
        "benchmark": task["benchmark"],
        "backend": "vllm",
        "vllm-python": runtime["vllm_python"],
        "generation-api": "chat_completions",
        "prompt-file": ROOT / runtime["prompt_file"],
        "base-url": f"http://127.0.0.1:{port}",
        "port": port,
        "dllm-algorithm": settings["algorithm"],
        "mem-fraction-static": settings["memory_fraction"],
        "max-running-requests": settings["concurrent"],
        "server-random-seed": settings["server_seed"],
        "val-batch-size": task["sample_count"],
        "shard-dp-size": settings["shards"],
        "shard-rank": task["rank"],
    }
    for key in (
        "num_samples",
        "seed",
        "temperature",
        "block_size",
        "max_steps",
        "threshold",
        "selection_policy",
        "enable_thinking",
        "thinking_budget",
        "max_new_tokens",
        "context_length",
        "concurrent",
        "top_p",
        "top_k",
    ):
        values[key.replace("_", "-")] = settings[key]
    return [
        runtime["python"],
        str(ROOT / "examples/eval_grpo_checkpoint_validation.py"),
    ] + [
        str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in values.items()
        for value in ("--" + key, value)
    ]


def scicode_environment(
    manifest: dict, task: dict, outdir: Path, port: int, sandbox_port: int
) -> dict[str, str]:
    settings, config = task["settings"], manifest["config"]
    samples = selected_samples(config, "scicode", settings)
    wanted = set(task["expected_ids"])
    input_path = outdir / "input.jsonl"
    input_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in samples
            if str(row["problem_id"]) in wanted
        )
    )
    values = {
        "MODEL": manifest["model"],
        "TOKENIZER": manifest["tokenizer"],
        "OUTDIR": outdir,
        "SCICODE_DATA_DIR": Path(config["runtime"]["data_root"]) / "scicode",
        "SCICODE_INPUT_FILE": input_path,
        "SERVER_PORT": port,
        "SANDBOX_PORT": sandbox_port,
        "BACKEND": "vllm",
        "TP_SIZE": 1,
        "VLLM_PY": config["runtime"]["vllm_python"],
        "NS_VENV": config["runtime"]["scicode_venv"],
    }
    mapping = {
        "DLLM_ALGORITHM": "algorithm",
        "BLOCK_SIZE": "block_size",
        "MAX_STEPS": "max_steps",
        "TEMPERATURE": "temperature",
        "TOP_P": "top_p",
        "TOP_K": "top_k",
        "THRESHOLD": "threshold",
        "SELECTION_POLICY": "selection_policy",
        "ENABLE_THINKING": "enable_thinking",
        "THINKING_BUDGET": "thinking_budget",
        "TOKENS_TO_GENERATE": "max_new_tokens",
        "MAX_MODEL_LEN": "context_length",
        "MEM_FRACTION_STATIC": "memory_fraction",
        "MAX_RUNNING_REQUESTS": "concurrent",
        "NUM_PARALLEL_REQUESTS": "concurrent",
        "SERVER_RANDOM_SEED": "server_seed",
        "SPLIT": "split",
        "PROMPT_CONFIG": "prompt_config",
        "SANDBOX_TIMEOUT": "sandbox_timeout",
    }
    values.update({key: settings[value] for key, value in mapping.items()})
    return {
        key: str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in values.items()
    }


def worker(manifest: dict, inside: bool) -> None:
    task = manifest["tasks"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
    if not inside:
        subprocess.run(
            container_command(manifest, "worker", task["benchmark"]), check=True
        )
        return
    verify_files(manifest)
    outdir = Path(manifest["outdir"]) / task["benchmark"] / f"shard-{task['rank']:03d}"
    outdir.mkdir(parents=True, exist_ok=False)
    port, sandbox_port = allocate_ports()
    env = (
        os.environ
        | dataset_environment(manifest["config"])
        | {
            "HF_HOME": manifest["config"]["runtime"]["hf_home"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "EXPANDABLE_SEGMENTS": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    env.pop("PYTHONPATH", None)
    if task["benchmark"] == "scicode":
        env.update(scicode_environment(manifest, task, outdir, port, sandbox_port))
        # This compatibility escape hatch must not override the resolved manifest.
        env.pop("NS_EXTRA_ARGS", None)
        command = ["bash", str(ROOT / "tools/scicode/run_scicode_ns_eval.sh")]
    else:
        command = generation_command(manifest, task, outdir, port)
    state = {
        "status": "running",
        "started": time.time(),
        "server_port": port,
        "sandbox_port": sandbox_port,
        "command": command,
        "settings": task["settings"],
    }
    write_json(outdir / "status.json", state)
    print(command_string(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env)
    state.update(
        status="complete" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        elapsed_seconds=time.time() - state["started"],
    )
    write_json(outdir / "status.json", state)
    if result.returncode:
        raise SystemExit(result.returncode)


def merge_records(manifest: dict, benchmark: str) -> tuple[list[dict], dict]:
    tasks = [task for task in manifest["tasks"] if task["benchmark"] == benchmark]
    merged = []
    for task in tasks:
        directory = Path(manifest["outdir"]) / benchmark / f"shard-{task['rank']:03d}"
        state = json.loads((directory / "status.json").read_text())
        if state["status"] != "complete":
            raise ValueError(f"Incomplete shard: {directory}")
        filename = (
            "eval-results/scicode/output.jsonl"
            if benchmark == "scicode"
            else "records.jsonl"
        )
        records = load_records(
            directory / filename,
            "problem_id" if benchmark == "scicode" else "source_id",
        )
        if set(sample_ids(records, benchmark)) != set(task["expected_ids"]):
            raise ValueError(f"Missing or unexpected sample IDs in {directory}")
        if benchmark == "scicode":
            expected_steps = json.loads((directory / "expected_steps.json").read_text())
            for record in records:
                if set(record["generation"]) != set(
                    expected_steps[str(record["problem_id"])]
                ):
                    raise ValueError(
                        f"Incomplete SciCode generation: {record['problem_id']}"
                    )
        merged.extend(records)
    sample_ids(merged, benchmark)  # also reject cross-shard duplicates
    return merged, tasks[0]["settings"]


def score_scicode(records: list[dict]) -> dict:
    # Same pass@1 definitions as NeMo-Skills SciCodeMetrics, with completeness checks.
    problems_passed = subtasks_passed = subtasks = 0
    for record in records:
        statuses = record["eval_status"]
        if not statuses or len(statuses) != len(record["generation"]):
            raise ValueError(
                f"Incomplete SciCode subtask grading: {record['problem_id']}"
            )
        passed = sum(status["process_status"] == "completed" for status in statuses)
        subtasks += len(statuses)
        subtasks_passed += passed
        problems_passed += passed == len(statuses)
    return {
        "num_problems": len(records),
        "num_subtasks": subtasks,
        "problems_passed": problems_passed,
        "subtasks_passed": subtasks_passed,
        "problem_accuracy": problems_passed / len(records),
        "subtask_accuracy": subtasks_passed / subtasks,
        "scorer": "NeMo-Skills inline tests; single attempt",
    }


def collect(manifest: dict, inside: bool) -> None:
    if not inside:
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError(
                "Collect executes model code; run it in a CPU Slurm allocation (see README)"
            )
        subprocess.run(container_command(manifest, "collect"), check=True)
        return
    verify_files(manifest)
    runtime = manifest["config"]["runtime"]
    summary = {}
    for benchmark in dict.fromkeys(task["benchmark"] for task in manifest["tasks"]):
        directory = Path(manifest["outdir"]) / benchmark
        try:
            records, settings = merge_records(manifest, benchmark)
            path = directory / "records.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            write_json(
                directory / "metrics.json",
                {"settings": settings, "total": len(records)},
            )
            score_path = directory / "scores.json"
            if benchmark == "scicode":
                metrics = score_scicode(records)
                write_json(score_path, metrics)
            else:
                scripts = {
                    "ifbench": "score_ifbench.py",
                    "livecodebench": "score_lcb.py",
                    "aa_lcr": "score_aa_lcr.py",
                }
                command = [
                    runtime["scoring_python"],
                    str(ROOT / "tools" / benchmark / scripts[benchmark]),
                    "--records",
                    str(path),
                    "--out",
                    str(score_path),
                ]
                if benchmark == "ifbench":
                    command += [
                        "--data",
                        str(Path(runtime["data_root"]) / settings["dataset"]),
                        "--ifbench-repo",
                        runtime["ifbench_repo"],
                    ]
                elif benchmark == "livecodebench":
                    command += [
                        "--tests",
                        str(Path(runtime["data_root"]) / settings["tests"]),
                    ]
                    for key in (
                        "workers",
                        "per_test_timeout",
                        "job_timeout",
                        "mem_limit_mb",
                    ):
                        command += ["--" + key.replace("_", "-"), str(settings[key])]
                else:
                    command += ["--judge", settings["judge"]]
                env = os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
                env.pop("PYTHONPATH", None)
                subprocess.run(command, check=True, env=env, cwd=ROOT)
                metrics = json.loads(score_path.read_text())
            summary[benchmark] = {
                "status": "complete",
                "samples": len(records),
                "scores": metrics,
            }
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            summary[benchmark] = {"status": "incomplete", "error": str(error)}
    write_json(Path(manifest["outdir"]) / "results.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if any(result["status"] != "complete" for result in summary.values()):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    submit_parser = commands.add_parser(
        "submit", help="Print/submit GPU shards followed by CPU scoring"
    )
    submit_parser.add_argument(
        "--config", type=Path, default=ROOT / "examples/configs/benchmark_eval.yaml"
    )
    submit_parser.add_argument("--model", type=Path, required=True)
    submit_parser.add_argument("--tokenizer", type=Path)
    submit_parser.add_argument("--outdir", type=Path, required=True)
    submit_parser.add_argument("--account", required=True)
    submit_parser.add_argument("--benchmarks", choices=BENCHMARKS, nargs="+")
    for flag in (
        "num-samples",
        "shards",
        "max-parallel",
        "max-new-tokens",
        "thinking-budget",
    ):
        submit_parser.add_argument("--" + flag, type=int)
    submit_parser.add_argument("--temperature", type=float)
    submit_parser.add_argument("--algorithm", choices=("FastDiffuser", "AR"))
    submit_parser.add_argument("--dry-run", action="store_true")
    for action in ("worker", "collect"):
        subparser = commands.add_parser(action)
        subparser.add_argument("--manifest", type=Path, required=True)
        subparser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == "submit":
        submit(args)
    else:
        manifest = json.loads(args.manifest.read_text())
        (worker if args.action == "worker" else collect)(manifest, args.inside)


if __name__ == "__main__":
    main()
