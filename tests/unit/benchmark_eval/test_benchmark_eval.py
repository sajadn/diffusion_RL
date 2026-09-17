"""Contract tests for requests, sharding, and benchmark score denominators."""

import argparse
import json
import os
import subprocess
import textwrap
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
import eval_grpo_checkpoint_validation as standalone
from tools.benchmark_eval import evaluate
from tools.benchmark_eval.scoring import (
    load_records,
    resolve_thinking_mode,
    strip_thinking,
)
from tools.aa_lcr import score_aa_lcr
from tools.livecodebench.score_lcb import grade_one
from tools.scicode import patch_ns_omit_seed


class RequestTests(unittest.TestCase):
    def request(
        self,
        algorithm="FastDiffuser",
        temperature=0.9,
        prompt_length=80,
        thinking_budget=32,
    ):
        args = argparse.Namespace(
            backend="vllm",
            dllm_algorithm=algorithm,
            served_model_name="default",
            enable_thinking="true",
            thinking_budget=thinking_budget,
            context_length=100,
        )
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "reason</think>answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 7},
        }
        with patch("requests.post", return_value=response) as post:
            result = standalone.generate_one_chat_completions(
                "http://localhost:123",
                "{}",
                "question",
                64,
                temperature,
                1.0,
                args,
                "livecodebench",
                system_message="LCB system",
                prompt_length=prompt_length,
            )
        return post.call_args.kwargs["json"], result

    def test_diffusion_request_preserves_system_and_caps_context(self):
        payload, result = self.request()
        self.assertEqual(payload["temperature"], 1)
        self.assertEqual(
            payload["messages"][0], {"role": "system", "content": "LCB system"}
        )
        self.assertEqual(payload["max_tokens"], 19)
        self.assertEqual(payload["thinking_token_budget"], 18)
        self.assertEqual(result["completion_tokens"], 7)
        self.assertNotIn("seed", payload)

    def test_zero_thinking_budget_omits_reasoning_cap(self):
        payload, _ = self.request(thinking_budget=0)
        self.assertNotIn("thinking_token_budget", payload)
        self.assertEqual(payload["max_tokens"], 19)

    def test_ar_keeps_requested_temperature(self):
        payload, _ = self.request(algorithm="AR")
        self.assertEqual(payload["temperature"], 0.9)

    def test_greedy_diffusion_request(self):
        payload, _ = self.request(temperature=0)
        self.assertEqual(payload["temperature"], 0)

    def test_overlong_prompt_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            self.request(prompt_length=100)


class LiveCodeBenchTests(unittest.TestCase):
    def grade(self, code, cases, functional=False):
        problem = {
            "functional": functional,
            "func_name": "solve" if functional else None,
            "tests": cases,
        }
        return grade_one(("fixture", code, problem, sys.executable, 2, 10, 4096))[1:]

    def test_buffered_and_text_stdin(self):
        for expression in (
            "sum(map(int, sys.stdin.buffer.read().split()))",
            "int(sys.stdin.buffer.readline()) + int(sys.stdin.buffer.readline())",
            "int(input()) + int(sys.stdin.readline())",
            "sum(map(int, sys.stdin.read().split()))",
        ):
            with self.subTest(expression=expression):
                self.assertEqual(
                    self.grade(
                        f"import sys; print({expression})",
                        [
                            {"input": "1\n2\n", "output": "3\n"},
                            {"input": "4\n5\n", "output": "9\n"},
                        ],
                    ),
                    (True, "ok"),
                )

    def test_upstream_stdout_comparison(self):
        cases = [
            ("1  2\n", "1 2\n", True),
            ("1.0 2e0\n", "1 2\n", True),
            ("  yes  \n  no  \n", "yes\nno", True),
            ("", "\n", True),
            ("9007199254740992", "9007199254740993", False),
            ("1.00000000000000000001", "1", False),
            ("yes  no", "yes no", False),
            ("1\n\n2", "1\n2", False),
            ("2 1", "1 2", False),
            ("1 2 3", "1 2", False),
        ]
        for actual, expected, passed in cases:
            with self.subTest(actual=actual, expected=expected):
                self.assertEqual(
                    self.grade(
                        f"print({actual!r}, end='')",
                        [{"input": "", "output": expected}],
                    ),
                    (passed, "ok" if passed else "wrong_answer"),
                )

    def test_functional_upstream_helpers(self):
        code = """
class Solution:
    @cache
    def cached(self, value):
        return value + 1

    def solve(self, nums: List[int]):
        heap = []
        heappush(heap, self.cached(2))
        return [bisect_left(nums, 2), heappop(heap), Counter(nums)[2],
                gcd(12, 8), reduce(add, nums), int(sqrt(9)),
                list(accumulate(nums)), sys.getrecursionlimit()]
"""
        self.assertEqual(
            self.grade(
                code,
                [
                    {
                        "input": "[1, 2, 2]",
                        "output": "[1, 3, 2, 4, 5, 3, [1, 3, 5], 50000]",
                    }
                ],
                functional=True,
            ),
            (True, "ok"),
        )

    def test_functional_stdout_does_not_corrupt_verdict(self):
        code = """
print('module', end='')
class Solution:
    def __init__(self):
        print('constructor', end='')

    def solve(self, value):
        print('call', end='')
        return value + 1
"""
        for expected, verdict in (("2", (True, "ok")), ("3", (False, "wrong_answer"))):
            with self.subTest(expected=expected):
                self.assertEqual(
                    self.grade(
                        code, [{"input": "1", "output": expected}], functional=True
                    ),
                    verdict,
                )

    def test_functional_exceptions_keep_valid_verdict(self):
        for body, reason in (
            ("raise ValueError('bad input')", "ValueError: bad input"),
            ("while True: pass", "timeout"),
        ):
            with self.subTest(body=body):
                code = (
                    "class Solution:\n    def solve(self, value):\n        print('debug', end='')\n        "
                    + body
                )
                self.assertEqual(
                    self.grade(code, [{"input": "1", "output": "2"}], functional=True),
                    (False, reason),
                )


class SciCodeClientPatchTests(unittest.TestCase):
    # Minimal client reproducing the pinned NeMo-Skills request mapping. Also
    # validate the patch against the actual configured container before submission.
    source = textwrap.dedent("""\
        import os

        class Client:
            def build(self, random_seed, repetition_penalty, extra_body=None):
                params = {
                    "seed": random_seed,
                }
                if repetition_penalty is not None:
                    params["presence_penalty"] = repetition_penalty
                return params
    """)

    def test_request_penalties_seed_and_extra_body(self):
        namespace = {}
        exec(patch_ns_omit_seed.patch_source(self.source), namespace)
        client = namespace["Client"]()
        for algorithm, omit_seed in (("AR", "0"), ("FastDiffuser", "1")):
            with (
                self.subTest(algorithm=algorithm),
                patch.dict(os.environ, {"NEMO_SKILLS_OMIT_SEED": omit_seed}),
            ):
                for repetition in (1.0, 1.2):
                    params = client.build(
                        42, repetition, {"thinking_token_budget": 128}
                    )
                    self.assertEqual(params["presence_penalty"], 0.0)
                    self.assertEqual(
                        params["extra_body"],
                        {
                            "repetition_penalty": repetition,
                            "thinking_token_budget": 128,
                        },
                    )
                    if algorithm == "AR":
                        self.assertEqual(params["seed"], 42)
                    else:
                        self.assertNotIn("seed", params)
                params = client.build(42, 1.0)
                self.assertEqual(params["extra_body"], {"repetition_penalty": 1.0})

    def test_idempotent_patch_and_upgrade(self):
        patched = patch_ns_omit_seed.patch_source(self.source)
        self.assertEqual(patch_ns_omit_seed.patch_source(patched), patched)
        earlier_patch = self.source.replace(
            patch_ns_omit_seed.OLD, patch_ns_omit_seed.NEW, 1
        )
        self.assertEqual(patch_ns_omit_seed.patch_source(earlier_patch), patched)

    def test_unknown_client_fails_without_partial_patch(self):
        with self.assertRaisesRegex(ValueError, "penalty mapping changed"):
            patch_ns_omit_seed.patch_source(
                self.source.replace(
                    'params["presence_penalty"] = repetition_penalty',
                    'params["presence_penalty"] = 99',
                )
            )


class SciCodeCommandTests(unittest.TestCase):
    def test_thinking_budget_in_generation_command(self):
        # Execute the real wrapper with fake external runtimes. Stop at generation
        # after capturing argv; no GPU, model code, /data writes, or Slurm required.
        for budget in (0, 128):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                binaries = directory / "bin"
                binaries.mkdir()
                data = directory / "data"
                data.mkdir()
                installed_data = directory / "installed_data"
                installed_data.mkdir()
                for name in (
                    "dev.jsonl",
                    "test.jsonl",
                    "test_aai.jsonl",
                    "test_data.h5",
                ):
                    (data / name).touch()
                runtime = binaries / "python"
                runtime.write_text(
                    f"#!{sys.executable}\n"
                    + r"""import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
if args[:2] == ["-m", "nemo_skills.inference.eval.scicode"]:
    Path(os.environ["CAPTURE"]).write_text(json.dumps(args))
    sys.exit(77)
if args[:2] == ["-m", "vllm.entrypoints.openai.api_server"] or args[0].endswith("run_sandbox.py"):
    time.sleep(60)
elif args[0] == "-c":
    if "import nemo_skills" in args[1]:
        print(os.environ["INSTALLED_DATA"])
    elif "float(sys.argv[1])" in args[1]:
        print(0)
"""
                )
                runtime.chmod(0o755)
                for name, body in {
                    "curl": "exit 0",
                    "ln": "exit 0",
                    "mkdir": 'if [[ "$*" != "-p /data" ]]; then /bin/mkdir "$@"; fi',
                }.items():
                    stub = binaries / name
                    stub.write_text("#!/bin/bash\n" + body + "\n")
                    stub.chmod(0o755)
                capture = directory / "argv.json"
                env = dict(
                    os.environ,
                    PATH=f"{binaries}:{os.environ['PATH']}",
                    OUTDIR=str(directory / "output"),
                    MODEL="fixture-model",
                    SCICODE_DATA_DIR=str(data),
                    NS_VENV=tmp,
                    VLLM_PY=str(runtime),
                    BACKEND="vllm",
                    DLLM_ALGORITHM="FastDiffuser",
                    THINKING_BUDGET=str(budget),
                    CAPTURE=str(capture),
                    INSTALLED_DATA=str(installed_data),
                )
                result = subprocess.run(
                    ["bash", str(ROOT / "tools/scicode/run_scicode_ns_eval.sh")],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
                args = json.loads(capture.read_text())
                budget_args = [arg for arg in args if "thinking_token_budget" in arg]
                self.assertEqual(
                    budget_args,
                    []
                    if budget == 0
                    else ["++inference.extra_body.thinking_token_budget=128"],
                )
                self.assertIn("++chat_template_kwargs.enable_thinking=true", args)


class ScoringTests(unittest.TestCase):
    def test_reasoning_is_not_an_answer(self):
        self.assertEqual(strip_thinking("correct reference in reasoning"), ("", False))
        self.assertEqual(
            strip_thinking("reference</think>wrong answer"), ("wrong answer", True)
        )

    def test_auto_thinking_reads_generation_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "metrics.json").write_text(
                json.dumps({"settings": {"enable_thinking": False}})
            )
            self.assertFalse(resolve_thinking_mode(directory / "records.jsonl", "auto"))
            self.assertTrue(resolve_thinking_mode(directory / "records.jsonl", "on"))

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text('{"source_id":"same"}\n' * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_records(path)

    def test_aa_scoring_ignores_reasoning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "source_id": "test-1",
                        "response": "Paris</think>London",
                        "gold": "Paris",
                    }
                )
                + "\n"
            )
            with patch.object(
                sys, "argv", ["score", "--records", str(path), "--thinking", "on"]
            ):
                score_aa_lcr.main()
            result = json.loads((path.parent / "aa_lcr_metrics.json").read_text())
            self.assertEqual(result["correct"], 0)
            self.assertIn("heuristic approximation", result["note"])

    def test_judge_url_and_strict_verdict(self):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "not CORRECT"}}]
        }
        with patch("requests.post", return_value=response) as post:
            result = score_aa_lcr.judge_openai(
                "http://localhost/v1", "judge", "q", "a", "b", 10, None
            )
        self.assertIsNone(result)
        self.assertEqual(post.call_args.args[0], "http://localhost/v1/chat/completions")

    def test_scicode_aggregates_subtasks_not_shard_percentages(self):
        rows = [
            {
                "problem_id": "1",
                "generation": {"1.1": "", "1.2": "", "1.3": ""},
                "sub_steps": [None] * 4,  # one upstream prefilled step is not graded
                "eval_status": [{"process_status": "completed"}] * 2
                + [{"process_status": "error"}],
            },
            {
                "problem_id": "2",
                "generation": {"2.1": ""},
                "sub_steps": [None],
                "eval_status": [{"process_status": "completed"}],
            },
        ]
        result = evaluate.score_scicode(rows)
        self.assertEqual(result["subtask_accuracy"], 0.75)
        self.assertEqual(result["problem_accuracy"], 0.5)
        rows[0]["eval_status"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            evaluate.score_scicode(rows)

    def test_incomplete_shard_cannot_be_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "ifbench/shard-000"
            directory.mkdir(parents=True)
            (directory / "status.json").write_text('{"status":"running"}')
            manifest = {"outdir": tmp, "tasks": [{"benchmark": "ifbench", "rank": 0}]}
            with self.assertRaisesRegex(ValueError, "Incomplete shard"):
                evaluate.merge_records(manifest, "ifbench")

    def test_missing_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "ifbench/shard-000"
            directory.mkdir(parents=True)
            (directory / "status.json").write_text('{"status":"complete"}')
            (directory / "records.jsonl").write_text('{"source_id":"a"}\n')
            manifest = {
                "outdir": tmp,
                "tasks": [
                    {"benchmark": "ifbench", "rank": 0, "expected_ids": ["a", "b"]}
                ],
            }
            with self.assertRaisesRegex(ValueError, "Missing or unexpected"):
                evaluate.merge_records(manifest, "ifbench")


class PlanTests(unittest.TestCase):
    def test_shards_cover_dataset_exactly_once(self):
        for count, shards in [(300, 1), (175, 1), (80, 4), (100, 10), (7, 3)]:
            samples = [{"source_id": str(i)} for i in range(count)]
            indices = [
                i
                for rank in range(shards)
                for i, _ in standalone.select_validation_shard(
                    samples, count, shards, rank
                )
            ]
            self.assertEqual(indices, list(range(count)))

    def test_ports_are_distinct(self):
        server, sandbox = evaluate.allocate_ports()
        self.assertNotEqual(server, sandbox)

    def test_invalid_sample_limit_is_rejected(self):
        import yaml

        config = yaml.safe_load(
            (ROOT / "examples/configs/benchmark_eval.yaml").read_text()
        )
        settings = config["common"] | config["benchmarks"]["ifbench"]
        settings["num_samples"] = 0
        with self.assertRaisesRegex(ValueError, "num_samples"):
            evaluate.validate_settings(settings)


if __name__ == "__main__":
    unittest.main()
