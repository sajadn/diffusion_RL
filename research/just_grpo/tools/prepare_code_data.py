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
"""Prepare the authors' AceCode-Hard and held-out HumanEval for ResponseDataset."""

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path


def acecode_record(row: dict) -> dict:
    question, tests = row["question"], row["test_cases"]
    if not isinstance(question, str) or not isinstance(tests, list):
        raise ValueError("AceCode-Hard requires a question and a list of tests")
    if not all(isinstance(test, str) for test in tests):
        raise ValueError("AceCode-Hard tests must be strings")
    if "\n```\n" not in question and tests:
        question += "\nTest cases: " + tests[0]
    return {
        "problem": question,
        "answer": json.dumps({"kind": "acecode", "test_cases": tests}),
    }


def humaneval_record(row: dict) -> dict:
    prompt = (
        "You are an expert Python programmer. Your task is to complete the "
        f"implementation of a function named `{row['entry_point']}`.\n\n"
        f"Here is the function to complete:\n```python\n{row['prompt'].rstrip()}\n```\n"
    )
    return {
        "problem": prompt,
        # Preserve the original prompt for reference test-program assembly.
        "answer": json.dumps(
            {
                "kind": "humaneval",
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
        ),
    }


# Prompt format from pinned JustGRPO eval.py (1a2fddb5c6655597e63081c0af5ebb718a849f39).
def mbpp_prompt(ex):
    func_name = ex["test_list"][0].split(" ")[1].split("(")[0]
    tests_str = "\n".join(f"    {t}" for t in ex["test_list"])
    try:
        tree = ast.parse(ex["test_list"][0].strip())
        n_args = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == func_name:
                n_args = len(node.args)
                break
    except Exception:
        n_args = 2
    params = ", ".join(f"input_param_{i + 1}" for i in range(n_args))
    return (
        "You are an expert Python programmer. Your task is to complete the "
        f"implementation of a function named `{func_name}`.\n\n"
        f"** TARGET FUNCTION **\n{ex['text']}\n\n"
        "** UNIT TESTS **\n"
        f"Your code should pass unit tests like:\n{tests_str}\n\n"
        "Here is the function to complete:\n"
        f"```python\ndef {func_name}({params}):\n"
        f'    """{ex["text"]}\n    """\n```\n'
    )


def mbpp_records(raw_path: Path, test_path: Path) -> list[dict]:
    """Preserve the reference's 500-task MBPP test split and test programs."""
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    tasks = raw[10:510]
    tests = {
        str(row["task_id"]): row
        for line in test_path.read_text().splitlines()
        if line
        for row in [json.loads(line)]
    }
    ids = [str(row["task_id"]) for row in tasks]
    if len(tasks) != 500 or len(set(ids)) != 500 or set(ids) != set(tests):
        raise ValueError("Expected the reference's 500 unique MBPP test tasks")
    records = []
    for row in tasks:
        test = tests[str(row["task_id"])]["test"]
        if (
            not isinstance(test, list)
            or not test
            or not all(isinstance(case, str) for case in test)
        ):
            raise ValueError("MBPP requires nonempty test programs")
        records.append(
            {
                "problem": mbpp_prompt(row),
                "answer": json.dumps(
                    {
                        "kind": "mbpp",
                        "task_id": row["task_id"],
                        "test": test,
                    }
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acecode", type=Path, required=True)
    parser.add_argument("--humaneval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mbpp", type=Path)
    parser.add_argument("--mbpp-tests", type=Path)
    args = parser.parse_args()
    if (args.mbpp is None) != (args.mbpp_tests is None):
        parser.error("--mbpp and --mbpp-tests must be supplied together")
    mbpp = mbpp_records(args.mbpp, args.mbpp_tests) if args.mbpp is not None else None
    train = [json.loads(line) for line in args.acecode.read_text().splitlines() if line]
    with gzip.open(args.humaneval, "rt") as stream:
        val = [json.loads(line) for line in stream if line.strip()]
    if len(val) != 164 or len({row["task_id"] for row in val}) != 164:
        raise ValueError("Expected all 164 unique HumanEval test problems")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records, convert in (
        ("train", train, acecode_record),
        ("validation", val, humaneval_record),
    ):
        target = args.output / f"{name}.jsonl"
        with target.open("x") as stream:
            for row in records:
                stream.write(json.dumps(convert(row)) + "\n")
    manifest = {
        "train_rows": len(train),
        "validation_rows": len(val),
        "acecode_sha256": hashlib.sha256(args.acecode.read_bytes()).hexdigest(),
        "humaneval_sha256": hashlib.sha256(args.humaneval.read_bytes()).hexdigest(),
        "validation_added_to_training": False,
        "prompt_filtering": False,
    }
    if mbpp is not None:
        with (args.output / "mbpp.jsonl").open("x") as stream:
            for row in mbpp:
                stream.write(json.dumps(row) + "\n")
        manifest["mbpp_validation_rows"] = len(mbpp)
        manifest["mbpp_sha256"] = hashlib.sha256(args.mbpp.read_bytes()).hexdigest()
        manifest["mbpp_tests_sha256"] = hashlib.sha256(
            args.mbpp_tests.read_bytes()
        ).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
