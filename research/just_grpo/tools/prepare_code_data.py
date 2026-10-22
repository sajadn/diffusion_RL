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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acecode", type=Path, required=True)
    parser.add_argument("--humaneval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
