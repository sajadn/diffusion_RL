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
"""Subprocess entrypoint for the unmodified upstream JustGRPO reward functions."""

import json
import sys
from pathlib import Path


def score_rows(rows: list[dict]) -> list[float]:
    if not rows:
        return []
    kinds = {row["kind"] for row in rows}
    if len(kinds) != 1:
        raise ValueError("A grading chunk must contain exactly one task type")
    if kinds == {"acecode"}:
        # Load the reference's Torch dependency only in the scoring subprocess.
        from data.code import reward_code

        scores = reward_code(
            {"test_cases": [row["test_cases"] for row in rows]},
            [row["completion"] for row in rows],
            num_generations=1,
            device="cpu",
        ).tolist()
    elif kinds == {"humaneval"}:
        # HumanEval grading uses the reference parser and execution harness.
        from utils.code_exec import build_test_code, check_correctness

        scores = []
        for row in rows:
            program = build_test_code(row, {row["task_id"]: row})
            result = check_correctness(
                row["task_id"], {"test_code": program}, timeout=3.0
            )
            scores.append(float(result["passed"]))
    else:
        raise ValueError(f"Unsupported task types: {kinds}")
    return scores


def main() -> None:
    rows = json.load(sys.stdin)
    sys.path.insert(0, sys.argv[1])
    result_path = Path(sys.argv[2])
    scores = score_rows(rows)
    # Candidate code and the reference grader may write arbitrary stdout.
    # Keep the reward protocol separate from those diagnostic streams.
    result_path.write_text(json.dumps(scores))


if __name__ == "__main__":
    main()
