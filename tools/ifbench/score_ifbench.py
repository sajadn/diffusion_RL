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
"""Offline IFBench grader for standalone-evaluator records.jsonl.

Generation and grading are split on purpose: the IFBench verifiers need
nltk/langdetect/emoji/syllapy, none of which exist in the eval container. The GPU
job only writes responses; this script grades them afterwards using the upstream
allenai/IFBench verifier package, so the scoring semantics (strict + 8-variant
loose) are theirs, not a reimplementation.

Usage:
  <venv>/bin/python score_ifbench.py --records <outdir>/records.jsonl
"""

from __future__ import annotations

import os

# The dfw login node caps RLIMIT_NPROC at 300; OpenBLAS otherwise tries to spawn one
# thread per core (64) and aborts. Set before numpy is pulled in by any dependency.
for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_IFBENCH_REPO = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/"
    "code/ifbench_src/IFBench-main"
)
DEFAULT_DATA = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/"
    "eval_data/ifbench/test.jsonl"
)


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.benchmark_eval.scoring import (
    load_records,
    resolve_thinking_mode,
    strip_thinking,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--ifbench-repo", type=Path, default=DEFAULT_IFBENCH_REPO)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--thinking",
        choices=("auto", "on", "off"),
        default="auto",
        help="Whether generation used thinking. 'auto' reads settings.enable_thinking "
        "from the sibling metrics.json.",
    )
    args = ap.parse_args()

    thinking_on = resolve_thinking_mode(args.records, args.thinking)
    print("thinking during generation: %s" % ("ON" if thinking_on else "OFF"))

    sys.path.insert(0, str(args.ifbench_repo))
    import evaluation_lib  # noqa: E402

    by_key = {}
    for line in open(args.data, "r", encoding="utf-8"):
        row = json.loads(line)
        by_key[str(row["key"])] = row

    records = load_records(args.records)

    n_unclosed = 0
    n_empty = 0
    results = []
    for rec in records:
        sid = str(rec.get("source_id") or "")
        key = sid.split("ifbench-", 1)[-1]
        row = by_key.get(key)
        if row is None:
            raise KeyError(f"record source_id {sid!r} not found in {args.data}")

        raw = rec.get("response") or ""
        if thinking_on:
            answer, closed = strip_thinking(raw)
            if not closed:
                n_unclosed += 1
        else:
            # thinking was closed in the prompt; the whole response is the answer
            answer, closed = raw.strip(), True
        if not answer.strip():
            n_empty += 1

        # Upstream's loose path calls build_description(**kwargs) WITHOUT dropping
        # None values; in their runner it only works because the strict path filters
        # inp.kwargs in place first. Filter here so each call is order-independent.
        clean_kwargs = [
            {k: v for k, v in kw.items() if v is not None} for kw in row["kwargs"]
        ]

        def _example():
            return evaluation_lib.InputExample(
                key=row["key"],
                instruction_id_list=list(row["instruction_id_list"]),
                prompt=row["prompt"],
                kwargs=[dict(k) for k in clean_kwargs],
            )

        # single-entry mapping avoids any prompt-collision ambiguity
        strict = evaluation_lib.test_instruction_following_strict(
            _example(), {row["prompt"]: answer}
        )
        loose = evaluation_lib.test_instruction_following_loose(
            _example(), {row["prompt"]: answer}
        )
        results.append((row, strict, loose, closed))

    n = len(results)
    p_strict = sum(1 for _, s, _, _ in results if s.follow_all_instructions)
    p_loose = sum(1 for _, _, l, _ in results if l.follow_all_instructions)
    i_tot = sum(len(r["instruction_id_list"]) for r, _, _, _ in results)
    i_strict = sum(sum(s.follow_instruction_list) for _, s, _, _ in results)
    i_loose = sum(sum(l.follow_instruction_list) for _, _, l, _ in results)

    per_type = defaultdict(lambda: [0, 0])
    for row, s, _, _ in results:
        for iid, ok in zip(row["instruction_id_list"], s.follow_instruction_list):
            per_type[iid][1] += 1
            per_type[iid][0] += int(ok)

    metrics = {
        "n_prompts": n,
        "prompt_level_strict": p_strict / n if n else 0.0,
        "prompt_level_loose": p_loose / n if n else 0.0,
        "instruction_level_strict": i_strict / i_tot if i_tot else 0.0,
        "instruction_level_loose": i_loose / i_tot if i_tot else 0.0,
        "prompt_strict_correct": p_strict,
        "prompt_loose_correct": p_loose,
        "instruction_total": i_tot,
        "responses_without_closing_think": n_unclosed,
        "responses_empty_after_strip": n_empty,
        "thinking_during_generation": thinking_on,
        "per_constraint_strict": {
            k: {"pass": v[0], "n": v[1]} for k, v in sorted(per_type.items())
        },
    }

    out = args.out or args.records.parent / "ifbench_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("prompts: %d" % n)
    print(
        "  prompt-level      strict %.4f (%d/%d)   loose %.4f (%d/%d)"
        % (
            metrics["prompt_level_strict"],
            p_strict,
            n,
            metrics["prompt_level_loose"],
            p_loose,
            n,
        )
    )
    print(
        "  instruction-level strict %.4f (%d/%d)   loose %.4f (%d/%d)"
        % (
            metrics["instruction_level_strict"],
            i_strict,
            i_tot,
            metrics["instruction_level_loose"],
            i_loose,
            i_tot,
        )
    )
    print("  responses with no closing </think>: %d/%d" % (n_unclosed, n))
    print("  empty after stripping thinking:     %d/%d" % (n_empty, n))
    print("wrote", out)


if __name__ == "__main__":
    main()
