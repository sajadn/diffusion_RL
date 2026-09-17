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
"""Offline LiveCodeBench grader for standalone-evaluator records.jsonl.

Generation and grading are split: the GPU job records responses only, and this
script executes the extracted programs against the problem test cases. Execution
happens here (not in the eval job) because there is no Docker on this cluster, so
model-authored code has to be confined with subprocess + rlimits + timeouts, and
that belongs in a CPU job rather than on a GPU node.

Code extraction matches lcb_runner/utils/extraction_utils.py: the LAST fenced block.

Usage:
  <venv>/bin/python score_lcb.py --records <outdir>/records.jsonl [--workers 16]
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import subprocess
import sys
import zlib
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.benchmark_eval.scoring import load_records

DEFAULT_TESTS = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/"
    "eval_data/livecodebench/test6.jsonl"
)
DRIVER = Path(__file__).resolve().parent / "_lcb_driver.py"


def extract_code(model_output: str) -> str:
    """Last fenced block, matching LCB's extraction (not the first)."""
    lines = model_output.split("\n")
    fences = [i for i, l in enumerate(lines) if "```" in l]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


def decode_private(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        )


def load_problems(tests_path: Path) -> dict:
    out = {}
    for line in open(tests_path, "r", encoding="utf-8"):
        r = json.loads(line)
        meta = json.loads(r["metadata"] or "{}")
        tests = json.loads(r["public_test_cases"]) + decode_private(
            r["private_test_cases"]
        )
        out[str(r["question_id"])] = {
            "tests": tests,
            "func_name": meta.get("func_name"),
            "difficulty": r["difficulty"],
            "platform": r["platform"],
            "functional": bool(r["starter_code"].strip()),
        }
    return out


def grade_one(args) -> tuple:
    qid, code, prob, py, per_test_timeout, job_timeout, mem_mb = args
    job = {
        "code": code,
        "tests": prob["tests"],
        "mode": "functional" if prob["functional"] else "stdin",
        "func_name": prob["func_name"],
        "per_test_timeout": per_test_timeout,
        "mem_limit_mb": mem_mb,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="lcb-") as workdir:
            p = subprocess.run(
                [py, str(DRIVER)],
                input=json.dumps(job),
                capture_output=True,
                text=True,
                timeout=job_timeout,
                cwd=workdir,
            )
    except subprocess.TimeoutExpired:
        return qid, False, "job_timeout"
    if p.returncode != 0:
        return qid, False, "driver_exit_%d" % p.returncode
    try:
        r = json.loads(p.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return qid, False, "driver_unparsable"
    return qid, bool(r.get("passed")), r.get("reason", "?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--tests", type=Path, default=DEFAULT_TESTS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--per-test-timeout", type=int, default=6)
    ap.add_argument("--job-timeout", type=int, default=300)
    ap.add_argument("--mem-limit-mb", type=int, default=4096)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    problems = load_problems(args.tests)
    records = load_records(args.records)

    jobs, no_code = [], 0
    for rec in records:
        qid = str(rec.get("source_id"))
        prob = problems.get(qid)
        if prob is None:
            raise KeyError("record source_id %r not in %s" % (qid, args.tests))
        code = extract_code(rec.get("response") or "")
        if not code.strip():
            no_code += 1
        jobs.append(
            (
                qid,
                code,
                prob,
                args.python,
                args.per_test_timeout,
                args.job_timeout,
                args.mem_limit_mb,
            )
        )

    results, reasons = {}, Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(grade_one, j) for j in jobs]
        done = 0
        for f in as_completed(futs):
            qid, ok, reason = f.result()
            results[qid] = ok
            reasons[reason] += 1
            done += 1
            if done % 25 == 0 or done == len(futs):
                print("  [%d/%d] graded" % (done, len(futs)), flush=True)

    n = len(results)
    npass = sum(1 for v in results.values() if v)
    by_diff = defaultdict(lambda: [0, 0])
    by_plat = defaultdict(lambda: [0, 0])
    for qid, ok in results.items():
        by_diff[problems[qid]["difficulty"]][1] += 1
        by_diff[problems[qid]["difficulty"]][0] += int(ok)
        by_plat[problems[qid]["platform"]][1] += 1
        by_plat[problems[qid]["platform"]][0] += int(ok)

    metrics = {
        "n_problems": n,
        "pass@1": npass / n if n else 0.0,
        "passed": npass,
        "responses_without_code_block": no_code,
        "by_difficulty": {
            k: {"pass": v[0], "n": v[1], "rate": v[0] / v[1]}
            for k, v in sorted(by_diff.items())
        },
        "by_platform": {
            k: {"pass": v[0], "n": v[1], "rate": v[0] / v[1]}
            for k, v in sorted(by_plat.items())
        },
        "failure_reasons": dict(reasons),
    }
    out = args.out or args.records.parent / "lcb_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("problems: %d" % n)
    print("  pass@1: %.4f (%d/%d)" % (metrics["pass@1"], npass, n))
    for k, v in metrics["by_difficulty"].items():
        print("    %-8s %.4f (%d/%d)" % (k, v["rate"], v["pass"], v["n"]))
    print("  responses with no code block: %d/%d" % (no_code, n))
    print("  reasons: %s" % dict(reasons))
    print("wrote", out)


if __name__ == "__main__":
    main()
