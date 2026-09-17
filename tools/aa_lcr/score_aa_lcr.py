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
r"""Offline AA-LCR grader for standalone-evaluator records.jsonl.

AA-LCR answers are free-form, so the publisher grades with an LLM equality checker
(their prompt is reproduced verbatim below; they use Qwen3 235B A22B 2507
non-reasoning). Two backends:

  --judge openai   POST to any OpenAI-compatible /v1/chat/completions endpoint.
                   This is the faithful path; the judge model is whatever you serve.
  --judge heuristic  No model. Conservative normalized containment check. This is
                   NOT the official metric and can have false positives and false
                   negatives; label it as a heuristic approximation.

Usage:
  score_aa_lcr.py --records <outdir>/records.jsonl --judge heuristic
  score_aa_lcr.py --records ... --judge openai --judge-base-url http://host:port \\
                  --judge-model <name>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.benchmark_eval.scoring import (
    load_records,
    resolve_thinking_mode,
    strip_thinking,
)

JUDGE_PROMPT = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT."""


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = s.replace("$", " ").replace(",", "")
    s = re.sub(r"[^a-z0-9\. ]+", " ", s)
    return " ".join(s.split())


def heuristic_correct(candidate: str, official: str) -> bool:
    """Every semicolon-separated criterion must appear in the candidate.

    This is an approximation, not the official metric or a guaranteed lower bound.
    """
    cand = norm(candidate)
    if not cand:
        return False
    parts = [p for p in str(official).split(";") if p.strip()]
    for p in parts:
        np = norm(p)
        if not np:
            continue
        # numbered-list answers ("1. Airline Industry (12)") -> require each token run
        if np not in cand:
            return False
    return True


def judge_openai(
    base_url: str,
    model: str,
    question: str,
    official: str,
    candidate: str,
    timeout: int,
    api_key: str | None,
) -> bool | None:
    import requests

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    official_answer=official,
                    candidate_answer=candidate,
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 8,
    }
    headers = {}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    r = requests.post(
        base_url + "/chat/completions", json=body, headers=headers, timeout=timeout
    )
    r.raise_for_status()
    text = (r.json()["choices"][0]["message"].get("content") or "").strip().upper()
    if text == "INCORRECT":
        return False
    if text == "CORRECT":
        return True
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--judge", choices=("heuristic", "openai"), default="heuristic")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-timeout", type=int, default=120)
    ap.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")
    args = ap.parse_args()

    if args.judge == "openai" and not (args.judge_base_url and args.judge_model):
        raise SystemExit("--judge openai requires --judge-base-url and --judge-model")

    records = load_records(args.records)
    thinking_on = resolve_thinking_mode(args.records, args.thinking)
    api_key = os.environ.get(args.judge_api_key_env)

    results, unparsed = {}, 0
    for rec in records:
        qid = str(rec.get("source_id"))
        cand = (rec.get("response") or "").strip()
        if thinking_on:
            cand, _ = strip_thinking(cand)
        official = rec.get("gold") or ""
        if not official.strip():
            raise ValueError(f"Empty reference answer for {qid}")
        question = rec.get("raw_answer") or ""  # loader stores the bare question here
        if args.judge == "heuristic":
            ok = heuristic_correct(cand, official)
        else:
            v = judge_openai(
                args.judge_base_url,
                args.judge_model,
                question,
                official,
                cand,
                args.judge_timeout,
                api_key,
            )
            if v is None:
                unparsed += 1
                v = False
            ok = v
        results[qid] = ok

    n = len(results)
    npass = sum(1 for v in results.values() if v)
    by_cat = defaultdict(lambda: [0, 0])
    for rec in records:
        qid = str(rec.get("source_id"))
        cat = qid.split("-")[0]
        by_cat[cat][1] += 1
        by_cat[cat][0] += int(results.get(qid, False))

    metrics = {
        "n_questions": n,
        "accuracy": npass / n if n else 0.0,
        "correct": npass,
        "judge": args.judge,
        "judge_model": args.judge_model,
        "judge_unparsed_replies": unparsed,
        "note": (
            "heuristic approximation, NOT the official AA-LCR metric or a guaranteed lower bound"
            if args.judge == "heuristic"
            else "official AA-LCR equality-checker prompt"
        ),
        "by_document_set": {
            k: {"pass": v[0], "n": v[1]} for k, v in sorted(by_cat.items())
        },
    }
    out = args.out or args.records.parent / "aa_lcr_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("questions: %d" % n)
    print(
        "  accuracy (%s judge): %.4f (%d/%d)"
        % (args.judge, metrics["accuracy"], npass, n)
    )
    if args.judge == "heuristic":
        print("  NOTE: heuristic approximation, not the official metric")
    if unparsed:
        print("  judge replies not parseable: %d" % unparsed)
    print("wrote", out)


if __name__ == "__main__":
    main()
