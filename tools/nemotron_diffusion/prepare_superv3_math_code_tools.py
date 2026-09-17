# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a reproducible nine-subset Superv3 mixture with grouped holdout splits."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

import orjson


PREFIX = "super_v3_lcsft_step1000_"
SUBSETS = {
    "skyworks_no_omni": ("math", "math_with_judge_simple_agent"),
    "dapo17k": ("math", "math_with_judge_simple_agent"),
    "math_holdout_small_igor": ("math", "math_with_judge_simple_agent"),
    "math_turing": ("math", "math_with_judge_simple_agent"),
    "math_tir_skywork_no_omni": ("math_tools", "ns_tools_simple_agent"),
    "math_tir_holdout_small_igor": ("math_tools", "ns_tools_simple_agent"),
    "math_tir_turing": ("math_tools", "ns_tools_simple_agent"),
    "comp_coding": ("coding", "code_gen_simple_agent"),
    "nemotronx_code": ("coding", "code_gen_simple_agent"),
}
DATASET_PATTERN = re.compile(rb'"dataset"\s*:\s*"([^"\\]+)"')


def problem_id(row):
    """Group identical normalized problem text across subsets and tool variants."""
    text = row.get("question") or row.get("problem")
    if not text:
        messages = row["responses_create_params"]["input"]
        if isinstance(messages, str):
            text = messages
        else:
            users = []
            for message in messages:
                if message.get("role") != "user":
                    continue
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(part.get("text", "") for part in content)
                users.append(content)
            text = "\n".join(users)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing problem text")
    text = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(text.encode()).hexdigest()


def validate_row(row, subset):
    """Return a rejection reason for ungradable or incorrectly routed rows."""
    category, agent = SUBSETS[subset]
    if row.get("agent_ref") != {"type": "responses_api_agents", "name": agent}:
        return "unexpected agent_ref"
    params = row.get("responses_create_params") or {}
    if not params.get("input"):
        return "missing input"
    if category == "coding":
        tests = (row.get("verifier_metadata") or {}).get("unit_tests")
        if not isinstance(tests, dict):
            return "missing unit_tests"
        inputs, outputs = tests.get("inputs"), tests.get("outputs")
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            return "invalid test arrays"
        if not inputs or len(inputs) != len(outputs):
            return "empty or unpaired tests"
        if not all(isinstance(item, str) for item in inputs + outputs):
            return "non-string test cases"
    elif (
        row.get("expected_answer") is None or str(row["expected_answer"]).strip() == ""
    ):
        return "missing expected_answer"
    if category == "math_tools":
        names = {tool.get("name") for tool in params.get("tools", [])}
        if "stateful_python_code_exec" not in names:
            return "missing Python tool"
        if row.get("verifier_type") != "math_with_judge":
            return "unexpected math tool verifier"
    elif params.get("tools"):
        return "unexpected tools in non-tool subset"
    return None


def holdout_groups(groups_by_subset, count, seed):
    """Choose deterministic per-subset holdouts, keeping shared groups together."""
    selected = set()
    for subset in sorted(groups_by_subset):
        groups = groups_by_subset[subset]
        if len(groups) <= count:
            raise ValueError(f"{subset}: need more than {count} unique problems")
        needed = max(0, count - len(groups & selected))
        ranked = sorted(
            groups - selected,
            key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest(),
        )
        selected.update(ranked[:needed])
    return selected


def build(source, output, validation_per_subset, seed):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if validation_per_subset < 1:
        raise ValueError("validation_per_subset must be positive")
    source_stat = source.stat()
    source_hash = hashlib.sha256()
    source_counts, rejected = Counter(), Counter()
    groups = defaultdict(set)
    records = []
    offset = 0
    with source.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            source_hash.update(raw)
            # Test metadata can occupy hundreds of MB. Read dataset tags from
            # the tail first; parse full JSON only for the selected nine subsets.
            matches = list(DATASET_PATTERN.finditer(raw[-16384:]))
            if not matches:
                matches = list(DATASET_PATTERN.finditer(raw))
            dataset = matches[-1][1].decode() if matches else ""
            subset = dataset.removeprefix(PREFIX)
            if dataset.startswith(PREFIX) and subset in SUBSETS:
                row = orjson.loads(raw)
                if row["dataset"] != dataset:
                    raise ValueError(f"Ambiguous dataset tag on line {line_number}")
                source_counts[subset] += 1
                reason = validate_row(row, subset)
                if reason:
                    rejected[f"{subset}: {reason}"] += 1
                else:
                    group = problem_id(row)
                    groups[subset].add(group)
                    records.append((offset, len(raw), subset, group))
            offset += len(raw)
            if line_number % 50000 == 0:
                print(
                    f"Scanned {line_number} rows; selected {len(records)}", flush=True
                )
    if set(groups) != set(SUBSETS):
        raise ValueError(f"Missing subsets: {set(SUBSETS) - set(groups)}")
    selected = holdout_groups(groups, validation_per_subset, seed)
    counts = {split: Counter() for split in ["train", "val"]}
    hashes = {split: hashlib.sha256() for split in counts}
    split_groups = {split: set() for split in counts}
    for _, _, subset, group in records:
        split = "val" if group in selected else "train"
        counts[split][subset] += 1
        split_groups[split].add(group)
    if any(not counts["train"][subset] for subset in SUBSETS):
        raise ValueError("Holdout leaves a subset with no training rows")
    assert split_groups["train"].isdisjoint(split_groups["val"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        with (
            source.open("rb") as stream,
            (temporary / "train.jsonl").open("wb") as train,
            (temporary / "val.jsonl").open("wb") as val,
            (temporary / "split_index.jsonl").open("w") as index,
        ):
            for position, length, subset, group in records:
                split = "val" if group in selected else "train"
                stream.seek(position)
                raw = stream.read(length)
                if len(raw) != length:
                    raise ValueError("Source changed during build")
                if not raw.endswith(b"\n"):
                    raw += b"\n"
                (val if split == "val" else train).write(raw)
                hashes[split].update(raw)
                index.write(
                    json.dumps(
                        {
                            "source_offset": position,
                            "subset": subset,
                            "problem_id": group,
                            "split": split,
                        }
                    )
                    + "\n"
                )
        final_stat = source.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (
            final_stat.st_size,
            final_stat.st_mtime_ns,
        ):
            raise ValueError("Source changed during build")
        manifest = {
            "source": str(source.resolve()),
            "source_sha256": source_hash.hexdigest(),
            "source_rows": line_number,
            "seed": seed,
            "validation_unique_problems_per_subset_minimum": validation_per_subset,
            "grouping": "SHA256 of NFKC/whitespace-normalized question/problem, falling back to user input; grouped across all subsets",
            "sampling": "Original source proportions; no resampling or row duplication",
            "source_subset_counts": dict(source_counts),
            "rejected": dict(rejected),
            "subsets": {
                name: {"category": category, "agent": agent}
                for name, (category, agent) in SUBSETS.items()
            },
            "counts": {split: dict(count) for split, count in counts.items()},
            "totals": {split: sum(count.values()) for split, count in counts.items()},
            "unique_problems": {
                split: len(value) for split, value in split_groups.items()
            },
            "output_sha256": {
                split: value.hexdigest() for split, value in hashes.items()
            },
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-per-subset", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    build(args.source, args.output, args.validation_per_subset, args.seed)
