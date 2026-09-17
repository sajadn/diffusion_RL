# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).resolve().parents[3]
    / "tools/nemotron_diffusion/prepare_superv3_math_code_tools.py"
)
spec = importlib.util.spec_from_file_location("prepare_superv3", MODULE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def make_row(subset, question):
    category, agent = prepare.SUBSETS[subset]
    row = {
        "dataset": prepare.PREFIX + subset,
        "agent_ref": {"type": "responses_api_agents", "name": agent},
        "responses_create_params": {"input": [{"role": "user", "content": question}]},
    }
    if category == "coding":
        row["verifier_metadata"] = {"unit_tests": {"inputs": ["1"], "outputs": ["2"]}}
    else:
        row.update(question=question, expected_answer="2")
    if category == "math_tools":
        row["verifier_type"] = "math_with_judge"
        row["responses_create_params"]["tools"] = [
            {"type": "function", "name": "stateful_python_code_exec"}
        ]
    return row


def test_problem_grouping_across_tool_variants():
    plain = make_row("dapo17k", "What is  1 + 1?")
    tool = make_row("math_tir_turing", "What is 1 + 1?\n")
    del tool["question"]
    assert prepare.problem_id(plain) == prepare.problem_id(tool)
    assert prepare.problem_id(plain) != prepare.problem_id(
        make_row("dapo17k", "What is 2 + 2?")
    )


def test_grouped_holdout_is_deterministic():
    groups = {"a": {"shared", "a1", "a2", "a3"}, "b": {"shared", "b1", "b2", "b3"}}
    selected = prepare.holdout_groups(groups, 2, 42)
    assert selected == prepare.holdout_groups(
        dict(reversed(list(groups.items()))), 2, 42
    )
    assert all(len(values & selected) >= 2 for values in groups.values())
    with pytest.raises(ValueError, match="unique problems"):
        prepare.holdout_groups({"a": {"one"}}, 1, 42)


def test_build_preserves_rows_routes_and_disjoint_groups(tmp_path):
    rows = [
        make_row(subset, f"problem {index}")
        for subset in prepare.SUBSETS
        for index in range(8)
    ]
    invalid = make_row("comp_coding", "invalid coding problem")
    invalid["verifier_metadata"]["unit_tests"]["outputs"] = []
    rows += [invalid, {"dataset": "unrelated"}]
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "prepared"
    prepare.build(source, output, 2, 42)
    manifest = json.loads((output / "manifest.json").read_text())
    loaded = {
        split: [
            json.loads(line)
            for line in (output / f"{split}.jsonl").read_text().splitlines()
        ]
        for split in ["train", "val"]
    }
    assert manifest["totals"] == {"train": 54, "val": 18}
    assert manifest["rejected"] == {"comp_coding: empty or unpaired tests": 1}
    assert {prepare.problem_id(row) for row in loaded["train"]}.isdisjoint(
        {prepare.problem_id(row) for row in loaded["val"]}
    )
    assert sorted(map(json.dumps, loaded["train"] + loaded["val"])) == sorted(
        map(json.dumps, rows[:-2])
    )
    assert all(len((output / f"{split}.jsonl").read_bytes()) > 0 for split in loaded)
    with pytest.raises(FileExistsError):
        prepare.build(source, output, 2, 42)


def test_rejects_wrong_agents_and_missing_tools():
    row = make_row("math_tir_turing", "problem")
    row["responses_create_params"]["tools"] = []
    assert prepare.validate_row(row, "math_tir_turing") == "missing Python tool"
    row["agent_ref"]["name"] = "wrong"
    assert prepare.validate_row(row, "math_tir_turing") == "unexpected agent_ref"
