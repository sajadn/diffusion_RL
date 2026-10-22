"""Check reference code rewards, held-out grading, and prompt preservation."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from just_grpo.environments.code_reference import CodeReferenceConfig, score_chunk

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prepare_code_data", ROOT / "tools/prepare_code_data.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def config():
    return CodeReferenceConfig(
        reference_dir=os.environ["JUSTGRPO_REFERENCE_DIR"],
        reference_revision="1a2fddb5c6655597e63081c0af5ebb718a849f39",
        python_executable=os.environ["POLICY_PYTHON"],
        num_workers=1,
        chunk_size=16,
        timeout_seconds=180,
    )


def test_training_reward_matches_reference():
    responses = [
        "def f(): return 1",
        "```python\ndef f(:\n```",
        "```python\ndef f(): return 0\n```",
        "```python\ndef f(x): return x\n```",
        "```python\ndef f(x): return x + 1\n```",
    ]
    tests = [
        ["assert f() == 1"],
        ["assert f() == 1"],
        ["assert f() == 1"],
        ["assert f(1) == 1", "assert f(2) == 3"],
        ["assert f(1) == 2", "assert f(2) == 3"],
    ]
    rows = [
        {"kind": "acecode", "test_cases": test, "completion": text}
        for text, test in zip(responses, tests, strict=True)
    ]
    assert score_chunk(rows, config()) == [0, 0.5, 1, 1.5, 2]


def test_humaneval_is_binary_without_format_reward():
    problem = {
        "kind": "humaneval",
        "task_id": "HumanEval/test",
        "prompt": 'def f(x):\n    """Add one."""\n',
        "test": "def check(candidate):\n    assert candidate(1) == 2",
        "entry_point": "f",
    }
    rows = [
        {**problem, "completion": completion}
        for completion in [
            "    return x + 1\n",
            "```python\ndef f(x):\n    return x + 1\n```",
            "```python\ndef f(x):\n    return 0\n```",
        ]
    ]
    assert score_chunk(rows, config()) == [1, 1, 0]


def test_preparation_preserves_reference_prompts():
    record = prepare.acecode_record(
        {"question": "Implement f.", "test_cases": ["assert f() == 1"]}
    )
    assert record["problem"] == "Implement f.\nTest cases: assert f() == 1"
    row = {
        "task_id": "HumanEval/0",
        "prompt": 'def f():\n    """f"""\n',
        "entry_point": "f",
        "test": "def check(f): pass",
    }
    record = prepare.humaneval_record(row)
    assert json.loads(record["answer"])["prompt"] == row["prompt"]
    assert record["problem"].endswith('def f():\n    """f"""\n```\n')


def test_unknown_task_fails_instead_of_zero_score():
    with pytest.raises(RuntimeError, match="Unsupported task"):
        score_chunk([{"kind": "unknown", "completion": ""}], config())
