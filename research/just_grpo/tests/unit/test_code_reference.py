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


@pytest.mark.parametrize("mode", ["long", "smoke"])
def test_code_recipe_sampling_uses_llada_validation_adapter(monkeypatch, mode):
    from just_grpo.config import validate_config
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    monkeypatch.setenv("LLADA_CODE_MAX_PROMPT_TOKENS", "1150")
    monkeypatch.setenv("LLADA_CODE_SEQUENCE_LENGTH", "1536")
    monkeypatch.setenv("LLADA_CODE_DATA_DIR", "/data")
    monkeypatch.setenv("JUSTGRPO_REFERENCE_DIR", "/reference")
    monkeypatch.setenv("POLICY_PYTHON", "/python")
    register_omegaconf_resolvers()
    cfg = load_config(
        ROOT
        / "configs/recipes"
        / f"just_grpo-acecode-llada8b-2n8g-megatron-humaneval-{mode}.yaml"
    )
    diffusion = validate_config(cfg)
    # Native GRPO rejects validation-only temperature overrides for Megatron.
    assert cfg.policy.generation.val_temperature == cfg.policy.generation.temperature
    # The LLaDA validation context owns the actual held-out decoding settings.
    assert diffusion.sampling.temperature == 1.0
    assert diffusion.validation_sampling.temperature == 0.0
    assert diffusion.validation_sampling.block_size == 32
    assert diffusion.validation_sampling.max_steps == 32
    cfg.policy.generation.val_temperature = 0.0
    with pytest.raises(ValueError, match="just_grpo.validation_sampling"):
        validate_config(cfg)


def test_training_reward_ignores_stdout_from_generated_code():
    # The reference guard disables print(), but generated code can write directly.
    rows = [
        {
            "kind": "acecode",
            "test_cases": ["assert f() == 1"],
            "completion": (
                "```python\nimport sys\n"
                "sys.stdout.write('candidate output\\n')\n"
                "sys.stdout.flush()\n"
                "def f(): return 1\n```"
            ),
        }
    ]
    assert score_chunk(rows, config()) == [2.0]


def test_empty_reward_batch():
    assert score_chunk([], config()) == []


def test_missing_reward_file_fails_even_with_json_stdout(monkeypatch, tmp_path):
    from just_grpo.environments import code_reference

    worker = tmp_path / "code_reference_worker.py"
    worker.write_text("print('[2.0]')\n")
    monkeypatch.setattr(code_reference, "__file__", str(tmp_path / "adapter.py"))
    with pytest.raises(RuntimeError, match="did not produce a valid reward result"):
        score_chunk([{"kind": "acecode", "completion": ""}], config())


def test_mbpp_is_binary_without_training_format_reward():
    problem = {
        "kind": "mbpp",
        "task_id": 11,
        "test": ["assert f(1) == 2", "assert f(3) == 4"],
    }
    rows = [
        {**problem, "completion": completion}
        for completion in [
            "def f(x): return x + 1",
            "```python\ndef f(x): return x + 1\n```",
            "def f(x): return 0",
            "def f(:",
        ]
    ]
    assert score_chunk(rows, config()) == [1, 1, 0, 0]


def test_mbpp_preparation_matches_reference():
    import ast

    reference = Path(os.environ["JUSTGRPO_REFERENCE_DIR"])
    source = ast.parse((reference / "eval.py").read_text())
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_format_mbpp_prompt"
    )
    namespace = {"ast": ast}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "reference_mbpp_prompt",
            "exec",
        ),
        namespace,
    )
    raw = reference / "datasets/mbpp.jsonl"
    records = prepare.mbpp_records(raw, reference / "datasets/mbpp_test.jsonl")
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line][10:510]
    assert len(records) == 500
    for row, record in zip(rows, records, strict=True):
        assert record["problem"] == namespace["_format_mbpp_prompt"](row)
        answer = json.loads(record["answer"])
        assert answer["task_id"] == row["task_id"]
        assert answer["kind"] == "mbpp" and answer["test"]
        assert "code" not in answer


def test_response_dataset_explicit_task_name(tmp_path):
    from nemo_rl.data.datasets.response_datasets.response_dataset import ResponseDataset

    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"problem": "Implement f.", "answer": "tests"}) + "\n")
    named = ResponseDataset(
        str(path),
        input_key="problem",
        output_key="answer",
        split="train",
        task_name="mbpp",
    )
    assert named.task_name == "mbpp"
    assert named.dataset["task_name"] == ["mbpp"]
    original = ResponseDataset(
        str(path), input_key="problem", output_key="answer", split="train"
    )
    assert original.task_name == f"{tmp_path.name}-questions"
