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
