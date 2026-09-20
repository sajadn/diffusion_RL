"""Corrected-confidence hooks must not activate for other training paths."""

import builtins
import copy
import sys
from types import SimpleNamespace

import pytest

from nemo_rl.algorithms.confidence_config import (
    clear_confidence_recording,
    configure_confidence_recording,
    get_confidence_experiment_config,
    is_corrected_confidence_trace,
    start_confidence_round,
)


def _policy(estimator):
    return {
        "logprob_estimation": estimator,
        "generation": {
            "backend": "vllm",
            "vllm_kwargs": {
                "diffusion_config": {"selection_policy": "confidence_threshold"}
            },
        },
    }


@pytest.mark.parametrize(
    "estimator",
    [
        None,
        {},
        *[
            {
                "type": name,
                "confidence_transition_correction": True,
                "confidence_experiment": {"mode": "filter"},
            }
            for name in (
                "ar",
                "diffu_grpo",
                "just_grpo",
                "block_just_grpo",
                "coupled_grpo",
                "espo_block_aware",
            )
        ],
        {
            "type": "trace_grpo",
            "confidence_transition_correction": False,
            "confidence_experiment": {"mode": "filter"},
        },
        {"type": "trace_grpo"},
    ],
)
def test_inactive_config_ignores_leftover_directories(estimator, monkeypatch, tmp_path):
    monkeypatch.setenv("NRL_CONFIDENCE_EXPERIMENT_DIR", str(tmp_path / "experiment"))
    monkeypatch.setenv("NRL_CONFIDENCE_AUDIT_DIR", str(tmp_path / "audit"))
    original_import = builtins.__import__

    def forbid_helpers(name, *args, **kwargs):
        assert name not in ("confidence_experiment", "confidence_audit_hooks"), name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_helpers)
    policy = _policy(estimator)
    before = copy.deepcopy(policy)
    assert not is_corrected_confidence_trace(policy)
    assert get_confidence_experiment_config(policy) is None
    configure_confidence_recording(policy, audit_requested=True)
    start_confidence_round(policy, 17)
    assert policy == before
    assert not list(tmp_path.iterdir())


def test_omitted_estimator_and_ar_without_diffusion_config(monkeypatch):
    monkeypatch.setenv("NRL_CONFIDENCE_EXPERIMENT_DIR", "/unused/stale/path")
    for policy in (
        {},
        {"generation": {"backend": "vllm", "vllm_kwargs": {"diffusion_config": None}}},
    ):
        before = copy.deepcopy(policy)
        configure_confidence_recording(policy, audit_requested=True)
        start_confidence_round(policy, 0)
        assert policy == before


@pytest.mark.parametrize("mode", ["filter", "importance"])
def test_active_experiment_starts_round_and_only_enables_its_hook(
    mode, monkeypatch, tmp_path
):
    monkeypatch.setenv("NRL_CONFIDENCE_EXPERIMENT_DIR", str(tmp_path))
    monkeypatch.setenv("NRL_CONFIDENCE_AUDIT_DIR", str(tmp_path / "unused"))
    calls = []
    monkeypatch.setitem(
        sys.modules, "confidence_experiment", SimpleNamespace(start_round=calls.append)
    )
    experiment = {"mode": mode}
    policy = _policy(
        {
            "type": "trace_grpo",
            "confidence_transition_correction": True,
            "confidence_experiment": experiment,
        }
    )
    configure_confidence_recording(policy, audit_requested=False)
    assert get_confidence_experiment_config(policy) is experiment
    decode = policy["generation"]["vllm_kwargs"]["diffusion_config"]
    assert decode["record_confidence_experiment"] is True
    assert "record_confidence_audit" not in decode
    start_confidence_round(policy, 12)
    assert calls == [12]


def test_audit_only_does_not_activate_stale_experiment(monkeypatch):
    monkeypatch.setenv("NRL_CONFIDENCE_EXPERIMENT_DIR", "/unused/stale/path")
    policy = _policy({"type": "trace_grpo", "confidence_transition_correction": True})
    configure_confidence_recording(policy, audit_requested=True)
    decode = policy["generation"]["vllm_kwargs"]["diffusion_config"]
    assert decode["record_confidence_audit"] is True
    assert "record_confidence_experiment" not in decode
    start_confidence_round(policy, 7)


def test_inherited_capture_flags_are_removed_from_other_estimators():
    policy = _policy({"type": "diffu_grpo"})
    generation = policy["generation"]
    decode = generation["vllm_kwargs"]["diffusion_config"]
    decode.update(record_confidence_experiment=True, record_confidence_audit=True)
    generation["vllm_val_dllm_overrides"] = {
        "vllm_kwargs": copy.deepcopy(generation["vllm_kwargs"])
    }
    generation["vllm_val_dllm_variants"] = {
        "rollout": {"vllm_kwargs": copy.deepcopy(generation["vllm_kwargs"])}
    }
    configure_confidence_recording(policy, audit_requested=True)
    for cfg in [
        generation,
        generation["vllm_val_dllm_overrides"],
        generation["vllm_val_dllm_variants"]["rollout"],
    ]:
        assert cfg["vllm_kwargs"]["diffusion_config"] == {
            "selection_policy": "confidence_threshold"
        }


def test_validation_copy_can_disable_capture_without_changing_training(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("NRL_CONFIDENCE_EXPERIMENT_DIR", str(tmp_path))
    policy = _policy(
        {
            "type": "trace_grpo",
            "confidence_transition_correction": True,
            "confidence_experiment": {"mode": "filter"},
        }
    )
    configure_confidence_recording(policy, audit_requested=True)
    val = copy.deepcopy(policy["generation"])
    clear_confidence_recording(val)
    assert val["vllm_kwargs"]["diffusion_config"] == {
        "selection_policy": "confidence_threshold"
    }
    assert policy["generation"]["vllm_kwargs"]["diffusion_config"][
        "record_confidence_experiment"
    ]
    assert policy["generation"]["vllm_kwargs"]["diffusion_config"][
        "record_confidence_audit"
    ]


def test_active_experiment_requires_directory(monkeypatch):
    monkeypatch.delenv("NRL_CONFIDENCE_EXPERIMENT_DIR", raising=False)
    policy = _policy(
        {
            "type": "trace_grpo",
            "confidence_transition_correction": True,
            "confidence_experiment": {"mode": "importance"},
        }
    )
    with pytest.raises(ValueError, match="NRL_CONFIDENCE_EXPERIMENT_DIR"):
        configure_confidence_recording(policy, audit_requested=False)
    with pytest.raises(ValueError, match="NRL_CONFIDENCE_EXPERIMENT_DIR"):
        start_confidence_round(policy, 0)


@pytest.mark.parametrize("variant", [None, {}])
def test_validation_variant_without_overrides_is_preserved(variant):
    policy = _policy({"type": "diffu_grpo"})
    policy["generation"]["vllm_val_dllm_variants"] = {"rollout": variant}
    before = copy.deepcopy(policy)
    configure_confidence_recording(policy, audit_requested=True)
    assert policy == before
