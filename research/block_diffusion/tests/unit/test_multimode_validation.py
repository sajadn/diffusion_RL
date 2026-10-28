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
"""Mode isolation, current-weight refits, matched data, and failure cleanup."""

import asyncio
import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from just_grpo.config import validate_config
from block_diffusion.generation.reference_vllm import (
    ReferenceVllmAsyncWorkerImpl,
    ReferenceVllmWorkerImpl,
    _diffusion_counters,
)
from block_diffusion.generation.validation import (
    MultiModeValidation,
    build_validation_config,
    validation_variants,
)
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from test_sudoku_and_config import load_reference_vllm

RESEARCH = Path(__file__).resolve().parents[3]
OVERRIDES = RESEARCH / "block_diffusion/configs/validation/ar_diffusion.yaml"


def variants():
    return OmegaConf.to_container(
        OmegaConf.load(OVERRIDES).policy.generation, resolve=True
    )


@pytest.mark.parametrize(
    "recipe",
    [
        "just_grpo/configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-vllm-dualval-long.yaml",
        "just_grpo/configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-vllm-async-dualval-long.yaml",
    ],
)
def test_runnable_dual_validation_recipes(recipe):
    register_omegaconf_resolvers()
    config = load_config(RESEARCH / recipe)
    validate_config(config)
    assert list(config.policy.generation.vllm_val_dllm_variants) == [
        "diffusion_conf09",
        "ar",
    ]


def test_build_modes_does_not_mutate_rollout_or_keep_training_channels():
    config = load_reference_vllm()
    rollout = OmegaConf.to_container(config.policy.generation, resolve=True)
    rollout["vllm_kwargs"]["diffusion_config"]["return_reveal_steps"] = True
    original = copy.deepcopy(rollout)
    parsed = validation_variants(variants())
    diffusion = build_validation_config(
        rollout, "diffusion_conf09", parsed["diffusion_conf09"]
    )
    ar = build_validation_config(rollout, "ar", parsed["ar"])
    assert rollout == original
    assert ar["vllm_kwargs"]["diffusion_config"] is None
    assert ar["vllm_kwargs"]["hf_overrides"]["architectures"] == [
        "NemotronLabsDiffusionForCausalLM"
    ]
    assert diffusion["vllm_kwargs"]["diffusion_config"]["max_denoising_steps"] == 16
    assert not diffusion["vllm_kwargs"]["diffusion_config"]["return_reveal_steps"]
    assert ar["refit_namespace"] != diffusion["refit_namespace"]


@pytest.mark.parametrize("runtime", ["megatron", "upstream_vllm"])
def test_unsupported_runtimes_fail_before_setup(runtime):
    config = OmegaConf.merge(
        load_reference_vllm(), {"policy": {"generation": variants()}}
    )
    config.just_grpo.runtime = runtime
    with pytest.raises(ValueError, match="reference_vllm|Upstream generation"):
        validate_config(config)


@pytest.mark.parametrize(
    "mutation", ["architecture", "budget", "decode", "name", "temperature"]
)
def test_invalid_variants_fail_fast(mutation):
    raw = variants()
    modes = raw["vllm_val_dllm_variants"]
    if mutation == "architecture":
        modes["ar"]["vllm_kwargs"]["hf_overrides"]["architectures"] = [
            "NemotronLabsDiffusionModel"
        ]
    elif mutation == "budget":
        del modes["ar"]["vllm_cfg"]["gpu_memory_utilization"]
    elif mutation == "decode":
        del modes["diffusion_conf09"]["vllm_kwargs"]["diffusion_config"][
            "max_denoising_steps"
        ]
    elif mutation == "name":
        modes["../ar"] = modes.pop("ar")
    else:
        modes["diffusion_conf09"]["temperature"] = 0.5
    with pytest.raises(ValueError):
        validation_variants(raw)


@pytest.mark.parametrize("buffer_size_gb", [None, 0.25])
def test_validation_refits_honor_buffer_size(monkeypatch, buffer_size_gb):
    from nemo_rl.weight_sync.ipc_weight_synchronizer import IPCWeightSynchronizer

    monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.3")
    runner = MultiModeValidation(validation_variants(variants()), colocated=True)
    runner.generations = {name: Mock(cfg={}) for name in runner.variants}
    policy = Mock()
    policy.get_free_memory_bytes.return_value = 8 * 1024**3
    runner.initialize(
        policy,
        train_cluster=Mock(),
        inference_cluster=Mock(),
        refit_buffer_size_gb=buffer_size_gb,
    )
    expected = int((0.25 if buffer_size_gb is not None else 8 * 0.3) * 1024**3)
    for group in runner.generations.values():
        synchronizer = group.weight_synchronizer
        assert isinstance(synchronizer, IPCWeightSynchronizer)
        assert synchronizer._compute_buffer_size() == expected
        group.prepare_refit_info.assert_called_once()
    if buffer_size_gb is not None:
        policy.get_free_memory_bytes.assert_not_called()


@pytest.mark.parametrize("fail_mode", [None, "diffusion_conf09", "ar"])
@pytest.mark.parametrize("colocated", [True, False])
@pytest.mark.parametrize("llada", [False, True])
def test_modes_refit_same_weights_and_prompts_and_restore_on_error(
    monkeypatch, fail_mode, colocated, llada, tmp_path
):
    events = []
    raw = variants()
    if llada:
        raw["vllm_val_dllm_variants"] = {
            "diffusion_conf09": raw["vllm_val_dllm_variants"]["diffusion_conf09"]
        }
        raw["vllm_val_dllm_variants"]["diffusion_conf09"]["vllm_kwargs"][
            "hf_overrides"
        ] = {"architectures": ["LLaDAModelLM"]}
        if fail_mode == "ar":
            pytest.skip("LLaDA has no causal-AR variant")
    runner = MultiModeValidation(validation_variants(raw), colocated=colocated)
    runner.policy = Mock()
    runner.policy.prepare_for_lp_inference.side_effect = lambda: events.append("onload")
    runner.policy.offload_after_refit.side_effect = lambda: events.append("offload")
    rollout = Mock()
    rollout.sleep.side_effect = lambda: events.append("sleep rollout") or True
    rollout.wake_up.side_effect = lambda: events.append("wake rollout") or True
    for name in runner.variants:
        group = Mock(cfg={"name": name})
        group.get_step_metrics.return_value = (
            {"vllm/denoising_nfe": 12.0, "vllm/tokens_per_denoising_nfe": 4.0}
            if name == "diffusion_conf09"
            else {}
        )
        group.sleep.side_effect = lambda n=name: events.append(f"sleep {n}") or True
        group.wake_up.side_effect = lambda n=name: events.append(f"wake {n}") or True
        runner.generations[name] = group

    class Loader:
        cursor = 7

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.cursor = state["cursor"]

    loader = Loader()
    master = Mock()
    master.logger = {"log_dir": str(tmp_path)}
    master.model_copy.side_effect = lambda **kw: SimpleNamespace(policy={})
    logger = Mock()
    controller = ModuleType("nemo_rl.algorithms.grpo")

    def refit(policy, group, **kwargs):
        assert policy is runner.policy
        assert kwargs == {"colocated_inference": colocated}
        assert events[-1] == ("onload" if colocated else "wake " + group.cfg["name"])
        events.append("refit " + group.cfg["name"])

    def validate(group, data, tokenizer, env, **kwargs):
        name = group.cfg["name"]
        assert data.cursor == 7
        data.cursor += 10
        assert kwargs["master_config"].policy["generation"] is group.cfg
        assert kwargs["logger"] is (logger if name == "diffusion_conf09" else None)
        events.append("validate " + name)
        if name == fail_mode:
            raise ValueError("validation failed")
        return {
            "accuracy": 0.7 if name == "ar" else 0.8,
            "avg_length": 4,
            "num_samples": 2,
        }, {"total_validation_time": 2}

    controller.refit_policy_generation = refit
    controller.validate = validate
    monkeypatch.setitem(sys.modules, controller.__name__, controller)
    if fail_mode:
        with pytest.raises(ValueError, match="validation failed"):
            runner(rollout, loader, None, None, 10, master, logger=logger)
    else:
        metrics, timings = runner(
            rollout, loader, None, None, 10, master, logger=logger
        )
        assert metrics["accuracy"] == metrics["accuracy/diffusion_conf09"] == 0.8
        if llada:
            assert (tmp_path / "accuracy_vs_nfe_step10.png").exists()
            rows = logger.log_table.call_args.args[1]
            assert rows == [[10, "diffusion_conf09", 0.9, 6.0, 0.8, 2]]
        else:
            assert metrics["accuracy/ar"] == 0.7
        assert timings["total_validation_time"] == (2 if llada else 4)
        assert metrics["vllm/denoising_nfe/diffusion_conf09"] == 12
        assert metrics["vllm/denoising_nfe"] == 12
        assert "vllm/denoising_nfe/ar" not in metrics
        for group in runner.generations.values():
            group.snapshot_step_metrics.assert_called_once()
            group.get_step_metrics.assert_called_once()
    assert loader.cursor == 7
    assert events[0] == "sleep rollout"
    assert events[-1] == "wake rollout"
    if colocated:
        assert events[-2] == "offload"
    else:
        runner.policy.offload_after_refit.assert_not_called()
        runner.policy.prepare_for_lp_inference.assert_not_called()
    assert all(
        events[events.index("validate " + name) + 1] == "sleep " + name
        for name in runner.variants
        if "validate " + name in events
    )


def test_ar_worker_uses_native_temperature_and_no_diffusion_metadata(monkeypatch):
    worker = object.__new__(ReferenceVllmWorkerImpl)
    worker.cfg = {"vllm_kwargs": {"diffusion_config": None}}
    params = SimpleNamespace(temperature=0.6, logprobs=0)
    monkeypatch.setattr(
        VllmGenerationWorkerImpl, "_build_sampling_params", Mock(return_value=params)
    )
    assert worker._build_sampling_params(greedy=False, stop_strings=None) is params
    assert params.temperature == 0.6
    assert worker._completion_metadata(None, input_length=3, padded_length=6) == {}
    create = Mock()
    monkeypatch.setattr(VllmGenerationWorkerImpl, "_create_engine", create)
    worker._create_engine({"diffusion_config": None})
    create.assert_called_once_with({"diffusion_config": None})


def test_ipc_sockets_are_isolated_reused_and_closed(monkeypatch):
    import nemo_rl.models.policy.workers.base_policy_worker as module

    context = Mock()
    sockets = [Mock(), Mock(), Mock()]
    context.socket.side_effect = sockets
    monkeypatch.setattr(module.zmq, "Context", Mock(return_value=context))
    worker = AbstractPolicyWorker()
    worker.report_device_id = lambda: "GPU-test"
    for name, socket in zip([None, "vllm_val_ar", "vllm_val_diffusion"], sockets):
        worker.maybe_init_zmq(name)
        assert worker.zmq_socket is socket
        socket.bind.assert_called_once_with(worker.get_zmq_address(name))
    worker.maybe_init_zmq()
    assert worker.zmq_socket is sockets[0]
    assert context.socket.call_count == 3
    assert worker.shutdown()
    for socket in sockets:
        socket.close.assert_called_once()
    context.term.assert_called_once()


@pytest.mark.parametrize("namespace", [None, "vllm_val_ar"])
def test_ipc_synchronizer_forwards_namespace_only_when_present(monkeypatch, namespace):
    import nemo_rl.weight_sync.ipc_weight_synchronizer as module

    generation = Mock(cfg={"refit_namespace": namespace})
    policy = Mock()
    monkeypatch.setattr(module.ray, "get", Mock(return_value=[True]))
    sync = module.IPCWeightSynchronizer(policy, generation, refit_buffer_size_gb=1)
    sync.sync_weights()
    expected = {"buffer_size_bytes": 1024**3, "kv_scales": None}
    if namespace is not None:
        expected["generation_group"] = namespace
    policy.stream_weights_via_ipc_zmq.assert_called_once_with(**expected)


def test_refit_receiver_receives_its_namespace():
    worker = object.__new__(ReferenceVllmWorkerImpl)
    worker.cfg = {"refit_namespace": "vllm_val_ar"}
    worker.llm = Mock()
    metadata = {"layer.weight": ([2, 2], "float32")}
    worker.prepare_refit_info(metadata)
    worker.llm.collective_rpc.assert_called_once_with(
        "prepare_refit_info",
        args=(metadata,),
        kwargs={"generation_group": "vllm_val_ar"},
    )


def test_validation_groups_sleep_before_next_engine_initialization(monkeypatch):
    import nemo_rl.models.generation.vllm.vllm_generation as module

    runner = MultiModeValidation(validation_variants(variants()), colocated=True)
    config = OmegaConf.to_container(
        load_reference_vllm().policy.generation, resolve=True
    )
    groups = []

    def create(**kwargs):
        if groups:
            groups[-1].sleep.assert_called_once()
        group = Mock(cfg=kwargs["config"])
        groups.append(group)
        return group

    monkeypatch.setattr(module, "VllmGeneration", Mock(side_effect=create))
    rollout = runner.create_generation(cluster=Mock(), config=config)
    assert rollout is groups[2]
    assert list(runner.generations.values()) == groups[:2]
    runner.shutdown()
    for group in groups[:2]:
        group.shutdown.assert_called_once()
    rollout.shutdown.assert_not_called()


def test_diffusion_nfe_counts_one_copy_per_tensor_parallel_engine():
    reports = [
        {"nfe": 0, "committed_tokens": 0},
        {"nfe": 7, "committed_tokens": 16},
        {"nfe": 7, "committed_tokens": 16},
    ]
    assert _diffusion_counters(reports) == {
        "vllm:diffusion_num_denoising_steps": 7.0,
        "vllm:diffusion_num_committed_tokens": 16.0,
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("diffusion", [False, True])
def test_reference_worker_reads_actual_sampler_counters(asynchronous, diffusion):
    cls = ReferenceVllmAsyncWorkerImpl if asynchronous else ReferenceVllmWorkerImpl
    worker = object.__new__(cls)
    worker.cfg = {"vllm_kwargs": {"diffusion_config": {} if diffusion else None}}
    reports = [{"nfe": 5, "committed_tokens": 16}]
    if not asynchronous:
        # Synchronous engines retain host totals because sleep discards GPU counters.
        worker._diffusion_totals = _diffusion_counters(reports)
    rpc = (
        AsyncMock(return_value=reports) if asynchronous else Mock(return_value=reports)
    )
    worker.llm = SimpleNamespace(
        collective_rpc=rpc,
        get_metrics=Mock(
            return_value=[SimpleNamespace(name="vllm:spec_decode_num_drafts", value=2)]
        ),
    )
    result = (
        asyncio.run(worker._get_raw_spec_counters())
        if asynchronous
        else worker._get_raw_spec_counters()
    )
    assert result["vllm:spec_decode_num_drafts"] == 2
    if diffusion:
        assert result["vllm:diffusion_num_denoising_steps"] == 5
        assert result["vllm:diffusion_num_committed_tokens"] == 16
        if asynchronous:
            rpc.assert_called_once()
        else:
            rpc.assert_not_called()
    else:
        assert "vllm:diffusion_num_denoising_steps" not in result
        rpc.assert_not_called()
