# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-worker LLaDA generation, replay parity, and validation boundaries."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from just_grpo.algorithms.llada_schedule import LLaDAJustGRPOSchedule
from just_grpo.config import validate_config
from just_grpo.generation import llada_megatron
from just_grpo.generation.llada import generate_llada

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_rollout_logprobs_match_full_canvas_replay(temperature):
    torch.manual_seed(12)
    embedding = torch.nn.Embedding(16, 8)
    projection = torch.nn.Linear(8, 16)

    def forward(ids):
        # Bidirectional dependence makes a shortened canvas observably different.
        hidden = embedding(ids)
        return projection(hidden + hidden.mean(dim=1, keepdim=True)).bfloat16()

    generated = generate_llada(
        forward,
        torch.tensor([[1, 2], [3, 4]]),
        completion_length=4,
        block_length=1,
        mask_token_id=15,
        temperature=temperature,
        generator=torch.Generator().manual_seed(19),
    )
    tokens = generated.token_ids
    rollout_probs = generated.logprobs
    mask = torch.zeros_like(tokens)
    mask[:, 2:] = 1
    data = BatchedDataDict(
        input_ids=tokens,
        input_lengths=torch.full((2,), 6),
        token_mask=mask,
        sample_mask=torch.ones(2),
    )
    schedule = LLaDAJustGRPOSchedule(data, canvas_length=4)
    for step in range(4):
        level = schedule.generate_single_trajectory(step)
        inputs = level["target_ids"].masked_fill(level["masked_indices"], 15)
        logits = forward(inputs).float() / temperature
        replay = logits.log_softmax(-1).gather(-1, tokens[..., None]).squeeze(-1)
        selected = level["token_mask"].bool()
        torch.testing.assert_close(replay[selected], rollout_probs[selected])


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("dataset", ["gsm8k", "math"])
def test_shared_worker_recipes(monkeypatch, auxiliary, smoke, dataset):
    monkeypatch.setenv("LLADA_MATH_DATA_DIR", "/data")
    register_omegaconf_resolvers()
    stem = "2n8g-megatron-inference-smoke" if smoke else "16n8g-megatron-inference-long"
    suffix = "-diffaux" if auxiliary else ""
    recipe = (
        Path(__file__).resolve().parents[2]
        / "configs/recipes"
        / (f"just_grpo-{dataset}-llada8b-canvas256-" + stem + suffix + ".yaml")
    )
    config = load_config(recipe)
    diffusion = validate_config(config)
    assert diffusion.runtime == "megatron_llada"
    assert diffusion.generation_python is None
    assert not config.grpo.async_grpo.enabled
    assert config.policy.generation.colocated.enabled
    assert config.policy.generation.backend == "megatron"
    assert config.cluster.num_nodes == (2 if smoke else 16)
    assert config.policy.generation.max_new_tokens == 256
    assert config.policy.train_micro_batch_size == 8
    assert config.policy.logprob_batch_size == 8
    assert config.policy.train_global_batch_size == (128 if smoke else 1024)
    assert bool(config.policy.get("diffusion_aux_loss")) is auxiliary
    assert config.checkpointing.keep_top_k == 3
    assert config.checkpointing.ft_keep_latest_k == 1
    assert config.checkpointing.metric_name == "val:accuracy"
    assert diffusion.validation_sampling.block_size == 32
    config.grpo.async_grpo.enabled = True
    with pytest.raises(ValueError, match="synchronous"):
        validate_config(config)


def test_required_llada_recipes_are_present():
    project = Path(__file__).resolve().parents[2]
    expected = {
        f"just_grpo-{dataset}-llada8b-canvas256-{mode}{suffix}.yaml"
        for dataset in ("gsm8k", "math")
        for mode in ("16n8g-megatron-inference-long", "2n8g-megatron-inference-smoke")
        for suffix in ("", "-diffaux")
    }
    expected.update(
        f"just_grpo-acecode-llada8b-2n8g-megatron-humaneval-{mode}.yaml"
        for mode in ("long", "smoke")
    )
    # Additional experiment recipes and protocol specifications are allowed.
    assert expected <= {p.name for p in (project / "configs").rglob("*llada*.yaml")}
    assert not list(
        (project.parents[1] / "examples/configs/recipes").rglob("*llada*.yaml")
    )


def test_removed_hf_runtime_is_rejected():
    register_omegaconf_resolvers()
    recipe = (
        Path(__file__).resolve().parents[2]
        / "configs/recipes/just_grpo-gsm8k-llada8b-canvas256-16n8g-megatron-inference-long.yaml"
    )
    config = load_config(recipe)
    config.just_grpo.runtime = "hf_llada"
    with pytest.raises(ValueError, match="runtime"):
        validate_config(config)


def test_validation_restores_training_decoder_after_failure(monkeypatch):
    monkeypatch.setattr(llada_megatron.ray, "get", lambda x: x)
    adapter = object.__new__(llada_megatron.LLaDAMegatronGeneration)
    adapter.forward_counts = []
    group = MagicMock()
    adapter.policy = SimpleNamespace(worker_group=group)
    with pytest.raises(RuntimeError, match="validator"):
        with adapter.validation_sampling():
            raise RuntimeError("validator")
    calls = group.run_all_workers_single_data.call_args_list
    assert [call.kwargs["enabled"] for call in calls] == [True, False]


def test_validation_batch_smaller_than_dp_preserves_every_example(monkeypatch):
    monkeypatch.setattr(llada_megatron.ray, "get", lambda x: x)
    adapter = object.__new__(llada_megatron.LLaDAMegatronGeneration)
    adapter.forward_counts = []
    group = MagicMock()

    def dispatch(method, *, data, common_kwargs):
        assert method == "generate"
        return [
            BatchedDataDict(
                output_ids=shard["input_ids"],
                num_forward_passes=torch.zeros(
                    len(shard["input_lengths"]), dtype=torch.long
                ),
                logprobs=torch.zeros_like(shard["input_ids"], dtype=torch.float32),
                generation_lengths=shard["input_lengths"],
                unpadded_sequence_lengths=shard["input_lengths"],
            )
            for shard in data
        ]

    group.run_all_workers_multiple_data.side_effect = dispatch
    adapter.policy = SimpleNamespace(
        worker_group=group,
        data_parallel_size=8,
        cfg={"generation": {"_pad_token_id": 0}},
    )
    data = BatchedDataDict(
        input_ids=torch.tensor([[1, 2], [3, 4], [5, 6]]),
        input_lengths=torch.tensor([2, 2, 2]),
    )
    result = adapter.generate(data)
    torch.testing.assert_close(result["output_ids"], data["input_ids"])


def test_replay_excludes_environment_feedback_without_dropping_response_tokens():
    from just_grpo.algorithms.llada_schedule import prepare_llada_training_data

    prompts = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    answers = [torch.tensor([6, 7, 8, 9]), torch.tensor([10, 11, 12, 13])]
    feedback = [torch.tensor([14, 14, 14]), torch.tensor([15])]
    logs = [
        [
            {"role": "user", "token_ids": prompt},
            {"role": "assistant", "token_ids": answer},
            {"role": "environment", "token_ids": env},
        ]
        for prompt, answer, env in zip(prompts, answers, feedback)
    ]
    tokens = torch.tensor(
        [[1, 2, 6, 7, 8, 9, 14, 14, 14], [3, 4, 5, 10, 11, 12, 13, 15, 0]]
    )
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0]])
    data = BatchedDataDict(
        input_ids=tokens,
        input_lengths=torch.tensor([9, 8]),
        token_mask=mask,
        sample_mask=torch.ones(2),
    )
    prepare_llada_training_data(data, logs, 0, canvas_length=4)
    assert data["input_lengths"].tolist() == [6, 7]
    schedule = LLaDAJustGRPOSchedule(data, canvas_length=4)
    assert schedule.layout.base["prompt_lengths"].tolist() == [2, 3]
    assert schedule.layout.base["canvas_lengths"].tolist() == [6, 7]
    torch.testing.assert_close(schedule.layout.response_mask, mask.bool())
    for step in range(4):
        level = schedule.generate_single_trajectory(step)
        assert level["token_mask"].sum(dim=1).tolist() == [1, 1]


@pytest.mark.parametrize("threshold, expected", [(None, 4), (0.9, 2)])
def test_threshold_reveals_parallel_tokens_and_preserves_future_canvas(
    threshold, expected
):
    seen = []

    def forward(ids):
        seen.append(ids.clone())
        logits = torch.zeros((*ids.shape, 16))
        logits[..., 7] = 10
        return logits

    generated = generate_llada(
        forward,
        torch.tensor([[1, 2]]),
        completion_length=4,
        block_length=2,
        mask_token_id=15,
        temperature=0,
        threshold=threshold,
    )
    tokens = generated.token_ids
    counts = generated.num_forward_passes
    assert len(seen) == expected
    assert counts.tolist() == [expected]
    assert tokens.tolist() == [[1, 2, 7, 7, 7, 7]]
    assert all(ids.shape == (1, 6) for ids in seen)
    assert (seen[0][:, 2:] == 15).all()
    assert (seen[1][:, 4:] == 15).all()


def test_threshold_fallback_and_finished_rows():
    def forward(ids):
        logits = torch.zeros((*ids.shape, 16))
        logits[0, :, 7] = 10
        logits[1, :, 7] = 0.1
        return logits

    generated = generate_llada(
        forward,
        torch.tensor([[1], [2]]),
        completion_length=4,
        block_length=4,
        mask_token_id=15,
        temperature=0,
        threshold=0.9,
    )
    tokens = generated.token_ids
    counts = generated.num_forward_passes
    assert counts.tolist() == [1, 4]
    assert (tokens[:, 1:] == 7).all()


def test_offline_threshold_overrides(monkeypatch):
    register_omegaconf_resolvers()
    monkeypatch.setenv("EVAL_CHECKPOINT", "/snapshot")
    monkeypatch.setenv("OUTPUT_DIR", "/evaluation")
    recipe = (
        Path(__file__).resolve().parents[2]
        / "configs/recipes/just_grpo-gsm8k-llada8b-canvas256-16n8g-megatron-inference-long.yaml"
    )
    config = load_config(recipe)
    config.just_grpo.validation_modes = {}
    config.just_grpo.validation_sampling.selection_policy = "confidence_threshold"
    config.policy.pretrained_checkpoint = {
        "path": "/snapshot/policy/weights",
        "format": "megatron_bridge",
    }
    config.checkpointing.enabled = False
    config.cluster.num_nodes = 1
    diffusion = validate_config(config)
    assert diffusion.validation_sampling.selection_policy == "confidence_threshold"
    assert diffusion.validation_sampling.threshold == 0.9
    assert diffusion.sampling.selection_policy == "leftmost"
    assert config.grpo.stop_at_validation_metric is None
    assert config.policy.pretrained_checkpoint.path == "/snapshot/policy/weights"
    assert not config.checkpointing.enabled
    assert config.cluster.num_nodes == 1


def test_async_validation_collects_forward_counts(monkeypatch):
    import asyncio

    async def dispatch(self, data, greedy=False):
        yield (
            0,
            BatchedDataDict(
                output_ids=torch.tensor([[1, 2]]),
                num_forward_passes=torch.tensor([7]),
            ),
        )

    monkeypatch.setattr(
        llada_megatron.MegatronDiffusionGeneration, "generate_async", dispatch
    )
    adapter = object.__new__(llada_megatron.LLaDAMegatronGeneration)
    adapter.forward_counts = []

    async def collect():
        return [item async for item in adapter.generate_async(BatchedDataDict())]

    results = asyncio.run(collect())
    assert adapter.forward_counts == [7]
    assert len(results) == 1
    assert "num_forward_passes" not in results[0][1]


def test_validation_saves_async_decoding_statistics(monkeypatch, tmp_path):
    import json
    import sys

    from just_grpo.generation.llada_generation import validate_llada

    monkeypatch.setattr(llada_megatron.ray, "get", lambda value: value)
    adapter = object.__new__(llada_megatron.LLaDAMegatronGeneration)
    adapter.forward_counts = []
    adapter.policy = SimpleNamespace(worker_group=MagicMock())

    def validate(generation, *args, **kwargs):
        generation.forward_counts.extend([5, 7])
        return {"accuracy": 0.5}, {"total": 1.0}

    monkeypatch.setitem(
        sys.modules, "nemo_rl.algorithms.grpo", SimpleNamespace(validate=validate)
    )
    master = SimpleNamespace(logger={"log_dir": str(tmp_path)})
    metrics, _ = validate_llada(adapter, None, None, None, 0, master)
    assert metrics["mean_forward_passes"] == 6
    saved = json.loads((tmp_path / "decode_stats_step0.json").read_text())
    assert saved["forward_passes"] == [5, 7]


def test_fixed_two_reveals_ranked_positions_in_full_canvas():
    seen = []

    def forward(ids):
        seen.append(ids.clone())
        logits = torch.zeros((*ids.shape, 16))
        logits[..., 7] = torch.arange(ids.shape[1]).float() + 1
        return logits

    generated = generate_llada(
        forward,
        torch.tensor([[1, 2]]),
        completion_length=8,
        block_length=4,
        mask_token_id=15,
        temperature=0,
        tokens_per_step=2,
    )
    tokens = generated.token_ids
    counts = generated.num_forward_passes
    assert counts.tolist() == [4]
    assert len(seen) == 4
    assert seen[1].tolist() == [[1, 2, 15, 15, 7, 7, 15, 15, 15, 15]]
    assert seen[2].tolist() == [[1, 2, 7, 7, 7, 7, 15, 15, 15, 15]]
    assert tokens.tolist() == [[1, 2] + [7] * 8]


def test_offline_fixed_two_overrides(monkeypatch):
    register_omegaconf_resolvers()
    monkeypatch.setenv("EVAL_CHECKPOINT", "/snapshot")
    monkeypatch.setenv("OUTPUT_DIR", "/evaluation")
    recipe = (
        Path(__file__).resolve().parents[2]
        / "configs/recipes/just_grpo-gsm8k-llada8b-canvas256-16n8g-megatron-inference-long.yaml"
    )
    config = load_config(recipe)
    config.just_grpo.validation_modes = {}
    config.just_grpo.validation_sampling.max_steps = 16
    diffusion = validate_config(config)
    assert diffusion.validation_sampling.selection_policy == "low_confidence"
    assert diffusion.validation_sampling.block_size == 32
    assert diffusion.validation_sampling.max_steps == 16
    assert diffusion.sampling.block_size // diffusion.sampling.max_steps == 1
    config.just_grpo.validation_sampling.max_steps = 15
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("dataset", ["gsm8k", "math"])
def test_recipes_validate_four_decoders(monkeypatch, auxiliary, smoke, dataset):
    register_omegaconf_resolvers()
    monkeypatch.setenv("LLADA_MATH_DATA_DIR", "/data")
    stem = "2n8g-megatron-inference-smoke" if smoke else "16n8g-megatron-inference-long"
    suffix = "-diffaux" if auxiliary else ""
    recipe = (
        Path(__file__).resolve().parents[2]
        / "configs/recipes"
        / (f"just_grpo-{dataset}-llada8b-canvas256-" + stem + suffix + ".yaml")
    )
    config = load_config(recipe)
    diffusion = validate_config(config)
    assert list(diffusion.validation_modes) == [
        "fixed1_t1",
        "threshold09_t1",
        "threshold08_t1",
        "threshold07_t1",
    ]
    assert config.policy.generation.val_temperature == 1
    fixed, *thresholds = diffusion.validation_modes.values()
    assert fixed.selection_policy == "low_confidence"
    assert fixed.block_size == fixed.max_steps == 32
    for params in diffusion.validation_modes.values():
        assert params.temperature == diffusion.sampling.temperature == 1
        assert params.block_size == params.max_steps == 32
    assert [params.threshold for params in thresholds] == [0.9, 0.8, 0.7]
    assert all(
        params.selection_policy == "confidence_threshold" for params in thresholds
    )
    config.just_grpo.validation_modes.threshold09_t1.max_steps = 16
    with pytest.raises(ValueError, match="Fixed parallel"):
        validate_config(config)


@pytest.mark.parametrize("fail_second", [False, True])
@pytest.mark.parametrize("with_logger", [False, True])
def test_named_validation_replays_data_restores_decoder_and_plots(
    monkeypatch, tmp_path, fail_second, with_logger
):
    import json
    import sys

    from just_grpo.generation.llada_generation import validate_llada

    monkeypatch.setattr(llada_megatron.ray, "get", lambda value: value)
    adapter = object.__new__(llada_megatron.LLaDAMegatronGeneration)
    adapter.forward_counts = []
    group = MagicMock()
    adapter.policy = SimpleNamespace(worker_group=group)
    loader = MagicMock()
    loader.state_dict.return_value = {"position": 0}
    logger = MagicMock() if with_logger else None
    modes = ["fixed1_t1", "threshold09_t1", "threshold08_t1", "threshold07_t1"]
    counts = [[256, 256], [40, 60], [30, 50], [20, 40]]
    accuracies = [0.4, 0.5, 0.45, 0.35]
    steps = []

    def validate(generation, dataloader, *args, **kwargs):
        index = len(steps)
        steps.append(kwargs["step"])
        assert dataloader.load_state_dict.call_args.args == ({"position": 0},)
        if fail_second and index == 1:
            raise RuntimeError("second pass failed")
        generation.forward_counts.extend(counts[index])
        if kwargs["logger"] is not None:
            kwargs["logger"].log_batched_dict_as_jsonl(
                {}, f"val_data_step{kwargs['step']}.jsonl"
            )
        return {"accuracy": accuracies[index]}, {"total": 2.0}

    monkeypatch.setitem(
        sys.modules, "nemo_rl.algorithms.grpo", SimpleNamespace(validate=validate)
    )
    master = SimpleNamespace(
        just_grpo={
            "validation_modes": {
                name: {
                    "selection_policy": "low_confidence"
                    if i == 0
                    else "confidence_threshold",
                    "threshold": 1.0 - i / 10,
                }
                for i, name in enumerate(modes)
            }
        },
        policy={"generation": {"max_new_tokens": 256}},
        logger={"log_dir": str(tmp_path)},
    )
    plot = tmp_path / "accuracy_vs_nfe_step10.png"
    if fail_second:
        with pytest.raises(RuntimeError, match="second pass"):
            validate_llada(adapter, loader, None, None, 10, master, logger)
        assert not plot.exists()
        if logger is not None:
            logger.log_plot.assert_not_called()
            logger.log_table.assert_not_called()
    else:
        metrics, timings = validate_llada(
            adapter, loader, None, None, 10, master, logger
        )
        assert metrics["accuracy"] == metrics["accuracy/fixed1_t1"] == 0.4
        assert timings["total"] == 8
        expected_rows = []
        for index, name in enumerate(modes):
            nfe = sum(counts[index]) / len(counts[index])
            assert metrics[f"accuracy/{name}"] == accuracies[index]
            assert metrics[f"mean_forward_passes/{name}"] == nfe
            assert metrics[f"tokens_per_forward/{name}"] == 256 / nfe
            expected_rows.append(
                [
                    10,
                    name,
                    None if index == 0 else 1.0 - index / 10,
                    nfe,
                    accuracies[index],
                    2,
                ]
            )
            suffix = f"_{name}" if index else ""
            saved = json.loads(
                (tmp_path / f"decode_stats_step10{suffix}.json").read_text()
            )
            assert saved["forward_passes"] == counts[index]
        assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        if logger is not None:
            logger.log_table.assert_called_once_with(
                [
                    "training_step",
                    "variant",
                    "confidence_threshold",
                    "mean_denoising_nfe",
                    "accuracy",
                    "num_samples",
                ],
                expected_rows,
                10,
                "validation/accuracy_vs_nfe_data",
            )
            figure, step, name = logger.log_plot.call_args.args
            assert (step, name) == (10, "validation/accuracy_vs_nfe")
            line = figure.axes[0].lines[0]
            ordered = sorted(expected_rows, key=lambda row: row[3])
            assert list(line.get_xdata()) == [row[3] for row in ordered]
            assert list(line.get_ydata()) == [100 * row[4] for row in ordered]
            assert figure.axes[0].get_ylabel() == "Accuracy (%)"
            assert [
                c.args[1] for c in logger.log_batched_dict_as_jsonl.call_args_list
            ] == [
                "val_data_step10.jsonl",
                *[f"val_data_step10_{mode}.jsonl" for mode in modes[1:]],
            ]
    evaluated = 2 if fail_second else 4
    assert steps == [10] * evaluated
    assert loader.load_state_dict.call_count == evaluated + 1
    calls = group.run_all_workers_single_data.call_args_list
    assert [c.kwargs["enabled"] for c in calls] == [True, False] * evaluated
    assert [c.kwargs["mode"] for c in calls[::2]] == modes[:evaluated]


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("threshold", [0.7, 0.9])
@pytest.mark.parametrize("named_mode", [False, True])
def test_llada_fork_validation_modes(
    temperature: float, threshold: float, named_mode: bool
) -> None:
    register_omegaconf_resolvers()
    recipe = (
        Path(__file__).resolve().parents[2]
        / "configs/recipes/just_grpo-gsm8k-llada8b-canvas256-16n8g-megatron-inference-long.yaml"
    )
    config = load_config(recipe)
    params = dict(
        config.just_grpo.validation_sampling,
        selection_policy="fork",
        temperature=temperature,
        threshold=threshold,
    )
    if named_mode:
        config.just_grpo.validation_modes = {"fork": params}
        selected = config.just_grpo.validation_modes.fork
    else:
        config.just_grpo.validation_sampling = params
        selected = config.just_grpo.validation_sampling
    validate_config(config)
    selected.max_steps = 16
    with pytest.raises(ValueError, match="Fixed parallel"):
        validate_config(config)


@pytest.mark.parametrize("auxiliary", [False, True])
def test_vllm_gsm8k_preserves_megatron_validation(auxiliary):
    register_omegaconf_resolvers()
    suffix = "-diffaux" if auxiliary else ""
    recipe = (
        Path(__file__).resolve().parents[2]
        / "configs/recipes"
        / ("just_grpo-gsm8k-llada8b-canvas256-16n8g-vllm-long" + suffix + ".yaml")
    )
    config = load_config(recipe)
    diffusion = validate_config(config)
    assert diffusion.runtime == "reference_vllm"
    assert diffusion.validation_runtime == "megatron_llada"
    assert diffusion.max_generation_kl == float("inf")
    assert config.logger.wandb.training_step_history is True
    assert len(diffusion.validation_modes) == 4
    assert config.policy.train_global_batch_size == 1024
    assert bool(config.policy.get("diffusion_aux_loss")) is auxiliary
    config.just_grpo.validation_runtime = "generation"
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("fail", [False, True])
def test_policy_validation_restores_rollout_lifecycle(monkeypatch, fail):
    from just_grpo.generation import llada_generation

    events = []
    policy = MagicMock()
    policy.prepare_for_lp_inference.side_effect = lambda: events.append("policy_on")
    policy.offload_after_refit.side_effect = lambda: events.append("policy_off")
    generation = MagicMock()
    generation.finish_generation.side_effect = lambda: events.append("sleep") or True
    generation.prepare_for_generation.side_effect = (
        lambda: events.append("wake") or True
    )
    adapter = object()
    monkeypatch.setattr(llada_megatron, "LLaDAMegatronGeneration", lambda **kw: adapter)

    def validate(actual_adapter, *args, **kwargs):
        assert actual_adapter is adapter
        events.append("validate")
        if fail:
            raise ValueError("validation failed")
        return {"accuracy": 0.5}, {}

    monkeypatch.setattr(llada_generation, "validate_llada", validate)
    validator = llada_generation.make_llada_policy_validator(policy=policy, config=None)
    if fail:
        with pytest.raises(ValueError, match="validation failed"):
            validator(generation)
    else:
        assert validator(generation) == ({"accuracy": 0.5}, {})
    assert events == ["sleep", "policy_on", "validate", "policy_off", "wake"]


def test_vllm_gsm8k_smoke_config():
    register_omegaconf_resolvers()
    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/recipes"
        / "just_grpo-gsm8k-llada8b-canvas256-2n8g-vllm-smoke.yaml"
    )
    validate_config(config)
    assert config.policy.train_global_batch_size == 32
    assert config.policy.train_micro_batch_size == 2
    assert config.grpo.val_at_start and config.grpo.val_at_end
