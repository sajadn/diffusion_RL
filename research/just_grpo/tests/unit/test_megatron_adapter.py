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
"""The upstream-worker adapter preserves trajectory coverage and gradients."""

from unittest.mock import Mock

import pytest
import torch
from just_grpo.algorithms.block_just_grpo import (
    BlockJustGRPOSchedule,
    select_low_confidence_tokens,
)
from test_fast_selection import confidence_batch, make_schedule

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

pytest.importorskip("megatron.bridge")
from block_diffusion.diffusion_processors import (
    DiffusionLogprobsPostProcessor,
    DiffusionLossPostProcessor,
    DiffusionMicrobatchProcessor,
    prepare_diffusion_microbatch,
    selected_diffusion_logprobs,
)
from block_diffusion.megatron_diffusion_policy import (
    MegatronDiffusionPolicyWorkerImpl,
)
from just_grpo.algorithms.block_just_grpo_policy_worker import (
    BlockJustGRPOPolicyWorkerImpl,
)
from megatron.core import parallel_state

from nemo_rl.models.megatron.data import ProcessedMicrobatch


@pytest.fixture(autouse=True)
def tensor_parallel_stubs(monkeypatch):
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: None)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", lambda: None
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_group", lambda: None
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_context_parallel_group", lambda: None
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_context_parallel_world_size", lambda: 1
    )


def processed(trajectory):
    return ProcessedMicrobatch(
        data_dict=trajectory,
        input_ids=trajectory["input_ids"],
        input_ids_cp_sharded=trajectory["input_ids"],
        position_ids=trajectory["position_ids"],
        attention_mask=None,
        packed_seq_params=None,
        cu_seqlens_padded=None,
        original_seq_length=trajectory["input_ids"].shape[1],
    )


def make_diffusion_schedule(data, fraction):
    schedule = make_schedule(data)
    return BlockJustGRPOSchedule(
        data,
        schedule.config,
        pad_token_id=0,
        selected_positions_per_block=1 if fraction < 1 else None,
    )


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_schedule_yields_one_level_at_a_time_with_aligned_rollout_fields(
    monkeypatch, fraction
):
    data = confidence_batch()
    data["training_token_mask"] = select_low_confidence_tokens(
        data, fraction=fraction, block_size=4
    )
    data["prev_logprobs"] = torch.randn_like(data["token_mask"].float())
    data["reference_policy_logprobs"] = data["prev_logprobs"] - 0.3
    schedule = make_diffusion_schedule(data, fraction)
    generated = []
    generate = schedule.generate_single_trajectory

    def track(step):
        generated.append(step)
        return generate(step)

    monkeypatch.setattr(schedule, "generate_single_trajectory", track)

    iterator = schedule.iter_levels(data)
    assert generated == []
    coverage = torch.zeros_like(data["token_mask"])
    for index in range(schedule.num_steps):
        micro = next(iterator)
        assert micro.size == data.size  # Never an expanded batch.
        assert generated == list(range(index + 1))
        for row, sample in enumerate(micro["sample_indices"]):
            active = micro["token_mask"][row]
            positions = micro["original_positions"][row][active]
            coverage[sample, positions] += 1
            for key in (
                "prev_logprobs",
                "reference_policy_logprobs",
                "generation_logprobs",
            ):
                torch.testing.assert_close(
                    micro[key][row][active], data[key][sample, positions]
                )
    with pytest.raises(StopIteration):
        next(iterator)
    torch.testing.assert_close(coverage.bool(), data["training_token_mask"])
    assert coverage.max() == 1


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_selected_same_position_scores_and_gradients_match_dense(fraction):
    data = confidence_batch()
    data["training_token_mask"] = select_low_confidence_tokens(
        data, fraction=fraction, block_size=4
    )
    trajectory = next(make_diffusion_schedule(data, fraction).return_schedule(2))
    torch.manual_seed(17)
    clean = trajectory["input_ids"]
    logits = torch.randn(*clean.shape, 32, requires_grad=True)

    micro = prepare_diffusion_microbatch(processed(trajectory), mask_token_id=31)
    torch.testing.assert_close(
        micro.input_ids[:, : clean.shape[1]],
        clean.masked_fill(trajectory["masked_indices"], 31),
    )
    torch.testing.assert_close(
        micro.input_ids[:, clean.shape[1] :], trajectory["clean_input_ids"]
    )
    torch.testing.assert_close(
        micro.position_ids[:, : clean.shape[1]],
        trajectory["position_ids"] + trajectory["prompt_lengths"][:, None],
    )
    assert micro.data_dict is trajectory
    active = trajectory["token_mask"]
    actual = selected_diffusion_logprobs(
        torch.cat([logits, logits], dim=1) / 0.7, trajectory
    )
    expected = (logits / 0.7).log_softmax(-1).gather(-1, clean[..., None]).squeeze(-1)
    torch.testing.assert_close(actual[active], expected[active])
    actual.sum().backward()
    actual_grad = logits.grad.clone()
    logits.grad = None
    expected[active].sum().backward()
    torch.testing.assert_close(actual_grad, logits.grad)
    assert not actual_grad[~active].any()


def test_empty_reveal_retains_backward_graph():
    data = confidence_batch()
    trajectory = next(make_schedule(data).return_schedule(2))
    trajectory["token_mask"].zero_()
    logits = torch.randn(2, 24, 32, requires_grad=True)
    values = selected_diffusion_logprobs(logits, trajectory)
    values.sum().backward()
    assert logits.grad is not None
    assert not logits.grad.any()


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_standard_forward_and_diffusion_processors_match_core_loss(
    monkeypatch, fraction
):
    pytest.importorskip("megatron.bridge")
    from megatron.core import parallel_state

    from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
    from nemo_rl.models.megatron.train import forward_with_post_processing_fn

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: None)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: None)
    # LossPostProcessor imports the same functions directly.
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", lambda: None
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_group", lambda: None
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_context_parallel_group", lambda: None
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_context_parallel_world_size", lambda: 1
    )
    data = confidence_batch()
    data["training_token_mask"] = select_low_confidence_tokens(
        data, fraction=fraction, block_size=4
    )
    data["generation_logprobs"] = torch.zeros_like(data["token_mask"].float()) - 3
    data["prev_logprobs"] = data["generation_logprobs"].clone()
    data["reference_policy_logprobs"] = data["prev_logprobs"] - 0.1
    data["advantages"] = torch.ones_like(data["prev_logprobs"])
    schedule = make_diffusion_schedule(data, fraction)
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.01))
    cfg = {"sequence_packing": {"enabled": False}}
    params = TrainingSamplingParams(temperature=0.7, top_k=None, top_p=1.0)
    processor = DiffusionLossPostProcessor(
        loss_fn,
        cfg,
        num_microbatches=schedule.num_steps,
    )
    scoring = DiffusionLogprobsPostProcessor(cfg, sampling_params=params)
    torch.manual_seed(13)
    logits = torch.randn(2, 24, 32, requires_grad=True)
    actual_losses, expected_losses = [], []
    n_seqs = data["sample_mask"].sum()
    n_toks = data["training_token_mask"].sum()
    for index, trajectory in enumerate(schedule.return_schedule(2)):
        micro = schedule.prepare_trajectory(index, data)
        prepared = prepare_diffusion_microbatch(processed(micro), mask_token_id=31)
        output, callback = forward_with_post_processing_fn(
            iter([prepared]),
            lambda **kwargs: logits.clone(),
            processor,
            global_valid_seqs=n_seqs,
            global_valid_toks=n_toks,
            sampling_params=params,
        )
        score_output, score_callback = forward_with_post_processing_fn(
            iter([prepared]),
            lambda **kwargs: logits.clone(),
            scoring,
            sampling_params=params,
        )
        values = score_callback(score_output)[1]["logprobs"]
        torch.testing.assert_close(values, selected_diffusion_logprobs(output, micro))
        # Undo the scaling that MCore's schedule applies, exactly as a real update.
        actual_losses.append(callback(output)[0] / schedule.num_steps)
        expected_losses.append(
            loss_fn(
                next_token_logprobs=values,
                data=micro,
                shift_labels=False,
                global_valid_seqs=n_seqs,
                global_valid_toks=n_toks,
            )[0]
        )
    actual, expected = sum(actual_losses), sum(expected_losses)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def _keep_worker_data_on_cpu(monkeypatch):
    move = BatchedDataDict.to
    monkeypatch.setattr(
        BatchedDataDict,
        "to",
        lambda self, device: self if device == "cuda" else move(self, device),
    )


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_worker_accumulates_levels_before_one_optimizer_step(monkeypatch, fraction):
    _keep_worker_data_on_cpu(monkeypatch)
    data = confidence_batch()
    data["training_token_mask"] = select_low_confidence_tokens(
        data, fraction=fraction, block_size=4
    )
    schedule = make_diffusion_schedule(data, fraction)
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    worker = Mock()
    worker.positive_diffusion = None
    worker._build_schedule.return_value = schedule
    worker.max_generation_kl = 0.05
    expected_tokens = float(
        (data.pop("training_token_mask") * data["sample_mask"][:, None]).sum()
    )
    counts = {"tokens": 0.0, "sequences": 0.0}

    def accumulate(level):
        # Mirror native streaming counts, including its AR first-column slice.
        counts["tokens"] += float(
            (
                MegatronDiffusionPolicyWorkerImpl._streaming_loss_token_mask(
                    worker, level
                )
                * level["sample_mask"][:, None]
            ).sum()
        )
        counts["sequences"] += float(level["sample_mask"].sum())

    def finish():
        return {
            "all_mb_metrics": {
                "gen_kl_error": [0.001],
                "global_valid_toks": [counts["tokens"]],
                "global_valid_seqs": [counts["sequences"]],
            }
        }

    worker.train_microbatch.side_effect = accumulate
    worker.finish_train_step.side_effect = finish
    loss_fn = Mock()
    result = MegatronDiffusionPolicyWorkerImpl.train(
        worker, data=data, loss_fn=loss_fn, gbs=16, mbs=2
    )
    worker.begin_train_step.assert_called_once_with(loss_fn, gbs=16, mbs=2)
    assert worker.train_microbatch.call_count == schedule.num_steps
    worker.finish_train_step.assert_called_once_with()
    worker.abort_train_step.assert_not_called()
    assert counts["tokens"] == expected_tokens
    assert result["all_mb_metrics"]["global_valid_toks"] == [expected_tokens]
    assert result["all_mb_metrics"]["global_valid_seqs"] == [
        float(data["sample_mask"].sum())
    ]
    assert result["all_mb_metrics"]["num_valid_samples"] == [data.size]


@pytest.mark.parametrize("failure", ["train_microbatch", "finish_train_step"])
def test_training_failure_aborts_open_step(monkeypatch, failure):
    _keep_worker_data_on_cpu(monkeypatch)
    data = confidence_batch()
    data["training_token_mask"] = data["token_mask"].bool()
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda: None)
    worker = Mock()
    worker.positive_diffusion = None
    worker._build_schedule.return_value = make_schedule(data)
    getattr(worker, failure).side_effect = RuntimeError("training failed")
    with pytest.raises(RuntimeError, match="training failed"):
        MegatronDiffusionPolicyWorkerImpl.train(worker, data=data, loss_fn=Mock())
    worker.abort_train_step.assert_called_once_with()
    if failure == "train_microbatch":
        worker.finish_train_step.assert_not_called()


@pytest.mark.parametrize("kl", [0.08, float("nan")])
def test_shared_worker_checks_generation_kl_over_schedule_targets(monkeypatch, kl):
    _keep_worker_data_on_cpu(monkeypatch)
    data = confidence_batch()
    data["training_token_mask"] = select_low_confidence_tokens(
        data, fraction=0.25, block_size=4
    )
    schedule = make_diffusion_schedule(data, 0.25)
    selected_tokens = float(data.pop("training_token_mask").sum())
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    worker = Mock()
    worker.positive_diffusion = None
    worker._build_schedule.return_value = schedule
    worker.max_generation_kl = 0.05
    worker.finish_train_step.return_value = {
        "all_mb_metrics": {
            "gen_kl_error": [kl],
            "global_valid_toks": [selected_tokens],
            "global_valid_seqs": [float(data.size * schedule.num_steps)],
        }
    }
    with pytest.raises(RuntimeError, match="Generation/training KL exceeds limit"):
        MegatronDiffusionPolicyWorkerImpl.train(worker, data=data, loss_fn=Mock())


@pytest.mark.parametrize("fraction", [1.0, 0.25])
@pytest.mark.parametrize("mode", ["policy-replay", "policy-no-replay", "reference"])
def test_scoring_coverage_depends_on_policy_role_not_router_replay(
    monkeypatch, fraction, mode
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    data = confidence_batch()
    worker = object.__new__(BlockJustGRPOPolicyWorkerImpl)
    worker.model = torch.nn.Module()
    worker.prepare_microbatch_fn = DiffusionMicrobatchProcessor(
        model=worker.model, mask_token_id=31
    )
    worker.just_grpo = SimpleNamespace(
        training_token_fraction=fraction, schedule=make_schedule(data).config
    )
    worker.cfg = {"max_total_sequence_length": data["input_ids"].shape[1]}
    worker.tokenizer = SimpleNamespace(pad_token_id=0)
    _keep_worker_data_on_cpu(monkeypatch)
    reference = mode == "reference"
    replay = mode == "policy-replay"
    expected_mask = select_low_confidence_tokens(
        data, fraction=fraction if reference else 1.0, block_size=4
    )
    active = []
    calls = []

    @contextmanager
    def reference_weights():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    worker.use_reference_model = reference_weights

    def score(self, *, data, micro_batch_size, require_router_replay):
        assert bool(active) == reference
        assert require_router_replay is replay
        assert micro_batch_size == 1
        calls.append(data)
        return {"logprobs": data["target_ids"].float()}

    monkeypatch.setattr(MegatronPolicyWorkerImpl, "get_logprobs", score)
    result = (
        worker.get_reference_policy_logprobs(data=data, micro_batch_size=1)
        if reference
        else worker.get_logprobs(
            data=data, micro_batch_size=1, require_router_replay=replay
        )
    )
    torch.testing.assert_close(result["logprob_token_mask"], expected_mask)
    result = result["reference_logprobs" if reference else "logprobs"]
    assert len(calls) == (1 if reference and fraction < 1 else 4)
    assert not active
    torch.testing.assert_close(result, data["input_ids"].float() * expected_mask)


def test_worker_preserves_fixed_model_canvas_for_short_rollouts():
    from types import SimpleNamespace

    data = confidence_batch()
    worker = object.__new__(BlockJustGRPOPolicyWorkerImpl)
    worker.model = torch.nn.Module()
    worker.prepare_microbatch_fn = DiffusionMicrobatchProcessor(
        model=worker.model, mask_token_id=31
    )
    worker.just_grpo = SimpleNamespace(
        training_token_fraction=0.25, schedule=make_schedule(data).config
    )
    worker.cfg = {"max_total_sequence_length": 32}
    worker.tokenizer = SimpleNamespace(pad_token_id=0)
    schedule = worker._build_schedule(data, purpose="train")
    assert schedule.base["input_ids"].shape == (data.size, 32)
    assert schedule.num_steps == 1
    assert not schedule.base["token_mask"][:, data["input_ids"].shape[1] :].any()


def test_diffusion_streaming_normalization_includes_first_response_token():
    data = BatchedDataDict(
        token_mask=torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        sample_mask=torch.tensor([1.0, 0.0]),
    )
    worker = object.__new__(MegatronDiffusionPolicyWorkerImpl)
    mask = worker._streaming_loss_token_mask(data)
    assert (mask * data["sample_mask"][:, None]).sum() == 2


def test_microbatch_keeps_original_context_and_response_logical_positions():
    data = BatchedDataDict(
        input_ids=torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]]),
        token_mask=torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]]),
        input_lengths=torch.tensor([8]),
        sample_mask=torch.ones(1),
    )
    trajectory = next(make_schedule(data).iter_levels(data))
    microbatch = prepare_diffusion_microbatch(processed(trajectory), mask_token_id=31)
    width = trajectory["input_ids"].shape[1]
    torch.testing.assert_close(microbatch.input_ids[:, width:], data["input_ids"])
    assert microbatch.position_ids[0, :5].tolist() == [3, 4, 5, 6, 7]
    assert trajectory["token_mask"][0, 0]


def test_metadata_traverses_a_single_wrapped_model_and_is_shared_between_layers():
    from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import (
        NemotronLabsDiffusionAttention,
    )

    layers = []
    for _ in range(2):
        layer = object.__new__(NemotronLabsDiffusionAttention)
        torch.nn.Module.__init__(layer)
        layer.block_size = 4
        layer._asymmetric_ar_metadata = None
        layers.append(layer)
    worker = object.__new__(MegatronDiffusionPolicyWorkerImpl)
    worker.model = torch.nn.Module()
    worker.prepare_microbatch_fn = DiffusionMicrobatchProcessor(
        model=worker.model, mask_token_id=31
    )
    worker.model.add_module("module", torch.nn.Sequential(*layers))
    worker.mask_token_id = 31
    trajectory = next(make_schedule(confidence_batch()).iter_levels())
    prepared = worker.prepare_microbatch_fn(processed(trajectory))
    metadata = layers[0]._asymmetric_ar_metadata
    assert metadata is layers[1]._asymmetric_ar_metadata
    torch.testing.assert_close(metadata.prompt_lengths, torch.tensor([4, 8]))
    assert prepared.input_ids.shape[1] == metadata.noisy_length + metadata.clean_length
    worker.prepare_microbatch_fn.clear_asymmetric_metadata()
    assert all(layer._asymmetric_ar_metadata is None for layer in layers)


def test_positive_diffusion_level_runs_once_after_the_schedule(monkeypatch):
    from types import SimpleNamespace

    _keep_worker_data_on_cpu(monkeypatch)
    data = confidence_batch()
    data["training_token_mask"] = data["token_mask"].bool()
    schedule = make_diffusion_schedule(data, 1.0)
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    worker = Mock()
    worker.positive_diffusion = SimpleNamespace(type="diffusion")
    worker.max_generation_kl = 0.05
    worker._build_schedule.return_value = schedule
    calls = []
    worker.train_microbatch.side_effect = lambda level: calls.append("level")
    worker._train_positive_diffusion_level.side_effect = lambda d: calls.append("aux")
    worker.finish_train_step.return_value = {
        "all_mb_metrics": {
            "gen_kl_error": [0.001],
            "global_valid_toks": [1.0],
            "global_valid_seqs": [float(data.size * (schedule.num_steps + 1))],
        }
    }
    result = MegatronDiffusionPolicyWorkerImpl.train(worker, data=data, loss_fn=Mock())
    assert calls == ["level"] * schedule.num_steps + ["aux"]
    assert result["all_mb_metrics"]["global_valid_seqs"] == [float(data.size)]


def test_positive_diffusion_level_uses_untempered_ce_and_response_denominator(
    monkeypatch,
):
    from types import SimpleNamespace

    from block_diffusion.positive_diffusion import (
        PositiveDiffusionLoss,
        resolve_positive_diffusion_config,
    )

    data = confidence_batch()
    data["advantages"] = data["token_mask"].float()
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    worker = object.__new__(MegatronDiffusionPolicyWorkerImpl)
    worker.positive_diffusion = resolve_positive_diffusion_config(
        {"mask_probability_min": 1.0}, mask_token_id=31, block_size=4
    )
    worker.positive_diffusion_generator = torch.Generator().manual_seed(0)
    worker.tokenizer = SimpleNamespace(pad_token_id=0)
    worker.cfg = {
        "max_total_sequence_length": 16,
        "generation": {"stop_token_ids": None},
    }
    worker.sampling_params = tempered = object()
    worker.loss_postprocessor_factory = factory = object()
    state = {
        "metric_normalizations": {},
        "local_valid_toks": torch.tensor(5.0, dtype=torch.float64),
    }
    worker._assert_step_open = lambda: state
    seen = {}

    def train_microbatch(level):
        seen["level"] = level
        seen["sampling_params"] = worker.sampling_params
        seen["processor"] = worker.loss_postprocessor_factory(
            loss_fn=Mock(), cfg={}, num_microbatches=1
        )

    worker.train_microbatch = train_microbatch
    worker._train_positive_diffusion_level(data)
    level = seen["level"]
    responses = float(data["token_mask"].sum())
    assert seen["sampling_params"] is None
    assert isinstance(seen["processor"].loss_fn, PositiveDiffusionLoss)
    # The policy-count finish scaling is converted to the response count.
    assert seen["processor"].loss_fn.weight == pytest.approx(0.1 * 5.0 / responses)
    assert worker.sampling_params is tempered
    assert worker.loss_postprocessor_factory is factory
    assert state["metric_normalizations"] == PositiveDiffusionLoss.metric_normalizations
    # Every included response token is a target; none dilute the policy count.
    assert level["token_mask"].sum() == responses
    assert not worker._streaming_loss_token_mask(level).any()


def test_causal_auxiliary_rides_on_the_last_level_without_an_extra_pass(monkeypatch):
    from types import SimpleNamespace

    _keep_worker_data_on_cpu(monkeypatch)
    data = confidence_batch()
    data["training_token_mask"] = data["token_mask"].bool()
    schedule = make_diffusion_schedule(data, 1.0)
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    worker = Mock()
    worker.positive_diffusion = SimpleNamespace(type="ar")
    worker.max_generation_kl = 0.05
    worker._build_schedule.return_value = schedule
    calls = []
    worker.train_microbatch.side_effect = lambda level: calls.append("level")
    worker._train_with_positive_ar.side_effect = lambda level, d: calls.append("ar")
    worker.finish_train_step.return_value = {
        "all_mb_metrics": {
            "gen_kl_error": [0.001],
            "global_valid_toks": [1.0],
            "global_valid_seqs": [float(data.size * schedule.num_steps)],
        }
    }
    result = MegatronDiffusionPolicyWorkerImpl.train(worker, data=data, loss_fn=Mock())
    assert calls == ["level"] * (schedule.num_steps - 1) + ["ar"]
    worker._train_positive_diffusion_level.assert_not_called()
    assert result["all_mb_metrics"]["global_valid_seqs"] == [float(data.size)]


def test_causal_auxiliary_level_rescales_to_the_response_denominator(monkeypatch):
    from types import SimpleNamespace

    from block_diffusion.diffusion_processors import (
        PolicyWithCausalAuxiliaryPostProcessor,
    )
    from block_diffusion.positive_diffusion import (
        PositiveARLoss,
        resolve_positive_diffusion_config,
    )
    from nemo_rl.models.megatron.train import LossPostProcessor

    data = confidence_batch()
    data["advantages"] = data["token_mask"].float()
    data["training_token_mask"] = data["token_mask"].bool()
    level = list(make_diffusion_schedule(data, 1.0).iter_levels(data))[-1]
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    worker = object.__new__(MegatronDiffusionPolicyWorkerImpl)
    worker.positive_diffusion = resolve_positive_diffusion_config(
        {"type": "ar"}, mask_token_id=31, block_size=4
    )
    worker.cfg = {"generation": {"stop_token_ids": None}}
    worker.sampling_params = tempered = SimpleNamespace(temperature=0.5)
    worker.loss_postprocessor_factory = factory = DiffusionLossPostProcessor
    state = {
        "metric_normalizations": {},
        "local_valid_toks": torch.tensor(5.0, dtype=torch.float64),
    }
    worker._assert_step_open = lambda: state
    seen = {}

    def train_microbatch(trained):
        seen["level"] = trained
        seen["sampling_params"] = worker.sampling_params
        seen["processor"] = worker.loss_postprocessor_factory(
            loss_fn=Mock(), cfg={}, num_microbatches=1
        )

    worker.train_microbatch = train_microbatch
    worker._train_with_positive_ar(level, data)
    processor = seen["processor"]
    assert isinstance(processor, PolicyWithCausalAuxiliaryPostProcessor)
    # Megatron's forward dispatches loss post-processors by this base class.
    assert isinstance(processor, LossPostProcessor)
    assert isinstance(processor.policy, DiffusionLossPostProcessor)
    assert processor.policy_sampling_params is tempered
    assert seen["sampling_params"] is None
    responses = float(data["token_mask"].sum())
    policy_tokens = 5.0 + float(level["token_mask"].sum())
    assert isinstance(processor.loss_fn, PositiveARLoss)
    assert processor.loss_fn.weight == pytest.approx(
        0.1 * policy_tokens / responses
    )
    targets = seen["level"]["causal_aux_target_mask"]
    assert targets.shape == level["clean_input_ids"].shape
    assert torch.equal(targets[:, : data["input_ids"].shape[1]], data["token_mask"].bool())
    assert worker.sampling_params is tempered
    assert worker.loss_postprocessor_factory is factory
    assert state["metric_normalizations"] == PositiveARLoss.metric_normalizations


@pytest.mark.parametrize("temperature", [1.0, 0.5])
def test_causal_auxiliary_processor_matches_manual_losses(temperature):
    from types import SimpleNamespace

    from block_diffusion.diffusion_processors import (
        PolicyWithCausalAuxiliaryPostProcessor,
    )
    from block_diffusion.positive_diffusion import (
        PositiveARLoss,
        resolve_positive_diffusion_config,
    )

    torch.manual_seed(0)
    batch, noisy_width, clean_width, vocab = 2, 4, 6, 11
    clean_ids = torch.randint(0, vocab, (batch, clean_width))
    targets = torch.zeros(batch, clean_width, dtype=torch.bool)
    targets[0, 3:5] = True
    data = BatchedDataDict(
        target_ids=torch.zeros(batch, noisy_width, dtype=torch.long),
        clean_input_ids=clean_ids,
        causal_aux_target_mask=targets,
        token_mask=torch.ones(batch, noisy_width),
        sample_mask=torch.ones(batch),
    )
    seen = {}

    class Policy:
        def __call__(self, data_dict, **kwargs):
            def process(logits):
                seen["policy_logits"] = logits
                return logits.sum() * 0, {"loss": 0.0}

            return process

    sampling = SimpleNamespace(temperature=temperature, top_k=None, top_p=1.0)
    processor = PolicyWithCausalAuxiliaryPostProcessor(
        policy_factory=lambda **kwargs: Policy(),
        auxiliary_loss_fn=PositiveARLoss(
            resolve_positive_diffusion_config(
                {"type": "ar", "weight": 0.3}, mask_token_id=31, block_size=2
            )
        ),
        policy_sampling_params=sampling,
        cfg={"sequence_packing": {"enabled": False}},
        num_microbatches=1,
    )
    logits = torch.randn(batch, noisy_width + clean_width, vocab, requires_grad=True)
    loss, metrics = processor(
        data, global_valid_seqs=torch.tensor(1.0), global_valid_toks=torch.tensor(1.0)
    )(logits)
    clean = logits[:, noisy_width:].log_softmax(-1)
    expected = -0.3 * (
        clean[:, :-1].gather(-1, clean_ids[:, 1:, None]).squeeze(-1) * targets[:, 1:]
    ).sum()
    torch.testing.assert_close(loss, expected)
    assert metrics["ar_aux_tokens"] == 2
    # The policy view is tempered; the causal CE is not.
    torch.testing.assert_close(
        seen["policy_logits"], logits[:, :noisy_width].float() / temperature
    )


@pytest.mark.parametrize(
    "backend,validation_mode,validation_runtime,error",
    [
        ("vllm", False, "megatron_llada", "Megatron backend"),
        ("vllm", True, "generation", "Megatron backend"),
        ("vllm", True, "megatron_llada", "TP=1"),
        ("megatron", False, None, "TP=1"),
    ],
)
def test_llada_generation_guard_uses_shared_diffusion_config(
    monkeypatch, backend, validation_mode, validation_runtime, error
):
    from types import SimpleNamespace
    from just_grpo.algorithms.llada_policy_worker import LLaDAJustGRPOPolicyWorkerImpl

    worker = object.__new__(LLaDAJustGRPOPolicyWorkerImpl)
    worker.cfg = {"generation": {"backend": backend}}
    worker.validation_mode = validation_mode
    worker.diffusion = SimpleNamespace(validation_runtime=validation_runtime)
    # Stop after backend selection, before CUDA or model execution.
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 2
    )
    with pytest.raises(ValueError, match=error):
        worker.generate(data=None)
