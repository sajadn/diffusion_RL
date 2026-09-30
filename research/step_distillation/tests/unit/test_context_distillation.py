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
"""Check positive-only next-state KD, full-distribution gradients and metadata."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from step_distillation.context_distillation import (
    ContextDistillationLoss,
    LessMaskedKDSchedule,
    prepare_context_distillation_data,
    score_teacher_context,
)
from step_distillation.config import StepDistillationConfig
from just_grpo.algorithms.trace_grpo import TraceScheduleConfig
from step_distillation.config import validate_config
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def state():
    # Student has revealed positions 8 and 10 in the current block; later blocks
    # and every remaining token in this block must be hidden from the teacher.
    ids = torch.tensor(
        [
            [
                29,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
            ]
        ]
    )
    levels = torch.tensor(
        [[-1] * 8 + [0, 1, 0, 1, 2, 2, 3, 3] + [4, 4, 5, 5, 6, 6, 7, 7]]
    )
    data = BatchedDataDict(
        input_ids=ids,
        input_lengths=torch.tensor([24]),
        token_mask=torch.tensor([[0.0] * 8 + [1.0] * 16]),
        sample_mask=torch.ones(1),
        trace_reveal_levels=levels,
        trace_loss_mask=levels >= 0,
        trace_sampled_levels=torch.tensor([[1]]),
        trace_loss_weights=torch.ones(1),
        distillation_seeds=torch.tensor([42]),
        rewards=torch.zeros(1),
    )
    schedule = TraceScheduleConfig(
        mask_token_id=29,
        block_size=8,
        num_level_samples=1,
        seed_base=42,
        sampled_level_reduction="mean",
    )
    return data, schedule, StepDistillationConfig()


def prepare(reward=1.0, seed=42):
    data, schedule, _ = state()
    data["rewards"].fill_(reward)
    config = StepDistillationConfig(target_source="less_masked_kd", positive_only=True)
    prepare_context_distillation_data(
        data, config=config, block_size=8, seeds=torch.tensor([seed])
    )
    return data, schedule


def test_next_state_is_same_block_and_targets_still_masked_in_both_views():
    seen = set()
    for seed in range(40):
        data, schedule = prepare(seed=seed)
        first = int(data["trace_sampled_levels"][0, 0])
        second = int(data["kd_teacher_levels"][0])
        start = int(data["kd_block_starts"][0])
        seen.add(first)
        assert second == first + 1
        expected = torch.zeros_like(data["input_ids"], dtype=torch.bool)
        expected[0, start : start + 8] = (
            data["trace_reveal_levels"][0, start : start + 8] >= second
        )
        torch.testing.assert_close(data["distillation_target_mask"], expected)
        recorded = []

        def forward(ids, positions):
            recorded.append(ids.clone())
            return torch.arange(48 * 32, dtype=torch.float32).reshape(1, 48, 32)

        targets = score_teacher_context(
            forward,
            data,
            schedule_config=schedule,
            pad_token_id=0,
            padded_width=24,
            vocab_size=32,
        )
        assert len(recorded) == 1
        row = recorded[0][0, :24]
        levels = data["trace_reveal_levels"][0]
        assert (
            row[start : start + 8][levels[start : start + 8] < second].tolist()
            == data["input_ids"][0, start : start + 8][
                levels[start : start + 8] < second
            ].tolist()
        )
        assert (row[expected[0]] == 29).all()
        assert (row[start + 8 :] == 29).all()
        assert row[:start].tolist() == data["input_ids"][0, :start].tolist()
        data.update(targets)
        view = next(
            LessMaskedKDSchedule(
                data, schedule, pad_token_id=0, weighted_loss=True
            ).iter_levels(data)
        )
        assert view["masked_indices"][expected].all()
        assert torch.equal(view["token_mask"].bool(), expected)
    assert seen == {0, 1, 2, 4, 5, 6}  # No cross-block pair or completed teacher block.


def test_negative_rollouts_run_no_teacher_and_have_zero_loss():
    for reward in (0.0, -1.0):
        data, schedule = prepare(reward)
        assert data["sample_mask"].item() == 0
        assert not data["distillation_target_mask"].any()

        def unexpected(*args):
            raise AssertionError("Teacher must not score filtered rows")

        data.update(
            score_teacher_context(
                unexpected,
                data,
                schedule_config=schedule,
                pad_token_id=0,
                padded_width=24,
                vocab_size=32,
            )
        )
        assert data["kd_teacher_logits"][0].numel() == 0
        view = next(
            LessMaskedKDSchedule(
                data, schedule, pad_token_id=0, weighted_loss=True
            ).iter_levels(data)
        )
        logits = torch.randn(1, 48, 32, requires_grad=True)
        loss, _ = ContextDistillationLoss(vocab_size=32, temperature=1.0)(
            view, torch.tensor(0.0), torch.tensor(0.0), logits=logits
        )
        assert loss.item() == 0
        loss.backward()
        assert not logits.grad.any()


def test_full_forward_kl_value_gradient_temperature_and_teacher_detachment():
    teacher = torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]], requires_grad=True)
    # Upstream has already divided student logits by temperature=2.
    logits = torch.tensor(
        [[[0.4, 0.1, -0.3], [0.1, -0.4, 0.2], [0.0, 0.0, 0.0]]], requires_grad=True
    )
    data = BatchedDataDict(
        target_ids=torch.zeros(1, 3, dtype=torch.long),
        token_mask=torch.tensor([[1.0, 1.0, 0.0]]),
        sample_mask=torch.ones(1),
        kd_block_starts=torch.tensor([0]),
        kd_teacher_logits=[teacher],
    )
    loss, metrics = ContextDistillationLoss(vocab_size=3, temperature=2.0)(
        data, torch.tensor(1.0), torch.tensor(2.0), logits=logits
    )
    p = (teacher.detach() / 2).softmax(-1)
    q = logits.detach()[0, :2].softmax(-1)
    expected = (p * (p.log() - q.log())).sum() / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(logits.grad[0, :2], (q - p) / 2)
    assert not logits.grad[0, 2].any()
    assert teacher.grad is None
    assert metrics["distillation_forward_kl"] == loss.item()


def test_block_logits_survive_gather_slice_and_remain_non_sequence_metadata():
    data, schedule = prepare()
    data.update(
        score_teacher_context(
            lambda *args: torch.randn(1, 48, 32),
            data,
            schedule_config=schedule,
            pad_token_id=0,
            padded_width=24,
            vocab_size=32,
        )
    )
    combined = BatchedDataDict.from_batches([data, data])
    assert len(combined["kd_teacher_logits"]) == 2
    assert combined["kd_teacher_logits"][0].shape == (8, 32)
    view = next(
        LessMaskedKDSchedule(
            combined.slice(1, 2),
            schedule,
            pad_token_id=0,
            padded_width=32,
            weighted_loss=True,
        ).iter_levels(combined.slice(1, 2))
    )
    assert view["input_ids"].shape == (1, 32)
    assert isinstance(view["kd_teacher_logits"], list)
    assert view["kd_teacher_logits"][0].shape == (8, 32)
    logits = torch.randn(1, 64, 32, requires_grad=True)
    loss, _ = ContextDistillationLoss(vocab_size=32, temperature=1.0)(
        view, torch.tensor(1.0), view["token_mask"].sum(), logits=logits
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_context_kd_excludes_post_eos_and_no_shared_mask_states():
    data, schedule, _ = state()
    data["trace_loss_mask"].zero_()
    data["trace_loss_mask"][0, 8] = True
    data["rewards"].fill_(1.0)
    prepare_context_distillation_data(
        data,
        config=StepDistillationConfig(target_source="less_masked_kd"),
        block_size=8,
        seeds=torch.tensor([42]),
    )
    assert (
        data["sample_mask"].item() == 0
    )  # Only scorable token is already visible in next state.


def test_positive_kd_recipe_and_guards():
    register_omegaconf_resolvers()
    path = (
        Path(__file__).parents[2]
        / "configs/recipes/step_distillation-deepscaler-3b-4n8g-megatron-vllm-positive-kd-long.yaml"
    )
    cfg = load_config(path)
    parsed = validate_config(cfg)
    assert cfg.grpo.num_prompts_per_step == 128
    assert cfg.grpo.num_generations_per_prompt == 8
    assert cfg.policy.train_global_batch_size == 1024
    assert not cfg.grpo.get("freeze_generation")
    assert parsed.step_distillation.target_source == "less_masked_kd"
    for key, value in (
        ("step_distillation.positive_only", False),
        ("grpo.num_generations_per_prompt", 1),
    ):
        bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.update(bad, key, value)
        with pytest.raises(
            ValueError, match="positive-only filtering and multiple rollouts"
        ):
            validate_config(bad)


def test_callback_prepares_kd_before_sharding_teacher_requests():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from step_distillation.context_distillation import ContextDistillation

    data, schedule, distillation = state()
    data["rewards"].fill_(1.0)
    expected = BatchedDataDict({key: value.clone() for key, value in data.items()})
    seeds = torch.tensor([schedule.seed_base + 7 * 1_000_003])
    prepare_context_distillation_data(
        expected, config=distillation, block_size=8, seeds=seeds
    )
    policy = Mock(data_parallel_size=1)
    group = policy.worker_group

    def dispatch(method, *, data, **kwargs):
        assert method == "get_frozen_teacher_targets"
        for key in (
            "trace_sampled_levels",
            "kd_teacher_levels",
            "distillation_target_mask",
        ):
            torch.testing.assert_close(data[0][key], expected[key])
        return ["teacher-request"]

    group.run_all_workers_sharded_data.side_effect = dispatch
    group.get_all_worker_results.return_value = [
        BatchedDataDict(
            kd_teacher_logits=[torch.zeros(8, 32, dtype=torch.bfloat16)],
            distillation_target_mask=expected["distillation_target_mask"],
        )
    ]
    callback = ContextDistillation(
        SimpleNamespace(schedule=schedule, step_distillation=distillation),
        stop_token_ids=[],
        policy=policy,
    )
    messages = [
        [
            {"token_ids": data["input_ids"][0, :8]},
            {
                "token_ids": data["input_ids"][0, 8:],
                "reveal_steps": data["trace_reveal_levels"][0, 8:],
            },
        ]
    ]
    callback.prepare_training_data(data, messages, step=7)
    policy.prepare_for_lp_inference.assert_called_once()
    group.run_all_workers_sharded_data.assert_called_once()
    assert data["kd_teacher_logits"][0].shape == (8, 32)
