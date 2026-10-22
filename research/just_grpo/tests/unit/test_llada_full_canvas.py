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
"""LLaDA canvas geometry, leakage checks, and positive auxiliary objective."""

import pytest
import torch
from block_diffusion.full_canvas import FullCanvasLayout
from block_diffusion.positive_diffusion import (
    PositiveDiffusionConfig,
    PositiveDiffusionLoss,
    build_positive_diffusion_level,
)
from just_grpo.algorithms.llada_schedule import LLaDAJustGRPOSchedule
from just_grpo.generation.llada import generate_llada

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def batch():
    return BatchedDataDict(
        input_ids=torch.tensor(
            [[11, 12, 21, 22, 23, 24, 0], [13, 14, 15, 31, 32, 33, 34]]
        ),
        input_lengths=torch.tensor([6, 7]),
        token_mask=torch.tensor([[0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1]]),
        sample_mask=torch.ones(2),
        advantages=torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            ]
        ),
    )


@pytest.mark.parametrize("step", range(4))
def test_replay_masks_entire_future_and_selects_one_token(step):
    data = batch()
    schedule = LLaDAJustGRPOSchedule(data, canvas_length=4)
    level = schedule.prepare_trajectory(step, data)
    noisy = level["target_ids"].masked_fill(level["masked_indices"], 99)
    for row, start in enumerate([2, 3]):
        assert torch.equal(
            noisy[row, : start + step], data["input_ids"][row, : start + step]
        )
        assert (noisy[row, start + step : start + 4] == 99).all()
        assert level["token_mask"][row].nonzero().flatten().tolist() == [start + step]
    assert level["canvas_lengths"].tolist() == [6, 7]


def test_replay_does_not_observe_different_future_targets():
    data = batch()
    other = batch()
    other["input_ids"][:, 5:] = 90
    a = LLaDAJustGRPOSchedule(data, canvas_length=4).generate_single_trajectory(0)
    b = LLaDAJustGRPOSchedule(other, canvas_length=4).generate_single_trajectory(0)
    for row, end in enumerate([6, 7]):
        assert torch.equal(
            a["target_ids"].masked_fill(a["masked_indices"], 99)[row, :end],
            b["target_ids"].masked_fill(b["masked_indices"], 99)[row, :end],
        )


def test_auxiliary_masks_only_positive_response_and_preserves_prompt():
    data = batch()
    config = PositiveDiffusionConfig(
        mask_token_id=99,
        block_size=4,
        mask_probability_min=1.0,
        mask_probability_max=1.0,
    )
    level, count = build_positive_diffusion_level(
        data,
        config,
        pad_token_id=0,
        padded_width=8,
        stop_token_ids=[],
        generator=torch.Generator().manual_seed(42),
        layout=FullCanvasLayout(data, canvas_length=4),
    )
    assert int(count) == 8
    assert level["diffusion_positive_samples"].tolist() == [True, False]
    assert level["token_mask"][0].nonzero().flatten().tolist() == [2, 3, 4, 5]
    assert not level["token_mask"][1].any()
    assert not level["masked_indices"][:, :2].any()
    assert not level["masked_indices"][0, 6]


def test_auxiliary_zero_positives_keeps_every_row():
    data = batch()
    data["advantages"].fill_(-1)
    config = PositiveDiffusionConfig(mask_token_id=99, block_size=4)
    level, count = build_positive_diffusion_level(
        data,
        config,
        pad_token_id=0,
        padded_width=8,
        stop_token_ids=[],
        generator=torch.Generator().manual_seed(42),
        layout=FullCanvasLayout(data, canvas_length=4),
    )
    assert level["input_ids"].shape == data["input_ids"].shape
    assert not level["token_mask"].any()
    assert int(count) == 8


def test_auxiliary_inverse_probability_weighting_and_gradients():
    config = PositiveDiffusionConfig(mask_token_id=99, block_size=4, weight=0.1)
    probs = torch.full((2, 4), -2.0, requires_grad=True)
    data = BatchedDataDict(
        token_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        sample_mask=torch.ones(2),
        diffusion_mask_probability=torch.tensor([0.5, 0.75]),
        diffusion_positive_samples=torch.tensor([True, False]),
    )
    loss, metrics = PositiveDiffusionLoss(config)(
        data,
        torch.tensor(2.0),
        torch.tensor(8.0),
        next_token_logprobs=probs,
        shift_labels=False,
    )
    assert float(loss.detach()) == pytest.approx(0.1)
    loss.backward()
    assert probs.grad[0, :2].tolist() == pytest.approx([-0.025, -0.025])
    assert not probs.grad[1].any()


@pytest.mark.parametrize("block", [1, 2])
def test_generation_allocates_full_canvas_and_uses_one_reveal_per_pass(block):
    seen = []

    def forward(tokens):
        seen.append(tokens.clone())
        logits = torch.zeros((*tokens.shape, 100), dtype=torch.bfloat16)
        logits[..., 7] = 8
        return logits

    generated = generate_llada(
        forward,
        torch.tensor([[11, 12]]),
        completion_length=4,
        block_length=block,
        mask_token_id=99,
        temperature=0.0,
    )
    result = generated.token_ids
    assert result.tolist() == [[11, 12, 7, 7, 7, 7]]
    assert len(seen) == 4
    assert all(t.shape[1] == 6 for t in seen)
    assert [int((t == 99).sum()) for t in seen] == [4, 3, 2, 1]


def test_leftmost_generation_advances_even_if_mask_is_sampled():
    seen = []

    def forward(tokens):
        seen.append(tokens.clone())
        logits = torch.zeros((*tokens.shape, 100), dtype=torch.bfloat16)
        logits[..., 99] = 8 if len(seen) == 1 else -8
        logits[..., 7] = 4
        return logits

    generated = generate_llada(
        forward,
        torch.tensor([[11]]),
        completion_length=4,
        block_length=1,
        mask_token_id=99,
        temperature=0.0,
    )
    result = generated.token_ids
    assert result.tolist() == [[11, 99, 7, 7, 7]]
    assert len(seen) == 4


@pytest.mark.parametrize("auxiliary", [False, True])
def test_production_accepts_128_dp_ranks_with_eight_completions_each(auxiliary):
    from pathlib import Path

    from just_grpo.config import validate_config

    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    suffix = "-diffaux" if auxiliary else ""
    recipe = (
        Path(__file__).parents[2]
        / "configs/recipes"
        / (
            "just_grpo-gsm8k-llada8b-canvas256-16n8g-megatron-inference-long"
            + suffix
            + ".yaml"
        )
    )
    config = load_config(recipe)
    parsed = validate_config(config)
    dp = config.cluster.num_nodes * config.cluster.gpus_per_node
    assert dp == 128
    assert config.grpo.num_prompts_per_step == 64
    assert config.grpo.num_generations_per_prompt == 16
    assert config.policy.train_global_batch_size // dp == 8
    assert parsed.schedule.block_size == 256
    assert parsed.validation_sampling.block_size == 32
