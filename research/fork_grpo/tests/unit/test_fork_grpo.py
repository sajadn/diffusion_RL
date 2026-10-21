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
"""Fork selection, recorded-state likelihood parity, and sparse loss coverage."""

from pathlib import Path

import pytest
import torch
from block_diffusion.denoising_schedule import aggregate_diffusion_logprobs
from block_diffusion.generation.megatron_generation import pack_responses, sample_batch
from fork_grpo.algorithm import ForkGRPO, ForkGRPOConfig
from fork_grpo.config import validate_config
from test_megatron_generation import TinyDoubled
from test_schedule import TinyDiffusion
from trace_grpo.algorithm import TraceGRPOSchedule

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def config() -> ForkGRPOConfig:
    sampling = dict(
        temperature=0.7,
        block_size=4,
        max_steps=4,
        selection_policy="fork",
        threshold=0.9,
        returns_reveal_steps=True,
        returns_entropy=False,
        emit_full_blocks=True,
    )
    return ForkGRPOConfig(
        runtime="megatron",
        generation_python=None,
        max_generation_kl=0.05,
        distributed=True,
        sampling=sampling,
        validation_sampling=sampling,
        schedule=dict(
            mask_token_id=31, block_size=4, num_level_samples=3, seed_base=42
        ),
    )


def prepare(
    prompts: list[list[int]],
    responses: list[dict],
    cfg: ForkGRPOConfig,
    stops: list[int],
    step: int = 0,
) -> BatchedDataDict:
    packed = pack_responses(
        prompts, responses, pad_token_id=0, max_new_tokens=8, stop_token_ids=stops
    )
    mask = torch.zeros_like(packed["output_ids"])
    messages = []
    for row, (prompt, response) in enumerate(zip(prompts, responses)):
        mask[row, len(prompt) : int(packed["unpadded_sequence_lengths"][row])] = 1
        messages.append(
            [
                {"role": "user", "token_ids": torch.tensor(prompt)},
                {
                    "role": "assistant",
                    "token_ids": torch.tensor(response["token_ids"]),
                    "reveal_steps": torch.tensor(response["reveal_steps"]),
                    "fallback_mask": torch.tensor(response["fallback_mask"]),
                },
            ]
        )
    data = BatchedDataDict(
        input_ids=packed["output_ids"],
        input_lengths=packed["unpadded_sequence_lengths"],
        token_mask=mask,
        sample_mask=torch.ones(len(prompts)),
        generation_logprobs=packed["logprobs"],
    )
    ForkGRPO(cfg, stop_token_ids=stops).prepare_training_data(data, messages, step)
    return data


def test_fork_parallel_then_leftmost_with_raw_confidence() -> None:
    cfg = config()
    cfg.sampling.temperature = 0.1

    def forward(ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        logits = torch.full((*ids.shape, 32), -100.0)
        logits[:, :, 0] = 0
        logits[:, :4, 1] = torch.tensor([0.0, 2.0, 8.0, 8.0])
        return logits

    response = sample_batch(
        forward,
        torch.tensor([[3]]),
        sampling=cfg.sampling,
        max_new_tokens=4,
        max_sequence_length=8,
        mask_token_id=31,
        stop_token_ids=[],
        generator=torch.Generator().manual_seed(2),
    )[0]
    # Positions 2/3 are confident; 0 and 1 require leftmost fallback. At T=.1,
    # position 1 would incorrectly exceed threshold if confidence were tempered.
    assert response["reveal_steps"] == [1, 2, 0, 0]
    assert response["fallback_mask"] == [True, True, False, False]


@pytest.mark.parametrize("prompt_length", [1, 3, 5])
@pytest.mark.parametrize("views", [1, 3, 4])
def test_fallback_logprobs_match_generation(prompt_length: int, views: int) -> None:
    torch.manual_seed(7)
    model = TinyDiffusion()
    cfg = config()
    cfg.schedule.num_level_samples = views
    prompts = [[1] * prompt_length, [2] * prompt_length]
    responses = sample_batch(
        TinyDoubled(model),
        torch.tensor(prompts),
        sampling=cfg.sampling,
        max_new_tokens=8,
        max_sequence_length=16,
        mask_token_id=31,
        stop_token_ids=[],
        generator=torch.Generator().manual_seed(19),
    )
    data = prepare(prompts, responses, cfg, [])
    with torch.no_grad():
        model.head.weight.div_(cfg.sampling.temperature)
        model.head.bias.div_(cfg.sampling.temperature)
    schedule = TraceGRPOSchedule(
        data, cfg.schedule, pad_token_id=0, padded_width=16, weighted_loss=False
    )
    scores = aggregate_diffusion_logprobs(
        schedule.iter_levels(data),
        model,
        original_shape=data["input_ids"].shape,
        device=torch.device("cpu"),
    )
    active = scores["logprob_token_mask"]
    assert active.any()
    assert schedule.num_steps == views
    assert not (active & ~data["trace_loss_mask"]).any()
    if views == cfg.schedule.block_size:
        torch.testing.assert_close(active, data["trace_loss_mask"])
    for row in range(data.size):
        for start in range(prompt_length, prompt_length + 8, 4):
            assert active[row, start : start + 4].sum() == min(
                views, int(data["trace_loss_mask"][row, start : start + 4].sum())
            )
    torch.testing.assert_close(
        scores["logprobs"][active],
        data["generation_logprobs"][active],
        atol=1e-6,
        rtol=1e-5,
    )
    train = TraceGRPOSchedule(
        data, cfg.schedule, pad_token_id=0, padded_width=16, weighted_loss=True
    )
    weights = torch.zeros(data.size)
    for scoring, training in zip(schedule.iter_levels(data), train.iter_levels(data)):
        torch.testing.assert_close(
            scoring["masked_indices"], training["masked_indices"]
        )
        weights += training["token_mask"].sum(1)
    torch.testing.assert_close(weights, torch.ones(data.size))


def test_mixed_empty_rollouts_and_terminal_context() -> None:
    cfg = config()
    responses = [
        dict(
            token_ids=[5, 0, 7, 31],
            logprobs=[-1.0] * 4,
            reveal_steps=[1, 2, 0, -1],
            fallback_mask=[True, True, True, False],
            response_length=2,
        ),
        dict(
            token_ids=[6] * 4,
            logprobs=[-1.0] * 4,
            reveal_steps=[0] * 4,
            fallback_mask=[False] * 4,
            response_length=4,
        ),
    ]
    data = prepare([[1], [2]], responses, cfg, [0])
    assert data["trace_loss_mask"].sum(1).tolist() == [2, 0]
    assert data["sample_mask"].tolist() == [1, 1]
    schedule = TraceGRPOSchedule(data, cfg.schedule, pad_token_id=0, weighted_loss=True)
    empty_shard = TraceGRPOSchedule(
        data.slice(1, 2), cfg.schedule, pad_token_id=0, weighted_loss=True
    )
    assert schedule.num_steps == empty_shard.num_steps == 3
    assert all(view["token_mask"].sum() == 0 for view in empty_shard.iter_levels())
    views = list(schedule.iter_levels())
    assert sum(view["token_mask"].sum() for view in views) == 1
    # The already committed post-stop token remains visible, never scored.
    assert all(not view["masked_indices"][0, 2] for view in views)
    assert all(view["token_mask"][0, 2] == 0 for view in views)


def test_no_fallback_has_finite_zero_loss_views() -> None:
    cfg = config()
    response = dict(
        token_ids=[5] * 4,
        logprobs=[-0.01] * 4,
        reveal_steps=[0] * 4,
        fallback_mask=[False] * 4,
        response_length=4,
    )
    data = prepare([[1]], [response], cfg, [])
    schedule = TraceGRPOSchedule(data, cfg.schedule, pad_token_id=0, weighted_loss=True)
    assert schedule.num_steps == 3
    view = next(schedule.iter_levels())
    assert torch.isfinite(view["token_mask"]).all()
    assert view["token_mask"].sum() == 0


def test_recipe_and_invalid_backend() -> None:
    register_omegaconf_resolvers()
    recipe = (
        Path(__file__).parents[2]
        / "configs/recipes/fork_grpo-deepscaler-3b-4n8g-megatron-vllm-async-long.yaml"
    )
    loaded = load_config(recipe)
    validate_config(loaded)
    # Prefix KV from the previous weights must not survive an async refit.
    assert loaded.grpo.async_grpo.recompute_kv_cache_after_weight_updates
    assert loaded.fork_grpo.schedule.num_level_samples == 3
    assert loaded.fork_grpo.sampling.max_steps == 16
    assert loaded.data.train.dataset_name == "DeepScaler"
    assert loaded.data.validation.dataset_name == "AIME2024"
    assert loaded.data.default.env_name == "math"
    assert set(loaded.env) == {"math"}
    bad = config().model_dump()
    bad["runtime"] = "upstream_vllm"
    with pytest.raises(ValueError, match="reference vLLM generation"):
        ForkGRPOConfig.model_validate(bad)
    bad = config().model_dump()
    bad["sampling"]["max_steps"] = 1
    with pytest.raises(ValueError, match="max_steps"):
        ForkGRPOConfig.model_validate(bad)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_rollout_transports_fallback_mask_without_exposing_context_tail(asynchronous):
    import asyncio
    from types import SimpleNamespace

    from nemo_rl.experience.rollouts import generate_responses, generate_responses_async

    output = pack_responses(
        [[1, 2, 3, 4]],
        [
            dict(
                token_ids=[5, 0, 7, 31],
                logprobs=[-1.0] * 4,
                reveal_steps=[0, 2, 1, -1],
                response_length=2,
                fallback_mask=[True, True, False, False],
            )
        ],
        pad_token_id=0,
        max_new_tokens=8,
        stop_token_ids=[0],
    )

    class Generation:
        cfg = {"backend": "megatron"}

        def generate(self, data, greedy=False):
            return output

        async def generate_async(self, data, greedy=False):
            yield 0, output

    tokenizer = SimpleNamespace(
        pad_token_id=0,
        batch_decode=lambda rows, **kw: [str(row.tolist()) for row in rows],
    )
    batch = BatchedDataDict(
        message_log=[
            [
                {
                    "role": "user",
                    "content": "prompt",
                    "token_ids": torch.tensor([1, 2, 3, 4]),
                }
            ]
        ]
    )
    inputs = BatchedDataDict(
        input_ids=torch.tensor([[1, 2, 3, 4]]), input_lengths=torch.tensor([4])
    )
    args = (Generation(), inputs, batch, tokenizer, inputs["input_lengths"])
    result, _, _ = (
        asyncio.run(generate_responses_async(*args))
        if asynchronous
        else generate_responses(*args)
    )
    message = result["message_log"][0][-1]
    assert message["content"] == "[5, 0]"
    assert message["token_ids"].tolist() == [5, 0, 7, 31]
    assert message["reveal_steps"].tolist() == [0, 2, 1, -1]
    assert message["fallback_mask"].tolist() == [True, True, False, False]


@pytest.mark.parametrize("confident_positions", [[0, 1, 2, 3], [3]])
def test_confident_singleton_is_not_a_fallback(confident_positions: list[int]) -> None:
    cfg = config()
    cfg.sampling.temperature = 0.0

    def forward(ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        logits = torch.full((*ids.shape, 32), -100.0)
        logits[:, :, :2] = 0.0
        logits[:, confident_positions, 0] = 12.0
        return logits

    output = sample_batch(
        forward,
        torch.tensor([[3]]),
        sampling=cfg.sampling,
        max_new_tokens=4,
        max_sequence_length=8,
        mask_token_id=31,
        stop_token_ids=[],
        generator=torch.Generator().manual_seed(2),
    )[0]
    for position in confident_positions:
        assert output["reveal_steps"][position] == 0
        assert not output["fallback_mask"][position]
    assert sum(output["fallback_mask"]) == 4 - len(confident_positions)


def test_missing_fallback_metadata_fails() -> None:
    data = BatchedDataDict(
        input_ids=torch.tensor([[1, 5]]),
        input_lengths=torch.tensor([2]),
        token_mask=torch.tensor([[0, 1]]),
        sample_mask=torch.ones(1),
    )
    messages = [
        [
            {"token_ids": torch.tensor([1])},
            {"token_ids": torch.tensor([5]), "reveal_steps": torch.tensor([0])},
        ]
    ]
    with pytest.raises(ValueError, match="fallback metadata"):
        ForkGRPO(config(), stop_token_ids=[]).prepare_training_data(data, messages, 0)


@pytest.mark.parametrize("prompt_length", [1, 3, 5])
def test_per_block_sampling_is_reproducible_and_unbiased(prompt_length: int) -> None:
    cfg = config()
    response = dict(
        token_ids=list(range(4, 12)),
        logprobs=[-1.0] * 8,
        reveal_steps=[0, 1, 2, 3, 0, 1, 0, 2],
        fallback_mask=[True] * 4 + [False, True, False, True],
        response_length=8,
    )
    estimates = {}
    for step in range(32):
        data = prepare([[1] * prompt_length], [response], cfg, [], step=step)
        repeat = prepare([[1] * prompt_length], [response], cfg, [], step=step)
        torch.testing.assert_close(
            data["trace_sampled_levels"], repeat["trace_sampled_levels"]
        )
        schedule = TraceGRPOSchedule(
            data, cfg.schedule, pad_token_id=0, padded_width=16, weighted_loss=True
        )
        assert schedule.num_steps == 3
        token_weights = torch.zeros(8)
        for view in schedule.iter_levels():
            token_weights += view["token_mask"][0, :8]
        for index, view in enumerate(schedule.iter_levels()):
            chosen = data["trace_sampled_levels"][0, index].repeat_interleave(4)
            expected_mask = torch.tensor(response["reveal_steps"]) >= chosen
            torch.testing.assert_close(view["masked_indices"][0, :8], expected_mask)
        selected = (token_weights[:4] != 0).nonzero().flatten().tolist()
        assert len(selected) == 3
        torch.testing.assert_close(token_weights[selected], torch.full((3,), 4 / 3 / 6))
        torch.testing.assert_close(token_weights[[5, 7]], torch.full((2,), 1 / 6))
        assert token_weights[[4, 6]].sum() == 0
        torch.testing.assert_close(token_weights.sum(), torch.tensor(1.0))
        # Enumerate the four possible subsets, then average their loss estimates.
        losses = torch.tensor([1.0, 2.0, 4.0, 8.0, 0.0, 16.0, 0.0, 32.0])
        estimates[tuple(selected)] = (losses * token_weights).sum()
    assert len(estimates) == 4
    torch.testing.assert_close(
        torch.stack(list(estimates.values())).mean(), torch.tensor(63 / 6)
    )


def test_replay_count_must_be_positive() -> None:
    cfg = config().model_dump()
    cfg["schedule"]["num_level_samples"] = 0
    with pytest.raises(ValueError, match="num_level_samples"):
        ForkGRPOConfig.model_validate(cfg)
