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
"""Recorded-context replay, consistent sampling, and terminal-block handling."""

from pathlib import Path

import pytest
import torch

from just_grpo.algorithms.trace_grpo import (
    TraceGRPO,
    TraceGRPOConfig,
    TraceGRPOSchedule,
)
from just_grpo.config import validate_config
from just_grpo.diffusion.denoising_schedule import aggregate_diffusion_logprobs
from just_grpo.generation.megatron_generation import pack_responses, sample_batch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from test_megatron_generation import TinyDoubled
from test_schedule import TinyDiffusion


def config(k=1, *, selection="confidence_threshold", reduction="sum"):
    sampling = dict(
        temperature=0.7,
        block_size=4,
        max_steps=4,
        selection_policy=selection,
        threshold=0.9,
        returns_reveal_steps=True,
        returns_entropy=False,
        emit_full_blocks=True,
    )
    return TraceGRPOConfig(
        runtime="megatron",
        generation_python=None,
        max_generation_kl=0.05,
        distributed=True,
        sampling=sampling,
        validation_sampling=sampling,
        schedule=dict(
            mask_token_id=31,
            block_size=4,
            num_level_samples=k,
            seed_base=42,
            sampled_level_reduction=reduction,
        ),
    )


def prepare(prompts, responses, cfg, *, step=3, stops=()):
    packed = pack_responses(
        prompts, responses, pad_token_id=0, max_new_tokens=8, stop_token_ids=list(stops)
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
    TraceGRPO(cfg, stop_token_ids=list(stops)).prepare_training_data(
        data, messages, step
    )
    return data


@pytest.mark.parametrize("k", [1, 2, 6])
@pytest.mark.parametrize("selection", ["leftmost", "confidence_threshold"])
def test_sampled_trace_scores_match_recorded_generation(k, selection):
    torch.manual_seed(7)
    model = TinyDiffusion()
    cfg = config(k, selection=selection)
    prompts = [[31, 1, 2, 3], [31, 4, 5, 6]]
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
    with torch.no_grad():
        model.head.weight.div_(cfg.sampling.temperature)
        model.head.bias.div_(cfg.sampling.temperature)
    data = prepare(prompts, responses, cfg)
    schedule = TraceGRPOSchedule(
        data, cfg.schedule, pad_token_id=0, padded_width=16, weighted_loss=False
    )
    calls = []

    def score(view):
        calls.append(view)
        return model(view)

    scores = aggregate_diffusion_logprobs(
        schedule.iter_levels(data),
        score,
        original_shape=data["input_ids"].shape,
        device=torch.device("cpu"),
    )
    selected = data["logprob_token_mask"]
    assert len(calls) == k
    torch.testing.assert_close(
        scores[selected], data["generation_logprobs"][selected], atol=1e-6, rtol=1e-5
    )
    assert (scores[~selected] == 0).all()
    # The same data sent separately for prev/reference/train must yield identical contexts.
    train = TraceGRPOSchedule(
        data, cfg.schedule, pad_token_id=0, padded_width=16, weighted_loss=True
    )
    for scored, trained in zip(calls, train.iter_levels(data)):
        assert torch.equal(scored["masked_indices"], trained["masked_indices"])
        assert torch.equal(scored["token_mask"].bool(), trained["token_mask"].bool())


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_level_draws_and_weights_survive_row_sharding(reduction):
    cfg = config(2, reduction=reduction)
    prompts = [[31, 1, 2, 3]] * 2
    responses = [
        dict(
            token_ids=[4, 5, 6, 7],
            logprobs=[-1.0] * 4,
            reveal_steps=[0, 7, 0, 2],
            response_length=4,
        ),
        dict(
            token_ids=[8, 9, 10, 11],
            logprobs=[-1.0] * 4,
            reveal_steps=[0, 0, 0, 0],
            response_length=4,
        ),
    ]
    data = prepare(prompts, responses, cfg)
    assert data["trace_reveal_levels"][0, 4:].tolist() == [0, 2, 0, 1]
    assert len(data["trace_sampled_levels"][0].unique()) == 2
    expected = [1.5, 1.0] if reduction == "sum" else [0.5, 1.0]
    torch.testing.assert_close(data["trace_loss_weights"], torch.tensor(expected))
    full = TraceGRPOSchedule(data, cfg.schedule, pad_token_id=0, weighted_loss=True)
    for row in range(2):
        shard = TraceGRPOSchedule(
            data.slice(row, row + 1), cfg.schedule, pad_token_id=0, weighted_loss=True
        )
        for all_rows, single in zip(full.iter_levels(), shard.iter_levels()):
            assert torch.equal(
                all_rows["masked_indices"][row : row + 1], single["masked_indices"]
            )
            assert torch.equal(
                all_rows["token_mask"][row : row + 1], single["token_mask"]
            )
    # Depth-one rows still execute both views, with one zero-loss placeholder.
    assert sum(int(v["token_mask"][1].count_nonzero()) for v in full.iter_levels()) == 4


def test_post_stop_context_is_replayed_but_never_trained():
    cfg = config(3)
    data = prepare(
        [[31, 1, 2, 3]],
        [
            dict(
                token_ids=[4, 0, 8, 31],
                logprobs=[-1.0] * 4,
                reveal_steps=[0, 7, 2, -1],
                response_length=2,
            )
        ],
        cfg,
        stops=[0],
    )
    schedule = TraceGRPOSchedule(
        data, cfg.schedule, pad_token_id=0, weighted_loss=False
    )
    data["trace_sampled_levels"][0] = torch.tensor([0, 1, 2])
    views = list(schedule.iter_levels())
    assert views[2]["masked_indices"][0, 4:].tolist() == [False, True, False, True]
    assert views[2]["token_mask"][0, 4:].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert not data["logprob_token_mask"][0, 6:].any()


def test_missing_reveal_metadata_fails_before_scoring():
    cfg = config()
    with pytest.raises(ValueError, match="recorded reveal step"):
        prepare(
            [[31, 1, 2, 3]],
            [
                dict(
                    token_ids=[4, 5],
                    logprobs=[-1.0, -1.0],
                    reveal_steps=[0, -1],
                    response_length=2,
                )
            ],
            cfg,
        )


def test_sampling_changes_by_step_and_repeats_exactly():
    cfg = config()
    response = dict(
        token_ids=list(range(8)),
        logprobs=[-1.0] * 8,
        reveal_steps=[0, 1, 2, 3] * 2,
        response_length=8,
    )
    first = prepare([[31, 1, 2, 3]], [response], cfg, step=0)
    same = prepare([[31, 1, 2, 3]], [response], cfg, step=0)
    assert torch.equal(first["trace_sampled_levels"], same["trace_sampled_levels"])
    draws = {
        int(
            prepare([[31, 1, 2, 3]], [response], cfg, step=step)[
                "trace_sampled_levels"
            ][0, 0]
        )
        for step in range(16)
    }
    assert draws == {0, 1, 2, 3}


def test_confidence_stop_retains_replay_context_and_semantic_length():
    cfg = config(4)
    # Position 2 commits EOS first. Earlier positions are filled on the next call.
    calls = 0

    def forward(ids, positions):
        nonlocal calls
        calls += 1
        logits = torch.zeros((*ids.shape, 32))
        logits[:, 6, 0] = 40
        if calls > 1:
            logits[:, 4:6, 5] = 40
        return logits

    responses = sample_batch(
        forward,
        torch.tensor([[31, 1, 2, 3]]),
        sampling=cfg.sampling,
        max_new_tokens=8,
        max_sequence_length=16,
        mask_token_id=31,
        stop_token_ids=[0],
        generator=torch.Generator().manual_seed(3),
    )
    response = responses[0]
    assert calls == 2
    assert response["token_ids"] == [5, 5, 0, 31]
    assert response["reveal_steps"] == [1, 1, 0, -1]
    assert response["response_length"] == 3
    packed = pack_responses(
        [[31, 1, 2, 3]], responses, pad_token_id=0, max_new_tokens=8, stop_token_ids=[0]
    )
    assert packed["response_lengths"].tolist() == [3]
    assert packed["generation_lengths"].tolist() == [4]
    prepare([[31, 1, 2, 3]], responses, cfg, stops=[0])


@pytest.mark.parametrize("async_mode", [False, True])
def test_trace_recipes_validate(async_mode):
    register_omegaconf_resolvers()
    suffix = "-async" if async_mode else ""
    path = (
        Path(__file__).parents[2]
        / f"configs/recipes/trace_grpo-sudoku6x6-4n8g-megatron-inference{suffix}-long.yaml"
    )
    raw = load_config(path)
    parsed = validate_config(raw)
    assert parsed.schedule.num_level_samples == 1
    raw.trace_grpo.schedule.num_level_samples = 3
    assert validate_config(raw).schedule.num_level_samples == 3
    raw.grpo.seq_logprob_error_threshold = 2
    with pytest.raises(ValueError, match="whole-response"):
        validate_config(raw)


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_sample_count_is_rejected(k):
    with pytest.raises(ValueError):
        config(k)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_rollout_transports_reveal_steps_without_exposing_context_tail(asynchronous):
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
