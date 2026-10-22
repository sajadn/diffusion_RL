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
"""Full-canvas Fork decoding, bounded replay, and exact context coverage."""

from pathlib import Path

import pytest
import torch
from block_diffusion.denoising_schedule import aggregate_diffusion_logprobs
from block_diffusion.full_canvas import FullCanvasLayout
from fork_grpo.algorithm import ForkGRPO, ForkGRPOConfig
from fork_grpo.config import validate_config
from just_grpo.generation.llada import generate_llada
from omegaconf import OmegaConf
from trace_grpo.algorithm import TraceGRPOSchedule

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def config(canvas: int, cap: int) -> ForkGRPOConfig:
    sampling = dict(
        temperature=0.7,
        block_size=canvas,
        max_steps=canvas,
        selection_policy="fork",
        threshold=0.9,
        returns_reveal_steps=True,
        returns_entropy=False,
        emit_full_blocks=True,
    )
    return ForkGRPOConfig(
        runtime="megatron_llada",
        generation_python=None,
        distributed=True,
        max_generation_kl=0.1,
        schedule=dict(
            mask_token_id=15, block_size=canvas, num_level_samples=cap, seed_base=42
        ),
        sampling=sampling,
        validation_sampling=sampling,
    )


def prepare(
    tokens: torch.Tensor,
    steps: torch.Tensor,
    fallback: torch.Tensor,
    cfg: ForkGRPOConfig,
    prompt_length: int,
    step: int = 0,
) -> BatchedDataDict:
    mask = torch.ones_like(tokens)
    mask[:, :prompt_length] = 0
    data = BatchedDataDict(
        input_ids=tokens,
        input_lengths=torch.full((len(tokens),), tokens.shape[1]),
        token_mask=mask,
        sample_mask=torch.ones(len(tokens)),
    )
    logs = [
        [
            {"role": "user", "token_ids": row[:prompt_length]},
            {
                "role": "assistant",
                "token_ids": row[prompt_length:],
                "reveal_steps": steps[i, prompt_length:],
                "fallback_mask": fallback[i, prompt_length:],
            },
        ]
        for i, row in enumerate(tokens)
    ]
    ForkGRPO(cfg, stop_token_ids=[]).prepare_training_data(data, logs, step)
    return data


def schedule(
    data: BatchedDataDict, cfg: ForkGRPOConfig, weighted: bool
) -> TraceGRPOSchedule:
    return TraceGRPOSchedule(
        data,
        cfg.schedule,
        pad_token_id=0,
        weighted_loss=weighted,
        layout=FullCanvasLayout(data, canvas_length=cfg.schedule.block_size),
    )


def test_leftmost_fallback_is_distinct_from_highest_confidence() -> None:
    def forward(ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full((*ids.shape, 16), -100.0)
        logits[:, :, :2] = 0
        logits[:, 2, 0] = 2
        logits[:, 3:, 0] = 10
        return logits

    out = generate_llada(
        forward,
        torch.tensor([[7]]),
        completion_length=4,
        block_length=4,
        mask_token_id=15,
        temperature=0,
        threshold=0.9,
        leftmost_fallback=True,
    )
    assert out.reveal_steps.tolist() == [[-1, 1, 2, 0, 0]]
    assert out.fallback_mask.tolist() == [[False, True, True, False, False]]
    assert out.num_forward_passes.tolist() == [3]


@pytest.mark.parametrize("cap", [1, 3, 64])
@pytest.mark.parametrize("prompt_length", [1, 3])
def test_full_canvas_replay_matches_generation(cap: int, prompt_length: int) -> None:
    torch.manual_seed(12)
    embedding = torch.nn.Embedding(16, 8)
    projection = torch.nn.Linear(8, 16)

    def forward(ids: torch.Tensor) -> torch.Tensor:
        hidden = embedding(ids)
        logits = projection(hidden + hidden.mean(1, keepdim=True)).bfloat16()
        logits[..., 15] = -100
        return logits

    cfg = config(8, cap)
    out = generate_llada(
        forward,
        torch.ones((2, prompt_length), dtype=torch.long),
        completion_length=8,
        block_length=4,
        mask_token_id=15,
        temperature=0.7,
        threshold=0.9,
        leftmost_fallback=True,
        generator=torch.Generator().manual_seed(19),
    )
    data = prepare(
        out.token_ids, out.reveal_steps, out.fallback_mask, cfg, prompt_length
    )
    calls = []

    def score(view: BatchedDataDict) -> torch.Tensor:
        calls.append(view)
        ids = view["target_ids"].masked_fill(view["masked_indices"], 15)
        logits = forward(ids).float() / cfg.sampling.temperature
        return (
            logits.log_softmax(-1).gather(-1, view["target_ids"][..., None]).squeeze(-1)
        )

    scoring = schedule(data, cfg, False)
    scores = aggregate_diffusion_logprobs(
        scoring.iter_levels(),
        score,
        original_shape=out.token_ids.shape,
        device=torch.device("cpu"),
    )
    active = scores["logprob_token_mask"]
    assert active.any()
    assert not (active & ~out.fallback_mask).any()
    assert len(calls) == min(cap, int(out.fallback_mask.sum(1).max()))
    torch.testing.assert_close(scores["logprobs"][active], out.logprobs[active])
    weighted = list(schedule(data, cfg, True).iter_levels())
    torch.testing.assert_close(
        sum(v["token_mask"].sum(1) for v in weighted), torch.ones(2)
    )
    for scoring_view, training_view in zip(calls, weighted):
        torch.testing.assert_close(
            scoring_view["masked_indices"], training_view["masked_indices"]
        )
    for row in range(2):
        shard = list(schedule(data.slice(row, row + 1), cfg, True).iter_levels())
        assert len(shard) == len(weighted)
        for full, single in zip(weighted, shard):
            torch.testing.assert_close(
                full["token_mask"][row : row + 1], single["token_mask"]
            )
            torch.testing.assert_close(
                full["masked_indices"][row : row + 1], single["masked_indices"]
            )


@pytest.mark.parametrize(
    "fallback_count, expected_views", [(0, 1), (3, 3), (64, 64), (130, 64)]
)
def test_64_view_bound_and_empty_shards(
    fallback_count: int, expected_views: int
) -> None:
    cfg = config(130, 64)
    tokens = torch.full((2, 131), 7)
    steps = torch.arange(131).expand(2, -1) - 1
    fallback = torch.zeros_like(tokens, dtype=torch.bool)
    fallback[0, 1 : 1 + fallback_count] = True
    data = prepare(tokens, steps, fallback, cfg, 1)
    replay = schedule(data, cfg, True)
    assert replay.num_steps == expected_views <= 64
    mask = sum(view["token_mask"] for view in replay.iter_levels())
    assert mask[0].count_nonzero() == min(fallback_count, 64)
    torch.testing.assert_close(
        mask.sum(1), torch.tensor([float(fallback_count > 0), 0.0])
    )
    empty = schedule(data.slice(1, 2), cfg, True)
    assert empty.num_steps == expected_views
    assert all(view["token_mask"].sum() == 0 for view in empty.iter_levels())


def test_uniform_subset_weights_preserve_fallback_mean() -> None:
    cfg = config(4, 2)
    tokens = torch.full((1, 5), 7)
    steps = torch.arange(5)[None] - 1
    fallback = steps >= 0
    estimates = {}
    for step in range(64):
        data = prepare(tokens, steps, fallback, cfg, 1, step)
        repeat = prepare(tokens, steps, fallback, cfg, 1, step)
        torch.testing.assert_close(
            data["trace_sampled_levels"], repeat["trace_sampled_levels"]
        )
        mask = sum(
            view["token_mask"] for view in schedule(data, cfg, True).iter_levels()
        )
        selected = tuple(mask[0].nonzero().flatten().tolist())
        estimates[selected] = (mask * torch.tensor([[0.0, 1.0, 2.0, 4.0, 8.0]])).sum()
    assert len(estimates) == 6
    torch.testing.assert_close(
        torch.stack(list(estimates.values())).mean(), torch.tensor(15 / 4)
    )


def test_sampled_mask_token_is_committed_only_once() -> None:
    def forward(ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*ids.shape, 16))
        logits[..., 15] = 0.1
        return logits

    out = generate_llada(
        forward,
        torch.tensor([[1]]),
        completion_length=4,
        block_length=4,
        mask_token_id=15,
        temperature=0,
        threshold=0.9,
        leftmost_fallback=True,
    )
    assert out.num_forward_passes.tolist() == [4]
    assert out.reveal_steps.tolist() == [[-1, 0, 1, 2, 3]]
    assert out.fallback_mask.sum() == 4


def test_llada_recipe_and_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    from block_diffusion import training
    from run_fork_grpo import run

    monkeypatch.setenv("MATH12K_DATA_DIR", "/data")
    register_omegaconf_resolvers()
    recipe = (
        Path(__file__).parents[2]
        / "configs/recipes/fork_grpo-math12k-llada8b-canvas256-4n8g-megatron-long.yaml"
    )
    raw = load_config(recipe)
    parsed = validate_config(raw)
    assert parsed.schedule.num_level_samples == 64
    assert raw.policy.generation.max_new_tokens == 256
    captured = []
    monkeypatch.setattr(
        training, "run", lambda *args, **kwargs: captured.append(kwargs)
    )
    run(raw)
    assert captured[0]["generation_factory"].__name__ == "LLaDAMegatronGeneration"
    assert captured[0]["validation_fn"].__name__ == "validate_llada"
    assert (
        captured[0]["prepare_training_data_factory"]([]).__self__.config.runtime
        == "megatron_llada"
    )


@pytest.mark.parametrize("dataset", ["math12k", "gsm8k"])
def test_llada_fork_validation_matches_primary_baseline(
    monkeypatch: pytest.MonkeyPatch, dataset: str
) -> None:
    monkeypatch.setenv("MATH12K_DATA_DIR", "/data")
    register_omegaconf_resolvers()
    root = Path(__file__).parents[4]
    raw = load_config(
        root
        / "research/fork_grpo/configs/recipes"
        / f"fork_grpo-{dataset}-llada8b-canvas256-4n8g-megatron-long.yaml"
    )
    baseline = load_config(
        root
        / "research/just_grpo/configs/recipes"
        / "just_grpo-gsm8k-llada8b-canvas256-16n8g-megatron-inference-long.yaml"
    )
    cfg = validate_config(raw)
    expected = baseline.just_grpo.validation_modes.fixed1_t1
    for key in (
        "temperature",
        "block_size",
        "max_steps",
        "selection_policy",
        "threshold",
        "returns_reveal_steps",
        "returns_entropy",
    ):
        assert getattr(cfg.validation_sampling, key) == expected[key]
    assert cfg.sampling.selection_policy == "fork"
    assert cfg.schedule.num_level_samples == 64
    assert raw.policy.generation.max_new_tokens == 256
    if dataset == "gsm8k":
        assert raw.data.train.dataset_name == "gsm8k"
        assert raw.data.validation.split == "test"
        assert raw.data.max_input_seq_length == baseline.data.max_input_seq_length
        assert raw.grpo.max_val_samples == baseline.grpo.max_val_samples
        assert raw.data.default.prompt_file == baseline.data.default.prompt_file

    invalid = OmegaConf.to_container(raw.fork_grpo, resolve=True)
    invalid["sampling"]["selection_policy"] = "low_confidence"
    with pytest.raises(ValueError, match="requires fork sampling"):
        ForkGRPOConfig.model_validate(invalid)


@pytest.mark.parametrize(
    "override", [{"block_size": 30}, {"max_steps": 16}, {"returns_entropy": True}]
)
def test_llada_baseline_validation_rejects_invalid_geometry(override: dict) -> None:
    raw = config(256, 64).model_dump()
    raw["validation_sampling"].update(
        selection_policy="low_confidence",
        block_size=32,
        max_steps=32,
        returns_reveal_steps=False,
        emit_full_blocks=False,
    )
    raw["validation_sampling"].update(override)
    with pytest.raises(ValueError, match="LLaDA baseline validation"):
        ForkGRPOConfig.model_validate(raw)
