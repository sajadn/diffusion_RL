# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""A complete first-step reveal is replayable; absent metadata still fails."""

import pytest
import torch

from nemo_rl.algorithms.trace_grpo_logprobs import (
    build_trace_base,
    make_trace_level_view,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _data():
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 11]]),
            "input_lengths": torch.tensor([5]),
            "token_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]]),
            "sample_mask": torch.ones(1),
            "reveal_steps": torch.zeros(1, 5, dtype=torch.long),
        }
    )


def test_all_tokens_revealed_at_step_zero_are_scored_from_masked_state():
    base, samples, levels = build_trace_base(
        _data(), mask_token_id=100, pad_token_id=11
    )
    assert samples == 1
    assert levels == 1
    view = make_trace_level_view(base, 0, ("diffu_grpo_score_mask",))
    scored = view["diffu_grpo_score_mask"].bool()
    assert scored.sum().item() == 3
    assert torch.all(view["input_ids"][scored] == 100)
    assert base["trace_reveal_level"][0, :3].tolist() == [0, 0, 0]


def test_missing_reveal_metadata_is_rejected():
    data = _data()
    del data["reveal_steps"]
    with pytest.raises(RuntimeError, match="reveal_steps is absent"):
        build_trace_base(data, mask_token_id=100, pad_token_id=11)
