"""Math prompt alignment preserves verifier inputs and source data."""

import torch
from just_grpo.environments.math_data import BlockAlignedMathDataset


def test_alignment_preserves_math_metadata_and_original_prompt():
    row = {
        "message_log": [
            {"role": "user", "content": "problem", "token_ids": torch.tensor([1, 2, 3])}
        ],
        "length": 3,
        "extra_env_info": {"ground_truth": "42"},
        "loss_multiplier": 1.0,
        "idx": 0,
        "task_name": "math",
    }
    data = BlockAlignedMathDataset([row], block_size=4, mask_token_id=100)
    actual = data[0]
    assert actual["message_log"][0]["token_ids"].tolist() == [100, 1, 2, 3]
    assert actual["length"] == 4
    assert actual["extra_env_info"] == {"ground_truth": "42"}
    assert actual["loss_multiplier"] == 1
    assert row["message_log"][0]["token_ids"].tolist() == [1, 2, 3]


def test_aligned_prompt_is_not_padded_again():
    row = {
        "message_log": [{"role": "user", "token_ids": torch.tensor([1, 2, 3, 4])}],
        "length": 4,
    }
    actual = BlockAlignedMathDataset([row], block_size=4, mask_token_id=100)[0]
    assert actual["message_log"][0]["token_ids"].tolist() == [1, 2, 3, 4]
