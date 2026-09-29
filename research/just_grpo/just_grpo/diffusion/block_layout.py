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
"""Shared aligned block canvases for diffusion schedules and generation."""

from typing import Any

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def align_prompt(
    tokens: list[int], *, block_size: int, mask_token_id: int
) -> list[int]:
    """Left-pad with fixed MASK context; use these exact IDs for generation too."""
    return [mask_token_id] * (-len(tokens) % block_size) + tokens


class BlockDiffusionLayout:
    """Clean targets, response coverage, and padded noisy canvas."""

    def __init__(
        self,
        data: BatchedDataDict[Any],
        *,
        block_size: int,
        pad_token_id: int,
        padded_width: int | None = None,
    ) -> None:
        ids = data["input_ids"]
        valid = data["token_mask"].bool()
        lengths = data["input_lengths"]
        if (
            ids.ndim != 2
            or valid.shape != ids.shape
            or lengths.shape != (ids.shape[0],)
        ):
            raise ValueError("Expected matching [N,S] IDs/masks and [N] lengths")
        if ((lengths < 0) | (lengths > ids.shape[1])).any():
            raise ValueError("Invalid input lengths")
        valid = valid & (
            torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
        )
        valid = valid & data["sample_mask"].bool()[:, None]
        block = block_size
        width = max(block, (ids.shape[1] + block - 1) // block * block)
        if padded_width is not None:
            if padded_width < width or padded_width % block:
                raise ValueError("padded_width must fit the batch and align to a block")
            width = padded_width
        clean = torch.nn.functional.pad(
            ids, (0, width - ids.shape[1]), value=pad_token_id
        )
        selected = data.get("training_token_mask")
        if selected is None:
            selected = valid
        elif selected.shape != valid.shape or (selected.bool() & ~valid).any():
            raise ValueError(
                "training_token_mask must be a subset of valid response tokens"
            )
        score_mask = torch.nn.functional.pad(selected.bool(), (0, width - ids.shape[1]))
        self.response_mask = torch.nn.functional.pad(valid, (0, width - ids.shape[1]))
        canvas_mask = torch.zeros_like(clean, dtype=torch.bool)
        for row, active in enumerate(valid):
            positions = active.nonzero().flatten()
            if not positions.numel():
                continue
            start, end = int(positions[0]), int(positions[-1]) + 1
            if end - start != positions.numel():
                raise ValueError(
                    "Block diffusion currently requires one contiguous response span"
                )
            if start % block:
                raise ValueError(
                    "Prompt must be block-aligned before generation; use align_prompt"
                )
            canvas_mask[row, start : (end + block - 1) // block * block] = True
        self.original_width = ids.shape[1]
        self.canvas_mask = canvas_mask
        position_ids = torch.arange(width, device=ids.device)[None].expand_as(clean)
        self.base = BatchedDataDict(
            input_ids=clean,
            target_ids=clean,
            token_mask=score_mask,
            position_ids=position_ids,
            original_positions=position_ids.clamp_max(ids.shape[1] - 1),
            sample_indices=torch.arange(ids.shape[0], device=ids.device),
        )
