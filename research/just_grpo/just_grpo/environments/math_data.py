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
"""Block-align the native math processor's pretokenized prompts."""

from copy import deepcopy

import torch
from torch.utils.data import Dataset

from just_grpo.diffusion.block_layout import align_prompt
from nemo_rl.data.interfaces import DatumSpec


class BlockAlignedMathDataset(Dataset):
    """Preserve native math prompts, grading metadata, and dataset splits."""

    def __init__(
        self, dataset: Dataset, *, block_size: int, mask_token_id: int
    ) -> None:
        self.dataset = dataset
        self.block_size = block_size
        self.mask_token_id = mask_token_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DatumSpec:
        datum = deepcopy(self.dataset[index])
        messages = datum["message_log"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("Math diffusion requires one pretokenized user prompt")
        tokens = messages[0]["token_ids"]
        aligned = align_prompt(
            tokens.tolist(),
            block_size=self.block_size,
            mask_token_id=self.mask_token_id,
        )
        messages[0]["token_ids"] = torch.tensor(aligned, dtype=tokens.dtype)
        datum["length"] = len(aligned)
        return datum
