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
"""Distillation integration with the native Megatron loss and empty-batch path."""

import pytest
import torch
from unittest.mock import Mock

pytest.importorskip("megatron.bridge")
from just_grpo.diffusion.diffusion_processors import DiffusionLossPostProcessor
from just_grpo.diffusion.megatron_diffusion_policy import (
    MegatronDiffusionPolicyWorkerImpl,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from megatron.core import parallel_state


@pytest.fixture(autouse=True)
def tensor_parallel_stubs(monkeypatch):
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: None)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: None)


def _keep_worker_data_on_cpu(monkeypatch):
    move = BatchedDataDict.to
    monkeypatch.setattr(
        BatchedDataDict,
        "to",
        lambda self, device: self if device == "cuda" else move(self, device),
    )


def test_context_distillation_processor_uses_logits_and_native_scaling(monkeypatch):
    from step_distillation.context_distillation import ContextDistillationLoss

    monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
    data = BatchedDataDict(
        target_ids=torch.tensor([[0, 2]]),
        kd_teacher_logits=[torch.randn(2, 3)],
        kd_block_starts=torch.tensor([0]),
        token_mask=torch.ones(1, 2),
        sample_mask=torch.ones(1),
    )
    loss_fn = ContextDistillationLoss(vocab_size=3, temperature=1.0)
    processor = DiffusionLossPostProcessor(
        loss_fn=loss_fn,
        cfg={"sequence_packing": {"enabled": False}},
        num_microbatches=3,
    )
    logits = torch.randn(1, 4, 3, requires_grad=True)
    loss, _ = processor(
        data, global_valid_seqs=torch.tensor(1), global_valid_toks=torch.tensor(2)
    )(logits)
    expected, _ = loss_fn(data, torch.tensor(1), torch.tensor(2), logits=logits)
    torch.testing.assert_close(loss / 3, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss / 3, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )


def test_empty_distillation_batch_skips_optimizer_and_scheduler(monkeypatch):
    from step_distillation.context_distillation import ContextDistillationLoss

    _keep_worker_data_on_cpu(monkeypatch)
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "cpu-test")
    data = BatchedDataDict(
        input_ids=torch.zeros(2, 8, dtype=torch.long),
        input_lengths=torch.full((2,), 8),
        token_mask=torch.ones(2, 8),
        sample_mask=torch.zeros(2),
    )
    worker = Mock(dtype=torch.bfloat16)
    result = MegatronDiffusionPolicyWorkerImpl.train(
        worker,
        data=data,
        loss_fn=ContextDistillationLoss(vocab_size=32, temperature=1.0),
    )
    worker.begin_train_step.assert_not_called()
    worker.finish_train_step.assert_not_called()
    worker._build_schedule.assert_not_called()
    assert result["all_mb_metrics"]["global_valid_toks"] == [0.0]
    assert result["global_loss"] == 0
