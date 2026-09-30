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
"""Full-distribution KD from a less-masked view of a correct student rollout."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
import time

import torch

from just_grpo.algorithms.trace_grpo import (
    TraceGRPO,
    TraceGRPOSchedule,
    TraceScheduleConfig,
)
from step_distillation.config import StepDistillationConfig
from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def prepare_context_distillation_data(
    data: BatchedDataDict[Any],
    *,
    config: StepDistillationConfig,
    block_size: int,
    seeds: torch.Tensor,
) -> None:
    """Filter positive rollouts, then sample a same-block student/next-state pair.

    Only positions still masked at the teacher state and before the rollout's
    terminal boundary contribute. A completed block is never a teacher view.
    """
    data["sample_mask"] *= (data["rewards"] > config.min_reward).to(
        data["sample_mask"].dtype
    )
    levels = data["trace_reveal_levels"]
    selected = torch.full((data.size, 1), -1, dtype=torch.long, device=levels.device)
    later = torch.full((data.size,), -1, dtype=torch.long, device=levels.device)
    blocks = torch.zeros(data.size, dtype=torch.long, device=levels.device)
    target_mask = torch.zeros_like(levels, dtype=torch.bool)
    for row in range(data.size):
        if not data["sample_mask"][row]:
            continue
        candidates = []
        for start in range(0, levels.shape[1], block_size):
            local = levels[row, start : start + block_size]
            scored = data["trace_loss_mask"][row, start : start + block_size]
            events = torch.unique(local[local >= 0], sorted=True).tolist()
            for first, second in zip(events[:-1], events[1:]):
                if (scored & (local >= second)).any():
                    candidates.append((first, second, start))
        if not candidates:
            data["sample_mask"][row] = 0
            continue
        rng = torch.Generator().manual_seed(int(seeds[row]) & 0x7FFFFFFFFFFFFFFF)
        first, second, start = candidates[
            int(torch.randint(len(candidates), (1,), generator=rng))
        ]
        selected[row, 0], later[row], blocks[row] = first, second, start
        target_mask[row, start : start + block_size] = data["trace_loss_mask"][
            row, start : start + block_size
        ] & (levels[row, start : start + block_size] >= second)
    data["trace_sampled_levels"] = selected
    data["trace_loss_weights"] = (selected[:, 0] >= 0).float()
    data["kd_teacher_levels"] = later
    data["kd_block_starts"] = blocks
    data["distillation_target_mask"] = target_mask


@torch.no_grad()
def score_teacher_context(
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: BatchedDataDict[Any],
    *,
    schedule_config: TraceScheduleConfig,
    pad_token_id: int,
    padded_width: int,
    vocab_size: int,
) -> BatchedDataDict[Any]:
    """One frozen forward per positive sampled view; no teacher token sampling.

    Transfer only a single block of full-vocabulary logits per row, in BF16
    (the model output precision), rather than sequence-by-vocabulary tensors.
    Both distributions are normalized in FP32 when the loss is evaluated.
    """
    teacher_data = BatchedDataDict(data)
    teacher_data["trace_sampled_levels"] = data["kd_teacher_levels"][:, None]
    schedule = TraceGRPOSchedule(
        teacher_data,
        schedule_config,
        pad_token_id=pad_token_id,
        padded_width=padded_width,
        weighted_loss=False,
    )
    view = schedule.generate_single_trajectory(0)
    block = schedule_config.block_size
    # Non-sequence-aligned per-row payloads use lists in BatchedDataDict. Keep
    # targets on CPU until each training microbatch; filtered rows allocate none.
    targets = [
        torch.empty(0, vocab_size, dtype=torch.bfloat16) for _ in range(data.size)
    ]
    for row in range(data.size):
        if (
            not data["sample_mask"][row]
            or not data["distillation_target_mask"][row].any()
        ):
            continue
        start = int(data["kd_block_starts"][row])
        scored = data["distillation_target_mask"][row, start : start + block]
        if not view["masked_indices"][row, start : start + block][scored].all():
            raise ValueError("KD targets must remain masked in the teacher view")
        tokens = view["input_ids"][row : row + 1].masked_fill(
            view["masked_indices"][row : row + 1], schedule_config.mask_token_id
        )
        logits = forward(
            torch.cat([tokens, tokens], dim=1),
            view["position_ids"][row : row + 1].repeat(1, 2),
        )
        targets[row] = (
            logits[0, start : start + block, :vocab_size]
            .detach()
            .to(device="cpu", dtype=torch.bfloat16)
        )
        del logits
    return BatchedDataDict(
        kd_teacher_logits=targets,
        distillation_target_mask=data["distillation_target_mask"].cpu(),
    )


class LessMaskedKDSchedule(TraceGRPOSchedule):
    """Student forward at the earlier state with shared-mask target coverage."""

    def __init__(self, data: BatchedDataDict[Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(data, *args, **kwargs)
        padding = self.layout.base["input_ids"].shape[1] - data["input_ids"].shape[1]
        self.kd_mask = torch.nn.functional.pad(
            data["distillation_target_mask"], (0, padding)
        )
        self.teacher_logits = data["kd_teacher_logits"]
        self.block_starts = data["kd_block_starts"]

    def generate_single_trajectory(self, time_step: int) -> BatchedDataDict[Any]:
        result = super().generate_single_trajectory(time_step)
        if (self.kd_mask & ~result["masked_indices"]).any():
            raise ValueError("KD targets must also be masked in the student view")
        result["token_mask"] = self.kd_mask.float() * self.weights[:, None]
        result["kd_teacher_logits"] = self.teacher_logits
        result["kd_block_starts"] = self.block_starts
        return result


class ContextDistillationLoss:
    """Forward KL(teacher || student), normalized by retained masked-token count.

    Native Megatron has already temperature-scaled student logits. Teacher
    logits are raw detached BF16 outputs, so scale them exactly once here.
    """

    input_type = LossInputType.LOGIT
    loss_type = LossType.TOKEN_LEVEL

    def __init__(self, *, vocab_size: int, temperature: float) -> None:
        self.vocab_size = vocab_size
        self.temperature = temperature

    def __call__(
        self,
        data: BatchedDataDict[Any],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        *,
        logits: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        weights = data["token_mask"] * data["sample_mask"][:, None]
        rows, positions = (weights > 0).nonzero(as_tuple=True)
        local_positions = positions - data["kd_block_starts"][rows]
        student = logits[rows, positions, : self.vocab_size].float().log_softmax(-1)
        teacher_chunks = []
        for row, target_logits in enumerate(data["kd_teacher_logits"]):
            selected = local_positions[rows == row]
            if selected.numel():
                teacher_chunks.append(
                    target_logits.detach()
                    .index_select(0, selected.cpu())
                    .to(logits.device)
                )
        teacher_logits = (
            torch.cat(teacher_chunks)
            if teacher_chunks
            else logits.new_empty((0, self.vocab_size))
        )
        teacher = (teacher_logits.float() / self.temperature).log_softmax(-1)
        probabilities = teacher.exp()
        selected_weights = weights[rows, positions]
        denominator = global_valid_toks.clamp_min(1)
        per_token = (probabilities * (teacher - student)).sum(-1)
        loss = (per_token * selected_weights).sum() / denominator
        match = (teacher.argmax(-1) == student.detach().argmax(-1)).float()
        return loss, {
            "loss": loss.detach().item(),
            "distillation_forward_kl": loss.detach().item(),
            "distillation_match_fraction": (match * selected_weights).sum().item()
            / denominator.item(),
        }


if TYPE_CHECKING:
    from nemo_rl.models.policy.lm_policy import Policy


class ContextDistillation(TraceGRPO):
    """Prepare positive student states and score the frozen next-state teacher."""

    def __init__(self, *args: Any, policy: "Policy", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.policy = policy

    def prepare_training_data(
        self, data: BatchedDataDict[Any], message_logs: list[list[dict]], step: int
    ) -> None:
        super().prepare_training_data(data, message_logs, step)
        schedule = self.config.schedule
        seeds = (
            schedule.seed_base
            + step * 1_000_003
            + torch.arange(data.size, device=data["input_ids"].device)
        )
        prepare_context_distillation_data(
            data,
            config=self.config.step_distillation,
            block_size=schedule.block_size,
            seeds=seeds,
        )
        started = time.perf_counter()
        self.policy.prepare_for_lp_inference()
        shards = data.shard_by_batch_size(
            self.policy.data_parallel_size, batch_size=None
        )
        group = self.policy.worker_group
        futures = group.run_all_workers_sharded_data(
            "get_frozen_teacher_targets",
            data=shards,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            output_is_replicated=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
        )
        targets = BatchedDataDict.from_batches(group.get_all_worker_results(futures))
        data.update(targets)
        print(
            f"Frozen context teacher: {int(targets['distillation_target_mask'].sum())} targets, "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
