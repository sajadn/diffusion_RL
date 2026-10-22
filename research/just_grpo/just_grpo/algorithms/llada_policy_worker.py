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
"""Exact left-to-right JustGRPO on LLaDA's fully bidirectional canvas."""

from typing import Any

import ray
import torch
from block_diffusion.config import DiffusionExperimentConfig, DiffusionSamplingParams
from block_diffusion.denoising_schedule import SchedulePurpose
from block_diffusion.full_canvas import FullCanvasLayout, FullCanvasMicrobatchProcessor
from block_diffusion.megatron_diffusion_policy import MegatronDiffusionPolicyWorkerImpl
from block_diffusion.positive_diffusion import build_positive_diffusion_level
from megatron.core import parallel_state
from omegaconf import OmegaConf

from just_grpo.algorithms.llada_schedule import LLaDAJustGRPOSchedule
from just_grpo.config import JustGRPOConfig
from just_grpo.generation.llada import generate_llada
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.train import model_forward
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker


class LLaDAJustGRPOPolicyWorkerImpl(MegatronDiffusionPolicyWorkerImpl):
    """Reuse native streaming accumulation and the existing positive auxiliary loss."""

    def __init__(self, config: Any, tokenizer: Any, **kwargs: Any) -> None:
        research = OmegaConf.create(config.pop("diffusion_config"))
        self.diffusion = self._load_diffusion_config(research)
        self.validation_mode = False
        self.validation_mode_name: str | None = None
        super().__init__(
            config,
            tokenizer,
            mask_token_id=self.diffusion.schedule.mask_token_id,
            block_size=self.diffusion.schedule.block_size,
            sampling=self.diffusion.sampling,
            generation_seed=research.grpo.seed,
            max_generation_kl=self.diffusion.max_generation_kl,
            **kwargs,
        )

    def _load_diffusion_config(self, research: Any) -> DiffusionExperimentConfig:
        return JustGRPOConfig.model_validate(
            OmegaConf.to_container(research.just_grpo, resolve=True)
        )

    def _generation_block_length(self, sampling: DiffusionSamplingParams) -> int:
        return sampling.block_size if self.validation_mode else 1

    def _configure_model(self, hf_config: Any, config: Any, block_size: int) -> None:
        # Register this optional architecture before native Bridge model construction.
        from just_grpo.algorithms import llada_bridge  # noqa: F401

        if hf_config.model_type != "llada":
            raise ValueError("LLaDA worker requires an original LLaDA checkpoint")
        if config["max_total_sequence_length"] > hf_config.max_sequence_length:
            raise ValueError("Full canvas exceeds LLaDA's position window")
        config["megatron_cfg"]["model_overrides"] = {
            "seq_length": config["max_total_sequence_length"]
        }

    def _create_microbatch_processor(self) -> FullCanvasMicrobatchProcessor:
        return FullCanvasMicrobatchProcessor(
            model=self.model, mask_token_id=self.mask_token_id
        )

    def _build_schedule(
        self, data: BatchedDataDict[Any], *, purpose: SchedulePurpose
    ) -> LLaDAJustGRPOSchedule:
        return LLaDAJustGRPOSchedule(
            data, canvas_length=self.diffusion.schedule.block_size
        )

    def _build_positive_diffusion_level(
        self, data: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], torch.Tensor]:
        return build_positive_diffusion_level(
            data,
            self.positive_diffusion,
            pad_token_id=self.tokenizer.pad_token_id,
            padded_width=self.cfg["max_total_sequence_length"],
            stop_token_ids=self.cfg["generation"]["stop_token_ids"],
            generator=self.positive_diffusion_generator,
            layout=FullCanvasLayout(
                data, canvas_length=self.diffusion.schedule.block_size
            ),
        )

    def set_validation_sampling(self, enabled: bool, mode: str | None = None) -> None:
        """Select the confidence decoder between synchronous generation batches."""
        if mode is not None and mode not in self.diffusion.validation_modes:
            raise ValueError(f"Unknown validation mode: {mode}")
        self.validation_mode = enabled
        self.validation_mode_name = mode if enabled else None

    @torch.no_grad()
    def generate(
        self, *, data: BatchedDataDict[Any], greedy: bool = False
    ) -> BatchedDataDict[Any]:
        """Sample from the live training model without a second model or refit."""
        if self.cfg["generation"]["backend"] != "megatron":
            raise ValueError("Shared LLaDA generation requires the Megatron backend")
        if parallel_state.get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Shared LLaDA generation currently requires TP=1")
        sampling = (
            (
                self.diffusion.validation_modes[self.validation_mode_name]
                if self.validation_mode_name is not None
                else self.diffusion.validation_sampling
            )
            if self.validation_mode
            else self.diffusion.sampling
        )
        block = self._generation_block_length(sampling)
        device = torch.device("cuda", torch.cuda.current_device())
        lengths = data["input_lengths"].to(device)
        inputs = data["input_ids"].to(device)
        completion = self.cfg["generation"]["max_new_tokens"]
        width = int(lengths.max()) + completion if lengths.numel() else 0
        if width > self.cfg["max_total_sequence_length"]:
            raise ValueError("Prompt plus LLaDA canvas exceeds the model context")
        output = torch.full(
            (len(lengths), width),
            self.tokenizer.pad_token_id,
            dtype=inputs.dtype,
            device=device,
        )
        logprobs = torch.zeros(output.shape, dtype=torch.float32, device=device)
        forward_counts = torch.zeros(len(lengths), dtype=torch.long, device=device)
        reveal_steps = torch.full_like(output, -1)
        fallback_mask = torch.zeros_like(output, dtype=torch.bool)
        rng = torch.Generator(device=device).manual_seed(
            self.generation_seed
            + parallel_state.get_data_parallel_rank()
            + self.generation_index * parallel_state.get_data_parallel_world_size()
        )

        def forward(ids: torch.Tensor) -> torch.Tensor:
            positions = torch.arange(ids.shape[1], device=device)[None].expand_as(ids)
            logits = model_forward(
                self.model,
                BatchedDataDict(),
                ids,
                positions,
                attention_mask=None,
                defer_fp32_logits=True,
            )
            return logits[..., : self.generation_vocab_size]

        # Replay may have left a per-row padding mask. Generation batches have
        # equal prompt lengths, so every position in their full canvas is valid.
        self.prepare_microbatch_fn.clear_asymmetric_metadata()
        try:
            for length in lengths.unique().tolist():
                indices = torch.nonzero(lengths == length).flatten()
                for chunk in indices.split(self.diffusion.generation_batch_size):
                    sampled = generate_llada(
                        forward,
                        inputs[chunk, :length],
                        completion_length=completion,
                        block_length=block,
                        mask_token_id=self.mask_token_id,
                        temperature=0.0 if greedy else sampling.temperature,
                        generator=rng,
                        tokens_per_step=sampling.block_size // sampling.max_steps,
                        threshold=(
                            sampling.threshold
                            if sampling.selection_policy
                            in ("confidence_threshold", "fork")
                            else None
                        ),
                        leftmost_fallback=sampling.selection_policy == "fork",
                    )
                    width = sampled.token_ids.shape[1]
                    forward_counts[chunk] = sampled.num_forward_passes
                    output[chunk, :width] = sampled.token_ids
                    logprobs[chunk, :width] = sampled.logprobs
                    reveal_steps[chunk, :width] = sampled.reveal_steps
                    fallback_mask[chunk, :width] = sampled.fallback_mask
        finally:
            self.prepare_microbatch_fn.clear_asymmetric_metadata()
        self.generation_index += 1
        result = BatchedDataDict(
            output_ids=output.cpu(),
            num_forward_passes=forward_counts.cpu(),
            logprobs=logprobs.cpu(),
            generation_lengths=torch.full_like(lengths, completion).cpu(),
            unpadded_sequence_lengths=(lengths + completion).cpu(),
            truncated=torch.zeros(len(lengths), dtype=torch.bool),
        )
        if sampling.returns_reveal_steps:
            result["reveal_steps"] = reveal_steps.cpu()
            result["fallback_mask"] = fallback_mask.cpu()
        return result


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class LLaDAJustGRPOPolicyWorker(LLaDAJustGRPOPolicyWorkerImpl):  # pragma: no cover
    pass
