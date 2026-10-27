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
"""Reference diffusion sampling on NeMo-RL's native vLLM worker."""

from collections.abc import AsyncIterator
from typing import Any

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.vllm.collective_rpc import resolve_collective_rpc_result
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


def _read_diffusion_counters(worker: Any, reset: bool = False) -> dict[str, int]:
    """Read sampler counters, optionally resetting the current generation window."""
    # The diffusion fork is only installed in generation workers.
    from vllm.model_executor.models.nemotron_dllm import get_diffusion_tpf_stats

    return get_diffusion_tpf_stats(reset=reset)


def _diffusion_counters(reports: list[dict[str, int]]) -> dict[str, float]:
    """Count one copy per TP/PP engine; the generation group sums DP engines."""
    # TP ranks replicate sampler counts; non-sampling PP ranks return zeros.
    report = max(reports, key=lambda item: item["nfe"])
    return {
        "vllm:diffusion_num_denoising_steps": float(report["nfe"]),
        "vllm:diffusion_num_committed_tokens": float(report["committed_tokens"]),
    }


class _ReferenceDiffusionSampling:
    """Keep native lifecycle/refit; specialize the reference fork's diffusion API."""

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        if llm_kwargs.get("diffusion_config") is None:
            return super()._create_engine(llm_kwargs)
        from vllm.config.diffusion import DiffusionConfig

        if "leftmost" not in str(
            DiffusionConfig.__annotations__.get("selection_policy")
        ):
            raise RuntimeError(
                "Reference vLLM must support leftmost diffusion sampling"
            )
        if (
            llm_kwargs["diffusion_config"].get("selection_policy")
            == "confidence_threshold_leftmost"
        ):
            from vllm.outputs import CompletionOutput

            if "fallback_mask" not in CompletionOutput.__annotations__:
                raise RuntimeError("Reference vLLM must return fallback_mask metadata")
        super()._create_engine(llm_kwargs)

    def _build_sampling_params(
        self,
        *,
        greedy: bool,
        stop_strings: list[str] | None,
        max_new_tokens: int | None = None,
    ) -> Any:
        params = super()._build_sampling_params(
            greedy=greedy, stop_strings=stop_strings, max_new_tokens=max_new_tokens
        )
        if self.cfg["vllm_kwargs"]["diffusion_config"] is None:
            return params
        # The reference fork uses 1 as a sentinel for engine-level temperature.
        # logprobs=0 returns sampled-token scores at commitment; top-k scores
        # instead describe the final canvas and cannot be used for GRPO.
        params.temperature = 0.0 if greedy else 1.0
        params.logprobs = 0
        params.top_k = -1
        return params

    def _completion_metadata(
        self,
        completion: Any,
        *,
        input_length: int,
        padded_length: int,
    ) -> dict[str, torch.Tensor]:
        diffusion = self.cfg["vllm_kwargs"]["diffusion_config"]
        if diffusion is None or not diffusion["return_reveal_steps"]:
            return {}
        recorded = completion.reveal_steps
        if recorded is None or len(recorded) != len(completion.token_ids):
            raise ValueError(
                "Reference vLLM must return a reveal step per generated token"
            )
        steps = torch.full((padded_length,), -1, dtype=torch.long)
        steps[input_length : input_length + len(recorded)] = torch.tensor(recorded)
        tokens = list(completion.token_ids)
        stop_ids = self.cfg["stop_token_ids"] or []
        response_length = next(
            (i + 1 for i, token in enumerate(tokens) if token in stop_ids), len(tokens)
        )
        metadata = {
            "reveal_steps": steps,
            "response_lengths": torch.tensor(response_length),
        }
        if diffusion.get("selection_policy") == "confidence_threshold_leftmost":
            flags = completion.fallback_mask
            if flags is None or len(flags) != len(tokens):
                raise ValueError(
                    "Reference vLLM must return a fallback flag per generated token"
                )
            fallback = torch.zeros(padded_length, dtype=torch.bool)
            fallback[input_length : input_length + len(flags)] = torch.tensor(
                flags, dtype=torch.bool
            )
            metadata["fallback_mask"] = fallback
        return metadata

    def _validate_context(self, data: BatchedDataDict[GenerationDatumSpec]) -> None:
        diffusion = self.cfg["vllm_kwargs"]["diffusion_config"]
        if len(data["input_lengths"]) and (
            int(data["input_lengths"].max())
            + self.cfg["max_new_tokens"]
            + (diffusion["canvas_length"] if diffusion is not None else 0)
            > self.cfg["vllm_cfg"]["max_model_len"]
        ):
            raise ValueError(
                "Prompt plus response canvas exceeds generation context length"
            )


class ReferenceVllmWorkerImpl(_ReferenceDiffusionSampling, VllmGenerationWorkerImpl):
    # The sampler counters live in device memory that sleep discards, and step
    # metrics are read after the engine sleeps: accumulate on the host instead.
    _diffusion_totals: dict[str, float] | None = None

    def _get_raw_spec_counters(self) -> dict[str, float | list[float]]:
        metrics = super()._get_raw_spec_counters()
        if self.cfg["vllm_kwargs"]["diffusion_config"] is not None:
            metrics.update(
                self._diffusion_totals
                or _diffusion_counters([{"nfe": 0, "committed_tokens": 0}])
            )
        return metrics

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        self._validate_context(data)
        if self.cfg["vllm_kwargs"]["diffusion_config"] is None:
            return super().generate(data, greedy=greedy)
        # Wake-up may remap the counters to unspecified contents: start from zero.
        self.llm.collective_rpc(_read_diffusion_counters, args=(True,))
        output = super().generate(data, greedy=greedy)
        window = _diffusion_counters(
            self.llm.collective_rpc(_read_diffusion_counters, args=(True,))
        )
        totals = self._diffusion_totals or dict.fromkeys(window, 0.0)
        self._diffusion_totals = {k: totals[k] + v for k, v in window.items()}
        return output


class ReferenceVllmAsyncWorkerImpl(
    _ReferenceDiffusionSampling, VllmAsyncGenerationWorkerImpl
):
    async def _get_raw_spec_counters(self) -> dict[str, float | list[float]]:
        metrics = super()._get_raw_spec_counters()
        if self.cfg["vllm_kwargs"]["diffusion_config"] is not None:
            reports = await resolve_collective_rpc_result(
                self.llm.collective_rpc(_read_diffusion_counters)
            )
            metrics.update(_diffusion_counters(reports))
        return metrics

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncIterator[tuple[int, BatchedDataDict[GenerationOutputSpec]]]:
        self._validate_context(data)
        async for result in super().generate_async(data, greedy=greedy):
            yield result


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)
class ReferenceVllmWorker(ReferenceVllmWorkerImpl):
    pass


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)
class ReferenceVllmAsyncWorker(ReferenceVllmAsyncWorkerImpl):
    pass
