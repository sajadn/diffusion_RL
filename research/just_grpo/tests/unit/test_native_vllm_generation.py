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
"""Reference-fork contracts on top of the native vLLM generation worker."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from block_diffusion.generation.reference_vllm import (
    ReferenceVllmAsyncWorkerImpl,
    ReferenceVllmWorkerImpl,
)

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl


@pytest.mark.parametrize(
    "worker_cls", [ReferenceVllmWorkerImpl, ReferenceVllmAsyncWorkerImpl]
)
@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
@pytest.mark.parametrize("greedy", [False, True])
def test_native_sampling_keeps_reference_temperature_and_commit_logprobs(
    temperature, greedy, worker_cls
):
    worker = object.__new__(worker_cls)
    worker.cfg = dict(
        temperature=temperature,
        vllm_kwargs={"diffusion_config": {"temperature": temperature}},
        top_p=1.0,
        top_k=None,
        max_new_tokens=512,
        stop_token_ids=[11],
    )
    worker.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    params = worker._build_sampling_params(
        greedy=greedy, stop_strings=None, max_new_tokens=32
    )
    assert params.temperature == (0.0 if greedy else 1.0)
    assert params.logprobs == 0
    assert params.max_tokens == 32
    assert params.stop_token_ids == [11]
    assert params.top_p == 1.0
    assert params.top_k == -1


@pytest.mark.parametrize("leftmost", [False, True])
def test_engine_requires_leftmost_support_and_reuses_native_creation(
    monkeypatch, leftmost
):
    for name in ("vllm", "vllm.config", "vllm.config.diffusion"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm.config.diffusion"].DiffusionConfig = SimpleNamespace(
        __annotations__={"selection_policy": "leftmost" if leftmost else "confidence"}
    )
    create = Mock()
    monkeypatch.setattr(VllmGenerationWorkerImpl, "_create_engine", create)
    worker = object.__new__(ReferenceVllmWorkerImpl)
    kwargs = {"diffusion_config": {"temperature": 0.7}, "enable_sleep_mode": True}
    if leftmost:
        worker._create_engine(kwargs)
        create.assert_called_once_with(kwargs)
    else:
        with pytest.raises(RuntimeError, match="leftmost"):
            worker._create_engine(kwargs)
        create.assert_not_called()


@pytest.mark.parametrize("max_model_len", [47, 48])
def test_context_reserves_diffusion_canvas_and_delegates_output_packing(
    monkeypatch, max_model_len
):
    worker = object.__new__(ReferenceVllmWorkerImpl)
    worker.cfg = dict(
        max_new_tokens=16,
        vllm_cfg={"max_model_len": max_model_len},
        vllm_kwargs={"diffusion_config": {"canvas_length": 16}},
    )
    batch = BatchedDataDict(
        input_ids=torch.ones((1, 16), dtype=torch.long),
        input_lengths=torch.tensor([16]),
    )
    worker.llm = Mock()
    worker.llm.collective_rpc.return_value = [{"nfe": 16, "committed_tokens": 16}]
    generate = Mock(return_value=object())
    monkeypatch.setattr(VllmGenerationWorkerImpl, "generate", generate)
    if max_model_len == 47:
        with pytest.raises(ValueError, match="context length"):
            worker.generate(batch)
        generate.assert_not_called()
    else:
        assert worker.generate(batch) is generate.return_value
        generate.assert_called_once_with(batch, greedy=False)


@pytest.mark.parametrize("max_model_len", [47, 48])
def test_async_context_check_and_native_stream_delegation(monkeypatch, max_model_len):
    import asyncio

    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = object.__new__(ReferenceVllmAsyncWorkerImpl)
    worker.cfg = dict(
        max_new_tokens=16,
        vllm_cfg={"max_model_len": max_model_len},
        vllm_kwargs={"diffusion_config": {"canvas_length": 16}},
    )
    batch = BatchedDataDict(
        input_ids=torch.ones((1, 16), dtype=torch.long),
        input_lengths=torch.tensor([16]),
    )
    called = []

    async def generate(self, data, greedy=False):
        called.append((data, greedy))
        yield 0, data

    monkeypatch.setattr(VllmAsyncGenerationWorkerImpl, "generate_async", generate)

    async def collect():
        return [item async for item in worker.generate_async(batch, greedy=True)]

    if max_model_len == 47:
        with pytest.raises(ValueError, match="context length"):
            asyncio.run(collect())
        assert not called
    else:
        assert asyncio.run(collect())[0][1] is batch
        assert called == [(batch, True)]


@pytest.mark.parametrize(
    "worker_cls", [ReferenceVllmWorkerImpl, ReferenceVllmAsyncWorkerImpl]
)
def test_reference_trace_metadata_retains_full_canvas_and_text_stop(worker_cls):
    worker = object.__new__(worker_cls)
    worker.cfg = {
        "vllm_kwargs": {"diffusion_config": {"return_reveal_steps": True}},
        "stop_token_ids": [11],
    }
    completion = SimpleNamespace(token_ids=[7, 11, 9, 100], reveal_steps=[1, 2, 0, -1])
    result = worker._completion_metadata(completion, input_length=4, padded_length=10)
    assert result["reveal_steps"].tolist() == [-1, -1, -1, -1, 1, 2, 0, -1, -1, -1]
    assert int(result["response_lengths"]) == 2
    completion.reveal_steps = None
    with pytest.raises(ValueError, match="reveal step"):
        worker._completion_metadata(completion, input_length=4, padded_length=10)


@pytest.mark.parametrize(
    "worker_cls", [ReferenceVllmWorkerImpl, ReferenceVllmAsyncWorkerImpl]
)
@pytest.mark.parametrize("flags", [[False] * 4, [True, False, True, False]])
def test_reference_fork_metadata_preserves_false_flags_and_terminal_canvas(
    worker_cls, flags
):
    worker = object.__new__(worker_cls)
    worker.cfg = {
        "vllm_kwargs": {
            "diffusion_config": {
                "return_reveal_steps": True,
                "selection_policy": "confidence_threshold_leftmost",
            }
        },
        "stop_token_ids": [11],
    }
    completion = SimpleNamespace(
        token_ids=[7, 11, 9, 8], reveal_steps=[1, 2, 0, 0], fallback_mask=flags
    )
    result = worker._completion_metadata(completion, input_length=2, padded_length=8)
    assert result["fallback_mask"].dtype == torch.bool
    assert result["fallback_mask"].tolist() == [False, False, *flags, False, False]
    assert int(result["response_lengths"]) == 2
    for invalid in (None, [], flags[:-1]):
        completion.fallback_mask = invalid
        with pytest.raises(ValueError, match="fallback flag"):
            worker._completion_metadata(completion, input_length=2, padded_length=8)


def test_sync_diffusion_metrics_survive_engine_sleep(monkeypatch):
    worker = object.__new__(ReferenceVllmWorkerImpl)
    worker.cfg = dict(
        max_new_tokens=16,
        vllm_cfg={"max_model_len": 48},
        vllm_kwargs={"diffusion_config": {"canvas_length": 16}},
    )
    worker.llm = Mock()
    worker.llm.collective_rpc.return_value = [{"nfe": 16, "committed_tokens": 16}]
    batch = BatchedDataDict(
        input_ids=torch.ones((1, 16), dtype=torch.long),
        input_lengths=torch.tensor([16]),
    )
    monkeypatch.setattr(VllmGenerationWorkerImpl, "generate", Mock(return_value=batch))
    monkeypatch.setattr(VllmGenerationWorkerImpl, "_get_raw_spec_counters", lambda _: {})
    worker.generate(batch)
    worker.generate(batch)
    # Metrics are polled after sleep; device-side counters are inaccessible then.
    worker.llm.collective_rpc.side_effect = RuntimeError("engine is asleep")
    assert worker._get_raw_spec_counters() == {
        "vllm:diffusion_num_denoising_steps": 32.0,
        "vllm:diffusion_num_committed_tokens": 32.0,
    }
