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
"""AceCode-Hard reward and HumanEval grading through pinned JustGRPO code."""

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import ray
import torch
from pydantic import BaseModel, Field

from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec, TokenizerType
from nemo_rl.data.processors import math_hf_data_processor, register_processor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.utils import register_env


class CodeReferenceConfig(BaseModel):
    reference_dir: str
    reference_revision: str
    python_executable: str
    num_workers: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)


def code_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Use the native chat template and reject, rather than truncate, long prompts."""
    result = math_hf_data_processor(
        datum_dict, task_data_spec, tokenizer, max_seq_length, idx
    )
    if result["loss_multiplier"] != 1.0:
        raise ValueError(f"Code prompt {idx} exceeds max_input_seq_length")
    return result


def score_chunk(rows: list[dict[str, Any]], cfg: CodeReferenceConfig) -> list[float]:
    """Run the reference in an isolated working directory and process group."""
    worker = Path(__file__).with_name("code_reference_worker.py")
    with tempfile.TemporaryDirectory(prefix="justgrpo_code_") as cwd:
        with subprocess.Popen(
            [cfg.python_executable, str(worker), cfg.reference_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""},
        ) as process:
            try:
                output, error = process.communicate(
                    json.dumps(rows), timeout=cfg.timeout_seconds
                )
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise RuntimeError(
                    "JustGRPO reference grader exceeded its batch timeout"
                )
            if process.returncode:
                raise RuntimeError(f"JustGRPO reference grader failed: {error[-4000:]}")
    scores = json.loads(output)
    if len(scores) != len(rows) or any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= (1 if row["kind"] == "humaneval" else 2)
        for value, row in zip(scores, rows, strict=True)
    ):
        raise ValueError("Invalid scores from JustGRPO reference grader")
    return scores


class CodeReferenceEnvironmentImpl(EnvironmentInterface):
    """Keep training's shaped reward separate from held-out binary pass@1."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = CodeReferenceConfig.model_validate(cfg)
        revision = subprocess.check_output(
            ["git", "-C", self.cfg.reference_dir, "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != self.cfg.reference_revision:
            raise ValueError(f"Reference revision mismatch: {revision}")
        changes = subprocess.check_output(
            [
                "git",
                "-C",
                self.cfg.reference_dir,
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
        )
        if changes:
            raise ValueError("Reference source has tracked modifications")
        self.pool = ThreadPoolExecutor(max_workers=self.cfg.num_workers)

    def step(
        self,
        message_log_batch: list,
        metadata: list[dict[str, Any]],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn:
        rows = [
            {**json.loads(info["ground_truth"]), "completion": messages[-1]["content"]}
            for messages, info in zip(message_log_batch, metadata, strict=True)
        ]
        chunks = [
            rows[i : i + self.cfg.chunk_size]
            for i in range(0, len(rows), self.cfg.chunk_size)
        ]
        futures = [self.pool.submit(score_chunk, chunk, self.cfg) for chunk in chunks]
        scores = [score for future in futures for score in future.result()]
        return EnvironmentReturn(
            observations=[{"role": "user", "content": ""} for _ in rows],
            metadata=metadata,
            next_stop_strings=[None] * len(rows),
            rewards=torch.tensor(scores, dtype=torch.float32),
            terminateds=torch.ones(len(rows), dtype=torch.bool),
            answers=[row["completion"] for row in rows],
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        return batch, {}

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)


CodeReferenceEnvironment = ray.remote(CodeReferenceEnvironmentImpl)


def register_code_reference() -> None:
    """Register the optional research adapter with NeMo-RL's native factories."""
    fqn = "just_grpo.environments.code_reference.CodeReferenceEnvironment"
    register_processor("justgrpo_code", code_data_processor)
    register_env("justgrpo_code", fqn)
    ACTOR_ENVIRONMENT_REGISTRY[fqn] = sys.executable
