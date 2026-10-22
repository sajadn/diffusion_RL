# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the per-sample reward calls used by native asynchronous rollouts."""

import asyncio
import json
import sys
import threading

import pytest

from just_grpo.environments import code_reference


def make_environment(monkeypatch, **overrides):
    monkeypatch.setattr(
        code_reference.subprocess,
        "check_output",
        lambda command, **kwargs: "revision" if "rev-parse" in command else "",
    )
    cfg = {
        "reference_dir": "/reference",
        "reference_revision": "revision",
        "python_executable": sys.executable,
        "num_workers": 2,
        "chunk_size": 4,
        "timeout_seconds": 10,
        "batch_wait_seconds": 0.01,
        **overrides,
    }
    return code_reference.CodeReferenceEnvironmentImpl(cfg)


def request(index, kind="acecode"):
    return (
        [[{"role": "assistant", "content": str(index)}]],
        [{"ground_truth": json.dumps({"kind": kind}), "index": index}],
    )


def test_singleton_requests_batch_parallelize_and_preserve_order(monkeypatch):
    environment = make_environment(monkeypatch)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    chunks = []
    active = 0
    peak = 0

    def score(rows, cfg):
        nonlocal active, peak
        with lock:
            ordinal = len(chunks)
            chunks.append([int(row["completion"]) for row in rows])
            active += 1
            peak = max(peak, active)
        try:
            if ordinal < 2:
                barrier.wait(timeout=10)
            return [int(row["completion"]) / 24 for row in rows]
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(code_reference, "score_chunk", score)

    async def run():
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(environment.step(*request(index)) for index in range(24))
                ),
                timeout=20,
            )
            assert [result.rewards.item() for result in results] == pytest.approx(
                [index / 24 for index in range(24)]
            )
            assert [result.metadata[0]["index"] for result in results] == list(
                range(24)
            )
            assert [result.answers for result in results] == [
                [str(index)] for index in range(24)
            ]
        finally:
            await environment.shutdown()

    asyncio.run(run())
    assert sorted(map(len, chunks)) == [4] * 6
    assert sorted(index for chunk in chunks for index in chunk) == list(range(24))
    assert peak == 2


def test_partial_batches_separate_task_kinds_and_handle_empty_calls(monkeypatch):
    environment = make_environment(monkeypatch, chunk_size=16)
    chunks = []

    def score(rows, cfg):
        assert len({row["kind"] for row in rows}) == 1
        chunks.append(rows)
        return [int(row["completion"]) / 10 for row in rows]

    monkeypatch.setattr(code_reference, "score_chunk", score)

    async def run():
        first_messages, first_metadata = request(1)
        second_messages, second_metadata = request(2, kind="humaneval")
        try:
            mixed, single, empty = await asyncio.wait_for(
                asyncio.gather(
                    environment.step(
                        first_messages + second_messages,
                        first_metadata + second_metadata,
                    ),
                    environment.step(*request(3)),
                    environment.step([], []),
                ),
                timeout=10,
            )
            assert mixed.rewards.tolist() == pytest.approx([0.1, 0.2])
            assert single.rewards.tolist() == pytest.approx([0.3])
            assert empty.rewards.numel() == 0
            assert mixed.terminateds.tolist() == [True, True]
            assert mixed.metadata == first_metadata + second_metadata
        finally:
            await environment.shutdown()

    asyncio.run(run())
    assert sorted(map(len, chunks)) == [1, 2]
    assert environment.flush_handle is None
    assert not environment.scoring_tasks


def test_grader_failure_reaches_every_caller_and_next_batch_recovers(monkeypatch):
    environment = make_environment(monkeypatch)

    def score(rows, cfg):
        if any(row["completion"] == "0" for row in rows):
            raise RuntimeError("reference grader timed out")
        return [1.0] * len(rows)

    monkeypatch.setattr(code_reference, "score_chunk", score)

    async def run():
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(environment.step(*request(index)) for index in range(4)),
                    return_exceptions=True,
                ),
                timeout=10,
            )
            assert all(
                isinstance(result, RuntimeError)
                and str(result) == "reference grader timed out"
                for result in results
            )
            result = await asyncio.wait_for(environment.step(*request(5)), timeout=10)
            assert result.rewards.tolist() == [1.0]
        finally:
            await environment.shutdown()

    asyncio.run(run())


def test_shutdown_flushes_pending_partial_batch(monkeypatch):
    environment = make_environment(monkeypatch, batch_wait_seconds=60)
    monkeypatch.setattr(
        code_reference, "score_chunk", lambda rows, cfg: [1.0] * len(rows)
    )

    async def run():
        result = asyncio.create_task(environment.step(*request(1)))
        await asyncio.sleep(0)
        assert environment.flush_handle is not None
        await asyncio.wait_for(environment.shutdown(), timeout=10)
        assert (await result).rewards.tolist() == [1.0]
        assert environment.flush_handle is None
        assert not environment.scoring_tasks

    asyncio.run(run())


def test_invalid_task_does_not_block_other_pending_requests(monkeypatch):
    environment = make_environment(monkeypatch)
    monkeypatch.setattr(
        code_reference, "score_chunk", lambda rows, cfg: [1.0] * len(rows)
    )

    async def run():
        try:
            valid = asyncio.create_task(environment.step(*request(1)))
            await asyncio.sleep(0)
            with pytest.raises(ValueError, match="Unsupported"):
                await environment.step(*request(2, kind="unknown"))
            messages, metadata = request(3)
            metadata[0]["ground_truth"] = "{}"
            with pytest.raises(KeyError, match="kind"):
                await environment.step(messages, metadata)
            assert (await asyncio.wait_for(valid, timeout=10)).rewards.tolist() == [1.0]
        finally:
            await environment.shutdown()

    asyncio.run(run())
