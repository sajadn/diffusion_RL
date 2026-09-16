"""Regression tests for lost rollout failures and indefinite refit drains."""

import threading
from unittest.mock import Mock, patch

import pytest

from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector, ReplayBuffer


def collector():
    cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    instance = cls.__new__(cls)
    instance.master_config = {
        "grpo": {"async_grpo": {"drain_no_progress_timeout_s": 2}}
    }
    instance._worker_error = None
    instance._threads_lock = threading.Lock()
    instance._pending_groups = {1: {"prompt_idx": 7, "target_weight_version": 30}}
    return instance


def test_failed_group_is_not_silently_lost():
    cls = ReplayBuffer.__ray_metadata__.modified_class
    buffer = cls(max_size=4)
    buffer.report_worker_error("prompt 7 failed")
    with pytest.raises(RuntimeError, match="prompt 7 failed"):
        buffer.size()
    with pytest.raises(RuntimeError, match="prompt 7 failed"):
        buffer.sample(num_prompt_groups=1, current_weight_version=0, max_age_steps=1)


def test_drain_timeout_reports_pending_group():
    instance = collector()
    instance._inflight_threads = {Mock(is_alive=Mock(return_value=True))}
    with (
        patch("nemo_rl.algorithms.async_utils.time.monotonic", side_effect=[0, 0, 3]),
        patch("nemo_rl.algorithms.async_utils.time.sleep"),
        patch("faulthandler.dump_traceback") as dump,
    ):
        with pytest.raises(TimeoutError, match="prompt_idx.*7"):
            instance.wait_for_pending_generations()
        dump.assert_called_once()


def test_completed_group_allows_refit():
    instance = collector()
    instance._inflight_threads = {Mock(is_alive=Mock(return_value=False))}
    instance.wait_for_pending_generations()
    assert not instance._inflight_threads


def test_drain_surfaces_worker_error():
    instance = collector()
    instance._worker_error = "tool HTTP request failed"
    instance._inflight_threads = set()
    with pytest.raises(RuntimeError, match="tool HTTP request failed"):
        instance.wait_for_pending_generations()


def test_last_worker_failure_cannot_look_like_successful_drain():
    instance = collector()
    def failed_worker():
        instance._worker_error = "last group failed"
        return False
    instance._inflight_threads = {Mock(is_alive=Mock(side_effect=failed_worker))}
    with pytest.raises(RuntimeError, match="last group failed"):
        instance.wait_for_pending_generations()


def test_drain_timeout_resets_when_a_group_completes():
    instance = collector()
    instance._inflight_threads = {
        Mock(is_alive=Mock(side_effect=[True, False])),
        Mock(is_alive=Mock(side_effect=[True, True, False])),
    }
    with (
        patch("nemo_rl.algorithms.async_utils.time.monotonic", side_effect=[0, 0, 3]),
        patch("nemo_rl.algorithms.async_utils.time.sleep"),
    ):
        instance.wait_for_pending_generations()
    assert not instance._inflight_threads
