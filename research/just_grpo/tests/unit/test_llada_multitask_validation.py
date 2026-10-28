"""Per-benchmark budgets, metrics and sweep artifacts use the same live policy."""

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from just_grpo.generation.llada_generation import validate_llada


class Tasks:
    # Raw row order deliberately differs from configured benchmark order.
    labels = ["mbpp"] * 5 + ["humaneval"] * 3
    dataset = {"task_name": labels}
    task_data_processors = {"humaneval": None, "mbpp": None}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"task": self.labels[index], "id": index}


@pytest.mark.parametrize("fail_mbpp", [False, True])
def test_benchmarks_have_independent_budgets_and_curves(
    monkeypatch, tmp_path, fail_mbpp
):
    loader = StatefulDataLoader(Tasks(), batch_size=2, collate_fn=list, num_workers=0)
    before = loader.state_dict()
    adapter = SimpleNamespace(forward_counts=[], mode=None)

    @contextmanager
    def sampling(mode=None):
        adapter.mode = mode
        adapter.forward_counts = []
        try:
            yield
        finally:
            adapter.mode = None

    adapter.validation_sampling = sampling
    modes = {
        "fixed1_t0": {"selection_policy": "low_confidence", "threshold": 0.9},
        **{
            f"threshold0{digit}_t0": {
                "selection_policy": "confidence_threshold",
                "threshold": digit / 10,
            }
            for digit in (9, 8, 7)
        },
    }
    calls = []

    def native_validate(
        policy, dataloader, tokenizer, envs, *, step, master_config, logger, processor
    ):
        rows = []
        for i, batch in enumerate(dataloader):
            if (
                i
                >= master_config.grpo.max_val_samples
                // master_config.grpo.val_batch_size
            ):
                break
            rows.extend(batch)
        tasks = {row["task"] for row in rows}
        assert len(tasks) == 1
        task = tasks.pop()
        calls.append((task, policy.mode, [row["id"] for row in rows]))
        if fail_mbpp and task == "mbpp" and policy.mode == "threshold09_t0":
            raise RuntimeError("grading failed")
        policy.forward_counts = [
            256 if policy.mode == "fixed1_t0" else (40 if task == "humaneval" else 80)
        ] * len(rows)
        logger.log_batched_dict_as_jsonl({"rewards": []}, f"val_data_step{step}.jsonl")
        return {"accuracy": 0.25 if task == "humaneval" else 0.75}, {
            "validation_time": 1.0
        }

    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.algorithms.grpo",
        SimpleNamespace(validate=native_validate),
    )
    config = SimpleNamespace(
        just_grpo={"validation_per_task": True, "validation_modes": modes},
        grpo=SimpleNamespace(max_val_samples=4, val_batch_size=2),
        policy={"generation": {"max_new_tokens": 256}},
        logger={"log_dir": str(tmp_path)},
    )
    logger = MagicMock()
    if fail_mbpp:
        with pytest.raises(RuntimeError, match="grading failed"):
            validate_llada(adapter, loader, None, {}, 7, config, logger)
        assert not (tmp_path / "mbpp/accuracy_vs_nfe_step7.png").exists()
    else:
        metrics, timings = validate_llada(adapter, loader, None, {}, 7, config, logger)
        assert metrics["accuracy"] == metrics["humaneval/accuracy"] == 0.25
        assert metrics["mbpp/accuracy"] == 0.75
        assert metrics["humaneval/num_samples"] == 3
        assert metrics["mbpp/num_samples"] == 4
        assert timings["validation_time"] == 8
        assert len(calls) == 8
        assert {call.args[2] for call in logger.log_plot.call_args_list} == {
            "validation/humaneval/accuracy_vs_nfe",
            "validation/mbpp/accuracy_vs_nfe",
        }
        assert (
            len(
                {
                    call.args[1]
                    for call in logger.log_batched_dict_as_jsonl.call_args_list
                }
            )
            == 8
        )
        for task, count in (("humaneval", 3), ("mbpp", 4)):
            assert (tmp_path / task / "accuracy_vs_nfe_step7.png").is_file()
            stats = json.loads(
                (tmp_path / task / "decode_stats_step7.json").read_text()
            )
            assert stats["metrics"]["num_samples"] == count
    for task, mode, ids in calls:
        assert len(ids) == (3 if task == "humaneval" else 4)
        assert ids == ([5, 6, 7] if task == "humaneval" else [0, 1, 2, 3])
    assert adapter.mode is None
    assert loader.state_dict() == before
