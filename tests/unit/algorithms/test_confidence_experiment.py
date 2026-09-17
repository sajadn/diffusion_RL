import sys
from pathlib import Path
import torch

repo = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(repo / "tools/nemotron_diffusion"), str(repo)]
from confidence_experiment import (
    response_weights,
    start_round,
    record_transitions,
    replay_fields,
)
from nemo_rl.algorithms.confidence_transition import (
    on_policy_correction,
    _transition_parts,
)


def test_weight_replaces_whole_actor_gradient():
    for weight in [0.0, 0.2, 1.0, 2.0]:
        logits = torch.tensor(
            [[[3.0, 0.0, -2.0], [0.0, 3.0, -2.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        targets = torch.tensor([[0, 1]])
        selected = torch.ones((1, 2), dtype=torch.bool)
        lp = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
        data = {
            "diffu_grpo_target_ids": targets,
            "block_reveal_harvest_mask": selected,
            "trace_reveal_level": torch.zeros_like(targets),
            "block_reveal_reveal_level": torch.zeros(1, dtype=torch.long),
            "diffu_grpo_response_lengths": torch.tensor([2]),
            "diffu_grpo_loss_mask": selected.double(),
            "advantages": torch.tensor([[0.7, 0.7]]),
            "sample_mask": torch.ones(1),
            "confidence_actor_weight": torch.tensor([weight]),
        }
        correction, _ = on_policy_correction(
            logits=logits,
            token_logprobs=lp,
            data=data,
            global_valid_toks=torch.tensor(2.0),
            block_size=2,
            mask_token_id=None,
        )
        actual = torch.autograd.grad(
            -0.7 * lp.sum() / 2 + correction, logits, retain_graph=True
        )[0]
        joint, *_ = _transition_parts(logits, targets, selected, selected, 0.9, None)
        expected = torch.autograd.grad(-0.7 * weight * joint.sum() / 2, logits)[0]
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)


def test_trajectory_weights_and_filter():
    logratio = torch.tensor([0.0, -4.64154, 10.0, 0.0])
    errors = torch.tensor([0.0, 4.64154, 0.1, 0.0])
    unsupported = torch.tensor([0, 0, 0, 1])
    torch.testing.assert_close(
        response_weights(logratio, errors, unsupported, mode="filter"),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    torch.testing.assert_close(
        response_weights(logratio, errors, unsupported, mode="importance"),
        torch.tensor([1.0, 0.00964285, 2.0, 0.0]),
        atol=1e-6,
        rtol=1e-5,
    )


def test_generation_replay_records_include_rejections_and_support(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NRL_CONFIDENCE_EXPERIMENT_DIR", str(tmp_path))
    start_round(0)
    logits = torch.full((1, 16, 128), -30.0)
    logits[:, :, 0] = 0.0
    logits[0, 0, 0] = 3.0
    logits[0, 0, 1] = 0.0
    logits[0, 1, 0] = 0.0
    logits[0, 1, 1] = 3.0
    before = torch.zeros((1, 16), dtype=torch.long)
    before[:, :2] = 100
    after = before.clone()
    after[0, 0] = 0
    after[0, 1] = 1
    record_transitions(logits, before, after, [[3, 4]])
    ids = torch.cat([before, torch.tensor([[3, 4]]), after], -1)
    target = after.clone()
    target[target == 100] = 0
    selected = before != after
    glp = logits.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
    data = {
        "diffu_grpo_noisy_lengths": torch.tensor([16]),
        "block_reveal_harvest_mask": selected,
        "diffu_grpo_target_ids": target,
        "input_ids": ids,
        "diffu_grpo_completion_starts": torch.tensor([2]),
        "generation_logprobs": glp,
    }
    result = replay_fields(logits, data)
    assert (
        result["confidence_count"].sum() == 1
        and result["confidence_unsupported"].sum() == 0
    )
    assert result["confidence_logratio"].abs().max() < 1e-6
    bad = logits.clone()
    bad[0, 1, 1] = 1.0
    result = replay_fields(bad, data)
    assert result["confidence_unsupported"].sum() == 1
    # A new round must not accidentally reuse probabilities from old weights.
    start_round(1)
    try:
        replay_fields(logits, data)
    except RuntimeError as error:
        assert "Missing generation transition" in str(error)
    else:
        raise AssertionError("stale generation record was used")


def test_retention_keeps_original_advantages(tmp_path, monkeypatch):
    from confidence_experiment import log_retained_advantages

    monkeypatch.setenv("NRL_CONFIDENCE_EXPERIMENT_DIR", str(tmp_path))
    start_round(0)
    data = {
        "token_mask": torch.ones(4, 2),
        "advantages": torch.tensor(
            [[1.0, 1.0], [-1.0, -1.0], [0.5, 0.5], [-0.5, -0.5]]
        ),
        "confidence_actor_weight": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        "sample_mask": torch.ones(4),
        "confidence_logratio": torch.zeros(4),
        "confidence_unsupported": torch.tensor([0, 1, 0, 0]),
        "confidence_ambiguous": torch.zeros(4),
    }
    before = data["advantages"].clone()
    stats = log_retained_advantages(
        data, [0, 0, 1, 1], torch.tensor([1.0, 0.0, 1.0, 0.0])
    )
    assert stats["positive_retained"] == 2 and stats["negative_retained"] == 1
    assert stats["groups_with_both_retained_advantage_signs"] == 1
    torch.testing.assert_close(data["advantages"], before)
