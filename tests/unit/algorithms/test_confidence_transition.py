import itertools
import math

import pytest
import torch

from nemo_rl.algorithms.confidence_transition import (
    block_confidence_logprobs,
    confidence_transition_logprob,
    on_policy_correction,
)


def enumerate_events(logits, threshold):
    lp = logits.log_softmax(-1)
    prob = lp.exp()
    events = {}
    for tokens in itertools.product(range(prob.shape[1]), repeat=prob.shape[0]):
        selected_p = [float(prob[i, t]) for i, t in enumerate(tokens)]
        selected = tuple(
            i for i, t in enumerate(tokens) if float(lp[i, t]) >= math.log(threshold)
        )
        if not selected:
            selected = (max(range(len(tokens)), key=lambda i: selected_p[i]),)
        key = tuple((i, tokens[i]) for i in selected)
        events[key] = events.get(key, 0) + math.prod(selected_p)
    return events


@pytest.mark.parametrize(
    "values,threshold",
    [
        ([[0.95, 0.03, 0.02], [0.92, 0.05, 0.03], [0.60, 0.30, 0.10]], 0.9),
        ([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]], 0.9),
        ([[0.9, 0.1], [0.7, 0.3]], 0.9),
        ([[0.4, 0.3, 0.3], [0.2, 0.3, 0.5]], 0.3),
    ],
)
def test_enumerated_probability_and_normalization(values, threshold):
    p = torch.tensor(values, dtype=torch.float64)
    logits = p.log().unsqueeze(0)
    total = 0
    for event, expected in enumerate_events(logits[0], threshold).items():
        target = torch.zeros((1, len(values)), dtype=torch.long)
        selected = torch.zeros_like(target, dtype=torch.bool)
        for pos, token in event:
            target[0, pos], selected[0, pos] = token, True
        lp, valid = confidence_transition_logprob(
            logits, target, torch.ones_like(selected), selected, threshold
        )
        assert valid.item()
        assert lp.exp().item() == pytest.approx(expected, abs=1e-12)
        total += lp.exp().item()
    assert total == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize(
    "target,selected",
    [([0, 0, 0], [1, 1, 0]), ([0, 0, 0], [1, 0, 0]), ([1, 1, 0], [0, 0, 1])],
)
def test_custom_backward_matches_reference_and_finite_difference(target, selected):
    # Strict margins keep support/ranking unchanged during finite differences.
    logits = (
        torch.tensor(
            [[[0.95, 0.03, 0.02], [0.92, 0.05, 0.03], [0.60, 0.30, 0.10]]],
            dtype=torch.float64,
        )
        .log()
        .requires_grad_()
    )
    target = torch.tensor([target])
    selected = torch.tensor([selected], dtype=torch.bool)
    masked = torch.ones_like(selected)
    reference, valid = confidence_transition_logprob(logits, target, masked, selected)
    assert valid.item()
    (reference_grad,) = torch.autograd.grad(reference.sum(), logits)
    custom, custom_valid = block_confidence_logprobs(
        logits, target, masked, selected, 3
    )
    (custom_grad,) = torch.autograd.grad(custom.sum(), logits)
    assert custom_valid.item()
    torch.testing.assert_close(custom_grad, reference_grad, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(
        lambda x: confidence_transition_logprob(x, target, masked, selected)[0],
        (logits,),
        atol=1e-6,
    )


def test_multiple_rows_blocks_inactive_and_zero_support():
    logits = torch.randn(2, 12, 5, dtype=torch.float64, requires_grad=True)
    target = torch.zeros((2, 8), dtype=torch.long)
    selected = torch.zeros_like(target, dtype=torch.bool)
    selected[0, 0] = True
    selected[1, 4:6] = True  # impossible multi-reveal, no p >= .9
    masked = torch.ones_like(selected)
    joint, valid = block_confidence_logprobs(logits, target, masked, selected, 4)
    (grad,) = torch.autograd.grad(joint.sum(), logits)
    assert not valid[0, 1] and not valid[1].any()
    assert not grad[1].any() and not grad[:, 8:].any() and torch.isfinite(grad).all()


def test_correction_replaces_actor_including_post_eos():
    logits = (
        torch.tensor(
            [[[0.95, 0.03, 0.02], [0.92, 0.05, 0.03], [0.60, 0.30, 0.10]]],
            dtype=torch.float64,
        )
        .log()
        .requires_grad_()
    )
    targets = torch.zeros((1, 3), dtype=torch.long)
    selected = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    token_lp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # Position 1 represents a post-EOS commit: removed by old token loss.
    loss_mask = torch.tensor([[1.0, 0.0, 0.0]])
    data = {
        "diffu_grpo_target_ids": targets,
        "block_reveal_harvest_mask": selected,
        "trace_reveal_level": torch.tensor([[0, 0, 1]]),
        "block_reveal_reveal_level": torch.tensor([0]),
        "diffu_grpo_response_lengths": torch.tensor([3]),
        "diffu_grpo_loss_mask": loss_mask,
        "advantages": torch.full((1, 3), -0.7),
        "sample_mask": torch.ones(1),
    }
    denominator = torch.tensor(2.0)
    correction, metrics = on_policy_correction(
        logits=logits,
        token_logprobs=token_lp,
        data=data,
        global_valid_toks=denominator,
        block_size=3,
    )
    old_actor = -(token_lp * loss_mask * data["advantages"]).sum() / denominator
    (actual,) = torch.autograd.grad(old_actor + correction, logits, retain_graph=True)
    q, _ = confidence_transition_logprob(
        logits, targets, torch.ones_like(selected), selected
    )
    (expected,) = torch.autograd.grad(
        -q.sum() * data["advantages"][0, 0] / denominator, logits
    )
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-7)
    assert metrics["confidence_zero_support_steps"] == 0
    assert metrics["confidence_selection_nll"] == pytest.approx(0, abs=1e-12)


def test_expected_score_is_zero():
    logits = (
        torch.tensor(
            [[0.95, 0.03, 0.02], [0.92, 0.05, 0.03], [0.60, 0.30, 0.10]],
            dtype=torch.float64,
        )
        .log()
        .requires_grad_()
    )
    expected_score = torch.zeros_like(logits)
    token_only_score = torch.zeros_like(logits)
    for event, probability in enumerate_events(logits.detach(), 0.9).items():
        targets = torch.zeros((1, 3), dtype=torch.long)
        selected = torch.zeros_like(targets, dtype=torch.bool)
        for pos, token in event:
            targets[0, pos], selected[0, pos] = token, True
        q, valid = confidence_transition_logprob(
            logits.unsqueeze(0), targets, torch.ones_like(selected), selected
        )
        (score,) = torch.autograd.grad(q.sum(), logits)
        expected_score += probability * score
        token_lp = logits.log_softmax(-1).gather(-1, targets[0, :, None]).squeeze(-1)
        (token_score,) = torch.autograd.grad((token_lp * selected[0]).sum(), logits)
        token_only_score += probability * token_score
    torch.testing.assert_close(
        expected_score, torch.zeros_like(logits), atol=1e-12, rtol=0
    )
    assert token_only_score.abs().max() > 0.01


def test_custom_multiple_chunks_matches_autograd():
    torch.manual_seed(12)
    logits = torch.randn(3, 16, 7, dtype=torch.float64).requires_grad_()
    targets = logits.argmax(-1)
    selected = torch.zeros_like(targets, dtype=torch.bool)
    # Highest candidate confidence position wins the fallback in every block.
    lp = (
        logits.detach()
        .log_softmax(-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
        .reshape(3, 4, 4)
    )
    winners = lp.argmax(-1)
    selected.reshape(3, 4, 4).scatter_(-1, winners.unsqueeze(-1), True)
    masked = torch.ones_like(selected)
    actual, valid = block_confidence_logprobs(logits, targets, masked, selected, 4)
    expected, expected_valid = confidence_transition_logprob(
        logits.reshape(12, 4, 7),
        targets.reshape(12, 4),
        masked.reshape(12, 4),
        selected.reshape(12, 4),
    )
    torch.testing.assert_close(actual.flatten(), expected)
    assert valid.all() and expected_valid.all()
    weights = torch.randn_like(actual)
    (ga,) = torch.autograd.grad((actual * weights).sum(), logits)
    (ge,) = torch.autograd.grad((expected * weights.flatten()).sum(), logits)
    torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)


def test_invisible_mask_proposal_is_marginalized():
    logits = (
        torch.tensor([[[0.95, 0.03, 0.02], [0.02, 0.03, 0.95]]], dtype=torch.float64)
        .log()
        .requires_grad_()
    )
    target = torch.zeros((1, 2), dtype=torch.long)
    selected = torch.tensor([[True, False]])
    q, valid = block_confidence_logprobs(
        logits, target, torch.ones_like(selected), selected, 2, 0.9, mask_token_id=2
    )
    assert valid.item()
    assert q.exp().item() == pytest.approx(0.95)
    (grad,) = torch.autograd.grad(q.sum(), logits)
    torch.testing.assert_close(
        grad[0, 1], torch.zeros_like(grad[0, 1]), atol=1e-12, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU integration preflight")
def test_cuda_bfloat16_full_vocab_backward():
    torch.manual_seed(42)
    logits = torch.randn(
        2, 80, 131072, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    target = logits[:, :64].argmax(-1)
    lp = (
        logits[:, :64]
        .detach()
        .float()
        .log_softmax(-1)
        .gather(-1, target.unsqueeze(-1))
        .squeeze(-1)
        .reshape(2, 4, 16)
    )
    selected = torch.zeros_like(target, dtype=torch.bool)
    selected.reshape(2, 4, 16).scatter_(-1, lp.argmax(-1).unsqueeze(-1), True)
    masked = torch.ones_like(selected)
    actual, valid = block_confidence_logprobs(logits, target, masked, selected, 16)
    expected, expected_valid = confidence_transition_logprob(
        logits[:, :64].reshape(8, 16, -1),
        target.reshape(8, 16),
        masked.reshape(8, 16),
        selected.reshape(8, 16),
    )
    assert valid.all() and expected_valid.all()
    torch.testing.assert_close(actual.flatten(), expected, atol=2e-5, rtol=2e-5)
    (ga,) = torch.autograd.grad(actual.sum(), logits)
    (ge,) = torch.autograd.grad(expected.sum(), logits)
    torch.testing.assert_close(ga, ge, atol=2e-5, rtol=0.02)
