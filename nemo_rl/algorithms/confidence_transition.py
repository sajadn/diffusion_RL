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
"""Marginal likelihood of sampled-token confidence threshold transitions.

T=1, full vocabulary, independent proposals, reveal all p >= threshold, and
fallback to the largest sampled p (ties choose the leftmost position).
Threshold/ranking membership is piecewise constant. This is not a smoothing
of the decoder and does not supply derivatives at support boundaries.
"""

import math

import torch


def _transition_parts(logits, targets, masked, selected, threshold, mask_token_id=None):
    # Float64 is retained for finite-difference tests; production logits are bf16.
    lp = logits.to(
        torch.float64 if logits.dtype == torch.float64 else torch.float32
    ).log_softmax(-1)
    chosen = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    count = selected.sum(-1)
    selected_min = chosen.masked_fill(~selected, float("inf")).min(-1).values
    threshold_event = selected_min >= math.log(threshold)
    winner = selected.to(torch.int64).argmax(-1)
    cutoff = torch.where(threshold_event, math.log(threshold), selected_min)
    positions = torch.arange(logits.shape[-2], device=logits.device)
    allow_tie = (~threshold_event).unsqueeze(-1) & (positions > winner.unsqueeze(-1))
    allowed = (lp < cutoff[:, None, None]) | (
        (lp == cutoff[:, None, None]) & allow_tie.unsqueeze(-1)
    )
    # A proposal equal to MASK leaves the visible canvas masked, even if
    # it passed the threshold. Marginalize that invisible transfer too.
    if mask_token_id is not None and 0 <= mask_token_id < logits.shape[-1]:
        allowed[..., mask_token_id] |= threshold_event.unsqueeze(-1)
    rejection_lp = lp.masked_fill(~allowed, float("-inf")).logsumexp(-1)
    rejected = masked & ~selected
    valid = (count > 0) & (threshold_event | (count == 1))
    valid &= (torch.isfinite(rejection_lp) | ~rejected).all(-1)
    valid &= (torch.isfinite(chosen) | ~selected).all(-1)
    joint = torch.where(selected, chosen, 0).sum(-1) + torch.where(
        rejected, rejection_lp, 0
    ).sum(-1)
    joint = torch.where(valid, joint, float("-inf"))
    return joint, valid, lp, allowed, rejection_lp


def confidence_transition_logprob(logits, targets, masked, selected, threshold=0.9):
    """Reference differentiable implementation, [steps, positions, vocabulary]."""
    return _transition_parts(logits, targets, masked, selected, threshold)[:2]


class _BlockConfidenceLogprob(torch.autograd.Function):
    """Recompute softmax in backward; keep no full-vocabulary probabilities.

    TP=CP=1 only. Each response block is one independent transition at a replay
    level. Output entries without a transition or with zero support are zero;
    the separate valid output distinguishes those entries from probability one.
    """

    @staticmethod
    def forward(
        ctx, logits, targets, masked, selected, block_size, threshold, mask_token_id
    ):
        batch, seq, vocab = logits.shape
        n = targets.shape[1]
        if n % block_size:
            raise ValueError(
                "Confidence correction requires block-aligned noisy length"
            )
        blocks = n // block_size
        selected_blocks = selected.reshape(batch, blocks, block_size)
        indices = selected_blocks.any(-1).nonzero()
        out = torch.zeros(
            (batch, blocks),
            device=logits.device,
            dtype=torch.float64 if logits.dtype == torch.float64 else torch.float32,
        )
        valid_out = torch.zeros_like(out, dtype=torch.bool)
        positions = torch.arange(block_size, device=logits.device)
        for start in range(0, len(indices), 4):
            idx = indices[start : start + 4]
            rows = idx[:, 0, None]
            cols = idx[:, 1, None] * block_size + positions
            joint, valid, *_ = _transition_parts(
                logits[rows, cols],
                targets[rows, cols],
                masked[rows, cols],
                selected[rows, cols],
                threshold,
                mask_token_id,
            )
            out[idx[:, 0], idx[:, 1]] = torch.where(valid, joint, 0).to(out.dtype)
            valid_out[idx[:, 0], idx[:, 1]] = valid
        ctx.save_for_backward(logits, targets, masked, selected, indices)
        ctx.block_size, ctx.threshold, ctx.mask_token_id = (
            block_size,
            threshold,
            mask_token_id,
        )
        ctx.mark_non_differentiable(valid_out)
        return out, valid_out

    @staticmethod
    def backward(ctx, grad_out, _grad_valid):
        logits, targets, masked, selected, indices = ctx.saved_tensors
        grad = torch.zeros_like(logits)
        positions = torch.arange(ctx.block_size, device=logits.device)
        for start in range(0, len(indices), 4):
            idx = indices[start : start + 4]
            rows = idx[:, 0, None]
            cols = idx[:, 1, None] * ctx.block_size + positions
            tgt, mask, sel = (
                targets[rows, cols],
                masked[rows, cols],
                selected[rows, cols],
            )
            _, valid, lp, allowed, rejection_lp = _transition_parts(
                logits[rows, cols], tgt, mask, sel, ctx.threshold, ctx.mask_token_id
            )
            p = lp.exp()
            selected_grad = -p
            selected_grad.scatter_add_(
                -1, tgt.unsqueeze(-1), torch.ones_like(tgt, dtype=p.dtype).unsqueeze(-1)
            )
            safe_rejection_lp = torch.where(
                torch.isfinite(rejection_lp), rejection_lp, 0
            )
            rejection_grad = (
                torch.where(allowed, (lp - safe_rejection_lp.unsqueeze(-1)).exp(), 0)
                - p
            )
            score = torch.where(sel.unsqueeze(-1), selected_grad, rejection_grad)
            score = torch.where((mask & valid.unsqueeze(-1)).unsqueeze(-1), score, 0)
            weight = grad_out[idx[:, 0], idx[:, 1]]
            grad[rows, cols] = (score * weight[:, None, None]).to(grad.dtype)
        return grad, None, None, None, None, None, None


def block_confidence_logprobs(
    logits, targets, masked, selected, block_size=16, threshold=0.9, mask_token_id=None
):
    return _BlockConfidenceLogprob.apply(
        logits, targets, masked, selected, block_size, threshold, mask_token_id
    )


def on_policy_correction(
    *,
    logits,
    token_logprobs,
    data,
    global_valid_toks,
    block_size=16,
    threshold=0.9,
    mask_token_id=100,
):
    """Replace the token-only actor gradient by the full transition score.

    Caller MUST use force_on_policy_ratio with one synchronous update, signed
    advantages, token normalization, no IS weights, and complete block traces.
    Includes post-EOS commits because they are part of the sampled latent trace;
    existing pre-EOS token KL regularization remains separate.

    Unsupported transitions caused by train/inference numerical mismatch have
    zero actor gradient and are counted explicitly (no epsilon likelihood floor).
    """
    n = token_logprobs.shape[1]
    targets = data["diffu_grpo_target_ids"][:, :n]
    selected = data["block_reveal_harvest_mask"][:, :n].bool()
    levels = data["trace_reveal_level"][:, :n]
    level = data["block_reveal_reveal_level"].unsqueeze(-1)
    lengths = data["diffu_grpo_response_lengths"]
    in_response = torch.arange(n, device=logits.device).unsqueeze(
        0
    ) < lengths.unsqueeze(-1)
    masked = in_response & (levels >= level)
    joint, valid = block_confidence_logprobs(
        logits, targets, masked, selected, block_size, threshold, mask_token_id
    )
    batch, blocks = joint.shape
    active = selected.reshape(batch, blocks, block_size).any(-1)
    # Cancel exactly the existing force_on_policy_ratio actor gradient, including
    # on zero-support events; add the supported full-transition score once.
    old_score = (
        (token_logprobs * data["diffu_grpo_loss_mask"][:, :n])
        .reshape(batch, blocks, block_size)
        .sum(-1)
    )
    actor_weight = data.get(
        "confidence_actor_weight", torch.ones_like(data["sample_mask"])
    )
    delta = actor_weight.detach().unsqueeze(-1) * joint - old_score
    advantage = data["advantages"][:, 0].unsqueeze(-1)
    sample_mask = data["sample_mask"].unsqueeze(-1)
    denominator = global_valid_toks + 1e-8
    correction = (
        -((delta - delta.detach()) * advantage * sample_mask).sum() / denominator
    )
    with torch.no_grad():
        count_mask = sample_mask.bool() & active
        selected_score = (
            torch.where(selected, token_logprobs, 0)
            .reshape(batch, blocks, block_size)
            .sum(-1)
        )
        metrics = {
            "confidence_steps": float(count_mask.sum().item()),
            "confidence_zero_support_steps": float((count_mask & ~valid).sum().item()),
            "confidence_selection_nll": float(
                (-(joint - selected_score) * count_mask * valid).sum().item()
                / denominator.item()
            ),
        }
    return correction, metrics


def validate_correction_config(grpo, policy, loss):
    est = policy["logprob_estimation"]
    gen = policy["generation"]
    decode = gen["vllm_kwargs"]["diffusion_config"]
    checks = {
        "synchronous rollout collection": not grpo.get("async_grpo", {}).get(
            "enabled", False
        ),
        "one update per rollout": int(grpo.get("num_updates_per_rollout", 1)) == 1,
        "one optimizer batch per rollout": policy["train_global_batch_size"]
        == grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"],
        "on-policy ratio": loss.get("force_on_policy_ratio", False),
        "no off-policy IS weighting": not loss.get(
            "use_importance_sampling_correction", False
        ),
        "token-level normalization": loss.get("token_level_loss", False),
        "full signed advantages": loss.get("negative_advantage_weight", 1) == 1,
        "no positive anchor": not est.get("positive_anchor_weight", 0),
        "exhaustive trace replay": est.get("num_level_samples") is None
        and est.get("max_reveal_levels") is None,
        "TP=CP=1": policy["megatron_cfg"]["tensor_model_parallel_size"] == 1
        and policy["megatron_cfg"]["context_parallel_size"] == 1,
        "full vocabulary": not est.get("exclude_mask_token_from_logits", True)
        and gen.get("top_p", 1) == 1
        and gen.get("top_k", -1) in (-1, 0, None),
        "T=1": gen["temperature"] == 1 and decode["temperature"] == 1,
        "threshold decoder": (
            gen["backend"] == "vllm"
            or (
                gen["backend"] == "megatron"
                and est.get("confidence_experiment", {}).get("mode") == "control"
            )
        )
        and decode["selection_policy"] == "confidence_threshold",
        "same threshold": decode["confidence_threshold"] == est["confidence_threshold"],
        "complete canvas traces": decode.get("emit_full_blocks", False)
        and decode.get("return_reveal_steps", False),
        "same block size": decode["canvas_length"] == est["block_size"],
        "sufficient denoising steps": decode["max_denoising_steps"]
        >= est["block_size"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Confidence transition correction requires: " + ", ".join(failed)
        )


def validate_complete_trace(base, block_size, mask_token_id, original_data=None):
    """Fail on truncated blocks or unrecorded no-progress denoising steps."""
    lengths = base["diffu_grpo_response_lengths"].tolist()
    targets = base["diffu_grpo_target_ids"].tolist()
    levels = base["trace_reveal_level"].tolist()
    if original_data is not None:
        raw = original_data["reveal_steps"].tolist()
        starts = base["diffu_grpo_completion_starts"].tolist()
        levels = [
            raw[row][start : start + lengths[row]] for row, start in enumerate(starts)
        ]
    for row, length in enumerate(lengths):
        if length % block_size:
            raise ValueError("Confidence correction needs complete emitted blocks")
        for start in range(0, length, block_size):
            if mask_token_id in targets[row][start : start + block_size]:
                raise ValueError("Unresolved MASK in confidence trace")
            steps = set(levels[row][start : start + block_size])
            if steps != set(range(max(steps) + 1)):
                raise ValueError("Confidence trace has unrecorded no-progress steps")
