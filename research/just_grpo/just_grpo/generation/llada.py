# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-canvas LLaDA sampling with leftmost or confidence position selection."""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class LLaDAGenerationOutput:
    """Full-canvas samples and globally ordered reveal metadata."""

    token_ids: torch.Tensor
    logprobs: torch.Tensor
    num_forward_passes: torch.Tensor
    reveal_steps: torch.Tensor
    fallback_mask: torch.Tensor


@torch.no_grad()
def generate_llada(
    forward: Callable[[torch.Tensor], torch.Tensor],
    prompts: torch.Tensor,
    *,
    completion_length: int,
    block_length: int,
    mask_token_id: int,
    temperature: float,
    generator: torch.Generator | None = None,
    threshold: float | None = None,
    tokens_per_step: int = 1,
    leftmost_fallback: bool = False,
) -> LLaDAGenerationOutput:
    """Reveal confident tokens while retaining the full masked future canvas.

    Each row must have the same prompt length and contain no prompt padding.
    ``block_length=1`` gives JustGRPO rollouts; 32 gives the paper's evaluation.
    Without a threshold, reveal exactly tokens_per_step positions per call.
    A threshold enables parallel reveals with a highest-confidence fallback.
    With leftmost_fallback, use Fork selection and record fallback decisions.
    Reveal steps are global across blocks so bidirectional replay retains the
    exact full-canvas context; they do not restart at block boundaries.
    """
    if completion_length <= 0 or block_length <= 0:
        raise ValueError("completion_length and block_length must be positive")
    if completion_length % block_length:
        raise ValueError("completion_length must be divisible by block_length")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if threshold is not None and not 0 < threshold < 1:
        raise ValueError("Confidence threshold must be between zero and one")
    if tokens_per_step <= 0 or block_length % tokens_per_step:
        raise ValueError("tokens_per_step must be positive and divide block_length")
    if threshold is not None and tokens_per_step != 1:
        raise ValueError("Threshold and fixed parallel reveals are mutually exclusive")
    if leftmost_fallback and threshold is None:
        raise ValueError("Fork decoding requires a confidence threshold")
    prompt_length = prompts.shape[1]
    tokens = torch.full(
        (prompts.shape[0], prompt_length + completion_length),
        mask_token_id,
        device=prompts.device,
        dtype=prompts.dtype,
    )
    tokens[:, :prompt_length] = prompts
    remaining = torch.zeros_like(tokens, dtype=torch.bool)
    remaining[:, prompt_length:] = True
    logprobs = torch.zeros_like(tokens, dtype=torch.float32)
    forward_counts = torch.zeros(
        prompts.shape[0], dtype=torch.long, device=prompts.device
    )
    reveal_steps = torch.full_like(tokens, -1)
    fallback_mask = torch.zeros_like(tokens, dtype=torch.bool)
    reveal_step = 0
    for start in range(prompt_length, tokens.shape[1], block_length):
        end = start + block_length
        for _ in range(block_length // tokens_per_step):
            active = remaining[:, start:end].any(dim=-1)
            if threshold is not None and not active.any():
                break
            forward_counts += active.long()
            # Preserve bf16 logits for confidence ranking, as the HF reference does.
            logits = forward(tokens).to(torch.bfloat16)
            if temperature == 0:
                candidates = logits.argmax(dim=-1)
            else:
                scores = logits.to(torch.float64)
                noise = -torch.log(
                    torch.rand(
                        scores.shape,
                        device=scores.device,
                        dtype=scores.dtype,
                        generator=generator,
                    )
                )
                candidates = (scores.exp() / noise.pow(temperature)).argmax(dim=-1)
            confidence = (
                logits.softmax(dim=-1).gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
            )
            confidence[:, :start] = -torch.inf
            confidence[:, end:] = -torch.inf
            confidence.masked_fill_(~remaining, -torch.inf)
            positions = confidence.topk(k=tokens_per_step, dim=-1).indices
            transfer = torch.zeros_like(tokens, dtype=torch.bool)
            if threshold is not None:
                transfer = confidence >= threshold
            if leftmost_fallback:
                fallback_rows = active & ~transfer.any(dim=1)
                leftmost = remaining[:, start:end].long().argmax(1) + start
                fallback_mask[fallback_rows, leftmost[fallback_rows]] = True
                transfer[fallback_rows, leftmost[fallback_rows]] = True
            else:
                transfer.scatter_(
                    1, positions, confidence.gather(1, positions).isfinite()
                )
            rows, selected_positions = transfer.nonzero(as_tuple=True)
            selected = candidates[rows, selected_positions]
            tokens[rows, selected_positions] = selected
            remaining[rows, selected_positions] = False
            reveal_steps[rows, selected_positions] = reveal_step
            reveal_step += 1
            selected_logits = logits[rows, selected_positions].float()
            if temperature > 0:
                selected_logits = selected_logits / temperature
            logprobs[rows, selected_positions] = (
                selected_logits.log_softmax(-1)
                .gather(-1, selected.unsqueeze(-1))
                .squeeze(-1)
            )
    return LLaDAGenerationOutput(
        tokens, logprobs, forward_counts, reveal_steps, fallback_mask
    )
