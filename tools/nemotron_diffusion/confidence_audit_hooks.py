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
"""Opt-in read-only paired transition capture for frozen diagnostic jobs."""

import hashlib
import json
import os
from pathlib import Path
import socket
import torch

from nemo_rl.algorithms import confidence_transition as math_module

ROOT = Path(os.environ["NRL_CONFIDENCE_AUDIT_DIR"])
COUNTERS = {"generation": 0, "replay": 0}


def transition_key(prefix, before, after):
    return hashlib.sha256(
        json.dumps([prefix, before, after], separators=(",", ":")).encode()
    ).hexdigest()


def save(kind, records):
    if not records:
        return
    folder = ROOT / kind
    folder.mkdir(parents=True, exist_ok=True)
    serial = COUNTERS[kind]
    COUNTERS[kind] += 1
    path = folder / f"{socket.gethostname()}_{os.getpid()}_{serial:05d}.pt"
    torch.save(records, path)


@torch.no_grad()
def before_sample(sampler, slots, is_committing):
    indices = (~is_committing).nonzero().flatten()
    if not len(indices):
        return None
    selected_slots = slots[indices]
    lengths = sampler.req_states.total_len.gpu[selected_slots].cpu().tolist()
    before = sampler.diffusion_states.canvas[selected_slots].clone()
    steps = sampler.diffusion_states.step[selected_slots].cpu().tolist()
    prefixes = [
        sampler.req_states.all_token_ids.gpu[int(slot), : int(length)].cpu().tolist()
        for slot, length in zip(selected_slots.cpu().tolist(), lengths)
    ]
    return indices, selected_slots, before, steps, prefixes


@torch.no_grad()
def after_sample(sampler, context, logits):
    if context is None:
        return
    indices, slots, before, steps, prefixes = context
    after = sampler.diffusion_states.canvas[slots].clone()
    masked = before == int(sampler.diffusion_states.mask_token_id)
    selected = masked & (after != before)
    rows = logits[indices]
    joint, valid, lp, _, _ = math_module._transition_parts(
        rows,
        after.long(),
        masked,
        selected,
        0.9,
        int(sampler.diffusion_states.mask_token_id),
    )
    values, ids = lp.topk(64, dim=-1)
    records = []
    for i in range(len(indices)):
        b = before[i].cpu().tolist()
        a = after[i].cpu().tolist()
        records.append(
            dict(
                key=transition_key(prefixes[i], b, a),
                prefix=prefixes[i],
                before=b,
                after=a,
                step=steps[i],
                selected=selected[i].cpu(),
                target_lp=lp[i]
                .gather(-1, after[i].long().unsqueeze(-1))
                .squeeze(-1)
                .cpu(),
                top_lp=values[i].cpu(),
                top_ids=ids[i].cpu(),
                joint_lp=float(joint[i]),
                valid=bool(valid[i]),
            )
        )
    save("generation", records)


@torch.no_grad()
def replay(logits, token_logprobs, data, block_size, threshold, mask_token_id):
    n = token_logprobs.shape[1]
    targets = data["diffu_grpo_target_ids"][:, :n]
    selected = data["block_reveal_harvest_mask"][:, :n].bool()
    levels = data["trace_reveal_level"][:, :n]
    level = data["block_reveal_reveal_level"].unsqueeze(-1)
    masked = (levels >= level) & (
        torch.arange(n, device=logits.device).unsqueeze(0)
        < data["diffu_grpo_response_lengths"].unsqueeze(-1)
    )
    active = selected.reshape(len(selected), -1, block_size).any(-1).nonzero()
    records = []
    for row, block in active.tolist():
        lo = block * block_size
        hi = lo + block_size
        sel = selected[row, lo:hi]
        tgt = targets[row, lo:hi].long()
        msk = masked[row, lo:hi]
        joint, valid, lp, _, rejection = math_module._transition_parts(
            logits[row : row + 1, lo:hi],
            tgt[None],
            msk[None],
            sel[None],
            threshold,
            mask_token_id,
        )
        values, ids = lp[0].topk(64, dim=-1)
        before = data["input_ids"][row, lo:hi].cpu().tolist()
        after = torch.where(sel, tgt, data["input_ids"][row, lo:hi]).cpu().tolist()
        clean = int(data["diffu_grpo_noisy_lengths"][row])
        prompt = int(data["diffu_grpo_completion_starts"][row])
        prefix = data["input_ids"][row, clean : clean + prompt + lo].cpu().tolist()
        selected_lp = lp[0].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        reason = "supported"
        if not bool(valid[0]):
            reason = (
                "multiple_selected_below_threshold"
                if int(sel.sum()) > 1
                and bool((selected_lp[sel] < __import__("math").log(threshold)).any())
                else "rejection_or_nonfinite"
            )
        records.append(
            dict(
                key=transition_key(prefix, before, after),
                prefix=prefix,
                before=before,
                after=after,
                step=int(level[row, 0]),
                block=block,
                selected=sel.cpu(),
                target_lp=selected_lp.cpu(),
                generation_lp=data["generation_logprobs"][row, lo:hi].cpu(),
                prev_lp=data["prev_logprobs"][row, lo:hi].cpu(),
                train_token_lp=token_logprobs[row, lo:hi].cpu(),
                logits_dtype=str(logits.dtype),
                top_lp=values.cpu(),
                top_ids=ids.cpu(),
                joint_lp=float(joint[0]),
                valid=bool(valid[0]),
                reason=reason,
                rejection_lp=rejection[0].cpu(),
                advantage=float(data["advantages"][row, 0]),
            )
        )
    save("replay", records)
