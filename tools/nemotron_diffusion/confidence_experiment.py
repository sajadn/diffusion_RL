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
"""Opt-in confidence likelihood experiments; no effect without configured hooks."""

import hashlib
import json
import math
import os
import socket
from collections import defaultdict
from pathlib import Path

import torch

G_INDEX_ROUND = None
G_INDEX = None
FIELDS = (
    "confidence_logratio",
    "confidence_unsupported",
    "confidence_count",
    "confidence_ambiguous",
)


def root():
    return Path(os.environ["NRL_CONFIDENCE_EXPERIMENT_DIR"])


def start_round(step):
    folder = root()
    folder.mkdir(parents=True, exist_ok=True)
    name = f"step_{step:05d}"
    (folder / name).mkdir(exist_ok=True)
    temporary = folder / f"round.{os.getpid()}.tmp"
    temporary.write_text(name)
    temporary.replace(folder / "round")


def key(prefix, before, after):
    return hashlib.sha256(
        json.dumps([prefix, before, after], separators=(",", ":")).encode()
    ).hexdigest()


def write_records(records):
    if not records or not (root() / "round").exists():
        return
    folder = root() / (root() / "round").read_text().strip()
    with (folder / f"generation_{socket.gethostname()}_{os.getpid()}.jsonl").open(
        "a"
    ) as stream:
        stream.writelines(
            json.dumps(record, separators=(",", ":")) + "\n" for record in records
        )


def generation_index():
    global G_INDEX_ROUND, G_INDEX
    round_path = root() / (root() / "round").read_text().strip()
    if G_INDEX_ROUND != str(round_path):
        result = defaultdict(list)
        for path in round_path.glob("generation_*.jsonl"):
            for line in path.open():
                item = json.loads(line)
                result[item[0]].append(item)
        G_INDEX, G_INDEX_ROUND = result, str(round_path)
    return G_INDEX


@torch.no_grad()
def before_sample(sampler, slots, committing):
    if not (root() / "round").exists():
        return None
    indices = (~committing).nonzero().flatten()
    if not len(indices):
        return None
    chosen_slots = slots[indices]
    lengths = sampler.req_states.total_len.gpu[chosen_slots].cpu().tolist()
    before = sampler.diffusion_states.canvas[chosen_slots].clone()
    prefixes = [
        sampler.req_states.all_token_ids.gpu[int(slot), :length].cpu().tolist()
        for slot, length in zip(chosen_slots.cpu().tolist(), lengths)
    ]
    return indices, chosen_slots, before, prefixes


@torch.no_grad()
def after_sample(sampler, context, logits):
    if context is None:
        return
    indices, slots, before, prefixes = context
    after = sampler.diffusion_states.canvas[slots].clone()
    record_transitions(logits[indices], before, after, prefixes)


@torch.no_grad()
def record_transitions(logits, before, after, prefixes):
    from nemo_rl.algorithms.confidence_transition import _transition_parts

    records = []
    for offset in range(0, len(before), 4):
        b = before[offset : offset + 4]
        a = after[offset : offset + 4]
        masked = b == 100
        selected = masked & (a != b)
        joint, valid, lp, *_ = _transition_parts(
            logits[offset : offset + 4], a.long(), masked, selected, 0.9, 100
        )
        target_lp = lp.gather(-1, a.long().unsqueeze(-1)).squeeze(-1)
        for j in range(len(b)):
            if not selected[j].any():
                raise RuntimeError(
                    "Confidence experiment encountered an unrecorded no-progress event"
                )
            if not valid[j]:
                raise RuntimeError(
                    "Generation transition is unsupported under its own logits"
                )
            records.append(
                [
                    key(prefixes[offset + j], b[j].cpu().tolist(), a[j].cpu().tolist()),
                    float(joint[j]),
                    target_lp[j, selected[j]].cpu().tolist(),
                ]
            )
    write_records(records)


@torch.no_grad()
def replay_fields(logits, data):
    from nemo_rl.algorithms.confidence_transition import _transition_parts

    n = int(data["diffu_grpo_noisy_lengths"][0])
    selected = data["block_reveal_harvest_mask"][:, :n].bool()
    targets = data["diffu_grpo_target_ids"][:, :n].long()
    before = data["input_ids"][:, :n]
    masked = before == 100
    output = {name: torch.zeros_like(selected, dtype=torch.float32) for name in FIELDS}
    active = selected.reshape(len(selected), -1, 16).any(-1).nonzero().tolist()
    index = generation_index()
    for row, block in active:
        lo, hi = block * 16, (block + 1) * 16
        sel = selected[row, lo:hi]
        target = targets[row, lo:hi]
        b = before[row, lo:hi]
        a = torch.where(sel, target, b)
        clean = int(data["diffu_grpo_noisy_lengths"][row])
        prompt = int(data["diffu_grpo_completion_starts"][row])
        prefix = data["input_ids"][row, clean : clean + prompt + lo].cpu().tolist()
        match_key = key(prefix, b.cpu().tolist(), a.cpu().tolist())
        options = index.get(match_key)
        if not options:
            raise RuntimeError(f"Missing generation transition: {match_key}")
        saved_lp = data["generation_logprobs"][row, lo:hi][sel].cpu()

        def error(item):
            return float((torch.tensor(item[2]) - saved_lp).abs().max())

        generation = min(options, key=error)
        if error(generation) > 1e-5:
            raise RuntimeError(
                f"Generation logprob alignment error: {error(generation)}"
            )
        joint, valid, *_ = _transition_parts(
            logits[row : row + 1, lo:hi],
            target[None],
            masked[row : row + 1, lo:hi],
            sel[None],
            0.9,
            100,
        )
        plausible = [item[1] for item in options if error(item) <= 1e-5]
        ambiguous = max(plausible) - min(plausible) > 1e-4
        anchor = lo + int(sel.nonzero()[0])
        output["confidence_ambiguous"][row, anchor] = float(ambiguous)
        output["confidence_logratio"][row, anchor] = (
            joint[0] - generation[1] if valid[0] else 0
        )
        output["confidence_unsupported"][row, anchor] = ~valid[0]
        output["confidence_count"][row, anchor] = 1
    return output


def response_weights(
    logratio, max_step_error, unsupported, *, mode, bound=4.0, clip=2.0
):
    supported = unsupported == 0
    if mode == "filter":
        return (supported & (max_step_error <= math.log(bound))).float()
    if mode == "importance":
        return torch.where(
            supported, torch.exp(logratio.clamp(max=math.log(clip))), 0.0
        )
    if mode == "control":
        return torch.ones_like(logratio)
    raise ValueError(mode)


def attach_weights(train_data, logprob_output, config):
    ratio = logprob_output["confidence_logratio"]
    maximum = logprob_output["confidence_max_step_error"]
    unsupported = logprob_output["confidence_unsupported"]
    ambiguous = logprob_output["confidence_ambiguous"]
    weight = response_weights(
        ratio,
        maximum,
        unsupported + ambiguous,
        mode=config["mode"],
        bound=config["filter_bound"],
        clip=config["importance_clip"],
    )
    train_data["confidence_actor_weight"] = weight
    train_data["confidence_logratio"] = ratio
    train_data["confidence_unsupported"] = unsupported
    train_data["confidence_ambiguous"] = ambiguous
    original = train_data["sample_mask"].bool()
    active = weight[original]
    effective = float(active.sum().square() / active.square().sum().clamp_min(1e-20))
    stats = {
        "responses": int(original.sum()),
        "retained": int((active > 0).sum()),
        "unsupported_responses": int((unsupported[original] > 0).sum()),
        "ambiguous_responses": int((ambiguous[original] > 0).sum()),
        "mean_actor_weight": float(active.mean()),
        "effective_sample_size": effective,
        "max_abs_supported_trajectory_logratio": float(
            ratio[original & (unsupported == 0)].abs().max()
        )
        if bool((original & (unsupported == 0)).any())
        else None,
    }
    print("CONFIDENCE_EXPERIMENT " + json.dumps(stats), flush=True)
    if not bool((active > 0).any()):
        raise RuntimeError("Confidence experiment retained no actor training signal")
    return stats


def log_retained_advantages(data, prompt_ids, rewards):
    token_mask = data["token_mask"]
    advantage = (data["advantages"] * token_mask).sum(-1) / token_mask.sum(
        -1
    ).clamp_min(1)
    weights = data["confidence_actor_weight"]
    valid = data["sample_mask"].bool()
    kept = valid & (weights > 0)
    stats = {}
    for name, sign in [
        ("positive", advantage > 1e-6),
        ("negative", advantage < -1e-6),
        ("near_zero", advantage.abs() <= 1e-6),
    ]:
        stats[name + "_original"] = int((valid & sign).sum())
        stats[name + "_retained"] = int((kept & sign).sum())
    groups = defaultdict(list)
    for index, prompt in enumerate(prompt_ids):
        groups[str(prompt)].append(index)
    mixed = 0
    for indices in groups.values():
        rows = torch.tensor(indices, device=kept.device)
        mixed += bool(
            (kept[rows] & (advantage[rows] > 1e-6)).any()
            and (kept[rows] & (advantage[rows] < -1e-6)).any()
        )
    stats["groups_with_both_retained_advantage_signs"] = mixed
    stats["groups"] = len(groups)
    print("CONFIDENCE_RETAINED " + json.dumps(stats), flush=True)
    folder = root() / (root() / "round").read_text().strip()
    values = {
        name: data[name].detach().cpu().tolist()
        for name in [
            "confidence_actor_weight",
            "confidence_logratio",
            "confidence_unsupported",
            "confidence_ambiguous",
        ]
    }
    values["advantages"] = advantage.detach().cpu().tolist()
    values["rewards"] = rewards.detach().cpu().tolist()
    values["retention"] = stats
    (folder / "response_weights.json").write_text(json.dumps(values))
    return stats
