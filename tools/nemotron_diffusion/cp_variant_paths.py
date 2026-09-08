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

"""Context-parallel variants V1/V2 for the SYMMETRIC diffusion layout.

Section 2 of `plans/full_cp_analysis.md`. Installed as a monkeypatch on
NemotronLabsDiffusionAttention rather than added to the bridge: these are
baselines for measurement, and the shipped symmetric path has no local-Q CP
branch at all (it gathers Q/K/V and runs the full attention on every rank).

  V1  symmetric, ONE global zigzag, all-gather K/V to full length
      Q per rank  S/cp        K/V per rank  S
  V2  symmetric, ONE global zigzag, all-gather only the CLEAN half
      Q per rank  S/cp        K/V per rank  S/(2cp) + S/2

Why one zigzag suffices for both, where the asymmetric layout needs two: the
symmetric halves are EQUAL, so with 2*cp chunks the noisy/clean midpoint lands
exactly on a chunk boundary. Chunks 0..cp-1 are entirely noisy and cp..2cp-1
entirely clean, so rank r owns noisy chunk r and clean chunk 2cp-1-r -- already
segment-aligned. The asymmetric layout (R vs P+R) has no such alignment, which
is what forces its segmented zigzag.

V2's noisy K/V locality is exact, not approximate: a noisy query reads noisy
keys only within its own `block_size` block, and rank r's noisy rows are the
CONTIGUOUS chunk r, so every noisy key it can reach is local -- provided the
chunk is block-aligned, which is asserted below.
"""

import torch

from megatron.bridge.diffusion.common.cp_utils import (
    all_gather_kv_seq_cp,
    all_gather_seq_cp,
    scatter_seq_cp,
)


class _AllGatherCleanCP(torch.autograd.Function):
    """All-gather ONE chunk per rank into ascending global order.

    `all_gather_kv_seq_cp` cannot be used for V2's clean half: its
    `_reorder_zigzag_chunks` assumes each rank contributes a zigzag PAIR
    (chunks r and 2cp-1-r) and splits the input in two. That holds for V1,
    which passes its whole shard, but not for the clean half alone -- rank r
    contributes the single chunk 2cp-1-r, so a plain all_gather returns the
    chunks DESCENDING and reversing restores global order.

    Backward mirrors `_AllGatherKVSeqCP`: with Q local, each rank's K/V grad
    carries only its own query rows, so the true grad is the all-reduced full
    grad sliced to this rank's chunk.
    """

    @staticmethod
    def forward(ctx, tensor, cp_group, seq_dim):
        cp_size = torch.distributed.get_world_size(cp_group)
        ctx.cp_group, ctx.cp_size, ctx.seq_dim = cp_group, cp_size, seq_dim
        ctx.cp_rank = torch.distributed.get_rank(cp_group)
        if cp_size == 1:
            return tensor.contiguous()
        tensor = tensor.contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, tensor, group=cp_group)
        # gathered[i] is global chunk 2cp-1-i, so reversed()[j] is chunk cp+j
        return torch.cat(gathered[::-1], dim=seq_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.cp_size == 1:
            return grad_output, None, None
        g = grad_output.contiguous().clone()
        torch.distributed.all_reduce(g, group=ctx.cp_group)
        # this rank owns global chunk 2cp-1-r, which sits at ascending slot cp-1-r
        slot = ctx.cp_size - 1 - ctx.cp_rank
        return torch.chunk(g, ctx.cp_size, dim=ctx.seq_dim)[slot].contiguous(), None, None


def _all_gather_clean(tensor, cp_group, seq_dim=0):
    return _AllGatherCleanCP.apply(tensor, cp_group, seq_dim)



# One live BlockMask, mirroring the shipped path's `_CURRENT_MASK`.
#
# WITHOUT this the variant paths rebuild the mask in EVERY layer -- 34 builds
# per step against the shipped path's 1 -- which shows up as a large fake
# slowdown for any custom variant measured against v3/ref. Same "one metadata
# generation live at a time" assumption the shipped cache documents; the bench
# calls reset_variant_mask_cache() when it switches variant.
_VARIANT_MASK = {"key": None, "mask": None}

# Counts cache MISSES, i.e. actual BlockMask constructions. Read by the bench's
# --count-masks to prove the build is outside the timed window. Counting here
# rather than wrapping `_build_block_mask` because that name is imported inside
# `install_symmetric_cp` and so lives in the forward's closure, out of reach of
# module-level patching.
MASK_BUILDS = [0]

# Per-microbatch RoPE tables for the symmetric variants. The shipped asymmetric
# path carries these on its metadata dict; the symmetric path has no metadata,
# so the bench holds the container here and clears it alongside the mask cache.
_SYM_ROPE_CACHE: dict = {}


def reset_variant_mask_cache():
    _VARIANT_MASK["key"] = None
    _VARIANT_MASK["mask"] = None
    _SYM_ROPE_CACHE.clear()


def _cached_mask(key, build):
    if _VARIANT_MASK["key"] != key:
        MASK_BUILDS[0] += 1
        _VARIANT_MASK["mask"] = build()
        _VARIANT_MASK["key"] = key
    return _VARIANT_MASK["mask"]

def _symmetric_mask_mod(block_size, half):
    """The sbd_block_diff predicate, in GLOBAL doubled-sequence coordinates.

    Same three cases as `compute_block_mask` / `compute_block_bias` in dllm.py;
    duplicated here only so the CP index remap can wrap it.
    """

    def sym(b, h, q_idx, kv_idx):
        del b, h
        x0_q = q_idx >= half
        x0_kv = kv_idx >= half
        blk_q = torch.where(x0_q, (q_idx - half) // block_size, q_idx // block_size)
        blk_kv = torch.where(x0_kv, (kv_idx - half) // block_size, kv_idx // block_size)
        block_diagonal = (blk_q == blk_kv) & (~x0_kv) & (~x0_q)
        offset_block_causal = (blk_q > blk_kv) & x0_kv & (~x0_q)
        fully_causal = (q_idx >= kv_idx) & x0_kv & x0_q
        return block_diagonal | offset_block_causal | fully_causal

    return sym


def _local_to_global_q(q_idx, cp_rank, cp_size, chunk):
    """Global position of a local query row under ONE global zigzag.

    Local rows are [noisy chunk r | clean chunk 2cp-1-r], each `chunk` long.
    Arithmetic rather than a gather table: this is a single zigzag, so the map
    is closed-form (the SEGMENTED case is what needs `segmented_zigzag_index_table`).
    """
    first = q_idx < chunk
    return torch.where(
        first,
        cp_rank * chunk + q_idx,
        (2 * cp_size - 1 - cp_rank) * chunk + (q_idx - chunk),
    )


def _local_to_global_kv_clean_only(kv_idx, cp_rank, cp_size, chunk, half):
    """Global position of a local K/V column under V2's [noisy local | clean full]."""
    del cp_size
    noisy = kv_idx < chunk
    return torch.where(noisy, cp_rank * chunk + kv_idx, half + (kv_idx - chunk))


def install_symmetric_cp(attn_mod, clean_only: bool = False, gather_q: bool = False):
    """Route the symmetric diffusion path onto a CP variant.

    `gather_q=True`  -> V0: all-gather Q, K AND V; every rank runs the FULL
                        attention and scatters its own rows back. CP shards
                        nothing that matters, so this costs cp_size x the
                        attention FLOPs. It is the naive baseline (rung 0 in
                        `cp_variants_overview.md`) and what the layer does when
                        no DIFFU_CP_* flag is set.
    `clean_only=False` -> V1: Q local, all-gather K/V to full length.
    `clean_only=True`  -> V2: Q local, all-gather only the clean half.

    Inference and the asymmetric path fall through to the shipped forward.
    """
    from megatron.core import parallel_state, tensor_parallel
    from megatron.bridge.diffusion.common.dllm import _build_block_mask

    cls = attn_mod.NemotronLabsDiffusionAttention
    orig_forward = cls.forward

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        if self._inference_mode or self._asymmetric_ar_metadata is not None:
            return orig_forward(self, query, key, value, attention_mask,
                                attn_mask_type, attention_bias, packed_seq_params)

        cp_size = self.cp_size
        cp_group = parallel_state.get_context_parallel_group() if cp_size > 1 else None
        cp_rank = parallel_state.get_context_parallel_rank() if cp_size > 1 else 0

        s_local = query.shape[0]
        full = s_local * cp_size
        half = full // 2
        chunk = s_local // 2                      # per-segment chunk on this rank
        bs = self.block_size

        if cp_size > 1:
            assert chunk % bs == 0, (
                f"V1/V2 symmetric CP: per-rank chunk {chunk} must be a multiple of "
                f"block_size {bs}; otherwise a rank's noisy queries need noisy keys "
                f"owned by another rank"
            )
            if gather_q:
                # V0: reconstruct the whole sequence on every rank. Uses
                # all_gather_seq_cp (slice-only backward) rather than the
                # local-Q K/V gather, because with Q gathered every rank
                # computes every query row, so the downstream grad IS replicated.
                query = all_gather_seq_cp(query, cp_group, seq_dim=0)
                key = all_gather_seq_cp(key, cp_group, seq_dim=0)
                value = all_gather_seq_cp(value, cp_group, seq_dim=0)
            elif clean_only:
                # keep this rank's noisy chunk local, gather only the clean half
                key = torch.cat([key[:chunk], _all_gather_clean(key[chunk:], cp_group)], dim=0)
                value = torch.cat([value[:chunk], _all_gather_clean(value[chunk:], cp_group)], dim=0)
            else:
                key = all_gather_kv_seq_cp(key, cp_group, seq_dim=0)
                value = all_gather_kv_seq_cp(value, cp_group, seq_dim=0)

        q_len, kv_len = query.shape[0], key.shape[0]
        expect = (chunk + half) if (clean_only and cp_size > 1) else full
        assert kv_len == expect, f"expected KV_LEN {expect}, got {kv_len}"
        assert q_len == (full if (gather_q or cp_size == 1) else s_local)

        # ---- position ids, in GLOBAL doubled-sequence coordinates ----
        dev = query.device
        if cp_size == 1 or gather_q:
            # rows and columns are already in global order
            q_global = torch.arange(q_len, device=dev)
            kv_global = torch.arange(kv_len, device=dev)
        else:
            q_global = _local_to_global_q(
                torch.arange(q_len, device=dev), cp_rank, cp_size, chunk)
            kv_local = torch.arange(kv_len, device=dev)
            kv_global = (_local_to_global_kv_clean_only(kv_local, cp_rank, cp_size, chunk, half)
                         if clean_only else kv_local)

        # RoPE is applied per HALF in this layout, so the table position is the
        # index within the half: g for the noisy half, g-half for the clean one.
        q_rope = (q_global % half).unsqueeze(0)
        k_rope = (kv_global % half).unsqueeze(0)

        # [s, b, np, hn] -> [b, np, s, hn]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        # Same per-microbatch RoPE sharing the shipped asymmetric path gets.
        # Without a role these variants would recompute and retain per layer
        # while v3/ref share, and the symmetric-vs-asymmetric comparison would
        # be measuring that difference instead of the layout.
        rope_cache = _SYM_ROPE_CACHE
        cos_q, sin_q = self.rope_embedding_module(query, q_rope, role="sym_q", cache=rope_cache)
        query = attn_mod.apply_rotary_pos_emb_single(query, cos_q, sin_q)
        cos_k, sin_k = self.rope_embedding_module(key, k_rope, role="sym_k", cache=rope_cache)
        key = attn_mod.apply_rotary_pos_emb_single(key, cos_k, sin_k)

        if self.beta is not None:
            # the shipped symmetric path scales by the DOUBLED-sequence index,
            # not the per-half one; reproduce that using global positions.
            scale = attn_mod._get_llama_4_attn_scale(
                q_global, self.beta, self.max_position_embeddings).to(query.dtype)
            query = query * scale.view(1, 1, -1, 1)

        n_rep = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        use_native_gqa = attn_mod._FLEX_SUPPORTS_GQA and n_rep > 1
        if n_rep > 1 and not use_native_gqa:
            key = attn_mod.repeat_kv(key, n_rep)
            value = attn_mod.repeat_kv(value, n_rep)

        sym = _symmetric_mask_mod(bs, half)
        if cp_size > 1 and not gather_q:
            # Arithmetic, NOT a gather table. mask_mod is lowered into the
            # attention Triton kernel and re-evaluated on every partial block,
            # so a table lookup is paid per element; the shipped local-Q path
            # uses closed-form arithmetic for exactly this reason. V1's K/V is
            # already in global order, so its kv index passes through untouched.
            _co = clean_only

            def mask_mod(b, h, q_idx, kv_idx):
                q_g = _local_to_global_q(q_idx, cp_rank, cp_size, chunk)
                kv_g = (_local_to_global_kv_clean_only(kv_idx, cp_rank, cp_size, chunk, half)
                        if _co else kv_idx)
                return sym(b, h, q_g, kv_g)
        else:
            mask_mod = sym

        block_mask = _cached_mask(
            ("sym", clean_only, gather_q, q_len, kv_len, cp_rank, cp_size, half, bs),
            lambda: _build_block_mask(
                mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=dev),
        )
        context = attn_mod.fused_flex_attention(
            query, key, value, block_mask=block_mask, enable_gqa=use_native_gqa
        )

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context = self.attention_dropout(context)
        else:
            context = self.attention_dropout(context)

        context = context.transpose(1, 2).transpose(0, 1)
        shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.contiguous().view(*shape)
        # V0 computed every query row; keep only this rank's zigzag slice.
        # (local-Q paths already produce exactly this rank's rows.)
        if cp_size > 1 and gather_q:
            context = scatter_seq_cp(context, cp_group, seq_dim=0)
        return context

    cls.forward = forward
