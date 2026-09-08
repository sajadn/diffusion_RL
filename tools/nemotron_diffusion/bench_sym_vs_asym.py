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

"""Fwd/bwd bench: SYMMETRIC vs ASYMMETRIC diffusion attention setup, cp=1.

NO context parallelism. This measures the two attention SETUPS on their own,
which is the only way to separate "the asymmetric LAYOUT is better" from
"the asymmetric path happens to run on a better KERNEL".

Three arms, chosen so each adjacent pair varies exactly one thing:

  variant         layout        attn seq   kernel                  mask builder
  -------------   -----------   --------   ---------------------   ----------------------
  symmetric_te    [xt|x0]       2*(P+R)    TE/cuDNN dense bias     compute_block_bias
  symmetric_flex  [xt|x0]       2*(P+R)    flex BlockMask          compute_block_mask
  asymmetric      [noisy|clean] P + 2R     flex BlockMask          compute_asymmetric_semi_ar_mask

  symmetric_te - symmetric_flex  =  pure KERNEL effect. `compute_block_bias` and
      `compute_block_mask` are the SAME predicate (block_diagonal |
      offset_block_causal | fully_causal, identical formulas in dllm.py) --
      one emits a dense additive [1,1,S,S] bias, the other a BlockMask. Same
      layout, same sequence, same allowed pairs. Only the kernel differs.

  symmetric_flex - asymmetric    =  pure LAYOUT effect. Same flex kernel, same
      GQA handling; the symmetric layout noises prompt+response and so doubles
      P+R, while the asymmetric one noises the response only and doubles R.

`symmetric_flex` is not hypothetical: it is what this module DID until commit
ba1c63e9 ("context parallelism for NemotronLabsDiffusion, PR #4447"), which
replaced the flex call with TE + a dense post_scale_bias in order to get CP --
TE has no backend for post_scale_bias + context_parallel on this stack, so the
symmetric path's CP is gather-Q/K/V-and-scatter. This arm reconstructs the
pre-ba1c63e9 forward, modernized in ONE respect: it uses `enable_gqa` like the
current asymmetric path instead of the historical `repeat_kv`, so the kernel
comparison is not contaminated by an obsolete GQA expansion.

One variant per process invocation, on purpose: the symmetric_te arm allocates
a dense [1,1,2(P+R),2(P+R)] bf16 bias PER LAYER (34 of them, and the tensor
does not shard with tp), so an in-process variant loop would let allocator
fragmentation from one arm contaminate the next arm's peak.

Usage (inside the container, from the repo root):

  torchrun --nproc_per_node=8 tools/nemotron_diffusion/bench_sym_vs_asym.py \\
      --tp 8 --prompt 2048 --response 4096 --variant asymmetric
"""

import argparse
import datetime
import os

import torch
import torch.distributed as dist

DEFAULT_MODEL = (
    "/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/"
    "Nemotron-Labs-Diffusion-8B-v1.5-sft"
)

VARIANTS = ("symmetric_te", "symmetric_flex", "asymmetric")


def layout_for(variant, prompt, response):
    """(noisy, clean, attention sequence length) for each arm."""
    if variant == "asymmetric":
        noisy, clean = response, prompt + response
        return noisy, clean, noisy + clean
    half = prompt + response
    return None, None, 2 * half


def install_symmetric_flex(attn_mod):
    """Route the symmetric diffusion path back onto flex_attention.

    Reconstructs the pre-ba1c63e9 forward rather than adding a branch to the
    shipped one: the point of this arm is to hold the mask semantics and the
    layout fixed while swapping ONLY the kernel, so it must reuse the module's
    own `compute_block_mask` and its own RoPE / Llama-4 / GQA / output-reshape
    steps verbatim. Anything hand-rolled here would show up as a kernel delta.

    The block mask is built at the RUNTIME half length rather than
    `self.mask_seq_length` (which the historical code used): the dense-bias arm
    derives its bias from the runtime shape too (`full_2l // 2`), so deriving
    both from the same source is what keeps the two arms comparable.
    """
    from megatron.core import tensor_parallel
    from megatron.bridge.diffusion.common.dllm import compute_block_mask

    cls = attn_mod.NemotronLabsDiffusionAttention
    orig_forward = cls.forward

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        # Only the symmetric diffusion path is redirected. Inference and the
        # asymmetric path keep the shipped implementation.
        if self._inference_mode or self._asymmetric_ar_metadata is not None:
            return orig_forward(self, query, key, value, attention_mask,
                                attn_mask_type, attention_bias, packed_seq_params)

        half_seq_len = query.shape[0] // 2
        position_ids = torch.arange(half_seq_len, device=query.device).unsqueeze(0)
        cos, sin = self.rope_embedding_module(query, position_ids)

        # [sq, b, np, hn] -> [b, np, sq, hn]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        # RoPE independently per half (xt and x0)
        q1, q2 = query.chunk(2, dim=2)
        k1, k2 = key.chunk(2, dim=2)
        q1, k1 = attn_mod.apply_rotary_pos_emb(q1, k1, cos, sin)
        q2, k2 = attn_mod.apply_rotary_pos_emb(q2, k2, cos, sin)
        query = torch.cat([q1, q2], dim=2)
        key = torch.cat([k1, k2], dim=2)

        if self.beta is not None:
            cache_position = torch.arange(query.shape[2], device=query.device)
            query = query * attn_mod._get_llama_4_attn_scale(
                cache_position, self.beta, self.max_position_embeddings
            ).to(query.dtype)

        # Same GQA gating as the shipped asymmetric path.
        n_rep = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        use_native_gqa = attn_mod._FLEX_SUPPORTS_GQA and n_rep > 1
        if n_rep > 1 and not use_native_gqa:
            key = attn_mod.repeat_kv(key, n_rep)
            value = attn_mod.repeat_kv(value, n_rep)

        block_mask = self._mask_cache.get(half_seq_len)
        if block_mask is None:
            block_mask = compute_block_mask(
                block_size=self.block_size, max_seq_length=half_seq_len
            )
            self._mask_cache[half_seq_len] = block_mask

        context = attn_mod.fused_flex_attention(
            query, key, value, block_mask=block_mask, enable_gqa=use_native_gqa
        )

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context = self.attention_dropout(context)
        else:
            context = self.attention_dropout(context)

        context = context.transpose(1, 2).transpose(0, 1)
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        return context.contiguous().view(*new_context_shape)

    cls.forward = forward



def canonicalize_te_inputs(modules):
    """Work around TE rejecting the shipped symmetric path's q/k/v at tp=8.

    `get_qkv_layout` compares k's and v's strides (`check_strides_kv`). In the
    symmetric path k is rebuilt by `torch.cat` + transposes while v is only
    transposed, and when `num_query_groups // tp == 1` (8 KV groups at tp=8)
    the two end up disagreeing on the strides of their size-1 head dim:

        k: stride=(128, 131072, 131072, 1)
        v: stride=(128,    128,    128, 1)

    so TE returns "not_supported" and raises. Its own fallback --
    `[x.contiguous() for x in (q, k, v)]` -- cannot fix this, because PyTorch
    reports a tensor with size-1 dims as contiguous REGARDLESS of the strides on
    those dims, so `.contiguous()` is a no-op that preserves the stale stride.

    This is a real bug in the shipped path, latent only because RL uses the
    asymmetric path; it reproduces at tp=8 and not at tp<=4. Rebuilding through
    a flat view restores canonical strides at zero copy (the tensor really is
    contiguous, so `reshape(-1)` is a view and the element order is correct).
    Applied ONLY to the symmetric_te arm, and it moves no data, so it does not
    flatter that arm's timing.
    """
    import torch

    def canonical(shape):
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    def fix(x):
        if x.is_contiguous() and x.stride() != canonical(tuple(x.shape)):
            return x.reshape(-1).view(x.shape)
        return x

    for m in modules:
        core = m.core_attention
        if getattr(core, "_symasym_wrapped", False):
            continue
        orig = core.forward

        def wrapped(q, k, v, *a, _orig=orig, **kw):
            return _orig(fix(q), fix(k), fix(v), *a, **kw)

        core.forward = wrapped
        core._symasym_wrapped = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="symmetric vs asymmetric attention bench (cp=1)")
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--prompt", type=int, required=True, help="prompt length P")
    p.add_argument("--response", type=int, required=True, help="response length R")
    p.add_argument("--variant", type=str, required=True, choices=VARIANTS)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--layers", type=int, default=0, help="0 = the checkpoint's own")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--no-sequence-parallel", action="store_true")
    return p.parse_args()


def build_model(args, device, seq_length):
    """Real NemotronLabsDiffusion, random-initialized.

    `to_megatron_provider(load_weights=False)` rather than `provider_bridge()`:
    the latter leaves `perform_initialization=True` on this family. Weight
    VALUES are irrelevant to a shape-driven benchmark, but the provider is also
    where parallelism and dtype are configured, so it has to be the same entry
    point production uses.

    `seq_length` is the LARGEST sequence any arm will run, not this arm's own,
    so every arm gets an identical RoPE/yarn configuration
    (`max_position_embeddings` feeds the yarn scaling factor). Per-arm values
    would change the rotation and the arms would stop being comparable.
    """
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = args.tp
    provider.context_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.sequence_parallel = not args.no_sequence_parallel
    provider.bf16 = True
    provider.fp16 = False
    provider.params_dtype = torch.bfloat16
    provider.seq_length = seq_length
    # The fused grad-accum kernel lives in APEX, which this container lacks.
    # Identical across arms, so it cancels -- and Megatron hard-errors without it.
    provider.gradient_accumulation_fusion = False
    if args.layers:
        provider.num_layers = args.layers
    if hasattr(provider, "finalize"):
        provider.finalize()

    model = provider.provide(pre_process=True, post_process=True)
    return model.to(device=device, dtype=torch.bfloat16), provider


def diffusion_attention_modules(model):
    return [m for m in model.modules() if hasattr(m, "set_asymmetric_ar_metadata")]


def apply_variant(modules, variant, args, noisy, clean, device):
    """Route every attention module onto the arm under test.

    Dispatch in the shipped `forward` is: metadata set -> the asymmetric flex
    path; metadata cleared -> the symmetric path. So clearing the metadata
    selects `symmetric_*`, and which symmetric KERNEL runs is decided earlier by
    whether `install_symmetric_flex` was applied.
    """
    if variant != "asymmetric":
        for m in modules:
            m.clear_asymmetric_ar_metadata()
            # Force a rebuild at this arm's shape rather than trusting the shape
            # check to notice: the cached bias is [1,1,S,S] and is the single
            # largest tensor in the run.
            m._sbd_bias = None
        return

    batch = args.batch
    prompt_len = clean - noisy
    for m in modules:
        m.set_asymmetric_ar_metadata(
            noisy_length=noisy,
            clean_length=clean,
            noisy_response_offset=0,
            prompt_lengths=torch.full((batch,), prompt_len, dtype=torch.long, device=device),
            response_lengths=torch.full((batch,), noisy, dtype=torch.long, device=device),
            noisy_valid_lengths=torch.full((batch,), noisy, dtype=torch.long, device=device),
            clean_lengths=torch.full((batch,), clean, dtype=torch.long, device=device),
        )


def main() -> None:
    args = parse_args()
    from megatron.core import parallel_state, tensor_parallel

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    assert args.tp == world, f"tp {args.tp} must equal world size {world} (cp=1, dp=1)"

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp, context_parallel_size=1
    )
    # Megatron's TP weight init forks the 'model-parallel-rng' tracker and
    # raises without this, even though the bench ignores weight VALUES.
    tensor_parallel.model_parallel_cuda_manual_seed(1234)

    def log(msg):
        if rank == 0:
            print(f"[symasym] {msg}", flush=True)

    noisy, clean, seq = layout_for(args.variant, args.prompt, args.response)
    max_seq = 2 * (args.prompt + args.response)

    from megatron.bridge.diffusion.models.common import (
        nemotron_labs_diffusion_attention as attn_mod,
    )
    if args.variant == "symmetric_flex":
        install_symmetric_flex(attn_mod)
        log("symmetric path REROUTED to flex_attention (pre-ba1c63e9 forward)")

    model, provider = build_model(args, device, max_seq)
    block_size = getattr(provider, "block_size", None) or 32
    if noisy is not None:
        assert noisy % (2 * block_size) == 0, (
            f"noisy {noisy} must divide 2*block_size = {2 * block_size}"
        )
    assert seq % args.tp == 0, f"seq {seq} must divide tp {args.tp} for sequence parallel"

    modules = diffusion_attention_modules(model)
    assert modules, "no NemotronLabsDiffusionAttention modules found"
    apply_variant(modules, args.variant, args, noisy, clean, device)
    if args.variant == "symmetric_te":
        canonicalize_te_inputs(modules)
        log("TE q/k/v stride canonicalization ENABLED (size-1 KV head dim at tp=8)")

    log(
        f"variant={args.variant} tp={args.tp} cp=1 | P={args.prompt} R={args.response} "
        f"| noisy={noisy} clean={clean} attn_seq={seq} batch={args.batch} "
        f"layers={provider.num_layers} block_size={block_size} "
        f"sp={provider.sequence_parallel} "
        f"flex_mode={os.environ.get('DIFFU_FLEX_COMPILE_MODE', '<default>')}"
    )

    vocab = getattr(model.config, "vocab_size", 131072)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    input_ids = torch.randint(1000, min(vocab, 131000), (args.batch, seq), generator=gen).to(device)
    position_ids = torch.arange(seq).unsqueeze(0).expand(args.batch, seq).contiguous().to(device)

    def one_step():
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        # Any scalar with a full-graph dependency works: the bench measures the
        # backward's cost, not its value.
        out.float().pow(2).mean().backward()

    try:
        for _ in range(args.warmup):
            one_step()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        static = torch.cuda.memory_allocated()
        times = []
        for _ in range(args.iters):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            one_step()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated()
    except torch.cuda.OutOfMemoryError as exc:
        # Reported, not raised: the sweep runs one variant per process and the
        # symmetric_te arm is EXPECTED to OOM first (its bias is O(S^2) and does
        # not shard with tp). A per-rank traceback would bury the arms that fit.
        log(f"RESULT variant={args.variant} P={args.prompt} R={args.response} seq={seq} OOM")
        log(f"  {str(exc).splitlines()[0]}")
        dist.destroy_process_group()
        os._exit(0)

    times.sort()
    median_ms = times[len(times) // 2]
    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    stats = torch.tensor(
        [median_ms, mean_ms, std_ms, peak / 2**30, static / 2**30],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    log(
        f"RESULT variant={args.variant} P={args.prompt} R={args.response} seq={seq} "
        f"median_ms={stats[0].item():.1f} mean_ms={stats[1].item():.1f} "
        f"std_ms={stats[2].item():.1f} peak_gib={stats[3].item():.3f} "
        f"static_gib={stats[4].item():.3f}"
    )
    log(f"  rank0 iters: {' '.join(f'{t:.0f}' for t in times)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
