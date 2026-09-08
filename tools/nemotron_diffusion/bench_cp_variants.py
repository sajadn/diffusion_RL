# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Fwd/bwd bench for the context-parallel variants in section 2 of
`plans/full_cp_analysis.md`.

  v0        symmetric,  1 global zigzag,   gather Q AND K/V (naive baseline)
  v1        symmetric,  1 global zigzag,   gather ALL K/V, Q local
  v2        symmetric,  1 global zigzag,   gather CLEAN K/V only, Q local
  v3        asymmetric, segmented zigzag,  gather CLEAN K/V only  (shipped block-aware)
  ref       asymmetric, 1 global zigzag,   gather ALL K/V         (shipped local-Q)

v0/v1/v2 come from `cp_variant_paths.py` (monkeypatch; the shipped symmetric path
has no local-Q CP branch). v3/ref are the real bridge code, selected by env var.

`--check` runs v0, v1 and v2 in one process on identical inputs and compares
each against v1. All three compute the same attention over the same layout and
differ only in what is gathered, so a mismatch localizes the fault: v2 means
its noisy-locality assumption is broken, v0 means the gather/scatter round trip
is.
"""

import argparse
import datetime
import os

import torch
import torch.distributed as dist

DEFAULT_MODEL = ("/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/"
                 "Nemotron-Labs-Diffusion-8B-v1.5-sft")
VARIANTS = ("v0", "v1", "v2", "v3", "ref")
SYMMETRIC = ("v0", "v1", "v2")



# ---------------------------------------------------------------------------
# Shared RoPE tables (measurement only -- see --share-rope)
# ---------------------------------------------------------------------------
# Each attention layer owns its own Ministral3RotaryEmbedding and rebuilds
# cos/sin from scratch, so an L-layer model materializes L identical tables and
# autograd retains every one (`x * cos` saves cos even though the table is
# built under no_grad). This memoizes them so all layers share one.
#
# Keyed on (shape, dtype), which is sound HERE and not in general: one variant
# runs per process against one fixed input, so every layer feeds identical
# position ids, and the q/k tensors differ in length wherever their values
# differ. In production the values vary per microbatch at a FIXED shape
# (per-sample prompt offsets), so a shape key would silently rotate Q at the
# wrong positions -- which is why this lives in the bench and not the bridge.
# `verify=True` re-checks the ids on every hit and turns that assumption into
# an assertion.
_ROPE_CACHE = {}


def enable_rope_sharing(attn_mod, verify: bool = False):
    orig = attn_mod.Ministral3RotaryEmbedding.forward

    @torch.no_grad()
    def cached(self, x, position_ids):
        key = (tuple(position_ids.shape), x.dtype)
        hit = _ROPE_CACHE.get(key)
        if hit is not None:
            cos, sin, ids = hit
            if verify and not torch.equal(ids, position_ids):
                raise RuntimeError(
                    f"RoPE cache collision at shape {tuple(position_ids.shape)}: "
                    "same shape, different position ids -- sharing would rotate "
                    "these rows at the wrong positions.")
            return cos, sin
        cos, sin = orig(self, x, position_ids)
        _ROPE_CACHE[key] = (cos, sin, position_ids.clone() if verify else None)
        return cos, sin

    attn_mod.Ministral3RotaryEmbedding.forward = cached
    return orig


def disable_rope_sharing(attn_mod, orig):
    attn_mod.Ministral3RotaryEmbedding.forward = orig
    _ROPE_CACHE.clear()


def seq_of(variant, P, R):
    """Attention sequence length. Symmetric doubles P+R; asymmetric doubles R."""
    return 2 * (P + R) if variant in SYMMETRIC else P + 2 * R


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--cp", type=int, required=True)
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--check", action="store_true", help="v1 vs v2 equivalence, then exit")
    p.add_argument("--share-rope", action="store_true",
                   help="share one cos/sin table across all layers")
    p.add_argument("--check-rope", action="store_true",
                   help="run the variant with and without RoPE sharing and "
                        "compare outputs; exits after")
    p.add_argument("--trace-alloc", action="store_true",
                   help="snapshot every LIVE allocation at end of forward with "
                        "its allocating python frame; the definitive answer to "
                        "what is alive that saved_tensors_hooks does not see")
    p.add_argument("--breakdown", action="store_true",
                   help="walk the autograd graph after the forward and sum every "
                        "retained tensor by the op that saved it; diffing two "
                        "variants localizes memory the K/V formula does not model")
    p.add_argument("--measure-kv", action="store_true",
                   help="sum the actual K/V bytes handed to flex per forward, "
                        "so the formula can be checked against tensors rather "
                        "than inferred from total memory")
    p.add_argument("--count-masks", action="store_true",
                   help="count BlockMask builds per timed step (should be 0 after warmup)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", type=int, required=True)
    p.add_argument("--response", type=int, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--layers", type=int, default=0)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--recompute", action="store_true",
                   help="full activation checkpointing, as production runs it "
                        "(recompute_granularity=full, method=uniform, num_layers=1 "
                        "-- nemo_rl/models/megatron/setup.py:508)")
    return p.parse_args()


def build(args, device, seq_length):
    from megatron.bridge import AutoBridge
    bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
    pr = bridge.to_megatron_provider(load_weights=False)
    pr.tensor_model_parallel_size = args.tp
    pr.context_parallel_size = args.cp
    pr.pipeline_model_parallel_size = 1
    pr.sequence_parallel = True
    pr.bf16 = True
    pr.params_dtype = torch.bfloat16
    pr.seq_length = seq_length
    pr.gradient_accumulation_fusion = False
    if args.recompute:
        # Matches _apply_performance_config in nemo_rl. Under full recompute the
        # gathered K/V is a per-layer TRANSIENT rather than a 34-layer sum, so the
        # variants' memory gap should shrink -- but the forward (including the
        # all-gather) re-runs in the backward, so their comm gap should grow.
        pr.recompute_granularity = "full"
        pr.recompute_method = "uniform"
        pr.recompute_num_layers = 1
    if args.layers:
        pr.num_layers = args.layers
    if hasattr(pr, "finalize"):
        pr.finalize()
    return pr.provide(pre_process=True, post_process=True).to(device, torch.bfloat16), pr


def shard(tensor, variant, P, R, cp_rank, cp_size):
    """Shard [b, S] to match the variant's zigzag."""
    from megatron.bridge.diffusion.common.cp_utils import (
        segmented_zigzag_slice, zigzag_slice)
    if cp_size == 1:
        return tensor
    if variant == "v3":                      # segmented: (noisy, clean)
        return segmented_zigzag_slice(tensor, (R, P + R), cp_rank, cp_size, seq_dim=1)
    return zigzag_slice(tensor, cp_rank, cp_size, seq_dim=1)   # v1/v2/ref


def set_asym_metadata(mods, P, R, batch, device):
    # Build ONCE and hand the same dict to every layer, as the policy worker
    # does. The keyword form is still accepted but gives each layer its own
    # metadata and its own RoPE container, which is correct yet forfeits the
    # sharing this bench is meant to measure.
    meta = type(mods[0]).build_asymmetric_ar_metadata(
        noisy_length=R, clean_length=P + R, noisy_response_offset=0,
        prompt_lengths=torch.full((batch,), P, dtype=torch.long, device=device),
        response_lengths=torch.full((batch,), R, dtype=torch.long, device=device),
        noisy_valid_lengths=torch.full((batch,), R, dtype=torch.long, device=device),
        clean_lengths=torch.full((batch,), P + R, dtype=torch.long, device=device),
    )
    for m in mods:
        m.set_asymmetric_ar_metadata(meta)


def configure(variant, attn_mod, pristine_forward, mods, args, device):
    """Put every attention module on the variant's path."""
    from tools.nemotron_diffusion.cp_variant_paths import (
        install_symmetric_cp, reset_variant_mask_cache)
    cls = attn_mod.NemotronLabsDiffusionAttention
    cls.forward = pristine_forward                      # undo any prior install
    reset_variant_mask_cache()                          # and drop the previous mask
    if variant in SYMMETRIC:
        for m in mods:
            m.clear_asymmetric_ar_metadata()
        install_symmetric_cp(attn_mod, clean_only=(variant == "v2"),
                             gather_q=(variant == "v0"))
        attn_mod._CP_LOCAL_Q, attn_mod._CP_BLOCK_AWARE = False, False
    else:
        set_asym_metadata(mods, args.prompt, args.response, args.batch, device)
        attn_mod._CP_LOCAL_Q = True
        attn_mod._CP_BLOCK_AWARE = (variant == "v3")


def main():
    args = parse_args()
    import sys
    sys.path.insert(0, "/home/snorouzi/diffusion_RL/RL")
    from megatron.core import parallel_state, tensor_parallel
    from megatron.bridge.diffusion.models.common import (
        nemotron_labs_diffusion_attention as attn_mod)

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    device = torch.device("cuda", torch.cuda.current_device())
    assert args.tp * args.cp == world, f"tp*cp must equal world {world}"
    parallel_state.initialize_model_parallel(args.tp, context_parallel_size=args.cp)
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    cp_rank = parallel_state.get_context_parallel_rank()
    log = (lambda m: print(f"[cpvar] {m}", flush=True)) if rank == 0 else (lambda m: None)

    P, R = args.prompt, args.response
    todo = ["v0", "v1", "v2"] if args.check else [args.variant]
    assert todo[0] is not None, "--variant required unless --check"
    max_seq = max(seq_of(v, P, R) for v in VARIANTS)

    model, pr = build(args, device, max_seq)
    bs = getattr(pr, "block_size", None) or 32
    mods = [m for m in model.modules() if hasattr(m, "set_asymmetric_ar_metadata")]
    pristine_forward = attn_mod.NemotronLabsDiffusionAttention.forward
    log(f"tp={args.tp} cp={args.cp} P={P} R={R} layers={pr.num_layers} "
        f"block_size={bs} recompute={'full' if args.recompute else 'off'} "
        f"share_rope={args.share_rope}")

    # divisibility, per variant family
    if args.cp > 1:
        assert (P + R) % (args.cp * bs) == 0, (
            f"symmetric CP needs (P+R)={P+R} divisible by cp*block_size={args.cp*bs}")
        assert R % (2 * args.cp * bs) == 0, (
            f"v3 needs R={R} divisible by 2*cp*block_size={2*args.cp*bs}")
        assert (P + R) % (2 * args.cp) == 0

    results = {}
    for variant in todo:
        seq = seq_of(variant, P, R)
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(1000, 100000, (args.batch, seq), generator=g)
        pos = torch.arange(seq).unsqueeze(0).expand(args.batch, seq).contiguous()
        ids_l = shard(ids, variant, P, R, cp_rank, args.cp).to(device)
        pos_l = shard(pos, variant, P, R, cp_rank, args.cp).to(device)
        configure(variant, attn_mod, pristine_forward, mods, args, device)
        rope_orig = enable_rope_sharing(attn_mod, verify=args.check_rope) \
            if (args.share_rope or args.check_rope) else None

        post_fwd = {"gib": 0.0}
        if args.trace_alloc:
            torch.cuda.memory._record_memory_history(
                enabled="all", stacks="python", max_entries=200000)
        kvb = {"bytes": 0, "n": 0, "mask": 0}
        if args.measure_kv:
            _flex = attn_mod.fused_flex_attention

            def _measured(q, k, v, **kw):
                kvb["bytes"] += k.numel() * k.element_size() + v.numel() * v.element_size()
                kvb["n"] += 1
                bm = kw.get("block_mask")
                if bm is not None and kvb["n"] == 1:
                    kvb["mask"] = sum(
                        t.numel() * t.element_size()
                        for t in vars(bm).values() if torch.is_tensor(t))
                return _flex(q, k, v, **kw)
            attn_mod.fused_flex_attention = _measured
            import tools.nemotron_diffusion.cp_variant_paths as _cvp_kv
            _cvp_kv.fused_flex_attention = _measured

        def _fwd():
            return model(input_ids=ids_l, position_ids=pos_l, attention_mask=None)

        def one_step():
            model.zero_grad(set_to_none=True)
            if args.breakdown and post_fwd.get("shapes") is None:
                # saved_tensors_hooks sees EVERY tensor stashed for backward,
                # including ones held by compiled regions and custom
                # autograd.Functions -- a grad_fn walk reading _saved_* misses
                # both and comes back empty. Identity hooks, so nothing changes
                # but the accounting. Deduplicated by storage so aliases and
                # views are counted once.
                rec, seen = {}, set()

                def _pack(t):
                    if t.is_cuda:
                        st = t.untyped_storage()
                        if st.data_ptr() not in seen:
                            seen.add(st.data_ptr())
                            k = tuple(t.shape)
                            rec[k] = rec.get(k, 0) + st.nbytes()
                    return t

                with torch.autograd.graph.saved_tensors_hooks(_pack, lambda t: t):
                    out = _fwd()
                post_fwd["shapes"] = rec
            else:
                out = _fwd()
            # RETAINED state with the graph complete and no backward transients
            # in flight. Peak is a max over the whole step, so it cannot separate
            # what is held from what is momentarily allocated; this can.
            post_fwd["gib"] = torch.cuda.memory_allocated() / 2**30
            if args.trace_alloc and post_fwd.get("live") is None:
                snap = torch.cuda.memory._snapshot()
                live = {}
                for seg in snap["segments"]:
                    for blk in seg["blocks"]:
                        if blk["state"] != "active_allocated":
                            continue
                        frames = blk.get("frames") or []
                        where = "?"
                        for fr in frames:                      # innermost non-torch frame
                            fn = fr.get("filename", "")
                            if "/torch/" not in fn and "site-packages" not in fn:
                                where = f"{fn.split(chr(47))[-1]}:{fr.get('line')} {fr.get('name','')}"
                                break
                        else:
                            if frames:
                                fr = frames[0]
                                where = f"{fr.get('filename','?').split(chr(47))[-1]}:{fr.get('line')} {fr.get('name','')}"
                        agg = live.setdefault(where, [0, 0, 0])
                        agg[0] += blk["size"]; agg[1] += 1
                        # block size vs what the tensor actually asked for.
                        # The allocator refuses to split a large block when the
                        # remainder would be tiny, so it hands over the whole
                        # block and charges memory_allocated for all of it.
                        agg[2] += blk.get("requested_size", blk["size"])
                post_fwd["live"] = live
            if args.measure_kv and kvb["n"] and post_fwd.get("kv_locked") is None:
                post_fwd["kv_locked"] = kvb["bytes"]
            out.float().pow(2).mean().backward()
            return out

        if args.check_rope:
            model.eval()
            with torch.no_grad():
                shared = model(input_ids=ids_l, position_ids=pos_l,
                               attention_mask=None).detach().float().clone()
            disable_rope_sharing(attn_mod, rope_orig)
            with torch.no_grad():
                plain = model(input_ids=ids_l, position_ids=pos_l,
                              attention_mask=None).detach().float().clone()
            d = (shared - plain).abs().max().item()
            log(f"ROPE-CHECK variant={variant} max_abs_diff={d:.3e} "
                f"bit_identical={torch.equal(shared, plain)} "
                f"tables_cached={len(_ROPE_CACHE)}")
            log("ROPE-CHECK " + ("PASS" if d == 0 else f"FAIL (diff {d:.3e})"))
            continue

        if args.check:
            model.eval()
            with torch.no_grad():
                results[variant] = model(input_ids=ids_l, position_ids=pos_l,
                                         attention_mask=None).detach().float().clone()
            log(f"{variant}: forward ok, seq={seq}")
            continue

        counter = {"n": 0}
        if args.count_masks:
            from megatron.bridge.diffusion.common import dllm as _dllm
            _orig_build = _dllm._build_block_mask

            def _counting(*a, **k):
                counter["n"] += 1
                return _orig_build(*a, **k)
            _dllm._build_block_mask = _counting
            # v0/v1/v2 build through _cached_mask, whose _build_block_mask is a
            # closure local; count its misses instead.
            import tools.nemotron_diffusion.cp_variant_paths as _cvp
            _cvp.MASK_BUILDS[0] = 0

        try:
            for _ in range(args.warmup):
                one_step()
            torch.cuda.synchronize()
            if args.count_masks:
                import tools.nemotron_diffusion.cp_variant_paths as _cvp
                warm = counter["n"] + _cvp.MASK_BUILDS[0]
                counter["n"] = 0
                _cvp.MASK_BUILDS[0] = 0
            torch.cuda.reset_peak_memory_stats()
            static = torch.cuda.memory_allocated()
            times = []
            for _ in range(args.iters):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); one_step(); e.record(); torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            peak = torch.cuda.max_memory_allocated()
            if args.count_masks:
                import tools.nemotron_diffusion.cp_variant_paths as _cvp2
                timed = counter["n"] + _cvp2.MASK_BUILDS[0]
                log(f"MASKS variant={variant} builds_during_warmup={warm} "
                    f"builds_during_{args.iters}_timed_iters={timed} "
                    f"(layers={pr.num_layers}; nonzero here means the timing is inflated)")
        except torch.cuda.OutOfMemoryError:
            log(f"RESULT variant={variant} tp={args.tp} cp={args.cp} P={P} R={R} seq={seq} OOM")
            dist.destroy_process_group(); os._exit(0)

        times.sort()
        stats = torch.tensor([times[len(times)//2], sum(times)/len(times),
                              peak/2**30, static/2**30, post_fwd["gib"]],
                             device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        log(f"RESULT variant={variant} tp={args.tp} cp={args.cp} P={P} R={R} seq={seq} "
            f"median_ms={stats[0].item():.1f} mean_ms={stats[1].item():.1f} "
            f"peak_gib={stats[2].item():.3f} static_gib={stats[3].item():.3f} "
            f"post_fwd_gib={stats[4].item():.3f}")
        if args.trace_alloc and post_fwd.get("live"):
            live = post_fwd["live"]
            tot = sum(v[0] for v in live.values())
            log(f"LIVEALLOC variant={variant} total={tot/2**30:.4f} GiB "
                f"sites={len(live)}")
            waste = sum(v[0] - v[2] for v in live.values())
            log(f"  ALLOC-WASTE variant={variant} block-minus-requested="
                f"{waste/2**20:.2f} MiB over all live blocks")
            for wh, (b, n, rq) in sorted(live.items(), key=lambda kv: -kv[1][0])[:18]:
                w = b - rq
                log(f"  LA {variant:4s} {b/2**20:10.2f} MiB  req={rq/2**20:9.2f}  "
                    f"waste={w/2**20:+7.2f}  n={n:<5d} {wh}")
        if args.breakdown and post_fwd.get("shapes"):
            sh = post_fwd["shapes"]
            tot = sum(sh.values())
            log(f"BREAKDOWN variant={variant} total_saved={tot/2**30:.4f} GiB "
                f"over {len(sh)} distinct shapes")
            for k, b in sorted(sh.items(), key=lambda kv: -kv[1])[:16]:
                log(f"  BD {variant:4s} {str(k):<30s} {b/2**20:10.2f} MiB")
        if args.measure_kv:
            log(f"KV variant={variant} kv_bytes_per_forward={post_fwd.get('kv_locked', 0)} "
                f"({post_fwd.get('kv_locked', 0)/2**30:.4f} GiB) flex_calls={kvb['n']} "
                f"blockmask_bytes={kvb['mask']}")

    if args.check:
        base = results["v1"]
        ok = True
        for other in ("v0", "v2"):
            d = (base - results[other]).abs().max().item()
            same = torch.equal(base, results[other])
            ok &= same
            log(f"CHECK v1 vs {other}: shapes {tuple(base.shape)} / "
                f"{tuple(results[other].shape)}  max_abs_diff={d:.3e}  "
                f"bit_identical={same}")
        log("CHECK " + ("PASS" if ok else "FAIL"))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
