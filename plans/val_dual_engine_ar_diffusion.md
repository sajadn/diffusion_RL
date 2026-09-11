# Validating under AR *and* diffusion in one cycle

## Goal

One validation cycle reports two curves for a diffusion-LLM run: the rollout
decode (diffusion) and an AR reference. On a dLLM the sampler is part of the
policy, so a single curve cannot separate "the weights degraded" from "the
sampler stopped working on these weights". AR vs diffusion is fixed when the
vLLM engine loads (`vllm_kwargs.hf_overrides.architectures` selects the causal-LM
class and `diffusion_config: null` drops the canvas), so this needs two engine
groups, not a runtime reconfigure.

Target: the non-colocated **async**-GRPO path (the DeepScaleR/kt40 configs), not
just the colocated sync path.
Primary curve: the rollout-matching decode (diffusion for BJG-Fast runs), so
`val:accuracy` and `keep_top_k` selection keep their current meaning.

## What already exists

- One dedicated validation engine group: `vllm_val_dllm_overrides` carrying a
  `vllm_cfg` key -> `_maybe_init_val_generation_group` (`grpo.py:363`) builds a
  second `VllmGeneration` with `name_prefix="vllm_val"`, launched before the
  rollout group while GPU memory is clean, then slept; woken + refit around
  validation.
- Validation overrides deep-merge LAST onto a copy of the rollout generation
  config, so a group can flip `hf_overrides.architectures` and null out
  `diffusion_config` (`_deep_update` replaces non-dict values wholesale).
- N-way side-by-side reporting: `vllm_val_dllm_variants` -> `accuracy/<name>`,
  same prompts and budget for every variant, first entry primary.
- Refit transport is already group-keyed: `base_policy_worker.maybe_init_zmq`
  holds `zmq_sockets: dict[group, socket]` bound at
  `ipc:///tmp/{group}-{device}.sock`, and `VllmGeneration.prepare_refit_info`
  passes `generation_group=self.name_prefix`. Multiple groups do not collide.

## What is missing

1. Exactly one group can be built. `_build_val_generation_config` reads the
   single `overrides_key`; nothing consumes `variants_key` at build time.
2. Engine-level variants are actively forbidden: assert at `grpo.py:2889`,
   runtime skip at `grpo.py:2995`, docs at `vllm/config.py:105-108`.
3. `validate()` has no per-variant engine routing -- one `policy_generation`,
   one `generation_config`, both closed over by `_run_val_pass`. Sampling params
   must come from the variant's own config (a diffusion engine rejects any
   per-request temperature other than 0 or 1).
4. Wake/refit/sleep happens once in the caller (`grpo.py:1778-1801`), outside
   `validate()`. With two val groups only one may be awake at a time.
5. `grpo.py:614` requires colocated inference and `grpo.py:619` forbids async
   GRPO for any dedicated val group.
6. `_assert_val_group_shares_memory` reasons about one val group vs the rollout
   group; the budget has to span all groups on the device.

## Design

### Config

Let a `vllm_val_dllm_variants` entry carry `vllm_cfg`. The existing
`_val_overrides_need_server_group` predicate is already the right discriminator:

- entry WITHOUT `vllm_cfg` -> soft knobs, `reconfigure_dllm` on the rollout
  engines (today's behavior).
- entry WITH `vllm_cfg` -> its own group, built by deep-merging the entry onto a
  copy of the rollout generation config, `name_prefix = f"vllm_val_{name}"`.

`vllm_val_dllm_overrides` with `vllm_cfg` keeps working as the single-group form.

```yaml
vllm_val_dllm_variants:
  diffusion:                        # primary: matches rollout, fills val:accuracy
    vllm_kwargs:
      diffusion_config:
        selection_policy: "confidence_threshold"
        confidence_threshold: 0.9
        temperature: 1.0
  ar:                               # reference: own engine group
    temperature: 1.0
    vllm_cfg:
      gpu_memory_utilization: 0.25
    vllm_kwargs:
      diffusion_config: null
      hf_overrides:
        architectures: ["NemotronLabsDiffusionForCausalLM"]
```

### Code

1. `_build_val_generation_config(generation_config, overrides)` -- take overrides
   as an argument so it can be called per variant.
2. Replace `_maybe_init_val_generation_group` with one that returns
   `dict[str, GenerationInterface]` (empty when no variant needs a group),
   looping over engine-level variants plus the legacy single override.
3. `validate()` takes that mapping; per variant it resolves
   `(generation, gen_cfg)` and, for an engine-level variant, wakes + refits it,
   runs the pass, and sleeps it again -- only one group awake at a time.
4. Replace the asserts at `grpo.py:614/619` with the real requirement: before
   waking a val group the rollout engines must be asleep and, under async GRPO,
   the collector must be drained. `pause_and_drain` now genuinely guarantees the
   latter (commit 4a8ab7f51), which is what makes async support tractable.
5. `val_reconfigures_engine` in the async loop becomes "validation needs the
   collector quiesced" = reconfigure OR any dedicated val group.
6. `_assert_val_group_shares_memory` sums `gpu_memory_utilization` across the
   rollout group and every val group sharing the device.

## Verification

- Unit tests on the pure selection/build helpers (no ray): which variants get a
  group, name prefixes, deep-merge of `diffusion_config: null` and
  `hf_overrides.architectures`, memory-budget assertion.
- Toy sudoku run on the async vLLM path with both variants; check the run log
  shows two engine groups, both curves appear in wandb, and `val:accuracy`
  equals the diffusion variant's.

## Refit is the real blocker on the non-colocated path (found 2026-09-09)

The colocated path was already multi-group ready -- `stream_weights_via_ipc_zmq`
takes `generation_group` and `base_policy_worker.maybe_init_zmq` keys its
sockets by it. The NON-colocated path was not, and that, not GPU memory, is
what `grpo.py`'s `assert colocated_inference` was really protecting:

- `setup()` built ONE collective sized `train_world_size + inference_nodes *
  inference_gpus_per_node`, and only the rollout group joined it.
- Each policy worker stored a single `self.model_update_group`, used by all
  three worker classes (megatron, dtensor, dtensor_v2).

A second engine group therefore had no rank in that collective and could not be
refit. Fix shape (mirroring the zmq path rather than inventing a new one):
`init_collective(..., generation_group=...)` keeping
`model_update_groups: dict[str, StatelessProcessGroup]`, one collective per
group on its own port, `broadcast_weights_for_collective(..., generation_group=)`
threaded through `lm_policy` and all three worker classes.

Two things to get right, both learned the hard way:

1. **Bring the groups up one at a time.** The port finder can hand out the same
   free port twice if nothing has bound it yet, so the previous group's process
   group must be up before the next port is requested.
2. **Change the registration key and its consumer in the same edit.** Keying
   registration by `name_prefix` while `refit_policy_generation` still asked for
   `None` broke EVERY non-colocated run, not just multi-group ones (job
   18208658: `RuntimeError: No refit collective for generation group None ...
   known groups: ['vllm_policy']`). The grep to run before believing a change
   here is behavior-neutral: every `generation_group=` producer and consumer
   must use the same expression.

Still unverified: whether a vLLM engine's NCCL communicator survives a
sleep/wake cycle. Only one group is awake at a time, so every val-group refit
happens on a communicator created before that group was ever slept.

## Process note

Do not edit `nemo_rl/` while a job is running out of this worktree -- the run
imports from it, and a job was killed mid-flight this way. Park work in progress
with `git stash push nemo_rl/` before submitting a verification run.
