# Corrected confidence-threshold Trace GRPO

This commit preserves the synchronous confidence-0.9 experiment and its
subsequent rollout/replay discrepancy filtering and importance-weighting arms.
It does not establish that the correction prevents collapse. The paused
Megatron generation prototype is not included.

## Transition likelihood and loss

`nemo_rl/algorithms/confidence_transition.py` marginalizes the proposals at
positions that stayed masked. It includes the probabilities of revealed tokens
and the probability mass of compatible rejected proposals. If no proposal
passes the threshold, the single revealed token is the highest-confidence
sampled proposal; rejection cutoffs then depend on that winner's probability,
with leftmost tie breaking. Invisible MASK proposals are marginalized too.

The correction replaces the token-only actor gradient with a full-transition
score-function gradient. Existing token KL regularization remains separate.
The log-likelihood derivative assumes unchanged threshold/ranking membership;
this is not a corrected off-policy PPO ratio. The provided configurations use
T=1, full vocabulary, complete 16-token blocks, exhaustive trace replay,
TP=CP=1, synchronous collection, and one optimizer update per rollout.
`validate_correction_config` checks these restrictions at setup.

A transition unsupported by the replay logits receives zero actor gradient
and is counted in `confidence_zero_support_steps`. Other supported transitions
in the same response can still contribute unless a response-level filter is
active. Complete-trace validation rejects truncated blocks and missing
no-progress steps. A valid all-zero reveal-step trace is accepted; the vLLM
workers check that requested reveal metadata is actually returned.

## Configurations

All filenames below are under `examples/configs/`:

- `grpo_sudoku6x6_trace_conf09_corrected_sync_20260910.yaml`: enables the
  transition correction using `confidence_transition_correction: true`, with
  no positive anchor and full signed advantages.
- `grpo_sudoku6x6_trace_conf09_corrected_filter_20260911.yaml`: inherits the
  correction baseline with the original 100-step filtering settings; retains
  a response only when all
  its transitions have support, unambiguous record matching, and absolute
  rollout/replay log-probability error at most `log(4)` per transition.
- `grpo_sudoku6x6_trace_conf09_corrected_importance_20260911.yaml`: inherits the
  correction baseline with the original 100-step importance-weighting settings;
  uses the replay/generation
  full-trajectory probability ratio capped at 2, and zero weight for unsupported
  or ambiguous responses.

For the filtering continuation, pass `grpo.max_num_steps=1000` with the filtering
config. Resume with its original checkpoint directory and nine-node topology.

Filtering and weighting act on the corrected actor gradient. Group advantages
are computed from the original reward groups and are not recomputed after
filtering. Logging reports retained signs, effective sample size, and support.
The base correction has no dependency on `confidence_experiment` when that
optional config section is absent, even if an experiment directory remains in
the environment. Experiment startup and weighting require both an active
`type: trace_grpo` estimator and `confidence_transition_correction: true`.

## Runtime helpers

The filtering/weighting arms additionally need rollout transition records.
Use `tools/nemotron_diffusion/run_confidence_experiment.py` as `RUN_SCRIPT` with
the normal cluster launcher, and export `NRL_VLLM_PY_EXECUTABLE` as the absolute
path to `tools/nemotron_diffusion/python-vllm-confidence`. Both require `RUN_NAME`.
The wrapper keeps the established DFW vLLM runtime and sets the repository and
helper import paths. The driver configures the Megatron worker environment.
`NRL_CONFIDENCE_EXPERIMENT_DIR` may override the shared record directory; use
a separate directory per run. The recorder currently assumes mask ID 100,
block size 16, threshold 0.9, and T=1; the entrypoint checks these settings.

The separate custom vLLM repo needs `patches/vllm_confidence_hooks.patch`,
applied after `patches/vllm_entropy_selection.patch` on its documented base:

```bash
git apply --check /path/to/RL-posanchor/tools/nemotron_diffusion/patches/vllm_confidence_hooks.patch
git apply /path/to/RL-posanchor/tools/nemotron_diffusion/patches/vllm_confidence_hooks.patch
```

The existing experiment runtime already contains both patches. Do not apply
them twice. The hook patch now includes separate sampler capture flags; a
checkout with the older environment-only hook patch must replace that patch
before using this version of the training integration. No decoder probability
calculation is changed by the hook patch; it records the logits and canvas
transitions needed by the mitigation arms.

`confidence_audit_hooks.py` is an optional diagnostic recorder for an active
corrected Trace config, requested with `NRL_CONFIDENCE_AUDIT_DIR` in both rollout
and Megatron worker environments.
Include this directory and the repository root on both workers' `PYTHONPATH`.
It captures paired transition probabilities and top-token values for the
confidence-0.9 audit; it does not change the objective.

The training setup derives `record_confidence_experiment` and
`record_confidence_audit` in the vLLM diffusion config independently. Directory
variables never activate sampler imports by themselves. Inactive estimators
remove inherited capture flags; ordinary vLLM configs gain no custom fields.
Separate validation engines disable capture. Shared-engine capture skips
validation policy/temperature changes outside confidence-0.9/T=1 and skips
per-request greedy rows.

## Verification

From the repository, in the established environment:

```bash
uv run --no-sync python -m pytest --confcutdir=tests/unit/algorithms \
  tests/unit/algorithms/test_confidence_config.py \
  tests/unit/algorithms/test_confidence_transition.py \
  tests/unit/algorithms/test_confidence_experiment.py \
  tests/unit/algorithms/test_trace_first_step_reveal.py -o addopts= -q
```

Tests enumerate transition probabilities and check normalization, finite-
difference gradients, fallback ties, unsupported events, post-EOS actor
replacement, actor weights, record alignment, and first-step-only reveals.
Isolation tests also cover omitted/null estimators, stale diagnostic variables,
inherited capture flags, and independent experiment/audit activation. The vLLM
patch includes hook-import and validation-switch tests. The full-vocabulary
CUDA backward test is skipped when no GPU is available.
