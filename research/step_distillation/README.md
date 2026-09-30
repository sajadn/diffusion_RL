# Diffusion step distillation

Run `research/step_distillation/run_context_distillation.py`, or pass
`configs/recipes/step_distillation-deepscaler-3b-4n8g-megatron-vllm-positive-kd-long.yaml`
to this entrypoint. The recipe uses DeepScaleR JSONLs with an explicit held-out
split; MATH-500 remains evaluation-only.

Generate eight stochastic current-student responses for each of 128 prompts.
Each 16-token block uses four fixed-count entropy-ranked reveal steps. Retain
responses with reward strictly above min_reward (default zero), then sample
one eligible recorded state per retained response. The original frozen teacher
makes one forward pass on the next recorded student state in the same block.
It generates no new tokens.

Optimize full-vocabulary forward KL(teacher || student) on positions still
masked in both views, before the response terminal boundary. Teacher-visible
tokens supply context and receive no loss. Completed teacher blocks have no
targets and cannot be selected. Reward filters samples; there is no GRPO, CE,
or entropy-regularization objective. A globally empty eligible batch skips the
optimizer and scheduler. Student generation weights continue to refit normally.

`step_distillation` enables this mode. Its supported target_source
is less_masked_kd; positive_only must be true. It requires synchronous,
colocated BF16 reference-vLLM generation, TP=PP=CP=1, and a fixed-count entropy
schedule. Full teacher logits are retained on CPU only for the chosen block;
normalization is FP32 at the training temperature. The teacher uses the native
reference weights loaded from the original checkpoint.

Student checkpoints save every 10 updates. The recipe disables inline validation;
evaluate saved students using the external MATH-500 step sweep. That evaluation
pipeline must be attached by the cluster submission wrapper. Checkpoints omit
optimizer state and do not support exact training resumes.

The algorithm, frozen teacher worker, configuration, entrypoint, and tests are
owned by this folder. Shared recorded-trajectory layouts, rollout generation,
and the native training controller are reused from JustGRPO and NeMo-RL.

From the repository root, using the existing training environment:

```bash
export PYTHONPATH="$PWD/research/step_distillation:$PWD/research/just_grpo:$PYTHONPATH"
uv run --no-sync python research/step_distillation/run_context_distillation.py --help
```

Keep both source roots available to Ray policy workers as well as the driver.
The configuration uses a top-level `step_distillation` block; `trace_grpo`
contains only the shared trajectory, generation, and sampling settings.

CPU tests in an existing environment (from the repository root):

```bash
PYTHONPATH=research/step_distillation:research/just_grpo:. \
  uv run --no-sync python -m pytest \
  research/step_distillation/tests/unit --noconftest -o addopts=""
```
