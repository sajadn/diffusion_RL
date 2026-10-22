# ForkGRPO

[Fork-dLLM / ForkGRPO](https://arxiv.org/html/2609.39859v1#S3) keeps
confidence-threshold parallel decoding and switches to the leftmost masked
position when no sampled token exceeds the threshold. Confidence uses the
untempered sampled-token probability; stored policy logprobs use the sampling
temperature. The sampler records an explicit `fallback_mask` and reveal steps.

ForkGRPO samples up to `schedule.num_level_samples` fallback decisions per
response block, uniformly without replacement. The DeepScaleR recipe uses three
views (`num_level_samples: 3`, positive integer) and `seed_base: 42` (integer).
Selections are made once on the driver using the seed, update number, and row,
then reused for policy scoring and training. Each view can replay a different
recorded level in each block. The shared asymmetric block attention supplies
clean earlier blocks. ForkGRPO directly uses `TraceGRPOSchedule`; no separate
Fork schedule is needed. Generation still permits 16 denoising steps per block.

Only sampled fallback tokens through the first stop token receive loss. All
recorded reveals, including unsampled fallbacks, parallel tokens, and retained
terminal context, reconstruct each selected decision's original masked state.
Three views means three scoring forwards and three training forward/backward
views per microbatch; activation checkpointing adds recomputation. It does not
imply an equivalent end-to-end speedup.

## Empty responses and normalization

Let `F_b` be a block's fallback count, `k_b = min(num_level_samples, F_b)`, and
`F` the response's total fallback count. A selected token has loss weight
`F_b / k_b / F`: inverse inclusion probability times the original response
normalization. Blocks with at most three fallbacks retain every decision; larger
blocks provide an unbiased estimate of the full fallback mean. Each nonempty
response still has total sampled loss weight one. Scoring uses unweighted
logprobs for the same selected tokens.

Responses without fallback tokens contribute zero policy loss and remain in the
reward group. The policy loss is averaged over nonempty responses. For block diffusion, all shards
execute the configured number of views, including zero-loss views. A globally
empty batch skips optimizer and scheduler updates, including Adam momentum and
weight decay.

## Run

From the repository root in a provisioned Megatron environment:

```bash
uv run --all-packages --extra mcore python research/fork_grpo/run_fork_grpo.py \
  policy.model_name=/path/to/Nemotron-Labs-Diffusion-3B \
  logger.log_dir=/path/to/fresh/results
```

The DeepScaleR recipe trains Nemotron-3B on DeepScaleR and validates on AIME2024,
using four nodes, 4K context, and 100 optimizer steps. It is an implementation baseline, not a reproduction
of the paper's LLaDA LoRA results. The recipe uses two training nodes and two
asynchronous vLLM generation nodes, with training microbatch 2 and logprob batch 4.
Set `NRL_VLLM_PY_EXECUTABLE` to the reference fork interpreter with explicit
`fallback_mask` output support. Megatron generation is also supported with
`runtime: megatron` and `selection_policy: fork`. Training uses the shared GRPO controller and
Megatron diffusion worker. The recipe disables the reference KL penalty.
`max_steps >= block_size` prevents forced parallel final-step commits.

For provisioned interpreters, add the repository, `research/block_diffusion`, and
`research/fork_grpo` and `research/trace_grpo` to `PYTHONPATH`.

For MATH-12K training and MATH-500 validation, use
`configs/recipes/fork_grpo-math12k-3b-4n8g-megatron-vllm-async-long.yaml`
and set `MATH12K_DATA_DIR` to the prepared `train.jsonl` and `test.jsonl` directory.
Both files need `problem` and `answer` fields. This recipe uses a 2,048-token
prompt limit, 6,144-token context, and the same 3,744-token generation budget.
It validates all 500 problems once each at temperature 0.6, at the start, every
10 updates, and at the end. Checkpoints are saved every 10 updates, with a
3-hour-30-minute save deadline for the four-hour allocation.

For matched Nemotron MATH-500 decoding comparisons, use
`configs/recipes/fork_grpo-math12k-3b-4n8g-megatron-vllm-async-multieval-long.yaml`.
It retains the three validation modes from
`trace_math12k_eb01_te1_ts06_sync_20261019`: entropy budget 0.1 with entropy
temperature 1.0, confidence threshold 0.9, and AR. It adds `diffusion_fork`
with confidence threshold 0.9 and leftmost fallback. All modes sample at 0.6,
use the same 3,744-token response budget, and validate all 500 problems.
The existing `MultiModeValidation` handles engine sleep, refits, and named metrics;
training retains the original asynchronous 2+2 node split and three replay views.
The runtime must support `entropy_temperature` as well as explicit Fork metadata.
Checkpoint ranking uses `val:accuracy/diffusion_entropy_budget` so earlier
Fork-only validation scores are not compared with the new primary decoder.

## LLaDA full-canvas replay

Use `configs/recipes/fork_grpo-math12k-llada8b-canvas256-4n8g-megatron-long.yaml`
with `MATH12K_DATA_DIR` set as above. This recipe uses LLaDA-8B, four nodes,
256 response tokens, temperature 1.0, confidence threshold 0.9, and training and
scoring microbatch size 2. Generation shares the Megatron training workers.
Validation matches the primary LLaDA JustGRPO `fixed1_t1` decoder: 32-token blocks,
one highest-confidence sampled token per step, temperature 1.0, and a 256-token
response canvas. Fork is used for training rollouts only.

For GSM8K, use
`configs/recipes/fork_grpo-gsm8k-llada8b-canvas256-4n8g-megatron-long.yaml`.
It uses the same training settings, the GSM8K train/test splits, a 768-token
prompt limit, and the same baseline validation decoder on all 1,319 test examples.
Only the primary baseline decoder is evaluated; additional baseline threshold
sweeps are not enabled.
Add `research/just_grpo` to `PYTHONPATH` for the shared LLaDA implementation.

LLaDA attends bidirectionally across the full canvas. Fork records globally
ordered reveal steps and the leftmost fallback flag. The existing
`TraceGRPOSchedule` reconstructs each selected decision with `FullCanvasLayout`;
the shared diffusion loss postprocessor applies the fallback mask.

The recipe caps replay at 64 views **per scoring or training phase**, independently
of generation. For a response with `F` eligible fallback decisions, sample
`min(F, 64)` uniformly without replacement and weight each by `1 / min(F, 64)`.
This estimates the full fallback mean without bias. The whole batch uses
`min(64, max(1, max(F)))` views, so every DP shard executes the same count; empty
rows and unused views have zero loss. Scoring and training reuse the selections.
Generation can still require up to 256 full-canvas forwards.

Relative to uncapped Fork, the cap helps only when a batch needs more than 64
fallback levels. Relative to 256-view LLaDA JustGRPO, 64 views reduces replay
views by 75%; this is not a prediction of total runtime savings. The driver logs
`views`, `fallback_mean`, and `fallback_max` for each prepared batch. Each replay
view still runs the full canvas. Activation checkpointing adds recomputation.

## Tests

```bash
uv run --package fork-grpo --group test python -m pytest \
  -c research/fork_grpo/pyproject.toml research/fork_grpo/tests/unit \
  --confcutdir=research
```

A two-GPU, three-update execution smoke is available at
`tests/functional/deepscaler.sh`, with `MODEL` and a fresh `OUTPUT_DIR` supplied by the
caller. GPU execution is required to validate real Megatron training; CPU tests
cover sampler ordering, temperature semantics, replay logprob parity, terminal
context, metadata transport, empty/mixed rows, sharding, and recipe validation.
