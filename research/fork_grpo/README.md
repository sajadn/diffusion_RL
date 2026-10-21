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
reward group. The policy loss is averaged over nonempty responses. All shards
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

The single recipe trains Nemotron-3B on DeepScaleR and validates on AIME2024,
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
