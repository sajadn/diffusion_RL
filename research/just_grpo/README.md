# Block JustGRPO

Research implementation on upstream NeMo-RL `main` (`612d5274059c821dc09c2c90767b546b6f2d50b5`), based on the [diffusion design document](https://docs.google.com/document/d/1HHzRpLfsRalQkqLGFe0vljZjRO7dwKaKWH43jZPxCu4/edit) and the [Flow GRPO research structure](https://github.com/triple-mu/RL/pull/1). The algorithm, 6x6 Sudoku generator, and few-shot prompt come from the Block JustGRPO reference implementation.

**Training uses Megatron only.** Generation can use the reference diffusion vLLM runtime or the in-process Megatron diffusion decoder. Both use the same driver, denoising schedule, loss, and default validation mode inherited from training. Reference vLLM also supports the named AR/diffusion validation modes described below. All experiment defaults live in YAML, inheriting `examples/configs/grpo_math_1B.yaml`.

## Upstream controller

`run_just_grpo.py` is the command-line entrypoint. Its `run(config)` function
validates JustGRPO settings and calls the shared diffusion driver. Shared runtime validation lives in `block_diffusion/validation.py`;
algorithm-specific configuration stays in `just_grpo/config.py`.

`block_diffusion/training.py` imports `setup()`, `grpo_train()`, and `async_grpo_train()` from `nemo_rl.algorithms.grpo`. Upstream owns Ray worker placement, data loading, rollout collection, GRPO advantages, validation, logging, and the training loop. Research code supplies the Sudoku data/environment and a `MegatronDiffusionGeneration` implementation of the existing `GenerationInterface`. vLLM uses upstream `VllmGeneration` directly. Setup constructs upstream `Policy` directly; the recipe selects `BlockJustGRPOPolicyWorker` through the existing `policy.worker_extension_cls_fqn` setting. The shared driver registers its runtime and passes the research configuration to the worker. For Megatron inference, the shared driver replaces the native generation wrapper after ordinary upstream setup and attaches a native colocated weight synchronizer. HTTP serving is disabled, so setup has not started an AR engine. Without named validation modes, vLLM retains its native setup path; multi-mode validation supplies a generation factory to upstream setup. Rollout logprobs are included in scoring inputs for Fast token selection. Previous-policy logprobs are always recomputed through upstream `Policy.get_logprobs()`.

## Diffusion training

Requires Megatron-Bridge asymmetric-attention support from [PR #6270](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6270) (upstream commit `baa58b55e17f`, or compatible backport `d61856208eed`).

`BlockJustGRPOSchedule` produces one denoising level at a time. Each view supplies clean token IDs, masked positions, selected targets, and their original rollout coordinates. `DiffusionMicrobatchProcessor` constructs `[noisy response | clean prompt + response]` inputs. Bridge receives per-sample prompt, response, and padding lengths through `build_asymmetric_ar_metadata` / `set_asymmetric_ar_metadata`. Noisy queries see their response-relative noisy block, the original prompt, and earlier clean response blocks; clean queries attend causally. RoPE maps response positions back to their original locations.

Prompts retain the exact tokenizer output, with no leading MASK tokens. Both AR and diffusion validation use those original prompts. Training blocks and Fast token selection start at the response boundary, independent of prompt length. Ungenerated positions in a trailing response block stay masked. Streaming loss normalization includes response position zero.

`BlockJustGRPOPolicyWorker` owns JustGRPO configuration, reveal schedules, and Fast token selection through `_build_schedule(data, purpose=...)`. It inherits `get_logprobs`, `get_reference_policy_logprobs`, `train`, and `generate` from `MegatronDiffusionPolicyWorkerImpl`. The shared worker owns diffusion input preparation, loss/logprob postprocessors, scoring and reference-weight handling, gradient accumulation, metrics, the generation-KL guard, and Megatron generation, while the upstream worker owns the model, optimizer, scheduler, reference weights, and forward/backward execution. Training uses:

1. `begin_train_step()` once per rollout batch.
2. `train_microbatch(level)` for each level from `DenoisingSchedule.iter_levels(data)`. This calls upstream `get_microbatch_iterator()`.
3. `finish_train_step()` once, normalizing with upstream's accumulated token-mask counts.

Each selected response token appears in exactly one level's loss mask. Gradients accumulate across levels before one reduction and optimizer update; source sample counts are corrected only for reporting. There is no expanded batch or custom iterator factory. Policy and reference scoring use `aggregate_diffusion_logprobs` to collect selected scores into the original `[batch, sequence]` layout. Reference weights remain loaded across the entire schedule. Scoring returns `logprobs` together with a boolean `logprob_token_mask` in rollout coordinates; coverage is collected from the scored views, including positions with a valid zero log-probability. Both synchronous and asynchronous GRPO diagnostics use this returned coverage. Diffusion sets `shift_labels=False` so diagnostics retain position zero; ordinary next-token callers retain their existing default.

`DiffusionLossPostProcessor` and `DiffusionLogprobsPostProcessor` share `selected_diffusion_logprobs`. The loss processor calls core `ClippedPGLossFn` with `shift_labels=False`; scores and metadata remain at the same positions without dummy-column padding. Temperature scaling and microbatch loss scaling remain in the upstream forward and postprocessor.

Supported training layouts are dense BF16 TP/DP with PP=CP=EP=1, unpacked fixed-width sequences, and synchronous gradient/parameter collectives. Sequence parallelism, PEFT, quantization, and dynamic batching are not supported by this research driver. Megatron generation shares live training weights; vLLM uses native weight synchronization.

## Generation

Shared generation code lives in `../block_diffusion/block_diffusion/generation/`. `megatron_generation.py` implements the existing `GenerationInterface` and the research denoising loop. Synchronous requests delegate to upstream `Policy.generate()`. Async requests await native Ray worker calls on TP=1 DP replicas, distributing inference microbatches across replicas and yielding rows with their original indices. This also handles single-row requests when DP exceeds the batch size. `MegatronDiffusionPolicyWorkerImpl` runs upstream `model_forward()` on its already-loaded model. Native `MegatronWeightSynchronizer` handles the colocated lifecycle, and `prepare_for_lp_inference()` restores model parameters, enters eval mode, and releases training buffers. Generation sees optimizer updates directly; there is no second model, checkpoint export, subprocess, or custom attention implementation.

The Megatron decoder uses uncached full forwards with the same asymmetric attention layout as training. Each current block starts fully masked and reveals positions from left to right. Sampling records temperature-scaled logprobs at commitment and stops independently per sample at EOS. Sampled MASK IDs count as committed tokens. Prompts are grouped by length, microbatched using `policy.logprob_batch_size`, and returned in their original order. This first implementation supports TP=1; KV caching and TP>1 decoding are deferred. It prioritizes matching the training forward, not inference throughput.

Use `policy.generation.backend: megatron`, `just_grpo.runtime: megatron`, and `generation_python: null` for this decoder. The driver constructs the research `GenerationInterface` after upstream setup; it does not initialize Megatron's AR inference engine. Colocated setup uses `refit_transport: mcore` and `refit_backend: gloo` to satisfy native refit configuration; weights are shared rather than transferred. Its recipe disables generation CUDA graphs. No vLLM/SGLang installation is needed for decoding.

`reference_vllm.py` extends native `VllmGenerationWorkerImpl` and `VllmAsyncGenerationWorkerImpl`, sharing the reference fork's temperature/logprob contract and diffusion context check. The recipe selects it with `policy.generation.worker_extension_cls_fqn`; upstream `VllmGeneration` owns worker placement, lifecycle, ports, output packing, and weight synchronization. The reference fork must support leftmost diffusion and sampled-token logprobs at commitment. Stock upstream vLLM/SGLang is not assumed to implement that contract; `upstream_vllm` and `upstream_sglang` remain rejected.

## Async collection

The Megatron decoder extends the native `MegatronGeneration` wrapper so upstream recognizes its colocated lifecycle. It implements `generate_async()` even with synchronous GRPO, because upstream selects async rollout dispatch for every Megatron generation backend. It yields completed microbatches without blocking the event loop and drains submitted calls before propagating an error or cancellation.

Set `grpo.async_grpo.enabled: true` to select upstream `async_grpo_train()` and its trajectory collector. The wrapper reports `blocks_training()` and `wake_carries_weight_updates()` as true. With `in_flight_weight_updates: false`, upstream drains pending rollouts before previous/reference scoring and training, then wakes the shared model and advances the collector's weight version. Generation and training alternate on the same model; this does not provide concurrent model execution. Importance-sampling correction is required. KV-cache recomputation, HTTP serving, and the async data plane are unsupported. Reference vLLM also supports upstream async GRPO using the native async engine and dedicated generation GPUs. Dedicated async configurations can allocate two training nodes and two generation nodes (32 GPUs total), with no in-flight weight updates. Training can overlap generation on the separate GPUs; refits drain pending requests first. Both runtimes must load the same NCCL version for dedicated weight synchronization. The DFW async vLLM submissions preload the existing vLLM runtime's NCCL library on both sides.

The retained Megatron recipes support async collection with the CLI override `grpo.async_grpo.enabled=true`. See `VERIFICATION.md` for GPU verification status.

## Recipes and launch

Four 32-GPU long-run recipes are published in `research/just_grpo/configs/recipes/`:

| Experiment | Recipe |
| --- | --- |
| Standard JustGRPO, TP=1/DP=32 | `just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml` |
| Fast JustGRPO, 25% per block, TP=1/DP=32 | `just_grpo-sudoku6x6-4n8g-megatron-inference-fast-long.yaml` |
| Async JustGRPO, TP=1/DP=32 | `just_grpo-sudoku6x6-4n8g-megatron-inference-async-long.yaml` |
| Fast Async JustGRPO, 25% per block, TP=1/DP=32 | `just_grpo-sudoku6x6-4n8g-megatron-inference-fast-async-long.yaml` |

The standard recipe inherits upstream `examples/configs/grpo_math_1B.yaml`. Fast inherits the standard recipe and overrides the token fraction and run name. Async inherits the same standard recipe and enables async GRPO with a separate log directory and run name. Fast Async inherits Async and overrides the token fraction and logging names. The functional smoke script applies two-GPU, three-update overrides to the standard recipe.

From the checkout root, with the NeMo-RL `mcore` environment:

```bash
export PYTHONPATH="$PWD:$PWD/research/block_diffusion:$PWD/research/just_grpo${PYTHONPATH:+:$PYTHONPATH}"
uv run --extra mcore python research/just_grpo/run_just_grpo.py \
  --config "$PWD/research/just_grpo/configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml" \
  policy.model_name=/path/to/Nemotron-Labs-Diffusion-3B \
  logger.log_dir=/path/to/fresh/run
```

For vLLM, `just_grpo.generation_python=/path/to/vllm/python` selects the native Ray generation worker interpreter; it must include the NeMo-RL worker dependencies as well as the reference fork. If omitted, the upstream vLLM actor environment is used. For Megatron generation, select its recipe and leave `just_grpo.generation_python` null. `tools/dfw_distributed.sbatch` handles both smoke and long runs; choose the allocation with `sbatch --nodes` and `--gpus-per-node`, matching the YAML cluster settings. Set `REPO_DIR`, `POLICY_PYTHON`, `MODEL`, `OUTPUT_DIR`, `CONFIG`, `CONTAINER_IMAGE`, and `CUDA_COMPAT_DIR`. `GENERATION_PYTHON` is optional. `POLICY_EXTRA_PYTHONPATH` supports a provisioned dependency overlay. The driver starts or connects to Ray; multi-node jobs use upstream `ray.sub`. Compilation caches use node-local storage. Native vLLM workers own port isolation. Megatron decoding executes inside the training workers.

`tests/functional/sudoku.sh` launches a two-GPU Ray smoke and checks native TensorBoard metrics for three updates, finite gradients, generation/training KL, timing metrics, and initial/final validation.

## Fast JustGRPO

For every sample and response block, select the lowest-confidence positions using detached sampled-token rollout logprobs. The budget is `ceil(training_token_fraction * block_size)`, capped by valid positions. Ties prefer earlier positions; prompts, padding, and excluded samples never enter selection. This is a confidence proxy, not entropy.

With block size 16 and fraction 0.25, the schedule supplies four views instead of sixteen. Each sample and block can choose a different offset in a view. Actual tokens before the selected offset are revealed, including unselected tokens; the selected offset and all later positions remain masked. Earlier-block context comes from the clean half, preserving ordinary JustGRPO conditioning at selected positions.

The objective is normalized by selected-token count, without selection-probability correction. Previous-policy logprobs are always recomputed before the update and serve as the PPO denominator. Generation logprobs remain separate for confidence selection, importance-sampling correction, and generation/training diagnostics. Previous-policy scoring covers all response positions for upstream sequence-level diagnostics, independently of `require_router_replay`. Reference scoring explicitly chooses the training targets. The router-replay argument only controls MoE routing replay; Fast selection reduces the training and reference-scoring views. Reference scoring remains enabled when reference KL is nonzero. `loss_fn.force_on_policy_ratio: true` is rejected because it can skip previous-policy scoring. `seq-mask-tis` is rejected because its whole-response acceptance decision cannot be computed independently per reveal level. Token-level `tis` remains supported. Reduced view count does not guarantee proportional end-to-end speedup.

## Sudoku and the matched experiment

The custom 6x6 environment uses the reference generator, 2x3 boxes, few-shot prompt, and **blank-cell accuracy** reward. `block_diffusion/environments/sudoku.py` contains the dataset, prompts, scoring, and NeMo-RL environment interface. `sudoku6x6_generator.py` implements puzzle generation; `sudoku6x6_fewshot.txt` holds the reference system prompt. Only this 6x6 task is supported; the 4x4 Reasoning Gym adapter and its dependency have been removed.

The 32-GPU recipe matches [cmp0913_sudoku6x6_block_justgrpo_val_t1_extend0914](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/sw1o4ibd): 128 prompts x 8 completions, 100 updates, LR 3e-7, 13 warmup updates starting at 3e-8, weight decay 0.01, reward normalization off, reference k3 KL 0.01, TIS capped at 2, and 512 output tokens. Block size is 16 and training temperature is 1.0. Validation uses the training decoder at temperature 1.0 every ten updates, on 256 puzzles repeated four times.

The reference seeds (training 1, validation 2) produce overlapping puzzles because the generator seeds each puzzle with `seed + index`. These matched splits are **not held out**. Runs start fresh from the base model, without the reference optimizer state or an identical RNG/data-order replay.

## Metrics and verification

W&B uses upstream GRPO names, including `train/reward`, `train/kl_penalty`, `train/gen_kl_error`, `train/loss`, `train/grad_norm`, and ratio/importance metrics. `validation/accuracy` reports the mean environment reward. By default, validation inherits the training decoder; named modes add suffixed metrics as described below.

Metrics and timers come directly from upstream GRPO, including `timing/train/total_step_time`, generation, scoring, training, validation, and worker timings. `global_valid_toks` counts selected training tokens. Upstream writes rollout JSONL files; `wandb_url.txt` contains the tracking link. The former custom `metrics.json` format is replaced by upstream logging.

See [VERIFICATION.md](VERIFICATION.md) for recorded Megatron checks and their scope.


## AR and diffusion validation

Synchronous colocated and asynchronous dedicated-GPU `reference_vllm` runs support
named validation modes through
`policy.generation.vllm_val_dllm_variants`, matching the engine-override format in
`diffusion_RL`.
The reusable override is `../block_diffusion/configs/validation/ar_diffusion.yaml`; add it after your
recipe in the YAML `defaults` list. Runnable examples end in
`-megatron-vllm-dualval-long.yaml` under `configs/recipes/`. Async examples end in
`-megatron-vllm-async-dualval-long.yaml`; their 32 GPUs are split evenly between
training and generation.

The example evaluates each current policy under confidence-threshold diffusion
(block size 16, 16 steps, threshold 0.9, temperature 1) and causal AR (temperature
1). Each mode uses the same validation prompts and budget. W&B records
`validation/accuracy/diffusion_conf09`, `validation/accuracy/ar`, and corresponding
`avg_length` metrics. The first configured mode also supplies `validation/accuracy`
and the primary `val_data_step*.jsonl` samples. Where checkpointing is enabled,
`checkpointing.metric_name: val:accuracy/ar` selects by AR accuracy; the research
checkpointing restrictions otherwise remain in force.

Each named mode must specify `temperature`, `vllm_cfg.gpu_memory_utilization`, and
`vllm_kwargs` with `hf_overrides.architectures` and `diffusion_config`. AR uses
`[NemotronLabsDiffusionForCausalLM]` and `diffusion_config: null`. Diffusion uses
`[NemotronLabsDiffusionModel]`, matching `hf_overrides.block_size` and
`diffusion_config.canvas_length`, and explicitly pinned sampling knobs. Extra
training-only reveal/entropy channels are disabled for validation. With no variants
configured, the existing `validation_sampling` behavior is unchanged.

Validation groups initialize before rollout workers and sleep between passes.
Every pass refits from current policy weights using a separate IPC socket for
colocated runs or a named NCCL collective for dedicated generation GPUs. Async
collection pauses and drains active rollouts before switching engines, then resumes
after validation. Worker registration and launch are atomic with the pause gate,
so a pending launch cannot race engine sleep. Rollout engines are restored even
if validation raises. The groups share generation GPU memory sequentially but
consume extra CPU memory for sleeping model weights.
Megatron generation and non-default refit transports are rejected for this option.
This port does not add runtime sampler reconfiguration or the SGLang variant format.

## Checkpointing and continuation

Set `checkpointing.enabled: true` and an explicit `checkpointing.checkpoint_dir`
to use upstream GRPO checkpointing. Keep `checkpointing.save_optimizer: true` for
training continuations. The upstream controller saves policy/optimizer state,
training counters, and dataloader state; async GRPO also saves rollout and replay
buffer state. It resumes the latest finalized `step_<n>` directory under the same
checkpoint root. Temporary `tmp_step_<n>` directories are not resumable.

Keep the training topology unchanged when restoring Megatron distributed optimizer
state. Queue continuation jobs with `afterany` and require a completed checkpoint
before launching them, so a missing checkpoint cannot silently start fresh.
`checkpointing.checkpoint_must_save_by` sets the save deadline within each allocation;
leave time for validation, checkpoint finalization, and shutdown before Slurm's limit.
These settings also apply to Trace through the shared driver.

## AR GRPO on DeepScaleR

`../ar_grpo/run_ar_grpo.py` trains Nemotron-Labs-Diffusion-3B with causal AR attention,
DeepScaleR training data, and AIME2024 validation. Its recipe uses LR `1e-6`,
2 training + 2 generation nodes, async rollouts, and no activation recomputation.
The same checkpoint is validated in `diffusion_conf09` and `ar` modes.

```bash
PYTHONPATH="$PWD:$PWD/research/block_diffusion:$PWD/research/ar_grpo:$PWD/research/just_grpo" EXPANDABLE_SEGMENTS=false \
NRL_VLLM_PY_EXECUTABLE="$NRL_VLLM_PY_EXECUTABLE" \
uv run research/ar_grpo/run_ar_grpo.py \
  --config research/just_grpo/configs/recipes/ar_grpo-deepscaler-3b-4n8g-megatron-vllm-async-dualval-long.yaml
```

AR, JustGRPO, and Trace import the shared `block-diffusion` package and use `block_diffusion/training.py` for controller setup and the
lifecycle of `MultiModeValidation` in `block_diffusion/generation/validation.py`.
AR uses shifted next-token labels and a causal policy adapter; diffusion
algorithms keep their existing unshifted labels and schedules. The canonical entrypoint is `research/ar_grpo/run_ar_grpo.py`. It uses the
legacy GRPO controllers for this recipe and SingleController for configs with
`grpo.async_grpo: null`, including the 128K ready-first configs.


## Optional LLaDA validation backend

The GSM8K vLLM long, diffusion-auxiliary, smoke, and full-canvas-validation
recipes use T=1 for training and T=0 for validation. Their accuracy/NFE sweep
uses `fixed1_t0`, `threshold09_t0`, `threshold08_t0`, and `threshold07_t0`.

For LLaDA vLLM training, `just_grpo.validation_runtime` selects validation:

- `megatron_llada`: use the live training policy. This remains the GSM8K recipe default and supports the existing 32-token selection blocks within a 256-token attention canvas.
- `generation`: use dedicated native vLLM validation engines, refitted from the current policy. Configure decoders through `just_grpo.validation_modes` (or `validation_sampling` for one mode). The rollout decoder remains leftmost.

The current vLLM LLaDA implementation selects across one full canvas, so validation `block_size` must equal `policy.generation.max_new_tokens`. For the 256-token GSM8K recipes, set both `just_grpo.validation_sampling.block_size=256` and `just_grpo.validation_sampling.max_steps=256`; the named modes inherit these values. The opt-in recipe `just_grpo-gsm8k-llada8b-canvas256-16n8g-vllm-fullcanvas-val-long.yaml` sets all three overrides. Apply the same overrides to the diffusion-auxiliary recipe to retain its loss configuration.

Full-canvas confidence selection differs from 32-token-block selection; accuracy and NFE are not an identical-decoder comparison. Both validation backends retain the training-step accuracy–NFE plot. Each vLLM mode owns an engine and refit namespace; only one generation group is awake at a time, but startup and host-memory use increase with the number of modes.


## MATH-500 greedy validation with vLLM rollouts

Use `configs/recipes/just_grpo-math-llada8b-canvas256-16n8g-vllm-long.yaml`
for plain JustGRPO, or the sibling `-long-diffaux.yaml` recipe for the positive
completion diffusion loss (weight 0.1, masking 5%–100%). Set
`LLADA_MATH_DATA_DIR` to the prepared MATH dataset directory containing
`train.jsonl` and the 500-question `test.jsonl`, as for the existing MATH recipes.

Both recipes keep the 16-node, 125-update setup and leftmost training rollouts at
T=1. Validation runs at the start, every 10 updates, and at the end, covering all
500 questions with T=0. The four modes are `fixed1_t0`, `threshold09_t0`,
`threshold08_t0`, and `threshold07_t0`; the fixed-one mode remains the primary
checkpoint metric. They retain the accuracy-versus-NFE plots and training-step
history. Validation uses the live Megatron policy with 32-token selection blocks
within the 256-token bidirectional canvas. The optional vLLM validation overrides
above also apply to these recipes when full-canvas selection is desired.

The matching smoke recipes are
`just_grpo-math-llada8b-canvas256-2n8g-vllm-smoke.yaml` and
`just_grpo-math-llada8b-canvas256-2n8g-vllm-smoke-diffaux.yaml`.
They match the GSM8K vLLM smoke size: 2 nodes, 2 updates, 2 prompts × 16
completions, and 16 validation questions per mode. Validation runs at the start
and after each update with the same four greedy decoders and local accuracy/NFE
artifacts. Checkpointing and W&B are disabled.
