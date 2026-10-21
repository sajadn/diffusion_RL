# Three-view fallback replay verification (2026-10-21)

- **140 CPU tests passed** across ForkGRPO, TraceGRPO, the shared schedule, and
  Megatron generation. Ruff (including import order), formatting, and diff checks
  passed for changed Python files.
- Fork tests cover 1, 3, and 4 replay views, prompt offsets, sampled-token
  generation-logprob parity, scoring/training context parity, per-block masking,
  exact coverage of small fallback sets, finite zero-loss views, and fixed view
  counts on empty DP shards.
- An enumerated four-subset test checks that inverse-inclusion weights reproduce
  the full fallback mean in expectation. Every sampled response has total loss
  weight one; draws reproduce exactly for the same seed and update.
- The single DeepScaleR recipe uses three replay views and retains 16 generation
  denoising steps. The shared `TraceGRPOSchedule` accepts per-block selected levels
  and weights while preserving its original per-response interface.
- Real GPU validation is performed by the two-update vLLM smoke in the submitted
  job before the production run. CPU success does not establish GPU performance.

```bash
PYTHONPATH=.:research/fork_grpo:research/trace_grpo:research/block_diffusion:research/just_grpo \
OMP_NUM_THREADS=1 uv run --no-project --python "$TEST_PYTHON" python -m pytest \
  -q -c research/fork_grpo/pyproject.toml \
  research/fork_grpo/tests/unit research/trace_grpo/tests/unit \
  research/just_grpo/tests/unit/test_schedule.py \
  research/just_grpo/tests/unit/test_megatron_generation.py --confcutdir=research
```

# ForkGRPO verification (2026-10-19)

- ForkGRPO, TraceGRPO, shared Megatron generation, native vLLM metadata,
  and logprob aggregation: **123 CPU tests passed**.
- The single recipe validates DeepScaleR training, AIME2024 validation, and the
  math environment. The launcher and functional smoke use that recipe.
- Replay tests cover recorded-generation logprob parity, fallback-only loss
  coverage and normalization, terminal context, empty responses, DP slicing,
  missing metadata, and sync/async fallback metadata transport.
- ForkGRPO directly uses `TraceGRPOSchedule`; no separate Fork schedule exists.
- Ruff check/format, shell syntax, and `git diff --check` passed.
- The Megatron split-state optimizer regression test was skipped because the
  CPU environment lacks `megatron.bridge`. An earlier focused check of the
  optimizer/scheduler guard with AdamW passed, including preservation of
  momentum, weight decay, and scheduler state on empty batches.
- Workspace dependency metadata is consistent. Full offline `uv lock --check`
  is unavailable because the host lacks the workspace-required Python 3.13.14;
  CPU tests used the existing Python 3.13.13 environment.
- vLLM compiled GPU sampler tests passed, including fallback capture, confident
  singleton exclusion, commit emission, and clearing flags between blocks.
- Real asynchronous Nemotron-3B generation returned aligned token IDs, reveal
  steps, fallback flags, and logprobs in cumulative and delta modes. Four normal
  128-token responses contained 18–24 fallback tokens each (job `20507465`).
  A separate easy-prompt run exercised all-false masks.
- The streaming collector regression passed with and without diffusion metadata.
  vLLM Ruff diagnostics match the three pre-existing HEAD diagnostics.
- The DeepScaleR recipe and production driver imports pass preflight with the
  reference vLLM interpreter. GPU training validation of the new backend and
  paper-level learning reproduction remain pending. The two-GPU execution
  smoke is `tests/functional/deepscaler.sh`.

From the repository root, with `TEST_PYTHON` pointing to a provisioned test
interpreter:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=.:research/block_diffusion:research/fork_grpo:research/just_grpo:research/trace_grpo \
uv run --no-project --python "$TEST_PYTHON" python -m pytest \
  -c research/fork_grpo/pyproject.toml \
  research/fork_grpo/tests/unit research/trace_grpo/tests/unit \
  research/just_grpo/tests/unit/test_megatron_generation.py \
  research/just_grpo/tests/unit/test_diffusion_logprob_aggregator.py \
  research/just_grpo/tests/unit/test_native_vllm_generation.py \
  -q --confcutdir=research
```
