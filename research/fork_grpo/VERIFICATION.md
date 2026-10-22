# Nemotron MATH-12K four-mode continuation (2026-10-21)

- Reuses `MultiModeValidation` for entropy-budget, confidence-0.9, AR, and
  additional Fork decoding. The first three mode dictionaries are checked for
  exact equality with `trace_math12k_eb01_te1_ts06_sync_20261019`.
- Multi-mode and async lifecycle regression tests: **41 passed**.
- The current vLLM sampler and the Trace reference sampler both pass the same
  independent scalar entropy-budget checks at coupled entropy temperature and
  explicit entropy temperature 1.0. Checks cover selection, absorbing positions,
  padding, sampled-token frequencies, logprobs, and emitted sampling entropy.
  These checks run the sampler body on CPU; they do not establish GPU execution.
- The continuation gates training on compiled GPU sampler tests, including
  separate entropy temperatures and the existing Fork fallback semantics.
- The recipe retains the original four-node 2+2 split, optimizer/scheduler target
  of 100 updates, three replay views, and 3,744-token generation budget.
  It resumes the latest finalized checkpoint and validates optimizer shard sizes.
- Checkpoint ranking uses the named entropy-budget metric to avoid mixing old
  Fork-only scores with the new primary evaluation mode.

# Baseline-matched LLaDA validation (2026-10-21)

- MATH-12K and the new GSM8K recipe use the primary JustGRPO `fixed1_t1`
  validation decoder: 32-token blocks, one highest-confidence sampled token per
  step, temperature 1.0, and 256 response tokens. Training retains Fork decoding
  and the 64-view replay cap. Additional baseline threshold sweeps are disabled.
- ForkGRPO and LLaDA regression suites: **88 passed, 1 deselected** (the existing
  stale JustGRPO recipe inventory assertion). New tests compare both recipes
  directly with the baseline decoder and reject non-Fork training or malformed
  validation blocks. Ruff and shell syntax checks passed.
- The preceding MATH job `20648526` passed two GPU training updates and Fork
  validation before its full run began. The smoke logged fallback means of
  72.28 and 63.40, maxima 202 and 226, and 64 replay views in both updates.
  This establishes GPU Fork replay execution, not the new validation results.
- Job `20648526` was cancelled at the user's request during initial production
  validation, before any production checkpoint. Replacement MATH and new GSM8K
  launches use fresh directories and begin with two updates plus baseline-decoder
  validation before their full runs.

# LLaDA Fork verification (2026-10-21)

- ForkGRPO, TraceGRPO, and LLaDA full-canvas/generation suites: **151 passed,
  1 deselected**. The deselected `exactly_eight_llada_recipes` test already has
  a stale inventory count at HEAD; this change does not add a JustGRPO recipe.
- Tests cover leftmost fallback selection, exact full-canvas replay logprobs,
  globally ordered multi-block reveals, prompt offsets, scoring/training mask
  parity, DP slicing, empty rows/shards, deterministic sampling, and unbiased
  normalization. Fallback counts 0/3/64/130 produce 1/3/64/64 batch views.
- Sampled MASK tokens commit once, preventing repeated fallback decisions at
  the same position. Existing JustGRPO decoding remains covered.
- The new MATH-12K/MATH-500 recipe validates with a 256-token generation canvas
  and at most 64 views per scoring/training phase. Generation is not capped at 64.
- Ruff check/format and `git diff --check` passed. Workspace lock metadata
  includes the new dependency on the existing `just-grpo` package. Offline
  `uv lock --check` cannot run because Python 3.13.14 is unavailable; the
  provisioned test interpreter uses Python 3.13.13.
- GPU execution and actual LLaDA fallback-count measurements remain pending.
  The launch runs two training updates plus validation before the full run.

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
