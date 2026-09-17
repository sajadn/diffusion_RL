# IFBench, LiveCodeBench, SciCode, and AA-LCR

Agent workflow: [benchmark-eval skill](../../skills/benchmark-eval/SKILL.md).

Run from the DFW checkout. `submit_benchmark_eval.sh` submits generation shards
(one GPU each) and a dependent CPU scoring job. It uses the existing vLLM
Nemotron diffusion runtime and existing driver/scoring environments; no installation
or additional model service is required. Terminal-Bench, SWE-bench, and tau2 are
outside this suite.

```bash
./submit_benchmark_eval.sh \
  --model /path/to/converted/hf/checkpoint \
  --account nvr_lpr_llm \
  --outdir /lustre/path/to/new/run \
  --dry-run
```

Check current fairshare with `sshare -U -o Account,FairShare`, choose an account,
and remove `--dry-run` to submit. The launcher prints the exact sbatch commands.
The output directory must be new. Tokenizer defaults to the checkpoint; use
`--tokenizer` only when a separate tokenizer/chat template is intended.

Evaluation presets and DFW paths live in
[`examples/configs/benchmark_eval.yaml`](../../examples/configs/benchmark_eval.yaml).
Use `--config` for a modified YAML file. CLI overrides are recorded in the manifest:

- `--benchmarks ifbench livecodebench scicode aa_lcr` (all four by default).
- `--algorithm FastDiffuser|AR`, `--temperature`, `--max-parallel`.
- `--num-samples`, `--shards`, `--max-new-tokens`, `--thinking-budget`.

`--thinking-budget 0` removes the separate reasoning cap for all four benchmarks;
`--max-new-tokens` still caps total generation. It does not disable thinking.

For a small plumbing check on all four benchmarks, add:

```bash
--num-samples 1 --shards 1 --max-new-tokens 2048 --thinking-budget 512
```

Reduced-budget smoke scores are not benchmark measurements. All sampling uses a
fixed seed from YAML. SciCode sharding preserves complete problem/subproblem chains.
The default four SciCode shards and ten AA-LCR shards keep individual jobs within
the two-hour `batch_short` allocation; at most two GPU shards run concurrently.
Timeouts remain possible for unusually slow models. No existing jobs are canceled.

`submit_all_step5500.sh` and `submit_all_step96.sh` are thin checkpoint aliases for
this launcher and require `--account` and `--outdir`. They now select only these four
benchmarks. Legacy standalone/math and SciCode launchers remain available, but the
common launcher is the maintained suite entrypoint.

## Outputs and scoring

- `manifest.json`: resolved settings, model/tokenizer identity, expected sample IDs,
  code/data/template SHA256 hashes, and repository commit. The worker and collector
  reject changed inputs/code, so do not edit evaluation sources during a run.
- `jobs.json`: generation array and scoring job IDs.
- `<benchmark>/shard-NNN/`: status, server log, records, generation metadata. SciCode
  also has sandbox logs, input problems, expected non-prefilled steps, and NeMo-Skills
  output. Standalone generation flushes `records.partial.jsonl` as requests complete.
- `<benchmark>/records.jsonl`, `scores.json`: merged responses and benchmark scores.
- `results.json`: completion status and scores for each benchmark. Missing/failed
  shards, duplicate IDs, missing questions, or incomplete SciCode subtask grading
  mark the benchmark incomplete; they are never silently averaged or counted as a
  complete evaluation. Partial records are diagnostic only.

| Benchmark | Scoring path |
|---|---|
| IFBench | Upstream IFBench strict/loose constraint verifiers. |
| LiveCodeBench | Existing local public/private test harness, last fenced-code extraction, one attempt per problem. |
| SciCode | Installed NeMo-Skills chained generation and inline tests; whole-problem and subtask accuracy, excluding upstream prefilled steps. |
| AA-LCR | **Heuristic answer matching, not the official AA-LCR metric or a guaranteed lower bound.** |

IFBench, AA-LCR, and SciCode strip reasoning according to the recorded thinking setting.
LiveCodeBench receives its dataset system message. Long standalone prompts are
never truncated: the output cap is reduced to the remaining context and recorded
per sample; prompts that do not fit fail explicitly. Diffusion temperature is set
in the engine; API temperature is 0 for greedy and 1 for sampling.

AA-LCR's optional separate LLM grading remains available through
`tools/aa_lcr/score_aa_lcr.py --judge openai --judge-base-url ... --judge-model ...`.
The suite never enables it automatically. Use a judge API key through the scorer's
configured environment variable, not a command-line secret. Heuristic and
LLM-judged scores must be reported separately.

LiveCodeBench and SciCode execute generated code using the existing local subprocess/
Flask harnesses. Time/resource limits and temporary working directories are not a
security isolation boundary. LiveCodeBench scoring runs in a CPU Slurm allocation.

To repeat scoring for an unchanged manifest, submit a CPU job (same account and
resources as the printed scoring command) with:

```bash
--wrap='/path/to/driver/bin/python tools/benchmark_eval/evaluate.py collect --manifest /path/to/run/manifest.json'
```

Use an absolute script path or submit from the repository root. `collect` enters
the configured container itself. It reports complete benchmarks even when another
failed, and exits nonzero if any requested benchmark is incomplete. A timed-out
shard may retain `status: running`; it is still treated as incomplete.

## Validation

```bash
UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache_tau2 \
uv run --no-project \
  --python /lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_mb_3rdparty_sglagn_local_fork/bin/python \
  python -m unittest discover -s tests/unit/benchmark_eval -v
```

Tests cover request temperature, system messages, context limits, reasoning removal,
shard coverage, incomplete/duplicate results, weighted SciCode aggregation, binary
stdin, upstream-compatible LiveCodeBench output comparison and functional helpers,
functional stdout isolation, SciCode client request parameters, and zero/positive
SciCode thinking-budget command arguments.
A real Slurm smoke run is also required after changing the launch/container path.

Validated on DFW on 2026-09-16: all 25 unit tests and generation/scoring smoke
checks for all four benchmarks at temperatures 0 and 0.9 passed. See
[`EVAL_STATUS.md`](../../EVAL_STATUS.md) for job IDs, artifact directories, and the
node-specific CUDA startup failure that required retrying two sampling workers.
After review fixes, an additional one-problem SciCode Slurm smoke with zero
thinking budget completed generation and collection successfully (8 subproblems).

A subsequent SciCode AR smoke at temperature 0.9 also completed generation and
collection after the penalty fix. Historical AR outputs from the unpatched client
need regeneration; changing scoring cannot remove a sampling penalty.
