---
name: benchmark-eval
description: Run, monitor, or troubleshoot IFBench, LiveCodeBench, SciCode, and AA-LCR checkpoint evaluations on DFW using submit_benchmark_eval.sh. Use for benchmark selection, launch commands, scoring interpretation, and changes to this evaluation suite; use the math evaluation skill for GSM8K/AIME.
---

# Checkpoint benchmarking

Use the repository's [submit_benchmark_eval.sh](../../submit_benchmark_eval.sh)
for these four benchmarks, including SciCode. The legacy `submit_scicode_eval.sh`
is not required by this workflow. This launcher serves supported Nemotron diffusion
HF checkpoints with vLLM; it is not a general-purpose evaluator for arbitrary models.

Read the [usage guide](../../tools/benchmark_eval/README.md) and
[YAML presets](../../examples/configs/benchmark_eval.yaml) before constructing a
run. YAML is the source of evaluation defaults and DFW runtime/data paths. Check
`./submit_benchmark_eval.sh --help` for current flags. Treat
[EVAL_STATUS.md](../../EVAL_STATUS.md) as historical validation evidence, not current
job state or a universal checkpoint recommendation.

## Choose scope and scoring

Honor the requested benchmarks, checkpoint, decode settings, and resource limit.
Omitting `--benchmarks` runs all four; pass an explicit subset when only some were
requested. An explanation or dry-run request does not authorize submitting jobs.

| CLI name | Scoring used by this launcher |
| --- | --- |
| `ifbench` | Upstream IFBench strict/loose constraint verifiers. |
| `livecodebench` | Existing local public/private test harness; one attempt per problem. |
| `scicode` | NeMo-Skills chained subproblem generation and inline tests; problem and subtask accuracy. |
| `aa_lcr` | Custom local heuristic answer matching; not official AA-LCR scoring. |

**Explain the AA-LCR distinction before proposing a run.** The heuristic predates
the unified launcher. It normalizes text and checks reference-answer containment,
including semicolon-separated reference components. It can produce false positives
and false negatives; it is neither a semantic judge nor a guaranteed lower bound.
Use it for debugging and label its results explicitly as heuristic.

The [published AA-LCR methodology](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR)
uses an LLM equality judge. For reportable results, verify the applicable dataset
revision and judging protocol. The current suite supports only heuristic AA-LCR
scoring; the separate `tools/aa_lcr/score_aa_lcr.py --judge openai` path accepts a
provided judge endpoint/model. Any compatible judge is not automatically equivalent
to the publisher's judge. Do not silently start another model or call a paid API.
If another model is excluded by the user, explain that official AA-LCR judging is
unavailable under that constraint; do not present heuristic results as its replacement.

## Prepare and submit

Work on DFW in place over SSH; do not mirror the repository to local scratch:

```bash
ssh dfw
cd /home/snorouzi/diffusion_RL/RL
./submit_benchmark_eval.sh --help
sshare -U -o Account,FairShare
```

Use an already converted Hugging Face checkpoint with its intended tokenizer/chat
template, and a fresh output directory on Lustre. `--tokenizer` defaults to the
model directory; override only when a separate tokenizer is intended. For a raw
Megatron checkpoint, follow the conversion section of the
[math checkpoint skill](../gsm8k-checkpoint-eval/SKILL.md); do not import its math
prompt or evaluation defaults into these benchmarks.

Use the existing environments, containers, and staged datasets from YAML. For another
user or cluster, adapt a config and pass `--config`; these paths are site-specific.
A dry run validates paths and plans, but does not validate GPU/container execution.

Preserve an explicitly chosen account. Otherwise select an eligible account using
current fairshare; never choose `coreai_dlalgo_modelopt`. See the
[DFW cluster skill](../dfw-cluster/SKILL.md) for cluster operations. This suite uses
`batch_short` for GPU generation and `cpu_short` for CPU scoring, with two-hour
limits in the current preset. The wrapper manages sharding, per-job server/sandbox
ports, and the scoring dependency; do not reconstruct those jobs by hand.

Example for all four benchmarks (replace the checkpoint, account, and output path):

```bash
./submit_benchmark_eval.sh \
  --model /path/to/converted-hf-checkpoint \
  --account YOUR_SLURM_ACCOUNT \
  --outdir /lustre/path/to/new-run \
  --benchmarks ifbench livecodebench scicode aa_lcr \
  --dry-run
```

For SciCode alone, use `--benchmarks scicode` in the same command. Remove
`--dry-run` to submit when execution is authorized. Show the exact submission
command and continue through the agreed run without asking for repeated approval.

For an initial smoke test after launch/runtime changes, use a separate output
directory and add:

```bash
--num-samples 1 --shards 1 --max-new-tokens 2048 --thinking-budget 512
```

The shard override prevents empty shards with a small sample count. These reduced
budgets validate the pipeline, not benchmark accuracy. Use the full presets for
measurements; remove the smoke overrides and use another fresh output directory.

Other useful overrides are `--algorithm FastDiffuser|AR`, `--temperature`, and
`--max-parallel`. Pass the desired diffusion temperature to the launcher: it maps
the API request temperature to 0/1 separately from the engine's sampling temperature.
Keep benchmark prompts and thinking settings consistent when comparing checkpoints.
SciCode shards contain complete problems, preserving each subproblem chain.

## Monitor, diagnose, and report

Read `jobs.json` for the generation array and scoring job IDs. Monitor both with
`squeue`; confirm terminal states and exit codes with `sacct`. A submitted job or
successful generation alone is not a completed scored evaluation. For an authorized
end-to-end run, continue monitoring through scoring and report failures promptly.

Read `results.json` for each benchmark's completion status and scores. Inspect
`<benchmark>/scores.json`, merged `records.jsonl`, and per-shard `status.json`,
`server.log`, and Slurm logs when diagnosing a problem. SciCode also has a sandbox
log and NeMo-Skills results. Use the manifest's expected IDs and scoring checks to
verify coverage; do not average shard percentages or score partial records as a
complete benchmark. A timed-out shard can retain `status: running`.

Input/code/template hashes are checked against `manifest.json`. Do not edit those
files during a run or alter the manifest to bypass a mismatch. Use a fresh run after
changes. `records.partial.jsonl` is diagnostic output, not automatic resume support.

For a failed run, identify the cause before retrying. Retry the affected benchmark
subset in a fresh directory within the agreed resource budget; preserve the original
artifacts. Do not cancel existing jobs. Stop repeated identical retries when no new
fix or evidence changes the outcome; report the blocker, and ask for a decision only
when resolving it requires a new choice outside the agreed scope. Past node failures
are diagnostic evidence, not grounds for permanent exclusions or automatic retries
on a hardcoded node list.

Report checkpoint, dataset/subset, decode and thinking settings, completed sample
counts, scoring method, job IDs, and artifact path. Distinguish smoke tests from full
benchmark runs and identify AA-LCR heuristic results explicitly. Consult the usage
guide's validation commands when changing code; validate the real Slurm path with a
small smoke run after launch/container changes. Documentation-only changes do not
require new GPU jobs.
