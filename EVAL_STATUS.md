# NLD 8B evaluation status

Last updated: 2026-09-16. Everything below is on dfw at `~/diffusion_RL/RL`
(branch `dllm_clean`), **uncommitted**.

## Four-benchmark cleanup (2026-09-16)

The maintained suite is now **IFBench, LiveCodeBench, SciCode, and AA-LCR**.
Use `submit_benchmark_eval.sh` with `--model`, `--account`, and a fresh `--outdir`.
Presets and DFW paths are in `examples/configs/benchmark_eval.yaml`; full usage and
scoring caveats are in `tools/benchmark_eval/README.md`. The step5500/step96 submit
scripts now only supply checkpoint identity and delegate to this launcher.

The launcher submits one-GPU generation shards (at most two concurrently) and a
CPU scoring job on `cpu_short`. SciCode uses four whole-problem shards; AA-LCR uses
ten shards. It checks sample coverage and rejects incomplete/duplicate results.
Run metadata records resolved settings, expected IDs, code/data/template hashes,
and per-shard status. Existing training and Terminal-Bench jobs were not canceled.

Correctness fixes: forward LiveCodeBench's system prompt; map vLLM diffusion
request temperature to 0/1 while preserving engine temperature; cap standalone
output tokens to remaining context; preserve AA-LCR's bare question for optional
LLM judging; strip reasoning for AA-LCR and SciCode; isolate SciCode sandbox ports;
make SciCode sampling/concurrency explicit; record actual chat completion tokens.
AA-LCR's default score is a **heuristic approximation, not the official metric or
a guaranteed lower bound**. No external judge is called by the suite. Historical
AA-LCR labels claiming a lower bound below should be read with this correction.

Initial validation: 15 focused unit tests, Ruff, shell syntax checks, and a completed
one-sample-per-benchmark greedy Slurm smoke (generation array `18771382`, CPU
scoring `18771409`). All four generation/scoring paths completed, including eight
SciCode subproblems and a 112,938-token AA-LCR prompt. This first smoke exposed the
CPU partition issue, which is fixed in the launcher. Nonzero-temperature (0.9)
generation and scoring also passed for all four benchmarks: SciCode and AA-LCR in
array `18772026` / scoring `18772036`, and IFBench and LiveCodeBench in retry array
`18772526` / scoring `18772527`. The initial IFBench/LCB sampling workers both failed
with CUDA initialization errors on `pool0-01815`; the collector correctly marked
those benchmarks incomplete. Retrying the identical code/settings on other nodes
with `SBATCH_EXCLUDE=pool0-01815` succeeded. No node exclusion is hardcoded.

Artifacts under `/lustre/fsw/portfolios/coreai/users/snorouzi/eval_results/`:
`benchmark_cleanup_smoke_20260916`, `benchmark_cleanup_sampling_20260916`, and
`benchmark_cleanup_sampling_retry_20260916`. Each has `manifest.json`, `jobs.json`,
and `results.json`. CPU smoke scoring time limits were shortened to ten minutes
for backfill; production presets retain two hours. These reduced-budget smoke
outputs are plumbing checks, not full benchmark scores. No full benchmark rerun
was submitted. Final unit tests, Ruff, shell syntax, and `git diff --check` pass.

Review fixes (2026-09-16): LiveCodeBench now provides buffered stdin and uses
upstream's per-line text/exact-Decimal comparison (including numeric whitespace
and equivalent decimal representations). Historical LiveCodeBench scores above
and below predate these grading fixes and need rescoring before comparison.
SciCode now treats `--thinking-budget 0` consistently with the standalone path:
omit the separate reasoning cap while retaining the total output-token limit.
Regression coverage after the first review was 19 tests, including real grader subprocesses and the
SciCode shell command with mocked runtimes at zero and positive budgets. The
reviewer independently reran the tests and found no remaining actionable issues.
The zero-budget SciCode Slurm smoke (`benchmark_review_fixes_20260916`, generation
`18793747`) completed in 4m09s with all eight subproblems graded. Its resolved
NeMo-Skills config has `extra_body={}`, thinking enabled, and a 512-token total
cap; truncated reasoning correctly produces empty answers. CPU collection job
`18793748` completed successfully in 1m11s, and `results.json` marks SciCode
complete (1 problem, 8 subproblems). The smoke scored zero under its short output
cap; it is not an accuracy measurement. For backfill, only this CPU smoke job
used a ten-minute time limit and 16 GB memory; production presets are unchanged.

Second review fixes (2026-09-16): LiveCodeBench functional tasks now receive
upstream's helper imports and recursion limit; stdout from module initialization,
constructors, and method calls cannot corrupt the grader's verdict. SciCode's
container-client patch now forwards repetition penalty as repetition penalty and
keeps presence penalty neutral, fixing silently biased AR sampling. Older SciCode
AR results using this wrapper require regeneration; rescoring cannot fix sampling.
All 25 regression tests pass. The exact configured container's patched OpenAI
builder also passed eight outgoing-JSON checks through the OpenAI SDK with a mock
transport (AR/diffusion, default/nondefault repetition, absent/present thinking cap).
The reviewer independently passed all 25 tests and found no remaining issues in
these fixes. A SciCode AR Slurm smoke at temperature 0.9 completed generation
`18794846` (5m06s) and collection `18794847`; `results.json` marks one problem and
all eight subproblems complete. Artifacts: `benchmark_review_ar_fixes_20260916`
under the eval-results root above. This smoke used 512 output tokens and a 128-token
thinking cap, so its zero score is not an accuracy measurement. Only its CPU
collector used a ten-minute limit and 16 GB memory for backfill.

## Checkpoints

| short name | path | notes |
| --- | --- | --- |
| v1.5-sft | `.../checkpoints/nld8b_v15sft_nodots` -> `Nemotron-Labs-Diffusion-8B-v1.5-sft-20260902` | the "refreshed" Sept-2 weights, sha256 `ae7be4d9...` |
| step_5500 | `.../checkpoints/nld8b_agentic_step5500` | agentic SFT, copied from EOS `sft-materialized-if-gbs512-hf/step_5500` |
| step_96 | `.../checkpoints/nld8b_topd_exp608_step96` | step_5500 + 96 top-d distillation steps from a Lightning-30B teacher |

Both copies verified against EOS by file-manifest md5 plus sha256 on the first
and last of 309 shards.

**Chat template matters and is not uniform.** step_5500 ships only a stub
template (concatenates `content`, no roles, no tools) because its training data
is pre-materialized; the Nemotron-3-Ultra template was installed over it
(byte-identical to the sibling `Agentic-SFT-NLD-8B-step_4918`). step_96 already
ships the correct Ultra template. v1.5-sft uses the Cascade-2 template. Ultra and
Cascade-2 render byte-identically whenever a system message is present; they
differ only in the default system text for bare-user prompts, where Cascade-2
injects "You are not allowed to use any tools."

## Results

All of ours are greedy (temperature 0) and vLLM FastDiffuser block diffusion
unless stated. **Read the caveats before quoting any of this.**

| benchmark | v1.5-sft | step_5500 | step_96 |
| --- | ---: | ---: | ---: |
| IFBench prompt-strict | 21.67% | 29.00% | failed |
| IFBench prompt-loose | 26.33% | 32.67% | failed |
| LiveCodeBench pass@1 | 18.29% | 24.57% | failed |
| AA-LCR (heuristic judge) | 18.00% | 21.00% | 24.44% (9/10 shards) |
| SciCode subtask | see caveat | 10.30% | 4.27% |
| SWE-bench Verified | 0 (agent never acted) | 1/5 smoke | not run |
| Tau2 Telecom | NOT MEASURED | NOT MEASURED | not run |

Caveats:

- **SciCode denominators differ.** The old v1.5 figure of 20.41% is the 15-problem
  dev split (49 subtasks). step_5500 and step_96 ran the full 80 problems
  (~330 subtasks). Do not put them in the same column.
- **step_96 is worse than step_5500 on SciCode** (4.27% vs 10.30%, same split) and
  generates longer outputs (avg 35,887 vs 26,872 tokens). AA-LCR is flat to
  slightly up. IFBench/LCB died on vLLM `EngineDeadError` (no OOM; 3 of 4 jobs on
  one node) and need a rerun before step_96 has a complete card.
- **IFBench and AA-LCR are not strictly prompt-identical across checkpoints**,
  because each uses its own correct template and they differ for bare-user prompts.
- **AA-LCR uses a heuristic judge**; it is a lower bound, not the official metric.
- **Tau2 was never measured.** See below.

## Comparison with nld-eval-next (rkorostik, branch codex/staging-next)

They evaluate the same step_5500 checkpoint with SGLang AR and a temperature
sweep. Same checkpoint, so this is directly comparable:

| benchmark | theirs t=0.0 | theirs t=0.9 (their best) | ours (greedy, diffusion) |
| --- | ---: | ---: | ---: |
| SciCode subtask | 15.98% (338 sub) | 17.36% | 10.30% (330 sub) |
| IFBench | 37.75% (300) | 46.92% | 29.00% strict (300) |
| LiveCodeBench | 30.84% (454) | 37.81% | 24.57% (175) -- different denominator |
| Tau2 Telecom | 28.07% (114) | 35.86% (912) | 0.0000 |
| SWE-bench Verified | -- | 31.40% (157/500) | 1/5 smoke |

Conclusions:

1. **Tau2's zero was our harness, not the model.** They score 28.07% on the same
   weights with a hosted qwen-235b user simulator. Ours simulated the user with
   the model itself over one endpoint, which the runner explicitly warns is not
   comparable and refuses without `ALLOW_SELF_SIM_FULL=1`.
2. **Temperature is worth 7-9 points and we ran the worst setting.** 0.0 -> 0.9
   gains +9.2 IFBench, +7.0 LCB, +7.8 Tau2. SciCode peaks at 0.2 (20.22%) and
   collapses at 1.0 (11.44%), so 1.0 is not universally right.
3. Residual gaps at matched t=0.0 (SciCode 10.30 vs 15.98, IFBench 29.00 vs 37.75)
   are most plausibly AR vs block diffusion.

## SWE-bench: why it was zero, and what fixed it

Two independent blockers, found in sequence. The second was invisible until the
first was cleared.

**v1.5-sft = 0: the model.** It emitted `<function=>` -- structurally valid tool
calls with an empty function name -- so the agent never acted (4 steps, 0 API
calls). Not context: it fires on the first turn and reproduces under plain
HuggingFace with a ~2,900-character prompt. Ruled out by measurement: the XML
parser, vLLM, decode mode (AR and diffusion both), and the tool names themselves
(`bash`/`file_editor`/`get_account`/`stateful_python_code_exec` all empty).
step_5500 emits correct names on the identical prompt and template.
Caveat: every v1.5 observation was at temperature 0, and greedy is now known to
make this model repeat bad outputs. v1.5 at t>0 is untested.

**step_5500 = 0: our harness.** Ablation on 5 astropy instances:

| run | change | resolved | note |
| --- | --- | ---: | --- |
| A | SWE-agent, AR, greedy, ctx 65536 | 0/5 | loops on `pip` x71, `cd` x92 |
| B | + OpenHands scaffold | 0/5 | 5/5 applied; `AgentStuckInLoopError` 4/5 |
| C | + temperature 1.0 | 0/5 | loops 4/5 -> 0/5; edits real source 2/5 |
| D | + 200 iterations | 0/5 | edits real source 4/5; 765/1001 requests 400 |
| E | + context 131072 | **1/5** | 400s 765 -> 285; 2/5 episodes finish cleanly |
| F | + context 262144 | n/a | vLLM rejects: exceeds `max_position_embeddings` |
| G | E but block diffusion | 0/5 | loops again -- see the hardcoded-temperature bug |

**Main cause: `MAX_MODEL_LEN` defaulted to 65536**, half the trained 131072. An
agent conversation reaches ~49-50K prompt plus a 16384 output request, crosses
the ceiling, and every later call 400s. C and D had *identical* 236 successes
despite D getting twice the iterations -- both hit the same wall, and D spent its
extra turns retrying failures. "Agent reached maximum iteration" was a symptom.

Secondary: **greedy decoding**. At temperature 0 the agent re-emits the identical
command on revisiting a context; OpenHands aborts with `AgentStuckInLoopError`.

Not a cause: the OpenHands-vs-SWE-agent scaffold (better diagnostics, no score
change), and the thinking budget (`THINKING_BUDGET` was dead code).

## Fixes applied to `tools/swebench/run_swebench_eval.sh`

| fix | was | now |
| --- | --- | --- |
| `MAX_MODEL_LEN` | 65536 | 131072 |
| `diffusion_config` temperature | hardcoded 0.0 | `${TEMPERATURE}` |
| `THINKING_BUDGET` | dead code | removed |
| OpenHands commit | `HEAD` (broken) | pinned `1.2.1`, auto-passed |
| `AGENT_FRAMEWORK` default | `mini_swe_agent` (invalid) | `swe_agent` |
| dataset for scoring | read from lustre | staged to `/nemo_run/code` |
| `--diffusion-config` | literal `PLACEHOLDER_DIFFUSION_CONFIG` | real config |

Working SWE-bench recipe: OpenHands + AR + temperature 1.0 + context 131072 +
200 iterations, `ENROOT_SHIM_SQUASH_PROCESSORS=16`, scratch on `/raid/scratch`.

## Not usable / open

- **Tau2**: self-simulated, so not comparable, and it does not finish -- 3/114 in
  4 hours on step_5500 versus all 114 in under 2 hours for v1.5. Needs an external
  simulator (`USER_BASE_URL`/`USER_LLM`) and sharding or higher `MAX_CONCURRENCY`.
- **Terminal-Bench**: never built. They run Terminal-Bench Hard and 2.1 via Harbor.
- **All our numbers are greedy**, i.e. the bottom of their temperature sweep.
- **SWE-bench is a 5-instance astropy smoke**, not a score. 285 requests still 400
  at 131072; 262144 is not available without `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
  (risks NaNs past trained context). Real fix is conversation compaction.
- **OpenHands patches carry a chmod artifact** (1,861 mode-only entries in one
  diff) from enroot extraction. Applies cleanly, but should be stripped.
- **OpenHands config TOMLs live outside the repo**, installed into
  `nemo_skills_host_venv/.../prompt/config/eval/swe-bench/openhands/`. They will
  not survive a fresh venv and must move into version control.
- **Nothing is committed.**

## Where the code is

Untracked: `tools/{swebench,tau2,scicode,enroot_shim,aa_lcr,ifbench,livecodebench}/`,
`submit_{swebench,scicode,tau2_telecom,all_step5500,all_step96}*.sh`.
Modified tracked: `examples/eval_grpo_checkpoint_validation.py`,
`submit_standalone_gsm8k_eval.sh`.

## Suggested next steps

1. Re-run the four scored benchmarks on **AR at t=0.9** (t=0.2 for SciCode) to
   separate method from model.
2. Point Tau2 at an external user simulator.
3. Move the OpenHands configs into the repo and commit the eval tooling.
4. Rerun step_96 IFBench and LiveCodeBench.
5. Scale SWE-bench past 5 astropy instances with a shuffled multi-project sample.
