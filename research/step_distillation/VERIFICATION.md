## Positive-rollout next-state KD (2026-09-30)

The standalone variant excludes the earlier sampled-token CE modes and frozen
rollout generation. It includes block-local reveal normalization, native math
prompt alignment, LOGIT-loss dispatch, reward exposure, and empty-batch handling
required to run KD independently.

The original implementation passed GPU smoke job 19576449: two updates, finite
losses 1.1400/1.3596, 16/64 and 20/64 retained responses, 113/161 KD targets,
and a finalized step-2 checkpoint. This used synthetic arithmetic examples to
exercise positive/negative filtering, not a benchmark. The isolated extraction
retains the same KD functions and frozen reference worker path; its final CPU
validation is recorded below.

Final isolated-branch validation: 167 unit tests passed, one optional Megatron
adapter module skipped. Ruff and whitespace checks passed. The four KD
computational functions/classes are AST-identical to the initial implementation.

Isolated GPU smoke 19578748 completed two updates and saved step_2. Retained
counts (16/64, 20/64), target counts (113, 161), losses (1.1400, 1.3596), and
gradient norms (178.9418945, 229.9425354) match the original smoke. This exercises
the standalone native-driver changes, live student refits, frozen reference
teacher, distributed target transfer, backward passes, and checkpoint saving.

## Dedicated folder extraction (2026-09-30)

Moved the algorithm, frozen teacher worker, KD configuration, recipe, entrypoint,
and tests to research/step_distillation. JustGRPO supplies shared trajectory and
runtime infrastructure and has no import of the distillation package.

After the move: 167 tests passed across both research suites; two optional
Megatron-dependent modules skipped. The dedicated driver test checks loss,
callback, and worker wiring; the callback test checks sampled targets before
teacher dispatch. Entrypoint --help, Ruff, and whitespace checks passed. The
four KD computational functions/classes remain AST-identical to commit
8bcbb5383. Earlier GPU smokes above precede this folder extraction; no new GPU
smoke was submitted for the move. The active training job uses the original
RL-trace-diffusion-ports checkout and is unaffected.

Workspace metadata adds only the local step-distillation package; all existing
locked packages are unchanged. A full uv lock check requires Python 3.13.14+
per the repository manifest; the reused CPU test environment is Python 3.13.13.
