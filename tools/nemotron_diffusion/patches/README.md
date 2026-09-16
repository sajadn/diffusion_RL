# Entropy selection for Trace GRPO

The four entropy configs and the NeMo-RL integration are committed together.
The decoder lives in a separate custom vLLM repository, so its implementation
and regression tests are preserved here as `vllm_entropy_selection.patch`.
This patch contains both fixed-count entropy selection and the entropy-budget
selection used for standalone evaluations. It excludes the experimental
confidence-likelihood correction and auditing hooks.

## Apply the vLLM dependency

The patch is based on custom vLLM commit
`99064e736080e90edb3b7d1cf72cc47440cc556a`, not stock vLLM. From a checkout of
that commit, use the absolute path to this directory:

```bash
git apply --check /path/to/RL/tools/nemotron_diffusion/patches/vllm_entropy_selection.patch
git apply /path/to/RL/tools/nemotron_diffusion/patches/vllm_entropy_selection.patch
```

The experiment runtime already contains these decoder changes in its working
tree at
`/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/vllm-nemotron-dllm`;
do not apply the patch again there. Configure `NRL_VLLM_PY_EXECUTABLE` to use
a runtime importing the patched vLLM checkout. The main NeMo-RL commit does
not modify or commit that separate working tree.

## Behavior

- `selection_policy: entropy` ranks masked positions by increasing entropy of
  `softmax(logits / temperature)`. Token draws follow that categorical
  distribution independently of the position ranking. A 16-token canvas and
  eight denoising steps reveal two tokens per step for complete blocks;
  partial blocks use the existing even transfer schedule.
- At temperature zero, token draws are greedy and position entropy is computed
  from the unscaled model distribution.
- `selection_policy: entropy_budget` selects the longest prefix of positions
  sorted by increasing entropy satisfying `sum(H) - max(H) <= entropy_bound`
  (nats). At least one masked position is revealed. Set the step budget to at
  least the canvas length to allow completion when only one token is selected
  per step. This mode was used in standalone evaluation; live NeMo-RL policy
  reconfiguration currently supports `entropy`, not `entropy_budget`.
- The Trace GRPO loss is unchanged. NeMo-RL's `vllm_backend.py` recognizes the
  entropy sampler flag so validation can switch to confidence selection and
  restore entropy selection afterward, together with temperature and steps.

## Four training configs

All paths below are relative to `examples/configs/`.

| Config | Training | Native validation |
| --- | --- | --- |
| `math_agrpo_trace_entropy_top2_t1_8t2g_20260910.yaml` | Entropy, T=1 | Confidence 0.9, T=1, 16 steps |
| `math_agrpo_trace_entropy_top2_t06_8t2g_20260911.yaml` | Entropy, T=0.6 | Entropy, T=0.6, 8 steps |
| `grpo_sudoku6x6_trace_top2_entropy_t1_20260910.yaml` | Entropy, T=1 | Confidence 0.9, T=1, 16 steps |
| `grpo_sudoku6x6_trace_top2_entropy_t06_20260916.yaml` | Entropy, T=0.6 | Confidence 0.9, T=1, 16 steps |

These retain the existing validation choices. Compare MATH training temperatures
using matched offline decoding settings, rather than the native validation
curves. The Sudoku T=0.6 config is a new counterpart, not a completed run.

## Tests

In the NeMo-RL environment with the vLLM dependency available:

```bash
uv run --no-sync python -m pytest tests/unit/models/generation/test_vllm_reconfigure.py -o addopts= -q
```

In the patched vLLM environment on a GPU node:

```bash
python -m pytest tests/models/language/generation/test_nemotron_dllm.py -k 'entropy_selection or entropy_budget' -q
```

The decoder tests cover low-probability token draws, reveal-step metadata,
padding and already-revealed positions, temperature, and the budget rule. The
NeMo-RL test covers validation switching and restoration at T=1 and T=0.6.
