#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/snorouzi/RL-Rebase/RL-research/research/just_grpo/tools/math_code_vllm_20261028/env.sh
stage=${1:?smoke or long}
task=${2:?math, code, or gsm8k}
variant=${3:?plain, diffaux, diffaux02, or diffaux03}
leg=${4:-0}
case "$stage" in
  smoke) nodes=2; size=2n8g ;;
  long) nodes=16; size=16n8g ;;
  *) exit 2 ;;
esac
[[ "$SLURM_NNODES" == "$nodes" ]]
case "$task" in
  math) recipe="just_grpo-math-llada8b-canvas256-${size}-vllm-${stage}" ;;
  code) recipe="just_grpo-acecode-llada8b-${size}-vllm-humaneval-${stage}" ;;
  gsm8k)
    [[ "$stage" == long && ( "$variant" == diffaux02 || "$variant" == diffaux03 ) ]]
    recipe="just_grpo-gsm8k-llada8b-canvas256-${size}-vllm-${stage}"
    ;;
  *) exit 2 ;;
esac
case "$variant" in
  plain) ;;
  diffaux) recipe+=-diffaux ;;
  diffaux02|diffaux03)
    [[ "$task" == gsm8k ]]
    recipe+="-$variant"
    ;;
  *) exit 2 ;;
esac
export OUTPUT_DIR="$RUN_ROOT/${task}_${variant}/$stage"
if [[ "$stage" == smoke ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR}_${SLURM_JOB_ID}"
fi
mkdir -p "$OUTPUT_DIR"
export LOG_DIR="$OUTPUT_DIR" CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"
export CONFIG="$REPO_DIR/research/just_grpo/configs/recipes/$recipe.yaml"
if [[ "$stage" == smoke ]]; then
  printf 'defaults: %s\n\njust_grpo:\n  max_generation_kl: 0.001\n' "$CONFIG" > "$OUTPUT_DIR/smoke-config.yaml"
  export CONFIG="$OUTPUT_DIR/smoke-config.yaml"
elif [[ "$leg" != 0 ]]; then
  step=$("$POLICY_PYTHON" -I - <<'CHECKPOINT'
import json, os
from pathlib import Path
root = Path(os.environ["CHECKPOINT_ROOT"])
complete = [p for p in root.glob("step_*") if p.name[5:].isdigit() and (p / "training_info.json").is_file()]
if not complete:
    raise SystemExit("Continuation requires a finalized checkpoint")
p = max(complete, key=lambda x: int(x.name[5:]))
step = json.loads((p / "training_info.json").read_text())["total_steps"]
if step < 1:
    raise SystemExit("Continuation requires a completed optimizer update")
print(step)
CHECKPOINT
  )
  if (( step >= 300 )); then
    echo "All 300 updates are complete."
    exit 0
  fi
  test -s "$OUTPUT_DIR/wandb_url.txt"
  export WANDB_RUN_ID WANDB_RESUME=must
  WANDB_RUN_ID=$(awk -F/ '{print $NF}' "$OUTPUT_DIR/wandb_url.txt")
elif compgen -G "$CHECKPOINT_ROOT/step_*" >/dev/null; then
  echo "Fresh run requires an unused checkpoint directory" >&2
  exit 1
fi
if [[ "$task" == code ]]; then
  label=acecode_humaneval_mbpp
elif [[ "$task" == gsm8k ]]; then
  label=gsm8k_canvas256
else
  label=math12k_math500
fi
export WANDB_RUN_NAME="llada_${label}_${variant}_vllm_greedy_20261028"
port_base=$((18000 + SLURM_JOB_ID % 1000 * 20))
export PORT=$port_base RAY_CLIENT_SERVER_PORT=$((port_base+1)) DASHBOARD_PORT=$((port_base+2))
export NODE_MANAGER_PORT=$((port_base+3)) OBJECT_MANAGER_PORT=$((port_base+5))
export RUNTIME_ENV_AGENT_PORT=$((port_base+7)) DASHBOARD_AGENT_GRPC_PORT=$((port_base+9))
export METRICS_EXPORT_PORT=$((port_base+11)) DASHBOARD_AGENT_LISTEN_PORT=$((port_base+13))
git -C "$REPO_DIR" rev-parse HEAD > "$OUTPUT_DIR/source-commit-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$OUTPUT_DIR/job-id.txt"
exec bash "$REPO_DIR/research/just_grpo/tools/llada_distributed.sbatch"
