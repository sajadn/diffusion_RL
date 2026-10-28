#!/usr/bin/env bash
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/llada_gsm8k_vllm_20261027/env.sh
variant=${1:?plain, diffaux, or smoke}
case "$variant" in
  plain) recipe=just_grpo-gsm8k-llada8b-canvas256-16n8g-vllm-long ;;
  diffaux) recipe=just_grpo-gsm8k-llada8b-canvas256-16n8g-vllm-long-diffaux ;;
  smoke) recipe=just_grpo-gsm8k-llada8b-canvas256-2n8g-vllm-smoke ;;
  *) exit 2 ;;
esac
export OUTPUT_DIR="$RUN_ROOT/$variant" LOG_DIR="$RUN_ROOT/$variant"
export CHECKPOINT_ROOT="$OUTPUT_DIR/checkpoints"
export CONFIG="$REPO_DIR/research/just_grpo/configs/recipes/$recipe.yaml"
export WANDB_RUN_NAME="llada_gsm8k_canvas256_${variant}_vllm_20261027_stepfix"
port_base=$((18000 + SLURM_JOB_ID % 1000 * 20))
export PORT=$port_base RAY_CLIENT_SERVER_PORT=$((port_base+1)) DASHBOARD_PORT=$((port_base+2))
export NODE_MANAGER_PORT=$((port_base+3)) OBJECT_MANAGER_PORT=$((port_base+5))
export RUNTIME_ENV_AGENT_PORT=$((port_base+7)) DASHBOARD_AGENT_GRPC_PORT=$((port_base+9))
export METRICS_EXPORT_PORT=$((port_base+11)) DASHBOARD_AGENT_LISTEN_PORT=$((port_base+13))
mkdir -p "$OUTPUT_DIR"
leg=${2:-0}
if [[ "$leg" != 0 ]]; then
  checkpoint_step=$("$POLICY_PYTHON" - <<'CHECKPOINT'
import json, os
from pathlib import Path
root = Path(os.environ["CHECKPOINT_ROOT"])
complete = [p for p in root.glob("step_*")
            if p.name.removeprefix("step_").isdigit()
            and (p / "training_info.json").is_file()]
if not complete:
    raise SystemExit("Continuation requires a finalized checkpoint")
latest = max(complete, key=lambda p: int(p.name.removeprefix("step_")))
step = json.loads((latest / "training_info.json").read_text())["total_steps"]
if step < 1:
    raise SystemExit("Continuation requires at least one completed update")
print(step)
CHECKPOINT
  )
  if (( checkpoint_step >= 125 )); then
    echo "All 125 updates are complete; no continuation needed."
    exit 0
  fi
  test -s "$OUTPUT_DIR/wandb_url.txt"
  export WANDB_RUN_ID WANDB_RESUME=must
  WANDB_RUN_ID=$(awk -F/ '{print $NF}' "$OUTPUT_DIR/wandb_url.txt")
else
  if compgen -G "$CHECKPOINT_ROOT/step_*" >/dev/null; then
    echo "Fresh run requires a checkpoint directory without previous steps" >&2
    exit 1
  fi
fi
exec bash "$REPO_DIR/research/just_grpo/tools/llada_distributed.sbatch"
