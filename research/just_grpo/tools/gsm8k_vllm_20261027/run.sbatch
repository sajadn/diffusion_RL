#!/usr/bin/env bash
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/snorouzi/RL-Rebase/RL-research/research/just_grpo/tools/gsm8k_vllm_20261027/env.sh
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
export WANDB_RUN_NAME="llada_gsm8k_canvas256_${variant}_vllm_20261027"
port_base=$((18000 + SLURM_JOB_ID % 1000 * 20))
export PORT=$port_base RAY_CLIENT_SERVER_PORT=$((port_base+1)) DASHBOARD_PORT=$((port_base+2))
export NODE_MANAGER_PORT=$((port_base+3)) OBJECT_MANAGER_PORT=$((port_base+5))
export RUNTIME_ENV_AGENT_PORT=$((port_base+7)) DASHBOARD_AGENT_GRPC_PORT=$((port_base+9))
export METRICS_EXPORT_PORT=$((port_base+11)) DASHBOARD_AGENT_LISTEN_PORT=$((port_base+13))
mkdir -p "$OUTPUT_DIR"
exec bash "$REPO_DIR/research/just_grpo/tools/llada_distributed.sbatch"
