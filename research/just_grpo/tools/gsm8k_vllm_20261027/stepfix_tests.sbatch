#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/llada_gsm8k_vllm_20261027/env.sh
export PYTHONPATH="$REPO_DIR:$REPO_DIR/research/block_diffusion:$REPO_DIR/research/just_grpo:$POLICY_EXTRA_PYTHONPATH"
export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset RAY_ADDRESS
export RAY_TMPDIR="/tmp/llada_stepfix_tests_${SLURM_JOB_ID}"
export UV_CACHE_DIR="$UV_TEST_CACHE_DIR" MPLCONFIGDIR=/tmp/llada-stepfix-tests-mpl
srun --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/home/snorouzi:/home/snorouzi,/lustre:/lustre \
  --container-workdir="$REPO_DIR" \
  /home/snorouzi/.local/bin/uv run --offline --no-project --python "$POLICY_PYTHON" --with pytest \
  python -m pytest tests/unit/utils/test_logger.py::TestWandbLogger \
  research/block_diffusion/tests/unit/test_confidence_sweep.py \
  research/just_grpo/tests/unit/test_llada_megatron.py -q
