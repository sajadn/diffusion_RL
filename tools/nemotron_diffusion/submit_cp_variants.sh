#!/bin/bash
#SBATCH --job-name=cpvar
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/home/snorouzi/logs/cpvar_%j.log

# Section 2 of plans/full_cp_analysis.md. Gate first (v1 must equal v2), then
# time each variant in its own process so no allocator carry-over between arms.
# --exclusive and srun --overlap are required: without them the inner srun
# blocks on GPUs the outer allocation already holds.
set -u
PY=/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_gsm8k_nd3b_sglang_a652eb48_mb500dac75/bin/python
IMG=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh
HF=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home
REPO=/home/snorouzi/diffusion_RL/RL
P=${P:-1024}
R=${R:-2048}
TP=${TP:-2}
CP=${CP:-4}
MODE=${MODE:-default}            # flex compile mode
SHARE=${SHARE:-}                 # "--share-rope" to share one cos/sin table across layers
RECOMPUTE=${RECOMPUTE:-}         # "--recompute" for full activation checkpointing

go () {
  srun --overlap --nodes=1 --ntasks=1 --container-image="$IMG" \
    --container-mounts=/home/snorouzi:/home/snorouzi,/lustre:/lustre \
    --container-workdir="$REPO" \
    bash -c "export HF_HOME=$HF MEGATRON_CONFIG_LOCK_DIR=$HF/megatron_config_locks \
               DIFFU_FLEX_COMPILE_MODE=$MODE PYTHONUNBUFFERED=1; \
             $PY -m torch.distributed.run --nproc_per_node=$((TP*CP)) --master_port=29561 \
               tools/nemotron_diffusion/bench_cp_variants.py \
               --tp $TP --cp $CP --prompt $P --response $R $RECOMPUTE $SHARE $*" 2>&1
}

echo "########## GATE: v0/v1/v2 equivalence (tp=$TP cp=$CP) ##########"
go --check --layers 4 | grep -E "^\[cpvar\]|Error|Traceback"

for v in v0 v1 v2 v3 ref; do
  echo "########## $v (tp=$TP cp=$CP P=$P R=$R) ##########"
  go --variant $v --iters 5 --warmup 2 | grep -E "^\[cpvar\]|Error|Traceback"
done
echo "########## DONE ##########"
