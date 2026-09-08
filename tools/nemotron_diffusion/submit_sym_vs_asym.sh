#!/bin/bash
#SBATCH --job-name=symasym
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/home/snorouzi/logs/symasym_%j.log

# Symmetric (TE dense sbd_block_diff bias) vs asymmetric (flex BlockMask)
# attention, fwd+bwd, cp=1, tp=8, real 8B v1.5. One process per (shape,
# variant, flex mode) so the symmetric arm's 34 dense [1,1,S,S] biases cannot
# contaminate another arm's peak-memory reading.

set -u

PY=/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_gsm8k_nd3b_sglang_a652eb48_mb500dac75/bin/python
IMG=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh
HF=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home
REPO=/home/snorouzi/diffusion_RL/RL
SCRIPT=tools/nemotron_diffusion/bench_sym_vs_asym.py

run_one () {
  local P=$1 R=$2 VARIANT=$3 MODE=$4
  echo "########## P=$P R=$R variant=$VARIANT flex_mode=$MODE ##########"
  srun --nodes=1 --ntasks=1 \
    --container-image="$IMG" \
    --container-mounts=/home/snorouzi:/home/snorouzi,/lustre:/lustre \
    --container-workdir="$REPO" \
    bash -c "export HF_HOME=$HF MEGATRON_CONFIG_LOCK_DIR=$HF/megatron_config_locks \
               DIFFU_FLEX_COMPILE_MODE=$MODE; \
             $PY -m torch.distributed.run --nproc_per_node=8 --master_port=29511 \
               $SCRIPT --tp 8 --prompt $P --response $R --variant $VARIANT \
               --iters 5 --warmup 2" 2>&1
  echo
}

# Shipped flex default, full shape ladder.
# Prompt:response 1:2 mirrors the CP bench's primary 16K+32K shape, scaled down
# until the symmetric arm's O(S^2) dense bias fits (34 * (2(P+R))^2 * 2 B).
for shape in "1024 2048" "2048 4096" "4096 8192" "4096 1024"; do
  set -- $shape
  for v in symmetric_te symmetric_flex asymmetric; do
    run_one $1 $2 $v max-autotune-no-cudagraphs
  done
done

# Autotune control: the CP bench found `default` FASTER than the shipped
# max-autotune at production shapes, so the flex arms are re-run without it.
# The symmetric arm is TE/cuDNN and unaffected, so it is not repeated.
for shape in "2048 4096" "4096 8192"; do
  set -- $shape
  for v in symmetric_flex asymmetric; do
    run_one $1 $2 $v default
  done
done

echo "########## SUMMARY ##########"
grep -h "RESULT" /home/snorouzi/logs/symasym_${SLURM_JOB_ID}.log
