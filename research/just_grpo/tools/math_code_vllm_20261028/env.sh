#!/usr/bin/env bash
# Reuse the provisioned policy and vLLM interpreters from the GSM8K campaign.
source /lustre/fsw/portfolios/coreai/users/snorouzi/RL-Rebase/RL-research/research/just_grpo/tools/gsm8k_vllm_20261027/env.sh
export RUN_ROOT=/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/llada_math_code_vllm_20261028
export LLADA_MATH_DATA_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/trace_entropy01_te1_ts06_tripleval_20261014/math12k/data
export LLADA_CODE_DATA_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/datasets/justgrpo_code/prepared_dual_20261028
export LLADA_CODE_MAX_PROMPT_TOKENS=3840 LLADA_CODE_SEQUENCE_LENGTH=4096

# CPU promotion jobs export their memory limit into nested sbatch submissions.
# Let srun use the GPU allocation's memory instead of the parent's 8 GB limit.
case "${SLURM_JOB_PARTITION:-}" in
  batch*) unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU ;;
esac
