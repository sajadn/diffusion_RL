# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
export RUN_ROOT=/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/llada_gsm8k_vllm_20261027
export REPO_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/RL-Rebase/RL-research
export POLICY_PYTHON=/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_mb_3rdparty_sglagn_local_fork/bin/python
export MODEL=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
export CONTAINER_IMAGE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh
export CUDA_COMPAT_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/cuda_compat_13.1/lib.real
export POLICY_EXTRA_PYTHONPATH="$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:/lustre/fsw/portfolios/coreai/users/snorouzi/just_grpo_megatron_overlay:/lustre/fsw/portfolios/coreai/users/snorouzi/ar_sc_overlay_20261001"
export NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/llada128_pair_retry_20261010_074524/smoke_justgrpo/initial_megatron_cache
export NRL_NCCL_LIBRARY=/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/nemotron_dllm_792ab07/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2
export UV_TEST_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/llada128_pair_retry_20261010_074524/uv_test_cache
export JUSTGRPO_REFERENCE_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/datasets/justgrpo_code/reference
export LLADA_CODE_DATA_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/datasets/justgrpo_code/prepared
export EXPANDABLE_SEGMENTS=false NRL_REFIT_BUFFER_MEMORY_RATIO=0.02
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset GENERATION_PYTHON NRL_LLADA_PY_EXECUTABLE NCCL_DEBUG_FILE EVALUATION_ONLY TRAINING_CONTINUATION WANDB_RUN_ID WANDB_RESUME
export HF_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home
export LLADA_CODE_MAX_PROMPT_TOKENS=1150
export LLADA_CODE_SEQUENCE_LENGTH=1536

export GENERATION_PYTHON=/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/vllm-nemotron-dllm/tools/llada/generation-python
