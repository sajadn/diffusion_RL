#!/usr/bin/env bash
set -euo pipefail
: "${NRL_NCCL_LIBRARY:?Set the shared NCCL library}"
# Apply the same explicit in-container preload as the successful NLD driver.
export LD_PRELOAD="${NRL_NCCL_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"
if [[ "$CONFIG" == *smoke* && "$CONFIG" != *resume* ]]; then
  export UV_CACHE_DIR="${UV_TEST_CACHE_DIR:?Set the prefetched test cache}"
  /home/snorouzi/.local/bin/uv run --offline --no-project --python "$POLICY_PYTHON" --with pytest \
    python -m pytest research/just_grpo/tests/unit/test_llada_full_canvas.py \
    research/just_grpo/tests/unit/test_megatron_adapter.py \
    research/just_grpo/tests/unit/test_llada_bridge_serialization.py \
    research/just_grpo/tests/unit/test_llada_megatron.py \
    research/just_grpo/tests/unit/test_native_vllm_generation.py \
    research/block_diffusion/tests/unit/test_positive_diffusion_level.py -q
  export UV_CACHE_DIR="${OUTPUT_DIR}/uv_cache"
fi
exec bash research/just_grpo/tools/driver.sh
