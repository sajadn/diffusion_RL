#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SciCode evaluation via NeMo-Skills. RUNS INSIDE THE PYXIS CONTAINER -- submit with
# submit_benchmark_eval.sh, do not invoke directly on the login node.
#
# SciCode is a multi-step benchmark: each problem is a chain of subproblems, and the
# model's own code for step i-1 is fed back as context for step i. NeMo-Skills ships
# that protocol (nemo_skills/inference/eval/scicode.py) along with the AA-matched
# `background` prompt, so this script only has to stand up the three things the
# container does not provide: the data, a code-execution sandbox, and a model server.
#
# Everything runs in ONE container on ONE node. That is deliberate: NeMo-Skills would
# normally start the sandbox as a second container, which needs docker (`ns` local
# executor) or multi-container pyxis. Neither is available here, so the sandbox runs
# as a plain Flask process alongside the server and the eval client.
set -euo pipefail

OUTDIR="${OUTDIR:?OUTDIR must be set}"
MODEL="${MODEL:?MODEL must be set}"
TOKENIZER="${TOKENIZER:-${MODEL}}"

SCICODE_DATA_DIR="${SCICODE_DATA_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/eval_data/scicode}"
BENCHMARK_SPEC="${BENCHMARK_SPEC:-scicode}"   # `scicode` = greedy; `scicode:3` = AA's 3 repeats
EXPNAME="${EXPNAME:-scicode-eval}"

# The sandbox is a single-process Flask app (upstream's multi-worker deployment is a
# docker/uwsgi image we cannot build here). Each concurrent test opens an IPython
# session that is never freed, so oversubscribing it wedges the run rather than
# slowing it down. 16 is the ceiling established by the ns_tools runs.
NUM_PARALLEL_REQUESTS="${NUM_PARALLEL_REQUESTS:-16}"
SANDBOX_TIMEOUT="${SANDBOX_TIMEOUT:-30.0}"
SANDBOX_PORT="${SANDBOX_PORT:-6000}"

# Empty means the benchmark default, which in this container is `test_aai`: dev+test,
# 80 problems, chosen upstream for consistency with AAI. SPLIT=test restricts to the
# 65-problem test split.
SPLIT="${SPLIT:-test_aai}"

# The dataset module in this container pins ++prompt_config=eval/scicode/default, but
# its own SciCodeGenerationConfig defaults to `background`, and upstream has since
# switched the dataset to `background` too, noting it is the prompt Artificial Analysis
# actually sends to the endpoint. `background` is also the only choice consistent with
# with_background=True, which is what renders the previous steps. User ++args are
# appended after the dataset GENERATION_ARGS, so this wins.
PROMPT_CONFIG="${PROMPT_CONFIG:-eval/scicode/background}"

SERVER_PORT="${SERVER_PORT:-32000}"
NS_VENV="${NS_VENV:-/opt/nemo_rl_venv}"
VENV="${VENV:-/lustre/fsw/portfolios/coreai/users/snorouzi/sglang_nemotron_torch291_cu129_uvpy312_venv}"
SGLANG_REPO="${SGLANG_REPO:-/home/snorouzi/code/sglang-nemotron-dllm-a652eb48}"

DLLM_ALGORITHM="${DLLM_ALGORITHM:-FastDiffuser}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-32}"
THRESHOLD="${THRESHOLD:-0.9}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:--1}"
SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-0}"
SELECTION_POLICY="${SELECTION_POLICY:-confidence}"
CAUSAL_CONTEXT="${CAUSAL_CONTEXT:-true}"
JSON_MODEL_OVERRIDE_ARGS="${JSON_MODEL_OVERRIDE_ARGS:-}"

DTYPE="${DTYPE:-bfloat16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
# SciCode accumulates: step i is prompted with the model's own code for steps 1..i-1.
# The test_aai split has 341 subproblems over 80 problems, one of them 15 steps deep,
# and static prompt text alone reaches ~23k chars before any generated code is added.
# Exceeding the window is not an error -- the generation module marks every remaining
# substep _ran_out_of_context_ and they score zero -- so an undersized context returns
# a quietly wrong number rather than failing. 16384 is the checkpoint's PRE-yarn window
# (rope_scaling factor 16 takes it to 262144), so it is the one value to avoid.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# 8k thinking + 8k answer. THINKING_BUDGET is enforced server-side (vLLM only);
# TOKENS_TO_GENERATE is the total, so the answer gets whatever thinking does not use.
# A zero THINKING_BUDGET omits the reasoning cap, as in the standalone evaluator.
TOKENS_TO_GENERATE="${TOKENS_TO_GENERATE:-16384}"
THINKING_BUDGET="${THINKING_BUDGET:-8192}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"

BACKEND="${BACKEND:-vllm}"
# Editable install resolving to vllm-nemotron-dllm @ nemotron-dllm-threshold, which is
# the branch carrying the thinking-budget work. python-compat prepends the CUDA 13.1
# compat libs; the bare python fails with "driver too old".
VLLM_PY="${VLLM_PY:-/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/nemotron_dllm_792ab07/bin/python-compat}"

mkdir -p "${OUTDIR}"
SCRIPT_DIR_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS_PY="${NS_VENV}/bin/python"

echo "=== SciCode eval ==="
echo "MODEL=${MODEL}"
echo "BENCHMARK_SPEC=${BENCHMARK_SPEC}"
echo "DLLM_ALGORITHM=${DLLM_ALGORITHM} BLOCK_SIZE=${BLOCK_SIZE} MAX_STEPS=${MAX_STEPS} THRESHOLD=${THRESHOLD}"
echo "NUM_PARALLEL_REQUESTS=${NUM_PARALLEL_REQUESTS}"
echo "OUTDIR=${OUTDIR}"

# --- 1. Data -----------------------------------------------------------------
# The dataset module resolves its data files relative to its own package directory,
# and `data_dir` is only consulted for benchmarks that are NOT built in, so for
# `scicode` the staged splits have to be copied into site-packages. The container
# root is a writable enroot overlay, so this lasts exactly as long as the job.
for f in test.jsonl test_aai.jsonl dev.jsonl test_data.h5; do
  if [[ ! -f "${SCICODE_DATA_DIR}/${f}" ]]; then
    echo "ERROR: ${SCICODE_DATA_DIR}/${f} missing. Run tools/scicode/stage_scicode_data.sh on the login node." >&2
    exit 1
  fi
done

NS_SCICODE_DIR="$(${NS_PY} -c 'import nemo_skills, pathlib; print(pathlib.Path(nemo_skills.__file__).parent / "dataset" / "scicode")')"
echo "copying splits into ${NS_SCICODE_DIR}"
cp "${SCICODE_DATA_DIR}"/dev.jsonl "${SCICODE_DATA_DIR}"/test.jsonl "${SCICODE_DATA_DIR}"/test_aai.jsonl "${NS_SCICODE_DIR}/"

# eval_prefix in scicode_utils.py hardcodes H5PY_FILE="/data/test_data.h5"; the
# evaluator aborts if it is not there. Symlink rather than copy: it is 1GB.
mkdir -p /data
ln -sf "${SCICODE_DATA_DIR}/test_data.h5" /data/test_data.h5

# --- 2. h5py -----------------------------------------------------------------
# Every SciCode test case runs eval_prefix, which imports h5py to read the targets.
# The eval container does not ship h5py and compute nodes have no egress, so install
# the staged wheel offline into a directory on PYTHONPATH rather than into the venv.
# The wheel is unzipped rather than pip-installed: this venv ships no pip (which is
# why the other eval pipelines here bootstrap it with ensurepip), and a wheel is just
# a zip. Unpacking keeps h5py.libs next to the extension modules, which is what their
# RPATH expects, so the bundled HDF5 resolves without a system install.
H5PY_TARGET="${OUTDIR}/pydeps"
if ! ${NS_PY} -c 'import h5py' 2>/dev/null; then
  echo "unpacking staged h5py wheel into ${H5PY_TARGET}"
  mkdir -p "${H5PY_TARGET}"
  ${NS_PY} - "${H5PY_TARGET}" "${SCICODE_DATA_DIR}"/wheels/h5py-*.whl <<'PYWHEEL'
import sys, zipfile
target, wheel = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(wheel) as z:
    z.extractall(target)
print("unpacked", wheel)
PYWHEEL
  export PYTHONPATH="${H5PY_TARGET}:${PYTHONPATH:-}"
fi
# The sandbox process is what actually runs eval_prefix, so h5py has to be importable
# there; PYTHONPATH is exported before the sandbox starts below.
${NS_PY} -c 'import h5py, scipy; print("h5py", h5py.__version__, "scipy", scipy.__version__)'

# Register cleanup before starting either child so startup failures cannot leak one.
SERVER_PID=""
SANDBOX_PID=""
cleanup() {
  for pid in "${SERVER_PID}" "${SANDBOX_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

# --- 3. Sandbox --------------------------------------------------------------
SANDBOX_LOG="${OUTDIR}/sandbox.log"
echo "starting local sandbox on 127.0.0.1:${SANDBOX_PORT} (log: ${SANDBOX_LOG})"
env PYTHONPATH="${PYTHONPATH:-}" \
  ${NS_PY} "${SCRIPT_DIR_SELF}/../benchmark_eval/run_sandbox.py" "${SANDBOX_PORT}" \
  >"${SANDBOX_LOG}" 2>&1 &
SANDBOX_PID=$!

# --- 4. Model server ---------------------------------------------------------
# BACKEND=vllm is the default because it is the only backend that can enforce a
# thinking budget on this model. SGLang's ThinkingBudgetLogitProcessor is applied in
# layers/sampler.py, and FastDiffuser never routes through that Sampler (it does its
# own Gumbel sampling), so the budget is silently inert there. The vLLM fork carries
# both halves on branch nemotron-dllm-threshold: upstream PR #46727 for AR and
# commit 99064e7360 for diffusion (budget enforced at each block commit by seeding
# the canvas with the reasoning-end token).
SERVER_LOG="${OUTDIR}/server.log"

if [[ "${BACKEND}" == "vllm" ]]; then
  # Diffusion decode on vLLM. The checkpoint's own architectures entry is not in the
  # fork registry, so name the block-diffusion class explicitly; TRITON_ATTN because
  # block diffusion passes a per-sequence is_causal tensor that FA3 rejects.
  if [[ "${SELECTION_POLICY}" != "confidence" ]]; then
    echo "Unsupported SciCode vLLM selection policy: ${SELECTION_POLICY}" >&2
    exit 2
  fi
  vllm_selection="confidence_threshold"
  diffusion_config="{\"selection_policy\": \"${vllm_selection}\", \"canvas_length\": ${BLOCK_SIZE}, \"max_denoising_steps\": ${MAX_STEPS}, \"temperature\": ${TEMPERATURE}, \"confidence_threshold\": ${THRESHOLD}}"
  server_args=(
    --model "${MODEL}"
    --tokenizer "${TOKENIZER}"
    --served-model-name default
    --trust-remote-code
    --host 127.0.0.1
    --port "${SERVER_PORT}"
    --tensor-parallel-size "${TP_SIZE:-1}"
    --dtype "${DTYPE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${MEM_FRACTION_STATIC}"
    --enforce-eager
    --max-num-seqs "${MAX_RUNNING_REQUESTS}"
    --seed "${SERVER_RANDOM_SEED}"
    --attention-backend TRITON_ATTN
    # Gives ReasoningConfig the delimiters to derive token ids from. Required for
    # thinking_token_budget to do anything; <think>/</think> are single tokens here
    # (12/13), which the diffusion implementation requires.
    --reasoning-config "{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": \"</think>\"}"
  )
  if [[ "${DLLM_ALGORITHM}" == "AR" ]]; then
    server_args+=(--hf-overrides '{"architectures": ["NemotronLabsDiffusionForCausalLM"]}')
  elif [[ "${DLLM_ALGORITHM}" == "FastDiffuser" ]]; then
    server_args+=(--hf-overrides '{"architectures": ["NemotronLabsDiffusionModel"]}' --diffusion-config "${diffusion_config}")
  else
    echo "Unsupported SciCode vLLM algorithm: ${DLLM_ALGORITHM}" >&2
    exit 2
  fi
  echo "--- diffusion config ---"; echo "${diffusion_config}"
  echo "starting vLLM on 127.0.0.1:${SERVER_PORT} (log: ${SERVER_LOG})"
  env -u PYTHONPATH "${VLLM_PY}" -m vllm.entrypoints.openai.api_server "${server_args[@]}" \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
else
  DLLM_CONFIG="${OUTDIR}/dllm_config.yaml"
  {
    echo "algorithm: ${DLLM_ALGORITHM}"
    echo "causal_context: ${CAUSAL_CONTEXT}"
    if [[ "${DLLM_ALGORITHM}" != "AR" ]]; then
      echo "block_size: ${BLOCK_SIZE}"
      echo "max_steps: ${MAX_STEPS}"
      echo "threshold: ${THRESHOLD}"
      echo "selection_policy: ${SELECTION_POLICY}"
    fi
  } >"${DLLM_CONFIG}"
  echo "--- dllm config ---"; cat "${DLLM_CONFIG}"
  server_args=(
    --model-path "${MODEL}"
    --tokenizer-path "${TOKENIZER}"
    --served-model-name default
    --trust-remote-code
    --host 127.0.0.1
    --port "${SERVER_PORT}"
    --tp-size "${TP_SIZE:-1}"
    --dtype "${DTYPE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --attention-backend "${ATTENTION_BACKEND}"
    --context-length "${MAX_MODEL_LEN}"
  )
  if [[ -n "${JSON_MODEL_OVERRIDE_ARGS}" ]]; then
    server_args+=(--json-model-override-args "${JSON_MODEL_OVERRIDE_ARGS}")
  fi
  if [[ "${DLLM_ALGORITHM}" != "NONE" && "${DLLM_ALGORITHM}" != "none" ]]; then
    server_args+=(--dllm-algorithm "${DLLM_ALGORITHM}" --dllm-algorithm-config "${DLLM_CONFIG}")
  fi
  echo "starting SGLang on 127.0.0.1:${SERVER_PORT} (log: ${SERVER_LOG})"
  env PYTHONPATH="${SGLANG_REPO}/python" \
    "${VENV}/bin/python" -m sglang.launch_server "${server_args[@]}" \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
fi



wait_for() {
  local name="$1" url="$2" pid="$3" tries="$4"
  for i in $(seq 1 "${tries}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: ${name} died during startup; last 80 lines:" >&2
      tail -80 "${OUTDIR}/$([[ ${name} == sandbox ]] && echo sandbox.log || echo server.log)" >&2 || true
      exit 1
    fi
    if curl -sf --max-time 3 "${url}" >/dev/null 2>&1; then
      echo "${name} ready after ~$((i * 5))s"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: ${name} not ready after $((tries * 5))s" >&2
  exit 1
}

wait_for sandbox "http://127.0.0.1:${SANDBOX_PORT}/health" "${SANDBOX_PID}" 24
wait_for server "http://127.0.0.1:${SERVER_PORT}/health" "${SERVER_PID}" 180

# --- 5. Evaluate -------------------------------------------------------------
# No --cluster: NeMo-Skills runs the pipeline in this process rather than submitting
# it. No --with_sandbox either -- that would try to launch a sandbox container; ours
# is already up and LocalSandbox just talks to 127.0.0.1:6000.
export NEMO_SKILLS_SANDBOX_HOST=127.0.0.1
export NEMO_SKILLS_SANDBOX_PORT="${SANDBOX_PORT}"

# This container leaks one IPython session per test into the single-process sandbox.
# Unpatched, test_aai wedges the run long before it finishes.
${NS_PY} "${SCRIPT_DIR_SELF}/patch_scicode_session_leak.py"

# Deliberately NOT `ns eval`. That drives the NeMo-Run/torchx pipeline, which is built
# for submitting jobs to a cluster -- but we are already inside the job. Locally it
# tries to launch three separate torchx apps (generation, its own sandbox, and
# summarize_results) and dies in the local scheduler with:
#     AssertionError: no app_id collisions expected since uuid4 suffix is used
# It would also start a second sandbox on top of the one above. Calling the two
# modules directly is what the pipeline would have run anyway -- the arguments below
# are lifted verbatim from the command it printed.
#
# The generation module evaluates inline: with ++eval_type set it runs the scicode
# evaluator over its own output, so there is no separate evaluation step.
# vLLM's diffusion path rejects any request carrying a seed:
#   _validate_diffusion() raises if `self.seed is not None`, and NeMo-Skills always
#   sends "seed": random_seed (default 0), so every request 400s with
#   "Diffusion models only support temperature 0 (greedy) or 1 ...".
# temperature 0.0 is already fine, and min_p/top_k are popped by the OpenAI client.
# ++inference.random_seed=null does not work: the field is typed `int`, so Hydra
# fails with "Error merging override". Patch the client to omit the key instead.
if [[ "${BACKEND}" == "vllm" ]]; then
  export NEMO_SKILLS_OMIT_SEED=0
  if [[ "${DLLM_ALGORITHM}" != "AR" ]]; then
    export NEMO_SKILLS_OMIT_SEED=1
  fi
  "${NS_PY}" "${SCRIPT_DIR_SELF}/patch_ns_omit_seed.py"
fi

REQUEST_TEMPERATURE="${TEMPERATURE}"
if [[ "${BACKEND}" == "vllm" && "${DLLM_ALGORITHM}" != "AR" ]]; then
  REQUEST_TEMPERATURE="$(${NS_PY} -c 'import sys; print(0 if float(sys.argv[1]) == 0 else 1)' "${TEMPERATURE}")"
fi
RESULTS_DIR="${OUTDIR}/eval-results/scicode"
mkdir -p "${RESULTS_DIR}"

# Zero means no separate reasoning cap, not zero reasoning tokens in vLLM.
thinking_args=()
if (( THINKING_BUDGET > 0 )); then
  thinking_args+=("++inference.extra_body.thinking_token_budget=${THINKING_BUDGET}")
fi

set -x
"${NS_PY}" -m nemo_skills.inference.eval.scicode \
  ++skip_filled=True \
  ++max_concurrent_requests="${NUM_PARALLEL_REQUESTS}" \
  ++parse_reasoning="${ENABLE_THINKING}" \
  ++input_file="${SCICODE_INPUT_FILE:-${NS_SCICODE_DIR}/${SPLIT}.jsonl}" \
  ++output_file="${RESULTS_DIR}/output.jsonl" \
  ++wait_for_sandbox=true \
  ++eval_type=scicode \
  ++eval_config.split="${SPLIT}" \
  ++inference.temperature="${REQUEST_TEMPERATURE}" \
  ++inference.top_p="${TOP_P}" \
  ++inference.top_k="${TOP_K}" \
  ++inference.random_seed="${SERVER_RANDOM_SEED}" \
  ++inference.tokens_to_generate="${TOKENS_TO_GENERATE}" \
  ++prompt_config="${PROMPT_CONFIG}" \
  ++chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
  "${thinking_args[@]}" \
  ++eval_config.num_parallel_requests="${NUM_PARALLEL_REQUESTS}" \
  ++eval_config.timeout="${SANDBOX_TIMEOUT}" \
  ++server.server_type=openai \
  ++server.base_url="http://127.0.0.1:${SERVER_PORT}/v1" \
  ++server.model=default \
  ${NS_EXTRA_ARGS:-}

# Record this installed version's prefilled steps for exact aggregation checks.
"${NS_PY}" - "${RESULTS_DIR}/output.jsonl" "${OUTDIR}/expected_steps.json" <<'PYMETA'
import json
import sys
from nemo_skills.inference.eval.scicode_utils import prefilled_steps_code
with open(sys.argv[1]) as stream:
    rows = [json.loads(line) for line in stream]
expected = {
    str(row["problem_id"]): [
        f"{row['problem_id']}.{i + 1}" for i in range(len(row["sub_steps"]))
        if (row["problem_id"], i) not in prefilled_steps_code
    ] for row in rows
}
with open(sys.argv[2], "w") as stream:
    json.dump(expected, stream)
PYMETA

"${NS_PY}" -m nemo_skills.pipeline.summarize_results "${OUTDIR}"
set +x

echo "=== done. results under ${OUTDIR}/eval-results/scicode ==="
find "${OUTDIR}/eval-results" -name "metrics.json" -exec echo {} \; -exec cat {} \; 2>/dev/null || true
