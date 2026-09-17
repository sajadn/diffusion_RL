#!/usr/bin/env bash
# Stage SciCode data for the NeMo-Skills `scicode` benchmark. RUN THIS ON THE LOGIN NODE.
#
# Two reasons this is a separate step rather than `ns prepare_data scicode`:
#   1. Compute nodes have no HuggingFace egress, and prepare.py calls load_dataset().
#   2. The eval container's site-packages copy of nemo_skills/dataset/scicode/ ships
#      the code but no data files, and its squashfs root is not a place to persist them.
#
# Reproduces nemo_skills/dataset/scicode/prepare.py exactly, minus the `datasets` dep:
#   problems_dev.jsonl  -> dev.jsonl       (HF "validation" split, 15 problems)
#   problems_test.jsonl -> test.jsonl      (HF "test" split, 65 problems / 288 subproblems)
#   dev then test       -> test_aai.jsonl  (80 problems; the split the `aai` group evaluates)
# The concatenation order matters: prepare.py iterates {validation: dev, test: test}.
#
# test_data.h5 carries the numerical targets every SciCode test case asserts against.
# The evaluator hard-fails unless it is visible at /data/test_data.h5 inside the job.
# Upstream SciCode distributes it via Google Drive, which is not reachable from here,
# so this pulls a HuggingFace mirror and verifies it by checksum.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/eval_data/scicode}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/hf_token.txt}"

# sha256 of the mirror as staged on 2026-09-09. A mismatch means the mirror moved or
# the download truncated; a truncated h5 does not error, it silently fails test cases.
TEST_DATA_SHA256="48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"

PROBLEMS_REPO="SciCode1/SciCode"
TEST_DATA_REPO="Srimadh/Scicode-test-data-h5"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

if [[ -r "${HF_TOKEN_FILE}" ]]; then
  HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
else
  echo "warning: ${HF_TOKEN_FILE} not readable; trying unauthenticated download" >&2
  HF_TOKEN=""
fi

fetch() {
  local repo="$1" file="$2" dest="$3"
  echo "fetching ${repo}/${file} -> ${dest}"
  curl -sSLf -H "Authorization: Bearer ${HF_TOKEN}" -o "${dest}" \
    "https://huggingface.co/datasets/${repo}/resolve/main/${file}"
}

fetch "${PROBLEMS_REPO}" problems_dev.jsonl problems_dev.jsonl
fetch "${PROBLEMS_REPO}" problems_test.jsonl problems_test.jsonl

cp problems_dev.jsonl dev.jsonl
cp problems_test.jsonl test.jsonl
cat dev.jsonl test.jsonl > test_aai.jsonl

# Guard the counts: a silently-empty split would otherwise show up as a suspiciously
# good score on a handful of problems rather than as a failure.
for split_and_count in "dev.jsonl 15" "test.jsonl 65" "test_aai.jsonl 80"; do
  set -- ${split_and_count}
  actual="$(wc -l < "$1")"
  if [[ "${actual}" -ne "$2" ]]; then
    echo "ERROR: $1 has ${actual} problems, expected $2" >&2
    exit 1
  fi
done

if [[ -f test_data.h5 ]] \
   && [[ "$(sha256sum test_data.h5 | cut -d' ' -f1)" == "${TEST_DATA_SHA256}" ]]; then
  echo "test_data.h5 already staged and matches checksum"
else
  fetch "${TEST_DATA_REPO}" test_data.h5 test_data.h5
  actual_sha="$(sha256sum test_data.h5 | cut -d' ' -f1)"
  if [[ "${actual_sha}" != "${TEST_DATA_SHA256}" ]]; then
    echo "ERROR: test_data.h5 sha256 ${actual_sha} != expected ${TEST_DATA_SHA256}" >&2
    exit 1
  fi
fi

echo
echo "staged in ${DATA_DIR}:"
ls -la dev.jsonl test.jsonl test_aai.jsonl test_data.h5
