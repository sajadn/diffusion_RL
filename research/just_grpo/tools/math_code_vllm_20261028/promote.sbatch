#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/snorouzi/RL-Rebase/RL-research/research/just_grpo/tools/math_code_vllm_20261028/env.sh
smoke_id=${1:?smoke job ID}
task=${2:?math or code}
variant=${3:?plain or diffaux}
[[ "$smoke_id" =~ ^[0-9]+$ ]]
[[ "$task" == math || "$task" == code ]]
[[ "$variant" == plain || "$variant" == diffaux ]]
tools="$REPO_DIR/research/just_grpo/tools/math_code_vllm_20261028"
output="$RUN_ROOT/${task}_${variant}/smoke_${smoke_id}"
# Serialize promotion of this experiment, including manual reruns of this gate.
exec 9>"$RUN_ROOT/${task}_${variant}/promotion.lock"
flock -n 9
receipt="$RUN_ROOT/${task}_${variant}/long-submissions.txt"
if [[ -s "$receipt" ]]; then
  echo "Long-run submissions already recorded in $receipt; refusing duplicates."
  exit 1
fi
"$POLICY_PYTHON" -I "$tools/check_smoke.py" \
  --job-id "$smoke_id" --output "$output" --task "$task" --kl-limit 0.001
# Four allocations provide up to 16 hours. Each continuation checks the latest
# finalized checkpoint and exits without starting training once step 125 exists.
previous=
for leg in 0 1 2 3; do
  dependency=()
  if [[ -n "$previous" ]]; then dependency=(--dependency="afterany:$previous"); fi
  cmd=(sbatch --parsable -A coreai_dlalgo_llm -p batch --nodes=16 --time=04:00:00
    --job-name="llada_${task}_${variant}_long_20261028_leg${leg}"
    --output="$RUN_ROOT/${task}_${variant}-long-%j.out"
    "${dependency[@]}" "$tools/run.sbatch" long "$task" "$variant" "$leg")
  printf 'Submitting: '; printf '%q ' "${cmd[@]}"; printf '\n'
  # Record attempts before calling sbatch: an RPC timeout may still submit a job.
  printf 'attempt leg=%s smoke=%s\n' "$leg" "$smoke_id" >> "$receipt"
  previous=$("${cmd[@]}")
  [[ "$previous" =~ ^[0-9]+$ ]]
  printf 'submitted leg=%s job=%s\n' "$leg" "$previous" | tee -a "$receipt"
done
