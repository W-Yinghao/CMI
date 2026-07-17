#!/usr/bin/env bash
#SBATCH --job-name=c85ep-regression
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI/c85ep_regression_logs/%x-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI/c85ep_regression_logs/%x-%j.err
set -euo pipefail

suite="${1:?focused|c65|c23|full required}"
repo=/home/infres/yinwang/CMI_AAAI_oaci
python=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python
cd "$repo"

head_commit=$(git rev-parse HEAD)
remote_commit=$(git rev-parse origin/oaci)
if [[ "$head_commit" != "$remote_commit" ]]; then
  echo "C85EP regression refused: HEAD != origin/oaci" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "C85EP regression refused: worktree is not clean" >&2
  exit 4
fi

mapfile -t tests < <("$python" -m oaci.multidataset.c84r_regression_suite "$suite")
if [[ "$suite" == "focused" ]]; then
  tests+=(oaci/tests/test_c85ep_input_availability.py)
fi
if [[ "${#tests[@]}" -eq 0 ]]; then
  echo "C85EP regression refused: empty suite" >&2
  exit 5
fi

export PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="/tmp/c85ep-reg-${suite}-${SLURM_JOB_ID:-local}/pycache"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

deselect=(
  --deselect=oaci/tests/test_c79p_post_seed3_protocol.py::test_real_execution_fails_closed_without_future_authorization_record
  --deselect=oaci/tests/test_c79p_post_seed3_protocol.py::test_show_binding_contract_is_the_only_unauthorized_adapter_command
  --deselect=oaci/tests/test_c79p_post_seed3_protocol.py::test_unauthorized_command_does_not_import_training_or_EEG_modules
  --deselect=oaci/tests/test_c85tr2_lock.py::test_no_authorization_result_proof_or_status_transition_exists
  --deselect=oaci/tests/test_c85vp_execution_lock.py::test_c85vp_has_no_authorization_result_or_status_transition
)

printf 'C85EP_REGRESSION suite=%s commit=%s python=%s CPU=48 GPU=0\n' \
  "$suite" "$head_commit" "$python"
"$python" -m pytest -q -rs \
  --basetemp="/tmp/c85ep-pytest-${suite}-${SLURM_JOB_ID:-local}" \
  -o "cache_dir=/tmp/c85ep-pytest-cache-${suite}-${SLURM_JOB_ID:-local}" \
  "${deselect[@]}" "${tests[@]}"
