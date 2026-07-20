#!/usr/bin/env bash
#SBATCH --job-name=c86p-regression
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI/c86p_regression_logs/%x-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI/c86p_regression_logs/%x-%j.err
set -euo pipefail

suite="${1:?focused|c65|c23|full required}"
repo=/home/infres/yinwang/CMI_AAAI_oaci
python=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python
cd "$repo"

head_commit=$(git rev-parse HEAD)
remote_commit=$(git rev-parse origin/oaci)
if [[ "$head_commit" != "$remote_commit" ]]; then
  echo "C86P regression refused: HEAD != origin/oaci" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "C86P regression refused: worktree is not clean" >&2
  exit 4
fi

mapfile -t tests < <("$python" -m oaci.multidataset.c84r_regression_suite "$suite")
if [[ "$suite" == "focused" ]]; then
  tests+=(
    oaci/tests/test_c86_active_program_protocol.py
    oaci/tests/test_c86_dataset_eligibility.py
    oaci/tests/test_c86_active_shadow_contract.py
  )
fi
if [[ "${#tests[@]}" -eq 0 ]]; then
  echo "C86P regression refused: empty suite" >&2
  exit 5
fi

export PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="/tmp/c86p-reg-${suite}-${SLURM_JOB_ID:-local}/pycache"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Each assertion below was correct at its historical pre-authorization
# milestone and was superseded by a later accepted, scope-specific execution.
deselect=(
  --deselect=oaci/tests/test_c79p_post_seed3_protocol.py::test_real_execution_fails_closed_without_future_authorization_record
  --deselect=oaci/tests/test_c79p_post_seed3_protocol.py::test_show_binding_contract_is_the_only_unauthorized_adapter_command
  --deselect=oaci/tests/test_c79p_post_seed3_protocol.py::test_unauthorized_command_does_not_import_training_or_EEG_modules
  --deselect=oaci/tests/test_c85_synthetic_contract.py::test_no_c85_real_data_or_active_execution_lock_exists
  --deselect=oaci/tests/test_c85r_protocol_lock.py::test_no_c85t_result_or_authorized_execution_exists
  --deselect=oaci/tests/test_c85ep_input_availability.py::test_no_c85e_lock_authorization_or_analysis_implementation_exists
  --deselect=oaci/tests/test_c85urp_isolation.py::test_confirmatory_results_and_theorem_statuses_are_immutable
  --deselect=oaci/tests/test_c85urp_lock.py::test_c85urp_has_no_authorization_execution_or_c85e_lock
  --deselect=oaci/tests/test_c85ur1_lock.py::test_no_authorization_real_execution_or_downstream_lock
  --deselect=oaci/tests/test_c85tr2_lock.py::test_no_authorization_result_proof_or_status_transition_exists
  --deselect=oaci/tests/test_c85vp_execution_lock.py::test_c85vp_has_no_authorization_result_or_status_transition
  --deselect=oaci/tests/test_c85e_execution_lock.py::test_lock_chronology_and_no_authorization_or_real_execution
)

printf 'C86P_REGRESSION suite=%s commit=%s python=%s CPU=32 GPU=0\n' \
  "$suite" "$head_commit" "$python"
"$python" -m pytest -q -rs \
  --basetemp="/tmp/c86p-pytest-${suite}-${SLURM_JOB_ID:-local}" \
  -o "cache_dir=/tmp/c86p-pytest-cache-${suite}-${SLURM_JOB_ID:-local}" \
  "${deselect[@]}" "${tests[@]}"
