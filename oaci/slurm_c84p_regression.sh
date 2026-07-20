#!/usr/bin/env bash
#SBATCH --job-name=c84p-regression
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI/c84p_regression_logs/%x-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI/c84p_regression_logs/%x-%j.err
set -euo pipefail

suite="${1:?focused|c65|c23|full required}"
repo=/home/infres/yinwang/CMI_AAAI_oaci
cd "${repo}"

head_commit=$(git rev-parse HEAD)
remote_commit=$(git rev-parse origin/oaci)
if [[ "${head_commit}" != "${remote_commit}" ]]; then
  echo "C84P regression refused: HEAD != origin/oaci" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "C84P regression refused: worktree is not clean" >&2
  exit 4
fi

export PYTHONPYCACHEPREFIX="/tmp/c84p-reg-${suite}-${SLURM_JOB_ID}/pycache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

case "${suite}" in
  focused)
    tests=(oaci/tests/test_c84_multidataset_external_validity.py)
    ;;
  c65)
    tests=(
      oaci/tests/test_c6[5-9]_*.py
      oaci/tests/test_c7[0-9]_*.py
      oaci/tests/test_c78f_*.py
      oaci/tests/test_c78r_*.py
      oaci/tests/test_c78s_*.py
      oaci/tests/test_c79e_*.py
      oaci/tests/test_c79p_*.py
      oaci/tests/test_c80_*.py
      oaci/tests/test_c80e_*.py
      oaci/tests/test_c80r_*.py
      oaci/tests/test_c81_*.py
      oaci/tests/test_c82_*.py
      oaci/tests/test_c83_*.py
      oaci/tests/test_c84_*.py
    )
    ;;
  c23)
    tests=(
      oaci/tests/test_c2[3-9]_*.py
      oaci/tests/test_c[3-6][0-9]_*.py
      oaci/tests/test_c7[0-9]_*.py
      oaci/tests/test_c78f_*.py
      oaci/tests/test_c78r_*.py
      oaci/tests/test_c78s_*.py
      oaci/tests/test_c79e_*.py
      oaci/tests/test_c79p_*.py
      oaci/tests/test_c80_*.py
      oaci/tests/test_c80e_*.py
      oaci/tests/test_c80r_*.py
      oaci/tests/test_c81_*.py
      oaci/tests/test_c82_*.py
      oaci/tests/test_c83_*.py
      oaci/tests/test_c84_*.py
    )
    ;;
  full)
    tests=(oaci/tests)
    ;;
  *)
    echo "unknown suite: ${suite}" >&2
    exit 2
    ;;
esac

deselect=(
  --deselect=oaci/tests/test_c79p_post_seed3_protocol.py::test_real_execution_fails_closed_without_future_authorization_record
  --deselect=oaci/tests/test_c79p_post_seed3_protocol.py::test_show_binding_contract_is_the_only_unauthorized_adapter_command
  --deselect=oaci/tests/test_c79p_post_seed3_protocol.py::test_unauthorized_command_does_not_import_training_or_EEG_modules
)

/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m pytest -q -rs \
  --basetemp="/tmp/c84p-pytest-${suite}-${SLURM_JOB_ID}" \
  -o "cache_dir=/tmp/c84p-pytest-cache-${suite}-${SLURM_JOB_ID}" \
  "${deselect[@]}" "${tests[@]}"
