#!/usr/bin/env bash
#SBATCH --job-name=c80r-regression
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI/c80r_regression_logs/c80r-regression-%j.stdout
#SBATCH --error=/home/infres/yinwang/CMI_AAAI/c80r_regression_logs/c80r-regression-%j.stderr
set -euo pipefail

suite="${1:?focused|c65|c23|full required}"
cd /home/infres/yinwang/CMI_AAAI/c78f_oaci
export PYTHONPYCACHEPREFIX="/tmp/c80r-reg-${suite}-${SLURM_JOB_ID}/pycache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

case "${suite}" in
  focused)
    tests=(
      oaci/tests/test_c80_label_budget_frontier.py
      oaci/tests/test_c80e_preflight_blocker.py
      oaci/tests/test_c80r_additive_repair.py
    )
    ;;
  c65)
    tests=(
      oaci/tests/test_c6[5-9]_*.py
      oaci/tests/test_c7[0-9]_*.py
      oaci/tests/test_c78r_*.py
      oaci/tests/test_c78f_*.py
      oaci/tests/test_c78s_*.py
      oaci/tests/test_c79_*.py
      oaci/tests/test_c79p_*.py
      oaci/tests/test_c79e_*.py
      oaci/tests/test_c80_*.py
      oaci/tests/test_c80e_*.py
      oaci/tests/test_c80r_*.py
    )
    ;;
  c23)
    tests=(
      oaci/tests/test_c2[3-9]_*.py
      oaci/tests/test_c[3-6][0-9]_*.py
      oaci/tests/test_c7[0-9]_*.py
      oaci/tests/test_c78r_*.py
      oaci/tests/test_c78f_*.py
      oaci/tests/test_c78s_*.py
      oaci/tests/test_c79_*.py
      oaci/tests/test_c79p_*.py
      oaci/tests/test_c79e_*.py
      oaci/tests/test_c80_*.py
      oaci/tests/test_c80e_*.py
      oaci/tests/test_c80r_*.py
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
  --basetemp="/tmp/c80r-pytest-${suite}-${SLURM_JOB_ID}" \
  -o "cache_dir=/tmp/c80r-pytest-cache-${suite}-${SLURM_JOB_ID}" \
  "${deselect[@]}" "${tests[@]}"
