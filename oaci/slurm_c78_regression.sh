#!/usr/bin/env bash
#SBATCH --job-name=c78-reg
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_oaci
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/c78-pycache-${SLURM_JOB_ID}"
suite="${1:?suite required}"
case "$suite" in
  focused)
    tests=(oaci/tests/test_c78_seed3_instrumented_pilot.py oaci/tests/test_c78_authorized_pilot.py oaci/tests/test_c78_authorized_collect.py)
    ;;
  c65)
    tests=(oaci/tests/test_c6[5-9]_*.py oaci/tests/test_c7[0-8]_*.py)
    ;;
  c23)
    tests=(oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c[3-6][0-9]_*.py oaci/tests/test_c7[0-8]_*.py)
    ;;
  full)
    tests=(oaci/tests)
    ;;
  *)
    echo "unknown suite: $suite" >&2
    exit 2
    ;;
esac
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m pytest -q \
  --basetemp="/tmp/c78-pytest-${SLURM_JOB_ID}" \
  -o "cache_dir=/tmp/c78-pytest-cache-${SLURM_JOB_ID}" "${tests[@]}"
