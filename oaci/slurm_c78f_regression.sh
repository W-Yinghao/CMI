#!/usr/bin/env bash
#SBATCH --job-name=c78f-regression
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
set -euo pipefail
suite="${1:?focused|c65|c23|full required}"
cd /home/infres/yinwang/CMI_AAAI_oaci
export PYTHONPYCACHEPREFIX="/tmp/c78f-reg-${suite}-${SLURM_JOB_ID}/pycache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
case "${suite}" in
  focused)
    tests=(oaci/tests/test_c78f_full_seed3_field.py)
    ;;
  c65)
    tests=(oaci/tests/test_c6[5-9]_*.py oaci/tests/test_c7[0-9]_*.py oaci/tests/test_c78r_*.py oaci/tests/test_c78f_*.py)
    ;;
  c23)
    tests=(oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c[3-6][0-9]_*.py oaci/tests/test_c7[0-9]_*.py oaci/tests/test_c78r_*.py oaci/tests/test_c78f_*.py)
    ;;
  full)
    tests=(oaci/tests)
    ;;
  *)
    echo "unknown suite: ${suite}" >&2
    exit 2
    ;;
esac
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m pytest -q \
  --basetemp="/tmp/c78f-pytest-${suite}-${SLURM_JOB_ID}" \
  -o "cache_dir=/tmp/c78f-pytest-cache-${suite}-${SLURM_JOB_ID}" "${tests[@]}"
