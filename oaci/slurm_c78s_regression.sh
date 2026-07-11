#!/usr/bin/env bash
#SBATCH --job-name=c78s-regression
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI/c78f_oaci/oaci/logs/c78s-regression-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI/c78f_oaci/oaci/logs/c78s-regression-%j.err
set -euo pipefail
suite="${1:?focused|c65|c23|full required}"
cd /home/infres/yinwang/CMI_AAAI/c78f_oaci
export PYTHONPYCACHEPREFIX="/tmp/c78s-reg-${suite}-${SLURM_JOB_ID}/pycache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
case "${suite}" in
  focused)
    tests=(oaci/tests/test_c78s_seed3_scientific_analysis.py)
    ;;
  c65)
    tests=(oaci/tests/test_c6[5-9]_*.py oaci/tests/test_c7[0-9]_*.py oaci/tests/test_c78r_*.py oaci/tests/test_c78f_*.py oaci/tests/test_c78s_*.py)
    ;;
  c23)
    tests=(oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c[3-6][0-9]_*.py oaci/tests/test_c7[0-9]_*.py oaci/tests/test_c78r_*.py oaci/tests/test_c78f_*.py oaci/tests/test_c78s_*.py)
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
  --basetemp="/tmp/c78s-pytest-${suite}-${SLURM_JOB_ID}" \
  -o "cache_dir=/tmp/c78s-pytest-cache-${suite}-${SLURM_JOB_ID}" "${tests[@]}"
