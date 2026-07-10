#!/usr/bin/env bash
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c76-representation-association/logs/%x_%j.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c76-representation-association/logs/%x_%j.err

set -euo pipefail

suite="${1:?suite must be c65_c76, c23_c76, or full}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/tmp/c76-pycache-${SLURM_JOB_ID}

mkdir -p /projects/EEG-foundation-model/yinghao/oaci-c76-representation-association/logs
cd /home/infres/yinwang/CMI_AAAI_oaci

case "${suite}" in
  c65_c76)
    tests=(oaci/tests/test_c6[5-9]_*.py oaci/tests/test_c7[0-6]_*.py)
    ;;
  c23_c76)
    tests=(oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c[3-6][0-9]_*.py oaci/tests/test_c7[0-6]_*.py)
    ;;
  full)
    tests=(oaci/tests)
    ;;
  *)
    printf 'unknown suite: %s\n' "${suite}" >&2
    exit 2
    ;;
esac

/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m pytest -q \
  --basetemp="/tmp/c76-pytest-${SLURM_JOB_ID}" \
  -o "cache_dir=/tmp/c76-pytest-cache-${SLURM_JOB_ID}" \
  "${tests[@]}"
