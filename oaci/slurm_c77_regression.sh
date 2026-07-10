#!/usr/bin/env bash
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs/%x_%j.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs/%x_%j.err
set -euo pipefail
suite="${1:?suite must be focused, c65_c77, c23_c77, or full}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/tmp/c77-pycache-${SLURM_JOB_ID}
mkdir -p /projects/EEG-foundation-model/yinghao/oaci-c77-multiregime/logs
cd /home/infres/yinwang/CMI_AAAI_oaci
case "${suite}" in
  focused) tests=(oaci/tests/test_c77_independent_multiregime_replication_protocol.py) ;;
  c65_c77) tests=(oaci/tests/test_c6[5-9]_*.py oaci/tests/test_c7[0-7]_*.py) ;;
  c23_c77) tests=(oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c[3-6][0-9]_*.py oaci/tests/test_c7[0-7]_*.py) ;;
  full) tests=(oaci/tests) ;;
  *) printf 'unknown suite: %s\n' "${suite}" >&2; exit 2 ;;
esac
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m pytest -q \
  --basetemp="/tmp/c77-pytest-${SLURM_JOB_ID}" \
  -o "cache_dir=/tmp/c77-pytest-cache-${SLURM_JOB_ID}" "${tests[@]}"
