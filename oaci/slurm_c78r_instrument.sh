#!/usr/bin/env bash
#SBATCH --job-name=c78r-instrument
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -euo pipefail
token="${1:?exact CLI authorization token required}"
shard="${2:?shard index required}"
count="${3:?shard count required}"
cd /home/infres/yinwang/CMI_AAAI_oaci
scratch="/tmp/c78r-instrument-${SLURM_JOB_ID}"
mkdir -p "${scratch}"
export TMPDIR="${scratch}"
export PYTHONPYCACHEPREFIX="${scratch}/pycache"
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 NUMEXPR_NUM_THREADS=48
/home/infres/yinwang/anaconda3/envs/icml/bin/python \
  -m oaci.conditioned_ceiling_coverage.c78r_instrument instrument \
  --authorization-token "${token}" --shard-index "${shard}" --num-shards "${count}"
