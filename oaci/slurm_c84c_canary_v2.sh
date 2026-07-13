#!/usr/bin/env bash
#SBATCH --job-name=c84c-canary-v3
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v3/slurm-%j.out
#SBATCH --error=/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v3/slurm-%j.err
set -euo pipefail

repo=/home/infres/yinwang/CMI_AAAI_oaci
env_prefix=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$repo"
"$env_prefix/bin/python" -m oaci.multidataset.c84c_real_canary_v2 run-real \
  --authorization-record oaci/reports/C84C_PI_AUTHORIZATION_RECORD_V2.json \
  --output-root /projects/EEG-foundation-model/yinghao/oaci-c84-canary-v3
