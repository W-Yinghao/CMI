#!/bin/bash
#SBATCH --job-name=c86l-build
#SBATCH --partition=cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_oaci/c86l_build_logs/c86l_%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_oaci/c86l_build_logs/c86l_%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_oaci
PY=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python
exec "$PY" -m oaci.active_testing.c86l_build \
  --output-root /projects/EEG-foundation-model/yinghao/oaci-c86l-development-field-v1 \
  --authorization '授权 C86L'
