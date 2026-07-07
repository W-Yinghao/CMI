#!/bin/bash
#SBATCH --job-name=fsr-cb-preflight
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# FSR Phase 8A CodeBrain preflight load-test on GPU (SSSM hardcodes .cuda() for the SWA mask).
#   sbatch scripts/codebrain_preflight.slurm
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs results/fsr_codebrain_preflight
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun "$PY" scripts/codebrain_preflight_loadtest.py --device cuda
