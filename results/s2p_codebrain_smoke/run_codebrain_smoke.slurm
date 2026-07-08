#!/bin/bash
#SBATCH --job-name=cb_smoke
#SBATCH --partition=V100-32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_codebrain_smoke/slurm-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_codebrain_smoke/slurm-%j.err
# NO --time / walltime flag (per project discipline).

set -euo pipefail
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
srun "$PY" /home/infres/yinwang/CMI_AAAI_s2p/results/s2p_codebrain_smoke/run_codebrain_smoke.py \
    --n-subjects 32 --hours-budget 50 --seed 0 \
    --batch-size 8 --max-steps 200 --target-windows 2048 --max-win-per-rec 48 --mask-ratio 0.5
