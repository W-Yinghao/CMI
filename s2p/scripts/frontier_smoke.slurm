#!/bin/bash
#SBATCH --job-name=fr_smoke
#SBATCH --partition=A40,A100,H100,V100-32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_cbramod/smoke/slurm-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_cbramod/smoke/slurm-%j.err
# NO --time (project discipline). Multi-partition: SLURM schedules on whichever GPU is free.
set -euo pipefail
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
echo "host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
srun "$PY" /home/infres/yinwang/CMI_AAAI_s2p/s2p/scripts/run_frontier_cbramod.py \
    --n-subjects 256 --subset-seed 0 --init-seed 0 --epochs 2 \
    --batch-size 32 --max-train-windows 512 --max-val-windows 128 --max-steps 15 --num-workers 4 --smoke \
    --out-dir /home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_cbramod/smoke
