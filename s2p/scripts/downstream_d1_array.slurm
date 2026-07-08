#!/bin/bash
#SBATCH --job-name=s2p_d1
#SBATCH --partition=A40,A100,H100,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --array=0-15
#SBATCH --time=4:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream/logs/d1-%A_%a.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream/logs/d1-%A_%a.err
# S2P_09 D1 fleet: 16 cells = 15 P1 checkpoints (5N x 3seed) + random-init floor. LAUNCH ONLY AFTER D0 PASSES + PM GO.
# Cells addressed by REAL name (not the array index); per-cell atomic JSON + --skip-if-done => a walltime kill or
# resubmit wastes nothing. 16 << 30 QOS cap so the array self-schedules across free A40/A100/H100/V100 GPUs.
# --time=4h << A100/H100 1d cap (all partitions eligible). Aggregate separately after coverage == 16/16.
set -euo pipefail
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
mkdir -p /home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream/logs
cd /home/infres/yinwang/CMI_AAAI_s2p

CELLS=(N128_s0 N128_s1 N128_s2 N256_s0 N256_s1 N256_s2 N512_s0 N512_s1 N512_s2 \
       N1024_s0 N1024_s1 N1024_s2 N2048_s0 N2048_s1 N2048_s2 random_init)
CELL=${CELLS[$SLURM_ARRAY_TASK_ID]}
echo "task=$SLURM_ARRAY_TASK_ID cell=$CELL host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
srun "$PY" s2p/scripts/run_downstream_frontier.py --cell "$CELL" --device cuda --skip-if-done
