#!/bin/bash
#SBATCH --job-name=s2p_d0
#SBATCH --partition=A40,A100,H100,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream/logs/d0-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream/logs/d0-%j.err
# S2P_09 D0 one-checkpoint probe gate (docs/S2P_09_DOWNSTREAM_AUDIT_FROZEN.md sec 5.1).
# Runs N512_s0 + random-init through the full SHU-MI source-only probe and evaluates the 8-item QC gate.
# --time=4h << the A100/H100 1d cap so all four partitions are eligible (see frontier_array.slurm rationale).
# Small: 2 cells x 25 subjects x ~500 trials forward on a 4.9M CBraMod -> a few minutes.
set -euo pipefail
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
mkdir -p /home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream/logs
cd /home/infres/yinwang/CMI_AAAI_s2p
echo "host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
srun "$PY" s2p/scripts/run_downstream_frontier.py --probe-gate --device cuda
