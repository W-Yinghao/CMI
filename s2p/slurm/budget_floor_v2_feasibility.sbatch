#!/bin/bash
#SBATCH --job-name=s2p9c0f
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p_launch/results/s2p_budget_floor_calibration_v2/logs/feas-only-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p_launch/results/s2p_budget_floor_calibration_v2/logs/feas-only-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_s2p_launch
mkdir -p results/s2p_budget_floor_calibration_v2/logs
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
OUT=/home/infres/yinwang/CMI_AAAI_s2p_launch/results/s2p_budget_floor_calibration_v2

echo "S2P 9C v2 feasibility-only job"
echo "host=$(hostname)"
echo "git=$(git rev-parse HEAD)"
git status --short

srun "$PY" s2p/scripts/budget_floor_v2_feasibility.py --out-dir "$OUT" --min-exposure-hours 0.25
