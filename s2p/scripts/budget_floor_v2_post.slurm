#!/bin/bash
#SBATCH --job-name=s2p9cpost
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2/logs/post-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_budget_floor_calibration_v2/logs/post-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_s2p
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
srun "$PY" s2p/scripts/budget_floor_v2_post.py --out-dir results/s2p_budget_floor_calibration_v2
