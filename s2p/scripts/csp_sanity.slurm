#!/bin/bash
#SBATCH --job-name=csp_san
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream/logs/csp-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_s2p/results/s2p_p1_downstream/logs/csp-%j.err
set -euo pipefail
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
cd /home/infres/yinwang/CMI_AAAI_s2p
srun /home/infres/yinwang/anaconda3/envs/eeg2025/bin/python s2p/scripts/shumi_csp_sanity.py
