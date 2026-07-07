#!/bin/bash
#SBATCH --job-name=fsr-inv-audit
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python scripts/cb_cbm_8b_audit.py --model codebrain --dataset shu --suffix _bs32
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python scripts/cb_cbm_8b_f1_audit.py --model codebrain --dataset shu --f0_target_bacc 0.530 --suffix _bs32
echo "=== bs32 invariance audits done ==="
