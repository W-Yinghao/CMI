#!/bin/bash
# Project B Step-2G: freeze Step-2E/2F results into paper tables. READ-ONLY, no training.
# Reads the SHARED-home copies of the locked outputs (/tmp is node-local).
#SBATCH --job-name=pb-step2g
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/infres/yinwang/project_b_step2g_report/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_step2g_report/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
OUT=/home/infres/yinwang/project_b_step2g_report
mkdir -p "$OUT/logs"
cd "$WT"
"$PY" scripts/project_b_step2g_report.py \
    --step2e /home/infres/yinwang/project_b_step2e_router \
    --step2f /home/infres/yinwang/project_b_step2f_support \
    --out "$OUT"
echo "done $(date)"
