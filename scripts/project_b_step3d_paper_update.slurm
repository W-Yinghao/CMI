#!/bin/bash
# Project B Step-3D: merge Step-3C real benchmark into the paper package (read-only, fast).
#SBATCH --job-name=pb-step3d
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/infres/yinwang/project_b_step3d_paper_update/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_step3d_paper_update/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p /home/infres/yinwang/project_b_step3d_paper_update/logs
cd "$WT"
"$PY" scripts/project_b_step3d_paper_update.py \
    --step2g /tmp/project_b_step2g_report \
    --step3a /tmp/project_b_step3a_real_bridge \
    --step3c /tmp/project_b_step3c_real_benchmark \
    --out /tmp/project_b_step3d_paper_update
