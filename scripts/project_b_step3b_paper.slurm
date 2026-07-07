#!/bin/bash
# Project B Step-3B: assemble the paper package from frozen Step-2G/3A outputs (read-only, fast).
#SBATCH --job-name=pb-step3b
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/infres/yinwang/project_b_step3b_paper/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_step3b_paper/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p /home/infres/yinwang/project_b_step3b_paper/logs
cd "$WT"
"$PY" scripts/project_b_step3b_paper_package.py \
    --step2g /tmp/project_b_step2g_report \
    --step3a /tmp/project_b_step3a_real_bridge \
    --out /tmp/project_b_step3b_paper
