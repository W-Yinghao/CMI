#!/bin/bash
# Project B-Next S2B: optional ACAR-error router integration eval (pure-numpy + router, fast).
#SBATCH --job-name=pb-s2b
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/infres/yinwang/project_b_s2b_acar_error_router/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_s2b_acar_error_router/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB_next
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p /home/infres/yinwang/project_b_s2b_acar_error_router/logs
cd "$WT"
"$PY" scripts/project_b_s2b_acar_error_router_eval.py \
    --records /tmp/project_b_s0_calibration_records \
    --alpha 0.10 --error_budget 0.45 --ridge_alpha 1.0 \
    --out /tmp/project_b_s2b_acar_error_router
