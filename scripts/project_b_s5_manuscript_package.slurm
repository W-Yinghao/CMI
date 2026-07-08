#!/bin/bash
# Project B-Next S5: manuscript consolidation package (read-only, fast).
#SBATCH --job-name=pb-s5
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/infres/yinwang/project_b_s5_manuscript/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_s5_manuscript/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB_next
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p /home/infres/yinwang/project_b_s5_manuscript/logs
cd "$WT"
"$PY" scripts/project_b_s5_manuscript_package.py \
    --step2g /tmp/project_b_step2g_report \
    --step3c /tmp/project_b_step3c_real_benchmark \
    --s0 /tmp/project_b_s0_calibration_records \
    --s2 /tmp/project_b_s2_acar_error \
    --s2b /tmp/project_b_s2b_acar_error_router \
    --s3 /tmp/project_b_s3_prior_only \
    --s1 /tmp/project_b_s1_phase_map \
    --s4 /tmp/project_b_s4_backend_feasibility \
    --out /tmp/project_b_s5_manuscript
