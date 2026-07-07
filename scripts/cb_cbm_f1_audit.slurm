#!/bin/bash
#SBATCH --job-name=fsr-f1-audit
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
rm -f results/fsr_codebrain_cbramod_8b/f1_task_gate.csv results/fsr_codebrain_cbramod_8b/f1_l1_subject_probe.csv results/fsr_codebrain_cbramod_8b/f1_l4_task_alignment.csv results/fsr_codebrain_cbramod_8b/f1_l5_replay.csv results/fsr_codebrain_cbramod_8b/f1_l6_target_consequence.csv
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python scripts/cb_cbm_8b_f1_audit.py --model codebrain --dataset shu --f0_target_bacc 0.530
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python scripts/cb_cbm_8b_f1_audit.py --model cbramod --dataset shu --f0_target_bacc 0.548
echo "=== F1 audits done ==="
