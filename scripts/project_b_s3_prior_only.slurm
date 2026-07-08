#!/bin/bash
# Project B-Next S3A: PRIOR_ONLY action study (synthetic worlds + real BNCI2014_004, CPU).
#SBATCH --job-name=pb-s3a
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/infres/yinwang/project_b_s3_prior_only/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_s3_prior_only/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB_next
OUT=/home/infres/yinwang/project_b_s3_prior_only
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p "$OUT/logs"
cd "$WT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

echo "start $(date) on $(hostname)"
"$PY" scripts/project_b_s3_prior_only_action_study.py \
    --include_synthetic \
    --include_real_bnci2014_004 \
    --synthetic_epochs 30 \
    --real_epochs 8 \
    --real_max_subjects 6 \
    --real_max_targets 4 \
    --tau 10.0 \
    --device cpu \
    --seed 0 \
    --allow_missing_data \
    --out "$OUT"
echo "done $(date)"
