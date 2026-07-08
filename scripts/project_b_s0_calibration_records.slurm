#!/bin/bash
# Project B-Next S0: calibration record builder (synthetic worlds + real BNCI2014_004, CPU).
#SBATCH --job-name=pb-s0
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/infres/yinwang/project_b_s0_calibration_records/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_s0_calibration_records/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB_next
OUT=/home/infres/yinwang/project_b_s0_calibration_records
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p "$OUT/logs"
cd "$WT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

echo "start $(date) on $(hostname)"
"$PY" scripts/project_b_calibration_records.py \
    --include_synthetic \
    --include_real_bnci2014_004 \
    --synthetic_epochs 30 \
    --real_epochs 8 \
    --real_max_subjects 6 \
    --real_max_targets 4 \
    --real_max_nested_folds 2 \
    --device cpu \
    --seed 0 \
    --allow_missing_data \
    --out "$OUT"
echo "done $(date)"
