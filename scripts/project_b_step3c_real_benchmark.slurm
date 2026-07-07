#!/bin/bash
# Project B Step-3C: bounded real-EEG benchmark expansion (BNCI2014_004 primary).
# Loads real data (may download), trains H2-CMI per target, routes subject+session x 2 support modes.
#SBATCH --job-name=pb-step3c
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/infres/yinwang/project_b_step3c_real_benchmark/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_step3c_real_benchmark/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB
OUT=/home/infres/yinwang/project_b_step3c_real_benchmark
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p "$OUT/logs"
cd "$WT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

echo "start $(date) on $(hostname)"
"$PY" scripts/project_b_real_eeg_benchmark.py \
    --datasets BNCI2014_004 --max_subjects 6 --max_targets 4 --epochs 8 --batch_size 64 \
    --resample 128 --eval_units subject,session \
    --support_modes in_source_subject_q95,nested_source_subject_excess_q95 \
    --max_nested_folds 2 --device cpu --seed 0 --allow_missing_data --allow_dataset_failures \
    --max_runtime_seconds 3000 --out "$OUT"
echo "done $(date)"
