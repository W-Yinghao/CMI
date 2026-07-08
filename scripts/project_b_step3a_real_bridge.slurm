#!/bin/bash
# Project B Step-3A: real-EEG LOSO router bridge smoke (MOABB BNCI2014_004).
# Loads real data (may download), trains H2-CMI per target, routes; degrades cleanly if data missing.
# Output to SHARED home storage (/tmp is node-local).
#SBATCH --job-name=pb-step3a
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/infres/yinwang/project_b_step3a_real_bridge/logs/driver-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_step3a_real_bridge/logs/driver-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB
OUT=/home/infres/yinwang/project_b_step3a_real_bridge
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p "$OUT/logs"
cd "$WT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

echo "start $(date) on $(hostname)"
"$PY" scripts/project_b_real_eeg_bridge.py \
    --dataset BNCI2014_004 --max_subjects 4 --max_targets 2 --epochs 8 --batch_size 64 \
    --resample 128 --eval_unit subject --support_mode both --max_nested_folds 2 \
    --device cpu --seed 0 --allow_missing_data --out "$OUT"
echo "done $(date)"
