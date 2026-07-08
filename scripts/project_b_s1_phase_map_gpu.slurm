#!/bin/bash
# Project B-Next S1A: real EEG TTA phase map + source-predictability test (GPU).
#SBATCH --job-name=pb-s1a-gpu
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/infres/yinwang/project_b_s1_phase_map/logs/gpu-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_s1_phase_map/logs/gpu-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB_next
OUT=/home/infres/yinwang/project_b_s1_phase_map
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p "$OUT/logs"
cd "$WT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

echo "start $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
"$PY" scripts/project_b_s1_phase_map.py \
    --datasets BNCI2014_004,BNCI2014_001,Lee2019_MI \
    --core_datasets BNCI2014_004,BNCI2014_001 \
    --max_subjects 6 --max_targets 4 --epochs 8 --batch_size 64 --resample 128 \
    --eval_units subject,session \
    --support_modes in_source_subject_q95,nested_source_subject_excess_q95 \
    --max_nested_folds 2 --ridge_alpha 1.0 --gain_margin 0.02 \
    --device cuda --seed 0 --allow_missing_data --allow_dataset_failures \
    --out "$OUT"
echo "done $(date)"
