#!/bin/bash
# Project B-Next S4A: backend feasibility (h2cmi_common vs cbramod_common) under common downstream (GPU).
#SBATCH --job-name=pb-s4a-gpu
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/project_b_s4_backend_feasibility/logs/gpu-%j.out
#SBATCH --error=/home/infres/yinwang/project_b_s4_backend_feasibility/logs/gpu-%j.err

WT=/home/infres/yinwang/CMI_AAAI_projectB_next
OUT=/home/infres/yinwang/project_b_s4_backend_feasibility
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p "$OUT/logs"
cd "$WT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

echo "start $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
"$PY" scripts/project_b_s4_backend_feasibility.py \
    --datasets BNCI2014_004,BNCI2014_001,Lee2019_MI \
    --core_datasets BNCI2014_004,BNCI2014_001 \
    --backends h2cmi_common,cbramod_common \
    --max_subjects 6 --max_targets 4 --epochs 8 --batch_size 64 --resample 200 \
    --eval_units subject,session --max_nested_folds 2 --pca_dim 128 \
    --tta_steps 100 --tta_lr 1e-2 --lambda_b 1e-2 --lambda_s 1e-2 --tau 10.0 \
    --error_budget 0.45 --gain_margin 0.02 \
    --device cuda --seed 0 --allow_missing_data --allow_missing_foundation --allow_dataset_failures \
    --out "$OUT"
echo "done $(date)"
