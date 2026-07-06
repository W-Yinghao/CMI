#!/bin/bash
#SBATCH --job-name=fsr-rq4-4b1
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# FSR Phase 4B-1: seed0 FULL LOSO ERM refit for one dataset (all folds) + checkpoint/latent dump.
# Exact F0 config (no --target_indices => all folds). Feeds the branch-local L1-L6 verification chain.
#   sbatch scripts/run_rq4_4b1_seed0.slurm BNCI2014_001
#   sbatch scripts/run_rq4_4b1_seed0.slurm BNCI2015_001
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
DATASET=${1:-BNCI2014_001}
mkdir -p logs results/fsr_rq4_refit/ckpt results/fsr_rq4_refit/latents

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mpl-${SLURM_JOB_ID:-0}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "FSR RQ4 4B-1 seed0 full LOSO: FBCSPLGGGraph ERM on $DATASET (all folds, F0-exact + ckpt/dump)"

srun "$PY" -m cmi.run_loso \
    --dataset "$DATASET" --backbone FBCSPLGGGraph \
    --source_val_early_stop --configs erm:0 \
    --epochs 300 --bs 64 --warmup 40 --n_inner 2 --device cuda \
    --weight_decay 0.0001 --sampler classbal --prior empirical --prior_alpha 1.0 --beta 0.0 \
    --protocol loso --tmin 0.5 --tmax 3.5 --resample 128 --seed 0 --transduct off \
    --save_ckpt_dir results/fsr_rq4_refit/ckpt \
    --dump_latent_dir results/fsr_rq4_refit/latents \
    --out "results/fsr_rq4_refit/4b1_${DATASET}_seed0_erm.json"

echo "=== 4B-1 $DATASET done; checkpoint + latent counts ==="
ls results/fsr_rq4_refit/ckpt/ | grep -c "$DATASET" || true
ls results/fsr_rq4_refit/latents/ | grep "$DATASET" | grep -c manifest || true
