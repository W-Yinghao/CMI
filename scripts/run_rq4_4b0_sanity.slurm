#!/bin/bash
#SBATCH --job-name=fsr-rq4-4b0
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# FSR Phase 4B-0 sanity: ONE dataset x ONE target fold x ONE seed, exact F0 config for
# 2a target_index 0 seed 0 + the default-off checkpoint/latent-dump instrument. Verifies:
# checkpoint saves, latent dumps save, recompose-identity < 1e-5, branch ablation runs,
# target labels not used in training/selection. No walltime (-t) per compute policy.
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs results/fsr_rq4_refit/ckpt results/fsr_rq4_refit/latents

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mpl-${SLURM_JOB_ID:-0}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "FSR RQ4 4B-0 sanity: FBCSPLGGGraph ERM, 2a target_index 0 seed 0 (F0-exact + ckpt/dump)"

srun "$PY" -m cmi.run_loso \
    --dataset BNCI2014_001 --backbone FBCSPLGGGraph --target_indices 0 \
    --source_val_early_stop --configs erm:0 \
    --epochs 300 --bs 64 --warmup 40 --n_inner 2 --device cuda \
    --weight_decay 0.0001 --sampler classbal --prior empirical --prior_alpha 1.0 --beta 0.0 \
    --protocol loso --tmin 0.5 --tmax 3.5 --resample 128 --seed 0 --transduct off \
    --save_ckpt_dir results/fsr_rq4_refit/ckpt \
    --dump_latent_dir results/fsr_rq4_refit/latents \
    --out results/fsr_rq4_refit/4b0_2a_t0_seed0_erm.json

echo "=== 4B-0 done; artifacts ==="
ls -la results/fsr_rq4_refit/ckpt results/fsr_rq4_refit/latents
