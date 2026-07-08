#!/usr/bin/env bash
#SBATCH --job-name=w1offdiag-probe
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%j.err

set -euo pipefail

cd /home/infres/yinwang/CMI_AAAI_qxu
mkdir -p h2cmi/results/review_completion/slurm/logs results/h2cmi/review_completion_offdiag

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${SLURM_JOB_ID}

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
PAIR="BNCI2014_001:0>1"
OUT="results/h2cmi/review_completion_offdiag/probe_bnci001_s1_seed0.jsonl"

echo "[probe] host=$(hostname) partition=${SLURM_JOB_PARTITION:-NA} job=${SLURM_JOB_ID:-NA}"
nvidia-smi || true
"$PY" -m h2cmi.run_v2p_geometry_offdiag \
  --pairs "$PAIR" \
  --subjects 1 \
  --seeds 0 \
  --out "$OUT" \
  --device cuda \
  --allow-dirty
