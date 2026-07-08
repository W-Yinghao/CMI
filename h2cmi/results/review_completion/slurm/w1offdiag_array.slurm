#!/usr/bin/env bash
#SBATCH --job-name=w1offdiag-array
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%A_%a.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%A_%a.err

set -euo pipefail

cd /home/infres/yinwang/CMI_AAAI_qxu
mkdir -p h2cmi/results/review_completion/slurm/logs results/h2cmi/review_completion_offdiag

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${SLURM_JOB_ID}

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
PAIRS=(
  "BNCI2014_001:0>1"
  "Lee2019_MI:0>1"
  "BNCI2014_004:0>1"
  "BNCI2014_004:2>3"
  "BNCI2014_004:2>4"
)
NAMES=(bnci001 lee2019 bnci004_01 bnci004_23 bnci004_24)

PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
OUT="results/h2cmi/review_completion_offdiag/w1offdiag_${NAME}.jsonl"

echo "[array] host=$(hostname) partition=${SLURM_JOB_PARTITION:-NA} job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-NA} pair=$PAIR"
nvidia-smi || true
"$PY" -m h2cmi.run_v2p_geometry_offdiag \
  --pairs "$PAIR" \
  --seeds 0,1,2 \
  --out "$OUT" \
  --device cuda \
  --allow-dirty
