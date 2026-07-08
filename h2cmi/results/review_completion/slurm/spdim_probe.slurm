#!/usr/bin/env bash
#SBATCH --job-name=spdim-probe
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%j.err

set -euo pipefail

cd /home/infres/yinwang/CMI_AAAI_qxu
mkdir -p h2cmi/results/review_completion/slurm/logs

EXT=/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$EXT:${PYTHONPATH:-}"
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${SLURM_JOB_ID}

echo "[spdim-probe] host=$(hostname) partition=${SLURM_JOB_PARTITION:-NA} job=${SLURM_JOB_ID:-NA}"
echo "[spdim-probe] repo=$(git rev-parse HEAD)"
echo "[spdim-probe] external=$(git -C "$EXT" rev-parse HEAD)"
nvidia-smi || true

"$PY" -m h2cmi.run_spdim_probe \
  --external-spdim-path "$EXT" \
  --dataset BNCI2014_001 \
  --subjects 1,9 \
  --seed 0 \
  --epochs 20 \
  --adapt-epochs 30 \
  --device cuda \
  --out h2cmi/results/review_completion/spdim_probe_results.csv \
  --audit h2cmi/results/review_completion/spdim_probe_audit.md \
  --failure-trace h2cmi/results/review_completion/spdim_probe_failure_trace.txt
