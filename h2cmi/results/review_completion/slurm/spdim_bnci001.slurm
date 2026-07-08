#!/usr/bin/env bash
#SBATCH --job-name=spdim-bnci001
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%j.err

set -euo pipefail

cd /home/infres/yinwang/CMI_AAAI_qxu

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
EXT=/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39
export PYTHONPATH="$PWD:$EXT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export MPLCONFIGDIR=/tmp/h2cmi-mpl-${SLURM_JOB_ID:-manual}
mkdir -p "$MPLCONFIGDIR" h2cmi/results/review_completion/slurm/logs

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "pwd=$PWD"
echo "repo_head=$(git rev-parse HEAD)"
echo "repo_status_begin"
git status --short
echo "repo_status_end"
echo "runner_diff_sha256=$(git diff -- h2cmi/run_spdim_probe.py | sha256sum | awk '{print $1}')"
echo "external_sha=$(git -C "$EXT" rev-parse HEAD)"
nvidia-smi || true

"$PY" -m h2cmi.run_spdim_probe \
  --external-spdim-path "$EXT" \
  --dataset BNCI2014_001 \
  --subjects all \
  --seed 0 \
  --epochs 20 \
  --adapt-epochs 30 \
  --device cuda \
  --result-schema bnci001 \
  --overwrite \
  --allow-dirty \
  --out h2cmi/results/review_completion/spdim_bnci001_results.csv \
  --audit h2cmi/results/review_completion/spdim_bnci001_audit.md \
  --failure-trace h2cmi/results/review_completion/spdim_bnci001_failure_trace.txt
