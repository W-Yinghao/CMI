#!/usr/bin/env bash
#SBATCH --job-name=spdim-w1-seed0
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=h2cmi/results/review_completion/slurm/logs/%x-%j.out
#SBATCH --error=h2cmi/results/review_completion/slurm/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

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
echo "repo_status_porcelain_begin"
git status --porcelain=v1 --untracked-files=all
echo "repo_status_porcelain_end"
echo "external_sha=$(git -C "$EXT" rev-parse HEAD)"
nvidia-smi || true

"$PY" -m h2cmi.run_spdim_w1_seed0 \
  --mode run \
  --external-spdim-path "$EXT" \
  --datasets BNCI2014_001,Cho2017,Lee2019_MI \
  --seed 0 \
  --epochs 20 \
  --adapt-epochs 30 \
  --device cuda \
  --out h2cmi/results/review_completion/spdim_w1_seed0_results.csv \
  --audit h2cmi/results/review_completion/spdim_w1_seed0_audit.md \
  --summary h2cmi/results/review_completion/spdim_w1_seed0_summary.json \
  --digest h2cmi/results/review_completion/spdim_w1_seed0_result_digest.md \
  --failure-trace h2cmi/results/review_completion/spdim_w1_seed0_failure_trace.txt
