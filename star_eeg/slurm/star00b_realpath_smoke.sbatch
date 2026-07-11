#!/usr/bin/env bash
#SBATCH --job-name=star00b
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_star
#SBATCH --output=results/star/star00b_preflight/logs/realpath-smoke-%j.out
#SBATCH --error=results/star/star00b_preflight/logs/realpath-smoke-%j.err

set -euo pipefail

REPO_ROOT="${STAR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/infres/yinwang/CMI_AAAI_star}}"
REPO_ROOT=$(readlink -f "$REPO_ROOT")
if [[ ! -d "$REPO_ROOT/star_eeg" || "$REPO_ROOT" == /var/spool/* ]]; then
  echo "ERROR: invalid STAR repo root: $REPO_ROOT" >&2
  exit 2
fi
cd "$REPO_ROOT"
mkdir -p results/star/star00b_preflight/logs

source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python

echo "STAR_00B real-path smoke host=$(hostname) head=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$PY" -m star_eeg.runners.run_star00b_realpath_smoke \
  --repo-root "$REPO_ROOT" \
  --out-dir results/star/star00b_preflight \
  --runtime-root /home/infres/yinwang/CMI_AAAI_star_runtime/results/star00b_realpath_smoke \
  --steps 10 \
  --device cuda:0
