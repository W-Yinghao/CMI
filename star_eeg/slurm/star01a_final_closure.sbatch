#!/usr/bin/env bash
# STAR_01A afterok closure. Submit only with --dependency=afterok:<array_job_id>.
#SBATCH --job-name=star01a-close
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_star
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_star_runtime/logs/star01a-closure-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_star_runtime/logs/star01a-closure-%j.err

set -euo pipefail

REPO_ROOT="${STAR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/infres/yinwang/CMI_AAAI_star}}"
REPO_ROOT=$(readlink -f "$REPO_ROOT")
cd "$REPO_ROOT"
: "${STAR_APPROVAL:?STAR_APPROVAL must name the SHA-bound read-only approval manifest}"
: "${STAR_ATTEMPT_ID:?STAR_ATTEMPT_ID must be attempt_NN}"
: "${STAR_ARRAY_JOB_ID:?STAR_ARRAY_JOB_ID must identify the six-cell parent array}"

source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
RUNTIME_ROOT=/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01
CLOSURE_DIR="/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01_closure/${STAR_ATTEMPT_ID}"

echo "STAR_01A closure host=$(hostname) head=$(git rev-parse HEAD) parent_array=${STAR_ARRAY_JOB_ID}"
srun "$PY" -m star_eeg.artifacts.close_star01a_finals \
  --repo-root "$REPO_ROOT" \
  --runtime-root "$RUNTIME_ROOT" \
  --attempt-id "$STAR_ATTEMPT_ID" \
  --approval-manifest "$STAR_APPROVAL" \
  --closure-output-dir "$CLOSURE_DIR"
