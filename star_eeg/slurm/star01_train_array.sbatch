#!/usr/bin/env bash
# STAR_01A six-cell array. Submission requires the SHA-bound 00C approval lock.
#SBATCH --job-name=star01
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_star
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_star_runtime/logs/star01a-train-%A_%a.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_star_runtime/logs/star01a-train-%A_%a.err

set -euo pipefail

REPO_ROOT="${STAR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/infres/yinwang/CMI_AAAI_star}}"
REPO_ROOT=$(readlink -f "$REPO_ROOT")
cd "$REPO_ROOT"
: "${STAR_APPROVAL:?STAR_APPROVAL must name the SHA-bound read-only approval manifest}"
: "${STAR_ATTEMPT_ID:?STAR_ATTEMPT_ID must be attempt_NN}"
APPROVAL=$(readlink -f "$STAR_APPROVAL")

source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
TASKS="$REPO_ROOT/results/star/star00b_preflight/star01_training_tasks.csv"
ROW=$("$PY" -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1])))[int(sys.argv[2])]; print(r["variant"]+","+r["seed"])' "$TASKS" "$SLURM_ARRAY_TASK_ID")
IFS=, read -r VARIANT SEED <<< "$ROW"
OUT="/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01/${VARIANT}_s${SEED}"
OUT="$OUT/$STAR_ATTEMPT_ID"

echo "STAR_01A cell=${VARIANT}_s${SEED} attempt=${STAR_ATTEMPT_ID} head=$(git rev-parse HEAD)"
srun "$PY" -m star_eeg.runners.run_star01_train \
  --variant "$VARIANT" \
  --model-seed "$SEED" \
  --approval-manifest "$APPROVAL" \
  --immutable-manifest "$REPO_ROOT/results/star/star00b_preflight/h200_immutable_manifest.json" \
  --immutable-go-nogo "$REPO_ROOT/results/star/star00b_preflight/h200_immutable_go_nogo.json" \
  --anchor-manifest "$REPO_ROOT/results/star/star00b_preflight/anchor_manifest.json" \
  --shuffled-manifest "$REPO_ROOT/results/star/star00b_preflight/shuffled_label_manifest.json" \
  --runtime-output-dir "$OUT" \
  --attempt-id "$STAR_ATTEMPT_ID" \
  --repo-root "$REPO_ROOT" \
  --device cuda:0
