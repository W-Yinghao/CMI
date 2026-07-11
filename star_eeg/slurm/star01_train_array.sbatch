#!/usr/bin/env bash
# STAR_01 TEMPLATE ONLY. STAR_00B does not authorize submitting this file.
#SBATCH --job-name=star01
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_star
#SBATCH --output=results/star/star01_runtime/logs/train-%A_%a.out
#SBATCH --error=results/star/star01_runtime/logs/train-%A_%a.err

set -euo pipefail

REPO_ROOT="${STAR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/infres/yinwang/CMI_AAAI_star}}"
REPO_ROOT=$(readlink -f "$REPO_ROOT")
cd "$REPO_ROOT"
APPROVAL="$REPO_ROOT/results/star/star01_approval.json"
if [[ ! -f "$APPROVAL" ]]; then
  echo "ERROR: STAR_01 PM approval manifest is absent" >&2
  exit 2
fi

source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
TASKS="$REPO_ROOT/results/star/star00b_preflight/star01_training_tasks.csv"
ROW=$("$PY" -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1])))[int(sys.argv[2])]; print(r["variant"]+","+r["seed"])' "$TASKS" "$SLURM_ARRAY_TASK_ID")
IFS=, read -r VARIANT SEED <<< "$ROW"
OUT="/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01/${VARIANT}_s${SEED}"

srun "$PY" -m star_eeg.runners.run_star01_train \
  --variant "$VARIANT" \
  --model-seed "$SEED" \
  --approval-manifest "$APPROVAL" \
  --immutable-manifest "$REPO_ROOT/results/star/star00b_preflight/h200_immutable_manifest.json" \
  --immutable-go-nogo "$REPO_ROOT/results/star/star00b_preflight/h200_immutable_go_nogo.json" \
  --anchor-manifest "$REPO_ROOT/results/star/star00b_preflight/anchor_manifest.json" \
  --shuffled-manifest "$REPO_ROOT/results/star/star00b_preflight/shuffled_label_manifest.json" \
  --runtime-output-dir "$OUT" \
  --repo-root "$REPO_ROOT" \
  --device cuda:0
