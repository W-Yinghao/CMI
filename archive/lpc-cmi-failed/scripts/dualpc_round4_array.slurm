#!/usr/bin/env bash
#SBATCH --job-name=dualpc-r4
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-5%6
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_round4 /tmp/matplotlib

export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib

TASK_FILE="${TASK_FILE:-scripts/dualpc_round4_tasks.tsv}"
line=$(awk -F '\t' -v id="${SLURM_ARRAY_TASK_ID}" '$1 == id {print $0}' "$TASK_FILE")
if [[ -z "$line" ]]; then
  echo "No task for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in $TASK_FILE" >&2
  exit 2
fi
IFS=$'\t' read -r task_id task_label task_cmd <<< "$line"
echo "task_id=$task_id label=$task_label"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "$task_cmd"
eval "$task_cmd"
