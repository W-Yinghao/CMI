#!/bin/bash
# Project A Step 13 — rerun the two ok datasets with rich R1 diagnostics + per-trial oracle payload,
# on a GPU array. 18 tasks = 2 datasets x 9 shards; task/9 selects the dataset, task%9 the shard.
# Same audited runner (now instrumented) with --subjects all --target-subjects all x seeds 0-2 x 50
# epochs. Idempotent with --resume. NO new datasets. Run the *_science_cpu.slurm pipeline after all
# tasks finish (afterok). NO --time (cluster convention).
#SBATCH --job-name=projA_step13_diag
#SBATCH --partition=A100,H100,V100,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-17
#SBATCH --output=logs/project_A_step13_diag_%A_%a.out
#SBATCH --error=logs/project_A_step13_diag_%A_%a.err

set -euo pipefail
mkdir -p logs

DATASETS=(BNCI2014_001 BNCI2014_004)
DSET_INDEX=$((SLURM_ARRAY_TASK_ID / 9))
SHARD=$((SLURM_ARRAY_TASK_ID % 9))
DATASET=${DATASETS[$DSET_INDEX]}
ROOT=notes/project_A_observability/results/step13_${DATASET}_diagnostics

conda run -n icml python -m h2cmi.run_real_audited_grid \
  --dataset "$DATASET" \
  --subjects all \
  --target-subjects all \
  --seeds 0 1 2 \
  --epochs 50 \
  --fast \
  --align-factor subject \
  --device cuda \
  --root-outdir "$ROOT" \
  --num-shards 9 \
  --shard-index "$SHARD" \
  --resume

echo "step13 diagnostics task $SLURM_ARRAY_TASK_ID ($DATASET shard $SHARD) done -> $ROOT"
