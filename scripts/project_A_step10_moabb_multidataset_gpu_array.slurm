#!/bin/bash
# Project A Step 10 — multi-dataset audited grids on a GPU array. 18 tasks = 2 datasets x 9 shards;
# task/9 selects the dataset, task%9 the shard. Each dataset uses --subjects all --target-subjects
# all (resolved post-load) x seeds 0-2 x 50 epochs. Idempotent with --resume. SLURM schedules each
# task onto whichever GPU is free among A100/H100/V100/A40. Run the *_validate.slurm digest+combine
# job AFTER all tasks finish (submit with --dependency=afterok:<arrayid> so no task races the
# summary). Not a SOTA run — an audited multi-dataset claim-boundary + normalized digest. NO --time.
#SBATCH --job-name=projA_step10_moabb
#SBATCH --partition=A100,H100,V100,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-17
#SBATCH --output=logs/project_A_step10_moabb_%A_%a.out
#SBATCH --error=logs/project_A_step10_moabb_%A_%a.err

set -euo pipefail
mkdir -p logs

DATASETS=(BNCI2014_004 BNCI2015_001)
DSET_INDEX=$((SLURM_ARRAY_TASK_ID / 9))
SHARD=$((SLURM_ARRAY_TASK_ID % 9))
DATASET=${DATASETS[$DSET_INDEX]}
ROOT=notes/project_A_observability/results/step10_${DATASET}

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

echo "step10 gpu task $SLURM_ARRAY_TASK_ID ($DATASET shard $SHARD) done -> $ROOT"
