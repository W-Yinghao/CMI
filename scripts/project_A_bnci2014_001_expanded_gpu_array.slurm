#!/bin/bash
# Project A — expanded audited BNCI2014_001 grid (subjects 1-9 x target 1-9 x seed 0-2 = 27 cells)
# sharded across a 9-task GPU ARRAY (3 cells/task) for wall-clock parallelism. SLURM schedules each
# task onto whichever GPU is free among A100/H100/V100/A40. Idempotent with --resume. Run the
# *_validate.slurm digest job AFTER all tasks finish (submit it with --dependency=afterok:<arrayid>
# so no task races to write the summary). Not a SOTA run — an audited claim-boundary validation.
# NO --time (cluster convention).
#SBATCH --job-name=projA_bnci_exp_gpu
#SBATCH --partition=A100,H100,V100,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-8
#SBATCH --output=logs/project_A_bnci_exp_gpu_%A_%a.out
#SBATCH --error=logs/project_A_bnci_exp_gpu_%A_%a.err

set -euo pipefail
mkdir -p logs
ROOT=notes/project_A_observability/results/step9_bnci2014_001_expanded

conda run -n icml python -m h2cmi.run_real_audited_grid \
  --dataset BNCI2014_001 \
  --subjects 1 2 3 4 5 6 7 8 9 \
  --target-subjects 1 2 3 4 5 6 7 8 9 \
  --seeds 0 1 2 \
  --epochs 50 \
  --fast \
  --align-factor subject \
  --device cuda \
  --root-outdir "$ROOT" \
  --num-shards 9 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --resume

echo "gpu array task $SLURM_ARRAY_TASK_ID done -> $ROOT (validate job runs after all tasks via afterok)"
