#!/bin/bash
# Project A — expanded audited BNCI2014_001 grid on CPU, MULTI-CORE via two levels of parallelism:
#   (1) PROCESS-level: a 9-task SLURM ARRAY, each task runs a shard (3 of the 27 cells) in its own
#       process on its own cores;  (2) intra-op: each cell's training uses cpus-per-task threads
#       (OMP/MKL + torch --threads). Idempotent with --resume. Run the *_validate.slurm job after
#       ALL array tasks finish (a separate job -> no race on the summary). NO --time (convention).
#SBATCH --job-name=projA_bnci_exp_cpu
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-8
#SBATCH --output=logs/project_A_bnci_exp_cpu_%A_%a.out
#SBATCH --error=logs/project_A_bnci_exp_cpu_%A_%a.err

set -euo pipefail
mkdir -p logs
ROOT=notes/project_A_observability/results/step9_bnci2014_001_expanded

NT=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS="$NT"
export MKL_NUM_THREADS="$NT"

conda run -n icml python -m h2cmi.run_real_audited_grid \
  --dataset BNCI2014_001 \
  --subjects 1 2 3 4 5 6 7 8 9 \
  --target-subjects 1 2 3 4 5 6 7 8 9 \
  --seeds 0 1 2 \
  --epochs 50 \
  --fast \
  --align-factor subject \
  --device cpu \
  --threads "$NT" \
  --root-outdir "$ROOT" \
  --num-shards 9 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --resume

echo "cpu array task $SLURM_ARRAY_TASK_ID done -> $ROOT (submit *_validate.slurm after all tasks finish)"
