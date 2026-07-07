#!/usr/bin/env bash
#SBATCH --job-name=dualpc-p3b-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_phase3b

INITIAL_P3B="${INITIAL_P3B:?set INITIAL_P3B to the submitted phase3b array job id}"
WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_MAX_HOURS="${WATCH_MAX_HOURS:-72}"

exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/watch_dualpc_slurm.py \
  --workdir /home/infres/yinwang/CMI_AAAI \
  --array-script scripts/dualpc_phase3b_array.slurm \
  --post-script scripts/dualpc_phase3b_post.slurm \
  --gpu-partition V100 \
  --group 'p3b=0-6%5' \
  --initial "p3b=${INITIAL_P3B}" \
  --state-json results/dualpc_phase3b/slurm_state.json \
  --log logs/dualpc_phase3b_slurm_watcher.log \
  --interval "${WATCH_INTERVAL}" \
  --max-hours "${WATCH_MAX_HOURS}"
