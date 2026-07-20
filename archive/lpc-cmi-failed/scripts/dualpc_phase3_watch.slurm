#!/usr/bin/env bash
#SBATCH --job-name=dualpc-p3-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_phase3

INITIAL_P3="${INITIAL_P3:?set INITIAL_P3 to the submitted phase3 array job id}"
WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_MAX_HOURS="${WATCH_MAX_HOURS:-72}"

exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/watch_dualpc_slurm.py \
  --workdir /home/infres/yinwang/CMI_AAAI \
  --array-script scripts/dualpc_phase3_array.slurm \
  --post-script scripts/dualpc_phase3_post.slurm \
  --gpu-partition V100 \
  --group 'p3=0-3%8' \
  --initial "p3=${INITIAL_P3}" \
  --state-json results/dualpc_phase3/slurm_state.json \
  --log logs/dualpc_phase3_slurm_watcher.log \
  --interval "${WATCH_INTERVAL}" \
  --max-hours "${WATCH_MAX_HOURS}"
