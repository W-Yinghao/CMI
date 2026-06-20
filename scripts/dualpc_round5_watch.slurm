#!/usr/bin/env bash
#SBATCH --job-name=dualpc-r5-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_round5

INITIAL_R5="${INITIAL_R5:?set INITIAL_R5 to the submitted round5 array job id}"
WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_MAX_HOURS="${WATCH_MAX_HOURS:-120}"

exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/watch_dualpc_slurm.py \
  --workdir /home/infres/yinwang/CMI_AAAI \
  --array-script scripts/dualpc_round5_array.slurm \
  --post-script scripts/dualpc_round5_post.slurm \
  --gpu-partition V100 \
  --group 'r5=0-5%6' \
  --initial "r5=${INITIAL_R5}" \
  --state-json results/dualpc_round5/slurm_state.json \
  --log logs/dualpc_round5_slurm_watcher.log \
  --interval "${WATCH_INTERVAL}" \
  --max-hours "${WATCH_MAX_HOURS}"
