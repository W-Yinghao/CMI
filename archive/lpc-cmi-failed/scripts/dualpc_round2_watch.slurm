#!/usr/bin/env bash
#SBATCH --job-name=dualpc-r2-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_opt_round2

INITIAL_R2="${INITIAL_R2:?set INITIAL_R2 to the submitted round2 array job id}"
WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_MAX_HOURS="${WATCH_MAX_HOURS:-72}"

exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/watch_dualpc_slurm.py \
  --workdir /home/infres/yinwang/CMI_AAAI \
  --array-script scripts/dualpc_round2_array.slurm \
  --post-script scripts/dualpc_round2_post.slurm \
  --gpu-partition A40 \
  --group 'r2=0-3%8' \
  --initial "r2=${INITIAL_R2}" \
  --state-json results/dualpc_opt_round2/slurm_state.json \
  --log logs/dualpc_round2_slurm_watcher.log \
  --interval "${WATCH_INTERVAL}" \
  --max-hours "${WATCH_MAX_HOURS}"
