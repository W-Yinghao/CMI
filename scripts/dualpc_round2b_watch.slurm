#!/usr/bin/env bash
#SBATCH --job-name=dualpc-r2b-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_opt_round2b

INITIAL_R2B="${INITIAL_R2B:?set INITIAL_R2B to the submitted round2b array job id}"
WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_MAX_HOURS="${WATCH_MAX_HOURS:-72}"

exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/watch_dualpc_slurm.py \
  --workdir /home/infres/yinwang/CMI_AAAI \
  --array-script scripts/dualpc_round2b_array.slurm \
  --post-script scripts/dualpc_round2b_post.slurm \
  --gpu-partition V100 \
  --group 'r2b=0-7%8' \
  --initial "r2b=${INITIAL_R2B}" \
  --state-json results/dualpc_opt_round2b/slurm_state.json \
  --log logs/dualpc_round2b_slurm_watcher.log \
  --interval "${WATCH_INTERVAL}" \
  --max-hours "${WATCH_MAX_HOURS}"
