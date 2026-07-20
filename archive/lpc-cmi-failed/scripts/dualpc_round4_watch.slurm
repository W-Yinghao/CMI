#!/usr/bin/env bash
#SBATCH --job-name=dualpc-r4-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_round4

INITIAL_R4="${INITIAL_R4:?set INITIAL_R4 to the submitted round4 array job id}"
WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_MAX_HOURS="${WATCH_MAX_HOURS:-120}"

exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/watch_dualpc_slurm.py \
  --workdir /home/infres/yinwang/CMI_AAAI \
  --array-script scripts/dualpc_round4_array.slurm \
  --post-script scripts/dualpc_round4_post.slurm \
  --gpu-partition V100 \
  --group 'r4=0-5%6' \
  --initial "r4=${INITIAL_R4}" \
  --state-json results/dualpc_round4/slurm_state.json \
  --log logs/dualpc_round4_slurm_watcher.log \
  --interval "${WATCH_INTERVAL}" \
  --max-hours "${WATCH_MAX_HOURS}"
