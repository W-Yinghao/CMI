#!/usr/bin/env bash
#SBATCH --job-name=dualpc-result-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_protocol_paper/per_task_analysis

WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_MAX_HOURS="${WATCH_MAX_HOURS:-120}"

exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/watch_dualpc_results.py \
  --workdir /home/infres/yinwang/CMI_AAAI \
  --interval "${WATCH_INTERVAL}" \
  --max-hours "${WATCH_MAX_HOURS}"
