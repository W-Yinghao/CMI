#!/usr/bin/env bash
#SBATCH --job-name=dualpc-p3-result-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_phase3/per_task_analysis

WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_MAX_HOURS="${WATCH_MAX_HOURS:-72}"

exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/watch_dualpc_results.py \
  --workdir /home/infres/yinwang/CMI_AAAI \
  --task-file scripts/dualpc_phase3_tasks.tsv \
  --slurm-state results/dualpc_phase3/slurm_state.json \
  --state-json results/dualpc_phase3/result_watch_state.json \
  --analysis-dir results/dualpc_phase3/per_task_analysis \
  --partial-dir results/dualpc_phase3 \
  --log logs/dualpc_phase3_result_watcher.log \
  --interval "${WATCH_INTERVAL}" \
  --max-hours "${WATCH_MAX_HOURS}"
