#!/usr/bin/env bash
#SBATCH --job-name=dualpc-p3b-result-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_phase3b/per_task_analysis

WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_MAX_HOURS="${WATCH_MAX_HOURS:-72}"

exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/watch_dualpc_results.py \
  --workdir /home/infres/yinwang/CMI_AAAI \
  --task-file scripts/dualpc_phase3b_tasks.tsv \
  --slurm-state results/dualpc_phase3b/slurm_state.json \
  --state-json results/dualpc_phase3b/result_watch_state.json \
  --analysis-dir results/dualpc_phase3b/per_task_analysis \
  --partial-dir results/dualpc_phase3b \
  --log logs/dualpc_phase3b_result_watcher.log \
  --interval "${WATCH_INTERVAL}" \
  --max-hours "${WATCH_MAX_HOURS}"
