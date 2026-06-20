#!/usr/bin/env bash
#SBATCH --job-name=dualpc-watch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_protocol_paper

INITIAL_G1="${INITIAL_G1:?set INITIAL_G1 to the submitted g1 job id}"
WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_MAX_HOURS="${WATCH_MAX_HOURS:-120}"

exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/watch_dualpc_slurm.py \
  --workdir /home/infres/yinwang/CMI_AAAI \
  --group 'g1=0-13%4' \
  --group 'g2=14-19%4' \
  --initial "g1=${INITIAL_G1}" \
  --interval "${WATCH_INTERVAL}" \
  --max-hours "${WATCH_MAX_HOURS}"
