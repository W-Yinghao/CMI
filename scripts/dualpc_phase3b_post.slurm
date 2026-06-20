#!/usr/bin/env bash
#SBATCH --job-name=dualpc-p3b-post
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_phase3b

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

"$PY" analysis/dualpc_readiness.py \
  results/dualpc_phase3/dualpc_p3_select_BNCI2014_001_guard_s0.json \
  results/dualpc_phase3b/*.json \
  --out-json results/dualpc_phase3b/dualpc_p3b_readiness.json
"$PY" analysis/dualpc_paper_summary.py \
  results/dualpc_phase3/dualpc_p3_select_BNCI2014_001_guard_s0.json \
  results/dualpc_phase3b/*.json \
  --out-json results/dualpc_phase3b/dualpc_p3b_paper_summary.json
