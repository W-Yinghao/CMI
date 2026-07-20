#!/usr/bin/env bash
#SBATCH --job-name=dualpc-r2b-post
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_opt_round2b

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

"$PY" analysis/dualpc_readiness.py results/dualpc_opt_round2b/*.json \
  --out-json results/dualpc_opt_round2b/dualpc_r2b_readiness.json
"$PY" analysis/dualpc_paper_summary.py results/dualpc_opt_round2b/*.json \
  --out-json results/dualpc_opt_round2b/dualpc_r2b_paper_summary.json
