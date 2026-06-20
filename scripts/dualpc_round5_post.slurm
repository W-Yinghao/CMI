#!/usr/bin/env bash
#SBATCH --job-name=dualpc-r5-post
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_round5

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

"$PY" analysis/dualpc_readiness.py results/dualpc_round5/*.json \
  --out-json results/dualpc_round5/dualpc_r5_readiness.json
"$PY" analysis/dualpc_paper_summary.py results/dualpc_round5/*.json \
  --out-json results/dualpc_round5/dualpc_r5_paper_summary.json
