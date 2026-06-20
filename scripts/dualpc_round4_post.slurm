#!/usr/bin/env bash
#SBATCH --job-name=dualpc-r4-post
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_round4

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python

"$PY" analysis/dualpc_readiness.py results/dualpc_round4/*.json \
  --out-json results/dualpc_round4/dualpc_r4_readiness.json
"$PY" analysis/dualpc_paper_summary.py results/dualpc_round4/*.json \
  --out-json results/dualpc_round4/dualpc_r4_paper_summary.json
