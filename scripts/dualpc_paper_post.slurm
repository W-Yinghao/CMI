#!/usr/bin/env bash
#SBATCH --job-name=dualpc-post
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI
mkdir -p logs results/dualpc_protocol_paper /tmp/matplotlib
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib

# Readiness summary after the JSONs above exist
/home/infres/yinwang/anaconda3/envs/icml/bin/python analysis/dualpc_readiness.py 'results/dualpc_protocol_paper/*.json' --out-json results/dualpc_protocol_paper/dualpc_readiness.json

# Paper aggregate summary after readiness
/home/infres/yinwang/anaconda3/envs/icml/bin/python analysis/dualpc_paper_summary.py 'results/dualpc_protocol_paper/*.json' --out-json results/dualpc_protocol_paper/dualpc_paper_summary.json

# Headline decision gate after summaries
/home/infres/yinwang/anaconda3/envs/icml/bin/python analysis/dualpc_decision_gate.py --readiness results/dualpc_protocol_paper/dualpc_readiness.json --paper-summary results/dualpc_protocol_paper/dualpc_paper_summary.json --min-comparison-tasks 4 --min-selector-tasks 2 --required-baselines erm lpc_prior --required-synthetic-groups null_prior concept all_three --out-json results/dualpc_protocol_paper/dualpc_decision.json
