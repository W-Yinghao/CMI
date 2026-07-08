#!/usr/bin/env bash
#SBATCH --job-name=w1offdiag-post
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%j.out
#SBATCH --error=/home/infres/yinwang/CMI_AAAI_qxu/h2cmi/results/review_completion/slurm/logs/%x-%j.err

set -euo pipefail

cd /home/infres/yinwang/CMI_AAAI_qxu
mkdir -p h2cmi/results/review_completion/slurm/logs h2cmi/results/review_completion

PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
"$PY" -m h2cmi.analyze_geometry_offdiag \
  --inputs 'results/h2cmi/review_completion_offdiag/w1offdiag_*.jsonl' \
  --out h2cmi/results/review_completion/geometry_capacity_offdiagonal_results.csv
