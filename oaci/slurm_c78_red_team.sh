#!/usr/bin/env bash
#SBATCH --job-name=c78-redteam
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_oaci
export PYTHONDONTWRITEBYTECODE=1
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m oaci.conditioned_ceiling_coverage.c78_red_team
