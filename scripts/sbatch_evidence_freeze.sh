#!/bin/bash
#SBATCH --job-name=cigl-freeze
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/cigl/cigl-freeze-%j.out
#SBATCH --error=logs/cigl/cigl-freeze-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cigl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"
"$PY" scripts/build_evidence_freeze.py --gate_dir results/cigl/r2_seed0_gate --out_dir results/cigl_r123/final --n_boot 4000 --workers 16
