#!/bin/bash
#SBATCH --job-name=b8_3_smoke
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/infres/yinwang/realeeg_feas/b8_3_smoke_%j.out
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/infres/yinwang/realeeg_feas
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[b8_3_smoke] host=$(hostname) START (Phase-A feasibility only, NO science verdict)"
python -u b8_3_smoke.py
echo "[b8_3_smoke] rc=$? DONE"
