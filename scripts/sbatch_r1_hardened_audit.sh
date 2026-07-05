#!/bin/bash
#SBATCH --job-name=cigl-r1-hardened
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --output=logs/cigl/cigl-r1-hardened-%j.out
#SBATCH --error=logs/cigl/cigl-r1-hardened-%j.out
# CPU-only confirmatory audit (no GPU). No --qos/--time (cluster convention).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cigl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1     # each worker is single-threaded; parallelism is over folds

PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
echo "host=$(hostname)  branch=$(git rev-parse --abbrev-ref HEAD)  commit=$(git rev-parse --short HEAD)"

# R1 hardened n_perm=1000 recompute from SEED0 saved features (no retrain). ERM + CIGL + CDAN.
"$PY" scripts/r1_hardened_audit.py \
  --gate_dir results/cigl/r2_seed0_gate \
  --seed 0 \
  --n_perm 1000 \
  --epochs 100 \
  --workers 32 \
  --methods erm cigl_graph_node cdan
