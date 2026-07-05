#!/bin/bash
#SBATCH --job-name=cigl-r1-hardened
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --mem=48G
#SBATCH --output=logs/cigl/cigl-r1-hardened-%j.out
#SBATCH --error=logs/cigl/cigl-r1-hardened-%j.out
# CPU-only confirmatory audit (no GPU). No --qos/--time (cluster convention). CHUNKED by method via R1_METHOD
# env (submit 3: erm, cigl_graph_node, cdan) so all 3 methods' folds run in parallel across nodes; each chunk =
# 21 folds on 24 physical cores (nomultithread avoids the worker oversubscription that made a single 32-worker
# job ~5x slow). Each chunk writes its own CSV; merge at append.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cigl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R1_METHOD="${R1_METHOD:-erm}"

PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) method=${R1_METHOD}"

"$PY" scripts/r1_hardened_audit.py \
  --gate_dir results/cigl/r2_seed0_gate \
  --seed 0 \
  --n_perm 1000 \
  --epochs 100 \
  --workers 21 \
  --methods "${R1_METHOD}" \
  --out_csv "results/cigl/r2_seed0_gate/R1_chunk_${R1_METHOD}_nperm1000.csv"
