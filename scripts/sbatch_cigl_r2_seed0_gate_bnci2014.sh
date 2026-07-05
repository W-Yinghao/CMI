#!/bin/bash
#SBATCH --job-name=cigl-r2gate-bnci2014
#SBATCH --partition=A100,V100,V100-32GB,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/cigl/cigl-r2gate-bnci2014-%j.out
#SBATCH --error=logs/cigl/cigl-r2gate-bnci2014-%j.out
# NOTE: no --qos and no --time on purpose (cluster convention: default QOS, no walltime cap).
# Seed parameterized via GATE_SEED (default 0): sbatch --export=ALL,GATE_SEED=1 <this>. Seed is NOT method
# semantics; training is identical across seeds (fixed epochs + warmup, no early stopping -- matches seed0).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cigl

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
GATE_SEED="${GATE_SEED:-0}"

PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
echo "host=$(hostname)  branch=$(git rev-parse --abbrev-ref HEAD)  commit=$(git rev-parse --short HEAD)  seed=${GATE_SEED}"

# Fail closed: the gate must run on GPU (never silently fall back to CPU).
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: CUDA not available on $(hostname); refusing to run on CPU." >&2
  exit 1
fi

# R2 gate: 5 approved methods (erm, cigl_graph_node, dann, cond_dann, cdan) on the SAME DGCNN adapter, ALL LOSO
# folds, seed=GATE_SEED. Resumable. Writes metrics JSON + Pareto rows + verified head-replay .audit.npz + firewall.
"$PY" scripts/run_cigl_r2_seed0_gate.py \
  --dataset BNCI2014_001 \
  --device cuda \
  --seed "${GATE_SEED}" \
  --epochs 80 \
  --probe_epochs 100 \
  --n_perm 50 \
  --gate_alpha 0.05
