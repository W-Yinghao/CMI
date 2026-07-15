#!/bin/bash
#SBATCH --job-name=cmitrace-obj
#SBATCH --partition=A100,V100,V100-32GB,A40,P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/cmi_trace/%x-%j.out
#SBATCH --error=logs/cmi_trace/%x-%j.out
# Cluster convention: no --qos and no --time (default QOS, no walltime cap).
# Parameters via env: OBJ_DATASET (BNCI2014_001|BNCI2015_001), OBJ_SEED (0|1|2), OBJ_METHODS (space list).
# Resumable: an existing per-cell JSON is skipped. Strict source-only firewall; target eval-only.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cmi_trace

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
# Pin the MOABB/MNE cache to the writable home cache. --export=ALL otherwise inherits a shell env that
# resolves BNCI to a permission-denied shared datalake (/projects/.../datalake/raw), which fails BNCI2015_001.
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data

OBJ_DATASET="${OBJ_DATASET:-BNCI2014_001}"
# Run ALL seeds in ONE job (data loaded once): avoids the concurrent-NFS-file-lock collision
# ("OSError: No locks available") seen when 3 per-seed jobs load the same moabb cache simultaneously.
OBJ_SEEDS="${OBJ_SEEDS:-0 1 2}"
OBJ_METHODS="${OBJ_METHODS:-erm cigl_graph_node cigl_nested coral label_coral irm vrex cond_dann}"

PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python   # CUDA torch 2.6.0+cu124 (GPU env)
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) dataset=${OBJ_DATASET} seeds=${OBJ_SEEDS}"

# Fail closed: must run on GPU (never silently fall back to CPU).
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: CUDA not available on $(hostname); refusing CPU fallback." >&2
  exit 1
fi

"$PY" scripts/run_cmi_trace_objective_comparison.py \
  --dataset "${OBJ_DATASET}" \
  --device cuda \
  --seeds ${OBJ_SEEDS} \
  --methods ${OBJ_METHODS} \
  --epochs 80 \
  --probe_epochs 100 \
  --n_perm 50 \
  --select_epochs 40 \
  --select_inner_folds 3 \
  --primary_k 2
