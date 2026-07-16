#!/bin/bash
#SBATCH --job-name=cmitrace-ladder
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/cmi_trace/%x-%j.out
#SBATCH --error=logs/cmi_trace/%x-%j.out
# Cluster convention: no --qos/--time. The DGCNN ladder is CPU-only (existing audit npz, sklearn readouts).
# Params via env: LAD_FAMILY (dgcnn_graph_z|tos_frozen), LAD_DATASET, LAD_METHODS, LAD_BACKBONE, LAD_NRANDOM.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/cmi_trace
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1   # sklearn: process-level parallelism, not intra-op
export MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data

LAD_FAMILY="${LAD_FAMILY:-dgcnn_graph_z}"
LAD_DATASET="${LAD_DATASET:-BNCI2014_001}"
LAD_METHODS="${LAD_METHODS:-erm cigl_graph_node}"
LAD_BACKBONE="${LAD_BACKBONE:-EEGNet}"
LAD_NRANDOM="${LAD_NRANDOM:-50}"

PY=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3/bin/python
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) family=${LAD_FAMILY} dataset=${LAD_DATASET}"

if [ "${LAD_FAMILY}" = "tos_frozen" ]; then
  "$PY" scripts/run_cmi_trace_relaxation_ladder.py --family tos_frozen --dataset "${LAD_DATASET}" \
      --backbone "${LAD_BACKBONE}" --n_random "${LAD_NRANDOM}" --with_tos_vd
else
  "$PY" scripts/run_cmi_trace_relaxation_ladder.py --family dgcnn_graph_z --dataset "${LAD_DATASET}" \
      --methods ${LAD_METHODS} --n_random "${LAD_NRANDOM}"
fi
