#!/bin/bash
# Target Readout Calibration Ladder -- 252-cell FAIL-RESUMABLE CPU ARRAY (BNCI2014_001 27 + BNCI2015_001 36 +
# Lee2019_MI 162 + BNCI2014_004 27). Pure frozen-feature analysis (L-BFGS softmax heads + numpy), NO GPU / NO
# re-inference. Each task writes its own cell json + .done; aggregate ONLY after 252/252. env c84c. Manuscript FROZEN.
#
#   sbatch --array=0-251 scripts/sbatch_readout_label_efficiency.sh
#   # after 252/252: python scripts/aggregate_readout_label_efficiency.py --from-dir results/cmi_trace_readout --expect 252
#SBATCH --job-name=ro-ladder
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/ro/ro-ladder-%A_%a.out
#SBATCH --error=logs/ro/ro-ladder-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/ro results/cmi_trace_readout/cells
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
PY=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3/bin/python
OUTDIR="results/cmi_trace_readout"
IDX="${SLURM_ARRAY_TASK_ID:-${1:?need a cell index}}"
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) cell=$IDX"
if ls "$OUTDIR"/cells/cell_$(printf '%03d' "$IDX")_*.done >/dev/null 2>&1; then echo "cell $IDX already done -> skip"; exit 0; fi
"$PY" -m scripts.run_readout_label_efficiency --cell-index "$IDX" --out-dir "$OUTDIR" --n-random 50 --n-draws 50
echo "[ro-ladder] cell $IDX complete"
