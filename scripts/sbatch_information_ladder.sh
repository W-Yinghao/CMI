#!/bin/bash
# Information-Regime Ladder (Track B) -- 63-cell FAIL-RESUMABLE CPU ARRAY. Pure frozen-feature analysis (sklearn
# logistic + numpy), NO GPU / NO re-inference. Each task writes its own cell json + .done; aggregate ONLY after 63/63.
# env c84c. Manuscript FROZEN; only the project owner stops/redirects a scientific line.
#
#   sbatch --array=0-62 scripts/sbatch_information_ladder.sh
#   # after 63/63: python scripts/aggregate_information_ladder.py --from-dir results/cmi_trace_info_ladder --expect 63
#SBATCH --job-name=info-ladder
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/il/info-ladder-%A_%a.out
#SBATCH --error=logs/il/info-ladder-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/il results/cmi_trace_info_ladder/cells
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1     # process-level parallelism per compute policy
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
PY=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3/bin/python
OUTDIR="results/cmi_trace_info_ladder"
IDX="${SLURM_ARRAY_TASK_ID:-${1:?need a cell index}}"
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) cell=$IDX"
STEM=$("$PY" -c "from scripts.run_information_ladder import enumerate_cells as e; import pathlib; ds,p=e()[$IDX]; print(f'cell_{$IDX:03d}_{ds}')")
if ls "$OUTDIR"/cells/${STEM}_*.done >/dev/null 2>&1; then echo "cell $IDX ($STEM) already done -> skip"; exit 0; fi
"$PY" -m scripts.run_information_ladder --cell-index "$IDX" --out-dir "$OUTDIR" --n-random 10 --n-draws 20
echo "[info-ladder] cell $IDX complete"
