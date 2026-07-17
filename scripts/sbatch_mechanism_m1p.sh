#!/bin/bash
# M1-P: confirmatory PRIMARY tranche (EEGNet x contrast_disagreement x (9+12) subjects x 3 seeds = 63 cells),
# amendment 03 (shared-null conditional estimand). FAIL-RESUMABLE SLURM ARRAY (one task per fold-seed cell). CPU
# only (frozen features + linear readout + low-dim eigensystem; no GPU). Each task writes its own cell jsonl +
# .done marker; the aggregator REFUSES to produce a confirmatory verdict until all 63 .done markers exist (no
# threshold edits on partial results). Estimator / threshold / control / gate / contrast are FROZEN in config v4 --
# this array only selects which cells run. Manuscript FROZEN. Only the project owner may stop a scientific line.
#
#   sbatch --array=0-62 scripts/sbatch_mechanism_m1p.sh
#   # after all complete:
#   python scripts/aggregate_mechanism_subspace_oracle.py --from-dir results/cmi_trace_mechanism_subspace/m1p --expect 63 --tag m1p
#SBATCH --job-name=mech-m1p
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/mech/m1p-%A_%a.out
#SBATCH --error=logs/mech/m1p-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/mech
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export _MNE_FAKE_HOME_DIR="/tmp/mne_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$_MNE_FAKE_HOME_DIR/.mne"
PY=/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3/bin/python
OUTDIR="results/cmi_trace_mechanism_subspace/m1p"
IDX="${SLURM_ARRAY_TASK_ID:-${1:?need a cell index}}"
# stale-code signature: print branch/SHA/config hash; the runner ALSO prints them at cell start (must match).
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) task=$IDX"
echo "config_hash=$("$PY" -c "from tos_cmi.eval.mechanism_subspace import config_hash; print(config_hash('configs/cmi_trace_mechanism_subspace_oracle_v4.yaml'))")"
DONE="$OUTDIR"/cell_$(printf '%03d' "$IDX")_*.done
if compgen -G "$DONE" > /dev/null; then echo "cell $IDX already done ($DONE) -> skip (resumable)"; exit 0; fi
"$PY" scripts/run_mechanism_subspace_oracle.py --cell-index "$IDX" \
  --backbone EEGNet --family contrast_disagreement --seeds 0 1 2 \
  --n_random 100 --blocks 2 --out-dir "$OUTDIR"
echo "[m1p] cell $IDX complete"
