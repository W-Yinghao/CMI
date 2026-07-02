#!/usr/bin/env bash
# C8 throttled WAVE controller (persistent, lightweight): trickle the 27 (seed, target) fold-runs into free
# submit-cap slots (shared with the user's other work). Each Phase-A self-chains its decision Phase-B. Submit:
#   OACI_LOSO_ROOT=/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012 \
#     sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out \
#       oaci/slurm_c8_wave_controller.sh
# Tunables (env): OACI_MAX_TOTAL (overall cap headroom, default 28), OACI_MAX_C8 (max concurrent C8 jobs,
# default 8), OACI_POLL (seconds, default 120), OACI_SEEDS (default 0,1,2). Per cluster policy: NO --time.
#SBATCH --job-name=oaci-c8-wave
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -u
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
REPO="$(pwd)"
: "${OACI_LOSO_ROOT:?the wave controller requires OACI_LOSO_ROOT (the sweep root)}"
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
case "$OACI_LOSO_ROOT" in "$REPO"/*|"$REPO") echo "loso root must be OUTSIDE the repo" >&2; exit 1;; esac
[ -d "$OACI_DATALAKE_ROOT/MNE-bnci-data" ] || { echo "no datalake at $OACI_DATALAKE_ROOT" >&2; exit 1; }
[ -z "$(git -C "$REPO" status --porcelain -- oaci)" ] || { echo "dirty oaci/ tree; commit before the wave run" >&2; exit 1; }
echo "[c8-wave-controller] node=$(hostname) commit=$(git -C "$REPO" rev-parse --short HEAD) root=$OACI_LOSO_ROOT"
$PY -m oaci.confirmatory.c8_wave --loso-root "$OACI_LOSO_ROOT" --repo-root "$REPO" \
    --datalake-root "$OACI_DATALAKE_ROOT" --seeds "${OACI_SEEDS:-0,1,2}" \
    --max-total "${OACI_MAX_TOTAL:-28}" --max-c8 "${OACI_MAX_C8:-8}" --poll "${OACI_POLL:-120}"
rc=$?
echo "=== c8-wave controller exited rc=$rc ==="
exit "$rc"
