#!/usr/bin/env bash
#SBATCH --job-name=s2pP2obj
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-19%8
#SBATCH --time=24:00:00
#SBATCH --output=results/s2p_fmscope_fsr_bridge_panel2/logs/object-%A_%a.out
#SBATCH --error=results/s2p_fmscope_fsr_bridge_panel2/logs/object-%A_%a.err

set -euo pipefail

if [[ "${S2P_PM_PANEL2_AUTHORIZED:-0}" != "1" ]]; then
  echo "ERROR: Panel 2 authorization is required" >&2
  exit 2
fi
for required in S2P_REPO_ROOT S2P_EXPECTED_COMMIT S2P_PYTHON PHASE_C_ARTIFACT_ROOT; do
  if [[ -z "${!required:-}" ]]; then
    echo "ERROR: $required is required" >&2
    exit 2
  fi
done
REPO_ROOT=$(readlink -f "$S2P_REPO_ROOT")
cd "$REPO_ROOT"
if [[ ! -d s2p || "$REPO_ROOT" == /var/spool/* ]]; then
  echo "ERROR: invalid repository root" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "$S2P_EXPECTED_COMMIT" || -n "$(git status --short --untracked-files=no)" ]]; then
  echo "ERROR: launch provenance mismatch" >&2
  exit 2
fi
TAGS=(random released H200_s0 H200_s1 H500_s0 H500_s1 H1000_s0 H1000_s1 H2000_s0 H2000_s1)
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-19}"
(( TASK_ID >= 0 && TASK_ID < 20 )) || { echo "ERROR: task must be 0-19" >&2; exit 2; }
TAG="${TAGS[$((TASK_ID % 10))]}"
ROOT=results/s2p_fmscope_fsr_bridge_panel2
if (( TASK_ID < 10 )); then
  DATASET=faced
  FEATURE="$ROOT/features/faced/${TAG}_faced_pooled_features.npz"
else
  DATASET=seedv
  FEATURE="$PHASE_C_ARTIFACT_ROOT/results/s2p_route_b_cross_task_seedv/features/${TAG}_features.npz"
fi
[[ -f "$FEATURE" ]] || { echo "ERROR: feature payload missing: $FEATURE" >&2; exit 2; }
OUT="$ROOT/objects/$DATASET/$TAG"
mkdir -p "$OUT" "$ROOT/logs"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_panel2_object_${SLURM_JOB_ID}_${TASK_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun "$S2P_PYTHON" s2p/scripts/fmscope_fsr_bridge_panel2_object.py \
  --dataset "$DATASET" --tag "$TAG" \
  --feature-payload "$FEATURE" --out-dir "$OUT" --n-random 100
