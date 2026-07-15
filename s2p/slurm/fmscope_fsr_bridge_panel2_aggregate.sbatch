#!/usr/bin/env bash
#SBATCH --job-name=s2pP2agg
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --output=results/s2p_fmscope_fsr_bridge_panel2/logs/aggregate-%A_%a.out
#SBATCH --error=results/s2p_fmscope_fsr_bridge_panel2/logs/aggregate-%A_%a.err

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
DATASETS=(faced seedv)
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-1}"
(( TASK_ID >= 0 && TASK_ID <= 1 )) || { echo "ERROR: task must be 0 or 1" >&2; exit 2; }
DATASET="${DATASETS[$TASK_ID]}"
ROOT=results/s2p_fmscope_fsr_bridge_panel2
OBJECT_ROOT="$ROOT/objects/$DATASET"
OUT="$ROOT/aggregate/$DATASET"
if [[ "$DATASET" == "faced" ]]; then
  FEATURE_ROOT="$ROOT/features/faced"
else
  FEATURE_ROOT="$PHASE_C_ARTIFACT_ROOT/results/s2p_route_b_cross_task_seedv/features"
fi
mkdir -p "$OUT" "$ROOT/logs"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_panel2_aggregate_${SLURM_JOB_ID}_${TASK_ID}"

srun "$S2P_PYTHON" s2p/scripts/fmscope_fsr_bridge_panel2_aggregate.py \
  --dataset "$DATASET" --object-root "$OBJECT_ROOT" --out-dir "$OUT"
srun "$S2P_PYTHON" s2p/scripts/fmscope_fsr_bridge_panel2_verify.py \
  --dataset "$DATASET" --object-root "$OBJECT_ROOT" --aggregate-dir "$OUT" \
  --feature-root "$FEATURE_ROOT" \
  --out-json "$OUT/${DATASET}_panel2_independent_verification.json"
