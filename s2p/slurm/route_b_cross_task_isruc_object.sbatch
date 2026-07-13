#!/usr/bin/env bash
#SBATCH --job-name=s2pCisruc
#SBATCH --partition=A100,H100,A40,L40S,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/s2p_route_b_cross_task_logs/isruc-object-%A_%a.out
#SBATCH --error=results/s2p_route_b_cross_task_logs/isruc-object-%A_%a.err

set -euo pipefail
if [[ "${S2P_PM_PHASE_C_GATE_AUTHORIZED:-0}" != "1" || -z "${S2P_EXPECTED_COMMIT:-}" || -z "${S2P_PYTHON:-}" ]]; then
  echo "ERROR: Phase C authorization, expected commit, and S2P_PYTHON are required" >&2
  exit 2
fi
REPO_ROOT="${S2P_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
REPO_ROOT=$(readlink -f "$REPO_ROOT")
if [[ ! -d "$REPO_ROOT/s2p" || "$REPO_ROOT" == /var/spool/* ]]; then
  echo "ERROR: invalid repository root" >&2
  exit 2
fi
cd "$REPO_ROOT"
if [[ "$(git rev-parse HEAD)" != "$S2P_EXPECTED_COMMIT" || -n "$(git status --short --untracked-files=no)" ]]; then
  echo "ERROR: launch provenance mismatch" >&2
  exit 2
fi
GATE_TAGS=(random released H200_s0 H1000_s0 H2000_s0)
FLEET_TAGS=(random released H200_s0 H200_s1 H500_s0 H500_s1 H1000_s0 H1000_s1 H2000_s0 H2000_s1)
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as an explicit array}"
STAGE="${S2P_PHASE_C_STAGE:-gate}"
ROOT=results/s2p_route_b_cross_task_isruc_s3
GATE_ARGS=()
if [[ "$STAGE" == "gate" ]]; then
  (( TASK_ID >= 0 && TASK_ID < 5 )) || { echo "ERROR: gate task must be 0-4" >&2; exit 2; }
  TAG="${GATE_TAGS[$TASK_ID]}"
  OBJECT_DIR="$ROOT/objects"
elif [[ "$STAGE" == "fleet" ]]; then
  "$S2P_PYTHON" - <<'PY'
import json
from pathlib import Path
if json.loads(Path("results/s2p_route_b_cross_task_isruc_s3/isruc_s3_gate_verdict.json").read_text()).get("status") != "PASS":
    raise SystemExit("ISRUC fleet is held until gate PASS")
PY
  (( TASK_ID >= 0 && TASK_ID < 10 )) || { echo "ERROR: fleet task must be 0-9" >&2; exit 2; }
  TAG="${FLEET_TAGS[$TASK_ID]}"
  OBJECT_DIR="$ROOT/objects_fleet"
  GATE_ARGS=(--gate-task-csv "$ROOT/isruc_s3_gate_task_performance.csv")
else
  echo "ERROR: invalid stage" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_phase_c_isruc_${SLURM_JOB_ID:-local}_${TASK_ID}"
PY="$S2P_PYTHON"
mkdir -p "$OBJECT_DIR"
srun "$PY" s2p/scripts/route_b_cross_task_isruc_object.py \
  --stage "$STAGE" \
  --tag "$TAG" \
  --checkpoint-manifest results/s2p_route_b_phase_b_checkpoint_closure/phase_b_checkpoint_immutable_manifest.csv \
  --feature-dir "$ROOT/features" \
  --rotation-manifest results/s2p_route_b_cross_task_c0/isruc_s3_rotation_manifest.csv \
  --out-dir "$OBJECT_DIR" \
  --device cuda:0 \
  "${GATE_ARGS[@]}"
