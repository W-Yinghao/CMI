#!/usr/bin/env bash
#SBATCH --job-name=s2pCisagg
#SBATCH --partition=A100,H100,A40,L40S,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/s2p_route_b_cross_task_logs/isruc-aggregate-%j.out
#SBATCH --error=results/s2p_route_b_cross_task_logs/isruc-aggregate-%j.err

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
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_phase_c_isruc_aggregate_${SLURM_JOB_ID:-local}"
PY="$S2P_PYTHON"
ROOT=results/s2p_route_b_cross_task_isruc_s3
STAGE="${S2P_PHASE_C_STAGE:-gate}"
GATE_ARGS=()
if [[ "$STAGE" == "fleet" ]]; then
  GATE_ARGS=(--gate-json "$ROOT/isruc_s3_gate_verdict.json")
  OBJECT_DIR="$ROOT/objects_fleet_a40"
else
  OBJECT_DIR="$ROOT/objects"
fi
srun "$PY" s2p/scripts/route_b_cross_task_isruc_aggregate.py \
  --stage "$STAGE" --object-dir "$OBJECT_DIR" --out-dir "$ROOT" "${GATE_ARGS[@]}"
