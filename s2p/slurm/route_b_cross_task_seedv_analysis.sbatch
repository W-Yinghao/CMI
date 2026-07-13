#!/usr/bin/env bash
#SBATCH --job-name=s2pCseedv
#SBATCH --partition=A100,H100,A40,L40S,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/s2p_route_b_cross_task_logs/seedv-analysis-%j.out
#SBATCH --error=results/s2p_route_b_cross_task_logs/seedv-analysis-%j.err

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
STAGE="${S2P_PHASE_C_STAGE:-gate}"
if [[ "$STAGE" != "gate" && "$STAGE" != "fleet" ]]; then
  echo "ERROR: invalid stage" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_phase_c_seedv_${SLURM_JOB_ID:-local}"
PY="$S2P_PYTHON"
ROOT=results/s2p_route_b_cross_task_seedv
GATE_ARGS=()
if [[ "$STAGE" == "fleet" ]]; then
  GATE_ARGS=(--gate-json "$ROOT/seedv_gate_verdict.json")
fi
srun "$PY" s2p/scripts/route_b_cross_task_seedv_analysis.py \
  --stage "$STAGE" \
  --checkpoint-manifest results/s2p_route_b_phase_b_checkpoint_closure/phase_b_checkpoint_immutable_manifest.csv \
  --feature-dir "$ROOT/features" \
  --out-dir "$ROOT" \
  "${GATE_ARGS[@]}"
if [[ "$STAGE" == "gate" ]]; then
  srun "$PY" s2p/scripts/route_b_cross_task_seedv_verify.py \
    --feature-dir "$ROOT/features" --result-dir "$ROOT"
fi
