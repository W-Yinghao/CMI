#!/usr/bin/env bash
#SBATCH --job-name=s2pCfeat
#SBATCH --partition=A100,H100,A40,L40S,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/s2p_route_b_cross_task_logs/features-%A_%a.out
#SBATCH --error=results/s2p_route_b_cross_task_logs/features-%A_%a.err

set -euo pipefail

if [[ "${S2P_PM_PHASE_C_GATE_AUTHORIZED:-0}" != "1" ]]; then
  echo "ERROR: Phase C gate authorization is required" >&2
  exit 2
fi
for required in S2P_EXPECTED_COMMIT S2P_PHASE_C_DATASET S2P_PHASE_C_STAGE CBRAMOD_ROOT PHASE_B_ARTIFACT_ROOT S2P_PYTHON; do
  if [[ -z "${!required:-}" ]]; then
    echo "ERROR: $required is required" >&2
    exit 2
  fi
done
if [[ -n "${S2P_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${S2P_REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/s2p" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  echo "ERROR: cannot resolve repository root" >&2
  exit 2
fi
REPO_ROOT=$(readlink -f "$REPO_ROOT")
if [[ ! -d "$REPO_ROOT/s2p" || "$REPO_ROOT" == /var/spool/* ]]; then
  echo "ERROR: invalid repository root: $REPO_ROOT" >&2
  exit 2
fi
cd "$REPO_ROOT"
if [[ "$(git rev-parse HEAD)" != "$S2P_EXPECTED_COMMIT" ]]; then
  echo "ERROR: expected commit does not match HEAD" >&2
  exit 2
fi
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
  echo "ERROR: tracked worktree changes detected" >&2
  exit 2
fi

python_check="$S2P_PYTHON"
"$python_check" - <<'PY'
import json
from pathlib import Path
gate = json.loads(Path("results/s2p_route_b_cross_task_c0/phase_c0_go_nogo.json").read_text())
if gate.get("status") != "PASS" or gate.get("scientific_endpoint_computed") is not False:
    raise SystemExit("C0 gate is not a metadata-only PASS")
PY

case "$S2P_PHASE_C_DATASET" in
  seedv)
    DATASET_ROOT="${SEEDV_ROOT:?SEEDV_ROOT is required}"
    RESULT_ROOT="results/s2p_route_b_cross_task_seedv"
    ;;
  isruc_s3)
    DATASET_ROOT="${ISRUC_PROCESSED_ROOT:?ISRUC_PROCESSED_ROOT is required}"
    RESULT_ROOT="results/s2p_route_b_cross_task_isruc_s3"
    ;;
  *)
    echo "ERROR: unsupported dataset $S2P_PHASE_C_DATASET" >&2
    exit 2
    ;;
esac

GATE_TAGS=(random released H200_s0 H1000_s0 H2000_s0)
FLEET_TAGS=(random released H200_s0 H200_s1 H500_s0 H500_s1 H1000_s0 H1000_s1 H2000_s0 H2000_s1)
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as an explicit array}"
if [[ "$S2P_PHASE_C_STAGE" == "gate" ]]; then
  if (( TASK_ID < 0 || TASK_ID >= ${#GATE_TAGS[@]} )); then
    echo "ERROR: gate array task must be 0-4" >&2
    exit 2
  fi
  TAG="${GATE_TAGS[$TASK_ID]}"
elif [[ "$S2P_PHASE_C_STAGE" == "fleet" ]]; then
  GATE_JSON="$RESULT_ROOT/${S2P_PHASE_C_DATASET}_gate_verdict.json"
  "$python_check" - "$GATE_JSON" <<'PY'
import json, sys
if json.load(open(sys.argv[1])).get("status") != "PASS":
    raise SystemExit("dataset fleet is held until its gate passes")
PY
  if (( TASK_ID < 0 || TASK_ID >= ${#FLEET_TAGS[@]} )); then
    echo "ERROR: fleet array task must be 0-9" >&2
    exit 2
  fi
  TAG="${FLEET_TAGS[$TASK_ID]}"
else
  echo "ERROR: stage must be gate or fleet" >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT/features"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_phase_c_features_${SLURM_JOB_ID:-local}_${TASK_ID}"

srun "$python_check" s2p/scripts/route_b_cross_task_extract_features.py \
  --dataset "$S2P_PHASE_C_DATASET" \
  --tag "$TAG" \
  --checkpoint-manifest results/s2p_route_b_phase_b_checkpoint_closure/phase_b_checkpoint_immutable_manifest.csv \
  --cbramod-root "$CBRAMOD_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --out-dir "$RESULT_ROOT/features" \
  --device cuda:0 \
  --batch-size 64
