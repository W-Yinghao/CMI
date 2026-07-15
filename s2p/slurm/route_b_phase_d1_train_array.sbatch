#!/usr/bin/env bash
#SBATCH --job-name=s2pD1train
#SBATCH --partition=A100,H100,A40,L40S,V100,V100-32GB,V100-16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-7
#SBATCH --time=96:00:00
#SBATCH --output=results/s2p_route_b_phase_d1_training/logs/train-%A_%a.out
#SBATCH --error=results/s2p_route_b_phase_d1_training/logs/train-%A_%a.err

set -euo pipefail

if [[ "${S2P_PM_D1_TRAINING_AUTHORIZED:-0}" != "1" ]]; then
  echo "ERROR: D1 training authorization is required" >&2
  exit 2
fi
for required in S2P_REPO_ROOT S2P_EXPECTED_COMMIT CBRAMOD_ROOT S2P_PYTHON; do
  if [[ -z "${!required:-}" ]]; then
    echo "ERROR: $required is required" >&2
    exit 2
  fi
done
REPO_ROOT=$(readlink -f "$S2P_REPO_ROOT")
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
"$S2P_PYTHON" - <<'PY'
import json
from pathlib import Path
gate = json.loads(Path("results/s2p_route_b_phase_d1_launch/phase_d1_canary_verdict.json").read_text())
if gate.get("status") != "PASS" or gate.get("launch_eight_trajectories_recommended") is not True:
    raise SystemExit("D1 training held until paired implementation canary PASS")
PY

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as explicit array 0-7}"
(( TASK_ID >= 0 && TASK_ID <= 7 )) || { echo "ERROR: task must be 0-7" >&2; exit 2; }
SUBSET_SEEDS=(0 0 0 0 1 1 1 1)
INIT_SEEDS=(0 0 1 1 0 0 1 1)
UNIQUE_H=(200 1000 200 1000 200 1000 200 1000)
SS="${SUBSET_SEEDS[$TASK_ID]}"
IS="${INIT_SEEDS[$TASK_ID]}"
H="${UNIQUE_H[$TASK_ID]}"
TRAJECTORY="SS${SS}_IS${IS}_U${H}"
ROOT=results/s2p_route_b_phase_d1_training
OUT="$ROOT/$TRAJECTORY"
mkdir -p "$OUT" "$ROOT/logs"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_d1_train_${SLURM_JOB_ID}_${TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$S2P_PYTHON" s2p/scripts/route_b_phase_d1_train.py \
  --repo-root "$REPO_ROOT" \
  --cbramod-root "$CBRAMOD_ROOT" \
  --trajectory-id "$TRAJECTORY" \
  --subset-seed "$SS" --init-seed "$IS" --unique-data-h "$H" \
  --output-dir "$OUT"
