#!/usr/bin/env bash
#SBATCH --job-name=s2pD1can
#SBATCH --partition=A100,H100,A40,L40S,V100,V100-32GB,V100-16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --time=01:00:00
#SBATCH --output=results/s2p_route_b_phase_d1_launch/logs/canary-%A_%a.out
#SBATCH --error=results/s2p_route_b_phase_d1_launch/logs/canary-%A_%a.err

set -euo pipefail

if [[ "${S2P_PM_D1_CANARY_AUTHORIZED:-0}" != "1" ]]; then
  echo "ERROR: D1 canary authorization is required" >&2
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

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-1}"
(( TASK_ID >= 0 && TASK_ID <= 1 )) || { echo "ERROR: task must be 0 or 1" >&2; exit 2; }
UNIQUE_H=(200 1000)
H="${UNIQUE_H[$TASK_ID]}"
TRAJECTORY="SS0_IS0_U${H}"
ROOT=results/s2p_route_b_phase_d1_launch/canary
OUT="$ROOT/$TRAJECTORY"
mkdir -p "$OUT" results/s2p_route_b_phase_d1_launch/logs
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_d1_canary_${SLURM_JOB_ID}_${TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$S2P_PYTHON" s2p/scripts/route_b_phase_d1_train.py \
  --repo-root "$REPO_ROOT" \
  --cbramod-root "$CBRAMOD_ROOT" \
  --trajectory-id "$TRAJECTORY" \
  --subset-seed 0 --init-seed 0 --unique-data-h "$H" \
  --output-dir "$OUT" \
  --low-update 2 --high-update 4 --validation-cadence 2 \
  --max-manifest-windows 256 --canary
