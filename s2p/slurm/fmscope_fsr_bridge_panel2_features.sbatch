#!/usr/bin/env bash
#SBATCH --job-name=s2pP2feat
#SBATCH --partition=A100,H100,A40,L40S,V100,V100-32GB,V100-16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-9
#SBATCH --time=12:00:00
#SBATCH --output=results/s2p_fmscope_fsr_bridge_panel2/logs/features-%A_%a.out
#SBATCH --error=results/s2p_fmscope_fsr_bridge_panel2/logs/features-%A_%a.err

set -euo pipefail

if [[ "${S2P_PM_PANEL2_AUTHORIZED:-0}" != "1" ]]; then
  echo "ERROR: Panel 2 authorization is required" >&2
  exit 2
fi
for required in S2P_REPO_ROOT S2P_EXPECTED_COMMIT S2P_PYTHON CBRAMOD_ROOT FACED_ROOT; do
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
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-9}"
(( TASK_ID >= 0 && TASK_ID < 10 )) || { echo "ERROR: task must be 0-9" >&2; exit 2; }
TAG="${TAGS[$TASK_ID]}"
ROOT=results/s2p_fmscope_fsr_bridge_panel2
mkdir -p "$ROOT/features/faced" "$ROOT/logs"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_panel2_features_${SLURM_JOB_ID}_${TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$S2P_PYTHON" s2p/scripts/fmscope_fsr_bridge_panel2_extract_faced.py \
  --tag "$TAG" \
  --checkpoint-manifest results/s2p_route_b_phase_b_checkpoint_closure/phase_b_checkpoint_immutable_manifest.csv \
  --cbramod-root "$CBRAMOD_ROOT" \
  --faced-lmdb "$FACED_ROOT" \
  --out-dir "$ROOT/features/faced" \
  --device cuda:0 --batch-size 64
