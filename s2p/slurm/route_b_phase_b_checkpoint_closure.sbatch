#!/usr/bin/env bash
#SBATCH --job-name=s2pBprov
#SBATCH --partition=A100,V100,H100,L40S,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_s2p_faced
#SBATCH --output=results/s2p_route_b_phase_b_checkpoint_closure/logs/phase-b-provenance-%j.out
#SBATCH --error=results/s2p_route_b_phase_b_checkpoint_closure/logs/phase-b-provenance-%j.err

set -euo pipefail

if [[ -n "${S2P_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${S2P_REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/s2p" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  echo "ERROR: cannot resolve REPO_ROOT" >&2
  exit 2
fi

REPO_ROOT=$(readlink -f "$REPO_ROOT")
if [[ ! -d "$REPO_ROOT/s2p" || "$REPO_ROOT" == /var/spool/* ]]; then
  echo "ERROR: invalid REPO_ROOT: $REPO_ROOT" >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p results/s2p_route_b_phase_b_checkpoint_closure/logs
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_phase_b_provenance_${SLURM_JOB_ID:-local}"
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python

echo "Phase B provenance closure host=$(hostname) git=$(git rev-parse HEAD)"
git status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" s2p/scripts/route_b_phase_b_checkpoint_closure.py --device cuda:0

srun "$PY" s2p/scripts/route_b_phase_b0_precheck.py --post-closure

echo "Phase B provenance closure and B0 rerun complete"
