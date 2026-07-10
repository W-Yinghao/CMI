#!/usr/bin/env bash
#SBATCH --job-name=s2pFACED2k
#SBATCH --partition=A100,V100,H100,L40S,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_s2p_faced
#SBATCH --output=results/s2p_route_b_33ch_b1_faced/logs/faced-h2000-%j.out
#SBATCH --error=results/s2p_route_b_33ch_b1_faced/logs/faced-h2000-%j.err

set -euo pipefail

if [[ -n "${S2P_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${S2P_REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/s2p" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  echo "ERROR: cannot resolve REPO_ROOT. Set S2P_REPO_ROOT or submit from repo root." >&2
  exit 2
fi

REPO_ROOT=$(readlink -f "$REPO_ROOT")
if [[ ! -d "$REPO_ROOT/s2p" || ! -d "$REPO_ROOT/results" ]]; then
  echo "ERROR: REPO_ROOT does not look like S2P repo: $REPO_ROOT" >&2
  exit 2
fi
if [[ "$REPO_ROOT" == /var/spool/* ]]; then
  echo "ERROR: REPO_ROOT resolved to SLURM spool path: $REPO_ROOT" >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p results/s2p_route_b_33ch_b1_faced/logs
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python

OUT_DIR="${FACED_OUT_DIR:-results/s2p_route_b_33ch_b1_faced}"
CKPT_ROOT="${FACED_CKPT_ROOT:-/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1}"
BATCH_SIZE="${FACED_BATCH_SIZE:-48}"

echo "Route B FACED H2000 downstream update host=$(hostname) git=$(git rev-parse HEAD)"
git status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" s2p/scripts/route_b_faced_h2000_update.py \
  --ckpt-root "$CKPT_ROOT" \
  --out-dir "$OUT_DIR" \
  --device cuda:0 \
  --batch-size "$BATCH_SIZE"
