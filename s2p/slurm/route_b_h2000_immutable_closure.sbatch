#!/usr/bin/env bash
#SBATCH --job-name=s2pH2kClose
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_s2p_faced
#SBATCH --output=results/s2p_route_b_h2000_immutable_closure/logs/closure-%j.out
#SBATCH --error=results/s2p_route_b_h2000_immutable_closure/logs/closure-%j.err

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
if [[ ! -d "$REPO_ROOT/s2p" || "$REPO_ROOT" == /var/spool/* ]]; then
  echo "ERROR: invalid REPO_ROOT: $REPO_ROOT" >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p results/s2p_route_b_h2000_immutable_closure/logs
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_h2000_closure_${SLURM_JOB_ID:-local}"
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python

SOURCE_ROOT="${H2000_SOURCE_ROOT:-/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1}"
IMMUTABLE_ROOT="${H2000_IMMUTABLE_ROOT:-/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1_immutable}"
MANIFEST_DIR="${H2000_CLOSURE_MANIFEST_DIR:-results/s2p_route_b_h2000_immutable_closure}"

echo "H2000 immutable closure host=$(hostname) git=$(git rev-parse HEAD) dependency=${SLURM_JOB_DEPENDENCY:-unset}"
git status --short

srun "$PY" s2p/scripts/route_b_h2000_immutable_closure.py \
  --source-root "$SOURCE_ROOT" \
  --immutable-root "$IMMUTABLE_ROOT" \
  --manifest-dir "$MANIFEST_DIR" \
  --require-job-ids 890151_6 890151_7
