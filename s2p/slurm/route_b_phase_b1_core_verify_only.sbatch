#!/usr/bin/env bash
#SBATCH --job-name=s2pB1verify
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_s2p_faced
#SBATCH --output=results/s2p_route_b_representation_emergence_b1_core/logs/phase-b1-core-verify-%j.out
#SBATCH --error=results/s2p_route_b_representation_emergence_b1_core/logs/phase-b1-core-verify-%j.err

set -euo pipefail

if [[ "${S2P_PM_B1_CORE_AUTHORIZED:-0}" != "1" ]]; then
  echo "ERROR: Phase B1-Core verification requires explicit PM authorization" >&2
  exit 2
fi
if [[ -z "${S2P_EXPECTED_COMMIT:-}" || -z "${S2P_REPO_ROOT:-}" ]]; then
  echo "ERROR: S2P_EXPECTED_COMMIT and S2P_REPO_ROOT are required" >&2
  exit 2
fi
REPO_ROOT=$(readlink -f "$S2P_REPO_ROOT")
if [[ ! -d "$REPO_ROOT/s2p" || "$REPO_ROOT" == /var/spool/* ]]; then
  echo "ERROR: invalid REPO_ROOT: $REPO_ROOT" >&2
  exit 2
fi
cd "$REPO_ROOT"
ACTUAL_COMMIT=$(git rev-parse HEAD)
if [[ "$ACTUAL_COMMIT" != "$S2P_EXPECTED_COMMIT" ]]; then
  echo "ERROR: verifier commit mismatch expected=$S2P_EXPECTED_COMMIT actual=$ACTUAL_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
  echo "ERROR: tracked worktree changes detected" >&2
  exit 2
fi

source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_phase_b1_core_verify_${SLURM_JOB_ID:-local}"
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python

echo "Phase B1-Core verifier-only host=$(hostname) git=$ACTUAL_COMMIT"
srun "$PY" s2p/scripts/route_b_phase_b1_core_verify.py --self-test
srun "$PY" s2p/scripts/route_b_phase_b1_core_verify.py

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("results/s2p_route_b_representation_emergence_b1_core")
core = json.loads((root / "phase_b1_core_verdict.json").read_text())
verification = json.loads((root / "phase_b1_core_adversarial_verification.json").read_text())
if core.get("status") != "PASS" or core.get("independent_verification_status") != "PASS":
    raise SystemExit("B1-Core final verdict is not independently verified")
if verification.get("status") != "PASS":
    raise SystemExit("B1-Core independent verifier did not pass")
if core.get("phase_b2_authorized") is not False:
    raise SystemExit("Phase B2 must remain held")
print("Phase B1-Core verifier-only: PASS")
PY
