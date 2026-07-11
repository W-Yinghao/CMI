#!/usr/bin/env bash
#SBATCH --job-name=s2pB1core
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_s2p_faced
#SBATCH --output=results/s2p_route_b_representation_emergence_b1_core/logs/phase-b1-core-%j.out
#SBATCH --error=results/s2p_route_b_representation_emergence_b1_core/logs/phase-b1-core-%j.err

set -euo pipefail

if [[ "${S2P_PM_B1_CORE_AUTHORIZED:-0}" != "1" ]]; then
  echo "ERROR: Phase B1-Core requires explicit PM authorization" >&2
  exit 2
fi
if [[ -z "${S2P_EXPECTED_COMMIT:-}" ]]; then
  echo "ERROR: S2P_EXPECTED_COMMIT is required" >&2
  exit 2
fi
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

ACTUAL_COMMIT=$(git rev-parse HEAD)
if [[ "$ACTUAL_COMMIT" != "$S2P_EXPECTED_COMMIT" ]]; then
  echo "ERROR: B1-Core commit mismatch expected=$S2P_EXPECTED_COMMIT actual=$ACTUAL_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
  echo "ERROR: tracked worktree changes detected at B1-Core launch" >&2
  git status --short --untracked-files=no >&2
  exit 2
fi

mkdir -p results/s2p_route_b_representation_emergence_b1_core/logs
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/s2p_phase_b1_core_${SLURM_JOB_ID:-local}"
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "$GPU_NAME" != *A100* ]]; then
  echo "ERROR: exact frozen-source replay requires A100; found $GPU_NAME" >&2
  exit 2
fi

echo "Phase B1-Core host=$(hostname) gpu=$GPU_NAME git=$ACTUAL_COMMIT"
srun "$PY" s2p/scripts/route_b_phase_b1_forensic_core_replay.py --repo-root "$REPO_ROOT" --self-test
srun "$PY" s2p/scripts/route_b_phase_b1_core_verify.py --self-test
srun "$PY" s2p/scripts/route_b_phase_b1_forensic_core_replay.py \
  --repo-root "$REPO_ROOT" --pm-authorized --device cuda:0
srun "$PY" s2p/scripts/route_b_phase_b1_core_verify.py

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("results/s2p_route_b_representation_emergence_b1_core")
disposition = json.loads((root / "phase_b1_variance_family_disposition.json").read_text())
core = json.loads((root / "phase_b1_core_verdict.json").read_text())
verification = json.loads((root / "phase_b1_core_adversarial_verification.json").read_text())
if disposition.get("status") != "FAILED_STABILITY_NOT_INTERPRETABLE":
    raise SystemExit("variance family disposition changed unexpectedly")
if core.get("status") != "PASS" or verification.get("status") != "PASS":
    raise SystemExit("B1-Core or independent verification did not pass")
if core.get("variance_used_in_mechanism_verdict") is not False:
    raise SystemExit("variance leaked into B1-Core verdict")
if core.get("phase_b2_authorized") is not False:
    raise SystemExit("Phase B2 must remain held")
print("Phase B1 variance forensic disposition and B1-Core verification: PASS")
PY
