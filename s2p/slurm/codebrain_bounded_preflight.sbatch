#!/usr/bin/env bash
#SBATCH --job-name=s2p_cb_gate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/s2p_codebrain_bounded_preflight/slurm-%j.out
#SBATCH --error=results/s2p_codebrain_bounded_preflight/slurm-%j.err

# Feasibility-only entry. It must never call Stage-2 training or downstream fine-tuning.
set -euo pipefail

if [[ -n "${S2P_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${S2P_REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/s2p" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  echo "ERROR: set S2P_REPO_ROOT or submit from the S2P repository root" >&2
  exit 2
fi
REPO_ROOT="$(readlink -f "${REPO_ROOT}")"
if [[ ! -d "${REPO_ROOT}/s2p/scripts" ]]; then
  echo "ERROR: invalid repository root: ${REPO_ROOT}" >&2
  exit 2
fi
if [[ -z "${S2P_EXPECTED_COMMIT:-}" ]]; then
  echo "ERROR: S2P_EXPECTED_COMMIT is required" >&2
  exit 2
fi
ACTUAL_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${S2P_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: commit mismatch actual=${ACTUAL_COMMIT} expected=${S2P_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --short --untracked-files=no)" ]]; then
  echo "ERROR: tracked worktree changes exist" >&2
  git -C "${REPO_ROOT}" status --short --untracked-files=no >&2
  exit 2
fi

cd "${REPO_ROOT}"
OUT="results/s2p_codebrain_bounded_preflight"
mkdir -p "${OUT}"

required_paths=(
  S2P_PYTHON
  CODEBRAIN_ROOT
  CODEBRAIN_TOKENIZER_PATH
  CODEBRAIN_RELEASED_PATH
  FACED_ROOT
  SEEDV_ROOT
  ISRUC_FLAT_ROOT
  ISRUC_PROCESSED_ROOT
)
for name in "${required_paths[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable is unset: ${name}" >&2
    exit 2
  fi
done

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}/s2p/scripts:${PYTHONPATH:-}"
PY="${S2P_PYTHON}"

echo "phase=S2P_CodeBrain_Bounded_preflight_only"
echo "git_head=${ACTUAL_COMMIT}"
echo "start_time=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun "${PY}" s2p/scripts/codebrain_bounded_preflight.py --out-dir "${OUT}"
srun "${PY}" s2p/scripts/codebrain_tokenizer_target_gate.py \
  --out-dir "${OUT}" --batch-size 16 --num-workers 4 --samples-per-stratum 256

echo "end_time=$(date -Is)"
cat "${OUT}/codebrain_bounded_go_nogo.json"
