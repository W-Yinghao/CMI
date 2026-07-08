#!/usr/bin/env bash
#SBATCH --job-name=s2pB1
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-7
#SBATCH --time=96:00:00
#SBATCH --chdir=/home/infres/yinwang/CMI_AAAI_s2p_b1_launch
#SBATCH --output=results/s2p_route_b_33ch_b1/logs/train-%A_%a.out
#SBATCH --error=results/s2p_route_b_33ch_b1/logs/train-%A_%a.err

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
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025
export PYTHONUNBUFFERED=1
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
CONTRACT="$REPO_ROOT/results/s2p_route_b_33ch_contract"
ROOT="$REPO_ROOT/results/s2p_route_b_33ch_b1"
TASKS="$CONTRACT/route_b_b1_training_tasks.csv"
mkdir -p "$ROOT/logs"

ROW=$("$PY" - "$TASKS" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv, sys
tasks = list(csv.DictReader(open(sys.argv[1])))
r = tasks[int(sys.argv[2])]
print(",".join(str(r[k]) for k in ["cell", "budget_h", "seed"]))
PY
)
IFS=, read -r CELL H SEED <<< "$ROW"
OUT="$ROOT/$CELL"
mkdir -p "$OUT"
echo "Route B B1 cell=$CELL H=$H seed=$SEED host=$(hostname) git=$(git rev-parse HEAD)"
git status --short

if [[ "${S2P_DRY_RUN:-0}" == "1" ]]; then
  CANARY_REPORT="$ROOT/route_b_b1_path_canary_report.csv"
  CHECKPOINT_DIR="$OUT"
  PATH_OK=true
  if [[ "$REPO_ROOT" == /var/spool/* || "$ROOT" != "$REPO_ROOT"/results/* || "$OUT" != "$ROOT"/* ]]; then
    PATH_OK=false
  fi
  "$PY" - "$CANARY_REPORT" "$SLURM_ARRAY_TASK_ID" "$CELL" "$REPO_ROOT" "$PWD" "$ROOT" "$ROOT/logs" "$CHECKPOINT_DIR" "$H" "$SEED" "$PATH_OK" <<'PY'
import csv
import fcntl
import os
import sys

report, task_id, run_id, repo_root, cwd, results_dir, log_dir, checkpoint_dir, h, seed, path_ok = sys.argv[1:]
row = {
    "array_task_id": task_id,
    "run_id": run_id,
    "repo_root": repo_root,
    "cwd": cwd,
    "results_dir": results_dir,
    "log_dir": log_dir,
    "checkpoint_dir": checkpoint_dir,
    "resolved_H": h,
    "resolved_seed": seed,
    "dry_run": "true",
    "path_ok": str(path_ok).lower(),
    "trainer_called": "false",
}
os.makedirs(os.path.dirname(report), exist_ok=True)
fields = list(row)
with open(report + ".lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    exists = os.path.exists(report) and os.path.getsize(report) > 0
    with open(report, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
PY
  if [[ "$PATH_OK" != "true" ]]; then
    echo "ERROR: dry-run path canary failed for $CELL" >&2
    exit 2
  fi
  echo "S2P_DRY_RUN=1 path canary passed for $CELL; trainer_called=false"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" s2p/scripts/route_b_train_cbramod.py \
  --budget-h "$H" --subset-seed "$SEED" --init-seed "$SEED" \
  --contract-dir "$CONTRACT" --out-dir "$OUT" \
  --epochs 50 --batch-size 64 --lr 5e-4 --weight-decay 5e-2 \
  --mask-ratio 0.5 --clip-value 1.0
