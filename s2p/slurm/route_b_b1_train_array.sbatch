#!/bin/bash
#SBATCH --job-name=s2pB1
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-7
#SBATCH --time=96:00:00
#SBATCH --output=results/s2p_route_b_33ch_b1/logs/train-%A_%a.out
#SBATCH --error=results/s2p_route_b_33ch_b1/logs/train-%A_%a.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" s2p/scripts/route_b_train_cbramod.py \
  --budget-h "$H" --subset-seed "$SEED" --init-seed "$SEED" \
  --contract-dir "$CONTRACT" --out-dir "$OUT" \
  --epochs 50 --batch-size 64 --lr 5e-4 --weight-decay 5e-2 \
  --mask-ratio 0.5 --clip-value 1.0
