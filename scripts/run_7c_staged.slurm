#!/bin/bash
#SBATCH --job-name=fsr-7c
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# FSR Phase 7C staged fail-closed run (CPU, process-parallel over 8 confirm seeds).
#   Q7C-a held-in learnability gate FIRST; ONLY if it passes -> Q7C-b transfer + target harm + repair.
#   sbatch scripts/run_7c_staged.slurm
set -euo pipefail
cd /home/infres/yinwang/CMI_AAAI_rq4
mkdir -p logs
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/fsr_head_only_label_conflict
SEEDS="20260721 20260722 20260723 20260724 20260725 20260726 20260727 20260728"

echo "=== 7C Q7C-a GATE (process-parallel over seeds) ==="
rm -rf $OUT/shard_gate_* ; pids=""
for s in $SEEDS; do
  "$PY" scripts/run_head_only_label_conflict.py --stage gate --seeds $s --outdir $OUT/shard_gate_$s \
        > logs/7c-gate-$s.log 2>&1 &
  pids="$pids $!"
done
for p in $pids; do wait $p; done
"$PY" scripts/merge_7c_shards.py shard_gate_
"$PY" scripts/aggregate_head_conflict_gate.py

PROCEED=$("$PY" -c "import json;print(json.load(open('$OUT/label_conflict_verdict.json'))['heldin_learnability_pass'])")
echo "=== Q7C-a heldin_learnability_pass = $PROCEED ==="
if [ "$PROCEED" != "True" ]; then
  echo "FAIL-CLOSED: Q7C-a did not pass; 7C-full (Q7C-b transfer + target harm + repair) does NOT run."
  exit 0
fi

echo "=== 7C-full: Q7C-b TRANSFER + target harm + repair (process-parallel over seeds) ==="
rm -rf $OUT/shard_full_* ; pids=""
for s in $SEEDS; do
  "$PY" scripts/run_head_only_label_conflict.py --stage full --seeds $s --outdir $OUT/shard_full_$s \
        > logs/7c-full-$s.log 2>&1 &
  pids="$pids $!"
done
for p in $pids; do wait $p; done
"$PY" scripts/merge_7c_shards.py shard_full_
"$PY" scripts/aggregate_head_conflict_transfer.py
echo "=== 7C staged run complete ==="
