#!/bin/bash
# B5.0 canary lifecycle: slot-watcher trickle-submit of the 12 shards (6 conds x 2 halves) under the QOS cap,
# then wait for completion, then merge. Runs in background. Respects the shared 30-job submit cap (CAP=28 margin).
set -uo pipefail
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_TASK_PID 2>/dev/null || true
cd /home/infres/yinwang/realeeg_feas
CAP=28
SB=b5_canary.sbatch
echo "[b5_run] start; trickle-submitting 12 shards under CAP=$CAP"
for IDX in $(seq 0 11); do
  until [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -lt "$CAP" ]; do sleep 30; done
  JID=$(sbatch --parsable --array="$IDX" "$SB" 2>&1) || { echo "[b5_run] submit IDX=$IDX FAILED: $JID"; sleep 45; JID=$(sbatch --parsable --array="$IDX" "$SB" 2>&1) || { echo "[b5_run] retry FAILED: $JID"; }; }
  echo "[b5_run] submitted IDX=$IDX -> $JID (queue=$(squeue -u "$USER" -h 2>/dev/null | wc -l))"
  sleep 8
done
echo "[b5_run] all 12 submitted; waiting for b5_canary jobs to drain"
until ! squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q b5_canary; do sleep 45; done
echo "[b5_run] all b5_canary jobs finished; running merge"
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
python -u b5_canary_merge.py
echo "[b5_run] merge rc=$? DONE"
