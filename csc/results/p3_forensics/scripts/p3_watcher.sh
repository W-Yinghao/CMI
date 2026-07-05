#!/usr/bin/env bash
# P3.0b/c slot-watcher: submit the 18 forensic shards (3 seed blocks x 6 conditions) THROTTLED so my total
# queued+running jobs stay <= CAP (QOS 30-job cap). Idempotent: never resubmits a shard whose output already
# exists or whose job name is already queued for that shard. After all 18 are submitted, wait for all
# p3_forensic jobs to finish, then exit (re-invokes the agent for merge).
set -u
BASES=(50000000 60000000 70000000)
CONDS=(NULL_cov NULL_label NULL_cov_plus_label random_label_control POS_concept POS_concept_plus_cov)
OUTDIR=/home/infres/yinwang/realeeg_feas/p3_forensic
SB=/home/infres/yinwang/realeeg_feas/p3_shard.sbatch
LOG=/home/infres/yinwang/realeeg_feas/p3_watcher.log
CAP=28          # keep total <= 28 (leaves margin under the 30 QOS cap)
POLL=60
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "watcher start; submitting 18 shards throttled to CAP=$CAP"
for BASE in "${BASES[@]}"; do
  for COND in "${CONDS[@]}"; do
    OUT="$OUTDIR/shard_${BASE}_${COND}.jsonl"
    # skip if a complete shard output already exists (idempotent restart)
    if [ -f "$OUT.prov.json" ]; then log "skip $BASE:$COND (output exists)"; continue; fi
    # wait for a slot
    while :; do
      N=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
      [ "$N" -lt "$CAP" ] && break
      sleep "$POLL"
    done
    OUTP=$(BASE="$BASE" COND="$COND" sbatch --export=ALL,BASE="$BASE",COND="$COND" "$SB" 2>&1)
    log "submit $BASE:$COND -> $OUTP"
    sleep 3
  done
done
log "all 18 shards submitted; waiting for p3_forensic jobs to drain"
i=0
while :; do
  i=$((i+1))
  R=$(squeue -u "$USER" -h -n p3_forensic 2>/dev/null | wc -l)
  [ "$R" -eq 0 ] && { log "all p3_forensic jobs finished"; break; }
  [ $((i % 10)) -eq 0 ] && log "still running: $R p3_forensic jobs"
  sleep "$POLL"
done
log "watcher done"
exit 0
