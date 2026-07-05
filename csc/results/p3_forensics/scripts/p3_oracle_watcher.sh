#!/usr/bin/env bash
# P3.0d oracle watcher: submit 12 shards (3 seed blocks x 4 conditions) throttled <= CAP, then wait to drain.
set -u
BASES=(50000000 60000000 70000000)
CONDS=(NULL_cov NULL_cov_plus_label POS_concept POS_concept_plus_cov)
OUTDIR=/home/infres/yinwang/realeeg_feas/p3_forensic/oracle
SB=/home/infres/yinwang/realeeg_feas/p3_oracle_shard.sbatch
LOG=/home/infres/yinwang/realeeg_feas/p3_oracle_watcher.log
CAP=28; POLL=60
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "oracle watcher start (12 shards, CAP=$CAP)"
for BASE in "${BASES[@]}"; do for COND in "${CONDS[@]}"; do
  OUT="$OUTDIR/oracle_${BASE}_${COND}.jsonl"
  [ -f "$OUT.prov.json" ] && { log "skip $BASE:$COND (exists)"; continue; }
  while :; do N=$(squeue -u "$USER" -h 2>/dev/null | wc -l); [ "$N" -lt "$CAP" ] && break; sleep "$POLL"; done
  R=$(BASE="$BASE" COND="$COND" sbatch --export=ALL,BASE="$BASE",COND="$COND",B=200 "$SB" 2>&1)
  log "submit $BASE:$COND -> $R"; sleep 3
done; done
log "all submitted; draining"
while :; do R=$(squeue -u "$USER" -h -n p3_oracle 2>/dev/null | wc -l); [ "$R" -eq 0 ] && break; sleep "$POLL"; done
log "oracle watcher done"; exit 0
