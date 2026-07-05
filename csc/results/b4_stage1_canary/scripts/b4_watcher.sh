#!/usr/bin/env bash
set -u
OUTDIR=/home/infres/yinwang/realeeg_feas/b4_stage1
SB=/home/infres/yinwang/realeeg_feas/b4_shard.sbatch
LOG=/home/infres/yinwang/realeeg_feas/b4_watcher.log
N=$(python3 -c "import json;print(len(json.load(open('$OUTDIR/b4_stage1_manifest.json'))['cohorts']))")
CH=26; CAP=28; POLL=60
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "b4 watcher start; N=$N chunk=$CH"
s=0
while [ "$s" -lt "$N" ]; do
  e=$((s+CH)); [ "$e" -gt "$N" ] && e=$N
  OUT="$OUTDIR/b4_shard_${s}_${e}.jsonl"
  if [ -f "$OUT.prov.json" ]; then log "skip $s:$e (exists)"; s=$e; continue; fi
  while :; do M=$(squeue -u "$USER" -h 2>/dev/null | wc -l); [ "$M" -lt "$CAP" ] && break; sleep "$POLL"; done
  R=$(SLICE="$s:$e" sbatch --export=ALL,SLICE="$s:$e" "$SB" 2>&1); log "submit $s:$e -> $R"; sleep 3
  s=$e
done
log "all submitted; draining"
while :; do R=$(squeue -u "$USER" -h -n b4_canary 2>/dev/null | wc -l); [ "$R" -eq 0 ] && break; sleep "$POLL"; done
log "b4 watcher done"; exit 0
