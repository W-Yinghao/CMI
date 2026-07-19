#!/bin/bash
set -uo pipefail
cd /home/infres/yinwang/CMI_AAAI_readout_prior
CELLS=/home/infres/yinwang/CMI_AAAI_readout_prior/results/precache_cells.txt
CAP=28; N=$(wc -l < $CELLS)
while :; do
  done_n=$(ls results/precache_done/*.done 2>/dev/null | wc -l)
  echo "[epoch-driver] done=$done_n/$N $(date -u +%H:%M:%S)"
  [ "$done_n" -ge "$N" ] && { echo "[epoch-driver] all $N done"; break; }
  running=" $(squeue -u yinwang -h -r -n lb-epoch -o '%K' 2>/dev/null | tr '\n' ' ') "
  missing=""
  for i in $(seq 0 $((N-1))); do
    LINE=$(sed -n "$((i+1))p" $CELLS); read DS S <<< "$LINE"
    [ -f results/precache_done/${DS}_${S}.done ] && continue
    case "$running" in *" $i "*) continue;; esac
    missing="$missing $i"
  done
  missing=$(echo "$missing" | tr ' ' '\n' | grep -v '^$')
  mine=$(squeue -u yinwang -h -r -n lb-epoch 2>/dev/null | wc -l); room=$((CAP - mine))
  if [ "$room" -ge 1 ] && [ -n "$missing" ]; then
    chunk=$(echo "$missing" | head -n "$room" | paste -sd, -)
    [ -n "$chunk" ] && sbatch --array="$chunk" scripts/precache_arr.sbatch >/dev/null 2>&1 && echo "[epoch-driver] submitted [$chunk]"
  fi
  sleep 30
done
