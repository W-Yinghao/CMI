#!/bin/bash
set -uo pipefail
cd /home/infres/yinwang/CMI_AAAI_readout_prior
CELLS=/home/infres/yinwang/CMI_AAAI_readout_prior/results/gpu_dump_cells.txt
CAP=28; N=$(wc -l < $CELLS)
while :; do
  done_n=$(ls results/gpu_dump_done/*.done 2>/dev/null | wc -l)
  echo "[dump-driver] done=$done_n/$N $(date -u +%H:%M:%S)"
  [ "$done_n" -ge "$N" ] && { echo "[dump-driver] all $N done"; break; }
  running=" $(squeue -u yinwang -h -r -n lb-dump -o '%K' 2>/dev/null | tr '\n' ' ') "
  missing=""
  for i in $(seq 0 $((N-1))); do
    LINE=$(sed -n "$((i+1))p" $CELLS); read DS S SEED <<< "$LINE"
    [ -f results/gpu_dump_done/${DS}_${S}_${SEED}.done ] && continue
    case "$running" in *" $i "*) continue;; esac
    missing="$missing $i"
  done
  missing=$(echo "$missing" | tr ' ' '\n' | grep -v '^$')
  mine=$(squeue -u yinwang -h -r -n lb-dump 2>/dev/null | wc -l); room=$((CAP - mine))
  if [ "$room" -ge 1 ] && [ -n "$missing" ]; then
    chunk=$(echo "$missing" | head -n "$room" | paste -sd, -)
    [ -n "$chunk" ] && sbatch --array="$chunk" scripts/gpu_dump_arr.sbatch >/dev/null 2>&1 && echo "[dump-driver] submitted [$chunk]"
  fi
  sleep 60
done
