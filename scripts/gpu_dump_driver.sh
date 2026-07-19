#!/bin/bash
set -uo pipefail
cd /home/infres/yinwang/CMI_AAAI_readout_prior
CELLS=/home/infres/yinwang/CMI_AAAI_readout_prior/results/gpu_dump_cells.txt
CAP=20; N=$(wc -l < $CELLS)
while :; do
  done_n=$(ls results/gpu_dump_done/*.done 2>/dev/null | wc -l)
  echo "[dump-driver] done=$done_n/$N $(date -u +%H:%M:%S)"
  [ "$done_n" -ge "$N" ] && { echo "[dump-driver] all $N done"; break; }
  running=" $(squeue -u yinwang -h -r -n lb-dump -o '%K' 2>/dev/null | tr '\n' ' ') "
  mine=$(squeue -u yinwang -h -r -n lb-dump 2>/dev/null | wc -l); room=$((CAP - mine))
  if [ "$room" -ge 1 ]; then
    subbed=0
    for i in $(seq 0 $((N-1))); do
      [ "$subbed" -ge "$room" ] && break
      LINE=$(sed -n "$((i+1))p" $CELLS); read DS S SEED <<< "$LINE"
      [ -f results/gpu_dump_done/${DS}_${S}_${SEED}.done ] && continue
      case "$running" in *" $i "*) continue;; esac
      # submit ONE cell; break on failure (submit-quota reached) so a tight quota fills exactly, not atomic-fails
      if sbatch --array="$i" scripts/gpu_dump_arr.sbatch >/dev/null 2>/tmp/lbdump_sberr.$$; then
        subbed=$((subbed+1)); running="$running $i "
      else
        echo "[dump-driver] sbatch stop at cell $i ($(tr -d '\n' </tmp/lbdump_sberr.$$ | tail -c 80))"; break
      fi
    done
    [ "$subbed" -gt 0 ] && echo "[dump-driver] submitted $subbed cells"
  fi
  sleep 60
done
