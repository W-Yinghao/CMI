#!/bin/bash
set -uo pipefail
cd /home/infres/yinwang/CMI_AAAI_readout_prior
OUT=results/cmi_trace_readout_prior_lockbox
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
CAP=28
N=$("$PY" -m scripts.run_readout_prior_decomposition --list-cells 2>/dev/null | tail -1 | tr -dc '0-9')
echo "[matrix-driver] N=$N cells"
while :; do
  done_n=$(ls $OUT/cells/*.done 2>/dev/null | wc -l)
  echo "[matrix-driver] done=$done_n/$N $(date -u +%H:%M:%S)"
  [ "$done_n" -ge "$N" ] && { echo "[matrix-driver] all $N done"; break; }
  running=" $(squeue -u yinwang -h -r -n lb-matrix -o '%K' 2>/dev/null | tr '\n' ' ') "
  missing=""
  for i in $(seq 0 $((N-1))); do
    ls $OUT/cells/cell_$(printf '%03d' $i)_*.done >/dev/null 2>&1 && continue
    case "$running" in *" $i "*) continue;; esac
    missing="$missing $i"
  done
  missing=$(echo "$missing" | tr ' ' '\n' | grep -v '^$')
  mine=$(squeue -u yinwang -h -r -n lb-matrix 2>/dev/null | wc -l); room=$((CAP - mine))
  if [ "$room" -ge 1 ] && [ -n "$missing" ]; then
    chunk=$(echo "$missing" | head -n "$room" | paste -sd, -)
    [ -n "$chunk" ] && sbatch --array="$chunk" scripts/rp_matrix_arr.sbatch >/dev/null 2>&1 && echo "[matrix-driver] submitted [$chunk]"
  fi
  sleep 45
done
