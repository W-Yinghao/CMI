#!/bin/bash
# 525-cell matrix driver: 1-cell-at-a-time submit, break on submit-quota, CAP=25.
# (matches the fixed gpu_dump_driver: an atomic-chunk array submit fails wholesale when the
#  shared 30-job per-user submit cap is contended by other-session jobs -> driver stalls.)
set -uo pipefail
cd /home/infres/yinwang/CMI_AAAI_readout_prior
OUT=results/cmi_trace_readout_prior_lockbox
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
CAP=25
N=$("$PY" -m scripts.run_readout_prior_decomposition --list-cells 2>/dev/null | tail -1 | tr -dc '0-9')
echo "[matrix-driver] N=$N cells (1-at-a-time submit)"
while :; do
  done_n=$(ls $OUT/cells/*.done 2>/dev/null | wc -l)
  echo "[matrix-driver] done=$done_n/$N $(date -u +%H:%M:%S)"
  [ "$done_n" -ge "$N" ] && { echo "[matrix-driver] all $N done"; break; }
  running=" $(squeue -u yinwang -h -r -n lb-matrix -o '%K' 2>/dev/null | tr '\n' ' ') "
  mine=$(squeue -u yinwang -h -r -n lb-matrix 2>/dev/null | wc -l); room=$((CAP - mine))
  if [ "$room" -ge 1 ]; then
    subbed=0
    for i in $(seq 0 $((N-1))); do
      [ "$subbed" -ge "$room" ] && break
      ls $OUT/cells/cell_$(printf '%03d' $i)_*.done >/dev/null 2>&1 && continue
      case "$running" in *" $i "*) continue;; esac
      if sbatch --array="$i" scripts/rp_matrix_arr.sbatch >/dev/null 2>/tmp/lbmx_sberr.$$; then
        subbed=$((subbed+1)); running="$running $i "
      else
        echo "[matrix-driver] sbatch stop at cell $i ($(tr -d '\n' </tmp/lbmx_sberr.$$ | tail -c 60))"; break
      fi
    done
    [ "$subbed" -gt 0 ] && echo "[matrix-driver] submitted $subbed cells"
  fi
  sleep 45
done
