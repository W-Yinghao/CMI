#!/bin/bash
cd /home/infres/yinwang/CMI_AAAI_readout_prior
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "[chain2] waiting for 273 GPU dumps..."
for i in $(seq 1 2880); do
  n=$(ls results/gpu_dump_done/*.done 2>/dev/null | wc -l)
  [ "$n" -ge 273 ] && { echo "[chain2] dumps done $n/273"; break; }
  sleep 60
done
n=$(ls results/gpu_dump_done/*.done 2>/dev/null | wc -l)
[ "$n" -lt 273 ] && { echo "[chain2] ABORT: dumps stalled at $n/273"; exit 1; }
ncells=$("$PY" -m scripts.run_readout_prior_decomposition --list-cells 2>/dev/null | tail -1 | tr -dc '0-9')
echo "[chain2] enumerate_cells = $ncells (expect 525)"
if [ "$ncells" != "525" ]; then echo "[chain2] ABORT: cell count $ncells != 525 (dump gaps?)"; exit 1; fi
echo "[chain2] launching CPU matrix driver"
nohup bash scripts/rp_matrix_driver.sh > logs/rp_matrix_driver.log 2>&1 &
echo "[chain2] matrix driver PID $! — chain2 complete"
