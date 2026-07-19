#!/bin/bash
cd /home/infres/yinwang/CMI_AAAI_readout_prior
echo "[chain] waiting for 91 epoch caches..."
for i in $(seq 1 960); do
  n=$(ls results/precache_done/*.done 2>/dev/null | wc -l)
  [ "$n" -ge 91 ] && { echo "[chain] epoch done $n/91"; break; }
  sleep 30
done
n=$(ls results/precache_done/*.done 2>/dev/null | wc -l)
[ "$n" -lt 91 ] && { echo "[chain] ABORT: epoch stalled at $n/91"; exit 1; }
echo "[chain] consolidating caches (once, avoids 28x race)..."
jid=$(sbatch --parsable --array=0-1 scripts/consolidate_arr.sbatch)
echo "[chain] consolidation array $jid"
for i in $(seq 1 240); do
  st=$(squeue -j "$jid" -h -r 2>/dev/null | wc -l)
  [ "$st" -eq 0 ] && break
  sleep 30
done
# verify both consolidated caches exist
sc=$(ls /home/infres/yinwang/cmi_epoch_cache/Stieger2021_*.npz 2>/dev/null | grep -v tmp | grep -v per_subject | wc -l)
hc=$(ls /home/infres/yinwang/cmi_epoch_cache/Shin2017A_*.npz 2>/dev/null | grep -v tmp | wc -l)
echo "[chain] consolidated caches: Stieger=$sc Shin=$hc"
if [ "$sc" -lt 1 ] || [ "$hc" -lt 1 ]; then echo "[chain] ABORT: consolidation failed (check logs/lb-consol-*)"; exit 1; fi
echo "[chain] launching GPU dump driver"
nohup bash scripts/gpu_dump_driver.sh > logs/gpu_dump_driver.log 2>&1 &
echo "[chain] GPU dump driver PID $! — chain complete"
