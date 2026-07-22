#!/bin/bash
cd /home/infres/yinwang/CMI_AAAI_readout_prior
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
OUT=results/cmi_trace_readout_prior_lockbox
echo "[chain3] waiting for 525 matrix cells..."
for i in $(seq 1 2880); do
  n=$(ls $OUT/cells/*.done 2>/dev/null | wc -l)
  [ "$n" -ge 525 ] && { echo "[chain3] matrix done $n/525"; break; }
  sleep 60
done
n=$(ls $OUT/cells/*.done 2>/dev/null | wc -l)
[ "$n" -lt 525 ] && { echo "[chain3] ABORT: matrix stalled at $n/525"; exit 1; }
echo "[chain3] running aggregator (hardened gate: refuses on any skip/solver-fail/degraded-lockbox)"
"$PY" scripts/aggregate_readout_prior_lockbox.py --from-dir "$OUT" --expect 525 > logs/aggregate.log 2>&1
rc=$?
echo "[chain3] aggregator exit=$rc (0=verdict written; 2/3/4/5=REFUSED)"
tail -20 logs/aggregate.log
