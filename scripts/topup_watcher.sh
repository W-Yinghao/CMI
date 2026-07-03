#!/bin/bash
# Auto-top-up: after all pilot jobs drain (A100 24h wall), resubmit ONLY missing folds per
# (dataset,backbone,seed) to V100 (48h). Loops until complete or MAXR rounds. Banked folds are safe
# (each fold writes its own npz), so this only fills gaps. Re-invokes the agent on exit.
cd /home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
MAXR=4
for round in $(seq 1 $MAXR); do
  until [ "$(squeue -u $USER -h -o '%j' 2>/dev/null | grep -c tos_eeg_pilot)" -eq 0 ]; do sleep 600; done
  MISS=$($PY - <<'PYEOF'
import json, glob, re
subs=json.load(open("artifact_build/subject_lists.json"))
R="tos_cmi/results/tos_cmi_eeg_frozen"
for D in ["Lee2019_MI","Cho2017","Schirrmeister2017"]:
    for BB in ["TSMNet","EEGNet"]:
        for S in [0,1,2]:
            d="%s/%s_%s_LOSO"%(R,D,BB)
            have={int(re.search(r'sub(\d+)_',p.split('/')[-1]).group(1)) for p in glob.glob("%s/sub*_erm_lam0_seed%d.npz"%(d,S))}
            miss=[s for s in subs[D] if s not in have]
            if miss: print("%s %s %d %s"%(D,BB,S," ".join(map(str,miss))))
PYEOF
)
  if [ -z "$MISS" ]; then echo "ALL_DUMPS_COMPLETE (round $round)"; exit 0; fi
  echo "=== round $round top-up (missing folds) ==="; echo "$MISS"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    D=$(echo "$line" | awk '{print $1}'); BB=$(echo "$line" | awk '{print $2}'); S=$(echo "$line" | awk '{print $3}'); M=$(echo "$line" | cut -d' ' -f4-)
    sbatch --partition=V100,V100-32GB,V100-16GB scripts/tos_eeg_frozen_pilot.sbatch \
      --dataset "$D" --backbone "$BB" --target-subjects $M --seed "$S" --configs erm:0 --epochs 300 --device cuda >/dev/null
    echo "  topped up $D $BB seed$S ($(echo $M | wc -w) folds)"
  done <<< "$MISS"
  sleep 60
done
echo "TOPUP_EXHAUSTED after $MAXR rounds (persistent missing folds -- investigate)"
