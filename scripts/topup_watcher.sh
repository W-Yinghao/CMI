#!/bin/bash
# Auto-top-up (group/array style): after all pilot+array jobs drain, resubmit ONLY combos with missing
# folds as a job ARRAY (max 15 tasks/group, %4 concurrent) on V100. Idempotent (--target-subjects all +
# skip-existing) so nothing recomputes. Loops until complete or MAXR rounds. Re-invokes agent on exit.
cd /home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
MAXR=8
for round in $(seq 1 $MAXR); do
  until [ "$(squeue -u $USER -h -o '%j' 2>/dev/null | grep -cE 'tos_eeg_pilot|tos_arr')" -eq 0 ]; do sleep 600; done
  $PY - <<'PYEOF' > artifact_build/topup_manifest.txt
import json, glob, re
subs=json.load(open("artifact_build/subject_lists.json"))
R="tos_cmi/results/tos_cmi_eeg_frozen"
for D in ["Lee2019_MI","Cho2017","Schirrmeister2017"]:
    for BB in ["TSMNet","EEGNet"]:
        for S in [0,1,2]:
            d="%s/%s_%s_LOSO"%(R,D,BB)
            have={int(re.search(r'sub(\d+)_',p.split('/')[-1]).group(1)) for p in glob.glob("%s/sub*_erm_lam0_seed%d.npz"%(d,S))}
            if [s for s in subs[D] if s not in have]: print("%s %s %d"%(D,BB,S))
PYEOF
  # 3-channel TSMNet-2b would be degenerate, but 2b is not in the big list; nothing to exclude here.
  head -15 artifact_build/topup_manifest.txt > artifact_build/topup_round.txt   # <=15 tasks/group
  K=$(wc -l < artifact_build/topup_round.txt)
  if [ "$K" -eq 0 ]; then echo "ALL_DUMPS_COMPLETE (round $round)"; exit 0; fi
  echo "=== round $round: $K combos need top-up (array %4, V100) ==="; cat artifact_build/topup_round.txt
  sbatch --array=0-$((K-1))%4 scripts/tos_eeg_array_pilot.sbatch artifact_build/topup_round.txt
  sleep 120
done
echo "TOPUP_EXHAUSTED after $MAXR rounds (persistent missing folds -- investigate)"
