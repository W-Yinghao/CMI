#!/bin/bash
#SBATCH --job-name=v2tsmrobust
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export EEG_DATALAKE_RAW=/projects/EEG-foundation-model/datalake/raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
OUT=tos_cmi/results/method_deepen/v2_tsm_robust
for FRAC in 0.15 0.20 0.25 0.30; do
  echo "===== nuisance_fraction=$FRAC (World A, TSMNet, Lee+Cho seed0 first5) ====="
  $PY -u -m tos_cmi.eeg.run_v2_certificate --worlds A --datasets Lee2019_MI Cho2017 --backbones TSMNet \
      --seeds 0 --n-source all --alphas 0.25 0.5 1.0 2.0 --folds 5 --n-pseudo 8 \
      --nuisance-fraction $FRAC --tag frac$FRAC --outdir $OUT 2>&1 | grep -E "nuisance=|World A \(" | head -3
done
echo "===== BY-FRACTION TAXONOMY (TSMNet) ====="
$PY - <<'PYEOF'
import json, glob, re
from tos_cmi.eeg.report_v2 import _worldA_taxonomy
OUT="tos_cmi/results/method_deepen/v2_tsm_robust"
print("%-6s %-8s %-16s %-9s %-8s %-8s %-8s"%("frac","m","oracle best[CI]","oracle_sup","clean","tgt_ben","ACCEPT"))
for f in ["0.15","0.20","0.25","0.30"]:
    p="%s/v2_frac%s_summary.json"%(OUT,f)
    try:
        S=json.load(open(p)); summ=S["summary"]; m=S["params"].get("m_TSMNet")
        A=[v for v in summ.values() if v["world"]=="A"]
        tax=_worldA_taxonomy(A,0.01)
        orc=[v for v in A if "oracle" in v["intervention"]]
        best=max(orc,key=lambda v:v["dtgt_bacc"])
        clean=sum(t["n_clean"] for t in tax); tb=sum(t["n_target_beneficial"] for t in tax)
        acc=sum(t["n_accept"] for t in tax); sup=all(t["oracle_supported"] for t in tax)
        rnd=max(v["dtgt_bacc"] for v in A if v["intervention"]=="random_k")
        print("%-6s m=%-6s %+.3f[%+.3f,%+.3f] %-9s %-8d %-8d %-8d (rnd %+.3f, oracle_sup_any=%s)"%(
            f,m,best["dtgt_bacc"],best["dtgt_bacc_lo"],best["dtgt_bacc_hi"],
            "Y" if any(t["oracle_supported"] for t in tax) else "n",clean,tb,acc,rnd,
            any(t["oracle_supported"] for t in tax)))
    except Exception as e: print("%-6s ERROR %r"%(f,e))
PYEOF
echo "V2TSMROBUST_SBATCH_DONE"
