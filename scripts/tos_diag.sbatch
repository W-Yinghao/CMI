#!/bin/bash
#SBATCH --job-name=tos_diag
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
cd /home/infres/yinwang/CMI_AAAI_tos
export PYTHONPATH=/home/infres/yinwang/CMI_AAAI_tos
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/infres/yinwang/anaconda3/envs/icml/bin/python - <<'PY'
import glob, traceback
from tos_cmi.eeg.erasure_baselines import analyze
p = sorted(glob.glob("tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_004_TSMNet_LOSO/sub1_erm_lam0_seed0.npz"))[0]
print("DIAG analyze TSMNet 2b:", p.split("/")[-1], flush=True)
try:
    r = analyze(p, with_rlace=False)
    print("OK keys:", {k: r[k] for k in r if k.startswith(("subj_","task_")) or k in ("z_dim","nDcand","n_subj")})
except Exception:
    traceback.print_exc()
PY
echo "DIAG_DONE"
