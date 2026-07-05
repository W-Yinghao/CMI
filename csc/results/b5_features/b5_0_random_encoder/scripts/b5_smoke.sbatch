#!/bin/bash
#SBATCH --job-name=b5_smoke
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/infres/yinwang/realeeg_feas/b5_features/b5_0_random_encoder/canary/_smoke/b5_smoke_%j.out
set -uo pipefail
cd /home/infres/yinwang/realeeg_feas
mkdir -p b5_features/b5_0_random_encoder/canary/_smoke
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[b5_smoke] host=$(hostname) job=${SLURM_JOB_ID:-na} START"
python -u -c "
import numpy as np, time
import realeeg_b5_canary as B5
import realeeg_internal_forensic as IF
from csc.mininfo import realeeg_engine as EG
print('CACHE swapped: Z', IF.CACHE['Z'].shape, 'nsubj', len(IF.SUBJECTS), flush=True)
print('feature_family', B5.FEATURE_FAMILY, 'b5_sha', B5._B5_CACHE_SHA[:12], flush=True)
# (1) structural: build_cohort on Z_deep for all 6 conditions (no certifier) -- fast
rng=np.random.default_rng(30000000+1)
subj=rng.choice(IF.SUBJECTS,size=min(IF.M,len(IF.SUBJECTS)),replace=False)
sel=np.isin(IF.CACHE['subject'],subj)
coh={k:IF.CACHE[k][sel] for k in ('Z','y','subject','session')}
for c in ['NULL_cov','NULL_cov_plus_label','NULL_label','random_label_control','POS_concept','POS_concept_plus_cov']:
    t=time.time(); Z,Y,D,G=EG.build_cohort(c,coh,np.random.default_rng(30000000+hash(c)%997))
    print(f'  build {c:24s} Z{Z.shape} Ymean={Y.mean():.3f} D={sorted(set(D.tolist()))} nsubj={len(set(G.tolist()))} ({time.time()-t:.1f}s)', flush=True)
# (2) time ONE full certifier cohort for NULL_cov and one for POS_concept
for c in ['NULL_cov','POS_concept']:
    t=time.time(); rec=B5.run_cohort_b5(c, 0, 30000000); dt=time.time()-t
    print(f'  CERTIFY {c}: {dt:.1f}s | state={rec.get(\"b3_state\")} false_confirm={rec.get(\"false_confirm\")} true_confirm={rec.get(\"true_confirm\")} T={rec.get(\"observed_T\")} null_sd={rec.get(\"null_sd_T\")} ffp={rec.get(\"fixed_margin_p\")} Tz={rec.get(\"T_z\")} fam={rec.get(\"feature_family\")}', flush=True)
print('SMOKE_OK', flush=True)
"
echo "[b5_smoke] rc=$? DONE"
