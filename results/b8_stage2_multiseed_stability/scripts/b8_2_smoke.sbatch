#!/bin/bash
#SBATCH --job-name=b8_2_smoke
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/infres/yinwang/realeeg_feas/b8_2_smoke_%j.out
set -uo pipefail
mkdir -p /home/infres/yinwang/realeeg_feas/b8_2_smoke_tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/infres/yinwang/realeeg_feas
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate icml
echo "[b8_2_smoke] host=$(hostname) START (engine UNCHANGED; smoke checks base-wiring + record fields + merge helpers)"
# run the UNCHANGED B8.1 worker at a B8.2 base (500e6) for 2 worlds, n=5, to verify wiring + output naming
python -u realeeg_b8_1_canary.py --base 500000000 --world CONTRACT_NULL_prior_only --start 0 --n 5 --jobs 8 --out b8_2_smoke_tmp/b8_2_500000000_CONTRACT_NULL_prior_only.jsonl
python -u realeeg_b8_1_canary.py --base 500000000 --world CONTRACT_POS_boundary     --start 0 --n 5 --jobs 8 --out b8_2_smoke_tmp/b8_2_500000000_CONTRACT_POS_boundary.jsonl
python -u realeeg_b8_1_canary.py --base 520000000 --world CONTRACT_NULL_prior_only --start 0 --n 5 --jobs 8 --out b8_2_smoke_tmp/b8_2_520000000_CONTRACT_NULL_prior_only.jsonl
python - <<'PY'
import json, sys
sys.path.insert(0,'/home/infres/yinwang/realeeg_feas'); sys.path.insert(0,'/home/infres/yinwang/CMI_AAAI_csc')
import b8_2_multiseed_merge as M
# record-field + seed-range + cross-base disjointness check
D='b8_2_smoke_tmp'
a=[json.loads(l) for l in open(f'{D}/b8_2_500000000_CONTRACT_NULL_prior_only.jsonl')]
b=[json.loads(l) for l in open(f'{D}/b8_2_500000000_CONTRACT_POS_boundary.jsonl')]
c=[json.loads(l) for l in open(f'{D}/b8_2_520000000_CONTRACT_NULL_prior_only.jsonl')]
req={'b8_state','contract_valid','provenance_match','p_exact_meanT','p_exact_stud','observed_T','exact_null_mean_T','exact_null_sd_T','contract_within_block_C_Z_auc','contract_n_support_blocks','seed','task_id'}
for nm,recs in (('500/prior',a),('500/POS',b),('520/prior',c)):
    miss=req-set(recs[0]); assert not miss, (nm,miss)
    print(nm,'seeds',[r['seed'] for r in recs[:2]],'state0',recs[0]['b8_state'])
# base-500 prior seeds must be 500e6+1*1e6+r (prior_only world_index=1); base-520 disjoint
s500=set(r['seed'] for r in a); s520=set(r['seed'] for r in c)
assert s500.isdisjoint(s520), 'base overlap!'
assert all(500_000_000<=s<=511_000_049 for s in s500), 'base500 range'
assert all(520_000_000<=s<=531_000_049 for s in s520), 'base520 range'
print('cp95u(0,300)=',round(M.cp95u(0,300),4),'cp95u(7,300)=',round(M.cp95u(7,300),4),'cp95u(2,50)=',round(M.cp95u(2,50),4))
print('B8_2_SMOKE_OK')
PY
echo "[b8_2_smoke] rc=$? DONE"
