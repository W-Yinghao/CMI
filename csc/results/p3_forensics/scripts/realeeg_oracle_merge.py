"""P3.0d oracle merge: gather the 12 oracle shards, verify sha256 + prov (0 worker errors, 0 fidelity fails),
assert every record fidelity_ok=True, exactly 1200 unique, canonical-sort. Fail-closed. Diagnostic-only."""
import os, sys, json, hashlib
OUTDIR="/home/infres/yinwang/realeeg_feas/p3_forensic/oracle"
BASES=[50000000,60000000,70000000]
CONDS=["NULL_cov","NULL_cov_plus_label","POS_concept","POS_concept_plus_cov"]
EXPECT=len(BASES)*len(CONDS)*100
def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()
def fail(m): print(f"INFRA-FAIL: {m}"); sys.exit(2)
records=[]
for base in BASES:
    for cond in CONDS:
        stem=f"{OUTDIR}/oracle_{base}_{cond}.jsonl"
        for suf in ("",".prov.json",".sha256"):
            if not os.path.exists(stem+suf): fail(f"missing {stem+suf}")
        if sha(stem)!=open(stem+".sha256").read().split()[0]: fail(f"sha mismatch {base}:{cond}")
        prov=json.load(open(stem+".prov.json"))
        if prov.get("n_worker_errors",1)!=0: fail(f"{base}:{cond} worker errors {prov['n_worker_errors']}")
        if prov.get("n_fidelity_fail",1)!=0: fail(f"{base}:{cond} fidelity fails {prov['n_fidelity_fail']}")
        recs=[json.loads(l) for l in open(stem) if l.strip()]
        if any("__worker_error__" in r for r in recs): fail(f"{base}:{cond} has worker-error records")
        if not all(r.get("fidelity_ok") for r in recs): fail(f"{base}:{cond} has fidelity_ok=False records")
        if len(recs)!=100: fail(f"{base}:{cond} has {len(recs)}!=100")
        records.extend(recs)
ids=[r["task_id"] for r in records]
if len(records)!=EXPECT or len(set(ids))!=EXPECT: fail(f"{len(records)}/{len(set(ids))} != {EXPECT}")
records.sort(key=lambda r:(r["seed_block"],r["condition_index"],r["cohort"]))
merged=dict(diagnostic_only=True,not_confirmatory=True,not_merged_with_v2=True,oracle_B=200,
            oracle_p_floor=1.0/201,n_records=len(records),seed_blocks=BASES,conditions=CONDS,per_cohort=records)
out=f"{OUTDIR}/p3_oracle_merged.json"; json.dump(merged,open(out,"w"),indent=1,default=str)
open(out+".sha256","w").write(sha(out)+"  "+os.path.basename(out)+"\n")
print(f"ORACLE MERGE OK: {len(records)} records, all fidelity_ok, hash-consistent -> {out}")
print(f"  sha256 {sha(out)}")
