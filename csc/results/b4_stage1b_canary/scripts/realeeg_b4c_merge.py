import os,sys,json,hashlib,glob
OUTDIR="/home/infres/yinwang/realeeg_feas/b4_stage1b"
EXPECT=json.load(open(f"{OUTDIR}/b4_stage1b_manifest.json"))["n_cohorts"]
def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()
def fail(m): print(f"INFRA-FAIL: {m}"); sys.exit(2)
recs=[]
for stem in sorted(glob.glob(f"{OUTDIR}/b4c_shard_*_*.jsonl")):
    if stem.endswith((".sha256",".prov.json")): continue
    for suf in (".sha256",".prov.json"):
        if not os.path.exists(stem+suf): fail(f"missing {stem+suf}")
    if sha(stem)!=open(stem+".sha256").read().split()[0]: fail(f"sha mismatch {stem}")
    prov=json.load(open(stem+".prov.json"))
    if prov.get("n_worker_errors",1)!=0: fail(f"{stem} worker errors")
    if prov.get("n_repro_fail",1)!=0: fail(f"{stem} repro fails")
    rs=[json.loads(l) for l in open(stem) if l.strip()]
    if any("__worker_error__" in r for r in rs): fail(f"{stem} worker-error rows")
    if any(r.get("observed_T_repro_ok") is False for r in rs): fail(f"{stem} repro-fail rows")
    recs.extend(rs)
ids=[r["task_id"] for r in recs]
if len(recs)!=EXPECT or len(set(ids))!=EXPECT: fail(f"{len(recs)}/{len(set(ids))} != {EXPECT}")
recs.sort(key=lambda r:(r.get("condition"),r.get("stratum"),r.get("seed_block"),r.get("cohort")))
merged=dict(diagnostic_only=True,not_confirmatory=True,not_deployable=True,stage="B4_stage1b_canary",candidate="B4c-Q3",n_records=len(recs),per_cohort=recs)
out=f"{OUTDIR}/b4_stage1b_merged.json"; json.dump(merged,open(out,"w"),indent=1,default=str)
open(out+".sha256","w").write(sha(out)+"  "+os.path.basename(out)+"\n")
print(f"B4c MERGE OK: {len(recs)} unique, hash-consistent -> {out} sha256 {sha(out)[:16]}")
