"""B4 Stage 1 fail-closed merge: gather shards, verify prov (0 worker/fidelity fail), 206 unique, sort, sha256."""
import os,sys,json,hashlib,glob
OUTDIR="/home/infres/yinwang/realeeg_feas/b4_stage1"
EXPECT=json.load(open(f"{OUTDIR}/b4_stage1_manifest.json"))["n_cohorts"]
def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()
def fail(m): print(f"INFRA-FAIL: {m}"); sys.exit(2)
recs=[]
for stem in sorted(glob.glob(f"{OUTDIR}/b4_shard_*_*.jsonl")):
    if stem.endswith(".sha256") or stem.endswith(".prov.json"): continue
    for suf in (".sha256",".prov.json"):
        if not os.path.exists(stem+suf): fail(f"missing {stem+suf}")
    if sha(stem)!=open(stem+".sha256").read().split()[0]: fail(f"sha mismatch {stem}")
    prov=json.load(open(stem+".prov.json"))
    if prov.get("n_worker_errors",1)!=0: fail(f"{stem} worker errors")
    if prov.get("n_fidelity_fail",1)!=0: fail(f"{stem} fidelity fails")
    rs=[json.loads(l) for l in open(stem) if l.strip()]
    if any("__worker_error__" in r for r in rs): fail(f"{stem} worker-error rows")
    if any(r.get("fidelity_ok") is False for r in rs): fail(f"{stem} fidelity_ok=False rows")
    recs.extend(rs)
ids=[r["task_id"] for r in recs]
if len(recs)!=EXPECT or len(set(ids))!=EXPECT: fail(f"{len(recs)}/{len(set(ids))} != {EXPECT}")
recs.sort(key=lambda r:(r.get("condition"),r.get("stratum"),r.get("seed_block"),r.get("cohort")))
merged=dict(diagnostic_only=True,not_confirmatory=True,not_deployable=True,no_method_change=True,
            stage="B4_stage1_canary",n_records=len(recs),per_cohort=recs)
out=f"{OUTDIR}/b4_stage1_merged.json"; json.dump(merged,open(out,"w"),indent=1,default=str)
open(out+".sha256","w").write(sha(out)+"  "+os.path.basename(out)+"\n")
print(f"B4 MERGE OK: {len(recs)} records unique, hash-consistent -> {out}  sha256 {sha(out)[:16]}")
