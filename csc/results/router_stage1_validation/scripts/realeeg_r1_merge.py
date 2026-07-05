import os,sys,json,hashlib,glob
PHASE=sys.argv[1]; OUTDIR=f"/home/infres/yinwang/realeeg_feas/router_stage1/{PHASE}"
CIDX={"NULL_cov":1,"NULL_label":2,"NULL_cov_plus_label":3,"POS_concept":4,"POS_concept_plus_cov":5,"random_label_control":7}
CONDS=(["NULL_cov","NULL_cov_plus_label"] if PHASE=="calibration" else list(CIDX)); EXPECT=len(CONDS)*300
def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()
def fail(m): print(f"INFRA-FAIL: {m}"); sys.exit(2)
recs=[]
for stem in sorted(glob.glob(f"{OUTDIR}/r1_*_*.jsonl")):
    if stem.endswith((".sha256",".prov.json")): continue
    for suf in (".sha256",".prov.json"):
        if not os.path.exists(stem+suf): fail(f"missing {stem+suf}")
    if sha(stem)!=open(stem+".sha256").read().split()[0]: fail(f"sha mismatch {stem}")
    if json.load(open(stem+".prov.json")).get("n_worker_errors",1)!=0: fail(f"{stem} worker errors")
    rs=[json.loads(l) for l in open(stem) if l.strip()]
    if any("__worker_error__" in r for r in rs): fail(f"{stem} worker-error rows")
    recs.extend(rs)
ids=[r["task_id"] for r in recs]
if len(recs)!=EXPECT or len(set(ids))!=EXPECT: fail(f"{len(recs)}/{len(set(ids))} != {EXPECT}")
import collections; cc=collections.Counter(r["condition"] for r in recs)
for c in CONDS:
    if cc[c]!=300: fail(f"{c} count {cc[c]}!=300")
recs.sort(key=lambda r:(r["condition"],r["cohort"]))
out=f"{OUTDIR}/r1_{PHASE}_merged.json"
json.dump(dict(diagnostic_only=True,phase=PHASE,n_records=len(recs),conditions=CONDS,per_cohort=recs),open(out,"w"),indent=1,default=str)
open(out+".sha256","w").write(sha(out)+"  "+os.path.basename(out)+"\n")
print(f"R1 {PHASE} MERGE OK: {len(recs)} unique ({', '.join(f'{c}={cc[c]}' for c in CONDS)}) -> {out} sha256 {sha(out)[:16]}")
