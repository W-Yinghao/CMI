"""Router R1 worker (development-only). Thin slicing wrapper around the VALIDATED P3 certifier-internal
run_cohort (realeeg_internal_forensic.run_cohort = certify_paired_calibrated) -- the B3 method path is UNCHANGED.
Captures b3_state (method_confirm), observed_T (deployable score), and report-only diagnostics. Fresh seeds via
--base / --start / --n. Fail-closed on worker error."""
import os, sys, json, argparse, hashlib, socket
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
sys.path.insert(0,"/home/infres/yinwang/realeeg_feas")
import realeeg_internal_forensic as IF
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",type=int,required=True); ap.add_argument("--condition",required=True)
    ap.add_argument("--start",type=int,default=0); ap.add_argument("--n",type=int,default=100)
    ap.add_argument("--jobs",type=int,default=8); ap.add_argument("--out",required=True)
    a=ap.parse_args(); assert a.condition in IF.CIDX
    from joblib import Parallel, delayed
    recs=Parallel(n_jobs=a.jobs,backend="loky")(delayed(IF.run_cohort)(a.condition,r,a.base) for r in range(a.start,a.start+a.n))
    errs=[x for x in recs if "__worker_error__" in x]
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    with open(a.out,"w") as f:
        for x in recs: f.write(json.dumps(x,default=str)+"\n")
    open(a.out+".sha256","w").write(hashlib.sha256(open(a.out,"rb").read()).hexdigest()+"  "+os.path.basename(a.out)+"\n")
    json.dump(dict(base=a.base,condition=a.condition,start=a.start,n=a.n,n_records=len(recs),n_worker_errors=len(errs),
                   host=socket.gethostname(),slurm=os.environ.get("SLURM_JOB_ID"),diagnostic_only=True),
              open(a.out+".prov.json","w"),indent=1,default=str)
    print(f"[r1 {a.base}:{a.condition}:{a.start}+{a.n}] {len(recs)} recs, {len(errs)} err, "
          f"confirm={sum(1 for x in recs if x.get('b3_state')=='CONCEPT_CONFIRMED')}",flush=True)
    sys.exit(2 if errs else 0)
