"""CSC B8.0 canary worker (development-only, NO tag). Per cohort: build a world (contract-satisfying or contract-
violating) on Lee2019 SM16 geometry, run b8_certify (contract-first refusal + EXACT block-stratified randomization
null). Records the B8 state + contract diagnostics. Fresh base 400e6. Fail-closed."""
import os, sys, json, argparse, hashlib, socket
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
import numpy as np
import realeeg_internal_forensic as IF
import realeeg_b8 as B8

STRIDE = IF.STRIDE; M = IF.M
WORLD_IDX = {w: i for i, w in enumerate(B8.WORLDS)}


def _f(x):
    try: return float(x)
    except Exception: return float("nan")


def run_cohort(world, r, base):
    idx = WORLD_IDX[world]; seed = base + idx * STRIDE + r
    kind, has_concept = B8.WORLDS[world]
    try:
        rng = np.random.default_rng(seed)
        subj = rng.choice(IF.SUBJECTS, size=min(M, len(IF.SUBJECTS)), replace=False)
        sel = np.isin(IF.CACHE["subject"], subj)
        coh = {k: IF.CACHE[k][sel] for k in ("Z", "y", "subject", "session")}
        Z, Y, C, G, Block, contract_intended = B8.build_b8_cohort(world, coh, rng)
        rr = B8.b8_certify(Z, Y, C, G, Block, m=len(subj), seed=seed, n_boot=IF.ML["n_boot"])
        st = str(rr.get("b8_state"))
        alert = (st == "B8_CONCEPT_ALERT")
        rec = dict(world=world, world_index=idx, world_class=kind, has_concept=bool(has_concept),
                   contract_intended=bool(contract_intended), cohort=r, seed=int(seed),
                   task_id=f"{base}:{idx:02d}:{world}:{r:04d}",
                   b8_state=st, contract_valid=bool(rr.get("contract_valid")),
                   b8_alert=bool(alert),
                   b8_false_alert=bool(alert and not has_concept),      # alert on a no-concept world = type-I error
                   b8_true_alert=bool(alert and has_concept),           # alert on a concept world = power
                   observed_T=_f(rr.get("observed_T")), p_exact_meanT=_f(rr.get("p_exact_meanT")),
                   p_exact_stud=_f(rr.get("p_exact_stud")),
                   exact_null_mean_T=_f(rr.get("exact_null_mean_T")), exact_null_sd_T=_f(rr.get("exact_null_sd_T")),
                   contract_within_block_C_Z_auc=_f(rr.get("contract_within_block_C_Z_auc")),
                   contract_n_support_blocks=int(rr.get("contract_n_support_blocks", 0) or 0),
                   n_exact_invalid=int(rr.get("n_exact_invalid", 0) or 0),
                   n_eligible=int(rr.get("n_eligible", 0) or 0), diagnostic_only=True)
        return rec
    except Exception as e:
        return dict(task_id=f"{base}:{idx}:{world}:{r:04d}", __worker_error__=f"{type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, required=True)
    ap.add_argument("--world", type=str, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    assert a.world in B8.WORLDS, f"unknown world {a.world}"
    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=a.jobs, backend="loky")(delayed(run_cohort)(a.world, r, a.base) for r in range(a.start, a.start + a.n))
    errs = [x for x in recs if "__worker_error__" in x]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for x in recs: f.write(json.dumps(x, default=str) + "\n")
    open(a.out + ".sha256", "w").write(hashlib.sha256(open(a.out, "rb").read()).hexdigest() + "  " + os.path.basename(a.out) + "\n")
    json.dump(dict(base=a.base, world=a.world, start=a.start, n=a.n, n_records=len(recs), n_worker_errors=len(errs),
                   host=socket.gethostname(), slurm=os.environ.get("SLURM_JOB_ID"), diagnostic_only=True),
              open(a.out + ".prov.json", "w"), indent=1, default=str)
    from collections import Counter
    sc = Counter(r.get("b8_state") for r in recs if "__worker_error__" not in r)
    print(f"[b8 {a.base}:{a.world}:{a.start}+{a.n}] {len(recs)} recs, {len(errs)} err | alert={sum(1 for r in recs if r.get('b8_alert'))} "
          f"contract_valid={sum(1 for r in recs if r.get('contract_valid'))} | states={dict(sc)}", flush=True)
    sys.exit(2 if errs else 0)
