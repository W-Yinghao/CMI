"""CSC B8.3 canary worker (development-only, NO tag). Per cohort: build a world (reuse byte-frozen B8.1
build_b8_1_cohort) on Lee2019 SM16, run b8_3_certify (label-balanced case-control audit selector + exact null that
RE-APPLIES the selector under each randomized C*). Records B8 state + audit-selector diagnostics + BOTH gates (both-gate
ALERT and mean-T-alone). Fail-closed."""
import os, sys, json, argparse, hashlib, socket
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
import numpy as np
import realeeg_internal_forensic as IF
import realeeg_b8_1 as B81
import realeeg_b8_3 as B83

STRIDE = IF.STRIDE; M = IF.M
WORLD_IDX = {w: i for i, w in enumerate(B83.WORLDS)}


def _f(x):
    try: return float(x)
    except Exception: return float("nan")


def run_cohort(world, r, base):
    idx = WORLD_IDX[world]; seed = base + idx * STRIDE + r
    kind, has_concept = B83.WORLDS[world]
    try:
        rng = np.random.default_rng(seed)
        subj = rng.choice(IF.SUBJECTS, size=min(M, len(IF.SUBJECTS)), replace=False)
        sel = np.isin(IF.CACHE["subject"], subj)
        coh = {k: IF.CACHE[k][sel] for k in ("Z", "y", "subject", "session")}
        Z, Y, C, G, Block, Dc, C_table, table_hash, contract_intended = B81.build_b8_1_cohort(world, coh, rng)
        rr = B83.b8_3_certify(Z, Y, C, G, Block, Dc, C_table, table_hash, m=len(subj), seed=seed, n_boot=IF.ML["n_boot"])
        st = str(rr.get("b8_state")); alert = (st == "B8_CONCEPT_ALERT")
        rec = dict(world=world, world_index=idx, world_class=kind, has_concept=bool(has_concept),
                   contract_intended=bool(contract_intended), cohort=r, seed=int(seed),
                   task_id=f"{base}:{idx:02d}:{world}:{r:04d}",
                   b8_state=st, contract_valid=bool(rr.get("contract_valid")),
                   provenance_match=_f(rr.get("provenance_match")),
                   contract_invalid_reasons=list(rr.get("contract_invalid_reasons", [])),
                   b8_alert=bool(alert), meanT_alone=bool(rr.get("meanT_alone")),
                   b8_false_alert=bool(alert and not has_concept), b8_true_alert=bool(alert and has_concept),
                   audit_selected_n=int(rr.get("audit_selected_n", 0) or 0),
                   audit_cxy_imbalance=int(rr.get("audit_cxy_imbalance", -1)),
                   audit_n_infeasible_strata=int(rr.get("audit_n_infeasible_strata", 0) or 0),
                   null_infeasible_draws=int(rr.get("null_infeasible_draws", 0) or 0),
                   null_selected_n_mean=_f(rr.get("null_selected_n_mean")),
                   selection_intensity_asymmetry=_f(rr.get("selection_intensity_asymmetry")),
                   observed_T=_f(rr.get("observed_T")), observed_Tz=_f(rr.get("observed_Tz")),
                   p_exact_meanT=_f(rr.get("p_exact_meanT")), p_exact_stud=_f(rr.get("p_exact_stud")),
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
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    assert a.world in B83.WORLDS, f"unknown world {a.world}"
    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=a.jobs, backend="loky")(delayed(run_cohort)(a.world, r, a.base) for r in range(a.start, a.start + a.n))
    errs = [x for x in recs if "__worker_error__" in x]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for x in recs: f.write(json.dumps(x, default=str) + "\n")
    open(a.out + ".sha256", "w").write(hashlib.sha256(open(a.out, "rb").read()).hexdigest() + "  " + os.path.basename(a.out) + "\n")
    json.dump(dict(base=a.base, world=a.world, start=a.start, n=a.n, n_records=len(recs), n_worker_errors=len(errs),
                   host=socket.gethostname(), slurm=os.environ.get("SLURM_JOB_ID"),
                   module_sha256=hashlib.sha256(open("/home/infres/yinwang/realeeg_feas/realeeg_b8_3.py", "rb").read()).hexdigest()[:16],
                   diagnostic_only=True),
              open(a.out + ".prov.json", "w"), indent=1, default=str)
    from collections import Counter
    sc = Counter(r.get("b8_state") for r in recs if "__worker_error__" not in r)
    print(f"[b8.3 {a.base}:{a.world}:{a.start}+{a.n}] {len(recs)} recs, {len(errs)} err | alert={sum(1 for r in recs if r.get('b8_alert'))} "
          f"meanT_alone={sum(1 for r in recs if r.get('meanT_alone'))} valid={sum(1 for r in recs if r.get('contract_valid'))} | states={dict(sc)}", flush=True)
    sys.exit(2 if errs else 0)
