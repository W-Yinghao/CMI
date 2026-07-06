"""R1 scale-up STRONG-COVARIATE null (development-only). Tests the router's PREDICTED failure mode (red-team #1
caveat): R1's type-I control was demonstrated only against a SOFT session covariate (session_auc~0.52). This module
builds a NO-CONCEPT null with a STRONG covariate: shift session-2's Z along a direction ORTHOGONAL to the pooled
logistic boundary w, raising session_auc toward a target, while drawing Y* from the pooled boundary evaluated at the
shifted Z. Because the shift v is orthogonal to w, w.Z' == w.Z, so P(Y|Z') = pooled boundary in BOTH sessions ->
GROUND TRUTH NO_CONCEPT by construction; only the marginal P(Z|session) changes.

Frozen-path safe: this does NOT mutate realeeg_engine.py or realeeg_internal_forensic.py. It reuses IF.CACHE (SM16),
EG._pooled_clf/_draw (byte-frozen injection helpers) and the byte-unchanged certify_paired_calibrated. Condition
index 6 (unused in IF.CIDX) for a disjoint seed stream. NO retune of tau; the router evaluation loads the locked tau.
"""
import os, sys, json, argparse, hashlib, socket
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
import numpy as np
import realeeg_internal_forensic as IF
from csc.mininfo import realeeg_engine as EG
from csc.mininfo.paired_calibrated import certify_paired_calibrated
from csc.mininfo.paired_certifier import CONCEPT_CONFIRMED

SC_CIDX = 6                     # disjoint from IF.CIDX (1,2,3,4,5,7)
STRIDE = IF.STRIDE
M = IF.M


def build_cohort_strongcov(cohort, rng, delta):
    """NO-CONCEPT cohort with an amplified session covariate. Returns (Z', Y*, D=session, G=subject)."""
    Z, y, subj, sess = cohort["Z"], cohort["y"], cohort["subject"], cohort["session"]
    clf = EG._pooled_clf(Z, y)                       # pooled boundary on the REAL features
    w = clf.coef_.ravel().astype(float)
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    # label-free shift direction: leading PC of Z, projected OFF the boundary normal (so w.v = 0 exactly)
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    v = Vt[0].astype(float)
    v = v - (v @ w_hat) * w_hat                       # orthogonalize against w
    v_hat = v / (np.linalg.norm(v) + 1e-12)
    scale = float(np.sqrt(np.mean((Zc @ v_hat) ** 2)))  # natural scale along v (std of projection)
    Zp = Z.copy()
    Zp[sess == 2] = Zp[sess == 2] + delta * scale * v_hat   # shift session-2 marginal along v (label-free)
    # Y* drawn from the pooled boundary at Zp; since w.v_hat=0, w.Zp == w.Z -> label session-invariant (NO concept)
    p = clf.predict_proba(Zp)[:, 1]
    Ystar = EG._draw(p, rng)
    return Zp, Ystar, sess.copy(), subj.copy()


def run_cohort_strongcov(r, base, delta):
    seed = base + SC_CIDX * STRIDE + r
    try:
        rng = np.random.default_rng(seed)
        subj = rng.choice(IF.SUBJECTS, size=min(M, len(IF.SUBJECTS)), replace=False)
        sel = np.isin(IF.CACHE["subject"], subj)
        coh = {k: IF.CACHE[k][sel] for k in ("Z", "y", "subject", "session")}
        Z, Y, D, G = build_cohort_strongcov(coh, rng, delta)
        log = certify_paired_calibrated(Z, Y, D, G, m=len(subj), min_confirm_pairs=IF.ML["min_confirm_pairs"],
              pair_integrity_min=IF.ML["pair_integrity_min"], min_epochs=IF.ML["min_epochs"], rank=IF.ML["rank"],
              C=IF.ML["C"], n_folds=IF.ML["n_folds"], n_boot=IF.ML["n_boot"], seed=seed,
              alpha_family=IF.ML["alpha_family"], n_decision_budgets=IF.ML["n_decision_budgets"])
        state = str(log.get("state"))
        T = log.get("observed_T", float("nan"))
        deltas = log.get("delta_subjects", []) or []
        rec = dict(condition="NULL_cov_strong", condition_index=SC_CIDX, cohort=r, seed=int(seed), delta=float(delta),
                   task_id=f"{base}:{SC_CIDX:02d}:NULL_cov_strong:{r:04d}",
                   b3_state=state, false_confirm=bool(state == CONCEPT_CONFIRMED),   # GT NO_CONCEPT -> any confirm is false
                   ground_truth_noconcept=True,
                   observed_T=IF._f(T), fixed_margin_p=IF._f(log.get("fixed_margin_null_p")),
                   studentized_p=IF._f(log.get("studentized_p_value")),
                   subject_consistency_lcb=IF._f(log.get("subject_consistency_lcb")),
                   null_sd_T=IF._f(log.get("null_sd")), null_mean_T=IF._f(log.get("null_mean")),
                   valid=bool(log.get("valid", False)), diagnostic_only=True, not_deployable=True)
        nm = log.get("null_mean", float("nan"))
        rec["T_z"] = IF._f((T - nm) / log.get("null_sd")) if IF._ok(T) and IF._ok(nm) and IF._ok(log.get("null_sd")) and log.get("null_sd") else float("nan")
        rec.update(IF.overlap_diag(Z, D, G))          # session_auc = strength of the injected covariate
        return rec
    except Exception as e:
        return dict(task_id=f"{base}:{SC_CIDX}:NULL_cov_strong:{r:04d}", __worker_error__=f"{type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=a.jobs, backend="loky")(delayed(run_cohort_strongcov)(r, a.base, a.delta) for r in range(a.start, a.start + a.n))
    errs = [x for x in recs if "__worker_error__" in x]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for x in recs: f.write(json.dumps(x, default=str) + "\n")
    open(a.out + ".sha256", "w").write(hashlib.sha256(open(a.out, "rb").read()).hexdigest() + "  " + os.path.basename(a.out) + "\n")
    json.dump(dict(base=a.base, delta=a.delta, start=a.start, n=a.n, n_records=len(recs), n_worker_errors=len(errs),
                   host=socket.gethostname(), slurm=os.environ.get("SLURM_JOB_ID"), diagnostic_only=True),
              open(a.out + ".prov.json", "w"), indent=1, default=str)
    print(f"[strongcov {a.base}:d{a.delta}:{a.start}+{a.n}] {len(recs)} recs, {len(errs)} err, "
          f"session_auc_med={np.median([r.get('session_auc', float('nan')) for r in recs if 'session_auc' in r]):.3f}, "
          f"false_confirm={sum(1 for r in recs if r.get('false_confirm'))}", flush=True)
    sys.exit(2 if errs else 0)
