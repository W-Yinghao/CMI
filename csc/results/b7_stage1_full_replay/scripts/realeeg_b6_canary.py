"""CSC B6.0 condition-randomization null CANARY worker (development-only, NO tag). Per cohort runs BOTH:
  (baseline) certify_paired_calibrated  -> observed_T, fixed_margin_p (OLD fitted-h0 Y-null), b3_state
  (B6)       crt_test                   -> p_C (C-randomization null), propensity/lock diagnostics
across the 8 reviewer-specified conditions INCLUDING the two strong-covariate levels. Answers the 3 mechanism Qs:
does the strong-cov null stop false-confirming under the C-null? is observed_T a typical draw from the C-null? does
POS retain signal? Fail-closed. Fresh base seed 200e6 (disjoint from all prior). B3 T byte-reused, injection unchanged.
"""
import os, sys, json, argparse, hashlib, socket
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
import numpy as np
import realeeg_internal_forensic as IF               # CACHE, SUBJECTS, M, ML (frozen)
import realeeg_strongcov as SC                        # build_cohort_strongcov (frozen this session)
from csc.mininfo import realeeg_engine as EG
from csc.mininfo.paired_calibrated import certify_paired_calibrated
from csc.mininfo.paired_certifier import CONCEPT_CONFIRMED
import realeeg_crt as CRT

STRIDE = IF.STRIDE; M = IF.M
# (name, kind, condition_index, delta[strong only]) -- strong deltas from the calibrated strong-cov bank
B6_CONDS = [
    ("NULL_cov_soft",            "std",    "NULL_cov",             1, None),
    ("NULL_cov_plus_label_soft", "std",    "NULL_cov_plus_label",  3, None),
    ("NULL_cov_strong_auc0.81",  "strong", None,                   8, 1.5),
    ("NULL_cov_strong_auc0.94",  "strong", None,                   9, 2.5),
    ("NULL_label",               "std",    "NULL_label",           2, None),
    ("random_label_control",     "std",    "random_label_control", 7, None),
    ("POS_concept",              "std",    "POS_concept",          4, None),
    ("POS_concept_plus_cov",     "std",    "POS_concept_plus_cov", 5, None),
]
CMAP = {c[0]: c for c in B6_CONDS}
NOCONCEPT = {"NULL_cov_soft", "NULL_cov_plus_label_soft", "NULL_cov_strong_auc0.81",
             "NULL_cov_strong_auc0.94", "NULL_label", "random_label_control"}


def _f(x):
    try: return float(x)
    except Exception: return float("nan")


def run_cohort(cond_name, r, base):
    name, kind, std_name, cidx, delta = CMAP[cond_name]
    seed = base + cidx * STRIDE + r
    try:
        rng = np.random.default_rng(seed)
        subj = rng.choice(IF.SUBJECTS, size=min(M, len(IF.SUBJECTS)), replace=False)
        sel = np.isin(IF.CACHE["subject"], subj)
        coh = {k: IF.CACHE[k][sel] for k in ("Z", "y", "subject", "session")}
        if kind == "strong":
            Z, Y, D, G = SC.build_cohort_strongcov(coh, rng, delta)
        else:
            Z, Y, D, G = EG.build_cohort(std_name, coh, rng)
        gt_noconcept = cond_name in NOCONCEPT
        # OLD baseline certifier (fitted-h0 Y-null)
        log = certify_paired_calibrated(Z, Y, D, G, m=len(subj), min_confirm_pairs=IF.ML["min_confirm_pairs"],
              pair_integrity_min=IF.ML["pair_integrity_min"], min_epochs=IF.ML["min_epochs"], rank=IF.ML["rank"],
              C=IF.ML["C"], n_folds=IF.ML["n_folds"], n_boot=IF.ML["n_boot"], seed=seed,
              alpha_family=IF.ML["alpha_family"], n_decision_budgets=IF.ML["n_decision_budgets"])
        old_state = str(log.get("state")); T_old = log.get("observed_T", float("nan"))
        # B6 C-randomization null
        crt = CRT.crt_test(Z, Y, D, G, m=len(subj), seed=seed, rank=IF.ML["rank"], C=IF.ML["C"],
                           n_folds=IF.ML["n_folds"], min_epochs=IF.ML["min_epochs"],
                           min_confirm_pairs=IF.ML["min_confirm_pairs"], n_boot=IF.ML["n_boot"])
        T_crt = crt.get("observed_T_crt", float("nan"))
        # FIDELITY: crt must reproduce the certifier's observed_T byte-for-byte (same eligible-pick + folds)
        fidelity_dT = _f(abs(T_old - T_crt)) if (T_old == T_old and T_crt == T_crt) else float("nan")
        rec = dict(condition=cond_name, condition_index=cidx, cohort=r, seed=int(seed), delta=delta,
                   task_id=f"{base}:{cidx:02d}:{cond_name}:{r:04d}", ground_truth_noconcept=gt_noconcept,
                   # OLD baseline
                   old_b3_state=old_state, observed_T=_f(T_old), old_fixed_margin_p=_f(log.get("fixed_margin_null_p")),
                   old_false_confirm=bool(old_state == CONCEPT_CONFIRMED and gt_noconcept),
                   old_true_confirm=bool(old_state == CONCEPT_CONFIRMED and not gt_noconcept),
                   old_null_sd_T=_f(log.get("null_sd")),
                   # B6 C-null
                   b6_state=crt.get("b6_state"), b6_valid=bool(crt.get("b6_valid")),
                   observed_T_crt=_f(T_crt), fidelity_dT=fidelity_dT,
                   p_C_meanT=_f(crt.get("p_C_meanT")), p_C_stud=_f(crt.get("p_C_stud")),
                   c_null_mean_T=_f(crt.get("c_null_mean_T")), c_null_sd_T=_f(crt.get("c_null_sd_T")),
                   b6_confirm=bool(crt.get("b6_confirm")),
                   # authoritative false/true-confirm key off the STATE (includes lock gate + size_ok), NOT the
                   # p-only b6_confirm flag (design-red-team fix): a lock-abstain must NOT count as a confirm.
                   b6_false_confirm=bool(crt.get("b6_state") == "CONCEPT_CONFIRMED_B6" and gt_noconcept),
                   b6_true_confirm=bool(crt.get("b6_state") == "CONCEPT_CONFIRMED_B6" and not gt_noconcept),
                   b6_confirm_p_only=bool(crt.get("b6_confirm")),   # diagnostic: p-gate alone (pre lock/size gates)
                   b6_abstain_lock=bool(crt.get("b6_state") == "UNIDENTIFIABLE_DUE_TO_COVARIATE_LOCK"),
                   n_crt_invalid=int(crt.get("n_crt_invalid", 0) or 0),
                   # propensity / condition-lock diagnostics
                   propensity_auc=_f(crt.get("propensity_auc")),
                   propensity_mean_entropy=_f(crt.get("propensity_mean_entropy")),
                   eff_randomization=_f(crt.get("eff_randomization")),
                   frac_condition_locked=_f(crt.get("frac_condition_locked")),
                   diagnostic_only=True, not_deployable=True)
        return rec
    except Exception as e:
        return dict(task_id=f"{base}:{cidx}:{cond_name}:{r:04d}", __worker_error__=f"{type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, required=True)
    ap.add_argument("--condition", type=str, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    assert a.condition in CMAP, f"unknown B6 condition {a.condition}"
    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=a.jobs, backend="loky")(delayed(run_cohort)(a.condition, r, a.base) for r in range(a.start, a.start + a.n))
    errs = [x for x in recs if "__worker_error__" in x]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for x in recs: f.write(json.dumps(x, default=str) + "\n")
    open(a.out + ".sha256", "w").write(hashlib.sha256(open(a.out, "rb").read()).hexdigest() + "  " + os.path.basename(a.out) + "\n")
    json.dump(dict(base=a.base, condition=a.condition, start=a.start, n=a.n, n_records=len(recs),
                   n_worker_errors=len(errs), host=socket.gethostname(), slurm=os.environ.get("SLURM_JOB_ID"),
                   diagnostic_only=True), open(a.out + ".prov.json", "w"), indent=1, default=str)
    ff_old = sum(1 for r in recs if r.get("old_false_confirm")); ff_b6 = sum(1 for r in recs if r.get("b6_false_confirm"))
    dts = [r.get("fidelity_dT") for r in recs if isinstance(r.get("fidelity_dT"), (int, float)) and r.get("fidelity_dT") == r.get("fidelity_dT")]
    n_nan_fid = sum(1 for r in recs if "fidelity_dT" in r and not (isinstance(r.get("fidelity_dT"), (int, float)) and r.get("fidelity_dT") == r.get("fidelity_dT")))
    print(f"[b6 {a.base}:{a.condition}:{a.start}+{a.n}] {len(recs)} recs, {len(errs)} err | "
          f"OLD false_confirm={ff_old} B6 false_confirm={ff_b6} | "
          f"max_fidelity_dT={(max(dts) if dts else float('nan')):.2e} (nan_fidelity={n_nan_fid})", flush=True)
    sys.exit(2 if errs else 0)
