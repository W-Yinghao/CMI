"""CSC B6-FM canary worker (development-only, NO tag). Reuses the EXACT B6.0 cohorts (base 200e6, same condition
indices) so B6-FM vs B6.0 differ ONLY in the null construction (plain C-randomization vs class-preserving/fixed-margin
C-randomization) -- directly comparable, same observed_T. Per cohort runs the OLD certifier (baseline comparator) +
crt_test_fm (the fixed-margin C-null) across the same 8 conditions incl the two strong-cov levels. Fail-closed."""
import os, sys, json, argparse, hashlib, socket
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
import numpy as np
import realeeg_internal_forensic as IF
import realeeg_strongcov as SC
import realeeg_b6_canary as B6          # reuse CMAP, NOCONCEPT, STRIDE, M (identical cohort construction)
from csc.mininfo import realeeg_engine as EG
from csc.mininfo.paired_calibrated import certify_paired_calibrated
from csc.mininfo.paired_certifier import CONCEPT_CONFIRMED
import realeeg_crt_fm as FM

CMAP = B6.CMAP; NOCONCEPT = B6.NOCONCEPT; STRIDE = B6.STRIDE; M = B6.M


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
        log = certify_paired_calibrated(Z, Y, D, G, m=len(subj), min_confirm_pairs=IF.ML["min_confirm_pairs"],
              pair_integrity_min=IF.ML["pair_integrity_min"], min_epochs=IF.ML["min_epochs"], rank=IF.ML["rank"],
              C=IF.ML["C"], n_folds=IF.ML["n_folds"], n_boot=IF.ML["n_boot"], seed=seed,
              alpha_family=IF.ML["alpha_family"], n_decision_budgets=IF.ML["n_decision_budgets"])
        old_state = str(log.get("state")); T_old = log.get("observed_T", float("nan"))
        fm = FM.crt_test_fm(Z, Y, D, G, m=len(subj), seed=seed, rank=IF.ML["rank"], C=IF.ML["C"],
                            n_folds=IF.ML["n_folds"], min_epochs=IF.ML["min_epochs"],
                            min_confirm_pairs=IF.ML["min_confirm_pairs"], n_boot=IF.ML["n_boot"])
        T_crt = fm.get("observed_T_crt", float("nan"))
        fidelity_dT = _f(abs(T_old - T_crt)) if (T_old == T_old and T_crt == T_crt) else float("nan")
        st = str(fm.get("fm_state"))
        rec = dict(condition=cond_name, condition_index=cidx, cohort=r, seed=int(seed), delta=delta,
                   task_id=f"{base}:{cidx:02d}:{cond_name}:{r:04d}", ground_truth_noconcept=gt_noconcept,
                   old_b3_state=old_state, observed_T=_f(T_old), old_fixed_margin_p=_f(log.get("fixed_margin_null_p")),
                   old_false_confirm=bool(old_state == CONCEPT_CONFIRMED and gt_noconcept),
                   old_true_confirm=bool(old_state == CONCEPT_CONFIRMED and not gt_noconcept),
                   observed_T_crt=_f(T_crt), fidelity_dT=fidelity_dT,
                   fm_state=st, fm_valid=bool(fm.get("fm_valid")),
                   fm_confirm=bool(st == "CONCEPT_CONFIRMED"),
                   fm_false_confirm=bool(st == "CONCEPT_CONFIRMED" and gt_noconcept),
                   fm_true_confirm=bool(st == "CONCEPT_CONFIRMED" and not gt_noconcept),
                   fm_margin_lock=bool(st == "UNIDENTIFIABLE_MARGIN_LOCK"),
                   fm_sampler_invalid=bool(st == "SAMPLER_INVALID"),
                   p_C_FM_meanT=_f(fm.get("p_C_FM_meanT")), p_C_FM_stud=_f(fm.get("p_C_FM_stud")),
                   c_null_mean_T=_f(fm.get("c_null_mean_T")), c_null_sd_T=_f(fm.get("c_null_sd_T")),
                   n_crt_invalid=int(fm.get("n_crt_invalid", 0) or 0),
                   # margin diagnostics (first-class)
                   propensity_auc=_f(fm.get("propensity_auc")),
                   margin_feasible_swaps=_f(fm.get("margin_feasible_swaps")),
                   frac_strata_single_condition=_f(fm.get("frac_strata_single_condition")),
                   n_strata=int(fm.get("n_strata", 0) or 0), unique_Cstar=int(fm.get("unique_Cstar", 0) or 0),
                   margin_fidelity_max_err=_f(fm.get("margin_fidelity_max_err")),
                   max_subject_count_err=_f(fm.get("max_subject_count_err")),
                   resampled_Cstar_auc=_f(fm.get("resampled_Cstar_auc")),
                   covariate_auc_gap=_f(fm.get("covariate_auc_gap")),
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
    ff_old = sum(1 for r in recs if r.get("old_false_confirm")); ff_fm = sum(1 for r in recs if r.get("fm_false_confirm"))
    lock = sum(1 for r in recs if r.get("fm_margin_lock")); inv = sum(1 for r in recs if r.get("fm_sampler_invalid"))
    mfe = max([r.get("margin_fidelity_max_err", 0) or 0 for r in recs] + [0])
    print(f"[b6fm {a.base}:{a.condition}:{a.start}+{a.n}] {len(recs)} recs, {len(errs)} err | "
          f"OLD ff={ff_old} FM ff={ff_fm} lock={lock} invalid={inv} | max_margin_err={mfe:.0f}", flush=True)
    sys.exit(2 if errs else 0)
