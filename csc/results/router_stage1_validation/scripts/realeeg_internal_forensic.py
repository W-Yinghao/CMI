"""CSC-realEEG-P3.0b/c certifier-INTERNAL forensic worker (ONE shard = one seed_block x one condition = 100
cohorts). DEVELOPMENT forensic: NO method change, NO freeze, NOT confirmatory, NOT merged with csc-realeeg-v2.
Reconstructs each cohort deterministically (seed = base + condition_index*stride + cohort_index) and calls the
BYTE-UNCHANGED certifier `certify_paired_calibrated` directly to capture its full internal log (T, fixed-margin
/ standard nulls, studentized stat, subject deltas, null mean/sd), plus overlap + subject-dominance diagnostics.
Writes a JSONL partial + sha256 + provenance; fail-closed on any worker error."""
import os, sys, json, argparse, hashlib, socket
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
REPO = "/home/infres/yinwang/CMI_AAAI_csc"; sys.path.insert(0, REPO)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from csc.mininfo import realeeg_engine as EG
from csc.mininfo.paired_calibrated import certify_paired_calibrated
from csc.mininfo.paired_certifier import CONCEPT_CONFIRMED

MDIR = os.path.join(REPO, "csc/mininfo")
L = lambda n: json.load(open(os.path.join(MDIR, n)))
CIDX = {"NULL_cov":1, "NULL_label":2, "NULL_cov_plus_label":3, "POS_concept":4,
        "POS_concept_plus_cov":5, "random_label_control":7}
NOCONCEPT = {"NULL_cov","NULL_label","NULL_cov_plus_label","random_label_control"}   # ground truth NO_CONCEPT
STRIDE = 1_000_000; M = 30

b3 = L("realeeg_b3_manifest.json"); mlk = b3["method_lock"]
ML = dict(min_confirm_pairs=20, pair_integrity_min=0.95, min_epochs=8, rank=3,
          C=mlk["regularisation_C"], n_folds=mlk["n_folds"],
          n_boot=b3["statistics"]["b_certifier_internal_null"], alpha_family=b3["statistics"]["alpha_family"],
          n_decision_budgets=len(mlk["positive_decision_budgets"]))
cache_path = L("realeeg_lee2019_cache_manifest.json")["provenance"]["cache_path"]
_npz = np.load(cache_path)
CACHE = dict(Z=_npz["Z"], y=_npz["y"], subject=_npz["subject_id"], session=_npz["session_id"])
SUBJECTS = np.unique(CACHE["subject"])


def overlap_diag(Z, D, G):
    D2 = (D == 2).astype(int); n = len(D)
    if len(np.unique(D2)) < 2:
        return dict(session_auc=float("nan"), overlap_frac=float("nan"), ess_frac=float("nan"), max_w_ratio=float("nan"))
    e = np.zeros(n); Zs = StandardScaler().fit_transform(Z)
    ng = len(np.unique(G)); ns = min(5, ng)
    if ns < 2:
        e[:] = D2.mean()
    else:
        for tr, te in GroupKFold(n_splits=ns).split(Zs, D2, groups=G):
            if len(np.unique(D2[tr])) < 2: e[te] = D2[tr].mean(); continue
            e[te] = LogisticRegression(C=1.0, max_iter=1000).fit(Zs[tr], D2[tr]).predict_proba(Zs[te])[:, 1]
    e = np.clip(e, 1e-6, 1-1e-6)
    auc = roc_auc_score(D2, e) if len(np.unique(D2)) == 2 else float("nan")
    w = np.where(D2 == 1, 1-e, e)
    return dict(session_auc=float(auc), overlap_frac=float(np.mean((e >= .1) & (e <= .9))),
                ess_frac=float(w.sum()**2/(w**2).sum()/n), max_w_ratio=float(w.max()/w.mean()))


def subject_dominance(deltas):
    v = np.asarray(deltas, float)
    if v.size == 0: return dict(frac_delta_pos=float("nan"), top1_abs_share=float("nan"),
                                top3_abs_share=float("nan"), gini_abs=float("nan"))
    a = np.abs(v); tot = a.sum() + 1e-12; s = np.sort(a)[::-1]
    gini = (2*np.arange(1, len(a)+1) - len(a) - 1) @ np.sort(a) / (len(a)*a.sum()+1e-12)
    return dict(frac_delta_pos=float(np.mean(v > 0)), top1_abs_share=float(s[0]/tot),
                top3_abs_share=float(s[:3].sum()/tot), gini_abs=float(gini))


def run_cohort(cond, r, base):
    seed = base + CIDX[cond]*STRIDE + r
    try:
        rng = np.random.default_rng(seed)
        subj = rng.choice(SUBJECTS, size=min(M, len(SUBJECTS)), replace=False)
        sel = np.isin(CACHE["subject"], subj)
        coh = {k: CACHE[k][sel] for k in ("Z","y","subject","session")}
        Z, Y, D, G = EG.build_cohort(cond, coh, rng)
        log = certify_paired_calibrated(Z, Y, D, G, m=len(subj), min_confirm_pairs=ML["min_confirm_pairs"],
              pair_integrity_min=ML["pair_integrity_min"], min_epochs=ML["min_epochs"], rank=ML["rank"],
              C=ML["C"], n_folds=ML["n_folds"], n_boot=ML["n_boot"], seed=seed,
              alpha_family=ML["alpha_family"], n_decision_budgets=ML["n_decision_budgets"])
        state = str(log.get("state"))
        gt_noconcept = cond in NOCONCEPT
        deltas = log.get("delta_subjects", []) or []
        nm, nsd = log.get("null_mean", float("nan")), log.get("null_sd", float("nan"))
        T = log.get("observed_T", float("nan"))
        rec = dict(seed_block=base, condition=cond, condition_index=CIDX[cond], cohort=r, seed=int(seed),
                   task_id=f"{base}:{CIDX[cond]:02d}:{cond}:{r:04d}",
                   b3_state=state, false_confirm=bool(state == CONCEPT_CONFIRMED and gt_noconcept),
                   true_confirm=bool(state == CONCEPT_CONFIRMED and not gt_noconcept),
                   ground_truth_noconcept=gt_noconcept,
                   # --- certifier-internal (from the byte-unchanged log) ---
                   observed_T=_f(T), fixed_margin_p=_f(log.get("fixed_margin_null_p")),
                   standard_null_p=_f(log.get("standard_null_p")),  # DIAGNOSTIC non-fixed-margin null
                   would_confirm_under_standard_null=bool(log.get("would_confirm_under_standard_null")),
                   studentized_p=_f(log.get("studentized_p_value")), studentized_stat=_f(log.get("studentized_stat")),
                   subject_consistency_lcb=_f(log.get("subject_consistency_lcb")),
                   lcb_budget=_f(log.get("subject_consistency_lcb_budget")),
                   mean_delta=_f(log.get("mean_delta")), sd_delta=_f(log.get("sd_delta")),
                   se_delta=_f(log.get("se_delta")), n_subject_deltas=int(log.get("n_subject_deltas", 0) or 0),
                   null_mean_T=_f(nm), null_sd_T=_f(nsd),
                   studentized_null_mean=_f(log.get("studentized_null_mean")),
                   studentized_null_sd=_f(log.get("studentized_null_sd")),
                   n_boot_invalid=int(log.get("n_boot_invalid", 0) or 0),
                   n_sampler_failures=int(log.get("n_sampler_failures", 0) or 0),
                   margin_preserved=bool(log.get("margin_preserved", False)), valid=bool(log.get("valid", False)),
                   n_eligible_queried=int(log.get("n_eligible_queried", 0) or 0),
                   pair_integrity=_f(log.get("pair_integrity")), fold_hash=log.get("fold_hash"),
                   # --- derived null-calibration signature ---
                   T_minus_nullmean=_f(T - nm) if _ok(T) and _ok(nm) else float("nan"),
                   T_z=_f((T - nm)/nsd) if _ok(T) and _ok(nm) and _ok(nsd) and nsd > 0 else float("nan"),
                   delta_subjects=[float(x) for x in deltas],
                   diagnostic_only=True, not_used_for_certificate=True)
        rec.update(subject_dominance(deltas))
        rec.update(overlap_diag(Z, D, G))
        return rec
    except Exception as e:
        return dict(task_id=f"{base}:{CIDX.get(cond,-1)}:{cond}:{r:04d}",
                    __worker_error__=f"{type(e).__name__}: {str(e)[:300]}")


def _ok(x):
    try: return x == x and abs(x) != float("inf")
    except Exception: return False
def _f(x):
    try: return float(x)
    except Exception: return float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, required=True)
    ap.add_argument("--condition", type=str, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    assert a.condition in CIDX, f"unknown condition {a.condition}"
    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=a.jobs, backend="loky")(delayed(run_cohort)(a.condition, r, a.base) for r in range(a.n))
    errs = [r for r in recs if "__worker_error__" in r]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs: f.write(json.dumps(r, default=str) + "\n")
    with open(a.out + ".sha256", "w") as f:
        f.write(hashlib.sha256(open(a.out, "rb").read()).hexdigest() + "  " + os.path.basename(a.out) + "\n")
    prov = dict(shard=f"{a.base}:{a.condition}", base=a.base, condition=a.condition, n=a.n, jobs=a.jobs,
                host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                n_records=len(recs), n_worker_errors=len(errs),
                cache_sha256=hashlib.sha256(open(cache_path, "rb").read()).hexdigest(),
                engine_sha256=hashlib.sha256(open(os.path.join(MDIR,"realeeg_engine.py"),"rb").read()).hexdigest(),
                b3_manifest_sha256=hashlib.sha256(open(os.path.join(MDIR,"realeeg_b3_manifest.json"),"rb").read()).hexdigest(),
                ml=ML, diagnostic_only=True)
    json.dump(prov, open(a.out + ".prov.json", "w"), indent=1, default=str)
    print(f"[shard {a.base}:{a.condition}] {len(recs)} records, {len(errs)} errors, "
          f"{sum(1 for r in recs if r.get('false_confirm'))} false-confirm, "
          f"{sum(1 for r in recs if r.get('true_confirm'))} true-confirm -> {a.out}", flush=True)
    sys.exit(2 if errs else 0)   # fail-closed: any worker error => infra failure exit 2
