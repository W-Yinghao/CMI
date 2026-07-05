"""CSC-realEEG-P3.0d ORACLE-GENERATOR fixed-margin diagnostic (development-only; diagnostic_only=true;
NOT a method, NOT confirmatory, no tag). For each cohort it recomputes the SAME B3 statistic path (same query,
folds, h0/h1 fits, subject-condition weighting, T, studentized stat, LCB, invalid accounting) but replaces ONLY
the null LABEL GENERATOR: instead of the fitted-h0 null it draws Y* from the bank's KNOWN oracle generator
p_oracle(Y|Z) = _pooled_clf(coh_Z, coh_y) (session-independent), conditioned on the SAME observed condition x
class margins (fixed-margin). The observed statistic is unchanged. A strict fidelity check asserts the re-derived
observed T equals the method's observed T (else STOP -- no silent approximation).

Reuses the BYTE-UNCHANGED certifier internals from paired_calibrated; changes no method code."""
import os, sys, json, argparse, hashlib, socket
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
REPO="/home/infres/yinwang/CMI_AAAI_csc"; sys.path.insert(0, REPO)
import numpy as np
from csc.mininfo import realeeg_engine as EG
from csc.mininfo import paired_calibrated as PC
from csc.mininfo.paired_certifier import CONCEPT_CONFIRMED
from csc.mininfo.paired_conditional_test import condition_code

MDIR=os.path.join(REPO,"csc/mininfo"); Lm=lambda n: json.load(open(os.path.join(MDIR,n)))
CIDX={"NULL_cov":1,"NULL_label":2,"NULL_cov_plus_label":3,"POS_concept":4,"POS_concept_plus_cov":5,"random_label_control":7}
NOCONCEPT={"NULL_cov","NULL_label","NULL_cov_plus_label","random_label_control"}
STRIDE=1_000_000; M=30; MIN_EP=8; RANK=3; C=0.5; NF=3; ALPHA_BUDGET=0.025
b3=Lm("realeeg_b3_manifest.json")
cache_path=Lm("realeeg_lee2019_cache_manifest.json")["provenance"]["cache_path"]
_npz=np.load(cache_path); CACHE=dict(Z=_npz["Z"],y=_npz["y"],subject=_npz["subject_id"],session=_npz["session_id"])
SUBJECTS=np.unique(CACHE["subject"])
def _fnan(x):
    try: return float(x)
    except Exception: return float("nan")


def _reconstruct(cond, r, base):
    seed=base+CIDX[cond]*STRIDE+r
    rng=np.random.default_rng(seed)
    subj=rng.choice(SUBJECTS, size=min(M,len(SUBJECTS)), replace=False)
    sel=np.isin(CACHE["subject"], subj)
    coh={k:CACHE[k][sel] for k in ("Z","y","subject","session")}
    clf_oracle=EG._pooled_clf(coh["Z"], coh["y"])              # bank generator p_oracle(Y|Z), session-independent
    Z,Y,D,G=EG.build_cohort(cond, coh, rng)                    # same (Z,Y,D,G) the certifier saw
    return seed, Z, Y, D, G, clf_oracle


def _oracle_diag(Z, Y, D, G, clf_oracle, seed, B):
    """Replicate paired_cv_test's fixed-margin path EXACTLY, swapping ONLY logp0 for the oracle generator's."""
    # ---- query the same eligible subset certify_paired_calibrated picks ----
    elig_all=PC.eligible_complete_pairs(D, G, MIN_EP)
    if len(elig_all) < 4: return dict(reason="too few eligible", fidelity_ok=False)
    rng=np.random.default_rng(seed)
    pick=rng.choice(np.array(sorted(elig_all)), size=min(M, len(elig_all)), replace=False)
    mask=np.isin(G, pick); Zq,Yq,Dq,Gq=Z[mask],Y[mask],D[mask],G[mask]
    # ---- paired_cv_test internals on the queried subset (byte-identical to method) ----
    elig=PC.eligible_complete_pairs(Dq, Gq, MIN_EP)
    if len(elig) < NF*2: return dict(reason="few eligible after query", fidelity_ok=False)
    mm=np.isin(Gq, elig); Z2,Y2,D2,g2=Zq[mm],Yq[mm],Dq[mm],Gq[mm]
    cl=np.array(sorted(np.unique(Y2)))
    folds,_=PC._make_folds(elig, NF, seed)
    prep=PC._prep_folds(Z2, D2, g2, folds, "centered", RANK, C)
    if prep is None: return dict(reason="fold prep degenerate", fidelity_ok=False)
    T, ok, deltas=PC._T_cv(prep, Y2, D2, g2, cl, C)
    if not ok: return dict(reason="observed cross-fit degenerate", fidelity_ok=False)
    st=PC._studentize(deltas); Z_obs=st["Z"]
    # ---- ORACLE logp0: p_oracle(Y|Z) from the bank generator, aligned to cl ----
    pr=np.clip(clf_oracle.predict_proba(Z2), 1e-12, 1.0)
    full=np.full((len(Z2), len(cl)), 1e-12); cli={c:j for j,c in enumerate(cl)}
    for j,c in enumerate(clf_oracle.classes_):
        if int(c) in cli: full[:, cli[int(c)]]=pr[:, j]
    logp0_or=np.log(full/full.sum(1, keepdims=True))
    y0_idx=np.searchsorted(cl, Y2)
    sampler_seed=int(seed+777); rng_fm=np.random.default_rng(sampler_seed)
    nsw=max(20*len(Y2), 300)
    def draw_or():
        ys=PC.sample_h0_fixed_condition_margins(logp0_or, D2, y0_idx, rng_fm, nsw)
        return cl[ys], False
    res=PC._bootstrap(prep, D2, g2, cl, C, T, Z_obs, draw_or, B, 0.20, 2)
    chk=PC.sample_h0_fixed_condition_margins(logp0_or, D2, y0_idx, np.random.default_rng(sampler_seed+1), nsw)
    margin_ok=all(np.array_equal(np.bincount(y0_idx[D2==c], minlength=len(cl)),
                                 np.bincount(chk[D2==c], minlength=len(cl))) for c in np.unique(D2))
    ghash=hashlib.sha256(np.concatenate([clf_oracle.coef_.ravel(), clf_oracle.intercept_.ravel()]).tobytes()).hexdigest()[:16]
    return dict(observed_T=float(T), studentized_stat=float(Z_obs), mean_delta=float(st["mean"]),
                se_delta=float(st["se"]), n_subject_deltas=int(st["S"]),
                oracle_fixed_margin_p=float(res["p"]), oracle_studentized_p=float(res["p_stud"]),
                oracle_null_mean_T=float(res["nmean"]), oracle_null_sd_T=float(res["nsd"]),
                oracle_studentized_null_mean=float(res["snmean"]), oracle_studentized_null_sd=float(res["snsd"]),
                oracle_invalid_boot=int(res["n_inv"]), oracle_sampler_failures=int(res["n_fail"]),
                oracle_estimable=bool(res["estimable"]), oracle_margin_preserved=bool(margin_ok),
                oracle_generator_spec=f"pooled_logistic C=0.5 on (coh_Z,coh_y); coef_hash={ghash}",
                fidelity_ok=True)


def run(cond, r, base, B):
    seed=base+CIDX[cond]*STRIDE+r
    tid=f"{base}:{CIDX[cond]:02d}:{cond}:{r:04d}"
    try:
        _, Z, Y, D, G, clf_oracle=_reconstruct(cond, r, base)
        # IN-PROCESS method certify (same node/process as the oracle -> method<->oracle perfectly comparable and
        # fidelity is exact; avoids cross-machine lbfgs(tol=1e-4) nondeterminism vs the SLURM-computed merged T).
        log=PC.certify_paired_calibrated(Z, Y, D, G, m=M, min_confirm_pairs=20, pair_integrity_min=0.95,
            min_epochs=MIN_EP, rank=RANK, C=C, n_folds=NF, n_boot=200, seed=seed, alpha_family=0.05,
            n_decision_budgets=2)
        _CC=CONCEPT_CONFIRMED
        import math
        mstate=str(log.get("state")); gt_noconcept=cond in NOCONCEPT
        mT=float(log.get("observed_T")); mZ=float(log.get("studentized_stat"))
        m_valid=bool(log.get("valid")) and not math.isnan(mT)
        if not m_valid:
            # the METHOD itself could not validly compute T (pair_integrity / min_pairs / a condition lacks
            # min_classes / degenerate cross-fit) -> the oracle diagnostic is N/A here; fidelity is trivially
            # satisfied (there is no observed statistic to match). NOT a fidelity break.
            return dict(task_id=tid, seed_block=base, condition=cond, condition_index=CIDX[cond], cohort=r,
                        seed=int(seed), ground_truth_noconcept=gt_noconcept, oracle_applicable=False,
                        method_state=mstate, method_false_confirm=False, method_true_confirm=False,
                        method_fixed_margin_p=_fnan(log.get("fixed_margin_null_p")),
                        method_studentized_p=_fnan(log.get("studentized_p_value")), method_observed_T=mT,
                        oracle_B=int(B), oracle_p_floor=1.0/(B+1), oracle_confirm=False, oracle_false_confirm=False,
                        oracle_true_confirm=False, fidelity_ok=True, fidelity_dT=None, method_invalid=True,
                        reason=str(log.get("reason")), diagnostic_only=True, not_used_for_certificate=True)
        od=_oracle_diag(Z, Y, D, G, clf_oracle, seed, B)
        if (not od.get("fidelity_ok")) or math.isnan(od.get("observed_T", float("nan"))):
            # method VALID but the oracle re-derivation degenerated -> genuine inconsistency: FAIL CLOSED.
            return dict(task_id=tid, seed_block=base, condition=cond, condition_index=CIDX[cond], cohort=r,
                        seed=int(seed), oracle_applicable=False, fidelity_ok=False,
                        reason=f"oracle-rederiv-invalid-where-method-valid:{od.get('reason')}", diagnostic_only=True)
        fid_T=abs(od["observed_T"]-mT)<1e-6; fid_Z=abs(od["studentized_stat"]-mZ)<1e-6
        lcb_budget=float(log.get("subject_consistency_lcb_budget"))
        n_q=int(log.get("n_queried_subjects", 0) or 0); n_elig=int(log.get("n_eligible_queried", 0) or 0)
        size_ok=(n_q>=20) and (n_elig>=20)
        oracle_confirm=bool(od["oracle_fixed_margin_p"]<=ALPHA_BUDGET and od["oracle_studentized_p"]<=ALPHA_BUDGET
                            and lcb_budget>0 and size_ok)
        rec=dict(task_id=tid, seed_block=base, condition=cond, condition_index=CIDX[cond], cohort=r, seed=int(seed),
                 ground_truth_noconcept=gt_noconcept, oracle_applicable=True,
                 method_state=mstate, method_false_confirm=bool(mstate==_CC and gt_noconcept),
                 method_true_confirm=bool(mstate==_CC and not gt_noconcept),
                 method_fixed_margin_p=float(log.get("fixed_margin_null_p")),
                 method_studentized_p=float(log.get("studentized_p_value")), method_observed_T=mT,
                 oracle_B=int(B), oracle_p_floor=1.0/(B+1), oracle_confirm=oracle_confirm,
                 oracle_false_confirm=bool(oracle_confirm and gt_noconcept),
                 oracle_true_confirm=bool(oracle_confirm and not gt_noconcept),
                 lcb_budget=lcb_budget, size_ok=size_ok, fidelity_ok=bool(fid_T and fid_Z),
                 fidelity_dT=float(abs(od["observed_T"]-mT)),
                 diagnostic_only=True, not_used_for_certificate=True,
                 **{k:od[k] for k in od if k not in ("fidelity_ok",)})
        return rec
    except Exception as e:
        return dict(task_id=tid, condition=cond, seed=int(seed), __worker_error__=f"{type(e).__name__}: {str(e)[:300]}")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", type=int, required=True); ap.add_argument("--condition", type=str, required=True)
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--B", type=int, default=200); ap.add_argument("--cohorts", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    a=ap.parse_args(); assert a.condition in CIDX
    from joblib import Parallel, delayed
    idxs=[int(x) for x in a.cohorts.split(",")] if a.cohorts else list(range(a.n))
    recs=Parallel(n_jobs=a.jobs, backend="loky")(delayed(run)(a.condition, r, a.base, a.B) for r in idxs)
    errs=[x for x in recs if "__worker_error__" in x]; fidbad=[x for x in recs if x.get("fidelity_ok") is False]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out,"w") as f:
        for x in recs: f.write(json.dumps(x, default=str)+"\n")
    with open(a.out+".sha256","w") as f:
        f.write(hashlib.sha256(open(a.out,"rb").read()).hexdigest()+"  "+os.path.basename(a.out)+"\n")
    json.dump(dict(shard=f"{a.base}:{a.condition}", n=len(recs), B=a.B, n_worker_errors=len(errs),
                   n_fidelity_fail=len(fidbad), host=socket.gethostname(), slurm=os.environ.get("SLURM_JOB_ID"),
                   diagnostic_only=True), open(a.out+".prov.json","w"), indent=1, default=str)
    print(f"[oracle {a.base}:{a.condition} B={a.B}] {len(recs)} recs, {len(errs)} errors, {len(fidbad)} fidelity-fail, "
          f"oracle_FC={sum(1 for x in recs if x.get('oracle_false_confirm'))} "
          f"oracle_TC={sum(1 for x in recs if x.get('oracle_true_confirm'))}", flush=True)
    sys.exit(2 if (errs or fidbad) else 0)   # fail-closed: worker error OR fidelity break => infra failure
