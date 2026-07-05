"""CSC-realEEG B4 Stage 1 canary worker (development-only; diagnostic; NOT deployable; NOT confirmatory).
Changes ONLY the NULL generation for the EXISTING B3 paired statistic. The observed T / studentized stat / LCB /
folds / features / condition-class margins / invalid accounting are UNCHANGED and computed ONCE, then shared
across every null arm -- so a B4 null arm provably cannot alter the observed statistic. Arms:
  - method  : the existing fixed-margin full-audit-h0 null (on-node baseline).
  - B4a     : BAGGED fixed-margin null -- K subject-cluster-bootstrap h0 fits, rotated per replicate (injects
              nuisance/estimation uncertainty into the null so its dispersion widens toward the oracle).
  - B4b     : NESTED fixed-margin null -- a FRESH subject-cluster-bootstrap h0 fit per replicate (K=B).
  - baseline: variance-inflation diagnostic (oracle/method null_sd ratio; method p under inflation) -- NOT a
              candidate method, reported only.
No overlap gate, no richer h1, no feature/montage change, no new statistic, no oracle generator in any arm."""
import os, sys, json, argparse, hashlib, socket, math
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
REPO="/home/infres/yinwang/CMI_AAAI_csc"; sys.path.insert(0, REPO)
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
import numpy as np
import realeeg_oracle_forensic as OF                       # _reconstruct, CACHE, SUBJECTS, M, ML
from csc.mininfo import paired_calibrated as PC
from csc.mininfo.paired_conditional_test import condition_code
CODING="centered"; RANK=3; C=0.5; NF=3; MIN_EP=8; A=0.025; M=OF.M


def _observed(Z, Y, D, G, seed):
    """Reproduce paired_cv_test's OBSERVED side EXACTLY (query, elig, folds, prep, T, studentized). Returns the
    shared observed statistic + the fold prep used by every null arm. None if the method itself is degenerate."""
    elig_all=PC.eligible_complete_pairs(D, G, MIN_EP)
    if len(elig_all)<4: return None
    rng=np.random.default_rng(seed)
    pick=rng.choice(np.array(sorted(elig_all)), size=min(M,len(elig_all)), replace=False)
    mask=np.isin(G, pick); Zq,Yq,Dq,Gq=Z[mask],Y[mask],D[mask],G[mask]
    elig=PC.eligible_complete_pairs(Dq, Gq, MIN_EP)
    if len(elig)<NF*2: return None
    mm=np.isin(Gq, elig); Z2,Y2,D2,g2=Zq[mm],Yq[mm],Dq[mm],Gq[mm]
    pcc=PC.per_condition_classes(Y2, D2)
    if any(len(v)<2 for v in pcc.values()): return None
    cl=np.array(sorted(np.unique(Y2))); folds,_=PC._make_folds(elig, NF, seed)
    prep=PC._prep_folds(Z2, D2, g2, folds, CODING, RANK, C)
    if prep is None: return None
    T,ok,deltas=PC._T_cv(prep, Y2, D2, g2, cl, C)
    if not ok: return None
    st=PC._studentize(deltas)
    return dict(prep=prep, T=float(T), Z_obs=float(st["Z"]), cl=cl, Z2=Z2, Y2=Y2, D2=D2, g2=g2,
                mean_delta=float(st["mean"]), se_delta=float(st["se"]), S=int(st["S"]), n_elig=len(elig))


def _h0_logp_fit_eval(Zf, Yf, Df, gf, Ze, De, cl, C):
    """Fit a class-balanced h0=[Z,c] on (Zf..) and return log p0(Y|Z,c) evaluated on (Ze,De), aligned to cl."""
    w=PC.class_balanced_weights(Yf, Df, gf);
    if w.sum()<=0: w=np.ones(len(Yf))
    W=w.sum(); mu=(w[:,None]*Zf).sum(0)/W; sd=np.sqrt(np.clip((w[:,None]*(Zf-mu)**2).sum(0)/W,0,None))+1e-8
    Zsf=(Zf-mu)/sd; cvf=condition_code(Df,CODING)[:,None]
    h0=PC._fit(np.hstack([Zsf,cvf]), Yf, C, w)
    Zse=(Ze-mu)/sd; cve=condition_code(De,CODING)[:,None]
    pr=np.clip(h0.predict_proba(np.hstack([Zse,cve])),1e-12,1.0)
    full=np.full((len(Ze),len(cl)),1e-12); cli={c:j for j,c in enumerate(cl)}
    for j,c in enumerate(h0.classes_):
        if int(c) in cli: full[:,cli[int(c)]]=pr[:,j]
    return np.log(full/full.sum(1,keepdims=True))


def _bagged_null(o, C, seed, B, K_bag, fallback_logp0):
    """Fixed-margin null with a BAGGED / NESTED h0 generator (K_bag subject-cluster bootstraps of the queried
    data). K_bag<B -> pre-fit K bags, rotate per replicate (B4a). K_bag>=B -> a fresh bag per replicate (B4b).
    Observed T/Z_obs are the SHARED observed statistic -- unchanged. Same margins/folds/statistic path."""
    Z2,Y2,D2,g2,cl,prep,T,Z_obs=o["Z2"],o["Y2"],o["D2"],o["g2"],o["cl"],o["prep"],o["T"],o["Z_obs"]
    subjects=np.unique(g2); y0=np.searchsorted(cl,Y2); nsw=max(20*len(Y2),300)
    rb=np.random.default_rng(seed+5555); rf=np.random.default_rng(seed+777)
    nbags=B if K_bag>=B else K_bag; bags=[]
    for k in range(nbags):
        bs=rb.choice(subjects, size=len(subjects), replace=True)
        parts=[np.where(g2==s)[0] for s in bs]; idx=np.concatenate(parts)
        gnew=np.concatenate([np.full(len(parts[i]), i) for i in range(len(parts))])
        try:
            lp=_h0_logp_fit_eval(Z2[idx],Y2[idx],D2[idx],gnew,Z2,D2,cl,C)
            if any(len(np.unique(Y2[idx][D2[idx]==c]))<2 for c in np.unique(D2[idx])): lp=fallback_logp0
        except Exception:
            lp=fallback_logp0
        bags.append(lp)
    ge=ge_s=1; ninv=0; ts=[]; zs=[]
    for b in range(B):
        lp=bags[b] if K_bag>=B else bags[b%K_bag]
        ys=PC.sample_h0_fixed_condition_margins(lp, D2, y0, rf, nsw); Ystar=cl[ys]
        if any(len(np.unique(Ystar[D2==c]))<2 for c in np.unique(D2)): ninv+=1;ge+=1;ge_s+=1;continue
        Ts,ok,ds=PC._T_cv(prep, Ystar, D2, g2, cl, C)
        if not ok: ninv+=1;ge+=1;ge_s+=1;continue
        Zst=PC._studentize(ds)["Z"]; ts.append(Ts); zs.append(Zst); ge+=int(Ts>=T); ge_s+=int(Zst>=Z_obs)
    return dict(p=ge/(B+1), p_stud=ge_s/(B+1), n_inv=ninv,
                null_sd=float(np.std(ts)) if ts else float("nan"),
                stud_null_mean=float(np.mean(zs)) if zs else float("nan"))


def run(spec, B, Ka, Kb):
    cond=spec["condition"]; r=spec["cohort"]; base=spec["seed_block"]; seed=spec["seed"]; tid=spec["task_id"]
    try:
        _,Z,Y,D,G,_=OF._reconstruct(cond, r, base)
        # in-process method certify: method p's + on-node observed T (fidelity reference, dT=0)
        log=PC.certify_paired_calibrated(Z,Y,D,G, m=M, min_confirm_pairs=20, pair_integrity_min=0.95,
            min_epochs=MIN_EP, rank=RANK, C=C, n_folds=NF, n_boot=B, seed=seed, alpha_family=0.05, n_decision_budgets=2)
        mstate=str(log.get("state")); mT=float(log.get("observed_T")); mZ=float(log.get("studentized_stat"))
        if not (bool(log.get("valid")) and not math.isnan(mT)):
            return dict(task_id=tid, condition=cond, stratum=spec["stratum"], seed=int(seed),
                        method_invalid=True, applicable=False, method_state=mstate, diagnostic_only=True)
        o=_observed(Z,Y,D,G,seed)
        if o is None or abs(o["T"]-mT)>1e-6:   # HARD STOP: B4 must not alter the observed statistic
            return dict(task_id=tid, condition=cond, seed=int(seed), fidelity_ok=False,
                        fidelity_dT=(abs(o["T"]-mT) if o else None), diagnostic_only=True)
        lcb=float(log.get("subject_consistency_lcb_budget")); nq=int(log.get("n_queried_subjects",0) or 0)
        ne=int(log.get("n_eligible_queried",0) or 0); size_ok=(nq>=20 and ne>=20)
        # shared full-audit fallback logp0 (used if a bootstrap bag degenerates)
        fb=PC._h0_full_logp(o["Z2"], o["Y2"], o["D2"], o["g2"], o["cl"], CODING, C)
        b4a=_bagged_null(o, C, seed, B, Ka, fb)
        b4b=_bagged_null(o, C, seed, B, Kb, fb)
        def confirm(p, ps): return bool(p<=A and ps<=A and lcb>0 and size_ok)
        m_fmp=float(log.get("fixed_margin_null_p")); m_stp=float(log.get("studentized_p_value")); m_sd=float(log.get("null_sd"))
        o_sd=spec.get("archived_oracle_null_sd_T")
        rec=dict(task_id=tid, condition=cond, seed_block=base, cohort=r, seed=int(seed), stratum=spec["stratum"],
                 ground_truth_noconcept=(cond in OF.NOCONCEPT), method_state=mstate,
                 method_fixed_margin_p=m_fmp, method_studentized_p=m_stp, method_null_sd=m_sd,
                 method_confirm=confirm(m_fmp, m_stp),
                 b4a_fixed_margin_p=b4a["p"], b4a_studentized_p=b4a["p_stud"], b4a_null_sd=b4a["null_sd"],
                 b4a_n_inv=b4a["n_inv"], b4a_confirm=confirm(b4a["p"], b4a["p_stud"]),
                 b4b_fixed_margin_p=b4b["p"], b4b_studentized_p=b4b["p_stud"], b4b_null_sd=b4b["null_sd"],
                 b4b_n_inv=b4b["n_inv"], b4b_confirm=confirm(b4b["p"], b4b["p_stud"]),
                 archived_oracle_fixed_margin_p=spec.get("archived_oracle_fixed_margin_p"),
                 archived_oracle_studentized_p=spec.get("archived_oracle_studentized_p"),
                 archived_oracle_null_sd_T=o_sd, archived_oracle_confirm=spec.get("oracle_confirm"),
                 baseline_inflation_ratio_oracle_over_method=(o_sd/m_sd if (o_sd and m_sd) else None),
                 observed_T=o["T"], studentized_stat=o["Z_obs"], lcb_budget=lcb, size_ok=size_ok,
                 fidelity_dT=abs(o["T"]-mT), fidelity_ok=True, K_bag_a=Ka, K_bag_b=Kb, B=B,
                 diagnostic_only=True, not_confirmatory=True, not_deployable=True, no_method_change=True)
        return rec
    except Exception as e:
        return dict(task_id=tid, condition=cond, seed=int(seed), __worker_error__=f"{type(e).__name__}: {str(e)[:300]}")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--slice", default=None)  # "start:end"
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--Ka", type=int, default=50); ap.add_argument("--Kb", type=int, default=200)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    man=json.load(open(a.manifest)); specs=man["cohorts"]
    if a.slice:
        s,e=a.slice.split(":"); specs=specs[int(s):int(e)]
    from joblib import Parallel, delayed
    recs=Parallel(n_jobs=a.jobs, backend="loky")(delayed(run)(sp, a.B, a.Ka, a.Kb) for sp in specs)
    errs=[x for x in recs if "__worker_error__" in x]; fid=[x for x in recs if x.get("fidelity_ok") is False]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out,"w") as f:
        for x in recs: f.write(json.dumps(x, default=str)+"\n")
    open(a.out+".sha256","w").write(hashlib.sha256(open(a.out,"rb").read()).hexdigest()+"  "+os.path.basename(a.out)+"\n")
    json.dump(dict(n=len(recs), B=a.B, Ka=a.Ka, Kb=a.Kb, n_worker_errors=len(errs), n_fidelity_fail=len(fid),
                   host=socket.gethostname(), slurm=os.environ.get("SLURM_JOB_ID"), diagnostic_only=True),
              open(a.out+".prov.json","w"), indent=1, default=str)
    print(f"[b4-stage1 slice={a.slice}] {len(recs)} recs, {len(errs)} errors, {len(fid)} fidelity-fail", flush=True)
    sys.exit(2 if (errs or fid) else 0)
