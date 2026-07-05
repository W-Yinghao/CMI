"""CSC-realEEG B4c Stage-1b canary worker (development-only; diagnostic; NOT deployable; NOT confirmatory).
Tests ONE change: a RICHER SHARED NUISANCE TRUNK f(Z) used symmetrically in h0 and h1. The B3 interaction
channel (c x PC1:3) is UNCHANGED, the estimand is the same contrast CE(h0)-CE(h1) subject-vote, the SM16
features / paired structure / folds / condition-class margin preservation / invalid accounting are unchanged.
Only the shared trunk gains fixed quadratic PC terms (u_i^2, u_i u_j of the SAME top-3 PC coords), in BOTH h0
and h1. C=0.25 predeclared. No overlap gate, no richer interaction channel, no new features/montage, no oracle
generator in fitting or null. Because the shared family changes, the observed T MAY change (logged; the old
fidelity_dT=0 invariant does NOT apply -- replaced by B4c observed-T reproducibility)."""
import os, sys, json, argparse, hashlib, socket
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
REPO="/home/infres/yinwang/CMI_AAAI_csc"; sys.path.insert(0, REPO); sys.path.insert(0,"/home/infres/yinwang/realeeg_feas")
import numpy as np
import realeeg_oracle_forensic as OF
from csc.mininfo import paired_calibrated as PC
from csc.mininfo.paired_conditional_test import condition_code
CODING="centered"; RANK=3; C_B4C=0.25; NF=3; MIN_EP=8; A=0.025; M=OF.M
# archived Stage-1a fields (method/B4a/B4b null_sd, B3 observed T) + manifest oracle, keyed by task_id
_S1=json.load(open("/home/infres/yinwang/realeeg_feas/b4_stage1/b4_stage1_merged.json"))["per_cohort"]
S1={r["task_id"]: r for r in _S1}


def _features_b4c(Z, D, mu, sd, Vr):
    """Shared trunk = [Zs(16), c, u1^2,u2^2,u3^2, u1u2,u1u3,u2u3]; h1 adds the UNCHANGED interaction c x PC1:3."""
    Zs=(Z-mu)/sd; c=condition_code(D,CODING)[:,None]; U=Zs@Vr                 # U = top-3 PC coords
    quad=np.column_stack([U[:,0]**2,U[:,1]**2,U[:,2]**2, U[:,0]*U[:,1],U[:,0]*U[:,2],U[:,1]*U[:,2]])
    X0=np.hstack([Zs, c, quad])                                              # richer SHARED trunk
    X1=np.hstack([Zs, c, quad, c*U])                                         # + c x PC1:3 (unchanged channel)
    return X0, X1


def _prep_b4c(Z, D, g, folds):
    prep=[]
    for f in folds:
        ho=np.isin(g,f); tr=~ho
        if tr.sum()==0 or ho.sum()==0: return None
        wz=PC.class_balanced_weights(np.zeros(tr.sum()), D[tr], g[tr]); wz=wz if wz.sum()>0 else np.ones(tr.sum())
        W=wz.sum(); mu=(wz[:,None]*Z[tr]).sum(0)/W
        sd=np.sqrt(np.clip((wz[:,None]*(Z[tr]-mu)**2).sum(0)/W,0,None))+1e-8
        Zs_tr=(Z[tr]-mu)/sd; Vr=np.linalg.svd(np.sqrt(wz)[:,None]*Zs_tr, full_matrices=False)[2][:RANK].T
        X0_tr,X1_tr=_features_b4c(Z[tr],D[tr],mu,sd,Vr); X0_ho,X1_ho=_features_b4c(Z[ho],D[ho],mu,sd,Vr)
        prep.append(dict(tr=tr,ho=ho,X0_tr=X0_tr,X1_tr=X1_tr,X0_ho=X0_ho,X1_ho=X1_ho))
    return prep


def _h0_logp_b4c(Z, Y, D, g, cl):
    w=PC.class_balanced_weights(Y,D,g);
    if w.sum()<=0: w=np.ones(len(Y))
    W=w.sum(); mu=(w[:,None]*Z).sum(0)/W; sd=np.sqrt(np.clip((w[:,None]*(Z-mu)**2).sum(0)/W,0,None))+1e-8
    Zs=(Z-mu)/sd; Vr=np.linalg.svd(np.sqrt(w)[:,None]*Zs, full_matrices=False)[2][:RANK].T
    X0,_=_features_b4c(Z,D,mu,sd,Vr); h0=PC._fit(X0,Y,C_B4C,w)
    pr=np.clip(h0.predict_proba(X0),1e-12,1.0); full=np.full((len(Z),len(cl)),1e-12); cli={c:j for j,c in enumerate(cl)}
    for j,c in enumerate(h0.classes_):
        if int(c) in cli: full[:,cli[int(c)]]=pr[:,j]
    return np.log(full/full.sum(1,keepdims=True))


def _b4c(Z, Y, D, G, seed, B):
    """B4c-Q3 paired conditional-change test (richer shared trunk). Returns observed T + fixed-margin null."""
    elig_all=PC.eligible_complete_pairs(D,G,MIN_EP)
    if len(elig_all)<4: return None
    rng=np.random.default_rng(seed); pick=rng.choice(np.array(sorted(elig_all)), size=min(M,len(elig_all)), replace=False)
    mask=np.isin(G,pick); Zq,Yq,Dq,Gq=Z[mask],Y[mask],D[mask],G[mask]
    elig=PC.eligible_complete_pairs(Dq,Gq,MIN_EP)
    if len(elig)<NF*2: return None
    mm=np.isin(Gq,elig); Z2,Y2,D2,g2=Zq[mm],Yq[mm],Dq[mm],Gq[mm]
    pcc=PC.per_condition_classes(Y2,D2)
    if any(len(v)<2 for v in pcc.values()): return None
    cl=np.array(sorted(np.unique(Y2))); folds,_=PC._make_folds(elig,NF,seed)
    prep=_prep_b4c(Z2,D2,g2,folds)
    if prep is None: return None
    T,ok,deltas=PC._T_cv(prep,Y2,D2,g2,cl,C_B4C)
    if not ok: return None
    st=PC._studentize(deltas); Z_obs=st["Z"]
    logp0=_h0_logp_b4c(Z2,Y2,D2,g2,cl); y0=np.searchsorted(cl,Y2); rf=np.random.default_rng(seed+777); nsw=max(20*len(Y2),300)
    def draw(): return cl[PC.sample_h0_fixed_condition_margins(logp0,D2,y0,rf,nsw)], False
    chk=PC.sample_h0_fixed_condition_margins(logp0,D2,y0,np.random.default_rng(seed+778),nsw)
    margin_ok=all(np.array_equal(np.bincount(y0[D2==c],minlength=len(cl)),np.bincount(chk[D2==c],minlength=len(cl))) for c in np.unique(D2))
    res=PC._bootstrap(prep,D2,g2,cl,C_B4C,T,Z_obs,draw,B,0.20,2)
    lcb=float(st["mean"]) - PC._t_quantile(0.975, max(st["S"]-1,1))*float(st["se"])
    return dict(observed_T=float(T), studentized_stat=float(Z_obs), fixed_margin_p=res["p"], studentized_p=res["p_stud"],
                null_sd=res["nsd"], null_mean=res["nmean"], n_boot_invalid=res["n_inv"], margin_preserved=bool(margin_ok),
                mean_delta=float(st["mean"]), se_delta=float(st["se"]), S=int(st["S"]), lcb_budget=lcb,
                n_eligible=len(elig), n_queried=len(pick))


def run(spec, B):
    cond=spec["condition"]; r=spec["cohort"]; base=spec["seed_block"]; seed=spec["seed"]; tid=spec["task_id"]
    try:
        _,Z,Y,D,G,_=OF._reconstruct(cond, r, base)
        b=_b4c(Z,Y,D,G,seed,B)
        if b is None:
            return dict(task_id=tid, condition=cond, stratum=spec["stratum"], seed=int(seed), b4c_applicable=False, diagnostic_only=True)
        # reproducibility self-check: recompute observed T; must be deterministic (same node)
        b2=_b4c(Z,Y,D,G,seed,5)   # tiny B for the observed re-check (observed T independent of B)
        repro_ok = abs(b["observed_T"]-b2["observed_T"])<1e-9
        size_ok=(b["n_queried"]>=20 and b["n_eligible"]>=20)
        confirm=bool(b["fixed_margin_p"]<=A and b["studentized_p"]<=A and b["lcb_budget"]>0 and size_ok)
        s1=S1.get(tid, {})
        gt_nc=cond in OF.NOCONCEPT
        rec=dict(task_id=tid, condition=cond, seed_block=base, cohort=r, seed=int(seed), stratum=spec["stratum"],
                 ground_truth_noconcept=gt_nc,
                 b4c_observed_T=b["observed_T"], archived_B3_observed_T=s1.get("observed_T"),
                 observed_T_repro_ok=repro_ok,
                 b4c_fixed_margin_p=b["fixed_margin_p"], b4c_studentized_p=b["studentized_p"],
                 b4c_studentized_stat=b["studentized_stat"], b4c_null_sd=b["null_sd"], b4c_lcb_budget=b["lcb_budget"],
                 b4c_n_boot_invalid=b["n_boot_invalid"], b4c_margin_preserved=b["margin_preserved"],
                 b4c_confirm=confirm, size_ok=size_ok,
                 b4c_T_z=(float((b["observed_T"]-b["null_mean"])/b["null_sd"]) if b["null_sd"] else None),
                 # archived comparisons (Stage 1a merged + manifest)
                 method_confirm=s1.get("method_confirm"), method_null_sd=s1.get("method_null_sd"),
                 b4a_null_sd=s1.get("b4a_null_sd"), b4b_null_sd=s1.get("b4b_null_sd"),
                 archived_oracle_null_sd_T=spec.get("archived_oracle_null_sd_T"),
                 archived_oracle_confirm=spec.get("oracle_confirm"), method_fixed_margin_p=s1.get("method_fixed_margin_p"),
                 shared_trunk_spec="[Z1:16, c, u1^2,u2^2,u3^2, u1u2,u1u3,u2u3]  (u=top3 PC of weighted-std Z)",
                 interaction_channel="c x PC1:3 (UNCHANGED from B3)", C=C_B4C, B=B,
                 b4c_applicable=True, diagnostic_only=True, not_confirmatory=True, not_deployable=True)
        return rec
    except Exception as e:
        return dict(task_id=tid, condition=cond, seed=int(seed), __worker_error__=f"{type(e).__name__}: {str(e)[:300]}")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--slice", default=None)
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--B", type=int, default=200); ap.add_argument("--out", required=True)
    a=ap.parse_args(); specs=json.load(open(a.manifest))["cohorts"]
    if a.slice: s,e=a.slice.split(":"); specs=specs[int(s):int(e)]
    from joblib import Parallel, delayed
    recs=Parallel(n_jobs=a.jobs, backend="loky")(delayed(run)(sp,a.B) for sp in specs)
    errs=[x for x in recs if "__worker_error__" in x]; repro=[x for x in recs if x.get("observed_T_repro_ok") is False]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out,"w") as f:
        for x in recs: f.write(json.dumps(x,default=str)+"\n")
    open(a.out+".sha256","w").write(hashlib.sha256(open(a.out,"rb").read()).hexdigest()+"  "+os.path.basename(a.out)+"\n")
    json.dump(dict(n=len(recs),B=a.B,C=C_B4C,n_worker_errors=len(errs),n_repro_fail=len(repro),host=socket.gethostname(),
                   slurm=os.environ.get("SLURM_JOB_ID"),diagnostic_only=True), open(a.out+".prov.json","w"), indent=1, default=str)
    print(f"[b4c slice={a.slice}] {len(recs)} recs, {len(errs)} err, {len(repro)} repro-fail, "
          f"confirm={sum(1 for x in recs if x.get('b4c_confirm'))}", flush=True)
    sys.exit(2 if (errs or repro) else 0)
