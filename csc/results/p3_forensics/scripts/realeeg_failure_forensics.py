"""CSC-realEEG-P3.0 FAILURE FORENSICS (NO method change, NO new experiment).
Reconstructs the EXACT frozen v2 cohorts (deterministic: seed = base + condition_index*stride + cohort_index)
and measures session-separability / covariate-overlap diagnostics, to test the hypothesis that NULL_cov
false-confirmations concentrate in high-session-AUC / low-overlap cohorts. Reads the frozen build_cohort
(byte-unchanged); does NOT touch the certifier or any manifest/tag."""
import os, sys, json
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_csc")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from csc.mininfo import realeeg_engine as EG

ART = "/home/infres/yinwang/CMI_AAAI_csc_realeeg_v2_frozen/csc/results/realeeg_validation_v2.final.json"
art = json.load(open(ART))
recs = art["per_cohort"]
base = art["base_seed"]; stride = 1_000_000
CIDX = {"NULL_cov":1, "NULL_label":2, "NULL_cov_plus_label":3, "POS_concept":4}
cache_path = json.load(open("/home/infres/yinwang/CMI_AAAI_csc/csc/mininfo/realeeg_lee2019_cache_manifest.json"))["provenance"]["cache_path"]
npz = np.load(cache_path)
cache = dict(Z=npz["Z"], y=npz["y"], subject=npz["subject_id"], session=npz["session_id"])
subjects = np.unique(cache["subject"]); M = 30

def reconstruct(cond, r):
    """Reproduce run_one_task's cohort construction EXACTLY for (condition, cohort_index r)."""
    seed = base + CIDX[cond]*stride + r
    rng = np.random.default_rng(seed)
    subj = rng.choice(subjects, size=min(M, len(subjects)), replace=False)
    sel = np.isin(cache["subject"], subj)
    coh = {k: cache[k][sel] for k in ("Z","y","subject","session")}
    Z, Y, D, G = EG.build_cohort(cond, coh, rng)
    return Z, Y, D, G, seed

def diagnostics(Z, Y, D, G):
    """D encodes the paired condition = session (1 vs 2). Predict D from Z (subject-grouped CV) -> AUC +
    cross-fit propensity e(z)=P(D==2|Z); overlap fraction, overlap-weight ESS, class/session balance."""
    D2 = (D == 2).astype(int)
    n = len(Y); nsub = len(np.unique(G))
    # subject-grouped OOF propensity
    e = np.zeros(n)
    ng = len(np.unique(G))
    nsplits = min(5, ng)
    Zs = StandardScaler().fit_transform(Z)
    if len(np.unique(D2)) < 2 or nsplits < 2:
        auc = float("nan"); e[:] = D2.mean()
    else:
        for tr, te in GroupKFold(n_splits=nsplits).split(Zs, D2, groups=G):
            if len(np.unique(D2[tr])) < 2:
                e[te] = D2[tr].mean(); continue
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(Zs[tr], D2[tr])
            e[te] = clf.predict_proba(Zs[te])[:, 1]
        auc = roc_auc_score(D2, e) if len(np.unique(D2)) == 2 else float("nan")
    e = np.clip(e, 1e-6, 1-1e-6)
    overlap = float(np.mean((e >= 0.1) & (e <= 0.9)))
    w = np.where(D2 == 1, 1-e, e)             # overlap weights
    ess = float(w.sum()**2 / (w**2).sum()) if (w**2).sum() > 0 else 0.0
    ess_frac = ess / n
    # class balance per session (rule out label/prior composition)
    def cb(sess):
        m = D == sess; return int(m.sum()), float(Y[m].mean()) if m.sum() else float("nan")
    n1, p1 = cb(1); n2, p2 = cb(2)
    return dict(n=n, n_subj=nsub, session_auc=auc, overlap_frac=overlap, ess=ess, ess_frac=ess_frac,
                max_w_ratio=float(w.max()/w.mean()), n_sess1=n1, n_sess2=n2,
                classbal_sess1=p1, classbal_sess2=p2, class_prior_gap=abs(p1-p2))

out = {}
for cond in CIDX:
    rows = []
    conf_idx = {r["cohort"] for r in recs if r["condition"]==cond and r["B3"]["state"]=="CONCEPT_CONFIRMED"}
    for r in range(100):
        Z, Y, D, G, seed = reconstruct(cond, r)
        dg = diagnostics(Z, Y, D, G)
        dg.update(cohort=r, seed=seed, confirmed=(r in conf_idx))
        rows.append(dg)
    out[cond] = rows
    print(f"[{cond}] done ({len(conf_idx)} confirmed)", flush=True)

json.dump(out, open("/home/infres/yinwang/realeeg_feas/forensics_diagnostics.json","w"), indent=1, default=str)

# ---- summary: within-condition contrast confirmed vs not ----
def summ(rows, key):
    a=[x[key] for x in rows if x["confirmed"] and x[key]==x[key]]
    b=[x[key] for x in rows if not x["confirmed"] and x[key]==x[key]]
    import statistics as st
    md=lambda v: (st.median(v) if v else float("nan"))
    return md(a), md(b), (len(a),len(b))
print("\n================ FORENSIC SUMMARY (median confirmed vs median NOT-confirmed) ================")
for cond in CIDX:
    rows=out[cond]
    print(f"\n[{cond}]  (n_confirmed={sum(r['confirmed'] for r in rows)})")
    for key in ("session_auc","overlap_frac","ess_frac","max_w_ratio","class_prior_gap"):
        mc, mn, (na,nb) = summ(rows, key)
        arrow = ""
        if mc==mc and mn==mn:
            if key in ("session_auc","max_w_ratio","class_prior_gap"): arrow = "  <== HIGHER in false-confirm" if mc>mn else ""
            if key in ("overlap_frac","ess_frac"): arrow = "  <== LOWER in false-confirm" if mc<mn else ""
        print(f"    {key:16s} confirmed={mc:.4f}  not={mn:.4f}  (n={na}/{nb}){arrow}")
print("\nsaved: forensics_diagnostics.json")
