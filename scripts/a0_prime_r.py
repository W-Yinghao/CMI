"""A0'-R — PROTOCOL-REPAIR rerun of A0'. Implements notes/A0_PRIME_R_PROTOCOL_REPAIR.md. Repairs only; no new
score/generator/severity/direction/threshold/admissibility freedom. GPU-free.

P0 fix: batch rollback score = WHOLE-batch label-blind mean (was conditioned on base_correct = target labels).
Also: serialized source state (no raw source at scoring), SHA-256 seeds, identity fallback (no label-based deletion),
metamorphic guard (permute y_target -> score path bit-identical), full metrics (AUPRC, C-index), per-disease 0.03
admissibility, double-run determinism.
"""
import os, json, hashlib
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, kendalltau
from cmi.eval.label_shift import transduct_predict
from cmi.eval.source_state import fit_source_state, pmct_predict_serialized
from a0_falsification import DISEASE, V4, B

SCORES = ["g_unc", "s_support", "s_sep", "pr_cmi_proxy"]    # pr_cmi_proxy renamed from 'cmi' (proto-vs-readout proxy)
SEV = {"lowmargin_rot": [10, 20, 30, 45], "highmargin_cbw": [0.05, 0.10, 0.15], "covariate_shift_beneficial": [0.5, 1.0, 2.0]}
FAMS = list(SEV); CONTEST = ["lowmargin_rot", "highmargin_cbw"]; R = 5; EQ = 0.03; RHO_SCORE = 0.2

def sha_seed(*parts):                                       # stable across processes (no PYTHONHASHSEED dependence)
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)

def _shrink(S, rho=RHO_SCORE, eps=1e-3):
    d = S.shape[0]; return (1 - rho) * S + rho * np.trace(S) / d * np.eye(d) + eps * np.eye(d)

def _maha2(z, mu, Winv):
    dz = z - mu; return np.einsum("ij,jk,ik->i", dz, Winv, dz)

# ---- serialized source-free state (frozen probe + source moments + priors; NO raw source at scoring) ----
def state_score_metric(state):
    return np.linalg.inv(_shrink(np.asarray(state["Sig_pool0"], float)))   # frozen score covariance metric

def sample_scores_sf(state, z, prob, Winv):
    p = np.clip(prob, 1e-12, 1)
    g_unc = -(p * np.log(p)).sum(1)
    s_support = _maha2(z, state["mu_pool"], Winv)
    m = np.stack([_maha2(z, state["mu_y"][c], Winv) for c in range(state["n_cls"])], 1)
    s_sep = -np.abs(m[:, 0] - m[:, 1])
    readout = state["clf"].predict(z); proto = m.argmin(1); margin = np.sort(m, 1)[:, 1] - np.sort(m, 1)[:, 0]
    pr_cmi_proxy = (proto != readout).astype(float) * margin + 0.01 * margin
    return dict(g_unc=g_unc, s_support=s_support, s_sep=s_sep, pr_cmi_proxy=pr_cmi_proxy)

def base_adapt_ztilde(state, z):                            # base = probe(z); adapted = serialized matched_coral
    base = state["clf"].predict_proba(z)
    prob, _, ztil = pmct_predict_serialized(state, z, ref="pooled", tmap="wc", em_iters=3, return_ztilde=True)
    return np.asarray(base, float), np.asarray(prob, float), np.asarray(ztil, float)

# ---- generators: z' is y-INDEPENDENT (pocket by base-confidence; rng by SHA seed); only y' (outcome) uses y ----
def _orth(w, rng):
    v = rng.standard_normal(len(w)); v -= (v @ w) * w; return v / (np.linalg.norm(v) + 1e-12)

def gen(name, sev, z, y, state, rng):
    w = state["clf"].coef_[0]; w = w / (np.linalg.norm(w) + 1e-12)
    if name == "lowmargin_rot":
        u = _orth(w, rng); th = np.deg2rad(sev); a = z @ w; b = z @ u
        return z + (np.cos(th)*a - np.sin(th)*b - a)[:, None]*w + (np.sin(th)*a + np.cos(th)*b - b)[:, None]*u, y.copy()
    if name == "highmargin_cbw":
        conf = np.abs(state["clf"].predict_proba(z)[:, 1] - 0.5)
        cand = np.argsort(-conf)[:max(2, int(2*sev*len(z)))]; pocket = rng.permutation(cand)[:max(1, int(sev*len(z)))]
        d = w + 0.5*_orth(w, rng); d /= np.linalg.norm(d) + 1e-12
        zp = z.copy(); yp = y.copy(); zp[pocket] += 1.5*np.sqrt(np.diag(_shrink(np.asarray(state["Sig_pool0"], float))).mean())*d
        yp[pocket] = 1 - yp[pocket]; return zp, yp
    if name == "covariate_shift_beneficial":
        d = _orth(w, rng) if rng.random() < 0.5 else rng.standard_normal(len(w)); d /= np.linalg.norm(d) + 1e-12
        return z + sev*z.std(0).mean()*d, y.copy()
    raise ValueError(name)

def _load(cond, coh):
    o = np.load(f"{V4}/audit_{cond}_{coh}_erm_0.npz", allow_pickle=True)
    zev, yev = np.asarray(o["z_ev"], float), np.asarray(o["y_ev"]).astype(int)
    zte, yte = np.asarray(o["z_te"], float), np.asarray(o["y_te"]).astype(int)
    order = np.argsort(np.asarray(o["window_index_te"]))
    return zev, yev, zte[order], yte[order]

def score_phase(state, zp, Winv):
    """PHASE-1: y-FREE. Returns per-batch (S_B per score, per-sample scores, fallback flag). NO target y touched."""
    out = []
    for s in range(0, len(zp), B):
        bz = zp[s:s+B]; fallback = len(bz) < 8                              # LABEL-BLIND fallback condition
        if fallback:
            base = state["clf"].predict_proba(bz); adapt = base; ztil = bz
        else:
            base, adapt, ztil = base_adapt_ztilde(state, bz)
        sc = sample_scores_sf(state, ztil, adapt, Winv)
        out.append(dict(scores=sc, SB={k: float(sc[k].mean()) for k in SCORES}, fallback=fallback,
                        base=base, adapt=adapt, n=len(bz)))
    return out

def outcome_phase(batches, yp):
    """PHASE-2: uses y ONLY for evaluation endpoints."""
    rows = []
    for bi, b in enumerate(batches):
        s = bi*B; by = yp[s:s+b["n"]]
        base, adapt = b["base"], b["adapt"]
        bc = base.argmax(1) == by; aw = adapt.argmax(1) != by; bw = base.argmax(1) != by
        lb = -np.log(np.clip(base[np.arange(len(by)), by], 1e-9, 1)); la = -np.log(np.clip(adapt[np.arange(len(by)), by], 1e-9, 1))
        rows.append(dict(batch=bi, base_correct=bc, harm_flip=bc & aw, ben_flip=bw & (adapt.argmax(1) == by),
                         base_err=bw, adapt_err=aw, dl=la-lb, mean_dl=float((la-lb).mean()),
                         scores=b["scores"], SB=b["SB"], fallback=b["fallback"]))
    return rows

def run_all():
    store = defaultdict(list); seedmap = {}; guard_fail = []
    for cond, cohs in DISEASE.items():
        for coh in cohs:
            zev, yev, zte, yte = _load(cond, coh)
            state = fit_source_state(zev, yev, 2, rho=0.1); Winv = state_score_metric(state)
            for fam in FAMS:
                for sev in SEV[fam]:
                    for real in range(R):
                        sd = sha_seed(coh, fam, sev, real); seedmap[f"{cond}/{coh}/{fam}/{sev}/{real}"] = sd
                        zp, yp = gen(fam, sev, zte, yte, state, np.random.default_rng(sd))
                        batches = score_phase(state, zp, Winv)              # phase-1: y-free
                        # METAMORPHIC GUARD: permute the EVAL y; score-path (recomputed) must be bit-identical
                        rngp = np.random.default_rng(sd); yperm = np.random.default_rng(sd + 1).permutation(yte)
                        zp2, _ = gen(fam, sev, zte, yperm, state, rngp)      # z' must not depend on eval-y
                        if not np.array_equal(zp, zp2):
                            guard_fail.append(f"{cond}/{coh}/{fam}/{sev}/{real}: z' depends on y_target")
                        b2 = score_phase(state, zp2, Winv)
                        if any(not np.allclose(batches[i]["scores"][k], b2[i]["scores"][k]) for i in range(len(batches)) for k in SCORES):
                            guard_fail.append(f"{cond}/{coh}/{fam}/{sev}/{real}: scores depend on y_target")
                        store[(cond, coh, fam)] += outcome_phase(batches, yp)
    return store, seedmap, guard_fail

def _auc(s, lab):
    lab = np.asarray(lab); return roc_auc_score(lab, s) if len(set(lab)) == 2 and np.std(s) > 1e-12 else np.nan
def _ap(s, lab):
    lab = np.asarray(lab); return average_precision_score(lab, s) if len(set(lab)) == 2 and np.std(s) > 1e-12 else np.nan
def _cindex(s, t):
    if len(s) < 8 or np.std(s) < 1e-12 or np.std(t) < 1e-12: return np.nan
    tau = kendalltau(s, t).correlation; return float((tau + 1) / 2) if tau is not None else np.nan

def within_batch(batches, score, target, subset):
    out = []
    for b in batches:
        m = b["base_correct"] if subset == "base_correct" else np.ones(len(b["base_correct"]), bool)
        if m.sum() < 8: continue
        a = _auc(b["scores"][score][m], b[target][m])
        if not np.isnan(a): out.append(a)
    return out

def analyze(store):
    res = {}
    # SAMPLE within-batch: harm-flip AUROC (+ AUPRC), macro batch->family->cohort, per disease
    def sample_macro(score, disease, metric):
        pc = []
        for coh in DISEASE[disease]:
            fm = []
            for fam in CONTEST:
                vals = []
                for b in store[(disease, coh, fam)]:
                    m = b["base_correct"]
                    if m.sum() < 8: continue
                    v = (_auc if metric == "auc" else _ap)(b["scores"][score][m], b["harm_flip"][m])
                    if not np.isnan(v): vals.append(v)
                if vals: fm.append(np.mean(vals))
            if fm: pc.append(np.mean(fm))
        return float(np.mean(pc)) if pc else np.nan
    res["sample_auc"] = {s: {d: sample_macro(s, d, "auc") for d in DISEASE} for s in SCORES}
    res["sample_auprc"] = {s: {d: sample_macro(s, d, "ap") for d in DISEASE} for s in SCORES}
    res["per_family_auc"] = {s: {f"{d}/{fam}": float(np.mean([v for coh in DISEASE[d] for v in within_batch(store[(d, coh, fam)], s, "harm_flip", "base_correct")] or [np.nan]))
                                 for d in DISEASE for fam in CONTEST} for s in SCORES}
    # adaptation specificity: base_err on FULL batch vs harm_flip on base-correct
    res["specificity"] = {s: dict(harm=float(np.mean([v for d in DISEASE for coh in DISEASE[d] for fam in CONTEST for v in within_batch(store[(d, coh, fam)], s, "harm_flip", "base_correct")] or [np.nan])),
                                  base_err=float(np.mean([v for d in DISEASE for coh in DISEASE[d] for fam in CONTEST for v in within_batch(store[(d, coh, fam)], s, "base_err", "all")] or [np.nan]))) for s in SCORES}
    # BATCH: whole-batch S_B vs mean Δℓ; Spearman + C-index + AUROC[Δℓ>0]; LOGFO; per-cohort robustness
    def brows(disease, excl=None):
        return [(b, b["mean_dl"]) for coh in DISEASE[disease] for fam in CONTEST if fam != excl for b in store[(disease, coh, fam)]]
    def bmetric(score, disease, excl=None):
        rows = brows(disease, excl); sB = np.array([b["SB"][score] for b, _ in rows]); dl = np.array([d for _, d in rows])
        if len(sB) < 8 or np.std(sB) < 1e-12 or np.std(dl) < 1e-12: return dict(spearman=np.nan, cindex=np.nan, auc_pos=np.nan, n=len(sB))
        return dict(spearman=float(spearmanr(sB, dl).correlation), cindex=_cindex(sB, dl), auc_pos=float(_auc(sB, dl > 0)), n=len(sB))
    res["batch"] = {s: {d: bmetric(s, d) for d in DISEASE} for s in SCORES}
    res["batch_logfo"] = {s: {d: {f"-{fam}": bmetric(s, d, fam) for fam in CONTEST} for d in DISEASE} for s in SCORES}
    def bcohort(score, disease):
        out = {}
        for coh in DISEASE[disease]:
            rows = [(b, b["mean_dl"]) for fam in CONTEST for b in store[(disease, coh, fam)]]
            sB = np.array([b["SB"][score] for b, _ in rows]); dl = np.array([d for _, d in rows])
            out[coh] = float(spearmanr(sB, dl).correlation) if len(sB) > 8 and np.std(sB) > 1e-12 and np.std(dl) > 1e-12 else None
        return out
    res["batch_by_cohort"] = {s: {d: bcohort(s, d) for d in DISEASE} for s in SCORES}
    return res

def decide(res):
    # per-disease, per-level best (0.03 band)
    def s_adm(s):
        for d in DISEASE:
            best = max(res["sample_auc"][k][d] for k in SCORES if not np.isnan(res["sample_auc"][k][d]))
            v = res["sample_auc"][s][d]
            if np.isnan(v) or v <= 0.5 or v < best - EQ: return False
        return all((res["per_family_auc"][s][f] or 0) > 0.5 for f in res["per_family_auc"][s])
    def b_adm(s):
        for d in DISEASE:
            best = max(res["batch"][k][d]["auc_pos"] for k in SCORES if not np.isnan(res["batch"][k][d]["auc_pos"]))
            m = res["batch"][s][d]
            if np.isnan(m["spearman"]) or m["spearman"] <= 0 or m["auc_pos"] <= 0.5 or m["auc_pos"] < best - EQ: return False
            if not all((res["batch_logfo"][s][d][f]["spearman"] or 0) > 0 for f in res["batch_logfo"][s][d]): return False
        vals = [v for d in DISEASE for v in res["batch_by_cohort"][s][d].values() if v is not None]
        return len(vals) >= 5 and sum(v > 0 for v in vals) >= len(vals) - 1
    sa = {s: bool(s_adm(s)) for s in SCORES}; ba = {s: bool(b_adm(s)) for s in SCORES}
    both = [s for s in SCORES if sa[s] and ba[s]]; hs, hb = any(sa.values()), any(ba.values())
    if both: dec = "SINGLE_SCALAR_CANDIDATE"
    elif hs and hb: dec = "TWO_LEVEL_CANDIDATE"
    elif hs: dec = "POST_ALIGN_ABSTENTION_ONLY"
    elif hb: dec = "ROLLBACK_ELIGIBILITY_ONLY"
    else: dec = "DIAGNOSTIC_ONLY"
    return dec, sa, ba

def canonical_hash(res, dec):
    key = dict(decision=dec, sample_auc=res["sample_auc"], batch={s: {d: {k: round(v, 6) if isinstance(v, float) and v == v else v for k, v in res["batch"][s][d].items()} for d in DISEASE} for s in SCORES})
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]

def main():
    fz = json.load(open("results/freeze_a1/manifest.json"))
    # serialized-state VERIFY GATE (vs deployed transduct, bit-exact)
    zev, yev, zte, _ = _load("PD", "ds002778")
    st = fit_source_state(zev, yev, 2, rho=0.1)
    on = transduct_predict(zev, yev, zte[:64], np.bincount(yev, minlength=2)/len(yev), 2, mode="matched_coral", shrink=0.1)["prob"]
    sp, _, _ = base_adapt_ztilde(st, zte[:64]); verify = float(np.abs(sp - on).max())  # sp is base; check adapt path:
    ad = pmct_predict_serialized(st, zte[:64], ref="pooled", tmap="wc", em_iters=3)[0]
    verify = float(np.abs(np.asarray(ad, float) - on).max())
    if verify > 1e-9:
        print(f"SERIALIZED-STATE VERIFY FAILED ({verify:.2e}) — abort"); raise SystemExit(2)
    store, seedmap, guard_fail = run_all()
    if guard_fail:
        print("METAMORPHIC GUARD FAILED (target-label leakage):"); [print("  ", g) for g in guard_fail[:10]]; raise SystemExit(3)
    res = analyze(store); dec, sa, ba = decide(res); h1 = canonical_hash(res, dec)
    # DOUBLE-RUN determinism
    store2, _, gf2 = run_all(); res2 = analyze(store2); dec2, _, _ = decide(res2); h2 = canonical_hash(res2, dec2)
    if h1 != h2:
        print(f"NON-DETERMINISTIC ({h1} != {h2}) — abort"); raise SystemExit(4)
    out = f"results/a0_prime_r/{fz['hash'][:16]}"; os.makedirs(out, exist_ok=True)
    summary = dict(decision=dec, sample_admissible=sa, batch_admissible=ba,
                   metamorphic_guard="PASS", serialized_verify_maxabs=verify, double_run_hash=h1, double_run_match=True,
                   sample_auc=res["sample_auc"], sample_auprc=res["sample_auprc"], per_family_auc=res["per_family_auc"],
                   batch=res["batch"], batch_logfo=res["batch_logfo"], batch_by_cohort=res["batch_by_cohort"],
                   adaptation_specificity=res["specificity"], n_seeds=len(seedmap),
                   note="P0 fix: batch S_B = WHOLE-batch label-blind mean (was base_correct-conditioned). pr_cmi_proxy "
                        "= prototype-vs-readout disagreement (NOT decoder-CMI residual).", freeze_a1_hash=fz["hash"])
    json.dump(summary, open(f"{out}/a0primer_summary.json", "w"), indent=2, default=str)
    json.dump(dict(pre_registration="notes/A0_PRIME_R_PROTOCOL_REPAIR.md", seed_map=seedmap, dumps=V4), open(f"{out}/run_manifest.json", "w"), indent=2)
    print(f"=== A0'-R DECISION: {dec} ===  (guard=PASS, serialized|d|={verify:.0e}, double-run={h1}=={h2})")
    print("  SAMPLE within-batch harm AUROC (AUPRC) PD/SCZ:")
    for s in SCORES:
        print(f"    {s:13s} AUROC PD={res['sample_auc'][s]['PD']:.3f} SCZ={res['sample_auc'][s]['SCZ']:.3f} | AUPRC PD={res['sample_auprc'][s]['PD']:.3f} SCZ={res['sample_auprc'][s]['SCZ']:.3f} | adm={sa[s]}")
    print("  adaptation-specificity (harm[base-corr] vs base_err[all] AUROC):")
    for s in SCORES:
        print(f"    {s:13s} harm={res['specificity'][s]['harm']:.3f} base_err={res['specificity'][s]['base_err']:.3f}")
    print("  BATCH whole-batch-mean vs meanΔℓ (Spearman | C-index | AUROC[Δℓ>0]):")
    for s in SCORES:
        b = res["batch"][s]
        print(f"    {s:13s} PD ρ={b['PD']['spearman']:+.2f} C={b['PD']['cindex']:.2f} auc={b['PD']['auc_pos']:.2f} | SCZ ρ={b['SCZ']['spearman']:+.2f} C={b['SCZ']['cindex']:.2f} auc={b['SCZ']['auc_pos']:.2f} | adm={ba[s]}")
    print(f"  -> {out}/a0primer_summary.json")

if __name__ == "__main__":
    main()
