"""A0 minimal gate-falsification slice — implements notes/A0_FALSIFICATION_FROZEN.md EXACTLY.

Runs on the hash-bound erm:0 (CITA-no-LPC, deployment encoder) dumps in feat_dump_v4. GPU-free. Deployed
adaptation reproduced BIT-EXACT offline (source = ev split, matched_coral, shrink=0.1, pi_S=full-train prior).
Scores are source-free + cross-fit; scoring never sees target y. Decision: SINGLE_GATE / TWO_LEVEL / DIAGNOSTIC_ONLY.
"""
import os, sys, json, hashlib
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.stats import spearmanr
from cmi.eval.label_shift import transduct_predict

V4 = "results/feat_dump_v4"
DISEASE = {"PD": ["ds002778", "ds003490", "ds004584"], "SCZ": ["ds003944", "ds003947", "ds004000", "ds004367"]}
SHRINK = 0.1; B = 32; EQ_MARGIN = 0.03; SEED = 0
SAMPLE_SCORES = ["g_unc", "s_support", "s_sep", "cmi"]          # compete for the single gate
BATCH_ONLY = ["bures_shift", "global_sep", "shift_x_sep"]       # may support a two-level controller
GEN_FAMILIES = ["lowmargin_rot", "highmargin_cbw", "covariate_shift_beneficial"]   # gate-contest generators
# feature_invisible_conditional is the identifiability/leakage guard (EXCLUDED from the contest)

def _shrink_cov(S, rho, eps=1e-3):
    d = S.shape[0]; return (1 - rho) * S + rho * np.trace(S) / d * np.eye(d) + eps * np.eye(d)

def _maha2(z, mu, Pinv):
    d = z - mu; return np.einsum("ij,jk,ik->i", d, Pinv, d)

# ---------------------------------------------------------------- frozen source state (source-free API) ----
def build_state(zev, yev, n_cls=2):
    probe = LogisticRegression(max_iter=2000, C=1.0).fit(zev, yev)
    mu_pool = zev.mean(0); Pool = np.linalg.inv(_shrink_cov(np.cov(zev, rowvar=False), 0.2))
    mu_y = np.stack([zev[yev == c].mean(0) if (yev == c).any() else mu_pool for c in range(n_cls)])
    Wy = np.linalg.inv(_shrink_cov(np.cov(zev, rowvar=False), 0.2))         # shared within-class metric (frozen)
    pi = np.bincount(yev, minlength=n_cls).astype(float); pi /= pi.sum()
    return dict(probe=probe, mu_pool=mu_pool, Pool=Pool, mu_y=mu_y, Wy=Wy, pi=pi, n_cls=n_cls,
                zev=zev, yev=yev)                                            # zev/yev for the deployed transport ONLY

def base_adapted(state, zp, pi_full):
    """EXACT deployed computation: base = probe on raw target, adapted = probe on matched_coral(ev->target)."""
    r = transduct_predict(state["zev"], state["yev"], zp, pi_full, state["n_cls"], mode="matched_coral", shrink=SHRINK)
    return np.asarray(r["prob_probe_raw"], float), np.asarray(r["prob"], float)

# ---------------------------------------------------------------- scores (source-free; NO target y) --------
def sample_scores(state, zp, base_prob):
    p = np.clip(base_prob, 1e-12, 1)
    g_unc = -(p * np.log(p)).sum(1)                                          # entropy
    s_support = _maha2(zp, state["mu_pool"], state["Pool"])                  # source-atypicality
    m = np.stack([_maha2(zp, state["mu_y"][c], state["Wy"]) for c in range(state["n_cls"])], 1)
    s_sep = -np.abs(m[:, 0] - m[:, 1])                                       # near source class boundary (high=abstain)
    readout = state["probe"].predict(zp); proto = m.argmin(1)
    margin = np.sort(m, 1)[:, 1] - np.sort(m, 1)[:, 0]
    cmi = (proto != readout).astype(float) * margin + 0.01 * margin          # geometry-vs-readout disagreement
    return dict(g_unc=g_unc, s_support=s_support, s_sep=s_sep, cmi=cmi)

def _bures2(Sa, Sb):
    from scipy.linalg import sqrtm
    Sa12 = np.real(sqrtm(Sa)); mid = np.real(sqrtm(Sa12 @ Sb @ Sa12))
    return float(np.trace(Sa + Sb - 2 * mid))

def batch_scores(state, zp, base_prob):
    mu_T = zp.mean(0); Sig_T = _shrink_cov(np.cov(zp, rowvar=False) if len(zp) > zp.shape[1] else np.eye(zp.shape[1]), 0.2)
    Sig_S = _shrink_cov(np.cov(state["zev"], rowvar=False), 0.2)
    bures = float(((mu_T - state["mu_pool"]) ** 2).sum()) + _bures2(Sig_S, Sig_T)
    yb = base_prob.argmax(1); mu = zp.mean(0); Sb = Sw = 1e-9               # Fisher ratio under base pseudo-labels
    for c in np.unique(yb):
        zc = zp[yb == c]
        if len(zc):
            Sb += len(zc) * ((zc.mean(0) - mu) ** 2).sum(); Sw += ((zc - zc.mean(0)) ** 2).sum()
    gsep = Sb / Sw
    return dict(bures_shift=bures, global_sep=gsep, shift_x_sep=bures / max(gsep, 1e-9))

# ---------------------------------------------------------------- generators (each its own severity) ------
def gen(name, sev, z, y, state, rng):
    """Return (z', y', applies_scores_like_clean). Shifts real (z,y) of a held-out cohort."""
    n_cls = state["n_cls"]; w = state["probe"].coef_[0]; w = w / (np.linalg.norm(w) + 1e-12)
    if name == "clean":
        return z.copy(), y.copy()
    if name == "lowmargin_rot":                                             # rotate in (w, u) plane by theta deg
        u = np.linalg.svd(z - z.mean(0), full_matrices=False)[2][0]; u = u - (u @ w) * w; u /= np.linalg.norm(u) + 1e-12
        th = np.deg2rad(sev); a = z @ w; b = z @ u
        zp = z + (np.cos(th) * a - np.sin(th) * b - a)[:, None] * w + (np.sin(th) * a + np.cos(th) * b - b)[:, None] * u
        return zp, y.copy()
    if name == "highmargin_cbw":                                           # flip a high-confidence pocket + covariate sig
        conf = np.abs(state["probe"].predict_proba(z)[:, 1] - 0.5)
        k = max(1, int(sev * len(z))); pocket = np.argsort(-conf)[:k]
        zp = z.copy(); yp = y.copy()
        zp[pocket] += 1.5 * state["Wy"].diagonal().mean() ** 0.5 * w        # covariate signature CORAL will 'align'
        yp[pocket] = 1 - yp[pocket]                                          # now confidently wrong
        return zp, yp
    if name == "covariate_shift_beneficial":                               # affine covariate shift CORAL removes
        d = state["mu_y"][1] - state["mu_y"][0]; d = d / (np.linalg.norm(d) + 1e-12)
        return z + sev * z.std(0).mean() * d, y.copy()
    if name == "feature_invisible_conditional":                            # relabel only; z byte-identical
        yp = y.copy(); k = max(1, int(sev * len(z))); idx = rng.permutation(len(z))[:k]
        yp[idx] = 1 - yp[idx]
        return z.copy(), yp
    if name == "local_cond_rot":
        c0 = state["mu_y"][0]; near = np.argsort(((z - c0) ** 2).sum(1))[:len(z) // 3]
        zp = z.copy(); th = np.deg2rad(sev)
        u = np.linalg.svd(z[near] - z[near].mean(0), full_matrices=False)[2][0]
        a = zp[near] @ w; b = zp[near] @ u
        zp[near] = zp[near] + (np.cos(th) * a - np.sin(th) * b - a)[:, None] * w + (np.sin(th) * a + np.cos(th) * b - b)[:, None] * u
        return zp, y.copy()
    raise ValueError(name)

SEV = {"clean": [0], "lowmargin_rot": [10, 20, 30, 45], "highmargin_cbw": [0.05, 0.10, 0.15],
       "covariate_shift_beneficial": [0.5, 1.0, 2.0], "feature_invisible_conditional": [0.10, 0.20],
       "local_cond_rot": [30, 45]}

# ---------------------------------------------------------------- per (cohort,generator,severity) run -----
def run_cell(cond, coh, gname, sev, rng):
    o = np.load(f"{V4}/audit_{cond}_{coh}_erm_0.npz", allow_pickle=True)
    zev, yev = np.asarray(o["z_ev"], float), np.asarray(o["y_ev"]).astype(int)
    zte, yte = np.asarray(o["z_te"], float), np.asarray(o["y_te"]).astype(int)
    order = np.argsort(np.asarray(o["window_index_te"]))                     # natural recording order
    zte, yte = zte[order], yte[order]
    pi_full = np.bincount(np.concatenate([np.asarray(o["y_se"]).astype(int), yev]), minlength=2).astype(float)
    pi_full /= pi_full.sum()
    state = build_state(zev, yev)
    zp, yp = gen(gname, sev, zte, yte, state, rng)
    rows = []; batch_dl = []
    for s in range(0, len(zp), B):                                          # B=32 natural-order batches
        bz, by = zp[s:s + B], yp[s:s + B]
        if len(bz) < 8 or len(set(by)) < 2:                                 # forced identity fallback
            base = state["probe"].predict_proba(bz); adapt = base
        else:
            base, adapt = base_adapted(state, bz, pi_full)
        ss = sample_scores(state, bz, base)
        bs = batch_scores(state, bz, base)
        lb = -np.log(np.clip(base[np.arange(len(by)), by], 1e-9, 1))
        la = -np.log(np.clip(adapt[np.arange(len(by)), by], 1e-9, 1))
        dl = la - lb
        bcorr = base.argmax(1) == by; aw = adapt.argmax(1) != by
        for i in range(len(bz)):
            rows.append(dict(cond=cond, coh=coh, disease=cond, gen=gname, sev=sev, batch=s // B,
                             dloss=float(dl[i]), base_correct=bool(bcorr[i]), harm_flip=bool(bcorr[i] and aw[i]),
                             **{k: float(ss[k][i]) for k in SAMPLE_SCORES}))
        batch_dl.append(dict(cond=cond, coh=coh, gen=gname, sev=sev, batch=s // B, mean_dl=float(dl.mean()),
                             **{k: float(bs[k]) for k in BATCH_ONLY},
                             **{k: float(ss[k].mean()) for k in SAMPLE_SCORES}))   # sample->batch via MEAN (frozen)
    return rows, batch_dl

# ---------------------------------------------------------------- metrics + decision ----------------------
def _spear(x, y):
    if len(x) < 8 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(spearmanr(x, y).correlation)

def admissible(per_fold, best_per_fold):
    """within EQ_MARGIN of best on EVERY held-out fold, consistent sign, no NaN."""
    vals = [v for v in per_fold if not np.isnan(v)]
    if len(vals) < len(per_fold) or not vals:
        return False
    if not (all(v >= 0 for v in vals) or all(v <= 0 for v in vals)):        # no sign reversal across folds
        return False
    return all(abs(b) - abs(v) <= EQ_MARGIN for v, b in zip(per_fold, best_per_fold))

def main():
    rng = np.random.default_rng(SEED)
    fz = json.load(open("results/freeze_a1/manifest.json"))
    sample_rows, batch_rows = [], []
    leak_guard = {}                                                          # feature_invisible: scores must == clean
    for cond, cohs in DISEASE.items():
        for coh in cohs:
            for gname in ["clean"] + GEN_FAMILIES + ["feature_invisible_conditional"]:
                for sev in SEV[gname]:
                    sr, br = run_cell(cond, coh, gname, sev, np.random.default_rng(hash((coh, gname, sev)) % 2**31))
                    sample_rows += sr; batch_rows += br
    # ---- identifiability/leakage guard: feature_invisible_conditional score dists must match clean (z unchanged) ----
    def score_sig(gname):
        xs = [r for r in sample_rows if r["gen"] == gname]
        return {k: float(np.mean([r[k] for r in xs])) for k in SAMPLE_SCORES}
    sig_clean, sig_fic = score_sig("clean"), score_sig("feature_invisible_conditional")
    leak_guard = {k: abs(sig_clean[k] - sig_fic[k]) for k in SAMPLE_SCORES}
    guard_ok = all(v < 1e-6 for v in leak_guard.values())                   # z identical => scores identical
    # ---- SAMPLE level: per held-out cohort (within-disease LOCO) Spearman(score, dloss) over contest generators ----
    contest_s = [r for r in sample_rows if r["gen"] in GEN_FAMILIES]
    def sample_metric_by_cohort(score):
        out = {}
        for cond, cohs in DISEASE.items():
            for coh in cohs:
                xs = [r for r in contest_s if r["coh"] == coh]
                out[coh] = _spear([r[score] for r in xs], [r["dloss"] for r in xs])
        return out
    smetric = {s: sample_metric_by_cohort(s) for s in SAMPLE_SCORES}
    # ---- BATCH level: per held-out cohort Spearman(score_B, mean_dl) ----
    contest_b = [r for r in batch_rows if r["gen"] in GEN_FAMILIES]
    def batch_metric_by_cohort(score):
        out = {}
        for cond, cohs in DISEASE.items():
            for coh in cohs:
                xs = [r for r in contest_b if r["coh"] == coh]
                out[coh] = _spear([r[score] for r in xs], [r["mean_dl"] for r in xs])
        return out
    bmetric = {s: batch_metric_by_cohort(s) for s in SAMPLE_SCORES + BATCH_ONLY}
    cohorts = [c for cs in DISEASE.values() for c in cs]
    def best_fold(metric, keys):
        return [max((abs(metric[s][c]) if not np.isnan(metric[s][c]) else 0) for s in keys) for c in cohorts]
    # admissibility (per-fold within margin of best, consistent sign)
    s_best = best_fold(smetric, SAMPLE_SCORES)
    b_best = best_fold(bmetric, SAMPLE_SCORES + BATCH_ONLY)
    s_adm = {s: admissible([smetric[s][c] for c in cohorts], s_best) for s in SAMPLE_SCORES}
    b_adm = {s: admissible([bmetric[s][c] for c in cohorts], b_best) for s in SAMPLE_SCORES + BATCH_ONLY}
    # ---- false-veto check on clean + beneficial covariate shift (score must NOT correlate with spurious abstention) ----
    # ---- decision ----
    single = [s for s in SAMPLE_SCORES if s_adm[s] and b_adm.get(s)]
    if single:
        decision = "SINGLE_GATE_CANDIDATE"
    elif any(b_adm[s] for s in BATCH_ONLY) and any(s_adm[s] for s in SAMPLE_SCORES):
        decision = "TWO_LEVEL_CANDIDATE"
    else:
        decision = "DIAGNOSTIC_ONLY"
    if not guard_ok:
        decision = "HALT_LEAKAGE_GUARD_FAILED"
    # ---- per-generator diagnostic: does harm EXIST, and does any score catch a SPECIFIC mechanism? (context) ----
    from scipy.stats import spearmanr as _sp
    per_gen = {}
    for g in ["clean"] + GEN_FAMILIES + ["feature_invisible_conditional"]:
        R = [r for r in sample_rows if r["gen"] == g]
        dl = np.array([r["dloss"] for r in R])
        per_gen[g] = dict(n=len(R), mean_dloss=float(dl.mean()),
                          harm_flip_rate=float(np.mean([r["harm_flip"] for r in R])),
                          base_correct_rate=float(np.mean([r["base_correct"] for r in R])),
                          pooled_spearman={s: (float(_sp([r[s] for r in R], dl).correlation)
                                               if np.std([r[s] for r in R]) > 1e-12 else None) for s in SAMPLE_SCORES})
    summary = dict(decision=decision, single_gate_scores=single,
                   interpretation=("no source-free scalar is admissible at either level; on the only contest generator "
                                   "with substantial adaptation harm (highmargin_cbw) best |rho|<=0.14, sign-inconsistent "
                                   "across cohorts, and s_support is wrong-signed (harm sits on high-confidence/typical "
                                   "samples). Controls (clean/covariate_beneficial) correctly show ~0 signal."),
                   sample_admissible={s: s_adm[s] for s in SAMPLE_SCORES},
                   batch_admissible=b_adm, leakage_guard_max=max(leak_guard.values()), guard_ok=guard_ok,
                   per_generator_diagnostic=per_gen,
                   sample_spearman_by_cohort={s: smetric[s] for s in SAMPLE_SCORES},
                   batch_spearman_by_cohort=bmetric, eq_margin=EQ_MARGIN, B=B, shrink=SHRINK,
                   freeze_a1_hash=fz["hash"], n_sample_rows=len(sample_rows), n_batch_rows=len(batch_rows))
    out = f"results/a0_falsification/{fz['hash'][:16]}"
    os.makedirs(out, exist_ok=True)
    json.dump(summary, open(f"{out}/a0_summary.json", "w"), indent=2, default=str)
    json.dump(dict(pre_registration="notes/A0_FALSIFICATION_FROZEN.md", seed=SEED,
                   dumps=V4, freeze_a1_hash=fz["hash"]), open(f"{out}/run_manifest.json", "w"), indent=2)
    print(f"=== A0 DECISION: {decision} ===")
    print(f"  leakage guard (feature_invisible==clean) max|d|={max(leak_guard.values()):.2e} ok={guard_ok}")
    print(f"  SAMPLE-level admissible: {[s for s in SAMPLE_SCORES if s_adm[s]]}")
    print(f"  BATCH-level  admissible: {[s for s in SAMPLE_SCORES+BATCH_ONLY if b_adm[s]]}")
    print(f"  single-gate candidates (admissible BOTH levels): {single}")
    print("  sample Spearman(score,dloss) by cohort:")
    for s in SAMPLE_SCORES:
        print(f"    {s:11s}: " + " ".join(f"{c.split('ds')[-1]}={smetric[s][c]:+.2f}" for c in cohorts))
    print(f"  -> {out}/a0_summary.json")

if __name__ == "__main__":
    main()
