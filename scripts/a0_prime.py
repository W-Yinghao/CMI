"""A0' — CONFIRMATORY gate slice. Implements notes/A0_PRIME_FROZEN.md on UNSEEN stochastic generator realizations.
Frozen (== A0): readout/adaptation (bit-exact), B=32, score formulas+directions, eq-margin 0.03. GPU-free.

Tests, with the batch-mean vs within-batch-residual decomposition:
  SAMPLE (abstention-after-tentative-adapt): within-batch AUROC of post-score vs harm_flip (base-correct), macro-avg.
  BATCH  (rollback eligibility): Spearman/C-index/AUROC of batch-mean post-score vs mean Δℓ_B, LOGFO, disease-strat.
Decision: SINGLE_SCALAR / TWO_LEVEL / POST_ALIGN_ABSTENTION_ONLY / DIAGNOSTIC_ONLY.
"""
import os, json, hashlib
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from cmi.eval.label_shift import transduct_predict
from a0_falsification import build_state, sample_scores, DISEASE, SAMPLE_SCORES, SHRINK, B, V4

SEV = {"lowmargin_rot": [10, 20, 30, 45], "highmargin_cbw": [0.05, 0.10, 0.15],
       "covariate_shift_beneficial": [0.5, 1.0, 2.0]}
FAMILIES = list(SEV); R = 5; SEED_BASE = 9000; EQ = 0.03

def _unit_orth(w, rng):
    v = rng.standard_normal(len(w)); v -= (v @ w) * w; return v / (np.linalg.norm(v) + 1e-12)

def gen_stoch(name, sev, z, y, st, rng):
    """UNSEEN stochastic realization (A0 used deterministic generators)."""
    w = st["probe"].coef_[0]; w = w / (np.linalg.norm(w) + 1e-12)
    if name == "lowmargin_rot":
        u = _unit_orth(w, rng); th = np.deg2rad(sev); a = z @ w; b = z @ u
        return z + (np.cos(th)*a - np.sin(th)*b - a)[:, None]*w + (np.sin(th)*a + np.cos(th)*b - b)[:, None]*u, y.copy()
    if name == "highmargin_cbw":
        conf = np.abs(st["probe"].predict_proba(z)[:, 1] - 0.5)
        cand = np.argsort(-conf)[:max(2, int(2*sev*len(z)))]
        pocket = rng.permutation(cand)[:max(1, int(sev*len(z)))]            # random subset of the pocket
        d = w + 0.5*_unit_orth(w, rng); d /= np.linalg.norm(d) + 1e-12      # random covariate direction (w-aligned)
        zp = z.copy(); yp = y.copy()
        zp[pocket] += 1.5*np.sqrt(st["Wy"].diagonal().mean())*d; yp[pocket] = 1 - yp[pocket]
        return zp, yp
    if name == "covariate_shift_beneficial":
        d = _unit_orth(w, rng) if rng.random() < 0.5 else rng.standard_normal(len(w))
        d /= np.linalg.norm(d) + 1e-12
        return z + sev*z.std(0).mean()*d, y.copy()
    raise ValueError(name)

def realize(cond, coh, name, sev, rng):
    o = np.load(f"{V4}/audit_{cond}_{coh}_erm_0.npz", allow_pickle=True)
    zev, yev = np.asarray(o["z_ev"], float), np.asarray(o["y_ev"]).astype(int)
    zte, yte = np.asarray(o["z_te"], float), np.asarray(o["y_te"]).astype(int)
    order = np.argsort(np.asarray(o["window_index_te"])); zte, yte = zte[order], yte[order]
    pi = np.bincount(np.concatenate([np.asarray(o["y_se"]).astype(int), yev]), minlength=2).astype(float); pi /= pi.sum()
    st = build_state(zev, yev); zp, yp = gen_stoch(name, sev, zte, yte, st, rng)
    batches = []
    for s in range(0, len(zp), B):
        bz, by = zp[s:s+B], yp[s:s+B]
        if len(bz) < 8 or len(set(by)) < 2:
            continue
        r = transduct_predict(zev, yev, bz, pi, 2, mode="matched_coral", shrink=SHRINK)
        base, adapt, ztil = np.asarray(r["prob_probe_raw"], float), np.asarray(r["prob"], float), np.asarray(r["z_tilde"], float)
        post = sample_scores(st, ztil, adapt)
        bc = base.argmax(1) == by; aw = adapt.argmax(1) != by; bw = base.argmax(1) != by
        hf = bc & aw; bf = bw & (adapt.argmax(1) == by)                      # harmful / beneficial flips
        lb = -np.log(np.clip(base[np.arange(len(by)), by], 1e-9, 1)); la = -np.log(np.clip(adapt[np.arange(len(by)), by], 1e-9, 1))
        batches.append(dict(post={k: post[k] for k in SAMPLE_SCORES}, base_correct=bc, harm_flip=hf, ben_flip=bf,
                            base_err=bw, adapt_err=aw, mean_dl=float((la-lb).mean())))
    return batches

def _auc(s, lab):
    lab = np.asarray(lab)
    return roc_auc_score(lab, s) if len(set(lab)) == 2 and np.std(s) > 1e-12 else np.nan

def within_batch_auc(batches, score, target, subset="base_correct"):
    """per-eligible-batch AUROC; subset='base_correct' for harm_flip, 'all' for base_err/adapt_err (else degenerate)."""
    out = []
    for b in batches:
        m = b["base_correct"] if subset == "base_correct" else np.ones(len(b["base_correct"]), bool)
        if m.sum() < 8:
            continue
        s = b["post"][score][m]; lab = b[target][m]
        a = _auc(s, lab)
        if not np.isnan(a):
            out.append(a)
    return out

def main():
    fz = json.load(open("results/freeze_a1/manifest.json"))
    # collect realizations: per (disease, cohort, family) -> list of batches
    store = defaultdict(list)
    for cond, cohs in DISEASE.items():
        for coh in cohs:
            for fam in FAMILIES:
                for sev in SEV[fam]:
                    for real in range(R):
                        rng = np.random.default_rng(SEED_BASE + real + (hash((coh, fam, sev)) % 100000))
                        store[(cond, coh, fam)] += realize(cond, coh, fam, sev, rng)
    contest_fams = ["lowmargin_rot", "highmargin_cbw"]                       # harm-bearing; beneficial is the control

    # ---- PRIMARY SAMPLE TEST: within-batch harm-flip AUROC, macro batch->family->cohort, PD/SCZ separate ----
    def sample_macro(score, disease):
        per_cohort = []
        for coh in DISEASE[disease]:
            fam_means = []
            for fam in contest_fams:
                aucs = within_batch_auc(store[(disease, coh, fam)], score, "harm_flip")
                if aucs:
                    fam_means.append(np.mean(aucs))
            if fam_means:
                per_cohort.append(np.mean(fam_means))
        return float(np.mean(per_cohort)) if per_cohort else np.nan, per_cohort
    # per-generator-family direction (no pooled-only): within-batch AUROC per family, per disease
    def fam_dir(score):
        out = {}
        for d in DISEASE:
            for fam in contest_fams:
                aucs = []
                for coh in DISEASE[d]:
                    aucs += within_batch_auc(store[(d, coh, fam)], score, "harm_flip")
                out[f"{d}/{fam}"] = float(np.mean(aucs)) if aucs else None
        return out
    sample_tbl = {s: {d: sample_macro(s, d)[0] for d in DISEASE} for s in SAMPLE_SCORES}
    # base-error vs adapted-error vs harm-flip (adaptation-specificity)
    spec = {}
    for s in SAMPLE_SCORES:
        row = {}
        for tgt, sub in (("harm_flip", "base_correct"), ("base_err", "all"), ("adapt_err", "all")):
            aucs = []
            for d in DISEASE:
                for coh in DISEASE[d]:
                    for fam in contest_fams:
                        aucs += within_batch_auc(store[(d, coh, fam)], s, tgt, subset=sub)
            row[tgt] = float(np.mean(aucs)) if aucs else None
        spec[s] = row

    # ---- PRIMARY BATCH TEST: batch-mean score vs mean Δℓ_B; Spearman + C-index + AUROC[meanΔℓ>0]; LOGFO ----
    def batch_rows(disease, exclude_fam=None):
        rows = []
        for coh in DISEASE[disease]:
            for fam in contest_fams:
                if fam == exclude_fam:
                    continue
                for b in store[(disease, coh, fam)]:
                    rows.append((b, b["mean_dl"]))
        return rows
    def batch_metric(score, disease, exclude_fam=None):
        rows = batch_rows(disease, exclude_fam)
        sB = np.array([b["post"][score][b["base_correct"]].mean() if b["base_correct"].any() else b["post"][score].mean() for b, _ in rows])
        dl = np.array([d for _, d in rows])
        if len(sB) < 8 or np.std(sB) < 1e-12 or np.std(dl) < 1e-12:
            return dict(spearman=np.nan, auc_pos=np.nan, n=len(sB))
        return dict(spearman=float(spearmanr(sB, dl).correlation), auc_pos=float(_auc(sB, dl > 0)), n=len(sB))
    batch_tbl = {s: {d: batch_metric(s, d) for d in DISEASE} for s in SAMPLE_SCORES}
    batch_logfo = {s: {d: {f"-{fam}": batch_metric(s, d, fam) for fam in contest_fams} for d in DISEASE} for s in SAMPLE_SCORES}

    # ---- selective risk @ 80% within-batch + global vs within-batch top-20% (primary score s_sep) ----
    def selective(score):
        kept_h, all_h, glob_top_h, wb_top_h = [], [], [], []
        alls, allh = [], []
        for d in DISEASE:
            for coh in DISEASE[d]:
                for fam in contest_fams:
                    for b in store[(d, coh, fam)]:
                        m = b["base_correct"]
                        if m.sum() < 8:
                            continue
                        s = b["post"][score][m]; h = b["harm_flip"][m]
                        k = int(0.8*len(s)); keep = np.argsort(s)[:k]
                        kept_h.append(h[keep].mean()); all_h.append(h.mean())
                        wb_top = np.argsort(-s)[:max(1, int(0.2*len(s)))]; wb_top_h.append(h[wb_top].mean())
                        alls += list(s); allh += list(h)
        alls, allh = np.array(alls), np.array(allh)
        gt = np.argsort(-alls)[:int(0.2*len(alls))]
        return dict(sel_risk_kept80=float(np.mean(kept_h)), overall=float(np.mean(all_h)),
                    withinbatch_top20_harm=float(np.mean(wb_top_h)), global_top20_harm=float(allh[gt].mean()),
                    base_rate=float(allh.mean()))

    # ---- 4-way decision (admissible = AUROC>0.5+within 0.03 of best, consistent direction across families & PD/SCZ) ----
    def adm_sample(s):
        vals = [sample_tbl[s][d] for d in DISEASE]; best = max(sample_tbl[k][d] for k in SAMPLE_SCORES for d in DISEASE)
        fams = fam_dir(s)
        return (all(v > 0.5 for v in vals if not np.isnan(v)) and all(not np.isnan(v) for v in vals)
                and all(abs(best) - v <= EQ for v in vals)
                and all((x is not None and x > 0.5) for x in fams.values()))
    def adm_batch(s):
        ok = []
        for d in DISEASE:
            sp = batch_tbl[s][d]["spearman"]; ap = batch_tbl[s][d]["auc_pos"]
            lo = all((batch_logfo[s][d][f]["spearman"] or 0) > 0 for f in batch_logfo[s][d])
            ok.append((not np.isnan(sp)) and sp > 0 and ap > 0.5 and lo)
        return all(ok)
    s_adm = {s: bool(adm_sample(s)) for s in SAMPLE_SCORES}
    b_adm = {s: bool(adm_batch(s)) for s in SAMPLE_SCORES}
    # cohort-macro robustness for the batch test (the pre-reg batch test pools batches within disease -> cohort
    # clustering; here we report per-cohort batch Spearman so a cohort-driven signal is visible, more conservative)
    def batch_by_cohort(score, disease):
        out = {}
        for coh in DISEASE[disease]:
            rows = [(b, b["mean_dl"]) for fam in contest_fams for b in store[(disease, coh, fam)]]
            sB = np.array([b["post"][score][b["base_correct"]].mean() if b["base_correct"].any() else b["post"][score].mean() for b, _ in rows])
            dl = np.array([d for _, d in rows])
            out[coh] = (float(spearmanr(sB, dl).correlation) if len(sB) > 8 and np.std(sB) > 1e-12 and np.std(dl) > 1e-12 else None)
        return out
    batch_cohort = {s: {d: batch_by_cohort(s, d) for d in DISEASE} for s in SAMPLE_SCORES}
    def batch_cohort_consistent(s):                                          # most cohorts same positive sign
        vals = [v for d in DISEASE for v in batch_cohort[s][d].values() if v is not None]
        return len(vals) >= 5 and sum(v > 0 for v in vals) >= len(vals) - 1
    b_adm_robust = {s: bool(b_adm[s] and batch_cohort_consistent(s)) for s in SAMPLE_SCORES}
    both = [s for s in SAMPLE_SCORES if s_adm[s] and b_adm_robust[s]]
    has_s, has_b = any(s_adm.values()), any(b_adm_robust.values())
    if both:
        decision = "SINGLE_SCALAR_CANDIDATE"
    elif has_s and has_b:
        decision = "TWO_LEVEL_CANDIDATE"
    elif has_s:
        decision = "POST_ALIGN_ABSTENTION_ONLY"
    elif has_b:
        decision = "ROLLBACK_ELIGIBILITY_ONLY"                              # batch-level rollback confirms; sample abstention does not
    else:
        decision = "DIAGNOSTIC_ONLY"

    flips = {}
    for s in ["dummy"]:
        pass
    hf = bf = nb = 0.0
    for k, v in store.items():
        for b in v:
            hf += b["harm_flip"].mean(); bf += b["ben_flip"].mean(); nb += 1
    summary = dict(decision=decision, sample_admissible=s_adm, batch_admissible=b_adm,
                   batch_admissible_cohort_robust=b_adm_robust, batch_spearman_by_cohort=batch_cohort, both=both,
                   sample_withinbatch_auc_harmflip=sample_tbl, batch_meanDl=batch_tbl, batch_logfo=batch_logfo,
                   adaptation_specificity=spec, per_family_direction={s: fam_dir(s) for s in SAMPLE_SCORES},
                   selective_risk={s: selective(s) for s in SAMPLE_SCORES},
                   net_flip=dict(harm_rate=hf/nb, beneficial_rate=bf/nb, net=(hf-bf)/nb),
                   mechanism_note="s_support/cmi AUROC<0.5 = anti-aligned with adaptation HARM (different endpoint "
                                  "from shift-affectedness they were validated on); NOT 'old result wrong'.",
                   seed_family=f"{SEED_BASE}+0..{R-1} (UNSEEN; A0 used deterministic gens)", freeze_a1_hash=fz["hash"])
    out = f"results/a0_prime/{fz['hash'][:16]}"; os.makedirs(out, exist_ok=True)
    json.dump(summary, open(f"{out}/a0prime_summary.json", "w"), indent=2, default=str)
    json.dump(dict(pre_registration="notes/A0_PRIME_FROZEN.md", seed_family=SEED_BASE, dumps=V4,
                   freeze_a1_hash=fz["hash"]), open(f"{out}/run_manifest.json", "w"), indent=2)
    print(f"=== A0' DECISION: {decision} ===")
    print("  SAMPLE within-batch harm-flip AUROC (PD / SCZ):")
    for s in SAMPLE_SCORES:
        print(f"    {s:11s} PD={sample_tbl[s]['PD']:.3f} SCZ={sample_tbl[s]['SCZ']:.3f}  | adm={s_adm[s]}")
    print("  adaptation-specificity (harm_flip[base-corr] vs base_err[all] vs adapt_err[all] AUROC):")
    f = lambda x: f"{x:.3f}" if x is not None else "n/a"
    for s in SAMPLE_SCORES:
        r = spec[s]; print(f"    {s:11s} harm={f(r['harm_flip'])} base_err={f(r['base_err'])} adapt_err={f(r['adapt_err'])}")
    print("  BATCH-mean vs mean Δℓ_B (Spearman | AUROC[Δℓ>0] | adm | cohort-robust):")
    for s in SAMPLE_SCORES:
        print(f"    {s:11s} PD ρ={batch_tbl[s]['PD']['spearman']:+.2f} auc={batch_tbl[s]['PD']['auc_pos']:.2f} | SCZ ρ={batch_tbl[s]['SCZ']['spearman']:+.2f} auc={batch_tbl[s]['SCZ']['auc_pos']:.2f} | adm={b_adm[s]} robust={b_adm_robust[s]}")
    print(f"  net flip: harm={hf/nb:.3f} beneficial={bf/nb:.3f} net={(hf-bf)/nb:+.3f}")
    print(f"  -> {out}/a0prime_summary.json")

if __name__ == "__main__":
    main()
