"""Adversarial audit of the A0 DIAGNOSTIC_ONLY verdict — try HARD to find a deployable harm signal A0's primary
lens (Spearman-on-dloss + per-cohort sign-consistency) could have masked. If a signal survives here, A0 was too
pessimistic; if nothing beats chance, DIAGNOSTIC_ONLY is robust. GPU-free; reuses the frozen A0 machinery.

Angles: (1) harm-flip AUROC on the base-correct subset; (2) selective-risk @ coverage; (3) POST-alignment gate
space T(z); (4) 2-feature combo (leave-one-cohort-out logistic); (5) within-disease pooling.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from cmi.eval.label_shift import transduct_predict
from a0_falsification import build_state, sample_scores, gen, DISEASE, SAMPLE_SCORES, SEV, GEN_FAMILIES, SHRINK, B, V4

def cell(cond, coh, gname, sev, rng):
    o = np.load(f"{V4}/audit_{cond}_{coh}_erm_0.npz", allow_pickle=True)
    zev, yev = np.asarray(o["z_ev"], float), np.asarray(o["y_ev"]).astype(int)
    zte, yte = np.asarray(o["z_te"], float), np.asarray(o["y_te"]).astype(int)
    order = np.argsort(np.asarray(o["window_index_te"])); zte, yte = zte[order], yte[order]
    pi = np.bincount(np.concatenate([np.asarray(o["y_se"]).astype(int), yev]), minlength=2).astype(float); pi /= pi.sum()
    st = build_state(zev, yev); zp, yp = gen(gname, sev, zte, yte, st, rng)
    rows = []
    for s in range(0, len(zp), B):
        bz, by = zp[s:s + B], yp[s:s + B]
        if len(bz) < 8 or len(set(by)) < 2:
            continue
        r = transduct_predict(zev, yev, bz, pi, 2, mode="matched_coral", shrink=SHRINK)
        base, adapt, ztil = np.asarray(r["prob_probe_raw"], float), np.asarray(r["prob"], float), np.asarray(r["z_tilde"], float)
        pre = sample_scores(st, bz, base)                      # PRE-alignment gate space
        post = sample_scores(st, ztil, adapt)                  # POST-alignment gate space
        bc = base.argmax(1) == by; aw = adapt.argmax(1) != by; hf = bc & aw
        lb = -np.log(np.clip(base[np.arange(len(by)), by], 1e-9, 1)); la = -np.log(np.clip(adapt[np.arange(len(by)), by], 1e-9, 1))
        for i in range(len(bz)):
            rows.append(dict(cohort=coh, disease=cond, gen=gname, base_correct=bool(bc[i]), harm_flip=bool(hf[i]),
                             dloss=float(la[i] - lb[i]),
                             **{f"pre_{k}": float(pre[k][i]) for k in SAMPLE_SCORES},
                             **{f"post_{k}": float(post[k][i]) for k in SAMPLE_SCORES}))
    return rows

def auc(score, label):
    label = np.asarray(label)
    if len(set(label)) < 2 or np.std(score) < 1e-12:
        return np.nan
    return roc_auc_score(label, score)

def sel_risk(score, harm, cov_keep=0.8):
    """harm-flip rate among the cov_keep fraction with the LOWEST score (abstain the top-(1-keep) by score)."""
    n = len(score); k = int(cov_keep * n); keep = np.argsort(score)[:k]
    return float(np.mean(np.asarray(harm)[keep])), float(np.mean(harm))

def main():
    allrows = []
    for cond, cohs in DISEASE.items():
        for coh in cohs:
            for g in ["clean"] + GEN_FAMILIES:
                for sev in SEV[g]:
                    allrows += cell(cond, coh, g, sev, np.random.default_rng(hash((coh, g, sev)) % 2**31))
    R = allrows
    print("=== ADVERSARIAL AUDIT of A0 DIAGNOSTIC_ONLY ===\n")
    # ---- (1)+(2)+(3): harm-flip AUROC (base-correct subset) + selective-risk, PRE vs POST space ----
    for scope_name, scope in [("highmargin_cbw", lambda r: r["gen"] == "highmargin_cbw"),
                              ("all-contest", lambda r: r["gen"] in GEN_FAMILIES)]:
        S = [r for r in R if scope(r) and r["base_correct"]]    # harm-flip only defined on base-correct samples
        harm = [r["harm_flip"] for r in S]
        print(f"--- {scope_name}: harm-flip AUROC on base-correct subset (n={len(S)}, harm-rate={np.mean(harm):.3f}) ---")
        for space in ("pre", "post"):
            line = []
            for k in SAMPLE_SCORES:
                a = auc([r[f"{space}_{k}"] for r in S], harm)
                kept, overall = sel_risk(np.array([r[f"{space}_{k}"] for r in S]), harm)
                line.append(f"{k}={a:.2f}(sr{kept:.2f}/{overall:.2f})")
            print(f"  [{space:4s}] " + "  ".join(line))
        # ---- (4): 2-feature combo, leave-one-cohort-out (no overfitting) ----
        feats = [f"pre_{k}" for k in SAMPLE_SCORES]
        cohs_all = sorted(set(r["cohort"] for r in S))
        oof = np.full(len(S), np.nan)
        Sidx = {c: [i for i, r in enumerate(S) if r["cohort"] == c] for c in cohs_all}
        for c in cohs_all:
            tr = [i for i in range(len(S)) if S[i]["cohort"] != c]
            if len(set(harm[i] for i in tr)) < 2:
                continue
            clf = LogisticRegression(max_iter=500).fit([[S[i][f] for f in feats] for i in tr], [harm[i] for i in tr])
            for i in Sidx[c]:
                oof[i] = clf.predict_proba([[S[i][f] for f in feats]])[0, 1]
        m = ~np.isnan(oof)
        combo_auc = auc(oof[m], np.asarray(harm)[m])
        print(f"  [combo] 4-feature LOCO logistic harm-flip AUROC = {combo_auc:.2f}\n")
    # ---- (5): within-disease pooled Spearman (more power than per-cohort) ----
    from scipy.stats import spearmanr
    print("--- within-disease pooled Spearman(score, dloss) on highmargin_cbw (power check) ---")
    for cond in DISEASE:
        S = [r for r in R if r["gen"] == "highmargin_cbw" and r["disease"] == cond]
        dl = [r["dloss"] for r in S]
        print(f"  {cond} (n={len(S)}): " + "  ".join(f"{k}={spearmanr([r[f'pre_{k}'] for r in S], dl).correlation:+.2f}" for k in SAMPLE_SCORES))
    print("\nVERDICT: A0 DIAGNOSTIC_ONLY is ROBUST iff every AUROC ~0.5 and selective-risk does not beat overall.")

if __name__ == "__main__":
    main()
