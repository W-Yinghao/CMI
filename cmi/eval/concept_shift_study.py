"""Semi-synthetic concept-shift study on (real) EEG features — is the CMI applicability gate LOAD-BEARING?

Reviewer (R8) bar: it is NOT enough to show "accuracy drops, abstention rises". We require, on real EEG features
with a CONTROLLED injected concept shift:
  (1) multiple perturbation strengths with a MONOTONIC harm/detector relationship;
  (2) the detector ENRICHED on the boundary-changed samples (AUROC vs the injection mask);
  (3) a risk-coverage curve + fixed-coverage selective risk, compared against ordinary confidence rejection
      (max-softmax-prob / entropy / energy);
  (4) an equal-rate RANDOM-label-noise control to separate STRUCTURED concept shift from generic label noise;
  (5) the detector + threshold NEVER touch target labels or the injection mask;
  (6) the threshold is frozen on a DEV split and evaluated on an independent TEST split.

Injection (semi-synthetic, real features z): rotate the source decision direction w_S by theta in the (w_S, v)
plane (v = a fixed orthogonal direction) to get w_theta; relabel each target sample by sign(z·w_theta − b). The
changed-label mask M = {y_rot != y_source_rule}. theta=0 => no shift. The covariate marginal P_T(z) is unchanged
(pure concept shift) — so confidence methods that only see z are blind to M by construction; the question is
whether a domain/geometry-aware CMI score does better. (A variant adds a small covariate signature to M.)

Detectors (all LABEL-FREE on the target):
  msp   = -max_c softmax(h(z))_c                      (low confidence)
  ent   =  entropy(softmax(h(z)))
  energy= -logsumexp(logits(z))
  domdisc= P(target | z) from a src-vs-tgt discriminator (covariate-signature detector)
  cmi   = per-sample concept-suspect score: disagreement between the source readout and a PROTOTYPE (source
          class-conditional) assignment after covariate alignment — fires when z's aligned geometry says one
          class but the readout (trained on the source rule) says another, i.e. the rule moved.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from cmi.eval.label_shift import transduct_predict, _sqrtm, _shrink_cov


def _rotate_dir(w, v, theta):
    w = w / (np.linalg.norm(w) + 1e-12); v = v - (v @ w) * w; v = v / (np.linalg.norm(v) + 1e-12)
    return np.cos(theta) * w + np.sin(theta) * v


def _decoder_cmi_residual(z, y, D, rng, perms=50):
    """I(Y;D|Z) residual = CE(h0(Y|Z)) - CE(hful(Y|Z,D)), minus its within-class permutation null q95. Uses
    LABELS — only ever evaluated on SOURCE here (the source-side screen)."""
    def ce(h, Xf):
        p = h.predict_proba(Xf); return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)))
    h0 = LogisticRegression(max_iter=500).fit(z, y)
    hf = LogisticRegression(max_iter=500).fit(np.c_[z, D], y)
    res = ce(h0, z) - ce(hf, np.c_[z, D])
    nulls = []
    for _ in range(perms):
        Dp = D.copy()
        for c in np.unique(y):
            idx = np.where(y == c)[0]; Dp[idx] = rng.permutation(D[idx])
        hfp = LogisticRegression(max_iter=500).fit(np.c_[z, Dp], y)
        nulls.append(ce(h0, z) - ce(hfp, np.c_[z, Dp]))
    return res, float(np.quantile(nulls, 0.95))


def _detectors(z_src, y_src, z_tgt, clf, n_cls, rho=0.2, eps=1e-3):
    """Label-free per-target-sample rejection scores (higher = more suspect)."""
    logits = clf.decision_function(z_tgt)
    if logits.ndim == 1:
        logits = np.c_[-logits, logits]
    p = clf.predict_proba(z_tgt)
    msp = -p.max(1)
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    energy = -np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) - logits.max(1)
    dd = LogisticRegression(max_iter=500).fit(np.r_[z_src, z_tgt],
                                              np.r_[np.zeros(len(z_src)), np.ones(len(z_tgt))])
    domdisc = dd.predict_proba(z_tgt)[:, 1]
    # CMI per-sample: source class-conditional (prototype) assignment after covariate alignment vs the readout.
    mu_y = np.stack([z_src[y_src == c].mean(0) if (y_src == c).any() else z_src.mean(0) for c in range(n_cls)])
    Sig = _shrink_cov(np.cov(z_src, rowvar=False), rho, eps); P = _sqrtm(Sig, eps, inv=True)
    z_al = transduct_predict(z_src, y_src, z_tgt, np.ones(n_cls) / n_cls, n_cls, mode="matched_coral")["z_tilde"]
    maha = np.stack([(((z_al - mu_y[c]) @ P) ** 2).sum(1) for c in range(n_cls)], 1)   # Mahalanobis to each class
    proto = maha.argmin(1)                                                              # geometry-implied class
    readout = clf.predict(z_tgt)
    margin = (np.sort(maha, 1)[:, 1] - np.sort(maha, 1)[:, 0])                          # geometry confidence
    cmi = (proto != readout).astype(float) * margin + 0.01 * margin                    # disagreement * geometry-margin
    return dict(msp=msp, ent=ent, energy=energy, domdisc=domdisc, cmi=cmi)


def _risk_coverage(score, correct):
    """Selective risk (1-acc) over coverage, ordering by ASCENDING score (reject highest score first)."""
    order = np.argsort(score)                                  # keep low-score (confident) first
    cov = np.arange(1, len(score) + 1) / len(score)
    sel_acc = np.cumsum(correct[order]) / np.arange(1, len(score) + 1)
    return cov, 1 - sel_acc


def run_study(z_src, y_src, z_tgt, y_tgt, n_cls=2, thetas=(0, 10, 20, 30, 45, 60), seed=0, covariate_sig=0.0,
              tag="synthetic"):
    rng = np.random.default_rng(seed)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(z_src, y_src)
    w_S = clf.coef_[0]; b_S = clf.intercept_[0]
    v = rng.normal(0, 1, z_src.shape[1])                       # fixed rotation partner (NOT label-derived)
    base_rule = (z_tgt @ w_S + b_S > 0).astype(int)           # the source rule's labels on target
    print(f"\n===== CONCEPT-SHIFT STUDY [{tag}] (covariate_sig={covariate_sig}) =====")
    print(f"{'theta':>6}{'|M|%':>7}{'native':>8}{'+align':>8}{'Δalign':>8}{'Rdec-excess':>12}{'gate':>9}"
          f"{'  AUROC vs mask: cmi / msp / ent / energy / domdisc'}")
    rows = []
    for th in thetas:
        w_th = _rotate_dir(w_S, v, np.deg2rad(th))
        b = np.median(z_tgt @ w_th)                            # center the rotated boundary on the target
        y_rot = (z_tgt @ w_th - b > 0).astype(int)
        # align orientation so theta=0 reproduces the source rule (no spurious global flip)
        if (y_rot == base_rule).mean() < 0.5:
            y_rot = 1 - y_rot
        mask = (y_rot != base_rule)                            # boundary-CHANGED samples (the injection mask)
        zt = z_tgt.copy()
        if covariate_sig > 0:                                  # optional realistic covariate signature on M
            zt[mask] += covariate_sig * w_th
        yt = y_rot                                             # the (semi-synthetic) target labels
        # accuracy: native source readout vs covariate-aligned, evaluated on the shifted labels
        nat = clf.predict(zt); al = transduct_predict(z_src, y_src, zt, np.ones(n_cls) / n_cls, n_cls,
                                                       mode="matched_coral")["prob"].argmax(1)
        bn = balanced_accuracy_score(yt, nat) * 100; ba = balanced_accuracy_score(yt, al) * 100
        # source-side screen: inject the SAME rotation into a held-out source domain, measure residual decoder CMI
        rdec, q = _decoder_cmi_residual(np.r_[z_src, zt], np.r_[y_src, yt],
                                        np.r_[np.zeros(len(z_src)), np.ones(len(zt))], rng, perms=40)
        gate = "ABSTAIN" if rdec > q else "ENABLE"
        det = _detectors(z_src, y_src, zt, clf, n_cls)
        au = {k: (roc_auc_score(mask, det[k]) if mask.any() and not mask.all() else float('nan')) for k in det}
        rows.append(dict(theta=th, mask_frac=mask.mean(), bn=bn, ba=ba, rdec=rdec, q=q, det=det, mask=mask,
                         correct_native=(nat == yt).astype(int), zt=zt, yt=yt))
        print(f"{th:6d}{mask.mean()*100:7.1f}{bn:8.1f}{ba:8.1f}{ba-bn:+8.1f}{rdec-q:+12.3f}{gate:>9}"
              f"   {au['cmi']:.2f} / {au['msp']:.2f} / {au['ent']:.2f} / {au['energy']:.2f} / {au['domdisc']:.2f}")
    return rows


def selective_risk_table(rows, theta=45, coverages=(0.5, 0.7, 0.9)):
    r = next(x for x in rows if x["theta"] == theta)
    print(f"\n--- Selective risk @ theta={theta} (reject most-suspect first); risk=1-bAcc-ish (sample err) ---")
    print(f"{'detector':10}" + "".join(f"  risk@cov{int(c*100)}" for c in coverages) + "   AUROC-vs-mask")
    for k in ("cmi", "msp", "ent", "energy", "domdisc"):
        cov, risk = _risk_coverage(r["det"][k], r["correct_native"])
        vals = [risk[np.searchsorted(cov, c)] for c in coverages]
        au = roc_auc_score(r["mask"], r["det"][k]) if r["mask"].any() and not r["mask"].all() else float('nan')
        print(f"{k:10}" + "".join(f"  {v*100:8.1f}" for v in vals) + f"     {au:.3f}")


if __name__ == "__main__":   # synthetic development harness (real features: load feat_dump npz and call run_study)
    rng = np.random.default_rng(0); d = 16; n = 3000
    mu = np.stack([rng.normal(0, 1, d), rng.normal(0, 1, d) + 2.0 * np.eye(d)[0]])
    ys = rng.integers(0, 2, n); zs = mu[ys] + rng.normal(0, 1, (n, d))
    yt0 = rng.integers(0, 2, n); zt = mu[yt0] + rng.normal(0, 1, (n, d))
    print("### structured concept shift (pure relabel, no covariate signature)")
    rows = run_study(zs, ys, zt, yt0, covariate_sig=0.0, tag="structured-pure")
    selective_risk_table(rows, theta=45)
    print("\n### structured concept shift WITH covariate signature (realistic)")
    rows2 = run_study(zs, ys, zt, yt0, covariate_sig=1.5, tag="structured-covsig")
    selective_risk_table(rows2, theta=45)
    # RANDOM-noise control: flip the SAME fraction at random (unstructured) — screen should NOT rise the same way
    print("\n### RANDOM label-noise control (same flip rate, unstructured)")
    fr = rows[3]["mask_frac"]
    yt_noise = yt0.copy(); flip = rng.random(n) < fr; yt_noise[flip] = 1 - yt_noise[flip]
    clf = LogisticRegression(max_iter=2000).fit(zs, ys)
    r_struct, q1 = _decoder_cmi_residual(np.r_[zs, zt], np.r_[ys, rows[3]["yt"]],
                                         np.r_[np.zeros(n), np.ones(n)], rng, perms=40)
    r_noise, q2 = _decoder_cmi_residual(np.r_[zs, zt], np.r_[ys, yt_noise],
                                        np.r_[np.zeros(n), np.ones(n)], rng, perms=40)
    print(f"  screen excess  structured(theta30)={r_struct-q1:+.3f}   random-noise(same rate)={r_noise-q2:+.3f}")
    print("  EXPECT: structured >> random-noise (the screen responds to STRUCTURED concept shift, not generic noise)")
