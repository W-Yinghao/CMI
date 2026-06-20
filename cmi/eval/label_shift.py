"""CIPC — Conditional-Invariant Prior Correction (transductive label-shift correction).

Under label shift pi_T(y) != pi_S(y) with conditional invariance P_S(z|y)=P_T(z|y) (enforced by lpc_prior),
the Bayes-optimal target rule is p_T(y|x) ∝ p_S(y|x) * pi_T(y)/pi_S(y). ERM bakes in pi_S; correcting to the
(unlabeled-target-estimated) pi_T re-ranks the per-sample argmax. This is a BOUNDARY-MOVER, not a symmetric
divergence — it escapes the accuracy-parity trap of pure invariance, and balanced accuracy rewards it exactly
(it recovers minority-class recall lost to prior mismatch). Post-hoc on saved softmax/logits => near-zero GPU.

pi_T estimated transductively (label-free) from unlabeled target predictions via BBSE (Lipton et al. 2018,
confusion-matrix solve) or MLLS-EM (Alexandari et al. 2020, more stable). Null-safe: if pi_T==pi_S the
correction factor is 1 (CIPC ⊇ lpc_prior ⊇ ERM).
"""
import numpy as np


def _simplex_clip(p, eps=1e-8):
    p = np.clip(np.asarray(p, float), eps, None)
    return p / p.sum()


def _softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def confusion_source(prob_src_eval, y_src_eval, n_cls):
    """C[k,j] = P_S(yhat=k | y=j) on a held-out SOURCE split (columns sum to 1)."""
    yhat = np.asarray(prob_src_eval).argmax(1)
    y = np.asarray(y_src_eval)
    C = np.zeros((n_cls, n_cls))
    for j in range(n_cls):
        m = (y == j)
        if m.sum() == 0:
            C[j, j] = 1.0
            continue
        for k in range(n_cls):
            C[k, j] = (yhat[m] == k).mean()
    return C


def _mlls_em(prob_tgt, pi_S, n_cls, iters=200, shrink=0.0, tol=1e-8):
    """MLLS-EM: estimate pi_T by EM on SOFT target predictions (stable, stays in simplex)."""
    pi_S = _simplex_clip(pi_S)
    pi_T = pi_S.copy()
    p = np.clip(np.asarray(prob_tgt, float), 1e-8, 1.0)
    for _ in range(iters):
        w = pi_T / pi_S
        q = p * w[None, :]
        q = q / q.sum(1, keepdims=True)
        new = q.mean(0)
        new = _simplex_clip((1.0 - shrink) * new + shrink * pi_S)
        if np.abs(new - pi_T).max() < tol:
            pi_T = new
            break
        pi_T = new
    return pi_T


def bbse_prior(prob_src_eval, y_src_eval, prob_tgt, pi_S, n_cls, method="em", shrink=0.0):
    """Estimate target class prior pi_T from unlabeled target predictions.
    method='em'  -> MLLS-EM on soft preds (default; robust to near-singular C).
    method='solve' -> BBSE hard confusion-matrix solve, with EM fallback on failure/out-of-simplex."""
    pi_S = _simplex_clip(pi_S)
    if method == "em":
        return _mlls_em(prob_tgt, pi_S, n_cls, shrink=shrink)
    # BBSE confusion-matrix solve: mu_hat[k] = sum_j P(yhat=k|y=j) pi_T(j) = (C @ pi_T)[k]
    C = confusion_source(prob_src_eval, y_src_eval, n_cls)
    mu_hat = np.bincount(np.asarray(prob_tgt).argmax(1), minlength=n_cls) / len(prob_tgt)
    try:
        pi_T = np.linalg.solve(C, mu_hat)
        if not np.all(np.isfinite(pi_T)) or pi_T.min() < -1e-3:
            raise np.linalg.LinAlgError
        pi_T = _simplex_clip(pi_T)
    except np.linalg.LinAlgError:
        pi_T = _mlls_em(prob_tgt, pi_S, n_cls, shrink=max(shrink, 0.1))
    if shrink > 0:
        pi_T = _simplex_clip((1.0 - shrink) * pi_T + shrink * pi_S)
    return pi_T


def fit_temperature(logits_tgt, grid=None):
    """Transductive temperature: scalar T minimizing mean target prediction entropy (low-density separation),
    with a floor to avoid over-sharpening. Label-free."""
    if grid is None:
        grid = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
    best_T, best_H = 1.0, np.inf
    for T in grid:
        p = _softmax(np.asarray(logits_tgt, float) / T)
        H = float(-(p * np.log(np.clip(p, 1e-8, 1))).sum(1).mean())
        if H < best_H:
            best_H, best_T = H, T
    return best_T


def apply_correction(prob_tgt, pi_T, pi_S, gate_l1=0.0, T=1.0, logits_tgt=None):
    """Re-prior the target predictions: corrected ∝ p(y|x) * pi_T/pi_S. Optionally temperature-scale logits
    first. NULL-SAFE GATE: if ||pi_T - pi_S||_1 <= gate_l1 return the uncorrected probs unchanged."""
    pi_T = _simplex_clip(pi_T)
    pi_S = _simplex_clip(pi_S)
    if float(np.abs(pi_T - pi_S).sum()) <= gate_l1:
        return np.asarray(prob_tgt, float)
    if logits_tgt is not None and T != 1.0:
        p = _softmax(np.asarray(logits_tgt, float) / T)
    else:
        p = np.asarray(prob_tgt, float)
    corr = p * (pi_T / pi_S)[None, :]
    return corr / corr.sum(1, keepdims=True)


def _sqrtm(M, eps, inv=False):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    s = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return (V * s) @ V.T


def feature_coral_recenter(z_src_pool, z_tgt, eps=1e-5):
    """Transductive covariate alignment (CORAL/EA in FEATURE space): whiten target features to the source-pool
    mean+covariance, z_t -> Sigma_S^{1/2} Sigma_T^{-1/2} (z_t - mu_T) + mu_S. Label-free; corrects an unseen
    per-domain affine covariate shift so the source boundary fits the target. NULL-SAFE: if P_T(z)=P_S(z) the
    map is ~identity. THIS is the balanced-accuracy lever (re-prioring is not; balanced acc is prior-invariant
    up to the boundary, optimized by a uniform prior)."""
    z_src_pool, z_tgt = np.asarray(z_src_pool, float), np.asarray(z_tgt, float)
    mu_S, mu_T = z_src_pool.mean(0), z_tgt.mean(0)
    d = z_src_pool.shape[1]
    Cs = np.cov(z_src_pool, rowvar=False) + eps * np.eye(d)
    Ct = np.cov(z_tgt, rowvar=False) + eps * np.eye(d)
    W = _sqrtm(Cs, eps) @ _sqrtm(Ct, eps, inv=True)      # Sigma_S^{1/2} Sigma_T^{-1/2}
    return (z_tgt - mu_T) @ W.T + mu_S


def transduct_predict(z_src, y_src, z_tgt, pi_S, n_cls, mode="coral", shrink=0.1, gate_l1=0.0):
    """Backbone-agnostic transductive correction on penultimate features z (one source-fit linear head).
    mode: 'probe' (head on raw target z, baseline) | 'coral' (CORAL-recenter target z first; the bAcc lever) |
          'prior' (probe + BBSE re-prior; plain-acc lever) | 'coral_prior' (both).
    Returns dict(prob, prob_probe_raw, pi_T)."""
    from sklearn.linear_model import LogisticRegression
    z_src, z_tgt = np.asarray(z_src, float), np.asarray(z_tgt, float)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(z_src, y_src)
    cls = clf.classes_

    def full(p):                                          # pad to n_cls columns if a class is absent in source-eval
        out = np.zeros((len(p), n_cls)); out[:, cls] = p; return out
    prob_raw = full(clf.predict_proba(z_tgt))
    pi_T = None
    if mode == "pmct":                                    # prior-matched conditional transport (unified)
        _, prob, pi_T = pmct_transport(z_src, y_src, z_tgt, n_cls, clf=clf)
        return dict(prob=prob, prob_probe_raw=prob_raw, pi_T=pi_T.tolist())
    z_in = feature_coral_recenter(z_src, z_tgt) if mode in ("coral", "coral_prior") else z_tgt
    prob = full(clf.predict_proba(z_in))
    if mode in ("prior", "coral_prior"):
        prob_se = full(clf.predict_proba(z_src))
        pi_T = bbse_prior(prob_se, y_src, prob, pi_S, n_cls, method="em", shrink=shrink)
        prob = apply_correction(prob, pi_T, pi_S, gate_l1=gate_l1)
    return dict(prob=prob, prob_probe_raw=prob_raw, pi_T=(pi_T.tolist() if pi_T is not None else None))


def _shrink_cov(C, rho, eps):
    d = C.shape[0]
    return (1 - rho) * C + rho * (np.trace(C) / d) * np.eye(d) + eps * np.eye(d)


def pmct_transport(z_src, y_src, z_tgt, n_cls, alpha=1.0, eps=1e-3, rho=0.2, em_iters=3, clf=None,
                   support_gate=True):
    """Prior-Matched Conditional Transport. Align the target to a PRIOR-MATCHED source mixture (not the pooled
    source), so label-prior shift is not mistaken for covariate shift. Unifies CORAL/EA: one-hot pi_hat_T ->
    single-class reference (no catastrophic single-class collapse); mixed pi -> mixture-matched CORAL. Uses the
    SOURCE class-conditional moments {mu_y^S, Sigma_y^S} (the same P(z|y) lpc_prior makes invariant) — the shared
    statistic that couples Stage 1 and Stage 2. EM-iterates pi_hat_T (align -> re-estimate -> re-align) since the
    prior estimate on a shifted target is itself biased. Returns (z_tilde, prob, pi_T)."""
    from sklearn.linear_model import LogisticRegression
    z_src, z_tgt = np.asarray(z_src, float), np.asarray(z_tgt, float)
    d = z_src.shape[1]
    mu_y = np.stack([z_src[y_src == c].mean(0) if (y_src == c).any() else z_src.mean(0) for c in range(n_cls)])
    Sig_y = [_shrink_cov(np.cov(z_src[y_src == c], rowvar=False), rho, eps) if (y_src == c).sum() > d
             else np.eye(d) for c in range(n_cls)]
    mu_T = z_tgt.mean(0)
    Wt_inv = _sqrtm(_shrink_cov(np.cov(z_tgt, rowvar=False), rho, eps), eps, inv=True)   # (Sigma_T)^{-1/2}
    if clf is None:
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(z_src, y_src)
    cls = clf.classes_

    def _full(p):
        out = np.zeros((len(p), n_cls)); out[:, cls] = p; return out
    # bootstrap pi_hat_T from a COVARIATE-corrected (global-CORAL) classification, not the raw (miscalibrated)
    # target — avoids the chicken-and-egg where a strongly-shifted target gives a wrong prior.
    pi = _simplex_clip(_full(clf.predict_proba(feature_coral_recenter(z_src, z_tgt))).mean(0))
    # SUPPORT-AWARE GATE alpha_T = alpha0 * g_support * g_cov: when the target support collapses toward a single
    # class (H(pi_hat_T)->0) or there are too few target samples for a reliable d-dim covariance, alpha_T->0 and
    # PMCT degrades to IDENTITY (null-safe) instead of distorting the single-class target (the R6/MUMTAZ crash).
    g_cov = float(np.clip(len(z_tgt) / (2.0 * d), 0.0, 1.0)) if support_gate else 1.0
    z_tilde, prob, alpha_eff = z_tgt, None, alpha
    for _ in range(max(1, em_iters)):
        if support_gate:
            H = float(-(pi * np.log(np.clip(pi, 1e-12, 1.0))).sum())
            g_support = (H / np.log(n_cls)) if n_cls > 1 else 1.0        # 1 at balanced, ->0 at single-class
            alpha_eff = alpha * g_support * g_cov
        mu_R = (pi[:, None] * mu_y).sum(0)
        Sig_R = sum(pi[c] * (Sig_y[c] + np.outer(mu_y[c] - mu_R, mu_y[c] - mu_R)) for c in range(n_cls))
        M = _sqrtm(Sig_R + eps * np.eye(d), eps) @ Wt_inv                     # Sigma_R^{1/2} Sigma_T^{-1/2}
        Tz = mu_R + (z_tgt - mu_T) @ M.T
        z_tilde = (1 - alpha_eff) * z_tgt + alpha_eff * Tz
        p = clf.predict_proba(z_tilde)
        prob = np.zeros((len(p), n_cls)); prob[:, cls] = p
        pi_new = _simplex_clip(prob.mean(0))
        if np.abs(pi_new - pi).max() < 1e-3:
            pi = pi_new; break
        pi = pi_new
    return z_tilde, prob, pi


def transduct_all(z_src, y_src, z_tgt, pi_S, n_cls, shrink=0.1):
    """Compute the predictions for ALL transduct modes from one (z_src, z_tgt) embedding — the ablation ladder
    in a single pass. Returns {mode: prob} for probe/coral/prior/coral_prior."""
    out = {}
    for mode in ("probe", "coral", "prior", "coral_prior", "pmct"):
        out[mode] = transduct_predict(z_src, y_src, z_tgt, pi_S, n_cls, mode=mode, shrink=shrink)["prob"]
    return out


def cipc_predict(prob_src_eval, y_src_eval, prob_tgt, pi_S, n_cls,
                 logits_tgt=None, method="em", shrink=0.0, gate_l1=0.0, use_temperature=False):
    """Full CIPC post-hoc step: estimate pi_T then re-prior. Returns (corrected_prob, pi_T_hat)."""
    pi_T = bbse_prior(prob_src_eval, y_src_eval, prob_tgt, pi_S, n_cls, method=method, shrink=shrink)
    T = fit_temperature(logits_tgt) if (use_temperature and logits_tgt is not None) else 1.0
    corr = apply_correction(prob_tgt, pi_T, pi_S, gate_l1=gate_l1, T=T, logits_tgt=logits_tgt)
    return corr, pi_T
