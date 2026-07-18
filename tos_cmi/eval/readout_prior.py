"""Readout Prior Decomposition core. Fixed-prior-precision heads on frozen source-standardised Z:
  L(theta) = SUM_i CE_i(theta) + tau * ||theta - theta_anchor||^2   (SUM, not mean -> prior precision fixed; shrinkage ~1/n)
Arms: H0 frozen source head; H1 hardened zero-centered ridge (anchor=0); H1-W same objective from source init
(solver init-invariance audit -> must equal H1); H2 source-centered MAP (anchor=source head); H3 bias+temperature;
H4 source-only budget-gated H2. tau selected SOURCE-ONLY, BUDGET-MATCHED via outer-source early->later pseudo-target.
Answers: is the anchoring win a genuine source-head PRIOR (H2 > H1) or a weak-fresh-baseline / optimisation-path /
budget-mismatch artifact? Pure numpy + scipy. Manuscript FROZEN."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, log_loss

from tos_cmi.eval.readout_calibration import standardize, _std, fit_head, fit_biastemp, session_macro_bacc

TAU_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def _ridge_map_obj(params, Z, y, C, Wa, ba, tau):
    """SUM_i CE_i + tau*(||W-Wa||^2 + ||b-ba||^2). Analytic gradient (SUM, not mean)."""
    d = Z.shape[1]; W = params[:C * d].reshape(C, d); b = params[C * d:]
    logits = Z @ W.T + b; logits = logits - logits.max(1, keepdims=True)
    ex = np.exp(logits); Zsum = ex.sum(1, keepdims=True); logp = logits - np.log(Zsum)
    ce = -logp[np.arange(len(y)), y].sum()
    dW = W - Wa; db = b - ba
    loss = ce + tau * ((dW ** 2).sum() + (db ** 2).sum())
    p = ex / Zsum; p[np.arange(len(y)), y] -= 1.0            # SUM (no /n)
    gW = p.T @ Z + 2 * tau * dW; gb = p.sum(0) + 2 * tau * db
    return loss, np.concatenate([gW.ravel(), gb])


def fit_ridge_map(Z, y, C, Wa=None, ba=None, tau=1.0, init=None):
    """Fixed-tau head. Wa/ba None -> zero-centered ridge (H1); else source-centered MAP (H2). init overrides the
    starting point (H1-W = same objective as H1 but init from the source head). Returns (W, b, audit)."""
    d = Z.shape[1]; Wa = np.zeros((C, d)) if Wa is None else Wa; ba = np.zeros(C) if ba is None else ba
    x0 = init if init is not None else np.concatenate([Wa.ravel(), ba])
    res = minimize(_ridge_map_obj, x0, args=(Z, y, C, Wa, ba, tau), jac=True, method="L-BFGS-B", options=dict(maxiter=1000))
    _, g = _ridge_map_obj(res.x, Z, y, C, Wa, ba, tau)
    return res.x[:C * d].reshape(C, d), res.x[C * d:], dict(success=bool(res.success), grad_norm=float(np.linalg.norm(g)))


def balanced_draw(ycal, k, rng):
    idx = []
    for c in np.unique(ycal):
        ci = np.where(ycal == c)[0]; idx.extend(rng.choice(ci, min(k, len(ci)), replace=False).tolist())
    return np.array(sorted(idx), dtype=int)


def _early_later(sess):
    s = np.asarray(list(map(str, sess))); u = sorted(set(s.tolist())); return s == u[0], np.isin(s, u[1:])


def select_tau_budget_matched(Zs_std, ys, ds, sess_s, C, k, n_draws, source_centered, rng_fn, grid=TAU_GRID):
    """Budget-matched SOURCE-ONLY tau selection: for each source subject d (anchor = source head fit on ds!=d if
    source_centered else 0), draw k class-balanced from d's EARLY session (n_draws draws), score CE on d's LATER
    session, pick tau minimising mean pseudo-query CE. Returns (tau*, full mean-CE curve)."""
    early, later = _early_later(sess_s); ds = np.asarray(ds)
    subs = [d for d in np.unique(ds)
            if ((ds == d) & early).sum() >= C and ((ds == d) & later).sum() >= C
            and len(np.unique(ys[(ds == d) & later])) >= 2 and len(np.unique(ys[ds != d])) == C]
    if len(subs) < 2:
        return 1.0, dict(reason="INSUFFICIENT_PSEUDO_SUBJECTS", n=len(subs))
    ce = {t: [] for t in grid}
    for d in subs:
        Wa, ba = (fit_head(Zs_std[ds != d], ys[ds != d], C) if source_centered else (None, None))
        cal_pool = np.where((ds == d) & early)[0]; qy = np.where((ds == d) & later)[0]
        ycd = ys[cal_pool]
        for di in range(n_draws):
            rng = rng_fn(d, di)
            drw = cal_pool[balanced_draw(ycd, k, rng)] if k != "Full" else cal_pool
            for t in grid:
                W, b, _ = fit_ridge_map(Zs_std[drw], ys[drw], C, Wa, ba, t)
                lg = Zs_std[qy] @ W.T + b; lg -= lg.max(1, keepdims=True)
                lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
                ce[t].append(-lp[np.arange(len(qy)), ys[qy]].mean())
    mean_ce = {t: float(np.mean(v)) for t, v in ce.items()}
    return min(mean_ce, key=mean_ce.get), dict(mean_ce=mean_ce, n_pseudo=len(subs))


def source_gate(Zs_std, ys, ds, sess_s, C, k, tau_s, n_draws, rng_fn):
    """H4 gate g_k = 1[mean delta - SE(delta) > 0], delta_d = U_H2 - U_H0 on the source pseudo-target (bAcc), budget-
    matched. SOURCE-ONLY (never sees target). Returns (g_k, diag)."""
    early, later = _early_later(sess_s); ds = np.asarray(ds)
    subs = [d for d in np.unique(ds)
            if ((ds == d) & early).sum() >= C and ((ds == d) & later).sum() >= C
            and len(np.unique(ys[(ds == d) & later])) >= 2 and len(np.unique(ys[ds != d])) == C]
    deltas = []
    for d in subs:
        Ws, bs = fit_head(Zs_std[ds != d], ys[ds != d], C)      # source head (anchor + H0 reference)
        cal_pool = np.where((ds == d) & early)[0]; qy = np.where((ds == d) & later)[0]; ycd = ys[cal_pool]
        u0 = balanced_accuracy_score(ys[qy], (Zs_std[qy] @ Ws.T + bs).argmax(1))
        for di in range(n_draws):
            drw = cal_pool[balanced_draw(ycd, k, rng_fn(d, di))] if k != "Full" else cal_pool
            W, b, _ = fit_ridge_map(Zs_std[drw], ys[drw], C, Ws, bs, tau_s)
            deltas.append(balanced_accuracy_score(ys[qy], (Zs_std[qy] @ W.T + b).argmax(1)) - u0)
    if not deltas:
        return 0, dict(reason="NO_PSEUDO", n=0)
    m = float(np.mean(deltas)); se = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 1.0
    return int(m - se > 0), dict(mean=m, se=se, n=len(deltas))


def _nll(W, b, Zq, yq):
    return float(log_loss(yq, _softmax(Zq @ W.T + b), labels=list(range(W.shape[0]))))


def _softmax(logits):
    logits = logits - logits.max(1, keepdims=True); ex = np.exp(logits); return ex / ex.sum(1, keepdims=True)
