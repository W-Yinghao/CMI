"""Target Readout Calibration Ladder — core (frozen encoder; only the readout head varies). All heads act on
SOURCE-standardized whitened(-deleted) features. Four heads:
  H0 frozen source head (no target labels)
  H1 fresh target head (logistic from k cal labels)
  H2 source-anchored MAP head (PRIMARY): min CE + alpha*||W-Ws||^2 + alpha*||b-bs||^2, alpha selected source-only
     via an outer-source early->later pseudo-target protocol (same alpha on bias)
  H3 bias-and-temperature calibration: fix Ws, optimise a per-class bias + one positive temperature T
Utility = session-macro query bAcc. Firewall: target QUERY (X,Y) enters ONLY the final utility; cal Y only adapts the
head; alpha is chosen from SOURCE only. Pure numpy + scipy. Manuscript FROZEN."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

RIDGE = 1e-6            # tiny conditioning ridge (H0/H1 softmax is otherwise shift-degenerate)
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def standardize(Zs):
    mu, sd = Zs.mean(0), Zs.std(0) + 1e-8
    return mu, sd


def _std(Z, mu, sd):
    return (Z - mu) / sd


def _softmax_obj(params, Z, y, C, Ws, bs, alpha):
    d = Z.shape[1]; W = params[:C * d].reshape(C, d); b = params[C * d:]
    logits = Z @ W.T + b; logits = logits - logits.max(1, keepdims=True)
    ex = np.exp(logits); Zsum = ex.sum(1, keepdims=True); logp = logits - np.log(Zsum)
    ce = -logp[np.arange(len(y)), y].mean()
    aW = Ws if Ws is not None else np.zeros_like(W); ab = bs if bs is not None else np.zeros_like(b)
    a = alpha + RIDGE
    loss = ce + a * (((W - aW) ** 2).sum() + ((b - ab) ** 2).sum())
    p = ex / Zsum; p[np.arange(len(y)), y] -= 1.0; p /= len(y)
    gW = p.T @ Z + 2 * a * (W - aW); gb = p.sum(0) + 2 * a * (b - ab)
    return loss, np.concatenate([gW.ravel(), gb])


def fit_head(Z, y, C, Ws=None, bs=None, alpha=0.0):
    """Softmax head via L-BFGS. Ws/bs None -> unanchored (H0 source / H1 fresh); else MAP-anchored (H2)."""
    d = Z.shape[1]
    x0 = np.concatenate([(Ws if Ws is not None else np.zeros((C, d))).ravel(),
                         (bs if bs is not None else np.zeros(C))])
    res = minimize(_softmax_obj, x0, args=(Z, y, C, Ws, bs, alpha), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=500))
    return res.x[:C * d].reshape(C, d), res.x[C * d:]


def _biastemp_obj(params, WsZ, y, C):
    """params = [logT, b_0..b_{C-1}]; logits = exp(logT)*WsZ + b. Numeric grad via analytic softmax."""
    T = np.exp(params[0]); b = params[1:]
    logits = T * WsZ + b; logits = logits - logits.max(1, keepdims=True)
    ex = np.exp(logits); Zsum = ex.sum(1, keepdims=True); logp = logits - np.log(Zsum)
    ce = -logp[np.arange(len(y)), y].mean()
    p = ex / Zsum; p[np.arange(len(y)), y] -= 1.0; p /= len(y)
    gb = p.sum(0); glogT = float((p * WsZ).sum() * T)
    return ce, np.concatenate([[glogT], gb])


def fit_biastemp(Z, y, C, Ws):
    """H3: fix the source direction Ws; optimise a positive temperature + per-class bias. Returns (W=T*Ws, b)."""
    WsZ = Z @ Ws.T
    res = minimize(_biastemp_obj, np.zeros(C + 1), args=(WsZ, y, C), jac=True, method="L-BFGS-B", options=dict(maxiter=300))
    T = float(np.exp(res.x[0])); return T * Ws, res.x[1:]


def session_macro_bacc(W, b, Zq, yq, sq):
    """Mean over query sessions (>=4 trials, >=2 classes) of bAcc; pooled fallback."""
    pred = (Zq @ W.T + b).argmax(1)
    per = [balanced_accuracy_score(yq[sq == s], pred[sq == s]) for s in np.unique(sq)
           if (sq == s).sum() >= 4 and len(np.unique(yq[sq == s])) >= 2]
    return float(np.mean(per)) if per else float(balanced_accuracy_score(yq, pred))


def _early_later_masks(sess):
    u = sorted(set(map(str, sess))); s = np.asarray(list(map(str, sess)))
    return s == u[0], np.isin(s, u[1:])


def select_alpha_pseudo_target(Zs_std, ys, ds, sess_s, C, grid=ALPHA_GRID):
    """SOURCE-ONLY alpha selection: for each source subject d, anchor = head fit on source\\{d}; adapt on d's EARLY
    session; score CE on d's LATER session. Pick alpha minimising mean pseudo-query CE over source subjects. No target
    data. Falls back to alpha=1.0 if too few usable pseudo-subjects."""
    early, later = _early_later_masks(sess_s); ds = np.asarray(ds)
    subs = [d for d in np.unique(ds)
            if ((ds == d) & early).sum() >= C and ((ds == d) & later).sum() >= C
            and len(np.unique(ys[(ds == d) & later])) >= 2 and len(np.unique(ys[(ds != d)])) == C]
    if len(subs) < 2:
        return 1.0, dict(reason="INSUFFICIENT_PSEUDO_SUBJECTS", n=len(subs))
    ce_by_alpha = {a: [] for a in grid}
    for d in subs:
        oth = ds != d; Ws, bs = fit_head(Zs_std[oth], ys[oth], C)         # anchor excludes d
        cal = (ds == d) & early; qy = (ds == d) & later
        for a in grid:
            W, b = fit_head(Zs_std[cal], ys[cal], C, Ws, bs, a)
            logits = Zs_std[qy] @ W.T + b; logits -= logits.max(1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
            ce_by_alpha[a].append(-logp[np.arange(qy.sum()), ys[qy]].mean())
    mean_ce = {a: float(np.mean(v)) for a, v in ce_by_alpha.items()}
    astar = min(mean_ce, key=mean_ce.get)
    return astar, dict(mean_ce=mean_ce, n_pseudo=len(subs))


def prepare_source_head(Zs_wd, ys, C):
    """Fit the frozen source head ONCE for a representation (reused across all k-budgets and draws). Returns
    (mu, sd, Ws, bs) in the source-standardised deleted space + the in-sample source bAcc (for the ZR retention filter)."""
    mu, sd = standardize(Zs_wd); Xs = _std(Zs_wd, mu, sd)
    Ws, bs = fit_head(Xs, ys, C)
    src_bacc = float(balanced_accuracy_score(ys, (Xs @ Ws.T + bs).argmax(1)))
    return mu, sd, Ws, bs, src_bacc


def adapt_and_score(prep, Xcal_wd, ycal, draw_idx, Zq_wd, yq, sq, C, alpha, heads=("frozen", "fresh", "map", "bias")):
    """Given a prepared source head, adapt on ONE cal draw and score the requested heads on the query session. The
    source head (frozen) is reused; only the small cal-draw heads are fit here."""
    mu, sd, Ws, bs = prep[0], prep[1], prep[2], prep[3]
    Xq = _std(Zq_wd, mu, sd); Xcalk = _std(Xcal_wd[draw_idx], mu, sd); yck = ycal[draw_idx]
    out = {}
    if "frozen" in heads:
        out["frozen"] = session_macro_bacc(Ws, bs, Xq, yq, sq)
    ok = len(np.unique(yck)) >= 2
    if "fresh" in heads:
        W, b = fit_head(Xcalk, yck, C) if ok else (Ws, bs); out["fresh"] = session_macro_bacc(W, b, Xq, yq, sq)
    if "map" in heads:
        W, b = fit_head(Xcalk, yck, C, Ws, bs, alpha) if ok else (Ws, bs); out["map"] = session_macro_bacc(W, b, Xq, yq, sq)
    if "bias" in heads:
        W, b = fit_biastemp(Xcalk, yck, C, Ws) if ok else (Ws, bs); out["bias"] = session_macro_bacc(W, b, Xq, yq, sq)
    return out


def readout_utilities(Zs_wd, ys, ds, sess_s, Xcal_wd, ycal, Zq_wd, yq, sq, C, draw_idx, alpha=None):
    """For ONE representation (already whitened+deleted) and ONE k-shot cal draw, return the 4 head utilities on the
    query session. alpha (H2) is passed in (selected once per representation, source-only). Standardisation is fit on
    the SOURCE features so query is scored under the source-fit transform (deployable)."""
    mu, sd = standardize(Zs_wd)
    Xs, Xcalk, Xq = _std(Zs_wd, mu, sd), _std(Xcal_wd[draw_idx], mu, sd), _std(Zq_wd, mu, sd)
    yck = ycal[draw_idx]
    Ws, bs = fit_head(Xs, ys, C)                                          # H0 source head
    u_frozen = session_macro_bacc(Ws, bs, Xq, yq, sq)
    if len(np.unique(yck)) < 2:                                           # a degenerate draw -> fall back to frozen
        return dict(frozen=u_frozen, fresh=u_frozen, map=u_frozen, bias=u_frozen, alpha=(alpha or 0.0), degenerate=True)
    if alpha is None:
        alpha = select_alpha_pseudo_target(_std(Zs_wd, mu, sd), ys, ds, sess_s, C)[0]
    Wf, bf = fit_head(Xcalk, yck, C)                                      # H1 fresh
    Wm, bm = fit_head(Xcalk, yck, C, Ws, bs, alpha)                       # H2 MAP
    Wt, bt = fit_biastemp(Xcalk, yck, C, Ws)                              # H3 bias+temp
    return dict(frozen=u_frozen, fresh=session_macro_bacc(Wf, bf, Xq, yq, sq),
                map=session_macro_bacc(Wm, bm, Xq, yq, sq), bias=session_macro_bacc(Wt, bt, Xq, yq, sq),
                alpha=float(alpha), degenerate=False)
