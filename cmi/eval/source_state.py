"""Source-state serialization for STRICT source-free deployment (reviewer R8 §6.1).

At training time we fit everything the target adaptation needs and freeze it into a `SourceState`:
the frozen linear readout `h_theta`, the class-conditional moments {mu_y^S, Sigma_y^S}, the pooled
source moments (mu_S, Sigma_S) used by the CORAL recenter, the source prior pi_S, and the shrinkage
parameters. A SHA-256 hash pins the artifact.

At deployment, `pmct_predict_serialized(state, z_tgt, ...)` consumes ONLY the state and the unlabeled
target features. It NEVER receives source examples or source labels — enforced by the signature (no
z_src/y_src argument) and asserted in `cmi/eval/test_label_shift.run_serialization`. This is what licenses
the "no source examples / no source labels / no gradient update at deployment" claim. The equivalence test
checks max|p_serialized - p_online| < 1e-6 against `pmct_transport`.
"""
import hashlib
import numpy as np
from cmi.eval.label_shift import _sqrtm, _shrink_cov, _simplex_clip


def _coral_recenter_from_stats(mu_S, Sig_S_raw, z_tgt, eps=1e-5):
    """feature_coral_recenter using PRECOMPUTED source pool stats (mu_S, Sig_S_raw) — no source examples."""
    z_tgt = np.asarray(z_tgt, float)
    d = z_tgt.shape[1]
    mu_T = z_tgt.mean(0)
    Cs = Sig_S_raw + eps * np.eye(d)
    Ct = np.cov(z_tgt, rowvar=False) + eps * np.eye(d)
    W = _sqrtm(Cs, eps) @ _sqrtm(Ct, eps, inv=True)
    return (z_tgt - mu_T) @ W.T + mu_S


def fit_source_state(z_src, y_src, n_cls, rho=0.2, eps=1e-3, clf=None):
    """Freeze everything the target adaptation needs into a serializable SourceState dict. Source examples are
    consumed HERE (training time) and never stored — only their sufficient statistics + the fitted readout."""
    from sklearn.linear_model import LogisticRegression
    z_src = np.asarray(z_src, float)
    d = z_src.shape[1]
    if clf is None:
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(z_src, y_src)
    mu_y = np.stack([z_src[y_src == c].mean(0) if (y_src == c).any() else z_src.mean(0) for c in range(n_cls)])
    Sig_y0 = [np.cov(z_src[y_src == c], rowvar=False) if (y_src == c).sum() > d else np.eye(d)
              for c in range(n_cls)]
    pi_S = np.bincount(y_src, minlength=n_cls).astype(float); pi_S /= pi_S.sum()
    state = dict(clf=clf, mu_y=mu_y, Sig_y0=[np.asarray(s, float) for s in Sig_y0],
                 mu_pool=z_src.mean(0), Sig_pool0=np.cov(z_src, rowvar=False),
                 pi_S=pi_S, n_cls=int(n_cls), d=int(d), rho=float(rho), eps=float(eps))
    state["hash"] = source_state_hash(state)
    return state


def source_state_hash(state):
    """SHA-256 over all numeric artifacts + the readout weights — pins the deployment state."""
    h = hashlib.sha256()
    for k in ("mu_y", "Sig_y0", "mu_pool", "Sig_pool0", "pi_S"):
        arr = np.ascontiguousarray(np.concatenate([np.ravel(a) for a in np.atleast_1d(state[k])])
                                   if k == "Sig_y0" else np.ravel(state[k]), dtype=np.float64)
        h.update(arr.tobytes())
    clf = state["clf"]
    for attr in ("coef_", "intercept_"):
        if hasattr(clf, attr):
            h.update(np.ascontiguousarray(getattr(clf, attr), dtype=np.float64).tobytes())
    h.update(np.array([state["n_cls"], state["d"], state["rho"], state["eps"]], dtype=np.float64).tobytes())
    return h.hexdigest()


def pmct_predict_serialized(state, z_tgt, alpha=1.0, gate="reliability", kappa=8.0,
                            ref="prior_matched", tmap="wc", em_iters=3):
    """STRICT source-free PMCT/CORAL transport+predict. Mirrors `label_shift.pmct_transport` EXACTLY but reads
    only `state` + the unlabeled target features. NO z_src / y_src argument — source data is unreachable here."""
    z_tgt = np.asarray(z_tgt, float)
    clf, n_cls, d = state["clf"], state["n_cls"], state["d"]
    rho, eps = state["rho"], state["eps"]
    mu_y, Sig_y0 = state["mu_y"], state["Sig_y0"]
    mu_pool, Sig_pool0 = state["mu_pool"], state["Sig_pool0"]
    cls = clf.classes_

    def _full(p):
        out = np.zeros((len(p), n_cls)); out[:, cls] = p; return out
    mu_T = z_tgt.mean(0); n_T = len(z_tgt)
    bar_Sig_T = _shrink_cov(np.cov(z_tgt, rowvar=False), rho, eps)
    Wt_inv = _sqrtm(bar_Sig_T, eps, inv=True)
    St_half = _sqrtm(bar_Sig_T, eps) if tmap == "ot" else None
    P0 = _full(clf.predict_proba(_coral_recenter_from_stats(mu_pool, Sig_pool0, z_tgt)))
    pi = _simplex_clip(P0.mean(0))
    if gate == "off":
        g = 1.0
    else:
        g_cov = float(np.clip(n_T / (2.0 * d), 0.0, 1.0))
        se = float(np.trace(np.cov(P0, rowvar=False)) / max(n_T, 1))
        g_unc = float(np.exp(-kappa * se * n_T)) if gate == "entropy" else float(np.exp(-kappa * se))
        g = g_cov * g_unc
    prob = None
    for _ in range(max(1, em_iters)):
        if ref == "pooled":
            mu_R, Sig_R0 = mu_pool, Sig_pool0
        else:
            mu_R = (pi[:, None] * mu_y).sum(0)
            Sig_R0 = sum(pi[c] * (Sig_y0[c] + np.outer(mu_y[c] - mu_R, mu_y[c] - mu_R)) for c in range(n_cls))
        bar_Sig_R = _shrink_cov(Sig_R0, rho, eps)
        if tmap == "ot":
            M = Wt_inv @ _sqrtm(St_half @ bar_Sig_R @ St_half, eps) @ Wt_inv
        else:
            M = _sqrtm(bar_Sig_R, eps) @ Wt_inv
        Tz = mu_R + (z_tgt - mu_T) @ M.T
        alpha_eff = alpha * g
        z_tilde = (1 - alpha_eff) * z_tgt + alpha_eff * Tz
        p = clf.predict_proba(z_tilde)
        prob = np.zeros((len(p), n_cls)); prob[:, cls] = p
        pi_new = _simplex_clip(prob.mean(0))
        if np.abs(pi_new - pi).max() < 1e-3:
            pi = pi_new; break
        pi = pi_new
    return prob, pi
