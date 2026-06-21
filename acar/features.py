"""Label-free paired pre->post observables phi_a(B). NONE of these read y_target — by construction, permuting the
labels leaves every value bit-identical (asserted by the metamorphic guard in run_gonogo / tests).

Paired features (the ACAR contribution): functions of (p_0 vs p_a) and (z vs z_tilde_a) — i.e. WHAT THE ACTION DID.
Context features: the A0 source-free scores, carried as raw regressor coordinates only (no asserted direction).
"""
from __future__ import annotations
import numpy as np

from .config import PAIRED_FEATURES, CONTEXT_FEATURES


# ---------- small numerics ----------
def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(1)


def _margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def _jsd(p, q):
    p = np.clip(p, 1e-12, 1.0); q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(a) - np.log(b))).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _shrink(S, rho=0.1, eps=1e-3):
    d = S.shape[0]
    return (1 - rho) * S + rho * np.trace(S) / d * np.eye(d) + eps * np.eye(d)


def _sqrtm_psd(S):
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def _bures_w2(x, y, rho=0.1):
    """Closed-form squared Bures-Wasserstein between Gaussian fits to feature batches x and y. Label-free."""
    if x is None or y is None or len(x) < 2 or len(y) < 2:
        return np.nan
    mu = ((x.mean(0) - y.mean(0)) ** 2).sum()
    Cx, Cy = _shrink(np.cov(x, rowvar=False), rho), _shrink(np.cov(y, rowvar=False), rho)
    Cxh = _sqrtm_psd(Cx)
    cross = _sqrtm_psd(Cxh @ Cy @ Cxh)
    tr = np.trace(Cx) + np.trace(Cy) - 2.0 * np.trace(cross)
    return float(max(mu + tr, 0.0))


def _fisher_ratio(z, pseudo):
    """tr(S_b)/tr(S_w) under hard pseudo-labels — label-free separability the action induced. Higher = cleaner."""
    if z is None or len(z) < 4:
        return np.nan
    classes = np.unique(pseudo)
    if len(classes) < 2:
        return 0.0
    mu = z.mean(0)
    sb = sw = 0.0
    for c in classes:
        zc = z[pseudo == c]
        if len(zc) < 1:
            continue
        sb += len(zc) * ((zc.mean(0) - mu) ** 2).sum()
        sw += ((zc - zc.mean(0)) ** 2).sum()
    return float(sb / (sw + 1e-9))


def _ess(p_a):
    """Effective sample size under soft responsibilities, min over classes. Label-free."""
    r = np.clip(p_a, 1e-12, 1.0)
    w = r.sum(0)
    return float((w ** 2 / (r ** 2).sum(0)).min())


# ---------- paired features ----------
def paired_features(p0, pa, z_pre, z_post) -> dict:
    """phi_a(B) — all label-free. z_post may be None (prob-only actions) -> bures/post_sep = nan."""
    return dict(
        d_entropy=float(_entropy(pa).mean() - _entropy(p0).mean()),
        d_margin=float(_margin(pa).mean() - _margin(p0).mean()),
        flip_rate=float((pa.argmax(1) != p0.argmax(1)).mean()),
        js=float(_jsd(p0, pa).mean()),
        bures=_bures_w2(z_post, z_pre),
        post_sep=_fisher_ratio(z_post, pa.argmax(1)),
        n_eff=_ess(pa),
    )


# ---------- A0 background scores (context coordinates; NO asserted direction) ----------
def _maha2(z, mu, Winv):
    dz = z - mu
    return np.einsum("ij,jk,ik->i", dz, Winv, dz)


def context_features(state, z_post, p_a) -> dict:
    """The A0 source-free scores, aggregated WHOLE-batch. Demoted to background regressor inputs only."""
    z = z_post if z_post is not None else state["mu_pool"][None] * 0  # prob-only -> geometry context unavailable
    if z_post is None:
        return dict(g_unc=float(_entropy(p_a).mean()), s_support=np.nan, s_sep=np.nan, pr_cmi_proxy=np.nan)
    Winv = np.linalg.inv(_shrink(np.asarray(state["Sig_pool0"], float)))
    g_unc = float(_entropy(p_a).mean())
    s_support = float(_maha2(z, state["mu_pool"], Winv).mean())
    m = np.stack([_maha2(z, state["mu_y"][c], Winv) for c in range(state["n_cls"])], 1)
    s_sep = float((-np.abs(m[:, 0] - m[:, 1])).mean())
    readout = state["clf"].predict(z); proto = m.argmin(1)
    margin = np.sort(m, 1)[:, 1] - np.sort(m, 1)[:, 0]
    pr_cmi_proxy = float(((proto != readout).astype(float) * margin + 0.01 * margin).mean())
    return dict(g_unc=g_unc, s_support=s_support, s_sep=s_sep, pr_cmi_proxy=pr_cmi_proxy)


def feature_vector(phi: dict, ctx: dict) -> np.ndarray:
    """Ordered [paired || context] vector for the regressor. NaNs (prob-only actions) -> 0.0 (mean-imputed)."""
    vals = [phi[k] for k in PAIRED_FEATURES] + [ctx[k] for k in CONTEXT_FEATURES]
    return np.array([0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in vals], float)
