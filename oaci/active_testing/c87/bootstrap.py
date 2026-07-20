"""Patient-cluster BCa bootstrap (CRN-paired across policies) + vectorized held-selection cross-fit.

Cluster = patient. Each replicate resamples patient COLUMNS with replacement; ALL policies are evaluated on
the SAME resample (common random numbers) so the gain G is a paired difference (C87P SPEC 4.F / v3). The
cross-fit reference is recomputed per replicate (folds carried by patient). BCa intervals per SPEC 4.D.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .estimand import XFIT_FOLDS


def crossfit_vec(Lbar: np.ndarray, fold_of: np.ndarray, F: int = XFIT_FOLDS):
    """Vectorized held-selection cross-fit on columns already selected. Returns (Lfold(A,F), sel(F,), ref).

    a_hat_{-f} = argmin_a mean-loss over patients NOT in fold f; ref = mean_f Lfold[a_hat_{-f}, f]."""
    onehot = (fold_of[:, None] == np.arange(F)[None, :]).astype(float)   # (n,F)
    foldsum = Lbar @ onehot                                              # (A,F)
    count = onehot.sum(0)                                                # (F,)
    total = Lbar.sum(1)                                                  # (A,)
    n_all = Lbar.shape[1]
    Lfold = foldsum / np.where(count > 0, count, 1.0)
    sel = np.full(F, -1, int)
    ref_terms = []
    for f in range(F):
        if count[f] == 0 or (n_all - count[f]) == 0:
            continue
        mean_other = (total - foldsum[:, f]) / (n_all - count[f])
        a = int(np.argmin(mean_other))
        sel[f] = a
        ref_terms.append(Lfold[a, f])
    ref = float(np.mean(ref_terms)) if ref_terms else float("nan")
    return Lfold, sel, ref


def excess_from_lfold(Lfold: np.ndarray, sel: np.ndarray, a_pick: int) -> float:
    """R^CF(a_pick) = mean_f [ Lfold[a_pick,f] - Lfold[sel_f,f] ] over evaluable folds."""
    F = sel.size
    terms = [Lfold[a_pick, f] - Lfold[sel[f], f] for f in range(F) if sel[f] >= 0]
    return float(np.mean(terms))


def mean_excess(Lfold, sel, picks) -> float:
    """Mean over K policy-seed picks of R^CF(pick)."""
    return float(np.mean([excess_from_lfold(Lfold, sel, int(a)) for a in picks]))


def _stat_on_columns(Lbar, fold_of, cols, picks_by_policy):
    """Compute {policy: mean R^CF} on a given set of (possibly resampled) patient columns."""
    Lb = Lbar[:, cols]
    fo = fold_of[cols]
    Lfold, sel, _ = crossfit_vec(Lb, fo)
    return {name: mean_excess(Lfold, sel, picks) for name, picks in picks_by_policy.items()}


def paired_gain_bootstrap(Lbar, fold_of, picks_pi, picks_p0, aC, B_boot, rng):
    """CRN-paired cluster bootstrap. Returns dict with observed + bootstrap arrays for R^CF(pi), R^CF(P0),
    G = R^CF(P0)-R^CF(pi), and T^CF = R^CF(aC). All on the SAME per-replicate patient resample."""
    n_pat = Lbar.shape[1]
    picks_by = {"pi": picks_pi, "P0": picks_p0, "aC": np.array([aC])}
    obs = _stat_on_columns(Lbar, fold_of, np.arange(n_pat), picks_by)
    obs_G = obs["P0"] - obs["pi"]
    boot_G = np.empty(B_boot)
    boot_pi = np.empty(B_boot)
    boot_T = np.empty(B_boot)
    for b in range(B_boot):
        idx = rng.integers(0, n_pat, n_pat)             # patient resample w/ replacement (CRN)
        s = _stat_on_columns(Lbar, fold_of, idx, picks_by)
        boot_G[b] = s["P0"] - s["pi"]
        boot_pi[b] = s["pi"]
        boot_T[b] = s["aC"]
    return dict(obs_G=obs_G, obs_pi=obs["pi"], obs_P0=obs["P0"], obs_T=obs["aC"],
                boot_G=boot_G, boot_pi=boot_pi, boot_T=boot_T)


def _jackknife_G(Lbar, fold_of, picks_pi, picks_p0):
    """Leave-one-patient-out jackknife of G for BCa acceleration."""
    n = Lbar.shape[1]
    picks_by = {"pi": picks_pi, "P0": picks_p0}
    vals = np.empty(n)
    allc = np.arange(n)
    for j in range(n):
        cols = np.delete(allc, j)
        s = _stat_on_columns(Lbar, fold_of, cols, picks_by)
        vals[j] = s["P0"] - s["pi"]
    return vals


def bca_lcb(obs, boot, jack, alpha=0.05):
    """One-sided BCa lower confidence bound at level 1-alpha for a scalar statistic."""
    boot = boot[np.isfinite(boot)]
    if boot.size < 10:
        return float("nan")
    z0 = norm.ppf(np.clip((boot < obs).mean(), 1e-6, 1 - 1e-6))
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5 + 1e-12)
    a = num / den
    zl = norm.ppf(alpha)
    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo = adj(zl)
    return float(np.quantile(boot, np.clip(lo, 1e-6, 1 - 1e-6)))


def percentile_lcb(boot, alpha=0.05):
    boot = boot[np.isfinite(boot)]
    return float(np.quantile(boot, alpha)) if boot.size else float("nan")


def bc_ci(obs, boot, alpha=0.05):
    """Bias-corrected (BC) two-sided bootstrap CI — corrects the plug-in bias of a non-smooth statistic
    (e.g. the cross-fit argmin) without the O(n) jackknife of full BCa acceleration. Returns (lo, hi)."""
    boot = boot[np.isfinite(boot)]
    if boot.size < 10:
        return float("nan"), float("nan")
    z0 = norm.ppf(np.clip((boot < obs).mean(), 1e-6, 1 - 1e-6))
    def adj(p):
        return float(np.quantile(boot, np.clip(norm.cdf(2 * z0 + norm.ppf(p)), 1e-6, 1 - 1e-6)))
    return adj(alpha / 2), adj(1 - alpha / 2)
