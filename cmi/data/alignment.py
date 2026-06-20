"""Euclidean Alignment (He & Wu, IEEE TBME 2020, arXiv:1808.05464) — label-free per-subject covariance
whitening for calibration-free cross-subject transfer. Per domain d: R̄_d = mean over that domain's trials
of (X Xᵀ / T); then X̃ = R̄_d^{-1/2} X. Reduces the marginal covariance shift across subjects with NO labels
and NO separate calibration session (it uses each subject's own trials).

NOTE (leakage interaction): EA aligns subject covariances -> it REDUCES the very subject-leakage we measure,
so it belongs to the ACCURACY pipeline only. The raw leakage probe must NOT be EA'd. Also a band-pass helper
(the dual-band: 8-30 Hz for covariance / 4-40 Hz for encoders)."""
from __future__ import annotations
import numpy as np


def _inv_sqrt(R):
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-10, None)
    return (V * (w ** -0.5)) @ V.T


def euclidean_align(X, d):
    """X [N,C,T] float, d [N] domain ids. Returns EA-aligned X̃ (same shape), per-domain."""
    X = X.astype("float64")
    Xa = np.empty_like(X)
    for dom in np.unique(d):
        idx = np.where(d == dom)[0]
        covs = np.einsum("nct,nkt->nck", X[idx], X[idx]) / X.shape[2]   # [n,C,C] per-trial cov
        P = _inv_sqrt(covs.mean(0))                                     # R̄_d^{-1/2}
        Xa[idx] = np.einsum("ck,nkt->nct", P, X[idx])
    return Xa.astype("float32")


def riemannian_align(X, d):
    """RA (Riemannian Alignment, Zanini et al. 2018): like EA but recenter with the AIRM GEOMETRIC
    (Fréchet) mean instead of the arithmetic mean -> respects the SPD manifold's negative curvature.
    Still signal-level (X̃ = Ḡ^{-1/2} X), so any downstream model can use it. Ḡ = geometric mean of
    the domain's trial covariances under the affine-invariant metric."""
    from pyriemann.utils.mean import mean_covariance
    X = X.astype("float64"); C = X.shape[1]
    Xa = np.empty_like(X)
    for dom in np.unique(d):
        idx = np.where(d == dom)[0]
        covs = np.einsum("nct,nkt->nck", X[idx], X[idx]) / X.shape[2] + 1e-6 * np.eye(C)
        P = _inv_sqrt(mean_covariance(covs, metric="riemann"))                # Ḡ^{-1/2}, AIRM Fréchet mean
        Xa[idx] = np.einsum("ck,nkt->nct", P, X[idx])
    return Xa.astype("float32")


def euclidean_align_strict(Xtr, dtr, Xte):
    """STRICT-DG EA: source subjects aligned per-subject (legal — source data); the TARGET is whitened
    by the SOURCE-POOL reference R̄_src only — it NEVER uses the target subject's own trials. Quantifies
    how much of standard (transductive) EA's gain comes from using unlabeled target statistics."""
    Xtr_a = euclidean_align(Xtr, dtr)                              # source: per-subject
    src = Xtr.astype("float64")
    Rbar = (np.einsum("nct,nkt->nck", src, src) / src.shape[2]).mean(0)   # raw source-pool reference
    P = _inv_sqrt(Rbar)
    Xte_a = np.einsum("ck,nkt->nct", P, Xte.astype("float64")).astype("float32")
    return Xtr_a, Xte_a


def hyperbolic_align(F, d, scale=0.1, c=1.0):
    """HA (EXPLORATORY): recenter FEATURES F [N,D] (e.g. LogCov tangent vectors) in the Poincaré ball.
    Per domain: embed via expmap0, take the tangent-space (Fréchet) centroid, and Möbius-translate it to
    the origin. Rationale: SPD-AIRM is a Hadamard (hyperbolic-LIKE) space (SPD(2)≅ℝ×ℍ²); HA asks whether an
    EXPLICIT constant-curvature embedding helps the cross-subject recentering. See notes/hyperbolic_alignment.md.
    NOTE: feature-level (not signal-level), so it pairs with the LogCov/covariance arm. Returns Euclidean
    tangent coords (logmap0 of the recentered points) so downstream code stays Euclidean."""
    import torch, geoopt
    ball = geoopt.PoincareBall(c=c)
    Ft = torch.tensor(F, dtype=torch.float64)
    Ft = (Ft - Ft.mean(0)) / (Ft.std(0) + 1e-7)
    P = ball.expmap0(scale * Ft)                                              # embed into the ball
    out = torch.empty_like(P)
    for dom in np.unique(d):
        idx = torch.tensor(np.where(d == dom)[0])
        mu = ball.expmap0(ball.logmap0(P[idx]).mean(0, keepdim=True))         # Fréchet-mean approx
        out[idx] = ball.mobius_add(-mu, P[idx])                               # translate centroid -> origin
    return ball.logmap0(out).numpy().astype("float32")                       # back to Euclidean tangent


def bandpass(X, fmin, fmax, sfreq, order=4):
    """Zero-phase Butterworth band-pass on [N,C,T]. fmin/fmax in Hz; None on either side = one-sided."""
    from scipy.signal import butter, sosfiltfilt
    nyq = sfreq / 2.0
    if fmin and fmax:
        sos = butter(order, [fmin / nyq, fmax / nyq], btype="bandpass", output="sos")
    elif fmin:
        sos = butter(order, fmin / nyq, btype="highpass", output="sos")
    elif fmax:
        sos = butter(order, fmax / nyq, btype="lowpass", output="sos")
    else:
        return X
    return sosfiltfilt(sos, X.astype("float64"), axis=-1).astype("float32")
