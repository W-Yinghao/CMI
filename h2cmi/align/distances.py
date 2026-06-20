"""Differentiable distribution-distance functions for reference-prior alignment.

These are the marginal-alignment primitives used by H2-CMI to pull each domain's
latent marginal ``q_d(Z)`` towards a shared reference prior (and towards each other)
*without* erasing class-conditional structure.  Every function here is

  * **pure** -- no global state, no side effects, no module-level RNG seeding;
  * **differentiable** -- gradients flow to the input tensors so the distance can be
    used directly as a training loss term;
  * **numerically stable** -- eigenvalue clamping, ``N >= 2`` guards, ``+eps`` under
    every square root.

All operate on float tensors of shape ``[N, d]`` (a batch of ``d``-dimensional
latents).  Only :mod:`torch` and :mod:`numpy` are used.

Public API
----------
``sliced_wasserstein(x, y, n_projections=64, p=2)``
    Sliced :math:`p`-Wasserstein distance between empirical samples.
``energy_distance(x, y)``
    Squared energy distance via pairwise Euclidean distances.
``gauss_w2(x, y, shrinkage=0.05)``
    Closed-form 2-Wasserstein-squared between Gaussian fits (Bures metric).
``ledoit_wolf_cov(x, shrinkage=None)``
    Shrinkage (well-conditioned) covariance estimator.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "sliced_wasserstein",
    "energy_distance",
    "gauss_w2",
    "ledoit_wolf_cov",
]

# A floor used everywhere we take a square root of a (clamped) nonnegative value.
_EPS = 1e-12
# Eigenvalue clamp for symmetric matrix square roots (well above float32 noise).
_EIG_FLOOR = 1e-6


# --------------------------------------------------------------------------- #
#  1. Sliced Wasserstein                                                       #
# --------------------------------------------------------------------------- #
def sliced_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 64,
    p: int = 2,
) -> torch.Tensor:
    """Sliced :math:`p`-Wasserstein distance between empirical samples.

    Draws ``n_projections`` random unit directions on the unit sphere of
    :math:`\\mathbb{R}^d`, projects both point clouds onto each, and averages the
    per-projection 1-D :math:`p`-Wasserstein distance.  In 1-D the Wasserstein
    distance has the closed form

    .. math::
        W_p(\\mu, \\nu)^p = \\int_0^1 |F_\\mu^{-1}(t) - F_\\nu^{-1}(t)|^p \\, dt ,

    which for equal sample sizes reduces to comparing the *sorted* samples.  When
    ``Nx != Ny`` the per-projection sorted samples (empirical quantiles) are linearly
    interpolated onto a common grid of ``max(Nx, Ny)`` quantile levels before the
    comparison.

    Parameters
    ----------
    x, y
        Float tensors ``[Nx, d]`` and ``[Ny, d]``.  Must share the last dimension.
    n_projections
        Number of random 1-D projections to average over.
    p
        Order of the Wasserstein distance (``p >= 1``).

    Returns
    -------
    torch.Tensor
        A scalar tensor (mean over projections), differentiable w.r.t. ``x`` and ``y``.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"expected [N, d] tensors, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"feature dims must match: {x.shape[1]} vs {y.shape[1]}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")

    nx, d = x.shape
    ny = y.shape[0]
    if nx < 1 or ny < 1:
        raise ValueError("both inputs need at least one sample")

    # Random unit directions on x's device/dtype (no global RNG mutation).
    directions = torch.randn(n_projections, d, device=x.device, dtype=x.dtype)
    directions = directions / (directions.norm(dim=1, keepdim=True) + _EPS)

    # Project: [n_proj, N] = [N, d] @ [d, n_proj] -> transpose.
    proj_x = (x @ directions.t()).t()  # [n_proj, Nx]
    proj_y = (y @ directions.t()).t()  # [n_proj, Ny]

    # Sort along the sample axis -> empirical quantile functions.
    sx, _ = torch.sort(proj_x, dim=1)  # [n_proj, Nx]
    sy, _ = torch.sort(proj_y, dim=1)  # [n_proj, Ny]

    if nx == ny:
        qx, qy = sx, sy
    else:
        # Interpolate both quantile functions onto a common grid of size max(Nx, Ny).
        m = max(nx, ny)
        grid = torch.linspace(0.0, 1.0, m, device=x.device, dtype=x.dtype)
        qx = _interp_quantiles(sx, grid)  # [n_proj, m]
        qy = _interp_quantiles(sy, grid)  # [n_proj, m]

    diff = (qx - qy).abs()
    # Mean over quantile levels approximates the integral, then take the p-th root.
    per_proj = diff.pow(p).mean(dim=1).clamp_min(_EPS).pow(1.0 / p)  # [n_proj]
    return per_proj.mean()


def _interp_quantiles(sorted_vals: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate sorted samples onto quantile levels ``grid`` in ``[0, 1]``.

    ``sorted_vals`` is ``[n_proj, n]`` (already sorted ascending along dim 1); the
    associated quantile levels are an equispaced grid in ``[0, 1]``.  Returns
    ``[n_proj, len(grid)]``.  Differentiable in ``sorted_vals`` (the interpolation
    weights depend only on the fixed grids).
    """
    n_proj, n = sorted_vals.shape
    if n == 1:
        return sorted_vals.expand(n_proj, grid.shape[0])

    src_levels = torch.linspace(0.0, 1.0, n, device=sorted_vals.device, dtype=sorted_vals.dtype)
    # Position of each query level within the source grid.
    pos = grid * (n - 1)                                   # [m] in [0, n-1]
    lo = pos.floor().clamp(0, n - 2).long()               # [m]
    hi = lo + 1                                            # [m]
    w = (pos - lo.to(grid.dtype)).clamp(0.0, 1.0)         # [m] fractional weight

    lo_vals = sorted_vals.index_select(1, lo)             # [n_proj, m]
    hi_vals = sorted_vals.index_select(1, hi)             # [n_proj, m]
    del src_levels  # only documents that the source levels are equispaced
    return lo_vals + (hi_vals - lo_vals) * w.unsqueeze(0)


# --------------------------------------------------------------------------- #
#  2. Energy distance                                                         #
# --------------------------------------------------------------------------- #
def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared energy distance between two empirical distributions.

    .. math::
        \\mathcal{E}(x, y) = 2\\,\\mathbb{E}\\|X - Y\\| - \\mathbb{E}\\|X - X'\\|
                              - \\mathbb{E}\\|Y - Y'\\| ,

    using pairwise Euclidean distances, where :math:`X, X'` are i.i.d. from the
    empirical law of ``x`` and likewise for ``y``.  This is nonnegative and zero iff
    the two empirical distributions coincide.

    Parameters
    ----------
    x, y
        Float tensors ``[Nx, d]``, ``[Ny, d]`` (same ``d``).

    Returns
    -------
    torch.Tensor
        A scalar tensor, differentiable w.r.t. ``x`` and ``y``.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"expected [N, d] tensors, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"feature dims must match: {x.shape[1]} vs {y.shape[1]}")
    if x.shape[0] < 1 or y.shape[0] < 1:
        raise ValueError("both inputs need at least one sample")

    d_xy = _pairwise_euclidean(x, y).mean()
    d_xx = _pairwise_euclidean(x, x).mean()
    d_yy = _pairwise_euclidean(y, y).mean()
    val = 2.0 * d_xy - d_xx - d_yy
    # Energy distance is nonnegative in the population; clamp tiny negative noise.
    return val.clamp_min(0.0)


def _pairwise_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stable pairwise Euclidean distance matrix ``[Na, Nb]``.

    Uses ``torch.cdist`` which is differentiable; a ``+eps`` is folded in by clamping
    the squared term away from zero implicitly (cdist handles the sqrt, we only guard
    the gradient at exact coincidence by adding a tiny epsilon to the inputs' diff is
    unnecessary because cdist already uses a numerically safe path for ``p=2``).
    """
    # cdist is O(Na*Nb*d); fine for N up to a few hundred.  Correct, not an approx.
    return torch.cdist(a, b, p=2.0)


# --------------------------------------------------------------------------- #
#  3 & 4. Gaussian W2 (Bures) and Ledoit-Wolf covariance                       #
# --------------------------------------------------------------------------- #
def ledoit_wolf_cov(x: torch.Tensor, shrinkage: float | None = None) -> torch.Tensor:
    """Shrinkage (Ledoit-Wolf) covariance estimate of ``x`` ``[N, d]``.

    Returns :math:`(1 - a) S + a\\,\\mu I` where :math:`S` is the empirical covariance,
    :math:`\\mu = \\operatorname{tr}(S) / d` is the average eigenvalue (the shrinkage
    target is :math:`\\mu I`), and :math:`a \\in [0, 1]` is the shrinkage intensity.

    If ``shrinkage`` is given it is used directly (and the whole computation is then
    differentiable in ``x``).  Otherwise the Ledoit-Wolf optimal shrinkage intensity
    is estimated from the data and clamped to ``[0, 1]``; in that data-driven case the
    intensity is treated as a constant (detached) so the returned matrix is still a
    valid differentiable function of ``x`` through ``S`` and ``mu``.

    Parameters
    ----------
    x
        Float tensor ``[N, d]`` with ``N >= 2``.
    shrinkage
        Fixed intensity in ``[0, 1]`` or ``None`` for the Ledoit-Wolf estimate.

    Returns
    -------
    torch.Tensor
        Symmetric PSD ``[d, d]`` covariance, differentiable when ``shrinkage`` is a
        fixed float.
    """
    if x.dim() != 2:
        raise ValueError(f"expected [N, d] tensor, got {tuple(x.shape)}")
    n, d = x.shape
    if n < 2:
        raise ValueError(f"need N >= 2 for a covariance, got N={n}")

    mu_vec = x.mean(dim=0, keepdim=True)            # [1, d]
    xc = x - mu_vec                                 # centred [N, d]
    # Empirical covariance with the unbiased 1/(N-1) factor (N>=2 guaranteed).
    cov = (xc.t() @ xc) / (n - 1)                   # [d, d]
    cov = 0.5 * (cov + cov.t())                     # symmetrise against round-off

    trace = torch.diagonal(cov).sum()
    mu = trace / d                                  # average eigenvalue (scalar)
    target = mu * torch.eye(d, device=x.device, dtype=x.dtype)

    if shrinkage is not None:
        a = float(shrinkage)
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"shrinkage must be in [0, 1], got {a}")
        a_t = torch.as_tensor(a, device=x.device, dtype=x.dtype)
    else:
        a_t = _ledoit_wolf_intensity(xc, cov, mu, n, d).detach()

    shrunk = (1.0 - a_t) * cov + a_t * target
    return 0.5 * (shrunk + shrunk.t())              # enforce exact symmetry


def _ledoit_wolf_intensity(
    xc: torch.Tensor,
    cov: torch.Tensor,
    mu: torch.Tensor,
    n: int,
    d: int,
) -> torch.Tensor:
    """Ledoit-Wolf (2004) optimal shrinkage intensity towards ``mu * I``.

    Implements the closed-form estimator ``a* = b^2 / delta^2`` where ``delta^2`` is
    the squared Frobenius distance of ``S`` from the target and ``b^2`` (clamped to
    ``<= delta^2``) measures the variance of the sample covariance entries.  Returned
    clamped to ``[0, 1]``.  Used only when the caller did not pass a fixed value.
    """
    # delta^2 = || S - mu I ||_F^2
    diff = cov - mu * torch.eye(d, device=cov.device, dtype=cov.dtype)
    delta_sq = (diff * diff).sum()

    # b_bar^2 : mean over samples of || x_i x_i^T - S ||_F^2 / n.
    # || x_i x_i^T ||_F^2 = ||x_i||^4 ; use the identity to avoid forming d x d per i.
    sq_norms = (xc * xc).sum(dim=1)                 # [N] = ||x_i||^2
    # E_i || x_i x_i^T - S ||_F^2 = E_i ||x_i||^4 - || S ||_F^2  (population identity);
    # the standard L-W sample plug-in for b^2:
    s_fro_sq = (cov * cov).sum()
    pi_hat = (sq_norms * sq_norms).mean() - s_fro_sq
    b_sq = torch.clamp(pi_hat / n, min=torch.zeros((), device=cov.device, dtype=cov.dtype))
    b_sq = torch.minimum(b_sq, delta_sq)            # b^2 <= delta^2 (L-W constraint)

    a = b_sq / (delta_sq + _EPS)
    return a.clamp(0.0, 1.0)


def _sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
    """Symmetric PSD matrix square root via eigendecomposition with clamping.

    ``mat`` is symmetrised, eigendecomposed with :func:`torch.linalg.eigh`, the
    eigenvalues are clamped at ``1e-6`` (guaranteeing a real, well-defined root and a
    finite gradient), and reassembled.  Differentiable.
    """
    sym = 0.5 * (mat + mat.t())
    evals, evecs = torch.linalg.eigh(sym)
    evals = evals.clamp_min(_EIG_FLOOR)
    root_evals = torch.sqrt(evals)
    return (evecs * root_evals.unsqueeze(0)) @ evecs.t()


def gauss_w2(x: torch.Tensor, y: torch.Tensor, shrinkage: float = 0.05) -> torch.Tensor:
    """Closed-form squared 2-Wasserstein between Gaussian fits to ``x`` and ``y``.

    For :math:`\\mathcal{N}(\\mu_x, C_x)` and :math:`\\mathcal{N}(\\mu_y, C_y)`,

    .. math::
        W_2^2 = \\|\\mu_x - \\mu_y\\|^2
                + \\operatorname{Tr}\\!\\left(C_x + C_y
                  - 2\\,(C_x^{1/2} C_y C_x^{1/2})^{1/2}\\right) ,

    the squared Bures-Wasserstein distance.  The covariances are estimated with
    :func:`ledoit_wolf_cov` (fixed ``shrinkage``) so they are well-conditioned, and all
    matrix square roots use the symmetric-eigendecomposition path with eigenvalue
    clamping at ``1e-6``.

    Parameters
    ----------
    x, y
        Float tensors ``[Nx, d]``, ``[Ny, d]`` (same ``d``, each ``N >= 2``).
    shrinkage
        Fixed Ledoit-Wolf shrinkage intensity for both covariances.

    Returns
    -------
    torch.Tensor
        A scalar tensor (the squared distance, clamped nonnegative), differentiable
        w.r.t. ``x`` and ``y``.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"expected [N, d] tensors, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"feature dims must match: {x.shape[1]} vs {y.shape[1]}")

    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)
    mean_term = ((mu_x - mu_y) ** 2).sum()

    cx = ledoit_wolf_cov(x, shrinkage=shrinkage)
    cy = ledoit_wolf_cov(y, shrinkage=shrinkage)

    cx_half = _sqrtm_psd(cx)
    inner = cx_half @ cy @ cx_half
    inner = 0.5 * (inner + inner.t())
    cross = _sqrtm_psd(inner)

    trace_term = torch.diagonal(cx).sum() + torch.diagonal(cy).sum() - 2.0 * torch.diagonal(cross).sum()
    val = mean_term + trace_term
    return val.clamp_min(0.0)


# --------------------------------------------------------------------------- #
#  Self-test                                                                   #
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    n, d = 200, 8
    x = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    y = (torch.randn(n, d, dtype=torch.float64) * 1.5 + 0.7)
    y = y.clone().detach().requires_grad_(True)

    sw = sliced_wasserstein(x, y, n_projections=64, p=2)
    ed = energy_distance(x, y)
    gw = gauss_w2(x, y, shrinkage=0.05)
    cov = ledoit_wolf_cov(x, shrinkage=0.05)
    cov_auto = ledoit_wolf_cov(x.detach())  # data-driven intensity path

    # --- shape / scalar / finiteness checks ---------------------------------
    for name, t in [("sliced_wasserstein", sw), ("energy_distance", ed), ("gauss_w2", gw)]:
        assert t.dim() == 0, f"{name} must be a scalar, got shape {tuple(t.shape)}"
        assert torch.isfinite(t), f"{name} produced a non-finite value: {t}"
        assert t.item() >= -1e-9, f"{name} should be nonnegative, got {t.item()}"
    assert cov.shape == (d, d), f"ledoit_wolf_cov shape {tuple(cov.shape)} != ({d},{d})"
    assert torch.isfinite(cov).all(), "ledoit_wolf_cov has non-finite entries"
    assert torch.allclose(cov, cov.t(), atol=1e-10), "ledoit_wolf_cov must be symmetric"
    assert torch.linalg.eigvalsh(cov).min().item() > 0, "ledoit_wolf_cov must be PD"
    assert cov_auto.shape == (d, d) and torch.isfinite(cov_auto).all()
    auto_a_eigmin = torch.linalg.eigvalsh(cov_auto).min().item()
    assert auto_a_eigmin > 0, "data-driven L-W cov must be PD"

    # --- self-distances ~ 0 -------------------------------------------------
    xd = x.detach().clone()
    sw_self = sliced_wasserstein(xd, xd, n_projections=128, p=2)
    gw_self = gauss_w2(xd, xd, shrinkage=0.05)
    ed_self = energy_distance(xd, xd)
    assert sw_self.item() < 1e-4, f"sliced_wasserstein(x,x) not ~0: {sw_self.item()}"
    assert gw_self.item() < 1e-6, f"gauss_w2(x,x) not ~0: {gw_self.item()}"
    assert ed_self.item() < 1e-6, f"energy_distance(x,x) not ~0: {ed_self.item()}"

    # --- unequal sample sizes (interpolation path) --------------------------
    x2 = torch.randn(137, d, dtype=torch.float64, requires_grad=True)
    y2 = torch.randn(211, d, dtype=torch.float64, requires_grad=True)
    sw_uneq = sliced_wasserstein(x2, y2, n_projections=32, p=2)
    assert sw_uneq.dim() == 0 and torch.isfinite(sw_uneq), "uneven-N sliced_wasserstein failed"

    # --- gradient flow ------------------------------------------------------
    total = sw + ed + gw + cov.sum() + sw_uneq
    total.backward()
    for name, g in [("x", x.grad), ("y", y.grad), ("x2", x2.grad), ("y2", y2.grad)]:
        assert g is not None, f"no gradient reached {name}"
        assert torch.isfinite(g).all(), f"non-finite gradient on {name}"
        assert g.abs().sum().item() > 0, f"zero gradient on {name}"

    print("h2cmi.align.distances self-test PASSED")
    print(f"  sliced_wasserstein(x,y) = {sw.item():.6f}   (x,x) = {sw_self.item():.2e}")
    print(f"  energy_distance(x,y)    = {ed.item():.6f}   (x,x) = {ed_self.item():.2e}")
    print(f"  gauss_w2(x,y)           = {gw.item():.6f}   (x,x) = {gw_self.item():.2e}")
    print(f"  ledoit_wolf_cov: PD, trace = {torch.diagonal(cov).sum().item():.6f}")
    print(f"  sliced_wasserstein uneven-N (137 vs 211) = {sw_uneq.item():.6f}")
    print("  gradients flow to x, y, x2, y2 (all finite, nonzero)")


if __name__ == "__main__":
    _selftest()
