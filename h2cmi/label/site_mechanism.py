"""Site-specific LABEL mechanism (review section 8).

This module models the assumption -- argued for in the review -- that the
*observed* clinical label ``Ytilde`` is not the ground truth but a noisy,
**site-dependent** corruption of a latent true state ``Ystar``::

    p(Ytilde = c' | Ystar = c, D_site = d) = C_d[c, c']

where ``C_d`` is a per-site, row-stochastic confusion matrix (rows index the
*true* state ``Ystar``, columns the *observed* label ``Ytilde``).  Because each
site only has a handful of subjects, the per-site matrices are unreliable, so we
shrink them hierarchically toward a shared global confusion::

    C_d = (1 - rho_d) * C_global + rho_d * C_d_local

with ``rho_d in [0, 1]`` the per-site amount of "local" deviation that survives
the shrinkage (rho_d = 0 -> fully pooled, rho_d = 1 -> fully local).

Two roles:

* **Generative** (simulator): :meth:`SiteLabelMechanism.corrupt` SAMPLES observed
  labels from known confusion matrices, i.e. it *injects* label-mechanism shift.
* **Inferential** (estimation): :func:`estimate_confusion_em` recovers the
  confusion matrices from the EEG model's soft posteriors over the latent state
  ``p_ystar`` together with the observed labels.  The review notes the model is
  only identifiable up to a permutation of classes, so an ``anchor`` confusion
  (a high-confidence prior) can be supplied to pin the labelling.

Only :mod:`numpy` and the standard library are used.
"""
from __future__ import annotations

import numpy as np

__all__ = ["SiteLabelMechanism", "estimate_confusion_em"]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _row_normalise(mat: np.ndarray) -> np.ndarray:
    """Return ``mat`` with every row renormalised to a probability simplex.

    Rows that are all-zero (or numerically negligible) become uniform so the
    result is always a valid row-stochastic matrix.
    """
    mat = np.asarray(mat, dtype=np.float64)
    mat = np.clip(mat, 0.0, None)
    row_sums = mat.sum(axis=-1, keepdims=True)
    k = mat.shape[-1]
    # where the row sum is ~0, fall back to uniform
    safe = row_sums < _EPS
    out = np.where(safe, 1.0 / k, mat / np.where(safe, 1.0, row_sums))
    return out


def _as_rho_vector(rho, n_sites: int) -> np.ndarray:
    """Coerce ``rho`` (scalar or per-site array) into a clipped length-``n_sites`` vector."""
    rho_arr = np.asarray(rho, dtype=np.float64)
    if rho_arr.ndim == 0:
        rho_arr = np.full(n_sites, float(rho_arr))
    else:
        rho_arr = rho_arr.reshape(-1)
        if rho_arr.shape[0] != n_sites:
            raise ValueError(
                f"rho array has length {rho_arr.shape[0]} but n_sites={n_sites}"
            )
    return np.clip(rho_arr, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# mechanism
# --------------------------------------------------------------------------- #
class SiteLabelMechanism:
    """Hierarchical, site-dependent label-corruption mechanism.

    Parameters
    ----------
    n_classes:
        Number of label classes ``K`` (shared by ``Ystar`` and ``Ytilde``).
    n_sites:
        Number of sites / domains.
    rho:
        Shrinkage weight on the *local* confusion, scalar or per-site array in
        ``[0, 1]``.  ``C_d = (1 - rho_d) C_global + rho_d C_d_local``.
    """

    def __init__(self, n_classes: int, n_sites: int, rho=0.3) -> None:
        if n_classes < 1:
            raise ValueError("n_classes must be >= 1")
        if n_sites < 1:
            raise ValueError("n_sites must be >= 1")
        self.n_classes = int(n_classes)
        self.n_sites = int(n_sites)
        self.rho = _as_rho_vector(rho, self.n_sites)

        # Until set_matrices is called the mechanism is the identity (no
        # corruption): C_global = I and every C_local = I.
        self._C_global: np.ndarray | None = None
        self._C_local: np.ndarray | None = None

    # -- configuration ----------------------------------------------------- #
    def set_matrices(self, C_global: np.ndarray, C_local: np.ndarray) -> None:
        """Install the global and per-site (pre-shrinkage) confusion matrices.

        ``C_global`` has shape ``[K, K]`` and ``C_local`` shape
        ``[n_sites, K, K]``; both are validated and defensively renormalised so
        their rows sum to one.
        """
        k = self.n_classes
        C_global = np.asarray(C_global, dtype=np.float64)
        C_local = np.asarray(C_local, dtype=np.float64)
        if C_global.shape != (k, k):
            raise ValueError(
                f"C_global must have shape {(k, k)}, got {C_global.shape}"
            )
        if C_local.shape != (self.n_sites, k, k):
            raise ValueError(
                f"C_local must have shape {(self.n_sites, k, k)}, got {C_local.shape}"
            )
        self._C_global = _row_normalise(C_global)
        self._C_local = _row_normalise(C_local)

    @property
    def is_fitted(self) -> bool:
        return self._C_global is not None and self._C_local is not None

    # -- queries ----------------------------------------------------------- #
    def _identity(self) -> np.ndarray:
        return np.eye(self.n_classes, dtype=np.float64)

    def confusion(self, site: int) -> np.ndarray:
        """Return the shrunk, row-stochastic confusion ``C_site`` for ``site``.

        If :meth:`set_matrices` has not been called the identity is returned
        (the no-corruption default).
        """
        if not (0 <= site < self.n_sites):
            raise IndexError(f"site {site} out of range [0, {self.n_sites})")
        if not self.is_fitted:
            return self._identity()
        rho_d = self.rho[site]
        C = (1.0 - rho_d) * self._C_global + rho_d * self._C_local[site]
        # the convex combination of two row-stochastic matrices is already
        # row-stochastic, but renormalise to kill any float drift.
        return _row_normalise(C)

    def all_confusions(self) -> np.ndarray:
        """Stack of every site's shrunk confusion, shape ``[n_sites, K, K]``."""
        return np.stack([self.confusion(d) for d in range(self.n_sites)], axis=0)

    # -- generative -------------------------------------------------------- #
    def corrupt(self, ystar: np.ndarray, sites: np.ndarray, rng) -> np.ndarray:
        """Sample observed labels ``ytilde`` from true ``ystar`` per site.

        Parameters
        ----------
        ystar:
            Integer array ``[N]`` of true latent states in ``[0, K)``.
        sites:
            Integer array ``[N]`` of site ids in ``[0, n_sites)``.
        rng:
            A :class:`numpy.random.Generator`.

        Returns
        -------
        ytilde:
            Integer array ``[N]`` of sampled observed labels.
        """
        ystar = np.asarray(ystar).astype(np.int64).reshape(-1)
        sites = np.asarray(sites).astype(np.int64).reshape(-1)
        if ystar.shape != sites.shape:
            raise ValueError("ystar and sites must have the same shape")
        n = ystar.shape[0]
        confs = self.all_confusions()  # [n_sites, K, K]
        ytilde = np.empty(n, dtype=np.int64)
        # Vectorise per (site) group: every sample in a site uses the same C.
        for d in range(self.n_sites):
            mask = sites == d
            if not np.any(mask):
                continue
            rows = confs[d][ystar[mask]]            # [n_d, K] sampling distributions
            # inverse-CDF sampling, one uniform per sample
            cdf = np.cumsum(rows, axis=1)
            cdf[:, -1] = 1.0                         # guard against float drift
            u = rng.random(rows.shape[0])[:, None]
            ytilde[mask] = (u < cdf).argmax(axis=1)
        return ytilde

    # -- inferential ------------------------------------------------------- #
    def loglik(
        self,
        p_ystar: np.ndarray,
        ytilde: np.ndarray,
        sites: np.ndarray,
    ) -> float:
        """Mean log-likelihood of observed labels under the mechanism.

        ``p(ytilde_i | .) = sum_ystar p_ystar[i, ystar] * C_site_i[ystar, ytilde_i]``

        where ``p_ystar`` ``[N, K]`` are the model's posteriors over the *latent*
        true state.  Numerically guarded with a small floor.
        """
        p_ystar = np.asarray(p_ystar, dtype=np.float64)
        ytilde = np.asarray(ytilde).astype(np.int64).reshape(-1)
        sites = np.asarray(sites).astype(np.int64).reshape(-1)
        n = ytilde.shape[0]
        if p_ystar.shape[0] != n:
            raise ValueError("p_ystar and ytilde length mismatch")
        confs = self.all_confusions()  # [n_sites, K, K]
        total = 0.0
        for d in range(self.n_sites):
            mask = sites == d
            if not np.any(mask):
                continue
            C = confs[d]                            # [K, K]
            # p(ytilde | x) over all classes for this group:  [n_d, K]
            #   marginal[i, c'] = sum_c p_ystar[i, c] * C[c, c']
            marginal = p_ystar[mask] @ C            # [n_d, K]
            obs = ytilde[mask]
            probs = marginal[np.arange(marginal.shape[0]), obs]
            total += np.log(np.clip(probs, _EPS, None)).sum()
        return float(total / max(n, 1))


# --------------------------------------------------------------------------- #
# EM estimation
# --------------------------------------------------------------------------- #
def estimate_confusion_em(
    p_ystar: np.ndarray,
    ytilde: np.ndarray,
    sites: np.ndarray,
    n_classes: int,
    n_sites: int,
    rho: float = 0.3,
    iters: int = 30,
    anchor: np.ndarray | None = None,
) -> SiteLabelMechanism:
    """Estimate site confusion matrices by EM over the latent true state.

    Treats the latent true state ``Ystar`` as the hidden variable.  The EEG model
    contributes a *prior* over ``Ystar`` per sample via the soft posteriors
    ``p_ystar`` ``[N, K]``; the confusion matrices are the parameters.

    * **E-step** -- responsibilities::

          r[i, c] ∝ p_ystar[i, c] * C_site_i[c, ytilde_i]

    * **M-step** -- per-site expected counts ``N_d[c, c'] = sum_{i in d} r[i, c] *
      [ytilde_i == c']`` give Laplace-smoothed local rows; the count-weighted
      average across sites (plus an optional ``anchor`` Dirichlet pseudocount)
      gives ``C_global``; the shrunk ``C_d`` is then re-formed for the next E-step.

    Parameters
    ----------
    anchor:
        Optional ``[K, K]`` high-confidence confusion folded into ``C_global`` as
        pseudocounts.  The review notes this is what makes the labelling
        identifiable (otherwise classes are recoverable only up to permutation).

    Returns
    -------
    SiteLabelMechanism
        A fitted mechanism whose :meth:`SiteLabelMechanism.confusion` returns the
        estimated shrunk per-site matrices.
    """
    K = int(n_classes)
    S = int(n_sites)
    p_ystar = np.asarray(p_ystar, dtype=np.float64)
    ytilde = np.asarray(ytilde).astype(np.int64).reshape(-1)
    sites = np.asarray(sites).astype(np.int64).reshape(-1)
    n = ytilde.shape[0]
    if p_ystar.shape != (n, K):
        raise ValueError(f"p_ystar must have shape {(n, K)}, got {p_ystar.shape}")

    p_ystar = _row_normalise(p_ystar)  # defensive

    rho_vec = _as_rho_vector(rho, S)

    # One-hot of the observed label, [N, K], reused every M-step.
    obs_onehot = np.zeros((n, K), dtype=np.float64)
    obs_onehot[np.arange(n), ytilde] = 1.0

    # Precompute per-site sample masks.
    masks = [sites == d for d in range(S)]

    # Anchor pseudocounts for C_global (Dirichlet-style).  Scale so the anchor
    # acts like a modest number of pseudo-observations rather than swamping data.
    if anchor is not None:
        anchor = _row_normalise(np.asarray(anchor, dtype=np.float64))
        if anchor.shape != (K, K):
            raise ValueError(f"anchor must have shape {(K, K)}, got {anchor.shape}")
        anchor_strength = max(1.0, 0.02 * n / S)  # gentle, data-relative
        anchor_counts = anchor_strength * anchor
    else:
        anchor_counts = np.zeros((K, K), dtype=np.float64)

    laplace = 1.0  # Laplace smoothing on confusion rows

    # --- initialise ------------------------------------------------------- #
    mech = SiteLabelMechanism(K, S, rho=rho_vec)
    # Warm start: a confusion biased toward the identity (diagonal-dominant).
    C_global = _row_normalise(np.eye(K) + 0.1)
    C_local = np.stack([_row_normalise(np.eye(K) + 0.1) for _ in range(S)], axis=0)
    mech.set_matrices(C_global, C_local)

    for _ in range(int(iters)):
        confs = mech.all_confusions()  # [S, K, K], shrunk -- used in E-step

        # accumulate per-site expected counts and the per-site total responsibility
        site_counts = np.zeros((S, K, K), dtype=np.float64)
        site_mass = np.zeros((S, K), dtype=np.float64)  # expected count of Ystar=c

        for d in range(S):
            mask = masks[d]
            if not np.any(mask):
                continue
            C = confs[d]                                 # [K, K]
            p_d = p_ystar[mask]                          # [n_d, K]
            y_d = ytilde[mask]                           # [n_d]
            # E-step responsibilities: r[i,c] ∝ p_ystar[i,c] * C[c, y_i]
            likely = C[:, y_d].T                         # [n_d, K]  (C[c, y_i])
            r = p_d * likely                             # [n_d, K]
            r_sum = r.sum(axis=1, keepdims=True)
            r = r / np.clip(r_sum, _EPS, None)
            # expected counts N_d[c, c'] = sum_i r[i,c] * onehot(y_i)[c']
            oh_d = obs_onehot[mask]                       # [n_d, K]
            site_counts[d] = r.T @ oh_d                   # [K, K]
            site_mass[d] = r.sum(axis=0)                  # [K]

        # --- M-step: global confusion = count-weighted avg across sites + anchor
        global_counts = site_counts.sum(axis=0) + anchor_counts  # [K, K]
        C_global = _row_normalise(global_counts + laplace)

        # --- M-step: per-site LOCAL confusion (Laplace-smoothed rows)
        C_local = _row_normalise(site_counts + laplace)          # [S, K, K]

        mech.set_matrices(C_global, C_local)

    return mech


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    rng = np.random.default_rng(0)
    K, S, rho = 3, 4, 0.4
    N = 4000

    # --- generating mechanism: C_global close to identity, random C_local --- #
    C_global_true = _row_normalise(np.eye(K) * 8.0 + 1.0)  # diagonal-dominant
    C_local_true = np.stack(
        [rng.dirichlet(np.ones(K) * 1.5, size=K) for _ in range(S)], axis=0
    )
    gen = SiteLabelMechanism(K, S, rho=rho)
    gen.set_matrices(C_global_true, C_local_true)
    gen_confs = gen.all_confusions()  # the shrunk matrices we must recover

    # --- generate data ---------------------------------------------------- #
    ystar = rng.integers(0, K, size=N)
    sites = rng.integers(0, S, size=N)
    ytilde = gen.corrupt(ystar, sites, rng)

    # --- a "decent" model: one-hot(ystar) blurred slightly ---------------- #
    blur = 0.12
    p_ystar = np.full((N, K), blur / (K - 1), dtype=np.float64)
    p_ystar[np.arange(N), ystar] = 1.0 - blur
    p_ystar = _row_normalise(p_ystar)

    # anchor = the (known) global confusion structure, to pin the labelling
    anchor = C_global_true

    fitted = estimate_confusion_em(
        p_ystar, ytilde, sites, n_classes=K, n_sites=S, rho=rho, iters=40,
        anchor=anchor,
    )
    est_confs = fitted.all_confusions()

    # --- assess recovery (per-site mean-abs-error) ------------------------ #
    maes = [np.mean(np.abs(est_confs[d] - gen_confs[d])) for d in range(S)]
    mean_mae = float(np.mean(maes))
    print("per-site MAE :", [round(m, 4) for m in maes])
    print("mean MAE     :", round(mean_mae, 5))
    assert mean_mae < 0.15, f"confusion recovery too poor: mean MAE={mean_mae:.4f}"

    # --- loglik must beat an identity-confusion baseline ------------------ #
    ll_fitted = fitted.loglik(p_ystar, ytilde, sites)
    baseline = SiteLabelMechanism(K, S, rho=rho)  # not fitted -> identity confusion
    ll_base = baseline.loglik(p_ystar, ytilde, sites)
    print("loglik fitted:", round(ll_fitted, 5))
    print("loglik ident :", round(ll_base, 5))
    assert ll_fitted > ll_base, (
        f"fitted loglik {ll_fitted:.4f} did not beat identity baseline {ll_base:.4f}"
    )

    # --- sanity: corrupt is reproducible & valid -------------------------- #
    yt2 = gen.corrupt(ystar, sites, np.random.default_rng(123))
    assert yt2.shape == (N,) and yt2.min() >= 0 and yt2.max() < K

    print("OK: site label mechanism self-test passed.")


if __name__ == "__main__":
    _self_test()
