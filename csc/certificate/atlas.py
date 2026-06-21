"""
csc.certificate.atlas — the source shift atlas.

From the source (Z, Y, D) we estimate the *directions* the source actually moved along,
split into the two interpretable parts (mirroring the h0/h decomposition of the residual
test):

  covariate (nuisance) directions  a_d  = mean_y ( mu_{d,y} - mu_y_pool )
        the part common to all classes in a domain  ==  P(Z) moved, boundary did NOT
  concept directions               r_{d,y} = (mu_{d,y} - mu_y_pool) - a_d
        the class-specific residual  ==  the boundary moved across domains

cov_dirs   = principal axes of {a_d}        (where covariate shift is identifiable)
concept_dirs = principal axes of {r_{d,y}}  (where concept shift left a marginal trace)

We also record the *spread* of the source along each part (leave-one-domain-out scale),
so the target shift can be judged in units of "normal between-domain wobble" rather than
absolute distance. A target shift smaller than that wobble is, by construction, not
distinguishable from a pure invisible conditional shift -> the certifier abstains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ShiftAtlas:
    pooled_mean: np.ndarray       # overall mean of source Z (deployment reference)
    cov_dirs: np.ndarray          # (d, n_cov) orthonormal covariate/nuisance directions
    concept_dirs: np.ndarray      # (d, n_con) orthonormal concept directions
    sigma_cov: float              # RMS covariate between-domain spread
    sigma_concept: float          # RMS concept between-(domain,class) spread
    n_domains: int
    n_classes: int


def _pca_dirs(X, var_keep=0.95, max_k=None) -> np.ndarray:
    """Orthonormal principal directions of rows of X (already centered), keeping enough
    components to explain `var_keep` of the variance."""
    if X.shape[0] < 1 or np.allclose(X, 0):
        return np.zeros((X.shape[1], 0))
    U, S, Vt = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False)
    if S.sum() == 0:
        return np.zeros((X.shape[1], 0))
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(np.searchsorted(cum, var_keep) + 1)
    if max_k is not None:
        k = min(k, max_k)
    k = max(k, 1)
    return Vt[:k].T


def build_atlas(Z, Y, D, var_keep: float = 0.95) -> ShiftAtlas:
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    classes = list(np.unique(Y)); domains = list(np.unique(D))
    d = Z.shape[1]
    mu_pool = {c: Z[Y == c].mean(0) for c in classes}

    a_list, r_list = [], []
    for dd in domains:
        dev = {}
        for c in classes:
            m = (D == dd) & (Y == c)
            if m.sum() == 0:
                continue
            dev[c] = Z[m].mean(0) - mu_pool[c]
        if not dev:
            continue
        a_d = np.mean(np.stack(list(dev.values())), axis=0)   # common (covariate) part
        a_list.append(a_d)
        for c, v in dev.items():
            r_list.append(v - a_d)                            # class-specific (concept) part

    A = np.stack(a_list) if a_list else np.zeros((1, d))
    R = np.stack(r_list) if r_list else np.zeros((1, d))

    cov_dirs = _pca_dirs(A, var_keep)
    concept_dirs = _pca_dirs(R, var_keep)
    # de-correlate: remove the covariate-subspace component from concept directions so a
    # covariate move cannot masquerade as concept (and re-orthonormalise).
    if cov_dirs.shape[1] and concept_dirs.shape[1]:
        concept_dirs = concept_dirs - cov_dirs @ (cov_dirs.T @ concept_dirs)
        Q, _ = np.linalg.qr(concept_dirs)
        keep = np.linalg.norm(Q, axis=0) > 1e-8
        concept_dirs = Q[:, keep] if keep.any() else np.zeros((d, 0))

    sigma_cov = float(np.sqrt((A ** 2).sum(1).mean())) if a_list else 0.0
    sigma_concept = float(np.sqrt((R ** 2).sum(1).mean())) if r_list else 0.0
    return ShiftAtlas(pooled_mean=Z.mean(0), cov_dirs=cov_dirs, concept_dirs=concept_dirs,
                      sigma_cov=sigma_cov, sigma_concept=sigma_concept,
                      n_domains=len(domains), n_classes=len(classes))


if __name__ == "__main__":
    from csc.sim.shift_simulator import SimConfig, make_source
    src = make_source(SimConfig(seed=2), n_domains=8, concept_domains=3)
    atl = build_atlas(src.Z, src.Y, src.D)
    print(f"cov_dirs {atl.cov_dirs.shape}  concept_dirs {atl.concept_dirs.shape}")
    print(f"sigma_cov={atl.sigma_cov:.3f}  sigma_concept={atl.sigma_concept:.3f}")
    if atl.cov_dirs.shape[1] and atl.concept_dirs.shape[1]:
        ang = atl.cov_dirs.T @ atl.concept_dirs
        print(f"max |cos(cov, concept)| = {np.abs(ang).max():.3f} (should be ~0)")
