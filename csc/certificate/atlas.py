"""
csc.certificate.atlas — the source shift atlas (CSC-P0 rewrite).

We estimate THREE signature subspaces from the source (Z, Y, D), all mutually
orthogonalised, so a target marginal shift can be attributed (or not):

  label_dirs (U_pi)   span{ mu_y - mean_y mu_y }            where a change of P(Y) moves
                       (class-mean-difference subspace)       the pooled mean
  cov_dirs   (U_cov)  principal axes of the per-domain        where covariate shift lives
                       common offsets a_d, orthogonal to U_pi (P(Z) moves, boundary fixed)
  concept_dirs        principal axes of the class-specific    where boundary movement leaves
                       residuals r_{d,y}, orthogonal to        a marginal trace -- PRUNED to
                       {U_pi, U_cov}                           directions with bootstrap-
                                                               SIGNIFICANT boundary evidence

  a_d      = mean_y ( mu_{d,y} - mu_y_pool )       common (covariate) part
  r_{d,y}  = (mu_{d,y} - mu_y_pool) - a_d          class-specific (concept) part

Two review fixes baked in here:
  * the LABEL subspace U_pi is explicit, so a pure label (target) shift -- which moves the
    pooled mean along the class-mean subspace -- is NOT silently absorbed into concept_dirs;
  * concept_dirs are DIRECTION-LINKED to evidence: each candidate direction is kept only if
    its boundary-movement loading exceeds a parametric-bootstrap null (under fitted h0).
    Global "there is some boundary movement somewhere" no longer licenses CONCEPT_SUSPECT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .residual_test import (
    residual_decoder_test, ResidualTestResult, fit_h0_proba, sample_labels, _standardise,
)


@dataclass
class ShiftAtlas:
    pooled_mean: np.ndarray
    label_dirs: np.ndarray        # (d, n_label) class-mean-difference subspace U_pi
    cov_dirs: np.ndarray          # (d, n_cov) covariate/nuisance directions
    concept_dirs: np.ndarray      # (d, n_con) SIGNIFICANT concept directions
    sigma_label: float            # source between-domain spread along U_pi
    sigma_cov: float
    sigma_concept: float
    n_domains: int
    n_classes: int


@dataclass
class SourceAnalysis:
    atlas: ShiftAtlas
    test: ResidualTestResult          # global T, p, significant, support
    concept_evidenced: bool           # >= 1 direction-significant concept dir survives
    concept_dir_pvalues: list         # per-rank bootstrap p of the concept singular values
    detail: dict = field(default_factory=dict)


def _pca_dirs(X, var_keep=0.95, max_k=None) -> np.ndarray:
    if X.shape[0] < 1 or np.allclose(X, 0):
        return np.zeros((X.shape[1], 0))
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if S.sum() == 0:
        return np.zeros((X.shape[1], 0))
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(np.searchsorted(cum, var_keep) + 1)
    if max_k is not None:
        k = min(k, max_k)
    return Vt[:max(k, 1)].T


def _orthonormal_complement(dirs, against) -> np.ndarray:
    """Remove `against` subspace from `dirs` columns and re-orthonormalise."""
    if dirs.shape[1] == 0:
        return dirs
    if against.shape[1]:
        dirs = dirs - against @ (against.T @ dirs)
    Q, _ = np.linalg.qr(dirs)
    keep = np.linalg.norm(Q, axis=0) > 1e-8
    return Q[:, keep] if keep.any() else np.zeros((dirs.shape[0], 0))


def _means(Z, Y, classes):
    return {c: Z[Y == c].mean(0) for c in classes}


def _offsets_residuals(Z, Y, D, classes, domains):
    """Return per-domain common offsets a_d and class-specific residuals r_{d,y}."""
    mu_pool = _means(Z, Y, classes)
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
        a_d = np.mean(np.stack(list(dev.values())), axis=0)
        a_list.append(a_d)
        for c, v in dev.items():
            r_list.append(v - a_d)
    d = Z.shape[1]
    A = np.stack(a_list) if a_list else np.zeros((1, d))
    R = np.stack(r_list) if r_list else np.zeros((1, d))
    return A, R, mu_pool


def _residual_singular_values(Z, Y, D, classes, domains, basis) -> np.ndarray:
    """Singular values of the class-specific residuals R projected onto `basis` columns."""
    _, R, _ = _offsets_residuals(Z, Y, D, classes, domains)
    if basis.shape[1] == 0:
        return np.zeros(0)
    Rp = (R - R.mean(0, keepdims=True)) @ basis        # load onto candidate concept dirs
    s = np.linalg.svd(Rp, compute_uv=False)
    return s


def _domain_pooled_means(Z, D, domains):
    return np.stack([Z[D == dd].mean(0) for dd in domains])


def build_atlas(Z, Y, D, var_keep: float = 0.95) -> ShiftAtlas:
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    classes = list(np.unique(Y)); domains = list(np.unique(D))

    A, R, mu_pool = _offsets_residuals(Z, Y, D, classes, domains)

    # Estimate cleanest subspace FIRST and orthogonalise outward, so cross-leakage flows
    # away from the dangerous directions:
    #   cov_dirs     <- a_d              (highest SNR; truly nuisance)
    #   concept_dirs <- r_{d,y}          (label-INVARIANT by construction) , orthog vs cov
    #   label_dirs   <- class-mean diffs , orthog vs {cov, concept}  -> pure task subspace
    cov_dirs = _pca_dirs(A, var_keep)
    concept_dirs = _orthonormal_complement(_pca_dirs(R, var_keep), cov_dirs)
    M = np.stack([mu_pool[c] for c in classes])
    label_dirs = _pca_dirs(M - M.mean(0, keepdims=True), var_keep=0.999)
    label_dirs = _orthonormal_complement(label_dirs, cov_dirs)
    label_dirs = _orthonormal_complement(label_dirs, concept_dirs)

    pooled = Z.mean(0)
    Mden = _domain_pooled_means(Z, D, domains) - pooled       # how domain means move (prior var)
    sigma_label = float(np.sqrt(((Mden @ label_dirs) ** 2).sum(1).mean())) \
        if label_dirs.shape[1] else 0.0
    sigma_cov = float(np.sqrt(((A @ cov_dirs) ** 2).sum(1).mean())) \
        if cov_dirs.shape[1] else 0.0
    sigma_concept = float(np.sqrt(((R @ concept_dirs) ** 2).sum(1).mean())) \
        if concept_dirs.shape[1] else 0.0
    return ShiftAtlas(pooled_mean=pooled, label_dirs=label_dirs, cov_dirs=cov_dirs,
                      concept_dirs=concept_dirs, sigma_label=max(sigma_label, 1e-6),
                      sigma_cov=sigma_cov, sigma_concept=sigma_concept,
                      n_domains=len(domains), n_classes=len(classes))


def analyze_source(Z, Y, D,
                   n_boot: int = 100,
                   n_dir_boot: int = 200,
                   alpha: float = 0.05,
                   var_keep: float = 0.95,
                   C: float = 1.0,
                   seed: int = 0) -> SourceAnalysis:
    """Build the atlas, run the residual test (global concept evidence), and PRUNE the
    concept directions to those carrying boundary evidence above a parametric-bootstrap
    null (direction-linked evidence)."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    classes = list(np.unique(Y)); domains = list(np.unique(D))

    test = residual_decoder_test(Z, Y, D, n_boot=n_boot, alpha=alpha, C=C, seed=seed)
    atlas = build_atlas(Z, Y, D, var_keep=var_keep)

    if test.status != "VALID" or atlas.concept_dirs.shape[1] == 0:
        return SourceAnalysis(atlas=atlas, test=test, concept_evidenced=False,
                              concept_dir_pvalues=[],
                              detail=dict(reason="invalid support or no concept directions"))

    # direction-linked evidence: parametric bootstrap under fitted h0 (boundary domain-
    # independent). Compare observed concept singular values to their bootstrap null.
    cand = atlas.concept_dirs
    obs_sv = _residual_singular_values(Z, Y, D, classes, domains, cand)
    Zs = _standardise(Z)
    p0 = fit_h0_proba(Zs, Y, D, domains, classes, C)
    rng = np.random.default_rng(seed + 7)
    null_sv = np.zeros((n_dir_boot, obs_sv.size))
    for b in range(n_dir_boot):
        Yb = sample_labels(p0, classes, rng)
        sv = _residual_singular_values(Z, Yb, D, classes, domains, cand)
        null_sv[b, : sv.size] = sv[: obs_sv.size]
    # per-rank one-sided p; keep the leading run of significant ranks (parallel analysis)
    pvals = [float((1.0 + np.sum(null_sv[:, k] >= obs_sv[k])) / (1.0 + n_dir_boot))
             for k in range(obs_sv.size)]
    n_keep = 0
    for k in range(len(pvals)):
        if pvals[k] <= alpha:
            n_keep += 1
        else:
            break
    atlas.concept_dirs = cand[:, :n_keep] if n_keep else np.zeros((Z.shape[1], 0))
    if n_keep == 0:
        atlas.sigma_concept = 0.0

    # concept evidence is DIRECTION-LINKED: >=1 concept direction must survive its own
    # parametric-bootstrap null. (We do NOT also require the coarser global-T significance --
    # that is the redundant, budget-noisy gate the review flagged; it is kept for reporting.)
    return SourceAnalysis(atlas=atlas, test=test,
                          concept_evidenced=(n_keep > 0),
                          concept_dir_pvalues=pvals,
                          detail=dict(n_concept_kept=n_keep, obs_sv=obs_sv.tolist(),
                                      global_significant=test.significant))


if __name__ == "__main__":
    from csc.sim.shift_simulator import SimConfig, make_source
    src = make_source(SimConfig(seed=2), n_domains=8, concept_domains=3)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=60, n_dir_boot=120)
    a = sa.atlas
    print(f"label_dirs {a.label_dirs.shape}  cov_dirs {a.cov_dirs.shape}  "
          f"concept_dirs(kept) {a.concept_dirs.shape}")
    print(f"sigma_label={a.sigma_label:.3f} sigma_cov={a.sigma_cov:.3f} "
          f"sigma_concept={a.sigma_concept:.3f}")
    print(f"residual test T={sa.test.T:+.3f} p={sa.test.p_value:.3f} "
          f"concept_evidenced={sa.concept_evidenced} dir_p={[round(p,3) for p in sa.concept_dir_pvalues]}")
    for nm, B in [("label", a.label_dirs), ("cov", a.cov_dirs), ("concept", a.concept_dirs)]:
        for nm2, B2 in [("label", a.label_dirs), ("cov", a.cov_dirs), ("concept", a.concept_dirs)]:
            if nm < nm2 and B.shape[1] and B2.shape[1]:
                print(f"  max|cos({nm},{nm2})|={np.abs(B.T@B2).max():.3f}")
