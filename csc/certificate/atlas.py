"""
csc.certificate.atlas — the source shift atlas (CSC-P1.1).

Three signature subspaces from the source (Z, Y, D), mutually orthogonalised in SNR order:

  cov_dirs   (U_cov)  principal axes of per-domain common offsets a_d   (nuisance; P(Z) moves)
  concept_dirs        class-specific residuals r_{d,y}, orthog. vs cov  (boundary moves) --
                       PRUNED to directions with BOUNDARY EVIDENCE
  label_dirs (U_pi)   span{mu_y - mean_y mu_y}, orthog. vs {cov,concept} (where P(Y) moves)

  a_d = mean_y(mu_{d,y}-mu_y_pool)   ;   r_{d,y} = (mu_{d,y}-mu_y_pool) - a_d

CSC-P1.1 review fixes vs the v0/P0 versions:
  * _orthonormal_complement uses the SVD RANK of the residualised matrix (the old QR-norm
    test kept spurious unit-norm columns even when the residual was zero -> the leakage fix
    was unreliable). + a principal-angle SIGNATURE_OVERLAP flag: if the raw subspaces are too
    close to separate, the certifier abstains instead of trusting a forced Gram-Schmidt order.
  * concept evidence is a FULL-bootstrap MAX-statistic / step-down test: every replicate
    RE-ESTIMATES the subspaces (no post-selection bias) and we compare the observed singular
    spectrum to the null of the *re-estimated* spectrum (FWER-controlled for "any direction").
  * COVARIATE_COMPATIBLE now needs POSITIVE equivalence evidence: the boundary movement along
    the covariate subspace must be statistically indistinguishable from the no-boundary null
    (cov_stable), not merely "the concept test did not fire".
  * shared `components` / `visibility_statistic` so the calibrator thresholds the EXACT
    statistic the certifier uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .residual_test import (
    residual_decoder_test, ResidualTestResult, fit_h0_proba, sample_labels, _standardise,
)

# pre-registered geometry constant: raw signature subspaces closer than this principal angle
# are declared non-separable (SIGNATURE_OVERLAP) -> abstain.
MIN_PRINCIPAL_ANGLE_DEG = 20.0


@dataclass
class ShiftAtlas:
    pooled_mean: np.ndarray
    label_dirs: np.ndarray
    cov_dirs: np.ndarray
    concept_dirs: np.ndarray      # only the EVIDENCED concept directions
    sigma_label: float
    sigma_cov: float
    sigma_concept: float
    n_domains: int
    n_classes: int
    min_principal_angle_deg: float = 90.0


@dataclass
class SourceAnalysis:
    atlas: ShiftAtlas
    test: ResidualTestResult
    concept_evidenced: bool           # >=1 concept direction beats the re-estimated null (max-stat)
    cov_stable: bool                  # covariate subspace boundary movement ~ no-boundary null
    signature_overlap: bool           # raw subspaces not separable -> attribution unreliable
    concept_dir_pvalues: list
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# linear-algebra helpers (the bug fixes)
# --------------------------------------------------------------------------------------
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
    """Project `against` OUT of `dirs` and return an orthonormal basis of the RESIDUAL
    subspace, using the SVD rank of the residualised matrix (not QR column norms, which are
    ~1 regardless of rank and so kept spurious directions). The rank tolerance is scaled by
    the ORIGINAL `dirs` magnitude -- NOT the residual's own top singular value, which is ~0
    when dirs lies in `against` and would otherwise let fp noise (~1e-16) pass as rank."""
    if dirs.shape[1] == 0:
        return dirs
    s_in = np.linalg.svd(dirs, compute_uv=False)
    scale = float(s_in[0]) if s_in.size else 1.0
    R = dirs - against @ (against.T @ dirs) if against.shape[1] else dirs
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    if s.size == 0 or scale == 0:
        return np.zeros((dirs.shape[0], 0))
    tol = max(R.shape) * np.finfo(float).eps * scale
    rank = int(np.sum(s > tol))
    return U[:, :rank]


def _principal_angle_cos(B1, B2) -> float:
    """cos of the SMALLEST principal angle between two orthonormal column spaces (1.0 ==
    they share a direction). 0.0 if either is empty."""
    if B1.shape[1] == 0 or B2.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(B1.T @ B2, compute_uv=False)
    return float(np.clip(s[0], 0.0, 1.0))


# --------------------------------------------------------------------------------------
# source moments
# --------------------------------------------------------------------------------------
def _means(Z, Y, classes):
    return {c: Z[Y == c].mean(0) for c in classes}


def _offsets_residuals(Z, Y, D, classes, domains):
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


def _domain_pooled_means(Z, D, domains):
    return np.stack([Z[D == dd].mean(0) for dd in domains])


def _concept_geometry(Z, Y, D, classes, domains, var_keep):
    """Re-estimable concept pipeline: cov_dirs from a_d, then the class-residual matrix with
    the cov subspace removed; returns (cov_dirs, Rperp, singular_values, right_vectors)."""
    A, R, mu_pool = _offsets_residuals(Z, Y, D, classes, domains)
    cov_dirs = _pca_dirs(A, var_keep)
    Rc = R - R.mean(0, keepdims=True)
    Rperp = Rc - (Rc @ cov_dirs) @ cov_dirs.T if cov_dirs.shape[1] else Rc
    U, s, Vt = np.linalg.svd(Rperp, full_matrices=False)
    return cov_dirs, Rperp, s, Vt, A, R, mu_pool


# --------------------------------------------------------------------------------------
# atlas + analysis
# --------------------------------------------------------------------------------------
def build_atlas(Z, Y, D, var_keep: float = 0.95) -> ShiftAtlas:
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    classes = list(np.unique(Y)); domains = list(np.unique(D))

    cov_dirs, Rperp, s, Vt, A, R, mu_pool = _concept_geometry(Z, Y, D, classes, domains, var_keep)
    # raw concept directions (top var_keep of the cov-removed residual spectrum)
    if s.sum() > 0:
        cum = np.cumsum(s ** 2) / np.sum(s ** 2)
        kc = int(np.searchsorted(cum, var_keep) + 1)
        concept_raw = Vt[:max(kc, 1)].T
    else:
        concept_raw = np.zeros((Z.shape[1], 0))

    M = np.stack([mu_pool[c] for c in classes])
    label_raw = _pca_dirs(M - M.mean(0, keepdims=True), var_keep=0.999)
    label_dirs = _orthonormal_complement(_orthonormal_complement(label_raw, cov_dirs), concept_raw)

    # separability of the RAW subspaces (before forced orthogonalisation)
    cos_max = max(_principal_angle_cos(cov_dirs, concept_raw),
                  _principal_angle_cos(cov_dirs, label_raw),
                  _principal_angle_cos(concept_raw, label_raw))
    min_angle = float(np.degrees(np.arccos(min(cos_max, 1.0)))) if cos_max > 0 else 90.0

    pooled = Z.mean(0)
    Mden = _domain_pooled_means(Z, D, domains) - pooled
    sigma_label = float(np.sqrt(((Mden @ label_dirs) ** 2).sum(1).mean())) if label_dirs.shape[1] else 0.0
    sigma_cov = float(np.sqrt(((A @ cov_dirs) ** 2).sum(1).mean())) if cov_dirs.shape[1] else 0.0
    sigma_concept = float(np.sqrt(((R @ concept_raw) ** 2).sum(1).mean())) if concept_raw.shape[1] else 0.0
    return ShiftAtlas(pooled_mean=pooled, label_dirs=label_dirs, cov_dirs=cov_dirs,
                      concept_dirs=concept_raw, sigma_label=max(sigma_label, 1e-6),
                      sigma_cov=max(sigma_cov, 1e-6), sigma_concept=max(sigma_concept, 1e-6),
                      n_domains=len(domains), n_classes=len(classes),
                      min_principal_angle_deg=min_angle)


def analyze_source(Z, Y, D,
                   n_boot: int = 80,
                   n_dir_boot: int = 200,
                   alpha: float = 0.05,
                   var_keep: float = 0.95,
                   C: float = 1.0,
                   min_angle_deg: float = MIN_PRINCIPAL_ANGLE_DEG,
                   cov_stable_margin: float = 1.5,
                   seed: int = 0) -> SourceAnalysis:
    """Atlas + (a) FULL-bootstrap max-statistic concept evidence (no post-selection bias),
    (b) covariate equivalence-stability, (c) signature-overlap separability flag."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    classes = list(np.unique(Y)); domains = list(np.unique(D))

    test = residual_decoder_test(Z, Y, D, n_boot=n_boot, alpha=alpha, C=C, seed=seed)
    atlas = build_atlas(Z, Y, D, var_keep=var_keep)
    overlap = atlas.min_principal_angle_deg < min_angle_deg

    if test.status != "VALID":
        return SourceAnalysis(atlas, test, False, False, overlap, [],
                              detail=dict(reason="invalid support graph"))

    # observed cov-removed concept spectrum + directions
    cov_dirs, Rperp, s_obs, Vt_obs, A, R, _ = _concept_geometry(Z, Y, D, classes, domains, var_keep)
    n_rank = s_obs.size

    # NULL bootstrap under fitted h0 (NOT a row bootstrap): each replicate draws Y* ~ p_hat0
    # (boundary domain-independent) and RE-RUNS the whole concept pipeline -- re-estimating
    # cov/concept subspaces and the singular spectrum. A row bootstrap would resample the
    # OBSERVED (alternative) data and could not calibrate the Type-I error of "any direction".
    Zs = _standardise(Z)
    p0 = fit_h0_proba(Zs, Y, D, domains, classes, C)
    rng = np.random.default_rng(seed + 7)
    null_spec = np.zeros((n_dir_boot, n_rank))
    cov_load_null = np.zeros(n_dir_boot)               # h0-null loading onto the cov subspace
    for b in range(n_dir_boot):
        Yb = sample_labels(p0, classes, rng)
        _, _, s_b, _, _, R_b = _concept_geometry(Z, Yb, D, classes, domains, var_keep)[:6]
        null_spec[b, : min(n_rank, s_b.size)] = s_b[:n_rank]
        if atlas.cov_dirs.shape[1]:
            Rcb = R_b - R_b.mean(0, keepdims=True)
            cov_load_null[b] = float(np.linalg.svd(Rcb @ atlas.cov_dirs, compute_uv=False)[0])

    # step-down: keep leading ranks whose observed s_k beats the (1-alpha) null quantile of
    # the RE-ESTIMATED s_k (s_0 is the FWER-controlled max statistic for "any direction").
    pvals, n_keep = [], 0
    for k in range(n_rank):
        pk = float((1.0 + np.sum(null_spec[:, k] >= s_obs[k])) / (1.0 + n_dir_boot))
        pvals.append(pk)
        if k == n_keep and pk <= alpha:
            n_keep += 1
    concept_evidenced = n_keep > 0
    if concept_evidenced:
        cum = np.cumsum(s_obs ** 2) / np.sum(s_obs ** 2)
        kmax = int(np.searchsorted(cum, var_keep) + 1)
        atlas.concept_dirs = Vt_obs[:min(n_keep, kmax)].T
    else:
        atlas.concept_dirs = np.zeros((Z.shape[1], 0))
        atlas.sigma_concept = 1e-6

    # COVARIATE equivalence-stability -- an explicit TOST-style  U_cov < eps_stable  test
    # (both >= 0):
    #   statistic  = top singular value of R projected onto the cov subspace (a non-negative
    #                loading); its DATA-bootstrap upper CI is U_cov.
    #   noise scale = (1-alpha) quantile of the SAME loading under the h0 NULL (dimensionally
    #                matched cov-subspace null; empirically ~ the observed point, i.e. a
    #                well-calibrated noise floor).
    #   margin     = eps_stable = cov_stable_margin * noise_scale  -> a PRE-REGISTERED
    #                negligibility multiplier (>1), so the test genuinely AFFIRMS "cov-boundary
    #                movement <= cov_stable_margin x noise = negligible" (equivalence), rather
    #                than "failed to reject". cov_stable_margin is swept in the freeze.
    #   cov_stable iff  U_cov  <  eps_stable.
    noise_scale_cov = float(np.quantile(cov_load_null, 1 - alpha)) if atlas.cov_dirs.shape[1] else np.inf
    eps_stable_cov = cov_stable_margin * noise_scale_cov
    if atlas.cov_dirs.shape[1]:
        Rc = R - R.mean(0, keepdims=True)
        s_cov_obs = float(np.linalg.svd(Rc @ atlas.cov_dirs, compute_uv=False)[0])
        rng2 = np.random.default_rng(seed + 11)
        nrow = Rc.shape[0]
        cov_load_boot = np.array([
            float(np.linalg.svd(Rc[rng2.integers(0, nrow, nrow)] @ atlas.cov_dirs,
                                compute_uv=False)[0]) for _ in range(n_dir_boot)])
        cov_ub = float(np.quantile(cov_load_boot, 1 - alpha))
        cov_stable = cov_ub < eps_stable_cov
    else:
        s_cov_obs, cov_ub, cov_stable = 0.0, 0.0, True

    return SourceAnalysis(atlas, test, concept_evidenced, bool(cov_stable), bool(overlap),
                          pvals,
                          detail=dict(n_concept_kept=n_keep, obs_spectrum=s_obs.tolist(),
                                      cov_loading=s_cov_obs, cov_loading_ub=cov_ub,
                                      cov_noise_scale=noise_scale_cov,
                                      cov_stable_margin=cov_stable_margin,
                                      eps_stable_cov=eps_stable_cov,
                                      min_principal_angle_deg=atlas.min_principal_angle_deg,
                                      global_significant=test.significant))


# --------------------------------------------------------------------------------------
# shared decomposition used by BOTH the certifier and the calibrator (same statistic!)
# --------------------------------------------------------------------------------------
def _proj(delta, basis):
    if basis.shape[1] == 0:
        return np.zeros_like(delta)
    return basis @ (basis.T @ delta)


def components(atlas: ShiftAtlas, delta: np.ndarray) -> dict:
    p_lab = _proj(delta, atlas.label_dirs)
    p_cov = _proj(delta, atlas.cov_dirs)
    p_con = _proj(delta, atlas.concept_dirs)
    resid = delta - p_lab - p_cov - p_con
    c_lab, c_cov, c_con, c_res = (float(np.linalg.norm(v)) for v in (p_lab, p_cov, p_con, resid))
    s_lab = atlas.sigma_label if atlas.sigma_label > 1e-8 else 1.0
    s_cov = atlas.sigma_cov if atlas.sigma_cov > 1e-8 else 1.0
    s_con = atlas.sigma_concept if atlas.sigma_concept > 1e-8 else 1.0
    return dict(n_label=c_lab / s_lab, n_cov=c_cov / s_cov, n_concept=c_con / s_con,
                n_resid=c_res / s_cov, c_label=c_lab, c_cov=c_cov, c_concept=c_con, c_resid=c_res)


def visibility_statistic(atlas: ShiftAtlas, Z_target: np.ndarray) -> float:
    """The EXACT 'is it visible?' statistic the certifier thresholds: max over the visible
    (non-label) normalised components. The calibrator must threshold THIS."""
    delta = np.asarray(Z_target, float).mean(0) - atlas.pooled_mean
    c = components(atlas, delta)
    return max(c["n_cov"], c["n_concept"], c["n_resid"])


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from csc.sim.shift_simulator import SimConfig, make_source
    src = make_source(SimConfig(seed=2), n_domains=8, concept_domains=3)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=40, n_dir_boot=150)
    a = sa.atlas
    print(f"label {a.label_dirs.shape} cov {a.cov_dirs.shape} concept(kept) {a.concept_dirs.shape}")
    print(f"sigma_label={a.sigma_label:.3f} sigma_cov={a.sigma_cov:.3f} sigma_concept={a.sigma_concept:.3f}")
    print(f"concept_evidenced={sa.concept_evidenced} cov_stable={sa.cov_stable} "
          f"overlap={sa.signature_overlap} min_angle={a.min_principal_angle_deg:.1f}deg")
    print(f"dir_p={[round(p,3) for p in sa.concept_dir_pvalues]} "
          f"cov_load={sa.detail['cov_loading']:.3f} ub={sa.detail['cov_loading_ub']:.3f} "
          f"null_q={sa.detail['null_top_q']:.3f}")
