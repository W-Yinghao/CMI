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
# source moments -- SAME voting convention everywhere: a (domain,class) cell mean is the
# mean over its SUBJECTS' cell means (one vote per subject) when group_ids are given, so a
# subject with many epochs does not dominate the atlas (matching the target / gate / bootstrap
# / oracle units). Epoch mean when no group_ids.
# --------------------------------------------------------------------------------------
def _cell_mean(Z, mask, g=None):
    if g is None:
        return Z[mask].mean(0)
    gm = g[mask]
    return np.stack([Z[mask][gm == u].mean(0) for u in np.unique(gm)]).mean(0)


def _means(Z, Y, classes, g=None):
    return {c: _cell_mean(Z, Y == c, g) for c in classes}


def _offsets_residuals(Z, Y, D, classes, domains, g=None):
    mu_pool = _means(Z, Y, classes, g)
    a_list, r_list = [], []
    for dd in domains:
        dev = {}
        for c in classes:
            m = (D == dd) & (Y == c)
            if m.sum() == 0:
                continue
            dev[c] = _cell_mean(Z, m, g) - mu_pool[c]      # subject-vote cell mean
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


def _domain_pooled_means(Z, D, domains, g=None):
    return np.stack([_cell_mean(Z, D == dd, g) for dd in domains])


def _concept_geometry(Z, Y, D, classes, domains, var_keep, g=None):
    """Re-estimable concept pipeline (subject-vote when g given): cov_dirs from a_d, then the
    class-residual matrix with the cov subspace removed."""
    A, R, mu_pool = _offsets_residuals(Z, Y, D, classes, domains, g)
    cov_dirs = _pca_dirs(A, var_keep)
    Rc = R - R.mean(0, keepdims=True)
    Rperp = Rc - (Rc @ cov_dirs) @ cov_dirs.T if cov_dirs.shape[1] else Rc
    U, s, Vt = np.linalg.svd(Rperp, full_matrices=False)
    return cov_dirs, Rperp, s, Vt, A, R, mu_pool


# --------------------------------------------------------------------------------------
# atlas + analysis
# --------------------------------------------------------------------------------------
def build_atlas(Z, Y, D, var_keep: float = 0.95, group_ids=None) -> ShiftAtlas:
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    g = None if group_ids is None else np.asarray(group_ids)
    classes = list(np.unique(Y)); domains = list(np.unique(D))

    cov_dirs, Rperp, s, Vt, A, R, mu_pool = _concept_geometry(Z, Y, D, classes, domains, var_keep, g)
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

    pooled = cluster_mean(Z, g)                            # subject-vote pooled mean
    Mden = _domain_pooled_means(Z, D, domains, g) - pooled
    sigma_label = float(np.sqrt(((Mden @ label_dirs) ** 2).sum(1).mean())) if label_dirs.shape[1] else 0.0
    sigma_cov = float(np.sqrt(((A @ cov_dirs) ** 2).sum(1).mean())) if cov_dirs.shape[1] else 0.0
    sigma_concept = float(np.sqrt(((R @ concept_raw) ** 2).sum(1).mean())) if concept_raw.shape[1] else 0.0
    return ShiftAtlas(pooled_mean=pooled, label_dirs=label_dirs, cov_dirs=cov_dirs,
                      concept_dirs=concept_raw, sigma_label=max(sigma_label, 1e-6),
                      sigma_cov=max(sigma_cov, 1e-6), sigma_concept=max(sigma_concept, 1e-6),
                      n_domains=len(domains), n_classes=len(classes),
                      min_principal_angle_deg=min_angle)


def _cov_loading(Z, Y, D, classes, domains, var_keep, cov_dirs=None, g=None):
    """Top singular value of the class-residuals R projected onto the covariate subspace
    (subject-vote when g given). If `cov_dirs` is None it is RE-ESTIMATED (full estimator)."""
    A, R, _ = _offsets_residuals(Z, Y, D, classes, domains, g)
    cd = cov_dirs if cov_dirs is not None else _pca_dirs(A, var_keep)
    if cd.shape[1] == 0:
        return 0.0
    Rc = R - R.mean(0, keepdims=True)
    return float(np.linalg.svd(Rc @ cd, compute_uv=False)[0])


def analyze_source(Z, Y, D,
                   n_boot: int = 80,
                   n_dir_boot: int = 200,
                   alpha: float = 0.05,
                   var_keep: float = 0.95,
                   C: float = 1.0,
                   min_angle_deg: float = MIN_PRINCIPAL_ANGLE_DEG,
                   cov_loading_margin_kappa: float = 1.5,
                   n_folds: int = 4,
                   group_ids=None,
                   seed: int = 0) -> SourceAnalysis:
    """Atlas + (a) h0-NULL max-statistic concept evidence (keeps ONLY the global-max-passing
    direction -> strong-FWER for 'any direction'), (b) covariate equivalence-stability via a
    FULL cluster bootstrap, (c) signature-overlap flag. `group_ids` (subject/session) makes
    the residual test and the cov-stability bootstrap CLUSTER-aware."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    groups = None if group_ids is None else np.asarray(group_ids)
    classes = list(np.unique(Y)); domains = list(np.unique(D))

    test = residual_decoder_test(Z, Y, D, n_folds=n_folds, n_boot=n_boot, alpha=alpha, C=C,
                                 group_ids=group_ids, seed=seed)
    atlas = build_atlas(Z, Y, D, var_keep=var_keep, group_ids=groups)
    overlap = atlas.min_principal_angle_deg < min_angle_deg

    if test.status != "VALID":
        return SourceAnalysis(atlas, test, False, False, overlap, [],
                              detail=dict(reason="invalid support graph"))

    cov_dirs, Rperp, s_obs, Vt_obs, A, R, _ = _concept_geometry(Z, Y, D, classes, domains, var_keep, groups)
    n_rank = s_obs.size

    # NULL bootstrap under fitted h0 (NOT a row bootstrap): each replicate draws Y* ~ p_hat0
    # and RE-RUNS the whole concept pipeline (re-estimating subspaces + spectrum).
    Zs = _standardise(Z)
    p0 = fit_h0_proba(Zs, Y, D, domains, classes, C)
    rng = np.random.default_rng(seed + 7)
    null_top = np.zeros(n_dir_boot)                    # h0-null TOP singular value (max stat)
    cov_load_null = np.zeros(n_dir_boot)               # h0-null cov-subspace loading (noise scale)
    for b in range(n_dir_boot):
        Yb = sample_labels(p0, classes, rng)
        cov_dirs_b, _, s_b, _, _, R_b = _concept_geometry(Z, Yb, D, classes, domains, var_keep, groups)[:6]
        null_top[b] = s_b[0] if s_b.size else 0.0
        # cov null scale goes through the IDENTICAL estimator pipeline: cov subspace is
        # RE-ESTIMATED per replicate (cov_dirs_b), not projected onto the fixed observed dirs.
        if cov_dirs_b.shape[1]:
            Rcb = R_b - R_b.mean(0, keepdims=True)
            cov_load_null[b] = float(np.linalg.svd(Rcb @ cov_dirs_b, compute_uv=False)[0])

    # GLOBAL max-statistic test (strong-FWER for "is there ANY concept direction"): compare the
    # observed TOP singular value to the h0-null TOP singular value. We keep ONLY this first
    # direction if it passes -- a per-rank sequential test would need a rank-k null per rank
    # (deferred); keeping >1 direction off the rank-0 null would NOT control FWER.
    p_global = float((1.0 + np.sum(null_top >= s_obs[0])) / (1.0 + n_dir_boot)) if n_rank else 1.0
    # CONCEPT EVIDENCE == the residual-decoder gate: the geometric global max-statistic AND the
    # cross-fitted residual-decoder test T must BOTH be significant. (Geometry alone is not the
    # claimed residual-decoder certificate.)
    concept_evidenced = (p_global <= alpha) and bool(test.significant)
    n_keep = 1 if concept_evidenced else 0
    if concept_evidenced:
        atlas.concept_dirs = Vt_obs[:1].T
    else:
        atlas.concept_dirs = np.zeros((Z.shape[1], 0))
        atlas.sigma_concept = 1e-6

    # COVARIATE equivalence-stability -- TOST-style  U_cov < eps_stable  (both >= 0):
    #   U_cov = FULL cluster-bootstrap upper CI of the cov-subspace loading. Each replicate
    #           resamples whole CLUSTERS (subjects if group_ids, else domains) WITH replacement
    #           and RE-ESTIMATES cell means, pooled means, (A,R) and the cov subspace -> a valid
    #           upper confidence bound for the full estimator (not a fixed-subspace row boot).
    #   noise_scale = (1-alpha) quantile of the SAME loading under the h0 NULL (dimensionally
    #           matched; empirically ~ the observed point -> a calibrated noise floor).
    #   eps_stable = cov_loading_margin_kappa (kappa, PRE-REGISTERED >1) * noise_scale.
    #   cov_stable iff U_cov < eps_stable  (affirmative equivalence, not "failed to reject").
    cov_noise_scale = float(np.quantile(cov_load_null, 1 - alpha)) if atlas.cov_dirs.shape[1] else np.inf
    eps_stable = cov_loading_margin_kappa * cov_noise_scale
    if atlas.cov_dirs.shape[1]:
        s_cov_obs = _cov_loading(Z, Y, D, classes, domains, var_keep, atlas.cov_dirs, g=groups)
        clusters = groups if groups is not None else D
        uniq = np.unique(clusters)
        idx_by = {c: np.where(clusters == c)[0] for c in uniq}
        rng2 = np.random.default_rng(seed + 11)
        boot = np.empty(n_dir_boot)
        for b in range(n_dir_boot):
            pick = rng2.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by[c] for c in pick])
            gid_sub = None if groups is None else groups[idx]
            try:
                boot[b] = _cov_loading(Z[idx], Y[idx], D[idx], classes,
                                       list(np.unique(D[idx])), var_keep, g=gid_sub)  # cov RE-estimated
            except Exception:
                boot[b] = np.nan
        boot = boot[np.isfinite(boot)]
        cov_ub = float(np.quantile(boot, 1 - alpha)) if boot.size else np.inf
        cov_stable = cov_ub < eps_stable
    else:
        s_cov_obs, cov_ub, cov_stable = 0.0, 0.0, True

    return SourceAnalysis(atlas, test, concept_evidenced, bool(cov_stable), bool(overlap),
                          [p_global],
                          detail=dict(n_concept_kept=n_keep, p_global=p_global,
                                      obs_top_singular=float(s_obs[0]) if n_rank else 0.0,
                                      cov_loading=s_cov_obs, cov_loading_ub=cov_ub,
                                      cov_loading_null_scale=cov_noise_scale,
                                      cov_loading_margin_kappa=cov_loading_margin_kappa,
                                      eps_stable_cov_units=eps_stable,
                                      cluster_aware=(groups is not None),
                                      min_principal_angle_deg=atlas.min_principal_angle_deg,
                                      global_significant=test.significant))


# --------------------------------------------------------------------------------------
# shared decomposition used by BOTH the certifier and the calibrator (same statistic!)
# --------------------------------------------------------------------------------------
def cluster_mean(Z, group_ids=None):
    """Target mean as ONE VOTE PER CLUSTER (subject): mean of per-group means, so a subject
    with many epochs does not dominate. Row mean when no group_ids."""
    Z = np.asarray(Z, float)
    if group_ids is None:
        return Z.mean(0)
    g = np.asarray(group_ids)
    return np.stack([Z[g == u].mean(0) for u in np.unique(g)]).mean(0)


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
