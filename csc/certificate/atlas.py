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
    residual_decoder_test, ResidualTestResult, fit_h0_proba, sample_labels,
    subject_null_labels, _standardise, stage_seed, check_support_graph,
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
    """Re-estimable CONCEPT-FIRST pipeline (subject-vote when g given).

    A covariate offset is label-INDEPENDENT, so it cancels in the class residual r_{d,y} and can
    only appear in a_d. The DEFINING signature of a concept (boundary) shift is class-CONDITIONAL
    domain structure, i.e. structure in R. So we estimate concept from R FIRST, then take the
    covariate subspace as the per-domain offsets a_d ORTHOGONAL to concept (the pure nuisance).
    This fixes the asymmetric-signature leak: a VISIBLE concept (whose label-mean also enters a_d)
    was previously absorbed into a cov-first cov_dirs, emptying concept_dirs."""
    A, R, mu_pool = _offsets_residuals(Z, Y, D, classes, domains, g)
    Rc = R - R.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(Rc, full_matrices=False)            # concept candidates (class-cond.)
    if s.sum() > 0:
        cum = np.cumsum(s ** 2) / np.sum(s ** 2)
        kc = int(np.searchsorted(cum, var_keep) + 1)
        concept_raw = Vt[:max(kc, 1)].T
    else:
        concept_raw = np.zeros((Z.shape[1], 0))
    A_perp = A - (A @ concept_raw) @ concept_raw.T if concept_raw.shape[1] else A
    cov_dirs = _pca_dirs(A_perp, var_keep)                       # nuisance = a_d minus concept
    return cov_dirs, concept_raw, s, Vt, A, R, mu_pool


# --------------------------------------------------------------------------------------
# atlas + analysis
# --------------------------------------------------------------------------------------
def build_atlas(Z, Y, D, var_keep: float = 0.95, group_ids=None) -> ShiftAtlas:
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    g = None if group_ids is None else np.asarray(group_ids)
    classes = list(np.unique(Y)); domains = list(np.unique(D))

    cov_dirs, concept_raw, s, Vt, A, R, mu_pool = _concept_geometry(Z, Y, D, classes, domains,
                                                                    var_keep, g)
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


def support_signature_strata(groups, D, Y):
    """Group subjects by their SUPPORT SIGNATURE = the set of (domain,class) cells the subject
    occupies. Resampling WITHIN each stratum (to its own size) keeps every occupied domain-class
    cell populated (no silent disappearance, CSC-P1.4.2 #4) and draws WHOLE biological subjects
    (paired conditions intact, #2b). Returns (idx_by_subject, strata{signature:[subjects]})."""
    g = np.asarray(groups); D = np.asarray(D); Y = np.asarray(Y)
    idx_by = {s: np.where(g == s)[0] for s in np.unique(g)}
    sig_of = {s: frozenset(zip(D[idx_by[s]].tolist(), Y[idx_by[s]].tolist())) for s in idx_by}
    strata = {}
    for s, sig in sig_of.items():
        strata.setdefault(sig, []).append(s)
    return idx_by, strata


def stratified_subject_resample(idx_by, strata, rng):
    """Draw whole subjects within each support-signature stratum; FRESH cluster id per copy.
    Returns (row_index, cluster_id)."""
    il, gl, gc = [], [], 0
    for sig, subs in strata.items():
        for s in rng.choice(subs, size=len(subs), replace=True):
            il.append(idx_by[s]); gl.append(np.full(len(idx_by[s]), gc)); gc += 1
    return np.concatenate(il), np.concatenate(gl)


def _cross_split_separability(Z, Y, D, classes, domains, var_keep, groups, seed, min_angle_deg):
    """CROSS-SPLIT cov/concept separability (CSC-P1.4.2 #5), the reviewer's option (a). Split
    subjects into two INDEPENDENT halves. From split A take the RAW per-domain nuisance offsets
    A_cov = {a_d} (label-AVERAGED -> the covariate signature, NOT forced orthogonal to concept).
    From split B take the concept direction concept_B (class-conditional). The genuine question is
    how much of the nuisance offset survives projecting OUT the concept direction:

        frac = ||A_cov - A_cov P_concept_B|| / ||A_cov||,   angle = arcsin(frac).

    angle ~ 90 deg  => the nuisance lives OFF the concept axis  -> separable (incl. a VISIBLE
    concept, whose label-mean component is removed via the INDEPENDENT split, not by construction).
    angle small     => the nuisance offset is itself ~the concept direction -> cov & concept are
    NOT separable (Assumption T fails) -> overlap=True -> abstain. REACTS as the true cov/concept
    angle shrinks. Returns (angle_deg, overlap); UNASSESSED (too few subjects) -> (90, False)."""
    g = np.asarray(groups) if groups is not None else np.arange(len(Y))
    subs = np.unique(g)
    if len(subs) < 4:
        return 90.0, False
    rng = np.random.default_rng(stage_seed(seed, "separability_split"))
    perm = rng.permutation(subs)
    inA = np.isin(g, perm[:len(perm) // 2])
    try:
        A_cov, _, _ = _offsets_residuals(Z[inA], Y[inA], D[inA], classes,
                                         list(np.unique(D[inA])), g[inA])
        concept_B = _concept_geometry(Z[~inA], Y[~inA], D[~inA], classes,
                                      list(np.unique(D[~inA])), var_keep, g[~inA])[1]
    except Exception:
        return 90.0, False
    angle = offset_concept_angle(A_cov, concept_B)
    return angle, bool(angle < min_angle_deg)


def offset_concept_angle(A_cov, concept_dirs):
    """Angle (deg) between the nuisance-offset matrix A_cov and the concept subspace:
    arcsin(||A_cov - A_cov P_concept|| / ||A_cov||). 90 deg = nuisance lies OFF the concept axis
    (separable); -> 0 as the nuisance offset aligns with the concept direction (overlap). Pure +
    deterministic so the diagnostic's REACTION to the true angle is directly unit-testable."""
    na = float(np.linalg.norm(A_cov))
    if concept_dirs.shape[1] == 0 or na < 1e-9:
        return 90.0
    resid = A_cov - (A_cov @ concept_dirs) @ concept_dirs.T
    return float(np.degrees(np.arcsin(np.clip(np.linalg.norm(resid) / na, 0.0, 1.0))))


def _cov_loading(Z, Y, D, classes, domains, var_keep, cov_dirs=None, g=None):
    """Top singular value of the class-residuals R projected onto the NUISANCE subspace (residual
    class-conditional structure leaking into the covariate directions). cov subspace is the
    concept-first nuisance (a_d orthogonal to concept); RE-ESTIMATED when `cov_dirs` is None."""
    cd_est, _, _, _, A, R, _ = _concept_geometry(Z, Y, D, classes, domains, var_keep, g)
    cd = cov_dirs if cov_dirs is not None else cd_est
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
                   invalid_frac_max: float = 0.20,
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
                                 group_ids=group_ids, invalid_frac_max=invalid_frac_max, seed=seed)
    atlas = build_atlas(Z, Y, D, var_keep=var_keep, group_ids=groups)
    # CSC-P1.4.2 #5: the within-split principal angle was an ALGORITHMIC artifact (cov_dirs is
    # built orthogonal to concept), so it could never detect cov/concept overlap. The operational
    # separability gate is CROSS-SPLIT attribution STABILITY: estimate the atlas on two independent
    # subject-halves; if a cov direction in one half aligns with a concept direction in the other
    # (cross angle < min_angle_deg) the cov/concept assignment is NOT stable -> abstain. This has no
    # forced-orthogonality artifact and REACTS as the true cov/concept angle shrinks.
    sep_angle, overlap = _cross_split_separability(Z, Y, D, classes, domains, var_keep,
                                                   groups, seed, min_angle_deg)

    if test.status != "VALID":
        return SourceAnalysis(atlas, test, False, False, overlap, [],
                              detail=dict(reason="invalid support graph"))

    cov_dirs, concept_raw, s_obs, Vt_obs, A, R, _ = _concept_geometry(Z, Y, D, classes, domains,
                                                                      var_keep, groups)
    n_rank = s_obs.size

    # NULL bootstrap under fitted h0 (NOT a row bootstrap): each replicate draws Y* ~ p_hat0
    # and RE-RUNS the whole concept pipeline (re-estimating subspaces + spectrum).
    Zs = _standardise(Z)
    p0 = fit_h0_proba(Zs, Y, D, domains, classes, C, groups)
    rng = np.random.default_rng(stage_seed(seed, "geometry_null"))
    cl_set = set(classes)
    null_top, cov_load_null, n_geom_invalid = [], [], 0
    for b in range(n_dir_boot):
        # cluster-consistent null: one label per (subject,condition) cell when groups given
        Yb = (subject_null_labels(p0, groups, classes, rng, D=D) if groups is not None
              else sample_labels(p0, classes, rng))
        # SAME support-validity check as the residual null (CSC-P1.4.2 #1): a relabel that drops
        # a class / empties the spectrum is an INVALID replicate, charged as extreme below.
        if set(np.unique(Yb)) != cl_set:
            n_geom_invalid += 1
            continue
        try:
            cov_dirs_b, _, s_b, _, _, R_b = _concept_geometry(Z, Yb, D, classes, domains, var_keep, groups)[:6]
        except Exception:
            n_geom_invalid += 1
            continue
        if not s_b.size:
            n_geom_invalid += 1
            continue
        null_top.append(float(s_b[0]))
        # cov null scale: cov subspace RE-ESTIMATED per replicate (not the fixed observed dirs).
        if cov_dirs_b.shape[1]:
            Rcb = R_b - R_b.mean(0, keepdims=True)
            cov_load_null.append(float(np.linalg.svd(Rcb @ cov_dirs_b, compute_uv=False)[0]))
    null_top = np.asarray(null_top)

    # GEOMETRIC concept gate: h0-parametric-bootstrap global max-statistic. On a NO-concept source
    # this has CORRECT type-I (re-estimated Y*~p0 spectrum matches the observed -> p_global high);
    # subject-level power is LOW (cluster-consistent null is conservative) -- the honest envelope
    # cost. It is the type-I-controlling gate; the cross-fitted DECODER is the second required gate.
    # p_global conservative: invalid replicates charged as extreme (CSC-P1.4.2 #1); too many invalid
    # -> the geometric null is not estimable -> concept NOT evidenced (fail closed).
    geom_estimable = n_geom_invalid <= invalid_frac_max * n_dir_boot
    if n_rank and geom_estimable:
        p_global = float((1.0 + np.sum(null_top >= s_obs[0]) + n_geom_invalid) / (1.0 + n_dir_boot))
    else:
        p_global = 1.0
    cov_noise_scale = (float(np.quantile(np.asarray(cov_load_null), 1 - alpha))
                       if (cov_dirs.shape[1] and cov_load_null) else 0.0)
    concept_top = float(s_obs[0]) if n_rank else 0.0

    # CONCEPT EVIDENCE = geometric global max-stat (p_global, type-I controlled) AND the
    # cross-fitted, subject-level residual-DECODER T significant. BOTH required: geometry alone
    # is direction-only; the decoder alone is NOT type-I-valid on a no-concept source (a covariate
    # source's finite-sample class-conditional noise inflates a decoder-only gate). The earlier
    # magnitude-only gate (concept_top >= kappa*cov_noise_scale) was UNCALIBRATED (full-R top vs a
    # cov-subspace projection -> ~always passes); it is reported as a diagnostic, not a gate.
    concept_geom_present = bool(concept_top >= cov_loading_margin_kappa * cov_noise_scale)  # diag
    concept_evidenced = bool(p_global <= alpha) and bool(test.significant)
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
    #   noise_scale = (1-alpha) quantile of the SAME loading under the h0 NULL (the floor above).
    #   eps_stable = cov_loading_margin_kappa (kappa, PRE-REGISTERED >1) * noise_scale.
    #   cov_stable iff U_cov < eps_stable  (affirmative equivalence, not "failed to reject").
    eps_stable = cov_loading_margin_kappa * cov_noise_scale if cov_dirs.shape[1] else np.inf
    if atlas.cov_dirs.shape[1]:
        s_cov_obs = _cov_loading(Z, Y, D, classes, domains, var_keep, atlas.cov_dirs, g=groups)
        clusters = groups if groups is not None else D
        # CSC-P1.4.2 #2b/#4: resample WHOLE biological subjects within each support-signature
        # stratum -> paired conditions intact AND every occupied domain-class cell stays populated.
        idx_by, strata = support_signature_strata(clusters, D, Y)
        rng2 = np.random.default_rng(stage_seed(seed, "cov_bootstrap"))
        boot, n_inv = [], 0
        for b in range(n_dir_boot):
            idx, gid = stratified_subject_resample(idx_by, strata, rng2)
            # RE-RUN the support gate per replicate (cell support + connectivity; the costly
            # interaction-design rank SVD is irrelevant to the cov-loading statistic -> skipped).
            sup_b = check_support_graph(Y[idx], D[idx], group_ids=gid, n_folds=n_folds,
                                        check_design=False)
            val = (_cov_loading(Z[idx], Y[idx], D[idx], classes, list(np.unique(D[idx])),
                                var_keep, g=gid) if sup_b.valid else float("nan"))
            boot.append(val) if np.isfinite(val) else None
            if not (sup_b.valid and np.isfinite(val)):
                n_inv += 1
        boot = np.array(boot)
        # conservative (CSC-P1.4.2 #1/#4): too many invalid replicates -> stability NOT certifiable
        cov_ub = float(np.quantile(boot, 1 - alpha)) if boot.size else np.inf
        cov_stable = bool(cov_ub < eps_stable and n_inv <= invalid_frac_max * n_dir_boot)
    else:
        s_cov_obs, cov_ub, cov_stable, n_inv = 0.0, 0.0, True, 0

    return SourceAnalysis(atlas, test, concept_evidenced, bool(cov_stable), bool(overlap),
                          [p_global],
                          detail=dict(n_concept_kept=n_keep, p_global=p_global,
                                      obs_top_singular=float(s_obs[0]) if n_rank else 0.0,
                                      concept_top=concept_top,
                                      concept_noise_floor=cov_loading_margin_kappa * cov_noise_scale,
                                      concept_geom_present_diag=concept_geom_present,
                                      geometric_maxstat_significant=bool(p_global <= alpha),
                                      cov_loading=s_cov_obs, cov_loading_ub=cov_ub,
                                      cov_loading_null_scale=cov_noise_scale,
                                      cov_loading_margin_kappa=cov_loading_margin_kappa,
                                      eps_stable_cov_units=eps_stable,
                                      cov_boot_invalid=n_inv,
                                      geom_null_invalid=n_geom_invalid,
                                      geom_null_estimable=bool(geom_estimable),
                                      residual_T_significant=bool(test.significant),
                                      null_invalid_replicates=test.null_invalid,
                                      cluster_aware=(groups is not None),
                                      min_principal_angle_deg=atlas.min_principal_angle_deg,
                                      separability_cross_angle_deg=sep_angle,
                                      signature_overlap=bool(overlap),
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
