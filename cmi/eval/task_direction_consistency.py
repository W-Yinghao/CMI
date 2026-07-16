"""CMI-Trace Stage 4 — task-direction / subspace-overlap diagnostics.

Purpose (ladder Stage 4 / gate G2 in configs/cmi_trace_relaxation_ladder.yaml): quantify, on SOURCE data
only, WHETHER the class-discriminative (task) direction is SHARED across subjects, and HOW GEOMETRICALLY
ENTANGLED the task (Y) and subject-identity (D) subspaces are. If the within-subject task contrast points the
same way across subjects, then a subject-axis eraser has a chance of preserving task signal; if task and
identity subspaces coincide, erasing identity necessarily damages the task readout.

Everything here is pure numpy (+ optional scipy/sklearn); NO torch is required. Statistical units are
SUBJECTS, never windows — the cross-subject cosine, its CI (subject-cluster bootstrap) and the within-subject
label-permutation null all treat the subject as the resampling unit, matching the cluster style in
cmi/eval/conditional_subject_leakage.py and cmi/eval/objective_effect_report.py.

Contents
  1. subject_task_contrast / direction_consistency_binary
        per-subject task contrast Delta_s = mean(Z|s,pos) - mean(Z|s,neg); mean cross-subject pairwise cosine,
        subject-cluster bootstrap CI, within-subject label-permutation null (one-sided p), per-subject
        contrast magnitude + contrast SNR.
  2. direction_consistency_multiclass
        4-class (BNCI2014_001) handling WITHOUT arbitrary class collapse: PRIMARY = every unordered class pair,
        macro-averaged, with per-pair values preserved; SENSITIVITY = one-vs-rest contrasts, kept separate.
  3. task_subject_overlap
        normalized Frobenius overlap ||P_Y P_D||_F^2 / min(rank P_Y, rank P_D) between the class-mean (task)
        and subject-mean (identity) subspaces on centered+whitened features, with a matched-rank random-subspace
        null and the principal angles between the two subspaces.
  4. representation_geometry
        feature_norm / covariance condition number / effective_rank / top_singular_value / deleted-rank ratio.

NOT calibrated CMI / not an information quantity — these are geometric witnesses that feed the ladder's G2
"shared task direction" gate and the head-separation reading.
"""
from __future__ import annotations
import numpy as np

from cmi.eval.objective_effect_report import feature_geometry


# ===================================================================== helpers
def _unit_rows(D, eps=1e-12):
    """L2-normalize each row; return (U, keep_mask) dropping (near-)zero rows whose direction is undefined."""
    D = np.asarray(D, float)
    norms = np.linalg.norm(D, axis=1)
    keep = norms > eps
    U = np.zeros_like(D)
    U[keep] = D[keep] / norms[keep, None]
    return U, keep


def _pairwise_cosine_stats(D):
    """Given per-subject contrast rows D [n, dim], drop undefined (zero-norm) rows, then return
    (mean_pairwise_cosine, per_subject_loo_cosine, n_effective). The mean of the per-subject leave-one-out
    mean-cosines equals the mean pairwise cosine exactly, giving a per-SUBJECT scalar for the cluster
    bootstrap."""
    U, keep = _unit_rows(D)
    U = U[keep]
    n = U.shape[0]
    if n < 2:
        return float("nan"), np.array([]), n
    G = U @ U.T
    iu = np.triu_indices(n, 1)
    mean_pc = float(G[iu].mean())
    loo = (G.sum(1) - 1.0) / (n - 1)          # per-subject mean cosine vs all OTHER subjects
    return mean_pc, loo.astype(float), n


def cluster_bootstrap_ci(cluster_values, n_boot=10000, ci=0.95, seed=0):
    """Subject-cluster bootstrap CI of the MEAN over clusters (one scalar per SUBJECT). Resamples subjects
    with replacement — the unit is the subject, never the window. Returns (mean, lo, hi, n_clusters). Mirrors
    objective_effect_report.cluster_bootstrap_ci; reimplemented locally to keep this module torch-free."""
    v = np.asarray([x for x in cluster_values if x is not None and np.isfinite(x)], float)
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        return (float(v[0]), float(v[0]), float(v[0]), 1)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, n, size=(n_boot, n))].mean(1)
    a = (1 - ci) / 2
    return (float(v.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)), n)


# ===================================================================== 1. binary within-subject task contrast
def subject_task_contrast(Z, y, subj, pos_label, neg_label):
    """Per-subject binary task contrast Delta_s = mean(Z | subj=s, y=pos) - mean(Z | subj=s, y=neg).

    Subjects that lack EITHER class (no window of pos or no window of neg) are skipped and recorded. Returns a
    dict:
      subjects        : int array of subjects with BOTH classes (sorted), one row of `deltas` each
      deltas          : [n_used, dim] contrast vectors
      magnitudes      : [n_used] ||Delta_s||
      subjects_skipped: list of (subject, reason) that lacked a class
      n_used / n_skipped
    """
    Z = np.asarray(Z, float)
    y = np.asarray(y)
    subj = np.asarray(subj)
    subjects, deltas, mags, skipped = [], [], [], []
    for s in np.unique(subj):
        ms = subj == s
        mp = ms & (y == pos_label)
        mn = ms & (y == neg_label)
        if mp.sum() == 0 or mn.sum() == 0:
            reason = ("no_pos" if mp.sum() == 0 else "") + ("no_neg" if mn.sum() == 0 else "")
            skipped.append((_py(s), reason))
            continue
        delta = Z[mp].mean(0) - Z[mn].mean(0)
        subjects.append(s)
        deltas.append(delta)
        mags.append(float(np.linalg.norm(delta)))
    subjects = np.array(subjects)
    deltas = np.vstack(deltas) if deltas else np.zeros((0, Z.shape[1]))
    return {"subjects": subjects, "deltas": deltas, "magnitudes": np.array(mags, float),
            "subjects_skipped": skipped, "n_used": len(subjects), "n_skipped": len(skipped)}


def _contrast_snr(Z, y, subj, subjects_used, deltas, pos_label, neg_label):
    """Contrast SNR = mean_s ||Delta_s|| / pooled within-subject-class std ALONG the mean contrast direction.

    Signal = typical magnitude of the per-subject class-mean shift. Noise = pooled standard deviation of single
    windows around their own (subject, class) mean, projected onto the unit average contrast direction u (the
    axis the contrast lives on). Returns (snr, pooled_std, mean_magnitude)."""
    Z = np.asarray(Z, float)
    y = np.asarray(y)
    subj = np.asarray(subj)
    if deltas.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    mean_dir = deltas.mean(0)
    nrm = np.linalg.norm(mean_dir)
    if nrm < 1e-12:
        return float("nan"), float("nan"), float(np.mean(np.linalg.norm(deltas, axis=1)))
    u = mean_dir / nrm
    ssq, dof = 0.0, 0                                   # pooled within-(subject,class) variance along u
    for s in subjects_used:
        ms = subj == s
        for lab in (pos_label, neg_label):
            m = ms & (y == lab)
            if m.sum() < 2:
                continue
            proj = Z[m] @ u
            ssq += float(np.sum((proj - proj.mean()) ** 2))
            dof += int(m.sum() - 1)
    pooled_std = float(np.sqrt(ssq / dof)) if dof > 0 else float("nan")
    mean_mag = float(np.mean(np.linalg.norm(deltas, axis=1)))
    snr = mean_mag / pooled_std if (pooled_std and np.isfinite(pooled_std) and pooled_std > 0) else float("nan")
    return snr, pooled_std, mean_mag


def direction_consistency_binary(Z, y, subj, pos_label, neg_label, n_boot=10000, n_perm=200, seed=0):
    """Cross-subject consistency of the within-subject binary task contrast.

    Returns a dict with:
      mean_pairwise_cosine     : mean cosine of {Delta_s} over unordered subject pairs (the point estimate)
      ci_lo / ci_hi            : subject-cluster bootstrap 95% CI (resamples SUBJECTS; n_clusters == n_used)
      n_clusters               : number of subjects entering the cosine (== n_used with defined contrast)
      perm_null_mean           : mean of the within-subject label-permutation null of mean_pairwise_cosine
      perm_p                   : one-sided p, P(null >= observed), (1+#ge)/(1+n_perm)
      n_perm                   : permutations actually run
      per_subject_magnitude    : {subject: ||Delta_s||}
      mean_magnitude           : mean_s ||Delta_s||
      contrast_snr             : mean ||Delta_s|| / pooled within-(subject,class) std along mean contrast dir
      pooled_within_std        : the denominator of the SNR
      n_used / n_skipped / subjects_skipped / subjects
    """
    con = subject_task_contrast(Z, y, subj, pos_label, neg_label)
    subjects, deltas = con["subjects"], con["deltas"]
    mean_pc, loo, n_eff = _pairwise_cosine_stats(deltas)

    mean_ci, lo, hi, n_clusters = cluster_bootstrap_ci(loo, n_boot=n_boot, ci=0.95, seed=seed)

    # within-subject label-permutation null: shuffle y among each subject's windows (restricted to the two
    # labels), recompute Delta_s and the mean pairwise cosine. Destroys any real class contrast -> ~0.
    Z = np.asarray(Z, float); y = np.asarray(y); subj = np.asarray(subj)
    rng = np.random.default_rng(seed + 101)
    null = []
    two = np.isin(y, [pos_label, neg_label])
    for _ in range(n_perm):
        yp = y.copy()
        for s in subjects:
            m = (subj == s) & two
            idx = np.where(m)[0]
            if idx.size:
                yp[idx] = rng.permutation(y[idx])
        d_perm = subject_task_contrast(Z, yp, subj, pos_label, neg_label)["deltas"]
        mp, _, npair = _pairwise_cosine_stats(d_perm)
        if npair >= 2 and np.isfinite(mp):
            null.append(mp)
    null = np.array(null, float)
    if null.size and np.isfinite(mean_pc):
        perm_p = float((1 + int(np.sum(null >= mean_pc))) / (1 + null.size))
        perm_null_mean = float(null.mean())
    else:
        perm_p, perm_null_mean = float("nan"), float("nan")

    snr, pooled_std, mean_mag = _contrast_snr(Z, y, subj, subjects, deltas, pos_label, neg_label)
    return {
        "pos_label": _py(pos_label), "neg_label": _py(neg_label),
        "mean_pairwise_cosine": mean_pc, "ci_lo": lo, "ci_hi": hi, "n_clusters": n_clusters,
        "perm_null_mean": perm_null_mean, "perm_p": perm_p, "n_perm": int(null.size),
        "per_subject_magnitude": {_py(s): float(m) for s, m in zip(subjects, con["magnitudes"])},
        "mean_magnitude": mean_mag, "contrast_snr": snr, "pooled_within_std": pooled_std,
        "n_used": con["n_used"], "n_skipped": con["n_skipped"],
        "subjects_skipped": con["subjects_skipped"], "subjects": [_py(s) for s in subjects],
    }


# ===================================================================== 2. four-class (no arbitrary collapse)
def direction_consistency_multiclass(Z, y, subj, classes, n_boot=10000, n_perm=200, seed=0):
    """Multiclass task-direction consistency WITHOUT arbitrarily collapsing classes.

    PRIMARY (pairwise): for every unordered class pair (a, b) present, restrict to windows with y in {a, b} and
    run direction_consistency_binary(pos=a, neg=b). Each pair's full report is preserved in `per_pair`. A pair
    is 'valid' when >= 2 subjects have both classes (so a cross-subject cosine exists). The headline number is
    the MACRO-AVERAGE of mean_pairwise_cosine over valid pairs (equal weight per pair, NOT per subject).

    SENSITIVITY (one-vs-rest): for each class a, contrast a vs (not-a). Kept in `ovr_sensitivity`, never mixed
    into the pairwise macro-average.

    Returns:
      macro_avg_consistency          : mean over valid pairs of per-pair mean_pairwise_cosine
      macro_avg_ci_lo / _ci_hi       : subject-cluster bootstrap CI of the per-subject scores POOLED over pairs
      n_valid_pairs / n_pairs        : valid / total unordered pairs
      per_pair                       : {(a, b): binary-report dict} — FULL per-pair breakdown, preserved
      valid_pairs                    : list of (a, b) that were valid
      ovr_sensitivity                : {a: binary-report dict} one-vs-rest, labeled sensitivity
      macro_avg_ovr_consistency      : mean over valid OVR classes of their mean_pairwise_cosine
      classes
    """
    Z = np.asarray(Z, float); y = np.asarray(y); subj = np.asarray(subj)
    classes = [_py(c) for c in classes]
    per_pair, valid_pairs, pair_vals, pooled_loo = {}, [], [], []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a, b = classes[i], classes[j]
            sub = np.isin(y, [a, b])
            rep = direction_consistency_binary(Z[sub], y[sub], subj[sub], a, b,
                                               n_boot=n_boot, n_perm=n_perm, seed=seed)
            per_pair[(a, b)] = rep
            if rep["n_clusters"] >= 2 and np.isfinite(rep["mean_pairwise_cosine"]):
                valid_pairs.append((a, b))
                pair_vals.append(rep["mean_pairwise_cosine"])
                # recover this pair's per-subject leave-one-out cosines for a pooled CI
                con = subject_task_contrast(Z[sub], y[sub], subj[sub], a, b)
                _, loo, _ = _pairwise_cosine_stats(con["deltas"])
                pooled_loo.extend(loo.tolist())
    macro = float(np.mean(pair_vals)) if pair_vals else float("nan")
    _, lo, hi, _ = cluster_bootstrap_ci(pooled_loo, n_boot=n_boot, ci=0.95, seed=seed)

    ovr, ovr_vals = {}, []
    for a in classes:
        yb = np.where(y == a, 1, 0)
        rep = direction_consistency_binary(Z, yb, subj, 1, 0, n_boot=n_boot, n_perm=n_perm, seed=seed)
        ovr[a] = rep
        if rep["n_clusters"] >= 2 and np.isfinite(rep["mean_pairwise_cosine"]):
            ovr_vals.append(rep["mean_pairwise_cosine"])
    return {
        "macro_avg_consistency": macro, "macro_avg_ci_lo": lo, "macro_avg_ci_hi": hi,
        "n_valid_pairs": len(valid_pairs), "n_pairs": len(per_pair),
        "per_pair": per_pair, "valid_pairs": valid_pairs,
        "ovr_sensitivity": ovr, "macro_avg_ovr_consistency": float(np.mean(ovr_vals)) if ovr_vals else float("nan"),
        "classes": classes,
    }


# ===================================================================== 3. task/subject subspace overlap
def _center_whiten(Z, eig_floor=1e-6):
    """CENTERING + WHITENING rule (documented, fixed): (1) subtract the GRAND mean (global centering, so both
    the between-class and between-subject scatters are measured relative to the same origin); (2) ZCA-whiten by
    the POOLED sample covariance so overlap is computed in an isotropic geometry and is NOT dominated by a few
    high-variance directions. Eigenvalues of the pooled covariance are floored at eig_floor * max_eig before
    inversion to stay numerically stable in rank-deficient / low-variance directions.

    Returns (Zw, whiten_matrix, mean, kept_dim). kept_dim = # covariance directions above the floor."""
    Z = np.asarray(Z, float)
    mu = Z.mean(0, keepdims=True)
    Zc = Z - mu
    cov = (Zc.T @ Zc) / max(1, Zc.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    thr = eig_floor * max(float(w.max()), 1e-12)
    inv_sqrt = np.where(w > thr, 1.0 / np.sqrt(np.maximum(w, thr)), 0.0)
    W = V @ np.diag(inv_sqrt) @ V.T                  # ZCA whitening (symmetric)
    return Zc @ W, W, mu, int(np.sum(w > thr))


def _group_mean_basis(Zw, labels, var_frac=0.999, eps=1e-9):
    """Orthonormal ROW basis of the centered group-mean subspace: stack group means (in whitened space),
    subtract their unweighted mean, SVD, keep leading directions capturing var_frac of the spectral energy
    (rank <= n_groups - 1). Returns B [r, dim] with B B^T = I_r, and the rank r."""
    groups = np.unique(labels)
    means = np.vstack([Zw[labels == g].mean(0) for g in groups])
    M = means - means.mean(0, keepdims=True)         # center the group means
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s = np.asarray(s, float)
    if s.size == 0 or s[0] <= eps:
        return np.zeros((0, Zw.shape[1])), 0
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    r = int(np.searchsorted(energy, var_frac) + 1)
    r = max(1, min(r, int(np.sum(s > eps * s[0]))))
    return Vt[:r], r


def _overlap_from_bases(B_Y, B_D):
    """Normalized Frobenius overlap ||P_Y P_D||_F^2 / min(rank P_Y, rank P_D) and cos(principal angles).

    With orthonormal row bases, ||P_Y P_D||_F^2 = ||B_Y B_D^T||_F^2 = sum cos^2(theta_i). Dividing by the number
    of principal angles min(r_Y, r_D) gives the MEAN cos^2 in [0, 1]: 0 iff orthogonal, 1 iff one subspace
    contains the other. Principal-angle cosines = singular values of B_Y B_D^T."""
    r_Y, r_D = B_Y.shape[0], B_D.shape[0]
    if r_Y == 0 or r_D == 0:
        return float("nan"), np.array([]), r_Y, r_D, float("nan")
    C = B_Y @ B_D.T
    sv = np.linalg.svd(C, compute_uv=False)
    cos_pa = np.clip(sv, 0.0, 1.0)
    raw = float(np.sum(cos_pa ** 2))                 # = ||P_Y P_D||_F^2
    norm = raw / min(r_Y, r_D)
    return float(norm), cos_pa.astype(float), r_Y, r_D, raw


def _random_row_basis(dim, r, rng):
    """Orthonormal row basis [r, dim] of a uniformly-random r-subspace of R^dim (QR of a Gaussian matrix)."""
    G = rng.standard_normal((dim, r))
    Q, _ = np.linalg.qr(G)
    return Q.T


def task_subject_overlap(Z, y, subj, n_random=50, var_frac=0.999, eig_floor=1e-6, seed=0):
    """Geometric overlap of the class-mean (task, P_Y) and subject-mean (identity, P_D) subspaces on SOURCE
    features, in a centered+whitened geometry (see _center_whiten).

    Returns:
      normalized_overlap     : ||P_Y P_D||_F^2 / min(rank P_Y, rank P_D)  in [0, 1]
      raw_overlap            : ||P_Y P_D||_F^2 (= sum cos^2 principal angles)
      rank_Y / rank_D        : subspace ranks (<= n_classes-1 / n_subjects-1)
      cos_principal_angles   : cosines of the principal angles between P_Y and P_D (len == min rank)
      null_mean / null_ci_lo / null_ci_hi : matched-rank random-subspace NULL — draw n_random random subspaces
                               of rank rank_D and overlap each with the fixed P_Y (percentile 95% CI)
      null_draws             : the raw null normalized-overlap values
      n_random               : draws actually used
      whitened_dim           : ambient dimension after whitening (covariance directions above the floor)
      centering / whitening  : text describing the fixed rule (grand-mean centering + ZCA pooled-cov whitening)
    """
    Z = np.asarray(Z, float); y = np.asarray(y); subj = np.asarray(subj)
    Zw, _, _, kept = _center_whiten(Z, eig_floor=eig_floor)
    B_Y, r_Y = _group_mean_basis(Zw, y, var_frac=var_frac)
    B_D, r_D = _group_mean_basis(Zw, subj, var_frac=var_frac)
    norm, cos_pa, rY, rD, raw = _overlap_from_bases(B_Y, B_D)

    rng = np.random.default_rng(seed + 202)
    dim = Zw.shape[1]
    null = []
    if r_Y > 0 and r_D > 0 and r_D <= dim:
        for _ in range(n_random):
            B_rand = _random_row_basis(dim, r_D, rng)
            nrm, *_ = _overlap_from_bases(B_Y, B_rand)
            if np.isfinite(nrm):
                null.append(nrm)
    null = np.array(null, float)
    _, nlo, nhi, _ = cluster_bootstrap_ci(null, n_boot=2000, ci=0.95, seed=seed) if null.size else (
        float("nan"), float("nan"), float("nan"), 0)
    return {
        "normalized_overlap": norm, "raw_overlap": raw,
        "rank_Y": rY, "rank_D": rD, "cos_principal_angles": cos_pa,
        "null_mean": float(null.mean()) if null.size else float("nan"),
        "null_ci_lo": nlo, "null_ci_hi": nhi, "null_draws": null, "n_random": int(null.size),
        "whitened_dim": kept,
        "centering": "grand-mean (global) centering",
        "whitening": "ZCA whitening by the pooled sample covariance; eigenvalues floored at "
                     f"{eig_floor}*max_eig before inverse-sqrt",
    }


# ===================================================================== 4. per-representation geometry
def representation_geometry(Z, deleted_rank=None, latent_dim=None, eig_floor=1e-12):
    """Per-representation geometry witnesses used across ladder levels/erasers.

    Returns:
      feature_norm          : mean row L2 norm (from objective_effect_report.feature_geometry)
      top_singular_value    : largest singular value of the centered matrix (idem)
      effective_rank        : entropy-based effective rank exp(H(normalized singular spectrum)) (idem)
      cov_condition_number  : max/min eigenvalue ratio of the covariance (min floored at eig_floor*max_eig)
      cov_max_eig / cov_min_eig
      deleted_rank / latent_dim / deleted_rank_ratio : passthrough + ratio when BOTH are given, else None
    """
    Z = np.asarray(Z, float)
    geom = feature_geometry(Z)                        # feature_norm, top_singular_value, effective_rank
    zc = Z - Z.mean(0, keepdims=True)
    cov = (zc.T @ zc) / max(1, zc.shape[0] - 1)
    w = np.linalg.eigvalsh(cov)
    w = np.clip(w, 0.0, None)
    max_eig = float(w.max()) if w.size else 0.0
    floor = eig_floor * max(max_eig, 1e-12)
    min_eig = float(w.min()) if w.size else 0.0
    cond = max_eig / max(min_eig, floor) if max_eig > 0 else float("nan")
    out = {
        "feature_norm": geom["feature_norm"], "top_singular_value": geom["top_singular_value"],
        "effective_rank": geom["effective_rank"],
        "cov_condition_number": float(cond), "cov_max_eig": max_eig, "cov_min_eig": min_eig,
        "deleted_rank": (int(deleted_rank) if deleted_rank is not None else None),
        "latent_dim": (int(latent_dim) if latent_dim is not None else None),
        "deleted_rank_ratio": (float(deleted_rank) / float(latent_dim)
                               if (deleted_rank is not None and latent_dim not in (None, 0)) else None),
    }
    return out


# ===================================================================== small util
def _py(x):
    """Convert numpy scalars to native python for clean dict keys / JSON."""
    return x.item() if isinstance(x, np.generic) else x
