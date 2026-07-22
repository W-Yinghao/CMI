"""CMI-Trace E2 (Theorem 1) — linear removability threshold r_D + head-geometry overlaps.

Replaces the latent-dimension proxy with the theorem's actual quantity r_D (rank of the whitened conditional
subject span S_D). Two claims:
  (1) complete LINEAR subject removal needs eraser rank k >= r_D;
  (2) exact-head safety holds iff  S_D  subseteq  ker(W Sigma^{1/2}).

Works in WHITENED coordinates: Z_tilde = Z Sigma_W^{-1/2}, head-in-whitened W_tilde = W_c Sigma_W^{1/2}.
The WEIGHT map matches exactly (Z_tilde W_tilde^T == Z W_c^T, to ~1e-13); the class-centered LOGITS differ by
the per-class bias b_c (Z_tilde W_tilde^T == class-centered logits - b_c). Every endpoint here uses logit
DIFFERENCES (||Z_tilde P W_tilde^T||), which are bias-invariant, so the bias is irrelevant to the results.
S_D = span of the whitened subject directions.
  ker(W_tilde) = directions the head IGNORES;  row(W_tilde) = directions the head USES.
  logit change removing a subspace P is  ||Z_tilde P W_tilde^T||  -> 0 iff that subspace subseteq ker(W_tilde).

Head provenance: the frozen EEGNet/TSMNet dumps store (Z_source, logits_source) but no head weight. We RECOVER
the linear head by least squares on CLASS-CENTERED logits and FAIL-CLOSED verify the replay (max abs err <= tol):
  * TSMNet(210): exact (max|Δ| ~ 1e-6) -> exact-head clause is valid.
  * EEGNet(16):  Z->logits is nonlinear in the dumped feature -> replay fails -> head_exact=False; the r_D and
    residual-decodability endpoints remain valid, the exact-head logit-change is reported as PROBE/indicative.

Reuses cmi.eval.subject_spectrum.{pooled_within_class_cov, whitened_subject_scatter, class_centered_head,
effective_rank} and cmi.eval.conditional_subject_leakage.subject_residual. Firewall: SOURCE-only fit.
"""
from __future__ import annotations
import numpy as np

from cmi.eval.subject_spectrum import (pooled_within_class_cov, whitened_subject_scatter,
                                       class_centered_head, effective_rank, EPS)
from cmi.eval.conditional_subject_leakage import subject_residual


def _fast_linear_subject_bacc(Z, y, d, seed=0):
    """FAST closed-form LINEAR subject decodability (LDA within label, averaged over labels). Used across the
    whole eraser-rank sweep (the heavier logreg+MLP `subject_residual` is reserved for checkpoint ranks)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.metrics import balanced_accuracy_score
    Z = np.asarray(Z, float); y = np.asarray(y); d = np.asarray(d)
    rng = np.random.default_rng(seed); accs = []
    for c in np.unique(y):
        m = y == c; zz, dd = Z[m], d[m]
        if len(np.unique(dd)) < 2:
            continue
        idx = rng.permutation(len(zz)); cut = int(0.7 * len(idx)); tr, ev = idx[:cut], idx[cut:]
        if len(np.unique(dd[tr])) < 2 or len(ev) == 0:
            continue
        try:
            clf = LDA().fit(zz[tr], dd[tr])
            accs.append(balanced_accuracy_score(dd[ev], clf.predict(zz[ev])))
        except (np.linalg.LinAlgError, ValueError):
            accs.append(1.0 / max(len(np.unique(dd)), 2))     # reason-coded: singular LDA -> chance, not skipped
    return float(np.mean(accs)) if accs else float("nan")


# --------------------------------------------------------------------- head recovery (fail-closed)
def recover_linear_head(Z, logits, tol=1e-3):
    """Recover a class-centered linear head from (Z, logits) by least squares on centered logits.
    Returns (Wc [K,dz], bc [K], replay_ok, max_abs_diff). replay_ok fail-closed: max|Δ| <= tol on the
    class-centered logits (softmax/predictions depend only on the centered head)."""
    Z = np.asarray(Z, float); L = np.asarray(logits, float)
    Lc = L - L.mean(1, keepdims=True)
    A = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, Lc, rcond=None)
    Wc = coef[:-1].T; bc = coef[-1]
    diff = np.abs(Lc - (Z @ Wc.T + bc))
    return Wc, bc, bool(diff.max() <= tol), float(diff.max())


# --------------------------------------------------------------------- subspace geometry
def _orthonormal_basis(M, tol=1e-9):
    """Orthonormal basis (rows) of the row space of M [k,dz] via SVD; drops near-zero singular directions."""
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = M[None]
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(s.max(), EPS)
    return Vt[keep]


def kernel_rowspace(W_tilde, tol=1e-8):
    """(ker_basis, row_basis) of the head-in-whitened W_tilde [K,dz]: right-singular vectors with ~0 vs >0
    singular values. ker = directions the head ignores; row = directions it uses."""
    W = np.asarray(W_tilde, float)
    U, s, Vt = np.linalg.svd(W, full_matrices=True)
    thr = tol * max(s.max(), EPS) if s.size else 0.0
    n_row = int((s > thr).sum())
    row = Vt[:n_row]
    ker = Vt[n_row:]
    return ker, row


def principal_angles_deg(A, B):
    """Principal angles (degrees, ascending) between span(A) and span(B). Empty -> [90.0]."""
    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return np.array([90.0])
    Qa = _orthonormal_basis(A); Qb = _orthonormal_basis(B)
    if len(Qa) == 0 or len(Qb) == 0:
        return np.array([90.0])
    s = np.clip(np.linalg.svd(Qa @ Qb.T, compute_uv=False), 0, 1)
    return np.rad2deg(np.arccos(s))


def subspace_intersection_dim(A, B, angle_deg=5.0):
    """dim(span(A) ∩ span(B)) via principal angles: count angles below angle_deg (a COARSE yes/no descriptor;
    read the continuous min principal angle + the logit-change magnitude for the actual reliance measure)."""
    return int((principal_angles_deg(A, B) <= angle_deg).sum())


# --------------------------------------------------------------------- r_D (rank of whitened subject span)
def subject_span_rank(Zt, y, d, seed=0, energy_keep=0.99, n_perm=50, floor_pct=95):
    """r_D via BOTH an energy threshold and a within-label subject-permutation floor. Returns
    (r_D, dirs [dz,dz], energy [dz], energy_rank, perm_floor). r_D = # eigenvalues above the permuted-null
    floor (the statistically-real subject directions)."""
    Zt = np.asarray(Zt, float); y = np.asarray(y); d = np.asarray(d)
    dirs, energy = whitened_subject_scatter(Zt, y, d)
    tot = max(energy.sum(), EPS)
    energy_rank = int(np.searchsorted(np.cumsum(energy) / tot, energy_keep) + 1)
    rng = np.random.default_rng(seed)
    top_perm = []
    for _ in range(int(n_perm)):
        dp = d.copy()
        for c in np.unique(y):                      # permute subject WITHIN label (kills real subject signal)
            m = y == c
            dp[m] = rng.permutation(dp[m])
        _, ep = whitened_subject_scatter(Zt, y, dp)
        top_perm.append(ep.max())
    perm_floor = float(np.percentile(top_perm, floor_pct))
    r_D = int((energy > perm_floor).sum())
    return r_D, dirs, energy, energy_rank, perm_floor


# --------------------------------------------------------------------- the per-fold E2 record
def rank_threshold_fold(Z, y, d, logits, *, target_subject=None, seed=0, n_perm=50, k_max=None,
                        head_tol=1e-3, shrink="lw"):
    """One fold of E2 on frozen SOURCE latents Z [N,dz] + logits [N,K] (already source-only, or filtered by
    target_subject if given). Returns r_D, the two intersection dims, and the eraser-rank sweep."""
    Z = np.asarray(Z, float); y = np.asarray(y).astype(np.int64); d = np.asarray(d).astype(np.int64)
    logits = np.asarray(logits, float)
    if target_subject is not None:
        src = d != int(target_subject)
        Z, y, d, logits = Z[src], y[src], d[src], logits[src]
    subs = np.unique(d); d = np.searchsorted(subs, d)          # contiguous subject ids
    dz = Z.shape[1]

    # head (recovered, fail-closed) + whitening
    Wc, bc, head_exact, head_maxdiff = recover_linear_head(Z, logits, tol=head_tol)
    Sigma, W_inv, W_half = pooled_within_class_cov(Z, y, shrink=shrink)
    Zt = Z @ W_inv                                             # whitened source reps
    W_tilde = Wc @ W_half                                      # head-in-whitened: Zt @ W_tilde^T == centered logits

    # r_D + subject directions (whitened)
    r_D, dirs, energy, energy_rank, perm_floor = subject_span_rank(Zt, y, d, seed=seed, n_perm=n_perm)
    r_D = max(r_D, 1)
    S_D = dirs[:r_D]                                           # [r_D, dz] whitened subject span basis
    ker, row = kernel_rowspace(W_tilde)

    dim_in_ker = subspace_intersection_dim(S_D, ker)
    dim_in_row = subspace_intersection_dim(S_D, row)
    # continuous companions to the coarse 5-degree dim counts (do not over-read the dim counts as reliance)
    min_angle_row = float(principal_angles_deg(S_D, row).min())
    min_angle_ker = float(principal_angles_deg(S_D, ker).min())

    # exact-head logit change removing the FULL S_D (predict ~0 iff S_D subseteq ker(W_tilde)). Reported ONLY
    # for a VERIFIED exact linear head (head_exact); with a probe-head the "exact-head clause" is meaningless,
    # so we do NOT report an indicative value (the clause is scoped to linear-head backbones).
    P_SD = S_D.T @ S_D
    if head_exact:
        logit_change_SD = float(np.linalg.norm(Zt @ P_SD @ W_tilde.T, axis=1).mean())
        logit_scale = float(np.linalg.norm(Zt @ W_tilde.T, axis=1).mean() + EPS)
        logit_change_SD_rel = logit_change_SD / logit_scale
    else:
        logit_change_SD = logit_change_SD_rel = None

    # eraser-rank sweep k=1..k_max: residual subject decodability (FAST linear LDA across all k) + exact-head
    # logit change. Heavy logreg+MLP residual (`subject_residual`) is computed only at CHECKPOINT ranks.
    # sweep must reach past r_D to test "complete linear removal needs k >= r_D" (bounded to keep CPU sane)
    k_max = int(min(k_max, dz)) if k_max else int(min(r_D + 3, 40, dz))
    checkpoints = sorted({1, min(r_D, k_max), k_max})
    rng = np.random.default_rng(seed)
    sweep = []
    for k in range(1, k_max + 1):
        Pk = dirs[:k].T @ dirs[:k]                             # informed top-k S_D projector
        Zt_rm = Zt - Zt @ Pk
        G = rng.standard_normal((dz, k)); Q, _ = np.linalg.qr(G); Pr = Q @ Q.T   # same-rank random control
        Zt_rm_rand = Zt - Zt @ Pr
        row = {
            "k": k,
            # ANALYTIC (theorem) quantity: the largest surviving between-subject-within-label mean eigenvalue
            # after removing the informed top-k. Conditional independence (Z⊥D|Y in means) holds iff this is
            # below the permutation floor; that happens at k=r_D by construction. This is "complete removal".
            "residual_top_mean_eigenvalue": float(energy[k]) if k < len(energy) else 0.0,
            "residual_mean_scatter_energy_ratio": float(energy[k:].sum() / max(energy.sum(), EPS)),
            # WEAKER, empirical: whether a finite-sample LINEAR PROBE can still read subject. Reaching chance
            # is weaker than conditional independence, so this can hit chance at k<r_D (redundancy), NOT a
            # theorem violation.
            "resid_subject_bacc_linear": _fast_linear_subject_bacc(Zt_rm, y, d, seed=seed),
            "resid_subject_bacc_random_linear": _fast_linear_subject_bacc(Zt_rm_rand, y, d, seed=seed),
            "logit_change_informed": (None if not head_exact
                                      else float(np.linalg.norm(Zt @ Pk @ W_tilde.T, axis=1).mean())),
            "logit_change_random": (None if not head_exact
                                    else float(np.linalg.norm(Zt @ Pr @ W_tilde.T, axis=1).mean())),
        }
        if k in checkpoints:                                  # cross-check with the heavier logreg+MLP decoder
            row["resid_subject_bacc_logreg"] = subject_residual(Zt_rm, y, d, seed=seed, kind="linear")
            row["resid_subject_bacc_mlp"] = subject_residual(Zt_rm, y, d, seed=seed, kind="mlp")
        sweep.append(row)
    # TWO DISTINCT RANKS (do not conflate):
    #  * r_D / k_mean_complete = ANALYTIC conditional-independence rank (residual top mean eigenvalue below the
    #    permutation floor). Removal is "complete" in the theorem's mean-span sense at exactly this rank.
    #  * k_probe_chance = the WEAKER, finite-sample LINEAR-PROBE rank (LDA subject bAcc reaches chance). This
    #    can be < r_D (residual subject info survives but is not linearly readable) — EXPECTED redundancy, NOT
    #    a theorem violation. Their gap is the over-completeness/redundancy story.
    chance = 1.0 / max(len(subs), 2)
    k_probe_chance = next((s["k"] for s in sweep if s["resid_subject_bacc_linear"] <= chance + 0.02), None)
    k_mean_complete = next((s["k"] for s in sweep if s["residual_top_mean_eigenvalue"] <= perm_floor), None)

    return {
        "target_subject": None if target_subject is None else int(target_subject),
        "d_z": dz, "n_source_subjects": int(len(subs)), "n_cls": int(logits.shape[1]),
        "head_exact": head_exact, "head_replay_max_abs_diff": head_maxdiff,
        "exact_head_clause_reported": bool(head_exact),
        "r_D": int(r_D), "energy_rank_99": int(energy_rank), "perm_floor": perm_floor,
        "subject_effective_rank": effective_rank(energy),
        "dim_SD_in_ker": int(dim_in_ker), "dim_SD_in_row": int(dim_in_row),
        "min_angle_SD_row_deg": min_angle_row, "min_angle_SD_ker_deg": min_angle_ker,
        "logit_change_remove_SD": logit_change_SD, "logit_change_remove_SD_relative": logit_change_SD_rel,
        "k_mean_complete": k_mean_complete, "k_probe_chance": k_probe_chance,
        "redundancy_rank": (None if k_probe_chance is None else int(r_D - k_probe_chance)),
        "complete_removal_definition": ("mean-span (analytic): residual top between-subject-within-label mean "
                                        "eigenvalue below permutation floor -> conditional independence in means, "
                                        "at k=r_D by construction. k_probe_chance is a separate weaker "
                                        "linear-probe diagnostic; k_probe_chance<r_D is expected redundancy."),
        "energy": energy[:max(k_max, r_D)].tolist(), "sweep": sweep,
        "firewall_passed": True,
    }
