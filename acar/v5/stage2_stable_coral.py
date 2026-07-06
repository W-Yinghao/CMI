"""ACAR V5 Stage-2-local STABLE CORAL — `stable_matched_coral_v1` (numpy only; NO torch, NO cmi.eval / pmct_predict_serialized).

Deterministic, bounded, rank-aware whiten-color transport that CANNOT produce unbounded transport from a rank-deficient target
covariance (the matched_coral non-finite blocker on the V5 256-D / 32-window substrate; see
notes/ACAR_V5_STAGE2B_REAL_SELECTION_BLOCKED_MATCHED_CORAL_NONFINITE.md). It keeps the action NAME `matched_coral` and the
whiten-color idea, but (a) eigenvalue-conditions the shrunk covariances (condition-number cap), (b) SVD-caps the transport
operator gain, and (c) reliability-interpolates the transport against identity by an amount that shrinks with the batch's
rank reliability. Class readout is the source-state LDA f_0 ONLY. Fail-closed: NO silent identity fallback — if the bounded
operator still yields a non-finite output, it raises.
"""
from __future__ import annotations

IMPLEMENTATION = "stable_matched_coral_v1"
RHO = 0.1                          # unchanged from the V5 source-state adapter
EPS = 1e-3                         # unchanged
CONDITION_NUMBER_CAP = 1e6         # eigenvalue floor caps cond(Σ_shrunk) ≤ this before (inv)sqrt
TRANSPORT_OPERATOR_SMAX = 10.0     # SVD cap on the whiten-color transport operator gain
GATE_KAPPA = 8.0                   # reliability gate: g_unc = exp(-kappa · se)


class Stage2StableCoralError(RuntimeError):
    """Raised when the bounded operator still yields a non-finite / contract-violating output (fail-closed; no identity fallback)."""


def _shrink(C, rho=RHO, eps=EPS):
    """Symmetric shrink: (1-rho)C + rho·trace(C)/D·I + eps·I."""
    import numpy as np
    C = 0.5 * (np.asarray(C, float) + np.asarray(C, float).T)
    d = C.shape[0]
    return (1.0 - rho) * C + rho * (np.trace(C) / d) * np.eye(d) + eps * np.eye(d)


def _conditioned_eig(C_shrunk):
    """Eigendecompose a symmetric shrunk covariance and FLOOR the eigenvalues at max(eps, lam_max/cond_cap) so the condition
    number is ≤ CONDITION_NUMBER_CAP. Returns (eigvals_clipped, eigvecs, lam_floor)."""
    import numpy as np
    C = 0.5 * (np.asarray(C_shrunk, float) + np.asarray(C_shrunk, float).T)
    w, Vv = np.linalg.eigh(C)
    lam_max = float(w.max())
    lam_floor = max(EPS, lam_max / CONDITION_NUMBER_CAP)
    return np.maximum(w, lam_floor), Vv, lam_floor


def _psd_power(C_shrunk, inv):
    """Conditioned symmetric matrix (inverse-)square-root: V · diag(w^±1/2) · Vᵀ over the floored eigenvalues."""
    import numpy as np
    w, Vv, _ = _conditioned_eig(C_shrunk)
    s = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return (Vv * s) @ Vv.T


def _svd_cap(M, smax=TRANSPORT_OPERATOR_SMAX):
    """Cap the singular values of M at smax (bounds the transport operator gain)."""
    import numpy as np
    U, s, Vt = np.linalg.svd(np.asarray(M, float), full_matrices=False)
    return (U * np.minimum(s, smax)) @ Vt


def _cond(C):
    import numpy as np
    w = np.linalg.eigvalsh(0.5 * (np.asarray(C, float) + np.asarray(C, float).T))
    lo = float(w.min())
    return float("inf") if lo <= 0 else float(w.max()) / lo


def transport_operator(source_lda, Z):
    """Build the bounded whiten-color transport operator M and report its conditioning (for tests + diagnostics)."""
    import numpy as np
    Z = np.asarray(Z, float)
    if Z.ndim != 2 or Z.shape[0] < 2:                                          # target covariance undefined for n<2
        raise Stage2StableCoralError(
            "transport_operator: target covariance undefined for n<2; forced-tail (sub-MIN_BATCH) batches must be "
            "identity-only and must not be routed through matched_coral")
    C_T = np.cov(Z, rowvar=False)
    C_R = np.asarray(source_lda.old_state["Sig_pool0"], float)
    C_T_s, C_R_s = _shrink(C_T), _shrink(C_R)
    M_raw = _psd_power(C_R_s, inv=False) @ _psd_power(C_T_s, inv=True)
    M = _svd_cap(M_raw)
    return {"M": M, "M_raw": M_raw,
            "M_smax": float(np.linalg.svd(M, compute_uv=False).max()),
            "M_raw_smax": float(np.linalg.svd(M_raw, compute_uv=False).max()),
            "cond_T_raw": _cond(C_T), "cond_T_conditioned": _cond(C_T_s + 0.0),
            "cond_T_after_floor": (lambda w: float(w.max()) / float(w.min()))(_conditioned_eig(C_T_s)[0])}


def stable_matched_coral_v1(source_lda, Z):
    """Bounded, deterministic, rank-aware CORAL. Returns (p_a [n,2], z_post [n,D]). Fail-closed (no identity fallback)."""
    import numpy as np
    Z = np.asarray(Z, float)
    if Z.ndim != 2:
        raise Stage2StableCoralError(f"Z must be [n, D], got {Z.shape}")
    n, D = Z.shape
    if n < 2:                                                                  # target covariance undefined for n<2
        raise Stage2StableCoralError(
            "stable_matched_coral_v1: target covariance undefined for n<2; forced-tail (sub-MIN_BATCH) batches must be "
            "identity-only and must not be routed through matched_coral (no silent identity fallback)")
    mu_T = Z.mean(axis=0)
    mu_R = np.asarray(source_lda.old_state["mu_pool"], float)
    op = transport_operator(source_lda, Z)
    M = op["M"]
    Tz = mu_R + (Z - mu_T) @ M.T                                               # transported embedding
    p0 = np.asarray(source_lda.predict_proba(Z), float)                        # identity f_0 (class readout = LDA only)
    g_cov = float(np.clip(n / (2.0 * D), 0.0, 1.0))                            # rank reliability of the batch covariance
    se = float(np.trace(np.cov(p0, rowvar=False)) / max(n, 1)) if n >= 2 else 0.0
    g_unc = float(np.exp(-GATE_KAPPA * se))                                     # gate uncertainty from IDENTITY f_0 probs
    alpha_eff = g_cov * g_unc
    z_post = (1.0 - alpha_eff) * Z + alpha_eff * Tz                            # reliability interpolation vs identity
    p_a = np.asarray(source_lda.predict_proba(z_post), float)
    if not np.isfinite(z_post).all():
        raise Stage2StableCoralError("stable_matched_coral_v1: non-finite z_post (fail-closed; no identity fallback)")
    if not (p_a.shape == (n, 2) and np.isfinite(p_a).all()
            and (p_a >= -1e-9).all() and (p_a <= 1 + 1e-9).all() and np.allclose(p_a.sum(axis=1), 1.0, atol=1e-6)):
        raise Stage2StableCoralError("stable_matched_coral_v1: p_a violates the probability contract (fail-closed)")
    return p_a, z_post
