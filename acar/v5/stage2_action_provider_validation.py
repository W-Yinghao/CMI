"""ACAR V5 Stage-2 real-action-seam VALIDATION (torch-free; numpy lazy). Structural, fail-closed validators for (a) the v5→old
source-state adapter the frozen acar.actions consume, (b) an action's probability output contract, and (c) the per-action
non-finite feature semantics. These operate on the LDA/adapter and on ALREADY-COMPUTED outputs, so they run on both Pythons; the
torch-only functional probe lives in stage2_real_action_provider.
"""
from __future__ import annotations
from acar.v5 import protocol as P

# old-state fields the frozen acar.actions expect (built by SourceLDA.old_state)
REQUIRED_OLD_STATE_FIELDS = ("clf", "n_cls", "mu_y", "mu_pool", "Sig_pool0", "Sig_y0", "pi_S", "d", "rho", "eps")
# paired features that MUST be finite for every action (routing depends on them)
REQUIRED_FINITE_FEATURES = ("d_entropy", "d_margin", "flip_rate", "JS", "n_eff")
# geometry features that may be NaN ONLY for a probability-only action (t3a, z_post=None)
GEOMETRY_FEATURES = ("Bures", "post_sep")
PROB_ONLY_ACTIONS = ("t3a",)


class Stage2ActionValidationError(RuntimeError):
    pass


def validate_source_state_adapter(source_lda):
    """Fail-closed: the v5→old source-state adapter has every required field, correct shapes ([2,D]/[D,D]/[2]), class order [0,1],
    a duck-typed LDA clf, invertible shared covariance, and finite entries."""
    import numpy as np
    st = getattr(source_lda, "old_state", None)
    if not isinstance(st, dict):
        raise Stage2ActionValidationError("source_lda.old_state must be a dict")
    missing = [f for f in REQUIRED_OLD_STATE_FIELDS if f not in st]
    if missing:
        raise Stage2ActionValidationError(f"adapter old_state missing field(s) {missing}")
    D = int(source_lda.D)
    means = np.asarray(st["mu_y"], float)
    cov = np.asarray(st["Sig_pool0"], float)
    priors = np.asarray(st["pi_S"], float)
    if means.shape != (2, D):
        raise Stage2ActionValidationError(f"mu_y must be [2,{D}], got {means.shape}")
    if cov.shape != (D, D):
        raise Stage2ActionValidationError(f"Sig_pool0 must be [{D},{D}], got {cov.shape}")
    if priors.shape != (2,):
        raise Stage2ActionValidationError(f"pi_S must be [2], got {priors.shape}")
    if int(st["n_cls"]) != 2:
        raise Stage2ActionValidationError("n_cls must be 2")
    if list(np.asarray(st["clf"].classes_).tolist()) != [0, 1]:
        raise Stage2ActionValidationError("clf.classes_ must be [0,1] = {control,case}")
    for name in ("predict_proba", "predict", "coef_", "intercept_"):
        if not hasattr(st["clf"], name):
            raise Stage2ActionValidationError(f"adapter clf missing sklearn-like attribute {name!r}")
    sig_y0 = st["Sig_y0"]
    if not (isinstance(sig_y0, (list, tuple)) and len(sig_y0) == 2):
        raise Stage2ActionValidationError("Sig_y0 must be a length-2 list of per-class covariances")
    if not (np.isfinite(means).all() and np.isfinite(cov).all() and np.isfinite(priors).all()):
        raise Stage2ActionValidationError("adapter means/cov/priors contain non-finite values")
    try:
        np.linalg.inv(cov)
    except np.linalg.LinAlgError as e:
        raise Stage2ActionValidationError(f"adapter covariance is not invertible: {e}")
    return True


def validate_action_output(name, pa, z_post, n):
    """Fail-closed: p_a is [n,2], finite, in [0,1], rows sum to 1; z_post is None ONLY for a probability-only action (t3a) and a
    finite [n,D] geometry array otherwise."""
    import numpy as np
    if name != "identity" and name not in P.ACTIONS:
        raise Stage2ActionValidationError(f"unknown action {name!r}")
    a = np.asarray(pa, float)
    if a.shape != (n, 2):
        raise Stage2ActionValidationError(f"{name}: p_a must be [{n},2], got {a.shape}")
    if not np.isfinite(a).all():
        raise Stage2ActionValidationError(f"{name}: p_a has non-finite entries")
    if (a < -1e-9).any() or (a > 1 + 1e-9).any():
        raise Stage2ActionValidationError(f"{name}: p_a not in [0,1]")
    if not np.allclose(a.sum(axis=1), 1.0, atol=1e-6):
        raise Stage2ActionValidationError(f"{name}: p_a rows must sum to 1")
    if name in PROB_ONLY_ACTIONS:
        if z_post is not None:
            raise Stage2ActionValidationError(f"{name}: probability-only action must return z_post=None")
    else:
        zp = np.asarray(z_post, float)
        if zp.ndim != 2 or zp.shape[0] != n:
            raise Stage2ActionValidationError(f"{name}: z_post must be [{n},D], got {getattr(zp,'shape',None)}")
        if not np.isfinite(zp).all():
            raise Stage2ActionValidationError(f"{name}: z_post has non-finite entries")
    return True


def validate_feature_finiteness(action, features):
    """Fail-closed: the 5 routing features are finite for EVERY action; the 2 geometry features may be NaN ONLY for t3a
    (probability-only). matched_coral/spdim must have finite geometry features."""
    import numpy as np
    for f in REQUIRED_FINITE_FEATURES:
        if f not in features or not np.isfinite(features[f]):
            raise Stage2ActionValidationError(f"{action}: routing feature {f} must be finite, got {features.get(f)}")
    for f in GEOMETRY_FEATURES:
        v = features.get(f)
        if action in PROB_ONLY_ACTIONS:
            continue                                              # NaN allowed (z_post=None)
        if v is None or not np.isfinite(v):
            raise Stage2ActionValidationError(
                f"{action}: geometry feature {f} must be finite (only {PROB_ONLY_ACTIONS} may be NaN), got {v}")
    return True
