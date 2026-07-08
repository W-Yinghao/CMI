"""Project B-Next (S2B) — cross-fitted identity-error risk layer for the RefusalFirstRouter.

Lightweight, pure numpy + stdlib. This module does NOT train an EEG model, run a TTA, or modify the
router policy. It fits a source-only, cross-fitted identity-error predictor over calibration records and
turns it into an ``ACARState`` (identity ERROR axis only) that plugs into the EXISTING
``RefusalFirstRouter`` via ``require_acar_error_for_output`` + ``risk_predictions``.

Design (leak-safe, mirrors ACAR's contract):
  - imputation / scaling statistics come ONLY from source training rows; targets never inform the fit;
  - a feature all-NaN in source is DROPPED (audited), never forced to 0;
  - leave-one-group-out cross-fitting produces genuinely out-of-fold source predictions;
  - the conformal qhat is calibrated on OOF residuals via the SAME ``acar.conformal_quantile`` the rest
    of the router uses (strict finite-sample by default);
  - ``make_identity_error_acar_state`` reuses ``acar.fit_acar_state`` on the OOF (true_error, pred_error)
    records, so it never fabricates a bound: an UNAVAILABLE fit yields no ACARState.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from h2cmi.router.features import CalibrationState
from h2cmi.router.reasons import OACIReason, normalize_reasons
from h2cmi.router.acar import ACARConfig, ACARState, conformal_quantile, fit_acar_state


@dataclass(frozen=True)
class ErrorRiskConfig:
    alpha: float = 0.10
    error_budget: float = 0.45
    ridge_alpha: float = 1.0
    min_groups: int = 3
    min_strict_examples: int = 9
    clip_lower: float = 0.0
    clip_upper: float = 1.0


@dataclass(frozen=True)
class ErrorRiskFeatureAudit:
    feature_names: tuple
    used_features: tuple
    dropped_all_nan: tuple
    dropped_zero_variance: tuple
    imputed_features: tuple
    imputation_values: dict
    imputation_source: str = "source_nested_training_mean"


@dataclass(frozen=True)
class ErrorRiskFit:
    state: CalibrationState
    qhat: "float | None"
    relaxed_qhat: "float | None"
    n_source: int
    n_groups: int
    source_oof_pred: np.ndarray
    source_oof_true: np.ndarray
    coef: "np.ndarray | None"
    intercept: "float | None"
    feature_audit: ErrorRiskFeatureAudit
    reason_codes: tuple
    config: ErrorRiskConfig = field(default_factory=ErrorRiskConfig)
    transform: dict = field(default_factory=dict)


# ------------------------------------------------------------------ numeric helpers
def _mat(rows, feats):
    return np.array([[float(r.get(f, float("nan"))) for f in feats] for r in rows], dtype=np.float64)


def _build_transform(Xsrc):
    """Source-only prep: drop all-NaN & zero-variance cols; impute from source mean; standardize."""
    finite = np.isfinite(Xsrc)
    miss = (~finite).sum(0)
    all_nan = ~finite.any(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        colmean = np.nanmean(np.where(finite, Xsrc, np.nan), axis=0)
    fill = np.where(np.isfinite(colmean), colmean, 0.0)
    Xi = np.where(finite, Xsrc, fill[None, :])
    mu = Xi.mean(0)
    sd = Xi.std(0)
    zero_var = sd < 1e-8
    use = (~all_nan) & (~zero_var)
    return dict(use_idx=[i for i in range(Xsrc.shape[1]) if use[i]], all_nan=all_nan,
                zero_var=zero_var, miss=miss, fill=fill, mu=mu,
                sd=np.where(sd < 1e-8, 1.0, sd))


def _apply_transform(tr, rows, feats):
    X = _mat(rows, feats)
    finite = np.isfinite(X)
    Xi = np.where(finite, X, tr["fill"][None, :])
    Z = (Xi - tr["mu"]) / tr["sd"]
    return Z[:, tr["use_idx"]]


def _ridge(Z, y, alpha):
    Z1 = np.hstack([np.ones((Z.shape[0], 1)), Z])
    A = Z1.T @ Z1 + alpha * np.eye(Z1.shape[1])
    A[0, 0] -= alpha
    return np.linalg.solve(A, Z1.T @ y)


def _predict_w(w, Z, lo, hi):
    Z1 = np.hstack([np.ones((Z.shape[0], 1)), Z])
    return np.clip(Z1 @ w, lo, hi)


# ------------------------------------------------------------------ fit
def fit_error_risk_crossfit(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    group_key: str,
    target_key: str = "identity_error",
    config: "ErrorRiskConfig | None" = None,
) -> ErrorRiskFit:
    cfg = config if config is not None else ErrorRiskConfig()
    feats = tuple(feature_names)
    rows = list(rows)
    n = len(rows)
    Xall = _mat(rows, feats)
    tr_full = _build_transform(Xall)
    fa = ErrorRiskFeatureAudit(
        feature_names=feats,
        used_features=tuple(feats[i] for i in tr_full["use_idx"]),
        dropped_all_nan=tuple(feats[i] for i in range(len(feats)) if tr_full["all_nan"][i]),
        dropped_zero_variance=tuple(feats[i] for i in range(len(feats))
                                    if tr_full["zero_var"][i] and not tr_full["all_nan"][i]),
        imputed_features=tuple(feats[i] for i in range(len(feats))
                               if tr_full["miss"][i] > 0 and not tr_full["all_nan"][i]),
        imputation_values={feats[i]: float(tr_full["fill"][i]) for i in tr_full["use_idx"]},
    )
    y = np.array([float(r.get(target_key, float("nan"))) for r in rows], dtype=np.float64)

    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(str(r.get(group_key, "")), []).append(i)
    n_groups = len(groups)

    def _unavailable(reason):
        return ErrorRiskFit(
            state=CalibrationState.UNAVAILABLE, qhat=None, relaxed_qhat=None, n_source=n,
            n_groups=n_groups, source_oof_pred=np.full(n, np.nan), source_oof_true=y,
            coef=None, intercept=None, feature_audit=fa, reason_codes=normalize_reasons([reason]),
            config=cfg, transform=tr_full)

    if not tr_full["use_idx"]:
        return _unavailable(OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION)
    if n_groups < cfg.min_groups:
        return _unavailable(OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION)

    # leave-one-group-out OOF predictions (transform re-fit per training fold, source-only)
    oof = np.full(n, np.nan)
    for g, te in groups.items():
        tri = [i for i in range(n) if i not in set(te)]
        if not tri:
            continue
        tr = _build_transform(_mat([rows[i] for i in tri], feats))
        if not tr["use_idx"]:
            return _unavailable(OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION)
        w = _ridge(_apply_transform(tr, [rows[i] for i in tri], feats), y[tri], cfg.ridge_alpha)
        pr = _predict_w(w, _apply_transform(tr, [rows[i] for i in te], feats), cfg.clip_lower, cfg.clip_upper)
        for j, idx in enumerate(te):
            oof[idx] = pr[j]

    residual = np.maximum(0.0, y - oof)
    qhat, qreasons = conformal_quantile(residual, alpha=cfg.alpha, finite_sample_strict=True)
    relaxed, _ = conformal_quantile(residual, alpha=cfg.alpha, finite_sample_strict=False)

    # final full-source model for target prediction
    w_full = _ridge(_apply_transform(tr_full, rows, feats), y, cfg.ridge_alpha)

    if qhat is None:
        return ErrorRiskFit(
            state=CalibrationState.UNAVAILABLE, qhat=None, relaxed_qhat=relaxed, n_source=n,
            n_groups=n_groups, source_oof_pred=oof, source_oof_true=y,
            coef=w_full[1:], intercept=float(w_full[0]), feature_audit=fa,
            reason_codes=normalize_reasons(qreasons or [OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION]),
            config=cfg, transform=tr_full)

    return ErrorRiskFit(
        state=CalibrationState.AVAILABLE, qhat=float(qhat), relaxed_qhat=relaxed, n_source=n,
        n_groups=n_groups, source_oof_pred=oof, source_oof_true=y,
        coef=w_full[1:], intercept=float(w_full[0]), feature_audit=fa,
        reason_codes=normalize_reasons([OACIReason.OACI_OK]), config=cfg, transform=tr_full)


def predict_error_risk(fit: ErrorRiskFit, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Point identity-error predictions for target rows (source-derived imputation/scaling)."""
    if fit.coef is None or fit.intercept is None or not fit.transform:
        return np.full(len(rows), np.nan)
    Z = _apply_transform(fit.transform, rows, fit.feature_audit.feature_names)
    w = np.concatenate([[fit.intercept], np.asarray(fit.coef, dtype=np.float64)])
    return _predict_w(w, Z, fit.config.clip_lower, fit.config.clip_upper)


def make_identity_error_acar_state(fit: ErrorRiskFit) -> "ACARState | None":
    """Turn an AVAILABLE fit into an ACARState with an IDENTITY ERROR calibration (no harm).

    Reuses acar.fit_acar_state on the OOF (true_error, pred_error) records so the conformal bound is
    NOT fabricated. Returns None when the fit is not AVAILABLE."""
    if fit.state != CalibrationState.AVAILABLE or fit.qhat is None:
        return None
    recs = [dict(action="identity", true_error=float(t), pred_error=float(p))
            for t, p in zip(fit.source_oof_true, fit.source_oof_pred)
            if math.isfinite(t) and math.isfinite(p)]
    acfg = ACARConfig(alpha_error=fit.config.alpha, min_calibration_examples=fit.config.min_strict_examples,
                      finite_sample_strict=True)
    return fit_acar_state(recs, config=acfg, actions=["identity"])


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    FEATS = ("f_signal", "f_noise", "f_optional")

    def _rows(n_groups, per, missing_opt=False, seed=0):
        r = np.random.RandomState(seed)
        rows = []
        for g in range(n_groups):
            base = 0.2 + 0.5 * (g / max(1, n_groups - 1))
            for _ in range(per):
                sig = base + r.normal(0, 0.03)
                err = float(np.clip(sig + r.normal(0, 0.02), 0, 1))
                rows.append(dict(cf_group=f"g{g}", identity_error=err, f_signal=sig,
                                 f_noise=r.normal(0, 1.0),
                                 f_optional=(float("nan") if missing_opt else r.normal(0, 1.0))))
        return rows

    # 1. mixed source errors, 3+ groups -> AVAILABLE
    fit = fit_error_risk_crossfit(_rows(5, 4, seed=1), feature_names=FEATS, group_key="cf_group")
    assert fit.state is CalibrationState.AVAILABLE and fit.qhat is not None, fit.reason_codes
    # 5. no NaN/inf predictions when AVAILABLE
    p = predict_error_risk(fit, _rows(2, 3, seed=2))
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))

    # 2. too few groups -> UNAVAILABLE
    fit2 = fit_error_risk_crossfit(_rows(2, 6, seed=3), feature_names=FEATS, group_key="cf_group")
    assert fit2.state is CalibrationState.UNAVAILABLE
    assert OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION in fit2.reason_codes

    # 3. all-NaN optional feature dropped
    fit3 = fit_error_risk_crossfit(_rows(5, 4, missing_opt=True, seed=4),
                                   feature_names=FEATS, group_key="cf_group")
    assert "f_optional" in fit3.feature_audit.dropped_all_nan
    assert "f_optional" not in fit3.feature_audit.used_features

    # 4. target prediction uses source-derived imputation (target missing feature -> source fill, finite)
    tgt_missing = [dict(f_signal=0.5, f_noise=0.1, f_optional=float("nan"))]
    pm = predict_error_risk(fit, tgt_missing)
    assert np.all(np.isfinite(pm))

    # 6. make_identity_error_acar_state -> identity error available, identity harm None
    st = make_identity_error_acar_state(fit)
    assert st is not None
    ident = st.get("identity")
    assert ident.error_available and ident.harm is None and ident.harm_available
    ub = ident.upper_bounds(predicted_error=0.4)
    assert ub["error"] is not None and ub["harm"] == 0.0
    # unavailable fit fabricates nothing
    assert make_identity_error_acar_state(fit2) is None

    print("error_risk self-test passed")
