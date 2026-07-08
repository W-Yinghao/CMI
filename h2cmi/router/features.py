"""Project B router — feature & calibration-state contract (Step-2B).

Pure Python + numpy. This module ONLY consumes a diagnostics mapping; it never runs a model,
never fits a TTA or a predictor, and imports NOTHING from torch / trainer / tta / gate / harness.
Its whole job is to turn a raw diagnostics dict into:
  - a finite router feature vector (no NaN/inf ever),
  - a legacy 8-d gate feature vector (SafetyGate diagnostic language, for baseline/compat),
  - an ACAR harm-calibration state (available / degenerate / unavailable),
  - a set of OACI reason codes with fail-loud semantics (every imputed/missing/non-finite field
    is reason-coded and recorded; nothing is silently zero-filled).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from h2cmi.router.reasons import OACIReason, has_blocking_reason


# ------------------------------------------------------------------ feature orders
# Legacy 8-d order kept to stay compatible with SafetyGate's diagnostic language (NOT imported).
LEGACY_GATE_FEATURE_KEYS = (
    "delta_density_nll",
    "transform_norm",
    "condition_number",
    "prior_shift",
    "pred_disagreement",
    "cmi_residual",
    "ood_score",
    "ess",
)

# 13-d router order.
ROUTER_FEATURE_KEYS = (
    "n_target",
    "ess",
    "density_nll_source_prior",
    "density_nll_target_prior",
    "support_gap",
    "min_class_responsibility",
    "delta_density_nll",
    "transform_norm",
    "condition_number",
    "prior_shift",
    "pred_disagreement",
    "ood_score",
    "cmi_residual",
)

_REQUIRED_KEYS = (
    "n_target", "ess", "delta_density_nll", "transform_norm",
    "condition_number", "prior_shift", "pred_disagreement", "ood_score",
)


class CalibrationState(str, Enum):
    AVAILABLE = "available"
    DEGENERATE = "degenerate"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RouterFeatureConfig:
    # TOS / support thresholds
    min_target_n: int = 20
    min_ess: float = 8.0
    max_condition_number: float = 50.0
    max_transform_norm: float = 25.0
    max_pred_disagreement: float = 0.50

    # Optional calibrated thresholds. None means "do not enforce".
    max_ood_score: "float | None" = None
    max_density_nll_target_prior: "float | None" = None
    max_support_gap_abs: "float | None" = None
    leakage_residual_budget: "float | None" = None

    # Prior-decoupled audit logic
    prior_shift_info_threshold: float = 0.20
    support_gap_info_threshold: float = 0.05

    # TTA evidence
    min_delta_density_nll_for_tta: float = 0.0

    # ACAR calibration-state detection
    min_calibration_examples: int = 4
    harm_margin: float = 0.02


@dataclass(frozen=True)
class CalibrationSummary:
    state: CalibrationState
    n: int
    n_harm: int
    n_nonharm: int
    gain_min: "float | None"
    gain_mean: "float | None"
    gain_max: "float | None"
    reason_codes: tuple[OACIReason, ...]


@dataclass(frozen=True)
class RouterFeatureBundle:
    vector: np.ndarray
    feature_names: tuple[str, ...]
    legacy_gate_vector: np.ndarray
    legacy_gate_feature_names: tuple[str, ...]
    diagnostics: dict[str, Any]
    reason_codes: tuple[OACIReason, ...]
    tos_pass: bool
    prior_shift_only: bool
    cmi_residual_available: bool
    acar_harm_calibration: CalibrationSummary


# ------------------------------------------------------------------ finite-float coercion
def as_finite_float(
    diagnostics: Mapping[str, Any],
    key: str,
    *,
    required: bool,
    default: float,
    missing_codes: list,
    nonfinite_codes: list,
    missing_keys: list,
    imputed_keys: list,
) -> float:
    """Coerce diagnostics[key] to a finite float, fail-loud.

    - missing REQUIRED key -> append OACI_DIAGNOSTIC_MISSING; record missing+imputed; return default
    - missing OPTIONAL key -> record missing+imputed; return default (NO auto DIAGNOSTIC_MISSING)
    - non-finite / non-numeric -> append OACI_DIAGNOSTIC_NONFINITE; record imputed; return default
    """
    if key not in diagnostics:
        if required:
            missing_codes.append(OACIReason.OACI_DIAGNOSTIC_MISSING)
        missing_keys.append(key)
        imputed_keys.append(key)
        return float(default)
    try:
        val = float(diagnostics[key])
    except (TypeError, ValueError):
        nonfinite_codes.append(OACIReason.OACI_DIAGNOSTIC_NONFINITE)
        imputed_keys.append(key)
        return float(default)
    if not np.isfinite(val):
        nonfinite_codes.append(OACIReason.OACI_DIAGNOSTIC_NONFINITE)
        imputed_keys.append(key)
        return float(default)
    return val


# ------------------------------------------------------------------ ACAR harm calibration state
def assess_acar_harm_calibration(
    gains: "Sequence[float] | None",
    *,
    min_examples: int,
    harm_margin: float,
) -> CalibrationSummary:
    """Classify the source-pseudo-target harm-calibration set.

    gains are ``bAcc_TTA - bAcc_IDENTITY`` per pseudo-target (negative == TTA harm).
      None / too few          -> UNAVAILABLE (OACI_ACAR_INSUFFICIENT_CALIBRATION)
      non-finite               -> UNAVAILABLE (OACI_DIAGNOSTIC_NONFINITE)
      all-harm or all-non-harm -> DEGENERATE  (OACI_ACAR_HARM_CALIBRATION_DEGENERATE)
      both classes present     -> AVAILABLE   (OACI_OK)
    harmed iff gain <= -harm_margin.
    """
    if gains is None:
        return CalibrationSummary(CalibrationState.UNAVAILABLE, 0, 0, 0, None, None, None,
                                  (OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION,))
    arr = np.asarray(list(gains), dtype=float)
    n = int(arr.size)
    if n < int(min_examples):
        return CalibrationSummary(CalibrationState.UNAVAILABLE, n, 0, 0, None, None, None,
                                  (OACIReason.OACI_ACAR_INSUFFICIENT_CALIBRATION,))
    if not bool(np.all(np.isfinite(arr))):
        return CalibrationSummary(CalibrationState.UNAVAILABLE, n, 0, 0, None, None, None,
                                  (OACIReason.OACI_DIAGNOSTIC_NONFINITE,))
    harmed = arr <= -float(harm_margin)
    n_harm = int(harmed.sum())
    n_nonharm = int((~harmed).sum())
    gmin, gmean, gmax = float(arr.min()), float(arr.mean()), float(arr.max())
    if n_harm == 0 or n_nonharm == 0:
        return CalibrationSummary(CalibrationState.DEGENERATE, n, n_harm, n_nonharm, gmin, gmean, gmax,
                                  (OACIReason.OACI_ACAR_HARM_CALIBRATION_DEGENERATE,))
    return CalibrationSummary(CalibrationState.AVAILABLE, n, n_harm, n_nonharm, gmin, gmean, gmax,
                              (OACIReason.OACI_OK,))


# ------------------------------------------------------------------ main entry
def _dedup(seq: list) -> list:
    return list(dict.fromkeys(seq))


def build_router_features(
    diagnostics: Mapping[str, Any],
    *,
    config: "RouterFeatureConfig | None" = None,
    acar_harm_gains: "Sequence[float] | None" = None,
) -> RouterFeatureBundle:
    cfg = config if config is not None else RouterFeatureConfig()
    d = diagnostics

    codes: list = []                 # accumulates ALL reason codes emitted while building features
    missing_keys: list = []
    imputed_keys: list = []

    def req(key: str) -> float:
        return as_finite_float(d, key, required=True, default=0.0,
                               missing_codes=codes, nonfinite_codes=codes,
                               missing_keys=missing_keys, imputed_keys=imputed_keys)

    def opt(key: str, default: float = 0.0) -> float:
        return as_finite_float(d, key, required=False, default=default,
                               missing_codes=codes, nonfinite_codes=codes,
                               missing_keys=missing_keys, imputed_keys=imputed_keys)

    # --- required diagnostics (missing/non-finite -> blocking diagnostic codes) ---
    n_target = req("n_target")
    ess = req("ess")
    delta_density_nll = req("delta_density_nll")
    transform_norm = req("transform_norm")
    condition_number = req("condition_number")
    prior_shift = req("prior_shift")
    pred_disagreement = req("pred_disagreement")
    ood_score = req("ood_score")

    # --- optional: cmi_residual (leakage residual; unavailable at deployment w/o true y) ---
    _cmi_present = "cmi_residual" in d
    cmi_residual = opt("cmi_residual", 0.0)
    cmi_residual_available = _cmi_present and ("cmi_residual" not in imputed_keys)
    if not _cmi_present:
        codes.append(OACIReason.OACI_LEAKAGE_RESIDUAL_UNAVAILABLE)

    # --- optional: prior-decoupled density fields ---
    prior_decoupling_available = ("density_nll_source_prior" in d) and ("density_nll_target_prior" in d)
    if prior_decoupling_available:
        density_nll_source_prior = opt("density_nll_source_prior", 0.0)
        density_nll_target_prior = opt("density_nll_target_prior", 0.0)
        support_gap = density_nll_source_prior - density_nll_target_prior
    else:
        density_nll_source_prior = 0.0
        density_nll_target_prior = ood_score          # fall back to ood_score (0.0 if it too was absent)
        support_gap = 0.0
        # Both final values are defaulted/derived here — even a PRESENT field is discarded when we
        # cannot decouple — so BOTH are 'imputed'; only the genuinely-absent ones are 'missing'.
        for k in ("density_nll_source_prior", "density_nll_target_prior"):
            imputed_keys.append(k)
            if k not in d:
                missing_keys.append(k)
        codes.append(OACIReason.OACI_PRIOR_DECOUPLING_UNAVAILABLE)

    # --- optional: min class responsibility (feature only, no trigger in v1) ---
    min_class_responsibility = opt("min_class_responsibility", 0.0)

    # --- TOS / support / TTA-stability triggers ---
    if n_target < cfg.min_target_n:
        codes.append(OACIReason.OACI_TOS_TOO_FEW_TARGET)
    if ess < cfg.min_ess:
        codes.append(OACIReason.OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE)
    if condition_number > cfg.max_condition_number:
        codes.append(OACIReason.OACI_TTA_UNSTABLE_TRANSFORM)
    if transform_norm > cfg.max_transform_norm:
        codes.append(OACIReason.OACI_TTA_UNSTABLE_TRANSFORM)
    if pred_disagreement > cfg.max_pred_disagreement:
        codes.append(OACIReason.OACI_TTA_HIGH_PRED_DISAGREEMENT)
    if delta_density_nll < cfg.min_delta_density_nll_for_tta:
        codes.append(OACIReason.OACI_TTA_NEGATIVE_EVIDENCE)
    if cfg.max_ood_score is not None and ood_score > cfg.max_ood_score:
        codes.append(OACIReason.OACI_TOS_DENSITY_OOD)
    if (cfg.max_density_nll_target_prior is not None
            and density_nll_target_prior > cfg.max_density_nll_target_prior):
        codes.append(OACIReason.OACI_TOS_SUPPORT_MISMATCH)
    if cfg.max_support_gap_abs is not None and abs(support_gap) > cfg.max_support_gap_abs:
        codes.append(OACIReason.OACI_TOS_SUPPORT_MISMATCH)
    if (cfg.leakage_residual_budget is not None and cmi_residual_available
            and cmi_residual > cfg.leakage_residual_budget):
        codes.append(OACIReason.OACI_LEAKAGE_RESIDUAL_HIGH)

    # --- prior-shift-only (audit) vs prior-decoupling-failed (blocking) ---
    _density_ood = OACIReason.OACI_TOS_DENSITY_OOD in codes
    _support_mismatch = OACIReason.OACI_TOS_SUPPORT_MISMATCH in codes
    prior_shift_only = bool(
        prior_shift >= cfg.prior_shift_info_threshold
        and prior_decoupling_available
        and support_gap >= cfg.support_gap_info_threshold
        and not _density_ood
        and not _support_mismatch
    )
    if prior_shift_only:
        codes.append(OACIReason.OACI_PRIOR_SHIFT_ONLY_INFO)
    if prior_shift >= cfg.prior_shift_info_threshold and not prior_decoupling_available:
        codes.append(OACIReason.OACI_PRIOR_DECOUPLING_FAILED)

    # tos_pass is decided by the feature-building codes ONLY (ACAR calibration is separate)
    tos_pass = not has_blocking_reason(codes)

    # --- ACAR harm-calibration state (separate axis; its codes join reason_codes but not tos_pass) ---
    acar = assess_acar_harm_calibration(acar_harm_gains,
                                        min_examples=cfg.min_calibration_examples,
                                        harm_margin=cfg.harm_margin)

    all_codes = list(codes) + list(acar.reason_codes)
    from h2cmi.router.reasons import normalize_reasons  # local import to keep top imports minimal
    reason_codes = normalize_reasons(all_codes)

    # --- feature vectors (guaranteed finite) ---
    values = {
        "n_target": n_target,
        "ess": ess,
        "density_nll_source_prior": density_nll_source_prior,
        "density_nll_target_prior": density_nll_target_prior,
        "support_gap": support_gap,
        "min_class_responsibility": min_class_responsibility,
        "delta_density_nll": delta_density_nll,
        "transform_norm": transform_norm,
        "condition_number": condition_number,
        "prior_shift": prior_shift,
        "pred_disagreement": pred_disagreement,
        "ood_score": ood_score,
        "cmi_residual": cmi_residual,
    }
    vector = np.array([values[k] for k in ROUTER_FEATURE_KEYS], dtype=np.float64)
    legacy_vector = np.array([values[k] for k in LEGACY_GATE_FEATURE_KEYS], dtype=np.float64)
    if not (np.all(np.isfinite(vector)) and np.all(np.isfinite(legacy_vector))):
        # defence-in-depth: should be impossible given as_finite_float; reason-code and clamp
        reason_codes = normalize_reasons(list(reason_codes) + [OACIReason.OACI_INTERNAL_ERROR])
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        legacy_vector = np.nan_to_num(legacy_vector, nan=0.0, posinf=0.0, neginf=0.0)
        tos_pass = False

    out_diag: dict[str, Any] = {
        "missing_diagnostics": _dedup(missing_keys),
        "imputed_diagnostics": _dedup(imputed_keys),
        "prior_decoupling_available": bool(prior_decoupling_available),
        "prior_shift_only": bool(prior_shift_only),
        "cmi_residual_available": bool(cmi_residual_available),
        "acar_harm_calibration_state": acar.state.value,
        "acar_harm_n": acar.n,
        "acar_harm_n_harm": acar.n_harm,
        "acar_harm_n_nonharm": acar.n_nonharm,
        "tos_pass": bool(tos_pass),
        "reason_codes": [r.value for r in reason_codes],
        "support_gap": float(support_gap),
    }

    return RouterFeatureBundle(
        vector=vector,
        feature_names=ROUTER_FEATURE_KEYS,
        legacy_gate_vector=legacy_vector,
        legacy_gate_feature_names=LEGACY_GATE_FEATURE_KEYS,
        diagnostics=out_diag,
        reason_codes=reason_codes,
        tos_pass=bool(tos_pass),
        prior_shift_only=bool(prior_shift_only),
        cmi_residual_available=bool(cmi_residual_available),
        acar_harm_calibration=acar,
    )


if __name__ == "__main__":
    R = OACIReason

    def _base() -> dict:
        # a clean, TOS-passing diagnostics dict
        return dict(
            n_target=120.0, ess=42.0, delta_density_nll=0.10, transform_norm=2.5,
            condition_number=1.6, prior_shift=0.05, pred_disagreement=0.10, ood_score=1.2,
            cmi_residual=0.0, density_nll_source_prior=3.0, density_nll_target_prior=2.9,
            min_class_responsibility=0.25,
        )

    # 1. happy path
    b = build_router_features(_base())
    assert b.tos_pass is True, b.diagnostics["reason_codes"]
    assert np.all(np.isfinite(b.vector)) and np.all(np.isfinite(b.legacy_gate_vector))
    assert b.vector.shape == (len(ROUTER_FEATURE_KEYS),)
    assert b.legacy_gate_vector.shape == (len(LEGACY_GATE_FEATURE_KEYS),)
    from h2cmi.router.reasons import has_blocking_reason as _hbr
    assert not _hbr(b.reason_codes), b.diagnostics["reason_codes"]   # no blocking code on happy path

    # 2. missing cmi_residual -> LEAKAGE_RESIDUAL_UNAVAILABLE, still tos_pass
    dd = _base(); dd.pop("cmi_residual")
    b = build_router_features(dd)
    assert R.OACI_LEAKAGE_RESIDUAL_UNAVAILABLE in b.reason_codes
    assert b.cmi_residual_available is False
    assert b.tos_pass is True

    # 3. missing required ess -> DIAGNOSTIC_MISSING, tos_pass False
    dd = _base(); dd.pop("ess")
    b = build_router_features(dd)
    assert R.OACI_DIAGNOSTIC_MISSING in b.reason_codes
    assert "ess" in b.diagnostics["missing_diagnostics"]
    assert b.tos_pass is False

    # 4. low ESS -> LOW_EFFECTIVE_SAMPLE_SIZE, tos_pass False
    dd = _base(); dd["ess"] = 3.0
    b = build_router_features(dd)
    assert R.OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE in b.reason_codes
    assert b.tos_pass is False

    # 5. prior-shift-only -> INFO, tos_pass True
    dd = _base(); dd["prior_shift"] = 0.6
    dd["density_nll_source_prior"] = 5.0; dd["density_nll_target_prior"] = 2.0  # support_gap +3
    b = build_router_features(dd)
    assert b.prior_shift_only is True
    assert R.OACI_PRIOR_SHIFT_ONLY_INFO in b.reason_codes
    assert b.tos_pass is True

    # 6. prior high but decoupling unavailable -> DECOUPLING_FAILED, tos_pass False
    dd = _base(); dd["prior_shift"] = 0.6
    dd.pop("density_nll_source_prior"); dd.pop("density_nll_target_prior")
    b = build_router_features(dd)
    assert R.OACI_PRIOR_DECOUPLING_FAILED in b.reason_codes
    assert R.OACI_PRIOR_DECOUPLING_UNAVAILABLE in b.reason_codes
    assert b.tos_pass is False

    # 6b. PARTIAL prior fields: a SUPPLIED field that gets discarded must be recorded as imputed (fail-loud)
    dd = _base(); dd.pop("density_nll_target_prior")     # source (3.0) present, target absent
    b = build_router_features(dd)
    assert b.diagnostics["prior_decoupling_available"] is False
    assert "density_nll_source_prior" in b.diagnostics["imputed_diagnostics"], "discarded supplied value not recorded"
    assert "density_nll_target_prior" in b.diagnostics["missing_diagnostics"]
    assert "density_nll_source_prior" not in b.diagnostics["missing_diagnostics"]  # it WAS supplied

    # 7. ACAR degenerate (all-zero gains)
    b = build_router_features(_base(), acar_harm_gains=[0.0, 0.0, 0.0, 0.0, 0.0])
    assert b.acar_harm_calibration.state is CalibrationState.DEGENERATE
    assert R.OACI_ACAR_HARM_CALIBRATION_DEGENERATE in b.reason_codes
    assert b.tos_pass is True  # ACAR degeneracy does NOT flip tos_pass

    # 8. ACAR available (both harm classes under margin)
    b = build_router_features(_base(), acar_harm_gains=[-0.10, -0.20, 0.05, 0.10, 0.0])
    assert b.acar_harm_calibration.state is CalibrationState.AVAILABLE
    assert b.acar_harm_calibration.n_harm == 2 and b.acar_harm_calibration.n_nonharm == 3

    # ACAR insufficient
    b = build_router_features(_base(), acar_harm_gains=[0.1])
    assert b.acar_harm_calibration.state is CalibrationState.UNAVAILABLE
    assert R.OACI_ACAR_INSUFFICIENT_CALIBRATION in b.reason_codes

    # non-finite required -> DIAGNOSTIC_NONFINITE, imputed recorded, tos_pass False
    dd = _base(); dd["transform_norm"] = float("nan")
    b = build_router_features(dd)
    assert R.OACI_DIAGNOSTIC_NONFINITE in b.reason_codes
    assert "transform_norm" in b.diagnostics["imputed_diagnostics"]
    assert np.all(np.isfinite(b.vector))
    assert b.tos_pass is False

    print("features self-test passed")
