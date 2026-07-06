"""Project B deployment router (Step-2B: reason / action / feature contract only).

RefusalFirstRouter is intentionally NOT exported yet — router.py / acar.py / router_harness.py
are later steps. This package currently locks the fail-loud feature & reason-code contracts.
"""
from __future__ import annotations

from h2cmi.router.actions import RouterAction
from h2cmi.router.reasons import OACIReason
from h2cmi.router.features import (
    CalibrationState,
    CalibrationSummary,
    RouterFeatureConfig,
    RouterFeatureBundle,
    build_router_features,
    assess_acar_harm_calibration,
)
from h2cmi.router.acar import (
    ACARConfig,
    ACARRiskType,
    ACARRiskCalibration,
    ACARActionState,
    ACARState,
    conformal_quantile,
    fit_risk_calibration,
    fit_acar_state,
)

__all__ = [
    "RouterAction",
    "OACIReason",
    "CalibrationState",
    "CalibrationSummary",
    "RouterFeatureConfig",
    "RouterFeatureBundle",
    "build_router_features",
    "assess_acar_harm_calibration",
    # ACAR (Step-2C)
    "ACARConfig",
    "ACARRiskType",
    "ACARRiskCalibration",
    "ACARActionState",
    "ACARState",
    "conformal_quantile",
    "fit_risk_calibration",
    "fit_acar_state",
]
