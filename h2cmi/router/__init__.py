"""Project B deployment router.

Contract + policy layers (no model / TTA / harness):
  - actions / reasons / features  (Step-2B): action + OACI reason + fail-loud feature contracts;
  - acar                          (Step-2C): ACAR conformal calibration-state contract;
  - router                        (Step-2D): RefusalFirstRouter policy with action-specific blockers.

router_harness integration (route_target over a real model) is a later step and is NOT here.
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
from h2cmi.router.router import (
    RouterConfig,
    ActionRiskPrediction,
    RouterDecision,
    RefusalFirstRouter,
)
from h2cmi.router.error_risk import (
    ErrorRiskConfig,
    ErrorRiskFeatureAudit,
    ErrorRiskFit,
    fit_error_risk_crossfit,
    predict_error_risk,
    make_identity_error_acar_state,
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
    # Router policy (Step-2D)
    "RouterConfig",
    "ActionRiskPrediction",
    "RouterDecision",
    "RefusalFirstRouter",
    # Identity error-risk layer (S2B)
    "ErrorRiskConfig",
    "ErrorRiskFeatureAudit",
    "ErrorRiskFit",
    "fit_error_risk_crossfit",
    "predict_error_risk",
    "make_identity_error_acar_state",
]
