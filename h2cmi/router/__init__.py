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

__all__ = [
    "RouterAction",
    "OACIReason",
    "CalibrationState",
    "CalibrationSummary",
    "RouterFeatureConfig",
    "RouterFeatureBundle",
    "build_router_features",
    "assess_acar_harm_calibration",
]
