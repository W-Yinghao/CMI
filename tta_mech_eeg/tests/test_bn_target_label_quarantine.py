from __future__ import annotations

import numpy as np

from tta_mech_eeg.red_team.bn_target_label_quarantine import (
    BnTargetLabelQuarantineFailure,
    validate_bn_target_label_quarantine,
)
from tta_mech_eeg.runners.run_02b0_preflight import run_condition


def test_bn_target_label_quarantine_accepts_condition_api():
    result = validate_bn_target_label_quarantine(run_condition)
    assert result.passed
    out = run_condition(
        {"kind": "contract_only"},
        {"source": "contract_only"},
        np.zeros((2, 3)),
        {"condition": "ERM_FROZEN_EVAL"},
    )
    assert out["real_forward_run"] is False
    assert out["bn_refresh_run"] is False
    assert out["target_metrics_computed"] is False


def test_bn_target_label_quarantine_rejects_target_y_parameter():
    def bad_condition(model_or_state, source_state, target_x, condition_config, target_y):
        return None

    try:
        validate_bn_target_label_quarantine(bad_condition)
    except BnTargetLabelQuarantineFailure:
        return
    raise AssertionError("BN target-label quarantine must reject target_y")
