from __future__ import annotations

from tta_mech_eeg.normalization.condition_registry import ALLOWED_CONDITIONS, condition_registry_payload
from tta_mech_eeg.red_team.bn_condition_freeze import BnConditionFreezeFailure, validate_bn_condition_freeze


def test_bn_condition_registry_freezes_expected_conditions():
    payload = condition_registry_payload()
    result = validate_bn_condition_freeze(payload)
    assert result.passed
    assert tuple(payload["allowed_conditions"]) == ALLOWED_CONDITIONS
    assert payload["runtime_addition_allowed"] is False
    assert payload["target_labels_allowed"] is False
    assert payload["deployment_selection_allowed"] is False


def test_bn_condition_registry_rejects_added_condition():
    payload = condition_registry_payload()
    payload["allowed_conditions"] = payload["allowed_conditions"] + ["NEW_TTA_METHOD"]
    try:
        validate_bn_condition_freeze(payload)
    except BnConditionFreezeFailure:
        return
    raise AssertionError("condition universe freeze must reject added conditions")
