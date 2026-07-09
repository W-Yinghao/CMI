from __future__ import annotations

from tta_mech_eeg.normalization.bn_schema import bn_audit_schema_payload
from tta_mech_eeg.normalization.condition_registry import condition_registry_payload
from tta_mech_eeg.red_team.no_weight_update_guard import NoWeightUpdateFailure, validate_no_weight_update_guard


def _guard_payload():
    return {
        **condition_registry_payload(),
        "mutation_rules": bn_audit_schema_payload()["mutation_rules"],
    }


def test_no_weight_update_guard_accepts_frozen_registry():
    result = validate_no_weight_update_guard(_guard_payload())
    assert result.passed


def test_no_weight_update_guard_rejects_weight_mutation():
    payload = _guard_payload()
    payload["entries"][0]["mutates_weights_allowed"] = True
    try:
        validate_no_weight_update_guard(payload)
    except NoWeightUpdateFailure:
        return
    raise AssertionError("weight mutation must be rejected")


def test_no_weight_update_guard_rejects_optimizer_step_term():
    payload = _guard_payload()
    payload["active_code_path"] = "optimizer.step()"
    try:
        validate_no_weight_update_guard(payload)
    except NoWeightUpdateFailure:
        return
    raise AssertionError("optimizer.step active path must be rejected")
