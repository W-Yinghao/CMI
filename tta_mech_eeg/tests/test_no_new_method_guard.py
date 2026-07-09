from __future__ import annotations

from tta_mech_eeg.red_team.no_new_method_guard import (
    NoNewMethodFailure,
    validate_no_new_method,
)


def test_no_new_method_guard_accepts_active_audit_config():
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_00A",
        "new_method_claim": False,
        "active_baselines": ["ERM_NO_ADAPT", "TTA_CONTROL_REPLAY"],
    }
    result = validate_no_new_method(payload)
    assert result.passed


def test_no_new_method_guard_rejects_active_forbidden_term():
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_00A",
        "new_method_claim": False,
        "active_baselines": ["LOW_RANK_ADAPTER"],
    }
    try:
        validate_no_new_method(payload)
    except NoNewMethodFailure:
        return
    raise AssertionError("active forbidden method term must fail")
