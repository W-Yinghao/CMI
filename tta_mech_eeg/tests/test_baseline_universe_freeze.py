from __future__ import annotations

from tta_mech_eeg.baselines.registry import ALLOWED_BASELINES, registry_payload
from tta_mech_eeg.red_team.baseline_universe_freeze import (
    BaselineUniverseFailure,
    validate_baseline_universe,
)


def test_baseline_universe_freeze_accepts_exact_registry():
    payload = registry_payload()
    result = validate_baseline_universe(payload)
    assert result.passed
    assert tuple(payload["allowed_baselines"]) == ALLOWED_BASELINES


def test_baseline_universe_freeze_rejects_added_baseline():
    payload = registry_payload()
    payload["allowed_baselines"] = payload["allowed_baselines"] + ["NEW_METHOD"]
    try:
        validate_baseline_universe(payload)
    except BaselineUniverseFailure:
        return
    raise AssertionError("baseline universe freeze must reject runtime additions")
