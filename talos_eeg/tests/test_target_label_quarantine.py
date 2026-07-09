from __future__ import annotations

from talos_eeg.red_team.target_label_quarantine import validate_target_label_quarantine
from talos_eeg.runners.run_talos00_preflight import build_preflight_payload


def test_target_label_quarantine_preflight_passes():
    payload = build_preflight_payload(seed=4)
    result = validate_target_label_quarantine(payload["scenarios"])
    assert result.passed
    assert payload["real_eeg_readout_run"] is False
    assert payload["scientific_readout"] is False
    assert payload["source_free_deployment_claim"] is False
