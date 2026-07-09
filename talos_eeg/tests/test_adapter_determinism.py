from __future__ import annotations

from talos_eeg.red_team.adapter_determinism import validate_adapter_determinism
from talos_eeg.runners.run_talos00_preflight import (
    TrustRegionBounds,
    make_synthetic_preflight,
    run_preflight_scenario,
)


def test_adapter_determinism_same_seed_same_hashes():
    data = make_synthetic_preflight(seed=3)
    bounds = TrustRegionBounds()
    left = run_preflight_scenario(data=data, scenario="true_y_final_only", seed=3, bounds=bounds)
    right = run_preflight_scenario(data=data, scenario="true_y_final_only", seed=3, bounds=bounds)
    result = validate_adapter_determinism(left, right)
    assert result.passed
