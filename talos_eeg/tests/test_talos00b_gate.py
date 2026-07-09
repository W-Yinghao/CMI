from __future__ import annotations

from talos_eeg.adapters.trust_region import TrustRegionBounds
from talos_eeg.runners.run_talos00b_real_replay import (
    TALOS00B_VARIANTS,
    _build_variant_freeze,
    _validate_variant_freeze,
)


def test_talos00b_variant_freeze_uses_pm_approved_universe():
    payload = _build_variant_freeze(seed=0, bounds=TrustRegionBounds())
    result = _validate_variant_freeze(payload)
    assert result["passed"]
    assert tuple(result["allowed_variants"]) == TALOS00B_VARIANTS
    assert "TALOS_LR" in payload["forbidden_variants"]
    assert payload["target_labels_allowed_for"] == ["final_metrics_only"]
