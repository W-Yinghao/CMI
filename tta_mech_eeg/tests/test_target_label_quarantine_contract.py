from __future__ import annotations

import numpy as np

from tta_mech_eeg.red_team.replay_determinism import validate_replay_determinism
from tta_mech_eeg.red_team.target_label_quarantine import validate_target_label_contract
from tta_mech_eeg.runners.run_00a_preflight import adapt_or_replay


def test_target_label_quarantine_contract_has_no_target_label_path():
    result = validate_target_label_contract(adapt_or_replay)
    assert result.passed


def test_toy_replay_is_deterministic_without_target_labels():
    source_state = {"n_features": 2, "n_classes": 2, "source_prior": [0.5, 0.5]}
    target_x = np.arange(8, dtype=np.float64).reshape(4, 2)
    left = adapt_or_replay(source_state, target_x, "ERM_NO_ADAPT")
    right = adapt_or_replay(source_state, target_x, "ERM_NO_ADAPT")
    result = validate_replay_determinism(left, right)
    assert result.passed
    assert left["target_metrics_computed"] is False
