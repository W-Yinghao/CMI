from __future__ import annotations

import numpy as np

from tta_mech_eeg.runners.run_01_real_replay import (
    EXPECTED_ARTIFACT_INVENTORY_HASH,
    EXPECTED_BASELINE_REGISTRY_HASH,
    _balanced_accuracy,
    _macro_f1,
)


def test_tta_mech01_expected_hashes_are_frozen_from_00a():
    assert EXPECTED_BASELINE_REGISTRY_HASH == "0d8a00cdb2d0bf810a20c58056323c21c1fde8807fb12e65ebc9ff8334748da7"
    assert EXPECTED_ARTIFACT_INVENTORY_HASH == "6323097829f2d3277275f392ea33329a4197e7032969bb3ebf87d1ad2e090cb2"


def test_tta_mech01_metric_helpers_are_label_explicit():
    labels = (0, 1, 2)
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])
    assert _balanced_accuracy(y_true, y_pred, labels) == (1.0 + 1.0 + 0.5) / 3.0
    assert 0.0 <= _macro_f1(y_true, y_pred, labels) <= 1.0
