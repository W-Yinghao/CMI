from __future__ import annotations

import numpy as np

from talos_eeg.data.source_state import (
    SOURCE_STATE_MODE_P0_REPLAY,
    build_source_state,
    validate_source_state_schema,
)


def test_source_state_schema_marks_p0_replay_not_source_free():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(30, 4))
    y = np.tile([0, 1], 15)
    state = build_source_state(z, y)
    result = validate_source_state_schema(state)
    assert result.passed
    assert result.source_state_mode == SOURCE_STATE_MODE_P0_REPLAY
    assert result.source_free_deployment_claim is False
    assert state.readout_weight.shape == (4, 2)
    assert state.readout_bias.shape == (2,)
