from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from oaci.conditioned_ceiling_coverage import c78_authorized_collect as collect
from oaci.conditioned_ceiling_coverage import c78_seed3_instrumented_pilot as c78


def _rows(name: str) -> list[dict[str, str]]:
    with open(c78.TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def test_balanced_accuracy_and_ece_are_well_defined():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    prediction = y.copy()
    probability = np.full((8, 4), 0.05)
    probability[np.arange(8), y] = 0.85
    assert collect._balanced_accuracy(y, prediction) == 1.0
    assert 0.0 <= collect._ece(y, prediction, probability) <= 1.0


def test_authorized_collection_if_present_has_no_recommendation_and_exact_counts():
    if not collect.STATE_PATH.exists():
        pytest.skip("authorized collection has not run yet")
    state = json.loads(collect.STATE_PATH.read_text())
    assert state["field"] == {
        "units": 82, "ERM_anchors": 2, "OACI_checkpoints": 80,
        "SRC": 0, "levels": [0, 1],
    }
    assert state["final_gate_candidate"] == "PILOT_VALID_SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD"
    assert state["claims"]["checkpoint_recommendation"] is False
    geometry = _rows("effective_multiplicity_top_gap_smoke.csv")
    assert len(geometry) == 2
    assert all(row["candidate_count"] == "41" for row in geometry)
    assert all(row["best_checkpoint_id_emitted"] == "0" for row in geometry)
    assert not any("checkpoint_id" in row for row in geometry)


def test_authorized_hash_replay_if_present_is_82_of_82():
    path = c78.TABLE_DIR / "checkpoint_hash_replay.csv"
    if not path.exists():
        pytest.skip("authorized collection has not run yet")
    rows = _rows("checkpoint_hash_replay.csv")
    assert len(rows) == 82
    assert all(row["checkpoint_hash_match"] == "1" for row in rows)
    assert all(row["optimizer_hash_match"] == "1" for row in rows)


def test_authorized_target_isolation_if_present_is_runtime_not_plan_only():
    path = c78.TABLE_DIR / "target_isolation_runtime_audit.csv"
    if not path.exists():
        pytest.skip("authorized collection has not run yet")
    row = _rows("target_isolation_runtime_audit.csv")[0]
    assert row["passed"] == "1"
    assert row["training_process_target_rows"] == "0"
    assert row["training_process_source_audit_rows"] == "0"
    assert row["field_frozen_before_target_load"] == "1"
