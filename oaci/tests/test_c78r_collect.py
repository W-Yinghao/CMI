from __future__ import annotations

import csv
import json

import pytest

from oaci.conditioned_ceiling_coverage import c78r_collect as collect
from oaci.conditioned_ceiling_coverage import c78r_seed3_src_canary as c78r


def _rows(name: str):
    with open(c78r.TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def test_C78R_state_if_present_has_exact_scope_and_claims():
    if not collect.STATE_PATH.exists():
        pytest.skip("C78R collection has not run")
    state = json.loads(collect.STATE_PATH.read_text())
    assert state["final_gate_candidate"] == "SRC_CANARY_EXECUTED_AND_VALIDATED_FULL_SEED3_READY_BUT_NOT_AUTHORIZED"
    assert state["scope"] == {
        "target": 4, "seed": 3, "regime": "SRC", "levels": [0, 1],
        "SRC_units": 80, "ERM_retrained": 0, "OACI_retrained": 0,
    }
    assert state["claims"]["multiregime_scientific_replication"] is False
    assert state["claims"]["full_seed3_expansion_authorized"] is False


def test_C78R_hash_and_cadence_tables_if_present_are_complete():
    if not (c78r.TABLE_DIR / "SRC_checkpoint_manifest.csv").exists():
        pytest.skip("C78R collection has not run")
    checkpoints = _rows("SRC_checkpoint_manifest.csv")
    assert len(checkpoints) == 80
    assert all(row["all_hashes_passed"] == "1" for row in checkpoints)
    cadence = _rows("SRC_checkpoint_cadence_audit.csv")
    assert len(cadence) == 2
    assert all(row["actual_checkpoints"] == "40" and row["passed"] == "1" for row in cadence)


def test_C78R_resource_plan_is_phase_based_and_denies_expansion():
    if not (c78r.TABLE_DIR / "updated_full_seed3_compute_plan.csv").exists():
        pytest.skip("C78R collection has not run")
    compute = _rows("updated_full_seed3_compute_plan.csv")
    total = next(row for row in compute if row["phase"] == "TOTAL_48_PHASE_SCHEDULE")
    assert total["remaining_phases"] == "48"
    assert "phase-level" in total["measurement"]
    gate = _rows("full_seed3_expansion_gate.csv")[0]
    assert gate["remaining_units"] == "1296"
    assert gate["full_seed3_authorized"] == "0"


def test_C78R_final_report_if_present_requires_red_team_and_no_science_claim():
    result_path = c78r.REPORT_DIR / "C78R_SEED3_SRC_CANARY.json"
    if not result_path.exists():
        pytest.skip("C78R final report absent")
    result = json.loads(result_path.read_text())
    red_team = (c78r.REPORT_DIR / "C78R_RED_TEAM_VERIFICATION.md").read_text()
    assert "Final status: `PASS`" in red_team
    assert result["red_team"]["blocking_passed"] == result["red_team"]["blocking_total"]
    assert result["claims"]["multiregime_scientific_replication"] is False
    assert result["claims"]["full_seed3_expansion_authorized"] is False
