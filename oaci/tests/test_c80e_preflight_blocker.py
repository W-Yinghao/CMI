from __future__ import annotations

import json
from pathlib import Path

import pytest

from oaci.conditioned_ceiling_coverage import c80_label_budget_frontier as frontier


REPORT_DIR = frontier.REPORT_DIR
PREFLIGHT_PATH = REPORT_DIR / "C80E_AUTHORIZATION_AND_PREFLIGHT.json"
AUTHORIZATION_PATH = REPORT_DIR / "C80E_PI_AUTHORIZATION_RECORD.json"


def test_c80e_authorization_is_recorded_but_execution_is_blocked():
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    assert authorization["authorization_received"] is True
    assert authorization["scientific_execution_started"] is False
    assert authorization["resolved_binding"]["protocol_sha256"] == frontier.PROTOCOL_SHA_PATH.read_text().strip()
    assert preflight["final_gate"] == "C80E_AUTHORIZATION_PROTOCOL_LOCK_VIEW_OR_DEPENDENCE_BLOCKER"
    assert len(preflight["blocking_findings"]) == 3
    assert all(row["blocking"] for row in preflight["blocking_findings"])


def test_locked_protocol_has_no_required_final_taxonomy_decision_table():
    protocol_text = frontier.PROTOCOL_PATH.read_text()
    lock_text = frontier.LOCK_PATH.read_text()
    registry_text = (frontier.TABLE_DIR / "scientific_registry.csv").read_text()
    combined = protocol_text + lock_text + registry_text
    for case in ("C80-A_", "C80-B_", "C80-C_", "C80-D_", "C80-E_"):
        assert case not in combined
    assert "near-FULL" not in combined


def test_locked_authorization_guard_replays_schema_mismatch_without_data_access():
    assert AUTHORIZATION_PATH.exists()
    with pytest.raises(RuntimeError, match="blocked and not reusable"):
        frontier.assert_c80e_authorized()


def test_locked_real_route_has_no_adapter_and_fails_before_data_access():
    source = Path(frontier.__file__).read_text()
    assert "deliberately has no real-data loader" in source
    assert "C80E real-data adapter is intentionally unavailable in C80P" in source
    assert "/projects/" not in source
    assert "np.load" not in source


def test_preflight_safe_stop_precedes_repaired_scientific_result():
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    protected = preflight["protected_state"]
    assert protected["real_budget_statistics_computed"] == 0
    assert protected["evaluation_label_values_read_for_C80E"] == 0
    assert protected["same_label_oracle_accesses"] == 0
    assert protected["target4_primary_rows"] == 0

    repaired_authorization = json.loads(
        (REPORT_DIR / "C80E_REPAIRED_PI_AUTHORIZATION_RECORD.json").read_text()
    )
    result_freeze = json.loads(
        (REPORT_DIR / "C80E_PRIMARY_RESULT_FREEZE.json").read_text()
    )
    assert preflight["recorded_at_utc"] < repaired_authorization["recorded_at_utc"]
    assert repaired_authorization["recorded_at_utc"] < result_freeze["frozen_at_utc"]
    assert repaired_authorization["evaluation_outcomes_before_binding_refresh"] == 0
    assert repaired_authorization["real_budget_statistics_before_record"] == 0
    assert result_freeze["all_five_paths_unconditional"] is True
    assert result_freeze["target4_primary"] is False
    assert result_freeze["same_label_oracle_accessed"] is False
    assert (REPORT_DIR / result_freeze["result_path"].split("/")[-1]).exists()
