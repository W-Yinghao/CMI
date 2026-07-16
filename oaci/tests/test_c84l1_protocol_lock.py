from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

from oaci.multidataset import c84l1_protocols as protocol
from oaci.multidataset import c84r_v2_protocols as historical


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c84l1p_tables"


def _rows(name: str):
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_repair_protocol_precedes_implementation_and_declares_zero_access():
    payload = json.loads((REPORTS / "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.json").read_text())
    assert payload["timing"]["designed_before_level1_real_data_access"] is True
    assert payload["timing"]["level1_real_EEG_access_before_protocol"] == 0
    assert payload["timing"]["level1_label_reads_before_protocol"] == 0
    assert payload["timing"]["level1_training_forward_GPU_before_protocol"] == 0
    assert payload["levels"]["0"]["id"] == protocol.LEVEL0_ID
    assert payload["levels"]["1"]["id"] == protocol.LEVEL1_ID


def test_protocol_sidecars_replay_for_complete_additive_family():
    stems = (
        "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL",
        "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3",
        "C84_LEVEL1_CANARY_PROTOCOL_V1",
        "C84_FIELD_GENERATION_PROTOCOL_V5",
        "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3",
    )
    for stem in stems:
        path = REPORTS / f"{stem}.json"
        expected = (REPORTS / f"{stem}.sha256").read_text().split()[0]
        assert protocol.sha256_file(path) == expected


def test_exact_six_cell_registry_and_support_contract_are_locked():
    rows = _rows("level_intervention_registry.csv")
    observed = {
        (row["dataset"], row["panel"]): (int(row["deleted_source_subject"]), row["deleted_class"])
        for row in rows
    }
    assert observed == {key: (subject, "left_hand") for key, subject in protocol.DELETED_SUBJECTS.items()}
    assert all(row["deleted_subject_is_first_locked"] == "1" for row in rows)
    assert all(row["minimum_cell_support"] == "8" for row in rows)
    support = _rows("level_support_contract.csv")
    assert {row["required_value"] for row in support if row["condition"] == "post_deletion_observed_cells"} == {"23"}
    assert all(row["failure_action"] == "BLOCK" for row in support)


def test_only_historical_level1_ids_are_superseded():
    old = historical.candidate_units()
    old_level0 = {row["unit_id"] for row in old if row["level"] == 0}
    old_level1 = {row["unit_id"] for row in old if row["level"] == 1}
    operative = _rows("operative_complete_unit_registry_v2.csv")
    current_level0 = {row["unit_id"] for row in operative if row["level"] == "0"}
    current_level1 = {row["unit_id"] for row in operative if row["level"] == "1"}
    assert len(old_level0) == len(current_level0) == 972
    assert old_level0 == current_level0
    assert len(old_level1) == len(current_level1) == 972
    assert not old_level1 & current_level1
    assert len({row["unit_id"] for row in operative}) == 1944


def test_new_level1_ids_bind_intervention_and_registry_digest():
    rows = _rows("level1_candidate_id_registry.csv")
    registry_sha = protocol.sha256_file(TABLES / "level_intervention_registry.csv")
    assert len(rows) == 972
    assert all(row["level_intervention_id"] == protocol.LEVEL1_ID for row in rows)
    assert all(row["level_intervention_registry_sha256"] == registry_sha for row in rows)
    assert all(row["deleted_class"] == "left_hand" for row in rows)
    expected = hashlib.sha256(protocol.canonical_bytes(sorted(row["unit_id"] for row in rows))).hexdigest()
    assert (TABLES / "level1_candidate_id_digest.txt").read_text().split()[0] == expected


def test_level1_canary_scope_is_exact_and_science_is_forbidden():
    rows = _rows("level1_canary_scope.csv")
    assert len(rows) == 3
    assert sum(int(row["candidate_units"]) for row in rows) == 243
    assert sum(int(row["training_phases"]) for row in rows) == 9
    assert {row["target_subject"] for row in rows} == {"19", "24", "106"}
    assert all(row["target_scientific_metrics"] == "0" for row in rows)
    canary = json.loads((REPORTS / "C84_LEVEL1_CANARY_PROTOCOL_V1.json").read_text())
    assert canary["scope"]["total_units"] == 243
    assert canary["scope"]["training_phases"] == 9
    assert canary["authorization"]["fresh_direct_PI_authorization_required"] is True
    assert {"target_accuracy", "target_regret", "selector_scores", "Q1", "Q2"} <= set(
        canary["forbidden_outputs"]
    )


def test_C84L1P_historical_commit_only_created_canary_lock():
    historical = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "a0ec77b", "oaci/reports"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    historical_locks = {Path(path).name for path in historical if "EXECUTION_LOCK" in path}
    assert "C84L1C_EXECUTION_LOCK.json" in historical_locks
    assert not any(name.startswith(("C84F_", "C84S_")) for name in historical_locks)
    lock_names = {path.name for path in REPORTS.glob("C84*EXECUTION_LOCK*.json")}
    assert "C84L1C_EXECUTION_LOCK.json" in lock_names
    assert "C84F_EXECUTION_LOCK.json" in lock_names
    science_locks = {name for name in lock_names if name.startswith("C84S_")}
    assert science_locks == {
        "C84S_ANALYSIS_EXECUTION_LOCK.json",
        "C84S_ANALYSIS_EXECUTION_LOCK_V2.json",
    }
    science_lock = json.loads((REPORTS / "C84S_ANALYSIS_EXECUTION_LOCK_V2.json").read_text())
    assert science_lock["status"] == "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
    assert science_lock["authorization"]["record_present_at_lock"] is False
    assert not (REPORTS / "C84S_PI_AUTHORIZATION_RECORD.json").exists()
