from __future__ import annotations

import json
from pathlib import Path

import pytest

from oaci.conditioned_ceiling_coverage import c78r_common as common
from oaci.conditioned_ceiling_coverage import c78r_instrument as instrument
from oaci.conditioned_ceiling_coverage import c78r_seed3_src_canary as c78r


def test_c78r_scope_is_exact_80_SRC_units():
    rows = c78r.unit_manifest()
    assert len(rows) == len({row["unit_id"] for row in rows}) == 80
    assert {row["dataset"] for row in rows} == {"BNCI2014_001"}
    assert {row["target"] for row in rows} == {4}
    assert {row["seed"] for row in rows} == {3}
    assert {row["level"] for row in rows} == {0, 1}
    assert {row["regime"] for row in rows} == {"SRC"}
    assert {row["smooth_temperature"] for row in rows} == {0.1}
    for level in (0, 1):
        assert [row["epoch"] for row in rows if row["level"] == level] == list(range(4, 200, 5))


def test_c78r_authorization_is_exact_only(monkeypatch):
    token = c78r.AUTHORIZATION_TOKEN
    assert c78r.authorization_matches(token)
    assert not c78r.authorization_matches(None)
    assert not c78r.authorization_matches(f" {token}")
    assert not c78r.authorization_matches(token + "\n")
    assert not c78r.authorization_matches("Authorize C78R execution")
    monkeypatch.setenv("C78R_AUTHORIZATION_TOKEN", token)
    assert not c78r.authorization_matches(None)


def test_c78r_training_guard_precedes_data_and_cuda_imports():
    from oaci.conditioned_ceiling_coverage import c78r_train

    source = Path(c78r_train.__file__).read_text()
    function = source[source.index("def train_src_canary"):]
    assert function.index("common.require_authorization") < function.index("from oaci.data.eeg.bnci")
    assert function.index("common.require_authorization") < function.index("import torch")


def test_c78r_worker_does_not_train_ERM_or_OACI():
    from oaci.conditioned_ceiling_coverage import c78r_train

    function = Path(c78r_train.__file__).read_text().split("def train_src_canary", 1)[1]
    assert "run_stage1_once" not in function
    assert "make_objective(\"OACI\"" not in function
    assert "SRCObjective" in function
    assert "train_stage2" in function


def test_c78r_historical_SRC_files_are_byte_exact():
    rows = c78r._historical_hash_rows()
    assert len(rows) == 4
    assert all(row["byte_exact"] == 1 for row in rows)


def test_c78r_target_unlabeled_schema_has_no_target_labels():
    assert not instrument.TARGET_INPUT_FIELDS & instrument.FORBIDDEN_TARGET_FIELDS
    assert not instrument.TARGET_OUTPUT_FIELDS & instrument.FORBIDDEN_TARGET_FIELDS


def test_c78r_protocol_if_present_has_full_hash_and_boundaries():
    if not c78r.PROTOCOL_PATH.exists():
        pytest.skip("C78R protocol not emitted yet")
    protocol, digest, token = common.load_protocol()
    assert len(digest) == 64
    assert token == c78r.AUTHORIZATION_TOKEN
    assert protocol["scope"]["retained_units"] == 80
    assert protocol["execution_boundary"]["ERM_retraining"] is False
    assert protocol["execution_boundary"]["OACI_retraining"] is False
    assert protocol["execution_boundary"]["seed4"] is False
    assert protocol["execution_boundary"]["full_seed3_expansion"] is False


def test_c78r_execution_lock_if_present_replays_protocol_and_implementation():
    if not common.LOCK_PATH.exists():
        pytest.skip("C78R execution lock not emitted yet")
    lock = common.load_execution_lock()
    assert lock["scope"]["retained_units"] == 80
    assert lock["authorization"]["received"] is True


def test_c78r_field_if_present_is_exact_and_isolated():
    if not common.LOCK_PATH.exists():
        pytest.skip("C78R execution lock absent")
    lock = common.load_execution_lock()
    if not common.field_frozen_path(lock).exists():
        pytest.skip("C78R field not trained")
    field = common.require_field_frozen(lock)
    assert field["unit_count"] == field["SRC_count"] == 80
    assert field["ERM_retrained_count"] == 0
    assert field["OACI_weight_access_count"] == 0
    assert field["execution"]["target_data_rows_loaded_during_training"] == 0
    assert field["execution"]["target_label_reads_during_training"] == 0
    assert field["execution"]["seed4_access"] == 0
    assert field["execution"]["BNCI2014_004_access"] == 0


def test_c78r_instrumentation_if_present_has_exact_rows_and_identity():
    if not common.LOCK_PATH.exists():
        pytest.skip("C78R execution lock absent")
    lock = common.load_execution_lock()
    path = common.instrumentation_gate_path(lock)
    if not path.exists():
        pytest.skip("C78R instrumentation absent")
    gate = common.verify_manifest(path)
    assert gate["unit_count"] == gate["unique_unit_count"] == 80
    assert gate["source_rows"] == 80 * 8 * 576
    assert gate["target_unlabeled_rows"] == 80 * 576
    assert gate["identity"]["failed_units"] == 0
    assert gate["physical_isolation"]["target_unlabeled_contains_labels"] is False


def test_c78r_final_report_if_present_preserves_claim_boundary():
    path = c78r.REPORT_DIR / "C78R_SEED3_SRC_CANARY.json"
    if not path.exists():
        pytest.skip("C78R final report absent")
    result = json.loads(path.read_text())
    assert result["final_gate"] == "SRC_CANARY_EXECUTED_AND_VALIDATED_FULL_SEED3_READY_BUT_NOT_AUTHORIZED"
    assert result["scope"]["SRC_units"] == 80
    assert result["claims"]["multiregime_scientific_replication"] is False
    assert result["claims"]["full_seed3_expansion_authorized"] is False
    assert result["claims"]["seed4_access"] is False
