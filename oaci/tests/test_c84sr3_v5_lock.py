from pathlib import Path

import pytest

from oaci.multidataset.c84sr3_common import (
    AUTHORIZATION_PATH, LOCK_PATH, LOCK_READY_STATUS, METHOD_CONTEXT_ROWS,
    PROTOCOL_SHA256, Q0_RECORDS,
)
from oaci.multidataset.c84sr3_runtime_guard import (
    static_process_isolation_audit, verify_bound_repository_objects,
    verify_lock_bound_readiness, verify_lock_self, verify_protocol_inputs,
)
from oaci.multidataset.c84s_common import read_json, sha256_file


def test_repair_protocol_and_consumed_v4_attempt_replay():
    observed = verify_protocol_inputs()
    assert observed["repair"] == PROTOCOL_SHA256
    assert observed["historical_V4_authorization"] == (
        "4419303f8282ab132d2f95a5b76993bfb73191b29e7257e8220cefdde408ff5a"
    )
    assert observed["historical_V4_consumption"] == (
        "6dfc058e67ea8fa1ea8ddc0c1d398a4b468c4213a42455b5f864ced800fb0866"
    )


def test_v5_authorization_was_absent_at_lock_and_is_now_preserved():
    lock = read_json(LOCK_PATH)
    assert lock["authorization"]["record_present_at_lock"] is False
    assert AUTHORIZATION_PATH.is_file()
    assert sha256_file(AUTHORIZATION_PATH) == (
        "3446e3562a8dd5db51c9f56a03765bf040f9678ee527ea13a4cf75e63dd575e1"
    )
    record = read_json(AUTHORIZATION_PATH)
    assert record["authorized_stage"] == "C84S"
    assert record["historical_V4_authorization_reused"] is False


def test_static_isolation_binds_v5_stage_a_and_seals_stage_b():
    rows = static_process_isolation_audit()
    assert all(row["pass"] == 1 for row in rows)
    assert any(row["check"] == "immutable_Stage_A_replay_only" for row in rows)
    assert any(row["check"] == "evaluation_descriptor_absent" for row in rows)
    source = Path("oaci/multidataset/c84sr3_execute.py").read_text(encoding="utf-8")
    assert "c84sr3_stage_a_replay" in source
    assert "c84sr1_stage_a_labels" not in source


def test_v5_lock_replays_when_readiness_has_published_it():
    if not LOCK_PATH.exists():
        pytest.skip("V5 lock is created only after the implementation commit")
    lock, _ = verify_lock_self()
    assert lock["status"] == LOCK_READY_STATUS
    assert lock["analysis_contract"]["Q0_records"] == Q0_RECORDS
    assert lock["analysis_contract"]["method_context_rows"] == METHOD_CONTEXT_ROWS
    assert lock["analysis_contract"]["Lee_B32_status"] == (
        "INPUT_UNAVAILABLE_NO_SELECTION_OR_RESULT_ROW"
    )
    verify_bound_repository_objects(lock)
    replay = verify_lock_bound_readiness(lock)
    assert replay["tables"] >= 10
