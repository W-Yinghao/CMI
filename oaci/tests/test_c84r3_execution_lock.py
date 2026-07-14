from __future__ import annotations

import csv
import json

import pytest

from oaci.multidataset import c84r3_canary_runtime_repair as runtime


def _lock():
    return json.loads(runtime.EXECUTION_LOCK_PATH.read_text())


def test_execution_lock_v3_self_and_all_bound_objects_replay():
    digest = runtime.verify_lock_self(runtime.EXECUTION_LOCK_PATH, runtime.EXECUTION_LOCK_SHA_PATH)
    assert digest == "c198607fb9e46ea2353ffa57d6b71bfa966c36e8ece53fdc40292681bba8bd1a"
    lock = _lock()
    assert lock["status"] == runtime.LOCK_READY_STATUS
    replay = runtime.verify_bound_object_registry(lock)
    assert len(replay) == lock["runtime_bound_object_count"] == 72
    assert all(row["replay_pass"] for row in replay)
    assert len(runtime.verify_protocol_sidecars(lock)) == 7


def test_execution_lock_v3_binds_scope_identity_and_split_tolerances():
    lock = _lock()
    assert lock["candidate_identity"]["canary_unit_count"] == 243
    assert lock["candidate_identity"]["canary_unit_ids_sha256"] == (
        "4ada05be758975e7c28429819d804b4064a1bdcfd99fe7a4752a3bdbded6d396"
    )
    assert runtime.verify_candidate_identity(lock)["canary_unit_count"] == 243
    assert runtime.verify_montage_binding(lock)["channel_count"] == 20
    assert lock["instrumentation"]["linear_z_classifier_logits_abs_tolerance"] == 1e-5
    assert lock["instrumentation"]["softmax_repeat_logits_repeat_z_abs_tolerance"] == 1e-6
    assert lock["scope"]["retrain_all_units"] is True
    assert lock["runtime"]["failed_root_reusable"] is False


def test_execution_lock_v3_preserves_failed_attempt_and_requires_fresh_authorization(tmp_path):
    lock = _lock()
    historical = lock["historical_lock_supersession"]
    assert historical["authorization_consumed_by_job"] == 895366
    assert historical["operative_for_execution"] is False
    assert lock["authorization"]["historical_authorization_reusable"] is False
    assert lock["authorization"]["failed_authorization_reused"] is False
    assert not runtime.AUTHORIZATION_RECORD_PATH.exists()
    with pytest.raises(runtime.C84R3RuntimeError, match="fresh direct C84C replacement authorization"):
        runtime.verify_authorization_record(lock, "lock-sha", "lock-commit", tmp_path / "missing.json")


def test_no_c84f_or_c84s_execution_lock_exists():
    names = {path.name for path in runtime.REPORT_DIR.glob("C84*EXECUTION_LOCK*.json")}
    assert "C84C_EXECUTION_LOCK_V3.json" in names
    assert not any(name.startswith(("C84F_", "C84S_")) for name in names)


def test_c84r3_readiness_is_reauthorization_only_and_red_team_complete():
    readiness = json.loads((runtime.REPORT_DIR / "C84R3_PROTOCOL_READINESS.json").read_text())
    assert readiness["gate"] == (
        "C84C_FLOAT32_REPLAY_REPAIRED_AND_RELOCKED_READY_FOR_FRESH_PI_AUTHORIZATION"
    )
    assert readiness["failed_authorization_consumed"] is True
    assert readiness["fresh_authorization_record_present"] is False
    assert readiness["replacement_real_data_access"] == 0
    assert readiness["runtime_bound_objects"] == 72
    assert readiness["final_red_team"] == "37/37 PASS"
    with (runtime.REPORT_DIR / "c84r3_tables/final_report_red_team.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 37
    assert all(row["status"] == "PASS" for row in rows)
