from pathlib import Path

from oaci.multidataset.c84s_common import read_json, sha256_file
from oaci.multidataset.c84sr1_common import (
    AUTHORIZATION_PATH, LOCK_PATH, LOCK_SHA_PATH,
)
from oaci.multidataset.c84sr1_runtime_guard import (
    LOCK_READY_STATUS, verify_lock_bound_readiness,
    verify_lock_self,
)


def test_v3_lock_self_identity_and_status():
    lock, digest = verify_lock_self()
    assert digest == "15d8a84b7870021a90b1f0f103a8a4d733523b249321b143a89091978c0aa9fc"
    assert LOCK_SHA_PATH.read_text(encoding="ascii").split()[0] == digest
    assert lock["status"] == LOCK_READY_STATUS
    assert lock["analysis_contract"]["Q0_records"] == 9_110_448
    assert lock["analysis_contract"]["method_context_rows"] == 18_608


def test_v3_lock_is_preserved_but_nonoperative_after_additive_sr2_repair():
    lock = read_json(LOCK_PATH)
    replay = verify_lock_bound_readiness(lock)
    assert replay["readiness_tables"] >= 10
    assert replay["synthetic_summary_sha256"] == "26e80934c75caae512d038e7939283ddef0b0d620c1c0686fe8cf55c1d5e8799"
    assert len(lock["runtime_bound_repository_objects"]) == 21
    current = {
        row["path"]: sha256_file(Path(row["path"]))
        for row in lock["runtime_bound_repository_objects"]
    }
    drift = [row["path"] for row in lock["runtime_bound_repository_objects"]
             if current[row["path"]] != row["sha256"]]
    assert drift == ["oaci/multidataset/c84sr1_context_enumerator.py"]


def test_v3_lock_preserves_historical_nonoperative_locks():
    lock = read_json(LOCK_PATH)
    assert lock["historical_locks"] == {
        "V1_sha256": "e17e4da14b60ac77ca0ec8bec80a2ca249cda014baf5460cfd64627294f2047b",
        "V2_sha256": "94c896f0f00c53441095da6225f9ac574eb4a9baa904821a5dab3f11ea76f75c",
        "V2_authorization_consumed": False,
        "operative": False,
    }


def test_v3_authorization_is_preserved_as_consumed_failed_attempt_evidence():
    lock = read_json(LOCK_PATH)
    synthetic = lock["production_path_synthetic_calibration"]
    assert AUTHORIZATION_PATH.is_file()
    assert sha256_file(AUTHORIZATION_PATH) == "441de7edc9a40da6bdf66faad2677d0abca2ca6119a1a9f599bc8727087371ee"
    assert lock["authorization"]["record_present_at_lock"] is False
    assert lock["authorization"]["historical_authorization_migrates"] is False
    assert synthetic["real_field_array_access"] == 0
    assert synthetic["real_target_label_access"] == 0
    assert lock["chronology"]["target_label_access_at_lock"] == 0
    assert lock["chronology"]["real_selector_scores_at_lock"] == 0
    assert lock["chronology"]["scientific_statistics_at_lock"] == 0
