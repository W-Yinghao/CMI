from __future__ import annotations

import csv
import hashlib
import json

from oaci.multidataset import c84c_engineering_result as result


def _report():
    return json.loads((result.REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.json").read_text())


def test_c84c_result_identity_and_gate():
    report = _report()
    assert report["schema_version"] == "c84c_engineering_canary_result_v1"
    assert report["gate"] == result.GATE
    assert report["engineering_only"] is True
    assert report["scientific_result_available"] is False
    digest = hashlib.sha256(
        (result.REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.json").read_bytes()
    ).hexdigest()
    assert (result.REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.sha256").read_text().split()[0] == digest


def test_c84c_complete_gate_is_243_units_and_nine_phases():
    report = _report()
    assert report["scope"]["candidate_units"] == 243
    assert report["scope"]["training_phases"] == 9
    assert report["complete_gate"] == {
        "checkpoint_state_sidecar_units": 243,
        "complete": True,
        "persisted_replay_units": 243,
        "strict_source_audit_artifacts": 243,
        "target_unlabeled_artifacts": 243,
        "unit_count": 243,
    }


def test_c84c_dataset_interfaces_are_exact_and_complete():
    rows = _report()["datasets"]
    assert [row["dataset"] for row in rows] == list(result.DATASETS)
    assert all(row["unit_count"] == 81 for row in rows)
    assert all(row["channel_count"] == 20 for row in rows)
    assert all(row["channel_order"].split("|") == result.CHANNELS for row in rows)
    assert all(row["sfreq_hz"] == 160.0 and row["n_times"] == 480 for row in rows)
    assert all(row["interpolation_or_synthesis"] == 0 for row in rows)


def test_c84c_persisted_replay_uses_split_tolerances():
    replay = _report()["persisted_replay"]
    assert replay["checkpoint_replay_units"] == 243
    assert replay["optimizer_replay_units"] == 243
    assert replay["source_audit_replay_units"] == 243
    assert replay["target_unlabeled_replay_units"] == 243
    assert replay["max_linear_replay_error"] == 6.67572021484375e-06
    assert replay["max_linear_replay_error"] < 1e-5
    assert replay["max_softmax_error"] == 0
    assert replay["max_repeat_logits_error"] == 0
    assert replay["max_repeat_z_error"] == 0


def test_c84c_protected_target_state_is_zero_and_later_stages_unauthorized():
    report = _report()
    assert all(value == 0 for value in report["isolation"].values())
    assert report["C84F_authorized"] is False
    assert report["C84S_authorized"] is False
    assert report["authorization"]["failed_authorization_reused"] is False
    assert report["replacement_retries"] == 0


def test_c84c_artifact_field_sets_exclude_target_labels():
    replay = _report()["persisted_replay"]
    assert set(replay["source_fields"]) == result.SOURCE_FIELDS
    assert set(replay["target_fields"]) == result.TARGET_FIELDS
    assert replay["target_label_fields"] == 0
    assert not any("label" in name.lower() for name in replay["target_fields"])


def test_c84c_lock_manifest_and_candidate_hashes_replay_in_compact_result():
    report = _report()
    assert report["complete_manifest_sha256"] == result.EXPECTED["complete_manifest_sha256"]
    assert report["lock"]["execution_lock_v3_sha256"] == result.EXPECTED["execution_lock_v3_sha256"]
    assert report["lock"]["candidate_unit_ids_sha256"] == result.EXPECTED["candidate_unit_ids_sha256"]
    assert report["lock"]["runtime_bound_objects"] == 72
    assert report["lock"]["protocol_bindings"] == 7


def test_c84c_compact_tables_are_complete_and_pass():
    expected_rows = {
        "authorization_lock_replay.csv": 8,
        "dataset_interface_replay.csv": 3,
        "complete_gate.csv": 6,
        "artifact_replay.csv": 8,
        "artifact_count_replay.csv": 15,
        "target_label_isolation.csv": 11,
        "external_artifact_manifest.csv": 6,
        "job_resource_ledger.csv": 1,
        "warning_ledger.csv": 3,
        "failure_and_retry_ledger.csv": 3,
    }
    for name, count in expected_rows.items():
        with (result.TABLE_DIR / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == count, name
        if rows and "status" in rows[0]:
            assert all(row["status"] in {
                "PASS", "COMPLETED", "FAILED_PRESERVED", "FAILED_BEFORE_REPORT_WRITE",
            } for row in rows)


def test_c84c_warning_ledger_discloses_nonempty_stderr_without_failure():
    warnings = {row["warning"]: row for row in _report()["warnings"]}
    assert warnings["Physionet_unverified_HTTPS"]["count"] == 102
    assert warnings["Cho_continuous_stack_edge_effect_notice"]["count"] == 17
    assert warnings["traceback_or_runtime_failure"]["count"] == 0
