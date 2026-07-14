import hashlib
import json

from oaci.multidataset import c84l1c_engineering_result as result


def _load_result():
    return json.loads((result.REPORT_DIR / "C84L1C_ENGINEERING_CANARY_RESULT.json").read_text())


def test_result_identity_and_gate():
    payload = _load_result()
    assert payload["gate"] == "C84L1C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84FL2_REVIEW_REQUIRED"
    assert payload["engineering_only"] is True
    assert payload["scientific_result_available"] is False
    digest = hashlib.sha256(
        (result.REPORT_DIR / "C84L1C_ENGINEERING_CANARY_RESULT.json").read_bytes()
    ).hexdigest()
    assert (result.REPORT_DIR / "C84L1C_ENGINEERING_CANARY_RESULT.sha256").read_text().split()[0] == digest


def test_complete_gate_and_scope():
    payload = _load_result()
    assert payload["complete_gate"]["complete"] is True
    assert payload["complete_gate"]["unit_count"] == 243
    assert payload["complete_gate"]["checkpoint_optimizer_sidecar_units"] == 243
    assert payload["complete_gate"]["strict_source_audit_artifacts"] == 243
    assert payload["complete_gate"]["target_unlabeled_artifacts"] == 243
    assert payload["scope"] == {
        "candidate_units": 243,
        "datasets": ["Lee2019_MI", "Cho2017", "PhysionetMI"],
        "level": 1,
        "regimes": ["ERM", "OACI", "SRC"],
        "source_panel": "A",
        "training_phases": 9,
        "training_seed": 5,
    }


def test_protected_state_and_later_stages_remain_closed():
    payload = _load_result()
    assert all(value == 0 for value in payload["isolation"].values())
    assert payload["C84F_authorized"] is False
    assert payload["C84S_authorized"] is False
    assert payload["authorization"]["historical_authorization_reused"] is False


def test_persisted_replay_and_tolerances():
    replay = _load_result()["persisted_replay"]
    assert replay["units"] == 243
    assert replay["checkpoint_replay_units"] == 243
    assert replay["optimizer_replay_units"] == 243
    assert replay["source_audit_replay_units"] == 243
    assert replay["target_unlabeled_replay_units"] == 243
    assert replay["max_Wz_plus_b_error"] <= 2e-5
    assert replay["max_linear_replay_error"] <= 2e-5
    assert replay["max_softmax_error"] <= 1e-6
    assert replay["max_repeat_logits_error"] <= 1e-6
    assert replay["max_repeat_z_error"] <= 1e-6


def test_dataset_support_contracts():
    datasets = {row["dataset"]: row for row in _load_result()["datasets"]}
    assert {name: row["deleted_source_subject"] for name, row in datasets.items()} == {
        "Lee2019_MI": 31,
        "Cho2017": 17,
        "PhysionetMI": 103,
    }
    assert all(row["deleted_class"] == "left_hand" for row in datasets.values())
    assert all(row["pre_support_cells"] == 24 for row in datasets.values())
    assert all(row["post_support_cells"] == 23 for row in datasets.values())
    assert all(row["minimum_post_support"] >= 8 for row in datasets.values())
    assert all(row["target_label_access"] == row["scientific_metrics"] == 0 for row in datasets.values())


def test_scheduler_evidence_uses_squeue_only():
    job = _load_result()["job"]
    assert job["job_id"] == 896066
    assert job["sacct_used"] is False
    assert job["squeue_observed_running"] is True
    assert job["squeue_final_state"] == "ABSENT_AFTER_APPLICATION_COMPLETE"
    assert job["application_complete"] is True


def test_compact_tables_have_expected_rows():
    expected = {
        "authorization_lock_replay.csv": 8,
        "dataset_support_replay.csv": 3,
        "complete_gate.csv": 5,
        "artifact_count_replay.csv": 15,
        "artifact_replay.csv": 243,
        "target_label_isolation.csv": 9,
        "external_artifact_manifest.csv": 6,
        "job_resource_ledger.csv": 1,
        "warning_ledger.csv": 2,
        "failure_and_retry_ledger.csv": 2,
    }
    for name, count in expected.items():
        lines = (result.TABLE_DIR / name).read_text().splitlines()
        assert len(lines) - 1 == count, name


def test_external_payload_is_not_copied_into_git():
    payload = _load_result()
    assert payload["external_root"].startswith("/projects/")
    assert payload["external_root_bytes"] > 0
    assert not any(result.REPORT_DIR.rglob("*.pt"))
    assert not any(result.REPORT_DIR.rglob("*.npz"))
