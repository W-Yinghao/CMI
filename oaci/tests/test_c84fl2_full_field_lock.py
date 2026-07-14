from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oaci.multidataset import c84f_runtime_guard as runtime
from oaci.multidataset import c84fl2_protocol as protocol


REPORT_DIR = protocol.REPORT_DIR


def _sidecar(path: Path) -> str:
    return path.read_text(encoding="ascii").split()[0]


@pytest.mark.parametrize(
    "name",
    (
        "C84FL2_DUAL_LEVEL_FULL_FIELD_RECONCILIATION_PROTOCOL",
        "C84_FIELD_GENERATION_PROTOCOL_V7",
        "C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL_V2",
    ),
)
def test_protocol_hash_sidecars_replay(name: str) -> None:
    path = REPORT_DIR / f"{name}.json"
    sidecar = REPORT_DIR / f"{name}.sha256"
    assert path.is_file() and sidecar.is_file()
    assert protocol.sha256_file(path) == _sidecar(sidecar)


def test_protocol_chronology_and_protected_state() -> None:
    reconciliation = protocol.read_json(
        REPORT_DIR / "C84FL2_DUAL_LEVEL_FULL_FIELD_RECONCILIATION_PROTOCOL.json"
    )
    assert reconciliation["timing"]["prospective_to_remaining_1458_unit_training"] is True
    assert reconciliation["timing"]["prospective_to_complete_target_unlabeled_field"] is True
    assert reconciliation["timing"]["target_labels_read_before_protocol"] == 0
    assert reconciliation["reuse"]["model_state_source_audit_units"] == 486
    assert reconciliation["field_arithmetic"]["remaining_units"] == 1458
    assert reconciliation["field_arithmetic"]["target_contexts"] == 944
    assert reconciliation["field_arithmetic"]["candidate_context_slices"] == 76464


def test_no_authorization_or_scientific_execution_lock_in_protocol_stage() -> None:
    assert not runtime.AUTHORIZATION_RECORD_PATH.exists()
    assert not (REPORT_DIR / "C84S_EXECUTION_LOCK.json").exists()
    if runtime.EXECUTION_LOCK_PATH.exists():
        lock = json.loads(runtime.EXECUTION_LOCK_PATH.read_text(encoding="utf-8"))
        assert lock["status"] == runtime.LOCK_READY_STATUS
        assert lock["scope"]["C84S"] is False


def test_authorization_record_binding_fails_closed_when_absent(tmp_path: Path) -> None:
    lock = {
        "protocols": {
            "reconciliation": {"sha256": "a" * 64},
            "field_v7": {"sha256": "b" * 64},
            "full_field_v2": {"sha256": "c" * 64},
        },
        "candidate_identity": {"registry_sha256": "d" * 64},
    }
    with pytest.raises(runtime.C84FRuntimeError, match="authorization record is absent"):
        runtime.verify_authorization_record(lock, "e" * 64, "f" * 40, tmp_path / "absent.json")


def test_bound_object_drift_is_detected_before_execution(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "bound.txt"
    path.write_text("locked", encoding="ascii")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = {"runtime_bound_objects": [{
        "path": "bound.txt", "sha256": digest, "bytes": path.stat().st_size,
        "blob": "blob-1",
    }]}
    replay = runtime.verify_bound_object_registry(
        lock, repo_root=repo, current_blob_getter=lambda value: "blob-1",
        head_blob_getter=lambda value: "blob-1",
    )
    assert replay[0]["replay_pass"]
    path.write_text("drift", encoding="ascii")
    with pytest.raises(runtime.C84FRuntimeError, match="SHA-256 drift"):
        runtime.verify_bound_object_registry(
            lock, repo_root=repo, current_blob_getter=lambda value: "blob-1",
            head_blob_getter=lambda value: "blob-1",
        )


def test_interface_replay_is_exact_twenty_channel_contract() -> None:
    interface = {
        "channels": list(protocol.read_json(
            REPORT_DIR / "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3.json"
        )["interface"]["primary_channel_allowlist"]),
        "montage_sha256": protocol.HASHES["montage"],
        "id": protocol.INTERFACE_ID, "sample_rate_hz": 160, "sample_count": 480,
        "linear_replay_abs_tolerance": 2e-5, "strict_identity_abs_tolerance": 1e-6,
        "Fz_substitution": False, "FCz_interpolation": False,
        "zero_fill": False, "dataset_specific_mask": False,
    }
    replay = runtime.verify_interface({"interface": interface})
    assert replay["montage_sha256"] == protocol.HASHES["montage"]
    interface["channels"] = list(reversed(interface["channels"]))
    with pytest.raises(runtime.C84FRuntimeError, match="channel list/order"):
        runtime.verify_interface({"interface": interface})


def test_candidate_registry_and_reuse_registry_static_arithmetic() -> None:
    operative = protocol.read_csv(protocol.TABLE_DIR / "operative_complete_unit_registry_replay.csv")
    reuse = protocol.read_csv(protocol.TABLE_DIR / "dual_canary_reuse_registry.csv")
    assert len(operative) == len({row["unit_id"] for row in operative}) == 1944
    assert len(reuse) == len({row["unit_id"] for row in reuse}) == 486
    assert sum(row["reuse_source"] == "C84C" for row in reuse) == 243
    assert sum(row["reuse_source"] == "C84L1C" for row in reuse) == 243
    assert all(row["complete_target_artifact_reusable"] == "0" for row in reuse)


def test_runtime_lock_when_present_binds_required_scope() -> None:
    if not runtime.EXECUTION_LOCK_PATH.exists():
        pytest.skip("execution lock is generated after implementation commit")
    lock_sha = runtime.verify_lock_self(runtime.EXECUTION_LOCK_PATH, runtime.EXECUTION_LOCK_SHA_PATH)
    lock = runtime.read_json(runtime.EXECUTION_LOCK_PATH)
    assert len(lock_sha) == 64
    assert lock["status"] == runtime.LOCK_READY_STATUS
    assert lock["scope"] == {
        "C84F": True, "C84S": False, "real_execution_at_lock": False,
        "candidate_units": 1944, "reused_units": 486, "new_units": 1458,
        "training_phases": 72, "target_contexts": 944,
        "candidate_context_slices": 76464,
    }
    assert lock["authorization"]["record_present_at_lock"] is False
    assert lock["forbidden"]["target_labels"] is True
    assert lock["forbidden"]["scientific_inference"] is True
