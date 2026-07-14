from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

from oaci.multidataset import c84r2_canary_runtime_repair as runtime


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_module_imports_no_protected_package(monkeypatch):
    before = set(sys.modules)
    importlib.reload(runtime)
    loaded = {name.split(".")[0] for name in set(sys.modules) - before}
    assert not loaded & {"numpy", "torch", "mne", "moabb", "braindecode", "skorch"}


def test_bound_implementation_modified_after_lock_fails(tmp_path):
    path = tmp_path / "adapter.py"
    path.write_text("LOCKED = True\n", encoding="ascii")
    lock = {"runtime_bound_objects": [{"path": "adapter.py", "sha256": _sha(path), "bytes": path.stat().st_size}]}
    assert runtime.verify_bound_object_registry(lock, repo_root=tmp_path)[0]["replay_pass"]
    path.write_text("LOCKED = False\n", encoding="ascii")
    with pytest.raises(runtime.C84R2RuntimeError, match="SHA-256 drift"):
        runtime.verify_bound_object_registry(lock, repo_root=tmp_path)


def test_bound_registry_modified_after_lock_fails(tmp_path):
    path = tmp_path / "registry.csv"
    path.write_text("key,value\nchannels,20\n", encoding="ascii")
    lock = {"runtime_bound_objects": [{"path": "registry.csv", "sha256": _sha(path)}]}
    path.write_text("key,value\nchannels,19\n", encoding="ascii")
    with pytest.raises(runtime.C84R2RuntimeError, match="runtime-bound SHA-256 drift"):
        runtime.verify_bound_object_registry(lock, repo_root=tmp_path)


@pytest.mark.parametrize("changed", ["torch", "moabb", "mne"])
def test_wrong_distribution_version_fails_without_import(changed):
    lock = {"environment": {
        "python": "3.9.25", "conda_prefix": "/locked",
        "distributions": {"torch": "2.6.0", "moabb": "1.5.0", "mne": "1.11.0"},
    }}
    versions = {"torch": "2.6.0", "moabb": "1.5.0", "mne": "1.11.0"}
    versions[changed] = "0.0.0"
    with pytest.raises(runtime.C84R2RuntimeError, match=changed):
        runtime.verify_distribution_environment(
            lock, version_getter=versions.__getitem__, python_version="3.9.25", prefix="/locked",
        )


def test_wrong_python_version_fails():
    lock = {"environment": {"python": "3.9.25", "conda_prefix": "/locked", "distributions": {}}}
    with pytest.raises(runtime.C84R2RuntimeError, match="Python version"):
        runtime.verify_distribution_environment(lock, python_version="3.13.7", prefix="/locked")


def test_wrong_loader_source_hash_fails_before_data(tmp_path):
    path = tmp_path / "moabb/datasets/loader.py"
    path.parent.mkdir(parents=True)
    path.write_text("class Loader: pass\n", encoding="ascii")
    lock = {"loader_source_identity": {"files": [{
        "qualified_object": "moabb.datasets.Loader",
        "distribution_relative_path": "moabb/datasets/loader.py",
        "sha256": "0" * 64,
    }]}}
    with pytest.raises(runtime.C84R2RuntimeError, match="loader source identity drift"):
        runtime.verify_loader_source_files(lock, locate_distribution_file=lambda _: path)


def test_exact_loader_source_hash_passes(tmp_path):
    path = tmp_path / "moabb/paradigms/mi.py"
    path.parent.mkdir(parents=True)
    path.write_text("class MotorImagery: pass\n", encoding="ascii")
    lock = {"loader_source_identity": {"files": [{
        "qualified_object": "moabb.paradigms.MotorImagery",
        "distribution_relative_path": "moabb/paradigms/mi.py",
        "sha256": _sha(path),
    }]}}
    replay = runtime.verify_loader_source_files(lock, locate_distribution_file=lambda _: path)
    assert replay[0]["before_get_data"] is True


def test_montage_order_and_digest_are_both_enforced():
    lock = {"interface": {
        "channels": list(runtime.EXPECTED_CHANNELS),
        "montage_sha256": runtime.EXPECTED_MONTAGE_SHA256,
        "Fz_substitution": False, "FCz_interpolation": False,
        "zero_fill": False, "dataset_specific_mask": False,
    }}
    assert runtime.verify_montage_binding(lock)["channel_count"] == 20
    lock["interface"]["channels"] = list(reversed(runtime.EXPECTED_CHANNELS))
    with pytest.raises(runtime.C84R2RuntimeError, match="list/order"):
        runtime.verify_montage_binding(lock)


def test_deterministic_environment_requires_both_variables():
    assert runtime.verify_deterministic_environment(environ={
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0",
    })["PYTHONHASHSEED"] == "0"
    with pytest.raises(runtime.C84R2RuntimeError, match="deterministic runtime"):
        runtime.verify_deterministic_environment(environ={"PYTHONHASHSEED": "0"})


def test_attempt_ledger_persists_failure_and_partial_manifest(tmp_path):
    consumption = {"sha256": "a" * 64}
    ledger = runtime.ExecutionAttemptLedger(tmp_path, consumption)
    ledger.stage("package_imports")
    error = ImportError("synthetic protected import failure")
    ledger.fail(error)
    events = [json.loads(line) for line in (tmp_path / "execution_attempts.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "attempt_started"
    assert events[-1]["event"] == "failed"
    partial = json.loads((tmp_path / "partial_artifact_manifest.json").read_text())
    assert partial["status"] == "FAILED"
    assert partial["error_type"] == "ImportError"
    assert partial["retry_disposition"] == "NEW_ADDITIVE_REPAIR_AND_LOCK_REQUIRED"
    assert partial["counters"]["real_EEG_arrays_materialized"] == 0


def test_historical_lock_and_adapter_remain_byte_identical():
    historical_lock = runtime.REPORT_DIR / "C84C_EXECUTION_LOCK.json"
    historical_sidecar = runtime.REPORT_DIR / "C84C_EXECUTION_LOCK.sha256"
    assert runtime.sha256_file(historical_lock) == historical_sidecar.read_text().split()[0]
    assert runtime.sha256_file(historical_lock) == "f9cabf8f362917d663e13154910085d5b105740b265789a2323dd7bc0193222b"
    payload = json.loads(historical_lock.read_text())
    assert payload["schema_version"] == "c84c_execution_lock_v1"
    assert payload["status"] == runtime.LOCK_READY_STATUS


def test_repair_protocol_hash_replays_and_records_zero_access():
    expected = runtime.REPAIR_PROTOCOL_SHA_PATH.read_text().split()[0]
    assert runtime.sha256_file(runtime.REPAIR_PROTOCOL_PATH) == expected
    payload = json.loads(runtime.REPAIR_PROTOCOL_PATH.read_text())
    assert payload["epistemic_status"]["real_EEG_access_before_protocol"] == 0
    assert payload["forbidden"]["C84C_execution_in_C84R2"] is True


def test_V3_protocol_hashes_and_environment_repair_replay():
    canary_path = runtime.REPORT_DIR / "C84_CANARY_PROTOCOL_V3.json"
    field_path = runtime.REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V3.json"
    assert runtime.sha256_file(canary_path) == (runtime.REPORT_DIR / "C84_CANARY_PROTOCOL_V3.sha256").read_text().split()[0]
    assert runtime.sha256_file(field_path) == (runtime.REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V3.sha256").read_text().split()[0]
    canary = json.loads(canary_path.read_text())
    field = json.loads(field_path.read_text())
    assert canary["environment"]["python"] == "3.13.7"
    assert canary["environment"]["historical_python_3_9_25_replayable"] is False
    assert canary["parent_external_protocol"]["scientific_interface_changed"] is False
    assert field["canary_reuse"]["strict_source_artifacts_required"] == 243
    assert field["canary_reuse"]["target_unlabeled_artifacts_required"] == 243
    assert field["scope_specific_execution_lock_created_in_C84R2"] is False


def test_execution_lock_v2_hash_and_all_bound_objects_replay():
    lock_path = runtime.REPORT_DIR / "C84C_EXECUTION_LOCK_V2.json"
    sidecar = runtime.REPORT_DIR / "C84C_EXECUTION_LOCK_V2.sha256"
    assert runtime.verify_lock_self(lock_path, sidecar) == sidecar.read_text().split()[0]
    lock = json.loads(lock_path.read_text())
    replay = runtime.verify_bound_object_registry(lock)
    assert len(replay) == lock["runtime_bound_object_count"] == 63
    assert all(row["replay_pass"] for row in replay)
    assert len(runtime.verify_protocol_sidecars(lock)) == 6
    assert runtime.verify_candidate_identity(lock)["canary_unit_count"] == 243


def test_execution_lock_v2_exact_environment_and_loader_sources_replay():
    lock = json.loads((runtime.REPORT_DIR / "C84C_EXECUTION_LOCK_V2.json").read_text())
    observed = runtime.verify_distribution_environment(lock)
    assert observed["python"] == "3.13.7"
    assert observed["distributions"] == {
        "chardet": "5.2.0", "mne": "1.11.0", "moabb": "1.5.0", "torch": "2.6.0",
    }
    sources = runtime.verify_loader_source_files(lock)
    assert len(sources) == 4
    assert all(row["before_get_data"] for row in sources)


def test_execution_lock_v2_preserves_authorization_lifecycle_and_has_no_field_science_lock():
    lock = json.loads((runtime.REPORT_DIR / "C84C_EXECUTION_LOCK_V2.json").read_text())
    assert lock["status"] == runtime.LOCK_READY_STATUS
    assert lock["historical_lock_supersession"]["operative_for_execution"] is False
    assert lock["authorization"]["record_present_at_lock"] is False
    if runtime.AUTHORIZATION_RECORD_PATH.exists():
        record = json.loads(runtime.AUTHORIZATION_RECORD_PATH.read_text())
        failure = json.loads((runtime.REPORT_DIR / "C84C_FAILED_ATTEMPT_895366.json").read_text())
        assert record["direct_explicit_PI_authorization"] is True
        assert failure["authorization_consumed"] is True
        assert failure["job_id"] == 895366
    names = {path.name for path in runtime.REPORT_DIR.glob("C84*EXECUTION_LOCK*.json")}
    assert {"C84C_EXECUTION_LOCK.json", "C84C_EXECUTION_LOCK_V2.json"} <= names
    assert not any(name.startswith(("C84F_", "C84S_")) for name in names)


def test_missing_authorization_fails_before_output_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    with pytest.raises(runtime.C84R2RuntimeError):
        runtime.require_authorization_and_lock(
            authorization_path=tmp_path / "missing.json", output_root=tmp_path / "external",
        )
    assert not (tmp_path / "external").exists()
