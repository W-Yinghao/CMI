"""Fail-closed runtime guard for the C84F target-stage repair.

The guard is standard-library only. It replays repository bytes, the failed
attempt, and every frozen model-field artifact before creating a replacement
output root. It never imports a loader, numerical stack, or training module.
"""
from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping

from . import c84f_field_manifest as manifests
from . import c84f_runtime_guard as base


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
EXECUTION_LOCK_PATH = REPORT_DIR / "C84F_TARGET_STAGE_EXECUTION_LOCK.json"
EXECUTION_LOCK_SHA_PATH = REPORT_DIR / "C84F_TARGET_STAGE_EXECUTION_LOCK.sha256"
AUTHORIZATION_RECORD_PATH = REPORT_DIR / "C84F_TARGET_STAGE_PI_AUTHORIZATION_RECORD.json"
DEFAULT_EXTERNAL_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-repair-v1"
)
LOCK_READY_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"


class C84FR1RuntimeError(RuntimeError):
    """Raised before protected target access when a binding cannot replay."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C84FR1RuntimeError(message)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_failed_attempt_and_model_field(lock: Mapping[str, Any]) -> dict[str, Any]:
    binding = lock.get("frozen_failed_attempt", {})
    root = Path(str(binding.get("root", "")))
    _require(root.is_dir(), "frozen C84F failed root is absent")
    required = {
        "failure_evidence": REPORT_DIR / "C84F_FAILED_ATTEMPT_896185.json",
        "authorization_consumed": root / "authorization_consumed.json",
        "execution_attempts": root / "execution_attempts.jsonl",
        "partial_manifest": root / "partial_artifact_manifest.json",
        "model_manifest": root / "C84F_MODEL_FIELD_MANIFEST.json",
        "target_raw_manifest": root / "C84F_TARGET_RAW_INPUT_MANIFEST.json",
    }
    hashes = binding.get("sha256", {})
    for name, path in required.items():
        _require(path.is_file(), f"frozen failed-attempt object is absent: {name}")
        _require(base.sha256_file(path) == hashes.get(name), f"frozen object hash drift: {name}")

    partial = read_json(required["partial_manifest"])
    counters = partial.get("counters", {})
    _require(partial.get("status") == "FAILED", "historical partial manifest is not failed")
    _require(partial.get("error_type") == "TypeError", "historical failure type drift")
    _require(
        partial.get("error") == "'<' not supported between instances of 'dict' and 'dict'",
        "historical failure message drift",
    )
    expected_zero = (
        "target_registry_trials", "target_unlabeled_artifacts", "target_context_slices",
        "target_y_accesses", "target_scientific_metrics", "canary_contexts_replayed",
    )
    _require(all(int(counters.get(name, -1)) == 0 for name in expected_zero),
             "historical target/scientific counters are not zero")
    _require(int(counters.get("target_EEG_arrays", -1)) == 118,
             "historical target-X coverage is not 118")
    _require(partial.get("target_outcome_decision") is False,
             "historical retry used a target outcome")

    model_sha_path = root / "C84F_MODEL_FIELD_MANIFEST.sha256"
    replay = manifests.verify_model_field_freeze(required["model_manifest"], model_sha_path)
    manifest = read_json(required["model_manifest"])
    rows = list(manifest.get("units", ()))
    _require(len(rows) == 1944, "frozen model field does not contain 1,944 units")
    artifact_count = 0
    for row in rows:
        for path_field, hash_field in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("optimizer_path", "optimizer_sha256"),
            ("sidecar_path", "sidecar_sha256"),
            ("source_audit_path", "source_audit_sha256"),
        ):
            path = Path(str(row[path_field]))
            _require(path.is_file(), f"frozen model artifact is absent: {row['unit_id']}/{path_field}")
            _require(base.sha256_file(path) == row[hash_field],
                     f"frozen model artifact hash drift: {row['unit_id']}/{path_field}")
            artifact_count += 1
    _require(artifact_count == 7776, "frozen model artifact replay count is not 7,776")

    forbidden_old_outputs = (
        root / "C84F_TARGET_UNLABELED_TRIAL_REGISTRY.json",
        root / "C84F_COMPLETE_FIELD_MANIFEST.json",
        root / "complete_target_unlabeled",
        root / "target_context_index",
    )
    _require(not any(path.exists() for path in forbidden_old_outputs),
             "historical failed root contains a target/final output")
    return {
        "root": root,
        "model_manifest_path": required["model_manifest"],
        "model_manifest_sha_path": model_sha_path,
        "model_manifest_sha256": replay["sha256"],
        "target_raw_manifest_path": required["target_raw_manifest"],
        "model_rows": rows,
        "model_artifact_files_replayed": artifact_count,
        "historical_target_X_arrays": 118,
        "historical_target_y_accesses": 0,
        "historical_scientific_metrics": 0,
    }


def verify_authorization_record(
    lock: Mapping[str, Any], lock_sha256: str, lock_commit: str, authorization_path: Path,
) -> dict[str, Any]:
    _require(authorization_path.is_file(), "fresh target-stage PI authorization record is absent")
    record = read_json(authorization_path)
    required = {
        "schema_version": "c84fr1_direct_pi_authorization_record_v1",
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84F_TARGET_STAGE_REPAIR",
        "repair_protocol_sha256": lock["protocols"]["repair"]["sha256"],
        "execution_lock_sha256": lock_sha256,
        "execution_lock_commit": lock_commit,
        "model_field_manifest_sha256": lock["frozen_failed_attempt"]["sha256"]["model_manifest"],
        "model_retraining": False,
        "target_labels": False,
        "scientific_metrics": False,
        "same_label_oracle": False,
        "C84S": False,
    }
    mismatch = {key: (record.get(key), wanted) for key, wanted in required.items()
                if record.get(key) != wanted}
    _require(not mismatch, f"target-stage authorization binding mismatch: {mismatch}")
    return {**record, "record_sha256": base.sha256_file(authorization_path)}


def require_authorization_and_lock(
    *,
    authorization_path: Path = AUTHORIZATION_RECORD_PATH,
    output_root: Path = DEFAULT_EXTERNAL_ROOT,
    lock_path: Path = EXECUTION_LOCK_PATH,
    lock_sha_path: Path = EXECUTION_LOCK_SHA_PATH,
    repo_root: Path = REPO_ROOT,
    version_getter: Callable[[str], str] = metadata.version,
    python_version: str | None = None,
    prefix: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    _require(lock_path.is_file(), "target-stage execution lock is absent")
    lock_sha = base.verify_lock_self(lock_path, lock_sha_path)
    lock = read_json(lock_path)
    _require(lock.get("status") == LOCK_READY_STATUS, "target-stage lock is not ready")
    bound = base.verify_bound_object_registry(lock, repo_root=repo_root)
    protocols = base.verify_protocol_sidecars(lock, repo_root=repo_root)
    interface = base.verify_interface(lock)
    environment = base.verify_distribution_environment(
        lock, version_getter=version_getter,
        python_version=python_version or platform.python_version(),
        prefix=prefix or sys.prefix,
    )
    deterministic = base.verify_deterministic_environment(environ)
    candidates = base.verify_candidate_and_wave_registry(lock, repo_root=repo_root)
    reuse = base.verify_dual_canary_reuse(lock, repo_root=repo_root)
    frozen = verify_failed_attempt_and_model_field(lock)
    lock_commit = base.commit_for_path(lock_path, repo_root=repo_root)
    repository = base.verify_repository_state(lock, lock_commit=lock_commit, repo_root=repo_root)
    authorization = verify_authorization_record(lock, lock_sha, lock_commit, authorization_path)
    run_root = base.verify_output_root(Path(output_root), lock_sha)
    return {
        "lock": lock, "lock_sha256": lock_sha, "lock_commit": lock_commit,
        "authorization": authorization, "authorization_sha256": authorization["record_sha256"],
        "bound_object_replay": bound, "protocol_replay": protocols,
        "interface_replay": interface, "environment_replay": environment,
        "deterministic_environment": deterministic, "candidate_replay": candidates,
        "dual_canary_replay": reuse,
        "frozen_model_field_replay": frozen, "repository_replay": repository,
        "run_root": run_root,
    }


def consume_authorization(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(binding["run_root"]) / "authorization_consumed.json"
    _require(not path.exists(), "target-stage authorization was already consumed")
    payload = {
        "schema_version": "c84fr1_authorization_consumption_v1",
        "stage": "C84F_TARGET_STAGE_REPAIR",
        "execution_lock_sha256": binding["lock_sha256"],
        "execution_lock_commit": binding["lock_commit"],
        "authorization_record_sha256": binding["authorization_sha256"],
        "repair_protocol_sha256": binding["lock"]["protocols"]["repair"]["sha256"],
        "model_field_manifest_sha256": binding["frozen_model_field_replay"]["model_manifest_sha256"],
        "consumed_at_unix_ns": time.time_ns(),
        "before_protected_package_imports": True,
        "before_loader_import": True,
        "before_target_data_access": True,
        "model_retraining_authorized": False,
        "target_labels_authorized": False,
        "scientific_outcomes_authorized": False,
        "C84S_authorized": False,
    }
    digest = manifests.write_json_atomic(path, payload)
    return {**payload, "path": str(path), "sha256": digest}


ExecutionAttemptLedger = base.ExecutionAttemptLedger
verify_loader_source_files = base.verify_loader_source_files
verify_loader_runtime_objects = base.verify_loader_runtime_objects
verify_protected_runtime_versions = base.verify_protected_runtime_versions
