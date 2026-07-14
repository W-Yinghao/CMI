"""Fail-closed runtime guard for the future C84L1C engineering canary.

Only standard-library and existing no-import verification helpers are loaded
before authorization consumption. Torch, MOABB, MNE, NumPy, and loaders remain
protected imports inside the authorized canary entrypoint.
"""
from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from . import c84l1_protocols as protocol
from . import c84r3_canary_runtime_repair as prior


REPO_ROOT = prior.REPO_ROOT
REPORT_DIR = prior.REPORT_DIR
REPAIR_PROTOCOL_PATH = REPORT_DIR / "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.json"
REPAIR_PROTOCOL_SHA_PATH = REPORT_DIR / "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.sha256"
CANARY_PROTOCOL_PATH = REPORT_DIR / "C84_LEVEL1_CANARY_PROTOCOL_V1.json"
CANARY_PROTOCOL_SHA_PATH = REPORT_DIR / "C84_LEVEL1_CANARY_PROTOCOL_V1.sha256"
FIELD_PROTOCOL_PATH = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V5.json"
FIELD_PROTOCOL_SHA_PATH = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V5.sha256"
EXECUTION_LOCK_PATH = REPORT_DIR / "C84L1C_EXECUTION_LOCK.json"
EXECUTION_LOCK_SHA_PATH = REPORT_DIR / "C84L1C_EXECUTION_LOCK.sha256"
AUTHORIZATION_RECORD_PATH = REPORT_DIR / "C84L1C_PI_AUTHORIZATION_RECORD.json"
DEFAULT_EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84-level1-canary-v1")
LOCK_READY_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
LINEAR_REPLAY_ABS_TOLERANCE = prior.LINEAR_REPLAY_ABS_TOLERANCE
STRICT_IDENTITY_ABS_TOLERANCE = prior.STRICT_IDENTITY_ABS_TOLERANCE
MODEL_REUSE_FIELDS = (
    "unit_id", "dataset", "regime", "epoch", "trajectory_order", "model_state_hash",
    "parent_ERM_model_state_hash", "previous_trajectory_model_state_hash",
    "checkpoint_sha256", "optimizer_sha256", "source_audit_sha256", "sidecar_sha256",
)


class C84L1RuntimeError(prior.C84R3RuntimeError):
    """Raised whenever the level-1 canary cannot replay its complete lock."""


canonical_bytes = prior.canonical_bytes
sha256_bytes = prior.sha256_bytes
sha256_file = prior.sha256_file
read_json = prior.read_json
write_json_atomic = prior.write_json_atomic
ExecutionAttemptLedger = prior.ExecutionAttemptLedger
replay_checkpoint = prior.replay_checkpoint
replay_optimizer_state = prior.replay_optimizer_state
replay_source_audit_artifact = prior.replay_source_audit_artifact
replay_target_unlabeled_artifact = prior.replay_target_unlabeled_artifact
replay_sidecar = prior.replay_sidecar
validate_instrumentation_errors = prior.validate_instrumentation_errors
verify_loader_source_files = prior.verify_loader_source_files
verify_loader_runtime_objects = prior.verify_loader_runtime_objects
verify_protected_runtime_versions = prior.verify_protected_runtime_versions


def _sidecar_digest(path: Path) -> str:
    if not path.is_file():
        raise C84L1RuntimeError(f"missing SHA-256 sidecar: {path}")
    fields = path.read_text(encoding="ascii").split()
    if not fields or len(fields[0]) != 64:
        raise C84L1RuntimeError(f"malformed SHA-256 sidecar: {path}")
    return fields[0]


def verify_lock_self(lock_path: Path, lock_sha_path: Path) -> str:
    expected = _sidecar_digest(lock_sha_path)
    if sha256_file(lock_path) != expected:
        raise C84L1RuntimeError("C84L1C execution-lock SHA-256 replay failed")
    return expected


def verify_intervention_registry(lock: Mapping[str, Any]) -> dict[str, Any]:
    binding = lock.get("level_intervention", {})
    path = REPO_ROOT / str(binding.get("registry_path", ""))
    if not path.is_file() or sha256_file(path) != binding.get("registry_sha256"):
        raise C84L1RuntimeError("C84L1C level-intervention registry drift")
    rows = protocol.read_csv(path)
    observed = {(row["dataset"], row["panel"]): (int(row["deleted_source_subject"]), row["deleted_class"])
                for row in rows}
    expected = {key: (subject, protocol.DELETED_CLASS) for key, subject in protocol.DELETED_SUBJECTS.items()}
    if observed != expected:
        raise C84L1RuntimeError("C84L1C deletion cells differ from the locked registry")
    return {"registry_sha256": binding["registry_sha256"], "cells": len(rows), "replay_pass": True}


def verify_candidate_identity(lock: Mapping[str, Any]) -> dict[str, Any]:
    registry_sha = lock["level_intervention"]["registry_sha256"]
    _, rows, supersession, operative = protocol.identity_rows(registry_sha)
    canary = sorted(row["unit_id"] for row in rows if int(row["C84L1C_canary"]) == 1)
    level1 = sorted(row["unit_id"] for row in rows)
    operative_ids = sorted(row["unit_id"] for row in operative)
    blocked = sorted(row["historical_planned_level1_unit_id"] for row in supersession)
    digest = sha256_bytes(canonical_bytes(canary))
    binding = lock.get("candidate_identity", {})
    if len(canary) != 243 or digest != binding.get("canary_unit_ID_digest"):
        raise C84L1RuntimeError("C84L1C 243-unit candidate identity drift")
    if sha256_bytes(canonical_bytes(level1)) != binding.get("level1_unit_ID_digest"):
        raise C84L1RuntimeError("C84L1C complete level-1 candidate identity drift")
    if sha256_bytes(canonical_bytes(operative_ids)) != binding.get("operative_unit_ID_digest"):
        raise C84L1RuntimeError("C84L1C operative complete-field candidate identity drift")
    if sha256_bytes(canonical_bytes(blocked)) != binding.get("blocked_level1_unit_ID_digest"):
        raise C84L1RuntimeError("C84L1C historical blocked level-1 identity drift")
    if len(operative) != 1944 or int(binding.get("operative_unit_count", -1)) != 1944:
        raise C84L1RuntimeError("C84L1C lock does not bind the 1,944-unit operative universe")
    return {
        "canary_units": 243,
        "canary_unit_ID_digest": digest,
        "level1_units": 972,
        "operative_units": 1944,
        "replay_pass": True,
    }


def summarize_accepted_c84c_manifest(path: Path) -> dict[str, Any]:
    """Summarize only accepted engineering identities, never target values."""
    manifest = read_json(path)
    if manifest.get("unit_count") != 243 or manifest.get("training_phases") != 9:
        raise C84L1RuntimeError("accepted C84C manifest arithmetic drift")
    if manifest.get("target_label_access") != 0 or manifest.get("target_scientific_metrics") != 0:
        raise C84L1RuntimeError("accepted C84C manifest contains protected target access")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 3:
        raise C84L1RuntimeError("accepted C84C manifest dataset registry drift")
    summaries: dict[str, Any] = {}
    all_units: list[dict[str, Any]] = []
    for dataset_row in datasets:
        dataset = str(dataset_row.get("dataset"))
        units = dataset_row.get("units")
        fingerprint = dataset_row.get("deterministic_prefix")
        if dataset not in protocol.historical.DATASET_ORDER or not isinstance(units, list) or len(units) != 81:
            raise C84L1RuntimeError("accepted C84C dataset/unit identity drift")
        if not isinstance(fingerprint, Mapping) or len(fingerprint.get("plan_hashes", ())) != 4:
            raise C84L1RuntimeError("accepted C84C plan-hash registry drift")
        normalized = []
        for unit in units:
            if any(field not in unit for field in MODEL_REUSE_FIELDS):
                raise C84L1RuntimeError("accepted C84C model reuse field is absent")
            normalized.append({field: unit[field] for field in MODEL_REUSE_FIELDS})
        normalized.sort(key=lambda row: row["unit_id"])
        all_units.extend(normalized)
        summaries[dataset] = {
            "unit_count": 81,
            "unit_ID_digest": sha256_bytes(canonical_bytes([row["unit_id"] for row in normalized])),
            "model_unit_registry_sha256": sha256_bytes(canonical_bytes(normalized)),
            "plan_hashes": list(fingerprint["plan_hashes"]),
            "deterministic_prefix_sha256": dataset_row["deterministic_prefix_sha256"],
            "deterministic_prefix_model_init_sha256": fingerprint["model_init_state_sha256"],
        }
    if set(summaries) != set(protocol.historical.DATASET_ORDER):
        raise C84L1RuntimeError("accepted C84C dataset names drift")
    all_units.sort(key=lambda row: row["unit_id"])
    return {
        "unit_count": 243,
        "unit_ID_digest": sha256_bytes(canonical_bytes([row["unit_id"] for row in all_units])),
        "model_unit_registry_sha256": sha256_bytes(canonical_bytes(all_units)),
        "datasets": summaries,
    }


def verify_c84c_level0_binding(lock: Mapping[str, Any]) -> dict[str, Any]:
    binding = lock.get("accepted_C84C_level0", {})
    manifest = Path(str(binding.get("manifest_path", "")))
    if not manifest.is_file() or sha256_file(manifest) != binding.get("manifest_sha256"):
        raise C84L1RuntimeError("accepted C84C level-0 manifest replay failed")
    if int(binding.get("reusable_units", -1)) != 243:
        raise C84L1RuntimeError("accepted C84C level-0 reuse count drift")
    observed = summarize_accepted_c84c_manifest(manifest)
    expected = {
        "unit_count": binding.get("reusable_units"),
        "unit_ID_digest": binding.get("unit_ID_digest"),
        "model_unit_registry_sha256": binding.get("model_unit_registry_sha256"),
        "datasets": binding.get("datasets"),
    }
    if observed != expected:
        raise C84L1RuntimeError("accepted C84C plan/model registry replay failed")
    return {
        "manifest_sha256": binding["manifest_sha256"],
        "reusable_units": 243,
        "model_unit_registry_sha256": observed["model_unit_registry_sha256"],
        "replay_pass": True,
    }


def verify_authorization_record(
    lock: Mapping[str, Any],
    lock_sha256: str,
    lock_commit: str,
    authorization_path: Path,
) -> dict[str, Any]:
    if not authorization_path.is_file():
        raise C84L1RuntimeError("fresh direct C84L1C PI authorization record is absent")
    record = read_json(authorization_path)
    required = {
        "schema_version": "c84l1c_direct_pi_authorization_record_v1",
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84L1C",
        "canary_protocol_v1_sha256": lock["canary_protocol"]["sha256"],
        "execution_lock_sha256": lock_sha256,
        "execution_lock_commit": lock_commit,
        "repair_protocol_sha256": lock["repair_protocol"]["sha256"],
        "level_intervention_registry_sha256": lock["level_intervention"]["registry_sha256"],
        "canary_unit_ID_digest": lock["candidate_identity"]["canary_unit_ID_digest"],
        "C84F": False,
        "C84S": False,
        "scientific_metrics": False,
        "same_label_oracle": False,
    }
    mismatch = {key: (record.get(key), wanted) for key, wanted in required.items() if record.get(key) != wanted}
    if mismatch:
        raise C84L1RuntimeError(f"C84L1C authorization binding mismatch: {mismatch}")
    return {**record, "record_sha256": sha256_file(authorization_path)}


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
    if not lock_path.is_file():
        raise C84L1RuntimeError("C84L1C execution lock is absent")
    lock_sha = verify_lock_self(lock_path, lock_sha_path)
    lock = read_json(lock_path)
    if lock.get("status") != LOCK_READY_STATUS:
        raise C84L1RuntimeError("C84L1C lock status is not the unique readiness state")
    bound = prior.verify_bound_object_registry(lock, repo_root=repo_root)
    protocols = prior.verify_protocol_sidecars(lock, repo_root=repo_root)
    montage = prior.verify_montage_binding(lock)
    environment = prior.verify_distribution_environment(
        lock, version_getter=version_getter, python_version=python_version, prefix=prefix,
    )
    deterministic = prior.verify_deterministic_environment(environ=environ)
    intervention = verify_intervention_registry(lock)
    candidate = verify_candidate_identity(lock)
    c84c = verify_c84c_level0_binding(lock)
    lock_commit = prior.commit_for_path(lock_path, repo_root=repo_root)
    repository = prior.verify_repository_state(lock, lock_commit=lock_commit, repo_root=repo_root)
    authorization = verify_authorization_record(lock, lock_sha, lock_commit, authorization_path)
    run_root = prior.verify_output_root(Path(output_root), lock_sha)
    return {
        "lock": lock,
        "lock_sha256": lock_sha,
        "lock_commit": lock_commit,
        "authorization": authorization,
        "authorization_sha256": authorization["record_sha256"],
        "bound_object_replay": bound,
        "protocol_replay": protocols,
        "montage_replay": montage,
        "environment_replay": environment,
        "deterministic_environment": deterministic,
        "intervention_replay": intervention,
        "candidate_replay": candidate,
        "C84C_level0_replay": c84c,
        "repository_replay": repository,
        "run_root": run_root,
    }


def consume_authorization(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(binding["run_root"]) / "authorization_consumed.json"
    if path.exists():
        raise C84L1RuntimeError("C84L1C authorization was already consumed")
    payload = {
        "schema_version": "c84l1c_authorization_consumption_v1",
        "stage": "C84L1C",
        "execution_lock_sha256": binding["lock_sha256"],
        "execution_lock_commit": binding["lock_commit"],
        "canary_protocol_v1_sha256": binding["lock"]["canary_protocol"]["sha256"],
        "repair_protocol_sha256": binding["lock"]["repair_protocol"]["sha256"],
        "authorization_record_sha256": binding["authorization_sha256"],
        "consumed_at_unix_ns": time.time_ns(),
        "before_package_imports": True,
        "before_CUDA_check": True,
        "before_dataset_loader_import": True,
        "before_download_or_get_data": True,
        "target_scientific_outcomes_authorized": False,
        "C84F_authorized": False,
        "C84S_authorized": False,
    }
    write_json_atomic(path, payload)
    return {**payload, "path": str(path), "sha256": sha256_file(path)}


def validate_complete_level1_canary_gate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 243 or len({str(row.get("unit_id")) for row in rows}) != 243:
        raise C84L1RuntimeError("C84L1C complete gate requires 243 unique units")
    required_true = (
        "checkpoint_replay_pass", "optimizer_replay_pass", "source_audit_replay_pass",
        "target_unlabeled_replay_pass", "sidecar_replay_pass", "support_replay_pass",
        "paired_model_init_pass", "level0_plan_replay_pass",
    )
    for row in rows:
        if any(not bool(row.get(field)) for field in required_true):
            raise C84L1RuntimeError(f"C84L1C unit failed complete replay: {row.get('unit_id')}")
        if int(row.get("target_y_access", -1)) or int(row.get("target_scientific_metrics", -1)):
            raise C84L1RuntimeError("C84L1C complete gate observed target-label/scientific access")
    return {
        "complete": True,
        "unit_count": 243,
        "checkpoint_optimizer_sidecar_units": 243,
        "strict_source_audit_artifacts": 243,
        "target_unlabeled_artifacts": 243,
        "target_y_access": 0,
        "target_scientific_metrics": 0,
    }
