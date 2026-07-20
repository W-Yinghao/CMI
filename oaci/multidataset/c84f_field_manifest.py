"""Fail-closed manifests and barriers for the future C84F execution.

This module is deliberately standard-library only.  It validates complete
model and target-field metadata before opening a final manifest path, and it
provides the mechanical barrier that prevents new target access before the
1,944-unit model field has frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from . import c84_dataset_registry_v2 as dataset_registry
from . import c84fl2_protocol as protocol


MODEL_FIELDS = protocol.MODEL_FIELDS
TARGET_TRIAL_FIELDS = protocol.TARGET_TRIAL_FIELDS
FIELD_DESCRIPTOR_FIELDS = protocol.FIELD_DESCRIPTOR_FIELDS
MODEL_MANIFEST_NAME = "C84F_MODEL_FIELD_MANIFEST.json"
MODEL_MANIFEST_SHA_NAME = "C84F_MODEL_FIELD_MANIFEST.sha256"
TARGET_REGISTRY_NAME = "C84F_TARGET_UNLABELED_TRIAL_REGISTRY.json"
TARGET_REGISTRY_SHA_NAME = "C84F_TARGET_UNLABELED_TRIAL_REGISTRY.sha256"
COMPLETE_MANIFEST_NAME = "C84F_COMPLETE_FIELD_MANIFEST.json"
COMPLETE_MANIFEST_SHA_NAME = "C84F_COMPLETE_FIELD_MANIFEST.sha256"

LABEL_LIKE_TOKENS = (
    "label", "class", "event", "target_y", "ground_truth", "accuracy",
    "regret", "q1", "q2", "budget", "selector", "scientific_metric",
)


class C84FManifestError(RuntimeError):
    """Raised before an incomplete or protected C84F artifact can publish."""


@dataclass(frozen=True)
class FieldScope:
    units: int = protocol.TOTAL_UNITS
    reused_units: int = protocol.REUSED_UNITS
    new_units: int = protocol.REMAINING_UNITS
    level_units: int = 972
    phases: int = protocol.TOTAL_PHASES
    subjects: int = 118
    contexts: int = protocol.TOTAL_CONTEXTS
    slices: int = protocol.TOTAL_SLICES
    canary_contexts: int = 6


REAL_SCOPE = FieldScope()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json_atomic(path: str | Path, value: Any) -> str:
    target = Path(path)
    _atomic_write_bytes(target, canonical_bytes(value) + b"\n")
    return sha256_file(target)


def write_hash_sidecar(path: str | Path, digest: str) -> None:
    target = Path(path)
    _atomic_write_bytes(target, f"{digest}  {target.name.removesuffix('.sha256')}\n".encode("ascii"))


def _sidecar_digest(path: Path) -> str:
    if not path.is_file():
        raise C84FManifestError(f"missing SHA-256 sidecar: {path}")
    fields = path.read_text(encoding="ascii").split()
    if not fields or len(fields[0]) != 64:
        raise C84FManifestError(f"malformed SHA-256 sidecar: {path}")
    return fields[0]


def _require_exact_fields(row: Mapping[str, Any], fields: Sequence[str], object_name: str) -> None:
    observed = set(row)
    expected = set(fields)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise C84FManifestError(f"{object_name} field-set drift: missing={missing} unknown={unknown}")


def _require_sha(value: Any, object_name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise C84FManifestError(f"{object_name} is not a lowercase SHA-256")
    return text


def _require_zero(row: Mapping[str, Any], fields: Sequence[str], object_name: str) -> None:
    nonzero = {field: row.get(field) for field in fields if int(row.get(field, -1)) != 0}
    if nonzero:
        raise C84FManifestError(f"{object_name} protected counters are nonzero: {nonzero}")


def validate_model_field_rows(
    rows: Iterable[Mapping[str, Any]], *, scope: FieldScope = REAL_SCOPE,
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    if len(values) != scope.units:
        raise C84FManifestError(f"model field has {len(values)} units, expected {scope.units}")
    unit_ids: set[str] = set()
    level_counts = {0: 0, 1: 0}
    reuse_counts = {"C84C": 0, "C84L1C": 0, "C84F": 0}
    required_true = (
        "checkpoint_replay_pass", "optimizer_replay_pass", "sidecar_replay_pass",
        "source_audit_replay_pass",
    )
    protected_zero = (
        "training_target_rows", "training_target_labels", "source_audit_rows_used_in_training",
        "target_outcome_retention", "target_outcome_retry",
    )
    for row in values:
        _require_exact_fields(row, MODEL_FIELDS, "model-field row")
        unit_id = str(row["unit_id"])
        if not unit_id or unit_id in unit_ids:
            raise C84FManifestError(f"duplicate or empty model-field unit ID: {unit_id!r}")
        unit_ids.add(unit_id)
        level = int(row["level"])
        if level not in level_counts:
            raise C84FManifestError(f"unregistered C84F level: {level}")
        level_counts[level] += 1
        expected_intervention = protocol.LEVEL0_ID if level == 0 else protocol.LEVEL1_ID
        if row["level_intervention_id"] != expected_intervention:
            raise C84FManifestError(f"level intervention drift for {unit_id}")
        if any(not bool(int(row[field])) for field in required_true):
            raise C84FManifestError(f"model replay gate failed for {unit_id}")
        _require_zero(row, protected_zero, f"model-field unit {unit_id}")
        for field in ("checkpoint_sha256", "optimizer_sha256", "sidecar_sha256", "source_audit_sha256",
                      "model_state_hash", "parent_ERM_model_state_hash",
                      "previous_trajectory_model_state_hash", "population_signature_sha256",
                      "support_graph_sha256", "paired_model_init_hash"):
            _require_sha(row[field], f"{unit_id}.{field}")
        if not all(str(row[field]) for field in (
            "checkpoint_path", "optimizer_path", "sidecar_path", "source_audit_path",
            "plan_hashes",
        )):
            raise C84FManifestError(f"model-field identity is incomplete for {unit_id}")
        provenance = str(row["reuse_provenance"])
        if provenance not in reuse_counts:
            raise C84FManifestError(f"unknown model provenance for {unit_id}: {provenance}")
        reuse_counts[provenance] += 1

    if scope == REAL_SCOPE:
        if level_counts != {0: scope.level_units, 1: scope.level_units}:
            raise C84FManifestError(f"model-field level arithmetic drift: {level_counts}")
        expected_reuse = {"C84C": 243, "C84L1C": 243, "C84F": scope.new_units}
        if reuse_counts != expected_reuse:
            raise C84FManifestError(f"model-field reuse arithmetic drift: {reuse_counts}")
    elif sum(reuse_counts.values()) != scope.units:
        raise C84FManifestError("synthetic model-field provenance arithmetic drift")
    return {
        "complete": True,
        "unit_count": len(values),
        "unique_unit_ids": len(unit_ids),
        "level_counts": {str(key): value for key, value in level_counts.items()},
        "reuse_counts": reuse_counts,
        "training_target_rows": 0,
        "training_target_labels": 0,
        "source_audit_rows_used_in_training": 0,
        "target_outcome_retention": 0,
        "target_outcome_retry": 0,
    }


def publish_model_field_manifest(
    root: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    execution_identity: Mapping[str, Any],
    scope: FieldScope = REAL_SCOPE,
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    gate = validate_model_field_rows(values, scope=scope)
    payload = {
        "schema_version": "c84f_model_field_manifest_v1",
        "execution_identity": dict(execution_identity),
        "complete_gate": gate,
        "units": values,
        "unit_count": len(values),
        "training_phases": scope.phases,
        "target_access_before_freeze": 0,
        "target_label_access": 0,
        "scientific_metrics": 0,
        "published_at_unix_ns": time.time_ns(),
    }
    directory = Path(root)
    path = directory / MODEL_MANIFEST_NAME
    sidecar = directory / MODEL_MANIFEST_SHA_NAME
    if path.exists() or sidecar.exists():
        raise C84FManifestError("model-field manifest already exists; same-identity overwrite is forbidden")
    digest = write_json_atomic(path, payload)
    write_hash_sidecar(sidecar, digest)
    return {**payload, "path": str(path), "sha256": digest, "sha256_path": str(sidecar)}


def verify_model_field_freeze(
    path: str | Path,
    sha_path: str | Path,
    *,
    scope: FieldScope = REAL_SCOPE,
) -> dict[str, Any]:
    manifest_path = Path(path)
    sidecar_path = Path(sha_path)
    expected = _sidecar_digest(sidecar_path)
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected:
        raise C84FManifestError("model-field manifest identity replay failed")
    payload = read_json(manifest_path)
    if payload.get("schema_version") != "c84f_model_field_manifest_v1":
        raise C84FManifestError("model-field manifest schema drift")
    observed = validate_model_field_rows(payload.get("units", ()), scope=scope)
    if payload.get("complete_gate") != observed or payload.get("target_access_before_freeze") != 0:
        raise C84FManifestError("model-field complete gate replay failed")
    if int(payload.get("training_phases", -1)) != scope.phases:
        raise C84FManifestError("model-field training-phase count drift")
    return {"path": str(manifest_path), "sha256": expected, **observed}


def expected_target_subjects() -> dict[str, tuple[int, ...]]:
    values: dict[str, tuple[int, ...]] = {}
    for dataset, spec in dataset_registry.DATASETS.items():
        partition = dataset_registry.partition_subjects(spec)
        values[dataset] = tuple(int(subject) for subject in partition["targets"])
    if {key: len(value) for key, value in values.items()} != protocol.TARGET_COUNTS:
        raise C84FManifestError("target-subject partition arithmetic drift")
    return values


def _contains_label_like_field(row: Mapping[str, Any]) -> bool:
    for field in row:
        lowered = str(field).lower()
        if lowered in {"y", "target", "targets"} or any(token in lowered for token in LABEL_LIKE_TOKENS):
            return True
    return False


def validate_target_trial_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_subject_map: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    if not values:
        raise C84FManifestError("target-unlabeled trial registry is empty")
    subjects = expected_target_subjects() if expected_subject_map is None else {
        key: tuple(int(value) for value in sequence) for key, sequence in expected_subject_map.items()
    }
    observed_subjects = {dataset: set() for dataset in subjects}
    trial_ids: set[tuple[str, str]] = set()
    for row in values:
        _require_exact_fields(row, TARGET_TRIAL_FIELDS, "target-unlabeled trial row")
        if _contains_label_like_field(row):
            raise C84FManifestError("target-unlabeled registry contains a label/scientific field")
        dataset = str(row["dataset"])
        if dataset not in subjects:
            raise C84FManifestError(f"unregistered target dataset: {dataset}")
        subject = int(row["target_subject_id"])
        if subject not in set(subjects[dataset]):
            raise C84FManifestError(f"unregistered target subject: {dataset}/{subject}")
        observed_subjects[dataset].add(subject)
        key = (dataset, str(row["target_trial_id"]))
        if not key[1] or key in trial_ids:
            raise C84FManifestError(f"duplicate or empty target trial ID: {key}")
        trial_ids.add(key)
        if row["interface_id"] != protocol.INTERFACE_ID or row["montage_sha256"] != protocol.HASHES["montage"]:
            raise C84FManifestError(f"target interface drift for {key}")
        if float(row["sample_rate_hz"]) != 160.0 or int(row["sample_count"]) != 480:
            raise C84FManifestError(f"target sample interface drift for {key}")
        if int(row["finite_value_flag"]) != 1:
            raise C84FManifestError(f"target trial contains non-finite data: {key}")
        if int(row["raw_input_bytes"]) <= 0:
            raise C84FManifestError(f"target raw-input byte identity is absent: {key}")
        _require_sha(row["raw_input_sha256"], f"{key}.raw_input_sha256")
    expected_sets = {key: set(value) for key, value in subjects.items()}
    if observed_subjects != expected_sets:
        raise C84FManifestError(
            f"target-subject coverage drift: observed={observed_subjects} expected={expected_sets}"
        )
    return {
        "complete": True,
        "trial_rows": len(values),
        "target_subjects": sum(len(value) for value in expected_sets.values()),
        "dataset_subject_counts": {key: len(value) for key, value in observed_subjects.items()},
        "target_label_fields": 0,
        "target_y_operations": 0,
    }


def publish_target_trial_registry(
    root: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    model_manifest_path: str | Path,
    model_manifest_sha_path: str | Path,
    execution_identity: Mapping[str, Any],
    scope: FieldScope = REAL_SCOPE,
    expected_subject_map: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    model_replay = verify_model_field_freeze(model_manifest_path, model_manifest_sha_path, scope=scope)
    values = [dict(row) for row in rows]
    gate = validate_target_trial_rows(values, expected_subject_map=expected_subject_map)
    if scope == REAL_SCOPE and gate["target_subjects"] != scope.subjects:
        raise C84FManifestError("target registry does not contain all 118 subjects")
    payload = {
        "schema_version": "c84f_target_unlabeled_trial_registry_v1",
        "execution_identity": dict(execution_identity),
        "model_field_manifest_sha256": model_replay["sha256"],
        "complete_gate": gate,
        "trials": values,
        "target_label_access": 0,
        "target_y_operations": [],
        "published_at_unix_ns": time.time_ns(),
    }
    directory = Path(root)
    path = directory / TARGET_REGISTRY_NAME
    sidecar = directory / TARGET_REGISTRY_SHA_NAME
    if path.exists() or sidecar.exists():
        raise C84FManifestError("target registry already exists; same-identity overwrite is forbidden")
    digest = write_json_atomic(path, payload)
    write_hash_sidecar(sidecar, digest)
    return {**payload, "path": str(path), "sha256": digest, "sha256_path": str(sidecar)}


def verify_target_trial_registry(
    path: str | Path,
    sha_path: str | Path,
    *,
    expected_subject_map: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    target = Path(path)
    expected = _sidecar_digest(Path(sha_path))
    if not target.is_file() or sha256_file(target) != expected:
        raise C84FManifestError("target-unlabeled registry identity replay failed")
    payload = read_json(target)
    if payload.get("schema_version") != "c84f_target_unlabeled_trial_registry_v1":
        raise C84FManifestError("target-unlabeled registry schema drift")
    gate = validate_target_trial_rows(payload.get("trials", ()), expected_subject_map=expected_subject_map)
    if gate != payload.get("complete_gate") or payload.get("target_y_operations") != []:
        raise C84FManifestError("target-unlabeled complete gate replay failed")
    return {"path": str(target), "sha256": expected, **gate}


def validate_field_descriptors(
    descriptors: Iterable[Mapping[str, Any]],
    *,
    operative_unit_ids: Sequence[str],
) -> dict[str, Any]:
    values = [dict(row) for row in descriptors]
    expected = set(map(str, operative_unit_ids))
    observed: set[str] = set()
    replayed_canary_contexts = 0
    for row in values:
        _require_exact_fields(row, FIELD_DESCRIPTOR_FIELDS, "field descriptor")
        unit_id = str(row["unit_id"])
        if unit_id in observed:
            raise C84FManifestError(f"duplicate field descriptor: {unit_id}")
        observed.add(unit_id)
        for field in ("checkpoint", "optimizer", "training_sidecar", "source_audit",
                      "complete_target_unlabeled", "target_context_index"):
            value = row[field]
            if not isinstance(value, Mapping) or not value.get("path") or not value.get("sha256"):
                raise C84FManifestError(f"field descriptor {unit_id}.{field} is incomplete")
            _require_sha(value["sha256"], f"{unit_id}.{field}.sha256")
        if row["interface_id"] != protocol.INTERFACE_ID:
            raise C84FManifestError(f"field descriptor interface drift: {unit_id}")
        witness = row["canary_subset_replay"]
        if not isinstance(witness, Mapping) or not bool(witness.get("required")):
            raise C84FManifestError(f"field descriptor canary replay contract absent: {unit_id}")
        if bool(witness.get("applicable")):
            if not bool(witness.get("passed")):
                raise C84FManifestError(f"canary subset replay failed: {unit_id}")
            replayed_canary_contexts += 1
    if observed != expected:
        raise C84FManifestError(
            f"field descriptor unit coverage drift: missing={len(expected-observed)} unknown={len(observed-expected)}"
        )
    return {
        "complete": True,
        "unit_descriptors": len(values),
        "unique_unit_ids": len(observed),
        "canary_unit_witnesses": replayed_canary_contexts,
    }


def publish_complete_field_manifest(
    root: str | Path,
    descriptors: Iterable[Mapping[str, Any]],
    *,
    operative_unit_ids: Sequence[str],
    model_manifest_path: str | Path,
    model_manifest_sha_path: str | Path,
    target_registry_path: str | Path,
    target_registry_sha_path: str | Path,
    instrumentation_summary: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    scope: FieldScope = REAL_SCOPE,
) -> dict[str, Any]:
    model = verify_model_field_freeze(model_manifest_path, model_manifest_sha_path, scope=scope)
    target = verify_target_trial_registry(target_registry_path, target_registry_sha_path)
    values = [dict(row) for row in descriptors]
    descriptor_gate = validate_field_descriptors(values, operative_unit_ids=operative_unit_ids)
    expected_summary = {
        "unit_artifacts": scope.units,
        "target_contexts": scope.contexts,
        "candidate_context_slices": scope.slices,
        "target_label_fields": 0,
        "target_y_operations": 0,
        "target_scientific_metrics": 0,
        "training_invocations": 0,
        "linear_replay_abs_tolerance": protocol.LINEAR_TOLERANCE,
        "strict_identity_abs_tolerance": protocol.STRICT_TOLERANCE,
        "canary_contexts_replayed": scope.canary_contexts,
        "canary_subset_replay_pass": True,
    }
    mismatch = {
        key: (instrumentation_summary.get(key), wanted)
        for key, wanted in expected_summary.items()
        if instrumentation_summary.get(key) != wanted
    }
    if mismatch:
        raise C84FManifestError(f"complete target instrumentation gate failed: {mismatch}")
    payload = {
        "schema_version": "c84f_complete_field_manifest_v1",
        "execution_identity": dict(execution_identity),
        "model_field_manifest_sha256": model["sha256"],
        "target_trial_registry_sha256": target["sha256"],
        "descriptor_gate": descriptor_gate,
        "instrumentation_gate": dict(instrumentation_summary),
        "field_descriptors": values,
        "target_construction_labels": 0,
        "target_evaluation_labels": 0,
        "same_label_oracle": 0,
        "selector_scores": 0,
        "scientific_statistics": 0,
        "gate": protocol.FIELD_GATE,
        "published_at_unix_ns": time.time_ns(),
    }
    directory = Path(root)
    path = directory / COMPLETE_MANIFEST_NAME
    sidecar = directory / COMPLETE_MANIFEST_SHA_NAME
    if path.exists() or sidecar.exists():
        raise C84FManifestError("complete field manifest already exists; overwrite is forbidden")
    digest = write_json_atomic(path, payload)
    write_hash_sidecar(sidecar, digest)
    return {**payload, "path": str(path), "sha256": digest, "sha256_path": str(sidecar)}


def atomic_publish_directory(staging: str | Path, final: str | Path) -> Path:
    source = Path(staging)
    destination = Path(final)
    if not source.is_dir() or not any(source.iterdir()):
        raise C84FManifestError("staging result directory is absent or empty")
    if destination.exists():
        raise C84FManifestError("final result directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    return destination


def validate_finite_error(value: Any, *, tolerance: float, name: str) -> float:
    observed = float(value)
    if not math.isfinite(observed) or observed < 0.0 or observed > tolerance:
        raise C84FManifestError(f"{name} exceeds its locked tolerance: {observed} > {tolerance}")
    return observed
