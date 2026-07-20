"""Generate the C84C V3 scope-specific execution lock (V2 identity)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from . import c84_dataset_registry_v2 as registry
from . import c84r_v2_protocols as candidate_protocol
from . import c84r2_canary_runtime_repair as runtime
from . import c84r2_v3_protocols as v3
from .c84r_montage_repair import CLASS_MAPPING_VERSION, EPOCH_RULE, INTERFACE_ID, MONTAGE_SHA256, UNIT_ID_SALT


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84r2_tables"
IMPLEMENTATION_COMMIT = "ddaa6d4531f13922481f53b827f13e62280d7968"
CREATED_AT_UTC = "2026-07-13T23:49:00Z"


IMPLEMENTATION_PATHS = (
    "oaci/multidataset/c84c_real_canary_v2.py",
    "oaci/multidataset/c84r2_canary_runtime_repair.py",
    "oaci/multidataset/c84c_real_canary.py",
    "oaci/multidataset/c84_dataset_registry_v2.py",
    "oaci/multidataset/c84r_v2_protocols.py",
    "oaci/multidataset/c84r_montage_repair.py",
    "oaci/multidataset/c84_dataset_registry.py",
    "oaci/multidataset/c84_fixed_zoo_protocol.py",
    "oaci/slurm_c84c_canary_v2.sh",
    "oaci/models/__init__.py",
    "oaci/models/factory.py",
    "oaci/models/shallow.py",
    "oaci/models/output.py",
    "oaci/support_graph.py",
    "oaci/config.py",
    "oaci/data/__init__.py",
    "oaci/data/plan_materialize.py",
    "oaci/data/plan_sampler.py",
    "oaci/methods/__init__.py",
    "oaci/methods/oaci.py",
    "oaci/methods/source_robust.py",
    "oaci/train/__init__.py",
    "oaci/train/batch_plan.py",
    "oaci/train/bn.py",
    "oaci/train/checkpoint.py",
    "oaci/train/data.py",
    "oaci/train/engine.py",
    "oaci/train/evaluate.py",
    "oaci/train/objective.py",
    "oaci/train/risk.py",
    "oaci/train/rng.py",
)


REGISTRY_PATHS = (
    "oaci/reports/C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.json",
    "oaci/reports/C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.sha256",
    "oaci/reports/C84R2_CANARY_RUNTIME_AND_REPLAY_REPAIR_PROTOCOL.json",
    "oaci/reports/C84R2_CANARY_RUNTIME_AND_REPLAY_REPAIR_PROTOCOL.sha256",
    "oaci/reports/C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.json",
    "oaci/reports/C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.sha256",
    "oaci/reports/C84_CANARY_PROTOCOL_V3.json",
    "oaci/reports/C84_CANARY_PROTOCOL_V3.sha256",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V3.json",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V3.sha256",
    "oaci/reports/C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2.json",
    "oaci/reports/C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2.sha256",
    "oaci/reports/c84p_tables/subject_partition_registry.csv",
    "oaci/reports/c84p_tables/selector_registry_replay.csv",
    "oaci/reports/c84r_tables/subject_partition_identity_replay.csv",
    "oaci/reports/c84r_tables/resource_estimate.csv",
    "oaci/reports/c84r_tables/candidate_unit_id_migration.csv",
    "oaci/reports/c84r_tables/channel_order_registry.csv",
    "oaci/reports/c84r_tables/dataset_registry_v2_replay.csv",
    "oaci/reports/c84r2_tables/environment_version_registry.csv",
    "oaci/reports/c84r2_tables/loader_source_identity_registry.csv",
    "oaci/reports/c84r2_tables/exact_subject_identity_contract.csv",
    "oaci/reports/c84r2_tables/actual_epoch_interface_contract.csv",
    "oaci/reports/c84r2_tables/source_audit_instrumentation_contract.csv",
    "oaci/reports/c84r2_tables/target_unlabeled_instrumentation_contract.csv",
    "oaci/reports/c84r2_tables/persisted_artifact_replay_contract.csv",
    "oaci/reports/c84r2_tables/optimizer_replay_contract.csv",
    "oaci/reports/c84r2_tables/deterministic_prefix_contract.csv",
    "oaci/reports/c84r2_tables/attempt_ledger_contract.csv",
    "oaci/reports/c84r2_tables/canary_complete_gate.csv",
    "oaci/reports/c84r2_tables/runtime_hash_replay_fixture.csv",
    "oaci/reports/c84r2_tables/synthetic_calibration.csv",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: str | Path) -> str:
    return runtime.sha256_file(path)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()


def _blob(commit: str, path: str) -> str:
    return _git("rev-parse", f"{commit}:{path}")


def bind_path(path: str) -> dict[str, Any]:
    current = REPO_ROOT / path
    if not current.is_file():
        raise RuntimeError(f"C84C V2 lock path is absent: {path}")
    committed = subprocess.run(
        ["git", "show", f"{IMPLEMENTATION_COMMIT}:{path}"], cwd=REPO_ROOT,
        capture_output=True, check=True,
    ).stdout
    current_bytes = current.read_bytes()
    if committed != current_bytes:
        raise RuntimeError(f"C84C V2 lock path differs from implementation commit: {path}")
    return {
        "path": path,
        "sha256": hashlib.sha256(current_bytes).hexdigest(),
        "blob": _blob(IMPLEMENTATION_COMMIT, path),
        "bytes": len(current_bytes),
        "commit": IMPLEMENTATION_COMMIT,
    }


def protocol_binding(stem: str) -> dict[str, Any]:
    path = REPORT_DIR / f"{stem}.json"
    sidecar = REPORT_DIR / f"{stem}.sha256"
    digest = sidecar.read_text(encoding="ascii").split()[0]
    if sha256_file(path) != digest:
        raise RuntimeError(f"C84C V2 protocol hash replay failed: {stem}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256_path": str(sidecar.relative_to(REPO_ROOT)),
        "sha256": digest,
    }


def _loader_rows() -> list[dict[str, Any]]:
    path = TABLE_DIR / "loader_source_identity_registry.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4 or not all(row["verified_before_get_data"] == "1" for row in rows):
        raise RuntimeError("C84C loader identity registry is incomplete")
    return [{
        "qualified_object": row["qualified_object"],
        "distribution": row["distribution"],
        "distribution_relative_path": row["distribution_relative_path"],
        "sha256": row["sha256"],
    } for row in rows]


def _subject_rows() -> list[dict[str, Any]]:
    path = TABLE_DIR / "exact_subject_identity_contract.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3 or not all(row["sets_disjoint"] == "1" for row in rows):
        raise RuntimeError("C84C exact subject registry is incomplete")
    return rows


def build_lock() -> dict[str, Any]:
    if _git("rev-parse", "HEAD") != IMPLEMENTATION_COMMIT:
        raise RuntimeError("C84C V2 lock must be generated at the implementation/calibration commit")
    implementation = [bind_path(path) for path in IMPLEMENTATION_PATHS]
    registries = [bind_path(path) for path in REGISTRY_PATHS]
    all_bound = implementation + registries
    if len({row["path"] for row in all_bound}) != len(all_bound):
        raise RuntimeError("C84C V2 runtime-bound registry contains duplicate paths")
    canary_units = [row for row in candidate_protocol.candidate_units() if row["canary_subset"]]
    canary_digest = runtime.canary_unit_digest(row["unit_id"] for row in canary_units)
    canary = protocol_binding("C84_CANARY_PROTOCOL_V3")
    field = protocol_binding("C84_FIELD_GENERATION_PROTOCOL_V3")
    repair = protocol_binding("C84R2_CANARY_RUNTIME_AND_REPLAY_REPAIR_PROTOCOL")
    external = protocol_binding("C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2")
    science = protocol_binding("C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2")
    montage_repair = protocol_binding("C84R_COMMON_MONTAGE_REPAIR_PROTOCOL")
    subject_rows = _subject_rows()
    return {
        "schema_version": "c84c_execution_lock_v2",
        "milestone": "C84C",
        "status": runtime.LOCK_READY_STATUS,
        "created_at_utc": CREATED_AT_UTC,
        "chronology": {
            "C84R_HEAD": v3.HISTORICAL_HEAD,
            "C84R2_repair_protocol_commit": v3.REPAIR_PROTOCOL_COMMIT,
            "C84R2_implementation_and_protocol_commit": IMPLEMENTATION_COMMIT,
            "repair_protocol_precedes_implementation": True,
            "implementation_precedes_lock": True,
            "real_EEG_access_before_lock": 0,
            "real_labels_before_lock": 0,
            "dataset_downloads_before_lock": 0,
            "training_forward_GPU_before_lock": 0,
        },
        "historical_lock_supersession": {
            "path": "oaci/reports/C84C_EXECUTION_LOCK.json",
            "sha256": v3.HISTORICAL_LOCK_SHA256,
            "preserved": True,
            "operative_for_execution": False,
        },
        "montage_repair_protocol": montage_repair,
        "repair_protocol": repair,
        "external_protocol": external,
        "canary_protocol": canary,
        "future_field_protocol": field,
        "future_scientific_protocol": science,
        "protocol_bindings": [montage_repair, repair, external, canary, field, science],
        "interface": {
            "id": INTERFACE_ID,
            "schema": registry.SCHEMA_VERSION,
            "channels": list(registry.PRIMARY_CHANNELS),
            "channel_count": 20,
            "montage_sha256": MONTAGE_SHA256,
            "epoch_rule": EPOCH_RULE,
            "sample_rate_hz": 160,
            "input_shape": [20, 480],
            "class_mapping_version": CLASS_MAPPING_VERSION,
            "actual_Epochs_interface_required": True,
            "Fz_substitution": False,
            "FCz_interpolation": False,
            "zero_fill": False,
            "dataset_specific_mask": False,
        },
        "scope": {
            "datasets": list(candidate_protocol.DATASET_ORDER),
            "source_panel": "A", "training_seed": 5, "level": 0,
            "regimes": ["ERM", "OACI", "SRC"],
            "units_per_dataset": 81, "total_units": 243, "training_phases": 9,
            "engineering_only": True, "C84F": False, "C84S": False,
        },
        "candidate_identity": {
            "function_path": "oaci/multidataset/c84r_v2_protocols.py",
            "salt": UNIT_ID_SALT,
            "complete_unit_count": 1944,
            "canary_unit_count": 243,
            "canary_unit_ids_sha256": canary_digest,
            "historical_blocked_unit_ID_reuse": 0,
            "C84C_C84F_identity_function_same": True,
        },
        "subject_identity": {
            "contract_path": "oaci/reports/c84r2_tables/exact_subject_identity_contract.csv",
            "datasets": subject_rows,
            "exact_loaded_sets_required": True,
            "persist_lists_and_trial_counts": True,
        },
        "environment": {
            "conda_prefix": v3.ENV_PREFIX,
            "python": "3.13.7",
            "distributions": {"torch": "2.6.0", "moabb": "1.5.0", "mne": "1.11.0", "chardet": "5.2.0"},
            "runtime_versions": {"torch": "2.6.0+cu124", "moabb": "1.5.0", "mne": "1.11.0"},
            "historical_python_3_9_25_compatible": False,
            "historical_incompatibility": "MOABB_and_MNE_require_Python_at_least_3.10",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "CUDA_allocation": "Slurm_V100_one_GPU_after_authorization",
        },
        "loader_source_identity": {
            "files": _loader_rows(),
            "verified_after_authorization_consumption": True,
            "verified_before_dataset_loader_class_import": True,
            "verified_before_download_and_get_data": True,
        },
        "implementation": {
            "commit": IMPLEMENTATION_COMMIT,
            "entrypoint": "python -m oaci.multidataset.c84c_real_canary_v2 run-real",
            "slurm_entrypoint": "oaci/slurm_c84c_canary_v2.sh",
            "files": implementation,
            "runtime_transitive_files": len(implementation),
            "all_files_replayed_before_output_root": True,
            "historical_ERM_OACI_SRC_formulas_unchanged": True,
        },
        "runtime_bound_objects": all_bound,
        "runtime_bound_object_count": len(all_bound),
        "runtime_replay": {
            "SHA256_for_every_object": True,
            "Git_blob_for_every_object": True,
            "HEAD_blob_matches_lock": True,
            "protocol_sidecars": True,
            "montage_order_digest": True,
            "candidate_ID_digest": True,
            "clean_HEAD_equals_origin_oaci": True,
            "before_authorization_consumption": True,
            "before_output_root": True,
        },
        "views": {
            "source_training_view": {"X": True, "y": True, "training": True},
            "source_audit_view": {"X": True, "y": True, "training": False, "metrics": False},
            "target_unlabeled_view": {"X": True, "y": False, "target_y_slot_consumed": False},
            "target_construction_view": {"provisioned": False},
            "target_evaluation_view": {"provisioned": False},
            "same_label_oracle_view": {"reachable": False},
        },
        "instrumentation": {
            "checkpoint_state_sidecar_units": 243,
            "strict_source_audit_artifacts": 243,
            "target_unlabeled_artifacts": 243,
            "source_fields": sorted(runtime.SOURCE_AUDIT_FIELDS),
            "target_fields": sorted(runtime.TARGET_UNLABELED_FIELDS),
            "persisted_reload_required": True,
            "deterministic_prefix_per_dataset": True,
        },
        "complete_gate": {
            "checkpoint_replay_units": 243,
            "optimizer_replay_units": 243,
            "sidecar_replay_units": 243,
            "source_audit_replay_units": 243,
            "target_unlabeled_replay_units": 243,
            "partial_completion_reusable": False,
        },
        "attempt_ledger": {
            "created_after_authorization_consumption_before_import": True,
            "package_import_CUDA_loader_data_training_instrumentation_manifest_wrapped": True,
            "access_counters": True,
            "partial_artifact_manifest": True,
            "retry_disposition": "NEW_ADDITIVE_REPAIR_AND_LOCK_REQUIRED",
        },
        "runtime": {
            "external_root": str(runtime.DEFAULT_EXTERNAL_ROOT),
            "content_addressed_subroot": "lock_<execution_lock_sha256_prefix20>",
            "resource_estimate_path": "oaci/reports/c84r_tables/resource_estimate.csv",
            "resource_estimate_sha256": sha256_file(REPORT_DIR / "c84r_tables/resource_estimate.csv"),
            "GPU_phase_hours_max": 250,
            "external_payload_TiB_max": 2.0,
            "Git_file_MiB_max": 50,
        },
        "authorization": {
            "record_path": "oaci/reports/C84C_PI_AUTHORIZATION_RECORD_V2.json",
            "record_present_at_lock": False,
            "direct_PI_statement_required": "\u6388\u6743 C84C",
            "magic_token_required": False,
            "hash_recital_required": False,
            "fresh_binding_required": True,
            "historical_authorization_reusable": False,
        },
        "report_schema": {
            "authorization_consumed": "authorization_consumed.json",
            "execution_attempts": "execution_attempts.jsonl",
            "partial_artifact_manifest": "partial_artifact_manifest.json",
            "complete_manifest": "C84C_COMPLETE_CANARY_MANIFEST.json",
            "candidate_sidecars": "<dataset>/sidecars/<unit_id>.json",
            "checkpoints": "<dataset>/checkpoints/<unit_id>.pt",
            "optimizer_states": "<dataset>/optimizer_states/*.pt",
            "source_audit_instrumentation": "<dataset>/source_audit/<unit_id>.npz",
            "target_unlabeled_instrumentation": "<dataset>/target_unlabeled/<unit_id>.npz",
            "scientific_result_table": None,
        },
        "retry_policy": {
            "all_attempts_preserved": True,
            "silent_rerun": False,
            "target_outcome_decisions": 0,
            "post_consumption_failure_requires_additive_repair_and_new_lock": True,
        },
        "forbidden": {
            "target_accuracy_calibration_regret_or_label_counts": True,
            "selector_scores_Q1_Q2_budget_frontier": True,
            "construction_or_evaluation_labels": True,
            "same_label_oracle": True,
            "C84F_or_C84S_execution_or_lock": True,
            "target_outcome_retention_or_retry": True,
            "raw_EEG_weights_optimizer_states_or_caches_in_Git": True,
        },
        "protected_state_at_lock": {
            "real_EEG_arrays_loaded": 0,
            "real_labels_read": 0,
            "dataset_downloads": 0,
            "training_forward_GPU_jobs": 0,
            "candidate_units_created": 0,
            "source_audit_artifacts_created": 0,
            "target_unlabeled_artifacts_created": 0,
            "target_scientific_metrics": 0,
            "authorization_record_present": False,
            "C84F_execution_lock_present": False,
            "C84S_execution_lock_present": False,
        },
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = list(values[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def generate() -> dict[str, Any]:
    lock = build_lock()
    path = REPORT_DIR / "C84C_EXECUTION_LOCK_V2.json"
    path.write_bytes(canonical_bytes(lock) + b"\n")
    digest = sha256_file(path)
    sha_path = REPORT_DIR / "C84C_EXECUTION_LOCK_V2.sha256"
    sha_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    write_csv(TABLE_DIR / "runtime_bound_object_registry.csv", [{
        "path": row["path"], "object_class": "implementation" if row["path"] in IMPLEMENTATION_PATHS else "registry",
        "commit": row["commit"], "blob": row["blob"], "sha256": row["sha256"],
        "bytes": row["bytes"], "runtime_replay_required": 1,
    } for row in lock["runtime_bound_objects"]])
    return {
        "execution_lock_v2_sha256": digest,
        "runtime_bound_objects": lock["runtime_bound_object_count"],
        "implementation_files": len(IMPLEMENTATION_PATHS),
        "registry_files": len(REGISTRY_PATHS),
        "canary_unit_ids_sha256": lock["candidate_identity"]["canary_unit_ids_sha256"],
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
        "dataset_downloads": 0,
        "authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
