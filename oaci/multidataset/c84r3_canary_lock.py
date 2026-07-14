"""Generate the additive C84C V4 scope-specific execution lock (V3 identity)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from . import c84_dataset_registry_v2 as registry
from . import c84r_v2_protocols as candidate_protocol
from . import c84r2_canary_lock as prior_lock
from . import c84r2_v3_protocols as v3
from . import c84r3_canary_runtime_repair as runtime
from . import c84r3_v4_protocols as v4
from .c84r_montage_repair import CLASS_MAPPING_VERSION, EPOCH_RULE, INTERFACE_ID, MONTAGE_SHA256, UNIT_ID_SALT


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84r3_tables"
IMPLEMENTATION_COMMIT = "bf33ad635ba46cba38636a6a140f3f580f6dab78"
RUNTIME_REPAIR_COMMIT = "10c60d92f61dd091fef7a08f686a7ce85d99eb07"
CREATED_AT_UTC = "2026-07-14T00:32:00Z"


IMPLEMENTATION_PATHS = (
    "oaci/multidataset/c84c_real_canary_v3.py",
    "oaci/multidataset/c84r3_canary_runtime_repair.py",
    *tuple(path for path in prior_lock.IMPLEMENTATION_PATHS if path != "oaci/slurm_c84c_canary_v2.sh"),
    "oaci/slurm_c84c_canary_v3.sh",
)

REGISTRY_PATHS = (
    "oaci/reports/C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.json",
    "oaci/reports/C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.sha256",
    "oaci/reports/C84R2_CANARY_RUNTIME_AND_REPLAY_REPAIR_PROTOCOL.json",
    "oaci/reports/C84R2_CANARY_RUNTIME_AND_REPLAY_REPAIR_PROTOCOL.sha256",
    "oaci/reports/C84R3_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.json",
    "oaci/reports/C84R3_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.sha256",
    "oaci/reports/C84R3_PROTOCOL_TIMING_AUDIT.md",
    "oaci/reports/C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.json",
    "oaci/reports/C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.sha256",
    "oaci/reports/C84_CANARY_PROTOCOL_V4.json",
    "oaci/reports/C84_CANARY_PROTOCOL_V4.sha256",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V4.json",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V4.sha256",
    "oaci/reports/C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2.json",
    "oaci/reports/C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2.sha256",
    "oaci/reports/C84C_EXECUTION_LOCK_V2.json",
    "oaci/reports/C84C_EXECUTION_LOCK_V2.sha256",
    "oaci/reports/C84C_PI_AUTHORIZATION_RECORD_V2.json",
    "oaci/reports/C84C_FAILED_ATTEMPT_895366.json",
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
    "oaci/reports/c84r3_tables/repair_decision.csv",
    "oaci/reports/c84r3_tables/failed_attempt_ledger.csv",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: str | Path) -> str:
    return runtime.sha256_file(path)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()


def bind_path(path: str) -> dict[str, Any]:
    current = REPO_ROOT / path
    if not current.is_file():
        raise RuntimeError(f"C84C V3 lock path is absent: {path}")
    committed = subprocess.run(
        ["git", "show", f"{IMPLEMENTATION_COMMIT}:{path}"], cwd=REPO_ROOT,
        capture_output=True, check=True,
    ).stdout
    current_bytes = current.read_bytes()
    if committed != current_bytes:
        raise RuntimeError(f"C84C V3 lock path differs from protocol/implementation commit: {path}")
    return {
        "path": path,
        "sha256": hashlib.sha256(current_bytes).hexdigest(),
        "blob": _git("rev-parse", f"{IMPLEMENTATION_COMMIT}:{path}"),
        "bytes": len(current_bytes),
        "commit": IMPLEMENTATION_COMMIT,
    }


def protocol_binding(stem: str) -> dict[str, Any]:
    path = REPORT_DIR / f"{stem}.json"
    sidecar = REPORT_DIR / f"{stem}.sha256"
    digest = sidecar.read_text(encoding="ascii").split()[0]
    if sha256_file(path) != digest:
        raise RuntimeError(f"C84C V3 protocol hash replay failed: {stem}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256_path": str(sidecar.relative_to(REPO_ROOT)),
        "sha256": digest,
    }


def _csv_rows(relative: str) -> list[dict[str, str]]:
    with (REPO_ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_lock() -> dict[str, Any]:
    if _git("rev-parse", "HEAD") != IMPLEMENTATION_COMMIT:
        raise RuntimeError("C84C V3 lock must be generated at the committed V4 protocol base")
    implementation = [bind_path(path) for path in IMPLEMENTATION_PATHS]
    registries = [bind_path(path) for path in REGISTRY_PATHS]
    all_bound = implementation + registries
    if len({row["path"] for row in all_bound}) != len(all_bound):
        raise RuntimeError("C84C V3 runtime-bound registry contains duplicate paths")

    canary_units = [row for row in candidate_protocol.candidate_units() if row["canary_subset"]]
    canary_digest = runtime.canary_unit_digest(row["unit_id"] for row in canary_units)
    subject_rows = _csv_rows("oaci/reports/c84r2_tables/exact_subject_identity_contract.csv")
    loader_rows = _csv_rows("oaci/reports/c84r2_tables/loader_source_identity_registry.csv")
    if len(subject_rows) != 3 or not all(row["sets_disjoint"] == "1" for row in subject_rows):
        raise RuntimeError("C84C V3 subject identity registry is incomplete")
    if len(loader_rows) != 4 or not all(row["verified_before_get_data"] == "1" for row in loader_rows):
        raise RuntimeError("C84C V3 loader identity registry is incomplete")

    montage_repair = protocol_binding("C84R_COMMON_MONTAGE_REPAIR_PROTOCOL")
    runtime_repair = protocol_binding("C84R2_CANARY_RUNTIME_AND_REPLAY_REPAIR_PROTOCOL")
    float_repair = protocol_binding("C84R3_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL")
    external = protocol_binding("C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2")
    canary = protocol_binding("C84_CANARY_PROTOCOL_V4")
    field = protocol_binding("C84_FIELD_GENERATION_PROTOCOL_V4")
    science = protocol_binding("C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2")
    protocol_bindings = [montage_repair, runtime_repair, float_repair, external, canary, field, science]

    return {
        "schema_version": "c84c_execution_lock_v3",
        "milestone": "C84C",
        "status": runtime.LOCK_READY_STATUS,
        "created_at_utc": CREATED_AT_UTC,
        "chronology": {
            "C84R3_repair_protocol_commit": v4.REPAIR_PROTOCOL_COMMIT,
            "C84R3_runtime_implementation_commit": RUNTIME_REPAIR_COMMIT,
            "C84R3_V4_protocol_commit": IMPLEMENTATION_COMMIT,
            "repair_precedes_implementation": True,
            "implementation_precedes_V4_protocols": True,
            "V4_protocols_precede_execution_lock": True,
            "prior_real_EEG_access_job": 895366,
            "prior_target_y_access": 0,
            "prior_target_scientific_metrics": 0,
            "replacement_real_EEG_access_before_lock": 0,
            "replacement_training_forward_GPU_before_lock": 0,
        },
        "historical_lock_supersession": {
            "path": "oaci/reports/C84C_EXECUTION_LOCK_V2.json",
            "sha256": v4.HISTORICAL_LOCK_V2_SHA256,
            "commit": "270fbb0d9f47f9bf6a2888ee58fd7ca6eadff0ea",
            "authorization_consumed_by_job": 895366,
            "preserved": True,
            "operative_for_execution": False,
        },
        "montage_repair_protocol": montage_repair,
        "runtime_repair_protocol": runtime_repair,
        "repair_protocol": float_repair,
        "external_protocol": external,
        "canary_protocol": canary,
        "future_field_protocol": field,
        "future_scientific_protocol": science,
        "protocol_bindings": protocol_bindings,
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
            "source_panel": "A",
            "training_seed": 5,
            "level": 0,
            "regimes": ["ERM", "OACI", "SRC"],
            "units_per_dataset": 81,
            "total_units": 243,
            "training_phases": 9,
            "engineering_only": True,
            "retrain_all_units": True,
            "C84F": False,
            "C84S": False,
        },
        "candidate_identity": {
            "function_path": "oaci/multidataset/c84r_v2_protocols.py",
            "salt": UNIT_ID_SALT,
            "complete_unit_count": 1944,
            "canary_unit_count": 243,
            "canary_unit_ids_sha256": canary_digest,
            "failed_attempt_unit_IDs_recomputed": False,
            "failed_attempt_artifacts_reused": False,
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
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "CUDA_allocation": "Slurm_V100_one_GPU_after_fresh_authorization",
        },
        "loader_source_identity": {
            "files": [{
                "qualified_object": row["qualified_object"],
                "distribution": row["distribution"],
                "distribution_relative_path": row["distribution_relative_path"],
                "sha256": row["sha256"],
            } for row in loader_rows],
            "verified_after_authorization_consumption": True,
            "verified_before_dataset_loader_class_import": True,
            "verified_before_download_and_get_data": True,
        },
        "implementation": {
            "commit": IMPLEMENTATION_COMMIT,
            "runtime_repair_commit": RUNTIME_REPAIR_COMMIT,
            "entrypoint": "python -m oaci.multidataset.c84c_real_canary_v3 run-real",
            "slurm_entrypoint": "oaci/slurm_c84c_canary_v3.sh",
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
            "linear_z_classifier_logits_abs_tolerance": runtime.LINEAR_REPLAY_ABS_TOLERANCE,
            "softmax_repeat_logits_repeat_z_abs_tolerance": runtime.STRICT_IDENTITY_ABS_TOLERANCE,
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
            "failed_root": "/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v3/lock_2e38dcd63c02a887b1dc",
            "failed_root_reusable": False,
            "resource_estimate_path": "oaci/reports/c84r_tables/resource_estimate.csv",
            "resource_estimate_sha256": sha256_file(REPORT_DIR / "c84r_tables/resource_estimate.csv"),
            "GPU_phase_hours_max": 250,
            "external_payload_TiB_max": 2.0,
            "Git_file_MiB_max": 50,
        },
        "authorization": {
            "record_path": "oaci/reports/C84C_PI_AUTHORIZATION_RECORD_V3.json",
            "record_present_at_lock": False,
            "direct_PI_statement_required": "授权 C84C",
            "magic_token_required": False,
            "hash_recital_required": False,
            "fresh_binding_required": True,
            "historical_authorization_reusable": False,
            "failed_authorization_reused": False,
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
            "failed_job_895366_artifact_reuse": False,
            "retrain_all_243_units": True,
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
            "prior_job_895366_real_EEG_views": 3,
            "prior_job_895366_source_label_arrays": 2,
            "prior_job_895366_target_y_access": 0,
            "prior_job_895366_target_scientific_metrics": 0,
            "prior_job_895366_complete_units": 0,
            "replacement_real_EEG_arrays_loaded": 0,
            "replacement_real_labels_read": 0,
            "replacement_training_forward_GPU_jobs": 0,
            "replacement_candidate_units_created": 0,
            "replacement_authorization_record_present": False,
            "C84F_execution_lock_present": False,
            "C84S_execution_lock_present": False,
        },
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = list(values[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def generate() -> dict[str, Any]:
    lock = build_lock()
    path = REPORT_DIR / "C84C_EXECUTION_LOCK_V3.json"
    path.write_bytes(canonical_bytes(lock) + b"\n")
    digest = sha256_file(path)
    (REPORT_DIR / "C84C_EXECUTION_LOCK_V3.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii",
    )
    write_csv(TABLE_DIR / "runtime_bound_object_registry_v3.csv", [{
        "path": row["path"],
        "object_class": "implementation" if row["path"] in IMPLEMENTATION_PATHS else "registry",
        "commit": row["commit"],
        "blob": row["blob"],
        "sha256": row["sha256"],
        "bytes": row["bytes"],
        "runtime_replay_required": 1,
    } for row in lock["runtime_bound_objects"]])
    return {
        "execution_lock_v3_sha256": digest,
        "runtime_bound_objects": lock["runtime_bound_object_count"],
        "implementation_files": len(IMPLEMENTATION_PATHS),
        "registry_files": len(REGISTRY_PATHS),
        "canary_unit_ids_sha256": lock["candidate_identity"]["canary_unit_ids_sha256"],
        "replacement_real_EEG_access": 0,
        "replacement_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
