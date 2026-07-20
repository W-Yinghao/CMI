"""Generate additive C84 canary/field V3 protocols and no-data registries."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from . import c84_dataset_registry_v2 as registry
from . import c84r_v2_protocols as v2
from . import c84r2_canary_runtime_repair as runtime
from .c84r_montage_repair import EPOCH_RULE, INTERFACE_ID, MONTAGE_SHA256


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84r2_tables"
REPAIR_PROTOCOL_COMMIT = "6c7e59f907431e073b2f8e580c4f25cb9e052a50"
REPAIR_PROTOCOL_SHA256 = "ff7c01f1760aaa19f2019672ad5426c8c28eba6c7071cd3b78b39bfe69dc8874"
IMPLEMENTATION_COMMIT = "4eb21dd75d7fc932ac804b9c3cb4df532934d224"
HISTORICAL_HEAD = "2fc5e797119ce1defc5e24c9063bb103b219a705"
HISTORICAL_LOCK_SHA256 = "f9cabf8f362917d663e13154910085d5b105740b265789a2323dd7bc0193222b"
HISTORICAL_CANARY_SHA256 = "f8e265f0969b9343526c4f6e09fef145d64149d159ea79b803cc983ae2761988"
EXTERNAL_V2_SHA256 = "522e6fe8372f8c73741ed146a27068076db8c3d7087f4c4a36760fe0328b7c2f"
FIELD_V2_SHA256 = "b6ecd3fb5cc2f1ded872cefad42ca38c172696e4169efc58a86a5a3a90395b62"
SCIENCE_V2_SHA256 = "dc33b22527352bd42989c26f6771b4a49dc1443d458962587ca3d70ad76dd631"
ENV_PREFIX = "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact"
CREATED_AT_UTC = "2026-07-13T23:31:00Z"
SUCCESS_GATE = "C84C_RUNTIME_LOCK_AND_COMPLETE_ENGINEERING_REPLAY_READY_FOR_PI_AUTHORIZATION"


LOADER_SOURCES = (
    ("moabb.datasets.Lee2019_MI", "moabb/datasets/Lee2019.py",
     "a0234b81923fed15e4a221e011399f76a83873cd43d598ad5c8c71ba54678a6f"),
    ("moabb.datasets.Cho2017", "moabb/datasets/gigadb.py",
     "42e2ef372762cb86aab11a886e1707675477ac776e0468448233de7a4ba71e32"),
    ("moabb.datasets.PhysionetMI", "moabb/datasets/physionet_mi.py",
     "a8abe8097870d804a2d78f500f3c6820962c1c3402f53368e92e7a91068b84ba"),
    ("moabb.paradigms.MotorImagery", "moabb/paradigms/motor_imagery.py",
     "f941a3f17c1bca4211045c28f7df3704c9d428ef689dff2410a478b5bf68651e"),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload) + b"\n")


def write_sha(path: Path, digest: str, target: str) -> None:
    path.write_text(f"{digest}  {target}\n", encoding="ascii")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise RuntimeError(f"refusing empty C84R2 table: {path}")
    fields = list(values[0])
    if any(set(row) != set(fields) for row in values):
        raise RuntimeError(f"C84R2 table schema mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def _exact_environment() -> dict[str, Any]:
    script = (
        "import importlib.metadata as m,json,sys,torch,moabb,mne;"
        "print(json.dumps({'prefix':sys.prefix,'python':sys.version.split()[0],"
        "'distributions':{'torch':m.version('torch'),'moabb':m.version('moabb'),'mne':m.version('mne'),"
        "'chardet':m.version('chardet')},"
        "'runtime':{'torch':torch.__version__,'moabb':moabb.__version__,'mne':mne.__version__}}))"
    )
    completed = subprocess.run(
        [f"{ENV_PREFIX}/bin/python", "-c", script], text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout.strip())
    expected = {
        "prefix": ENV_PREFIX,
        "python": "3.13.7",
        "distributions": {"torch": "2.6.0", "moabb": "1.5.0", "mne": "1.11.0", "chardet": "5.2.0"},
        "runtime": {"torch": "2.6.0+cu124", "moabb": "1.5.0", "mne": "1.11.0"},
    }
    if payload != expected:
        raise RuntimeError(f"C84C exact environment replay failed: {payload}")
    return payload


def _verify_loader_sources() -> list[dict[str, Any]]:
    rows = []
    base = Path(ENV_PREFIX) / "lib/python3.13/site-packages"
    for qualified, relative, digest in LOADER_SOURCES:
        path = base / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"C84C loader source replay failed: {qualified}")
        rows.append({
            "qualified_object": qualified,
            "distribution": "moabb",
            "distribution_relative_path": relative,
            "sha256": digest,
            "verified_before_get_data": 1,
            "real_data_access": 0,
        })
    return rows


def build_canary_protocol() -> dict[str, Any]:
    environment = _exact_environment()
    loader_rows = _verify_loader_sources()
    return {
        "schema_version": "c84_canary_protocol_v3",
        "milestone": "C84C",
        "status": "LOCKED_PROTOCOL_RUNTIME_LOCK_REQUIRED_NOT_AUTHORIZED",
        "created_at_utc": CREATED_AT_UTC,
        "supersession": {
            "historical_C84C_protocol_V2_sha256": HISTORICAL_CANARY_SHA256,
            "historical_C84C_lock_sha256": HISTORICAL_LOCK_SHA256,
            "historical_objects_rewritten": False,
            "historical_objects_execution_authority": False,
            "repair_protocol_commit": REPAIR_PROTOCOL_COMMIT,
            "repair_protocol_sha256": REPAIR_PROTOCOL_SHA256,
        },
        "parent_external_protocol": {
            "path": "oaci/reports/C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.json",
            "sha256": EXTERNAL_V2_SHA256,
            "scientific_interface_changed": False,
        },
        "scope": {
            "datasets": list(v2.DATASET_ORDER), "source_panel": "A", "training_seed": 5, "level": 0,
            "target_subjects": v2.CANARY_TARGETS, "units_per_dataset": 81,
            "total_units": 243, "training_phases": 9, "engineering_only": True,
        },
        "interface": {
            "id": INTERFACE_ID, "channels": list(registry.PRIMARY_CHANNELS),
            "montage_sha256": MONTAGE_SHA256, "epoch_rule": EPOCH_RULE,
            "sample_rate_hz": 160, "final_input_shape": [20, 480],
            "actual_Epochs_ch_names_required": True, "actual_Epochs_sfreq_required": True,
            "persist_pre_half_open_n_times": True, "persist_first_last_time": True,
            "bad_channel_interpolation_or_synthesis": False,
        },
        "exact_subject_identity": {
            "source_training_per_dataset": 12, "source_audit_per_dataset": 4,
            "target_per_dataset": 1, "exact_sets_required": True, "sets_disjoint": True,
            "persist_lists_and_trial_counts": True,
        },
        "environment": {
            **environment,
            "historical_python_3_9_25_replayable": False,
            "historical_incompatibility_reason": "MOABB_1.5.0_and_MNE_1.11.0_require_Python_at_least_3.10",
            "environment_identity_repaired_before_real_access": True,
        },
        "loader_source_identity": {"files": loader_rows, "before_download_and_get_data": True},
        "runtime_replay": {
            "all_lock_bound_files_by_SHA256_and_Git_blob": True,
            "protocol_sidecars": True, "dataset_and_table_registries": True,
            "montage_and_candidate_ID_digest": True, "clean_HEAD_equals_origin_oaci": True,
            "before_authorization_consumption": True, "before_output_root": True,
        },
        "views": {
            "source_training": {"X": True, "y": True, "training": True},
            "source_audit": {"X": True, "y": True, "training": False, "metrics": False},
            "target_unlabeled": {"X": True, "y": False, "structural_y_slot_consumed": False},
            "target_construction": {"provisioned": False},
            "target_evaluation": {"provisioned": False},
            "same_label_oracle": {"reachable": False},
        },
        "instrumentation": {
            "checkpoint_state_sidecar_units": 243,
            "strict_source_audit_artifacts": 243,
            "target_unlabeled_artifacts": 243,
            "source_fields": sorted(runtime.SOURCE_AUDIT_FIELDS),
            "target_fields": sorted(runtime.TARGET_UNLABELED_FIELDS),
            "source_metrics_reported": False, "target_metrics_reported": False,
        },
        "persisted_replay": {
            "checkpoint_SHA_and_state_schema": True,
            "optimizer_load_labels_and_step_counts": True,
            "sidecar_exact_schema_and_identity": True,
            "saved_source_softmax": True, "saved_target_softmax": True,
            "saved_z_classifier_logits": True, "saved_repeat_logits_z": True,
            "trial_IDs_and_shapes": True, "genealogy": True,
        },
        "deterministic_prefix": {
            "per_dataset": True, "full_duplicate_training": False,
            "repeat_count": 2, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0",
            "model_init_plan_first_batch_and_short_prefix_hashes": True,
        },
        "attempt_ledger": {
            "created_immediately_after_authorization_consumption": True,
            "before_package_imports_and_CUDA": True,
            "all_post_consumption_stages_wrapped": True,
            "access_counters": True, "partial_artifact_manifest": True,
            "retry_disposition": "NEW_ADDITIVE_REPAIR_AND_LOCK_REQUIRED",
        },
        "authorization": {
            "fresh_direct_PI_authorization_required": True,
            "shortest_statement": "\u6388\u6743 C84C",
            "magic_token_required": False, "historical_authorization_reusable": False,
            "record_schema": "c84c_direct_pi_authorization_record_v2",
        },
        "forbidden_outputs": [
            "target_accuracy", "target_calibration", "target_regret", "target_label_counts",
            "selector_scores", "Q1", "Q2", "label_budget_frontier", "cross_dataset_science",
        ],
        "protected_state": {
            "real_EEG_arrays_loaded": 0, "real_labels_read": 0, "dataset_downloads": 0,
            "training_forward_GPU_jobs": 0, "candidate_units_created": 0,
        },
    }


def build_field_protocol(canary_sha256: str) -> dict[str, Any]:
    historical = json.loads((REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V2.json").read_text())
    return {
        **{key: value for key, value in historical.items()
           if key not in {"schema_version", "status", "parent_external_protocol_sha256",
                          "fresh_direct_PI_authorization_after_canary_review",
                          "scope_specific_execution_lock_created_in_C84R"}},
        "schema_version": "c84_field_generation_protocol_v3",
        "status": "LOCKED_PROTOCOL_ONLY_NO_EXECUTION_LOCK_NOT_AUTHORIZED",
        "parent_external_protocol_sha256": EXTERNAL_V2_SHA256,
        "parent_canary_protocol_v3_sha256": canary_sha256,
        "historical_field_protocol_v2_sha256": FIELD_V2_SHA256,
        "scientific_field_scope_changed": False,
        "canary_reuse": {
            "checkpoint_and_optimizer_reusable_only_after_complete_replay": True,
            "source_audit_and_target_unlabeled_reusable_only_when_both_complete_and_bound": True,
            "complete_units_required": 243,
            "strict_source_artifacts_required": 243,
            "target_unlabeled_artifacts_required": 243,
            "missing_instrumentation_requires_new_field_lock": True,
            "missing_instrumentation_may_be_added_without_retraining": True,
            "outcome_driven_choice": False,
        },
        "fresh_direct_PI_authorization_after_canary_review": True,
        "scope_specific_execution_lock_created_in_C84R2": False,
    }


def subject_contract_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in v2.DATASET_ORDER:
        partition = registry.partition_subjects(registry.DATASETS[dataset])
        split = registry.source_train_audit_split(dataset, "A", partition["source_panel_A"])
        rows.append({
            "dataset": dataset,
            "source_training_subjects": "|".join(map(str, split["source_training"])),
            "source_training_count": len(split["source_training"]),
            "source_audit_subjects": "|".join(map(str, split["source_audit"])),
            "source_audit_count": len(split["source_audit"]),
            "target_subjects": str(v2.CANARY_TARGETS[dataset]),
            "target_count": 1,
            "sets_disjoint": int(
                not set(split["source_training"]) & set(split["source_audit"])
                and v2.CANARY_TARGETS[dataset] not in set(split["source_training"]) | set(split["source_audit"])
            ),
            "persist_loaded_lists_and_trial_counts": 1,
            "real_data_access": 0,
        })
    return rows


def contract_rows() -> dict[str, list[dict[str, Any]]]:
    environment = _exact_environment()
    loader_rows = _verify_loader_sources()
    return {
        "historical_lock_supersession.csv": [{
            "object": "C84C_EXECUTION_LOCK_V1", "path": "oaci/reports/C84C_EXECUTION_LOCK.json",
            "sha256": HISTORICAL_LOCK_SHA256, "preserved": 1, "operative_for_execution": 0,
            "reason": "runtime_did_not_replay_22_bound_files_or_complete_declared_canary_checks",
            "real_outcome_access": 0,
        }],
        "environment_version_registry.csv": [
            {"object": "python", "historical_locked": "3.9.25", "V3_locked": environment["python"],
             "distribution_metadata": "NOT_APPLICABLE", "runtime_value": environment["python"],
             "compatibility": "V3_REPAIR_Python_3.9_incompatible_with_MOABB_MNE", "real_data_access": 0},
            {"object": "torch", "historical_locked": "2.6.0+cu124", "V3_locked": "2.6.0+cu124",
             "distribution_metadata": environment["distributions"]["torch"],
             "runtime_value": environment["runtime"]["torch"], "compatibility": "PASS", "real_data_access": 0},
            {"object": "moabb", "historical_locked": "1.5.0", "V3_locked": "1.5.0",
             "distribution_metadata": environment["distributions"]["moabb"],
             "runtime_value": environment["runtime"]["moabb"], "compatibility": "Requires-Python>=3.10", "real_data_access": 0},
            {"object": "mne", "historical_locked": "1.11.0", "V3_locked": "1.11.0",
             "distribution_metadata": environment["distributions"]["mne"],
             "runtime_value": environment["runtime"]["mne"], "compatibility": "Requires-Python>=3.10", "real_data_access": 0},
            {"object": "chardet", "historical_locked": "UNBOUND", "V3_locked": "5.2.0",
             "distribution_metadata": environment["distributions"]["chardet"],
             "runtime_value": "NOT_IMPORTED", "compatibility": "stderr_warning_guard", "real_data_access": 0},
        ],
        "loader_source_identity_registry.csv": loader_rows,
        "exact_subject_identity_contract.csv": subject_contract_rows(),
        "actual_epoch_interface_contract.csv": [{
            "check": check, "locked_value": value, "runtime_source": source,
            "persisted": 1, "failure_is_blocking": 1, "real_data_access_in_C84R2": 0,
        } for check, value, source in (
            ("actual_ch_names", "exact_20_channel_order", "returned_Epochs.ch_names"),
            ("actual_sfreq", "160.0", "returned_Epochs.info[sfreq]"),
            ("pre_half_open_n_times", "480_or_481", "returned_Epochs.times_and_get_data"),
            ("final_n_times", "480", "half_open_array"),
            ("first_time", "0.0", "returned_Epochs.times"),
            ("final_last_time", "479/160", "half_open_times"),
            ("input_shape", "trial_x_20_x_480", "normalized_tensor"),
            ("bad_or_synthesized_channels", "NONE", "returned_Epochs.info[bads]"),
        )],
        "source_audit_instrumentation_contract.csv": [{
            "field": field, "required_per_unit": 1, "unit_count": 243,
            "source_labels_allowed": int(field == "source_class_label"),
            "scientific_metric": 0, "persist_and_reload": 1,
        } for field in sorted(runtime.SOURCE_AUDIT_FIELDS)],
        "target_unlabeled_instrumentation_contract.csv": [{
            "field": field, "required_per_unit": 1, "unit_count": 243,
            "target_label": 0, "persist_and_reload": 1,
        } for field in sorted(runtime.TARGET_UNLABELED_FIELDS)],
        "persisted_artifact_replay_contract.csv": [{
            "artifact": artifact, "replay": replay, "per_unit": 1,
            "required_units": 243, "failure_is_blocking": 1,
        } for artifact, replay in (
            ("checkpoint", "file_SHA_state_schema_state_hash"),
            ("sidecar", "canonical_field_set_and_identity"),
            ("source_audit", "field_shape_row_ID_saved_softmax"),
            ("target_unlabeled", "field_shape_row_ID_saved_softmax_z_classifier_repeat"),
            ("genealogy", "parent_ERM_and_previous_trajectory_hash"),
        )],
        "optimizer_replay_contract.csv": [{
            "regime": regime, "expected_optimizer_labels": labels,
            "encoder_step_rule": steps, "load_required": 1,
            "file_SHA_required": 1, "required_units": count,
        } for regime, labels, steps, count in (
            ("ERM", "encoder", "200", 3),
            ("OACI", "critic|encoder", "trajectory_order*100", 120),
            ("SRC", "encoder", "trajectory_order*100", 120),
        )],
        "deterministic_prefix_contract.csv": [{
            "dataset": dataset, "repeat_count": 2, "full_duplicate_training": 0,
            "model_init_hash": 1, "materialized_plan_hashes": 1,
            "first_batch_ID_hash": 1, "one_step_state_hash": 1,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0",
        } for dataset in v2.DATASET_ORDER],
        "attempt_ledger_contract.csv": [{
            "order": order, "stage": stage, "inside_failure_ledger": int(order >= 3),
            "before_real_data": int(order <= 7), "access_counters_persisted": int(order >= 3),
        } for order, stage in enumerate((
            "preauthorization_bound_replay", "authorization_consumption", "attempt_ledger_start",
            "package_imports_and_versions", "CUDA_check", "loader_source_replay",
            "loader_import", "dataset_access", "training", "instrumentation", "manifest_publication",
        ), start=1)],
        "canary_complete_gate.csv": [{
            "component": component, "required": count, "replay_required": 1,
            "partial_completion_reusable": 0, "scientific_metric": 0,
        } for component, count in (
            ("checkpoint_state_sidecar_units", 243),
            ("strict_source_audit_artifacts", 243),
            ("target_unlabeled_artifacts", 243),
            ("persisted_replay_units", 243),
        )],
        "risk_register.csv": [{
            "risk": risk, "blocking": 0, "status": "CLOSED_BY_V3_LOCKED_CONTROL",
            "control": control, "real_outcome_access": 0,
        } for risk, control in (
            ("historical_lock_used_for_execution", "additive_V3_protocol_and_new_lock_required"),
            ("bound_file_drift_after_lock", "runtime_SHA_and_Git_blob_replay_before_output_root"),
            ("bound_registry_drift", "runtime_table_SHA_replay"),
            ("historical_python_version_incompatible", "dedicated_Python_3.13.7_exact_snapshot"),
            ("package_version_drift", "importlib_metadata_and_runtime_version_replay"),
            ("loader_source_drift", "four_source_hashes_before_get_data"),
            ("subject_omission_or_addition", "exact_loaded_subject_set_assertion"),
            ("channel_order_or_sfreq_drift", "returned_Epochs_interface_assertion"),
            ("source_audit_missing", "243_artifact_complete_gate"),
            ("target_label_consumed", "structural_slot_unbound_and_metadata_scanner"),
            ("saved_artifact_drift", "reload_and_recompute_all_identifies"),
            ("nondeterministic_training_prefix", "two_repeat_prefix_fingerprint"),
            ("post_consumption_failure_unrecorded", "attempt_ledger_before_import"),
            ("C84F_or_C84S_lock_created", "C84R2_creates_C84C_lock_only"),
            ("real_data_access_in_C84R2", "protocol_and_tests_are_synthetic_only"),
        )],
        "failure_reason_ledger.csv": [
            {"failure_id": "C84R2-F01", "stage": "environment_replay",
             "root_cause": "historical_Python_3.9.25_is_incompatible_with_MOABB_1.5.0_and_MNE_1.11.0",
             "failed_attempt": "historical_environment_identity_replay",
             "repair": "dedicated_exact_snapshot_Python_3.13.7_packages_unchanged",
             "scientific_registry_changed": 0, "real_outcome_access": 0, "status": "CLOSED_BEFORE_LOCK"},
            {"failure_id": "C84R2-F02", "stage": "environment_snapshot",
             "root_cause": "conda_clone_relinked_pip_overlays_to_CPU_torch_and_MNE_1.10.1",
             "failed_attempt": "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3",
             "repair": "byte_exact_hardlink_snapshot_c84c-eeg2025-v3-exact",
             "scientific_registry_changed": 0, "real_outcome_access": 0, "status": "CLOSED_BEFORE_LOCK"},
        ],
    }


def generate() -> dict[str, Any]:
    if sha256_file(runtime.REPAIR_PROTOCOL_PATH) != REPAIR_PROTOCOL_SHA256:
        raise RuntimeError("C84R2 repair protocol hash drift")
    canary = build_canary_protocol()
    canary_path = REPORT_DIR / "C84_CANARY_PROTOCOL_V3.json"
    write_json(canary_path, canary)
    canary_sha = sha256_file(canary_path)
    write_sha(canary_path.with_suffix(".sha256"), canary_sha, canary_path.name)
    field = build_field_protocol(canary_sha)
    field_path = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V3.json"
    write_json(field_path, field)
    field_sha = sha256_file(field_path)
    write_sha(field_path.with_suffix(".sha256"), field_sha, field_path.name)
    for name, rows in contract_rows().items():
        write_csv(TABLE_DIR / name, rows)
    return {
        "canary_protocol_v3_sha256": canary_sha,
        "field_protocol_v3_sha256": field_sha,
        "external_protocol_v2_sha256": EXTERNAL_V2_SHA256,
        "science_protocol_v2_sha256": SCIENCE_V2_SHA256,
        "scientific_interface_changed": False,
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
        "dataset_downloads": 0,
        "C84C_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
