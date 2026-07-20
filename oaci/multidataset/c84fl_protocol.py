"""Generate the C84F full-field execution and manifest protocol.

This module is metadata-only. It reads committed C84C engineering metadata and
the external C84C manifest, but never imports array, EEG, loader, or ML code.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import c84_dataset_registry_v2 as registry
from . import c84r_v2_protocols as identity
from .c84r_montage_repair import EPOCH_RULE, INTERFACE_ID, MONTAGE_SHA256


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84fl_tables"
CREATED_AT_UTC = "2026-07-14T12:30:32Z"

C84C_FINAL_HEAD = "f7bbd27579308e01ed5c0388cb728cc7417978ac"
C84C_AUTHORIZATION_COMMIT = "6949b62a51f7cd092c63be4ca24654e9ab7db068"
C84C_RESULT_COMMIT = "2f541e526deb79091ad164b0d37419941e6f662b"
C84C_RESULT_SHA256 = "bec3a8b205a3d13fdb848ce1f82f71f903d05a97f746fdae25b3b4cce40e67f0"
C84C_MANIFEST_SHA256 = "530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b"
C84C_VALID_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v4/"
    "lock_c198607fb9e46ea2353f"
)
C84C_FAILED_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v3/"
    "lock_2e38dcd63c02a887b1dc"
)

EXTERNAL_PROTOCOL_SHA256 = "522e6fe8372f8c73741ed146a27068076db8c3d7087f4c4a36760fe0328b7c2f"
FIELD_PROTOCOL_V4_SHA256 = "eff7ebbc2e4f91830a3df1d679adfcae6eae2ab8a1e91c64ed28df7fce96aa12"
SCIENCE_PROTOCOL_SHA256 = "dc33b22527352bd42989c26f6771b4a49dc1443d458962587ca3d70ad76dd631"
NUMERICAL_REPAIR_SHA256 = "cdbdb9a25dc29b6a37ac9eb65f130f44efa120042dfb7ddb140cf3db103ec196"

TOTAL_UNITS = 1944
REUSED_UNITS = 243
REMAINING_UNITS = 1701
TOTAL_PHASES = 72
REUSED_PHASES = 9
REMAINING_PHASES = 63
TOTAL_CONTEXTS = 944
CANARY_CONTEXTS = 3
REMAINING_CONTEXTS = 941
TOTAL_SLICES = 76464
CANARY_SLICES = 243
REMAINING_SLICES = 76221
TARGET_COUNTS = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
WAVE_BY_CELL = {
    ("A", 5, 0): "C84C_REUSE",
    ("A", 5, 1): "A",
    ("A", 6, 0): "A",
    ("A", 6, 1): "A",
    ("B", 5, 0): "B0",
    ("B", 5, 1): "B1",
    ("B", 6, 0): "B1",
    ("B", 6, 1): "B1",
}
WAVE_ORDER = {"C84C_REUSE": 0, "A": 1, "B0": 2, "B1": 3}

MODEL_MANIFEST_FIELDS = (
    "unit_id", "dataset", "panel", "training_seed", "level", "regime",
    "epoch", "trajectory_order", "checkpoint_path", "checkpoint_sha256",
    "optimizer_path", "optimizer_sha256", "sidecar_path", "sidecar_sha256",
    "source_audit_path", "source_audit_sha256", "model_state_hash",
    "parent_ERM_model_state_hash", "previous_trajectory_model_state_hash",
    "reuse_provenance", "checkpoint_replay_pass", "optimizer_replay_pass",
    "sidecar_replay_pass", "source_audit_replay_pass", "training_target_rows",
    "training_target_labels", "source_audit_rows_used_in_training",
    "target_outcome_retention", "target_outcome_retry",
)
TARGET_TRIAL_REGISTRY_FIELDS = (
    "dataset", "target_subject_id", "target_trial_id", "session", "run",
    "interface_id", "montage_sha256", "sample_rate_hz", "n_times",
    "input_finite", "target_label_field_count",
)
TARGET_ARTIFACT_FIELDS = (
    "unit_id", "dataset", "panel", "training_seed", "level", "regime",
    "epoch", "trajectory_order", "target_subject_id", "target_trial_id",
    "session", "run", "logits", "probabilities", "z", "Wz_plus_b",
    "classifier_weight", "classifier_bias",
)
FIELD_DESCRIPTOR_FIELDS = (
    "unit_id", "checkpoint", "optimizer", "training_sidecar", "source_audit",
    "complete_target_unlabeled", "target_context_index", "interface_id",
    "protocol_sha256", "model_reuse_provenance", "target_artifact_provenance",
    "canary_subset_replay", "failure_retry_provenance",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_bytes(payload) + b"\n")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise RuntimeError(f"refusing empty C84FL table: {path}")
    fieldnames = list(fields or materialized[0])
    if any(set(row) != set(fieldnames) for row in materialized):
        raise RuntimeError(f"C84FL CSV schema drift: {path}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _assert_hash(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"C84FL identity replay failed for {path}: {observed}")


def load_c84c() -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.json"
    manifest_path = C84C_VALID_ROOT / "C84C_COMPLETE_CANARY_MANIFEST.json"
    _assert_hash(result_path, C84C_RESULT_SHA256)
    _assert_hash(manifest_path, C84C_MANIFEST_SHA256)
    result = read_json(result_path)
    manifest = read_json(manifest_path)
    if result["gate"] != "C84C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84F_REVIEW_REQUIRED":
        raise RuntimeError("C84C accepted gate drift")
    if not manifest["complete_gate"]["complete"] or manifest["unit_count"] != REUSED_UNITS:
        raise RuntimeError("C84C complete manifest is not a reusable 243-unit field")
    return result, manifest


def _canary_manifest_units(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    units = {
        row["unit_id"]: {**row, "dataset": dataset["dataset"]}
        for dataset in manifest["datasets"] for row in dataset["units"]
    }
    if len(units) != REUSED_UNITS:
        raise RuntimeError("C84C manifest does not contain 243 unique units")
    return units


def _read_sidecar(dataset: str, unit_id: str) -> dict[str, Any]:
    return read_json(C84C_VALID_ROOT / dataset / "sidecars" / f"{unit_id}.json")


def reusable_unit_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_units = _canary_manifest_units(manifest)
    rows = []
    for planned in identity.candidate_units():
        if not planned["canary_subset"]:
            continue
        unit = manifest_units.get(planned["unit_id"])
        if unit is None:
            raise RuntimeError(f"C84C reusable unit absent: {planned['unit_id']}")
        sidecar = _read_sidecar(planned["dataset"], planned["unit_id"])
        expected = {
            "checkpoint_sha256": sidecar["checkpoint"]["sha256"],
            "optimizer_sha256": sidecar["optimizer"]["file_sha256"],
            "sidecar_sha256": sha256_file(C84C_VALID_ROOT / planned["dataset"] / "sidecars" / f"{planned['unit_id']}.json"),
            "source_audit_sha256": sidecar["source_audit"]["sha256"],
            "target_unlabeled_sha256": sidecar["target_unlabeled"]["sha256"],
        }
        if any(unit[key] != value for key, value in expected.items()):
            raise RuntimeError(f"C84C sidecar/manifest hash mismatch: {planned['unit_id']}")
        rows.append({
            "unit_id": planned["unit_id"], "dataset": planned["dataset"], "panel": "A",
            "training_seed": 5, "level": 0, "regime": planned["regime"],
            "epoch": planned["epoch"], "trajectory_order": planned["trajectory_order"],
            "checkpoint_path": sidecar["checkpoint"]["path"], "checkpoint_sha256": expected["checkpoint_sha256"],
            "optimizer_path": sidecar["optimizer"]["path"], "optimizer_sha256": expected["optimizer_sha256"],
            "sidecar_path": str(C84C_VALID_ROOT / planned["dataset"] / "sidecars" / f"{planned['unit_id']}.json"),
            "sidecar_sha256": expected["sidecar_sha256"],
            "source_audit_path": sidecar["source_audit"]["path"],
            "source_audit_sha256": expected["source_audit_sha256"],
            "canary_target_path": sidecar["target_unlabeled"]["path"],
            "canary_target_sha256": expected["target_unlabeled_sha256"],
            "model_state_hash": unit["model_state_hash"],
            "parent_ERM_model_state_hash": unit["parent_ERM_model_state_hash"],
            "previous_trajectory_model_state_hash": unit["previous_trajectory_model_state_hash"],
            "model_state_source_audit_reusable": 1, "canary_target_slice_only": 1,
            "complete_target_artifact_reusable": 0, "manifest_replay_pass": 1,
            "failed_job_895366_artifact_reused": 0,
        })
    if len(rows) != REUSED_UNITS:
        raise RuntimeError("C84FL reusable registry is not 243 rows")
    return rows


def complete_unit_rows() -> list[dict[str, Any]]:
    rows = []
    for unit in identity.candidate_units():
        cell = (unit["source_panel"], unit["training_seed"], unit["level"])
        wave = WAVE_BY_CELL[cell]
        rows.append({
            "unit_id": unit["unit_id"], "dataset": unit["dataset"],
            "panel": unit["source_panel"], "training_seed": unit["training_seed"],
            "level": unit["level"], "regime": unit["regime"], "epoch": unit["epoch"],
            "trajectory_order": unit["trajectory_order"], "interface_id": unit["interface_id"],
            "montage_sha256": unit["montage_sha256"], "wave": wave,
            "wave_order": WAVE_ORDER[wave], "reuse_C84C_model_state_source_audit": int(unit["canary_subset"]),
            "train_in_C84F": int(not unit["canary_subset"]),
            "target_subject_contexts": TARGET_COUNTS[unit["dataset"]],
            "complete_target_artifact_required": 1,
        })
    if len(rows) != TOTAL_UNITS or len({row["unit_id"] for row in rows}) != TOTAL_UNITS:
        raise RuntimeError("C84FL complete unit registry drift")
    return rows


def wave_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in identity.DATASET_ORDER:
        for panel in identity.PANELS:
            for seed in identity.SEEDS:
                for level in identity.LEVELS:
                    wave = WAVE_BY_CELL[(panel, seed, level)]
                    rows.append({
                        "wave": wave, "wave_order": WAVE_ORDER[wave], "dataset": dataset,
                        "panel": panel, "training_seed": seed, "level": level,
                        "zoos": 1, "candidate_units": 81, "training_phases": 3,
                        "action": "REPLAY_C84C" if wave == "C84C_REUSE" else "TRAIN",
                        "release_evidence": "engineering_only",
                        "target_value_release_evidence_allowed": 0,
                    })
    counts = {
        wave: sum(row["candidate_units"] for row in rows if row["wave"] == wave)
        for wave in ("C84C_REUSE", "A", "B0", "B1")
    }
    if counts != {"C84C_REUSE": 243, "A": 729, "B0": 243, "B1": 729}:
        raise RuntimeError(f"C84FL wave arithmetic drift: {counts}")
    return rows


def source_view_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in identity.DATASET_ORDER:
        partition = registry.partition_subjects(registry.DATASETS[dataset])
        for panel in identity.PANELS:
            subjects = partition[f"source_panel_{panel}"]
            split = registry.source_train_audit_split(dataset, panel, subjects)
            rows.append({
                "dataset": dataset, "panel": panel,
                "source_training_subjects": "|".join(map(str, split["source_training"])),
                "source_training_count": len(split["source_training"]),
                "source_audit_subjects": "|".join(map(str, split["source_audit"])),
                "source_audit_count": len(split["source_audit"]),
                "sets_disjoint": int(not set(split["source_training"]) & set(split["source_audit"])),
                "target_overlap": int(bool(set(subjects) & set(partition["targets"]))),
                "source_labels_allowed": 1, "target_rows_in_training": 0,
                "interface_id": INTERFACE_ID,
            })
    return rows


def schema_rows(fields: Sequence[str], artifact: str) -> list[dict[str, Any]]:
    label_fields = {"source_class_label"}
    array_fields = {"logits", "probabilities", "z", "Wz_plus_b", "classifier_weight", "classifier_bias"}
    return [{
        "artifact": artifact, "field_order": order, "field": field,
        "type_contract": "numeric_array" if field in array_fields else "scalar_or_row_key",
        "target_label_or_derived": 0, "source_label_allowed": int(field in label_fields),
        "required": 1,
    } for order, field in enumerate(fields)]


def canary_slice_rows(reusable: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for unit in reusable:
        target = identity.CANARY_TARGETS[unit["dataset"]]
        rows.append({
            "unit_id": unit["unit_id"], "dataset": unit["dataset"],
            "canary_target_subject_id": target, "canary_target_path": unit["canary_target_path"],
            "canary_target_sha256": unit["canary_target_sha256"],
            "historical_scope": "one_target_subject_slice",
            "complete_dataset_target_subjects": TARGET_COUNTS[unit["dataset"]],
            "complete_target_artifact_reusable": 0,
            "C84F_all_target_subset_replay_required": 1,
            "trial_ID_and_numerical_identity_required": 1,
        })
    return rows


def context_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset, targets in TARGET_COUNTS.items():
        contexts = targets * 2 * 2 * 2
        rows.append({
            "dataset": dataset, "target_subjects": targets, "panels": 2,
            "training_seeds": 2, "levels": 2, "target_contexts": contexts,
            "candidates_per_context": 81, "candidate_context_slices": contexts * 81,
            "C84C_target_contexts": 1, "C84C_candidate_context_slices": 81,
            "remaining_target_contexts": contexts - 1,
            "remaining_candidate_context_slices": contexts * 81 - 81,
        })
    rows.append({
        "dataset": "TOTAL", "target_subjects": 118, "panels": 2,
        "training_seeds": 2, "levels": 2, "target_contexts": TOTAL_CONTEXTS,
        "candidates_per_context": 81, "candidate_context_slices": TOTAL_SLICES,
        "C84C_target_contexts": CANARY_CONTEXTS, "C84C_candidate_context_slices": CANARY_SLICES,
        "remaining_target_contexts": REMAINING_CONTEXTS,
        "remaining_candidate_context_slices": REMAINING_SLICES,
    })
    return rows


def resource_rows() -> list[dict[str, Any]]:
    measured_hours = 1 + 46 / 60 + 19 / 3600
    remaining_hours = measured_hours * (REMAINING_PHASES / REUSED_PHASES)
    complete_hours = measured_hours * (TOTAL_PHASES / REUSED_PHASES)
    # Measured canary bytes are decomposed by artifact class; the target estimate
    # scales each dataset's one-target slice by 8 zoos and its target population.
    class_bytes: dict[str, int] = {}
    for artifact in ("checkpoints", "optimizer_states", "sidecars", "source_audit"):
        class_bytes[artifact] = sum(
            sum(path.stat().st_size for path in (C84C_VALID_ROOT / dataset / artifact).glob("*"))
            for dataset in identity.DATASET_ORDER
        )
    model_complete = sum(class_bytes.values()) * 8
    target_complete = sum(
        sum(path.stat().st_size for path in (C84C_VALID_ROOT / dataset / "target_unlabeled").glob("*"))
        * TARGET_COUNTS[dataset] * 8
        for dataset in identity.DATASET_ORDER
    )
    measured_root = sum(path.stat().st_size for path in C84C_VALID_ROOT.rglob("*") if path.is_file())
    projected_derived = model_complete + target_complete
    envelope_bytes = 2 * 1024**4
    values = (
        ("C84C_measured", "calendar_and_GPU_time", measured_hours, "hours", 250.0),
        ("C84F_remaining", "linear_training_GPU_time", remaining_hours, "hours", 250.0),
        ("C84_complete", "linear_training_GPU_time", complete_hours, "hours", 250.0),
        ("C84C_measured", "external_root", measured_root, "bytes", envelope_bytes),
        ("C84_complete", "model_state_source_artifacts", model_complete, "bytes", envelope_bytes),
        ("C84_complete", "all_target_instrumentation_projection", target_complete, "bytes", envelope_bytes),
        ("C84_complete", "derived_projection", projected_derived, "bytes", envelope_bytes),
        ("C84F", "raw_download_cache_upper_bound", 180 * 1024**3, "bytes", envelope_bytes),
        ("C84_complete", "download_plus_derived_projection", 180 * 1024**3 + projected_derived, "bytes", envelope_bytes),
        ("C84F", "CPU_manifest_replay", TOTAL_SLICES, "candidate_context_slices", TOTAL_SLICES),
        ("C84F_remaining", "candidate_units", REMAINING_UNITS, "units", REMAINING_UNITS),
        ("C84_complete", "candidate_units", TOTAL_UNITS, "units", TOTAL_UNITS),
    )
    return [{
        "scope": scope, "resource": resource, "estimate": f"{estimate:.6f}" if isinstance(estimate, float) else estimate,
        "unit": unit, "hard_or_planning_envelope": envelope,
        "within_envelope": int(float(estimate) <= float(envelope)),
        "basis": "C84C_job_895441_measured_and_scope_arithmetic",
    } for scope, resource, estimate, unit, envelope in values]


def retry_rows() -> list[dict[str, Any]]:
    return [
        {"failure_stage": "training_before_target_instrumentation", "retraining_allowed": 1,
         "target_instrumentation_allowed": 0, "implementation_change_allowed_under_same_lock": 0,
         "required_action": "preserve_attempt_same_lock_same_rows_IDs_RNG_new_empty_root_no_target_value"},
        {"failure_stage": "model_field_freeze", "retraining_allowed": 0,
         "target_instrumentation_allowed": 0, "implementation_change_allowed_under_same_lock": 0,
         "required_action": "stop_and_reconcile_incomplete_model_manifest"},
        {"failure_stage": "target_instrumentation", "retraining_allowed": 0,
         "target_instrumentation_allowed": 1, "implementation_change_allowed_under_same_lock": 0,
         "required_action": "preserve_attempt_additive_forward_repair_new_lock_no_retention_change"},
        {"failure_stage": "scientific_outcome", "retraining_allowed": 0,
         "target_instrumentation_allowed": 0, "implementation_change_allowed_under_same_lock": 0,
         "required_action": "not_applicable_target_labels_and_scientific_scores_forbidden"},
    ]


def risk_rows() -> list[dict[str, Any]]:
    risks = {
        "C84C_failed_root_reused": "valid root and manifest hashes fixed; job 895366 rejected",
        "canary_target_slices_called_complete": "3/944 and 243/76464 scope explicit",
        "canary_model_state_retrained": "243 IDs marked reuse-only",
        "complete_unit_arithmetic_drift": "1,944-ID registry exact-once validation",
        "remaining_wave_arithmetic_drift": "729/243/729 wave registry",
        "target_labels_consumed": "target trial schema has zero label fields",
        "target_value_drives_wave_release": "engineering-only gate allowlist",
        "target_value_drives_retention_retry": "zero counters and retry state machine",
        "target_instrumentation_before_model_freeze": "model-field barrier state machine",
        "target_instrumentation_retrains_model": "forward stage has no training transition",
        "source_subject_drift": "exact 12+4 source contract per panel",
        "target_subject_omission": "118-subject exact target registry required",
        "channel_or_sfreq_drift": "20-channel/160-Hz/480-sample interface replay",
        "canary_subset_numerical_mismatch": "trial-ID and numeric replay required",
        "raw_input_identity_unrecorded": "path/size/SHA manifest required",
        "warnings_hide_traceback": "warning classifier fails traceback/parse/nonfinite",
        "partial_model_manifest_published": "atomic complete gate 1,944/1,944",
        "partial_target_manifest_published": "atomic complete gate 76,464/76,464",
        "runtime_bytes_drift": "future execution lock binds every implementation byte",
        "resource_envelope_exceeded": "250 GPU-hour and 2-TiB hard guards",
        "C84S_scope_creep": "no C84S execution lock or target label view",
        "raw_EEG_weights_cache_in_Git": "50-MiB and prohibited-suffix scan",
        "prior_C84C_authorization_reused": "fresh C84F record required",
    }
    return [{
        "risk": risk, "status": "CLOSED_BY_PROTOCOL_CONTROL",
        "blocking": 0, "control": control, "real_data_access_in_C84FL": 0,
    } for risk, control in risks.items()]


def build_protocol() -> dict[str, Any]:
    return {
        "schema_version": "c84f_full_field_execution_and_manifest_protocol_v1",
        "milestone": "C84FL",
        "status": "PROTOCOL_LOCKED_IMPLEMENTATION_AND_EXECUTION_LOCK_PENDING_NOT_AUTHORIZED",
        "created_at_utc": CREATED_AT_UTC,
        "chronology": {
            "C84C_final_HEAD": C84C_FINAL_HEAD,
            "C84C_authorization_commit": C84C_AUTHORIZATION_COMMIT,
            "C84C_result_commit": C84C_RESULT_COMMIT,
            "C84F_protocol_precedes_real_adapter_implementation": True,
            "C84F_real_EEG_access_before_protocol": 0,
            "C84F_training_forward_GPU_before_protocol": 0,
            "C84_scientific_outcome_access": 0,
        },
        "parent_objects": {
            "external_protocol_V2_sha256": EXTERNAL_PROTOCOL_SHA256,
            "field_protocol_V4_sha256": FIELD_PROTOCOL_V4_SHA256,
            "scientific_protocol_V2_sha256": SCIENCE_PROTOCOL_SHA256,
            "C84R3_numerical_repair_sha256": NUMERICAL_REPAIR_SHA256,
            "C84C_result_sha256": C84C_RESULT_SHA256,
            "C84C_complete_manifest_sha256": C84C_MANIFEST_SHA256,
            "C84C_valid_external_root": str(C84C_VALID_ROOT),
            "failed_job_895366_root_reusable": False,
        },
        "epistemic_status": {
            "designed_after_C84C_engineering_results": True,
            "prospective_to_remaining_training": True,
            "prospective_to_complete_target_instrumentation": True,
            "prospective_to_all_C84_scientific_outcomes": True,
            "target_label_or_selector_outcome_access_before_protocol": 0,
            "C84C_engineering_only": True,
        },
        "interface": {
            "id": INTERFACE_ID, "montage_sha256": MONTAGE_SHA256,
            "epoch_rule": EPOCH_RULE, "sample_rate_hz": 160,
            "input_shape": [20, 480], "target_labels": False,
        },
        "field_arithmetic": {
            "complete_zoos": 24, "complete_training_phases": TOTAL_PHASES,
            "complete_units": TOTAL_UNITS, "C84C_reused_zoos": 3,
            "C84C_reused_phases": REUSED_PHASES, "C84C_reused_units": REUSED_UNITS,
            "remaining_zoos": 21, "remaining_phases": REMAINING_PHASES,
            "remaining_units": REMAINING_UNITS, "target_subjects": 118,
            "target_contexts": TOTAL_CONTEXTS, "candidate_context_slices": TOTAL_SLICES,
        },
        "C84C_reuse": {
            "reusable": ["candidate_ID", "checkpoint", "optimizer_state", "genealogy_sidecar", "source_audit"],
            "canary_target_artifacts": "historical_one_target_slices_only",
            "canary_target_contexts": CANARY_CONTEXTS,
            "canary_candidate_context_slices": CANARY_SLICES,
            "complete_target_artifact_reusable": False,
            "all_target_subset_replay_required": True,
            "failed_job_895366_artifacts_reusable": False,
        },
        "target_artifact_layout": {
            "choice": "one_all_target_artifact_per_candidate_unit_with_context_index",
            "artifacts": TOTAL_UNITS, "contexts": TOTAL_CONTEXTS,
            "candidate_context_slices": TOTAL_SLICES,
            "fields": list(TARGET_ARTIFACT_FIELDS),
            "target_label_fields": 0,
        },
        "waves": {
            "A": {"units": 729, "phases": 27},
            "B0": {"units": 243, "phases": 9},
            "B1": {"units": 729, "phases": 27},
            "release_evidence": "engineering_only",
            "target_values_allowed": False,
        },
        "barriers": {
            "dataset_input_freeze_before_training": True,
            "model_field_1944_freeze_before_complete_target_forward": True,
            "target_trial_registry_before_target_forward": True,
            "complete_target_field_atomic_freeze": True,
            "target_construction_evaluation_same_label_oracle_views": "not_provisioned",
        },
        "model_complete_gate": {
            "candidate_units": TOTAL_UNITS, "training_phases": TOTAL_PHASES,
            "checkpoint_optimizer_sidecar_source_audit_each": TOTAL_UNITS,
            "unique_unit_IDs": TOTAL_UNITS, "C84C_reused_units": REUSED_UNITS,
            "training_target_rows": 0, "training_target_labels": 0,
            "source_audit_rows_in_training": 0,
            "target_outcome_retention": 0, "target_outcome_retry": 0,
        },
        "target_complete_gate": {
            "target_subjects": TARGET_COUNTS, "target_contexts": TOTAL_CONTEXTS,
            "candidate_context_slices": TOTAL_SLICES,
            "linear_replay_abs_tolerance": 1e-5,
            "softmax_repeat_logits_repeat_z_abs_tolerance": 1e-6,
            "C84C_canary_slice_replay": True, "target_label_fields": 0,
        },
        "retry_policy": {
            "training_retry_same_lock_only_without_target_value": True,
            "implementation_change_requires_new_lock": True,
            "target_instrumentation_failure_may_not_retrain": True,
            "failed_attempts_preserved": True,
        },
        "runtime_lock_requirements": {
            "all_implementation_and_registry_bytes": True,
            "protocol_hashes": True, "C84C_manifest_and_reuse_registry": True,
            "environment_and_loader_source": True, "clean_HEAD_equals_origin_oaci": True,
            "fresh_direct_C84F_authorization": True,
            "attempt_ledger_before_protected_import": True,
        },
        "authorization": {
            "C84F_authorized": False, "C84S_authorized": False,
            "shortest_future_statement": "\u6388\u6743 C84F",
            "record_path": "oaci/reports/C84F_PI_AUTHORIZATION_RECORD.json",
            "prior_C84C_authorization_reusable": False,
        },
        "resource_envelopes": {"GPU_hours": 250, "external_bytes": 2 * 1024**4, "Git_file_MiB": 50},
        "final_gate": "C84_MULTI_DATASET_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED",
        "C84S_execution_lock_created": False,
    }


def timing_markdown(protocol_sha: str) -> str:
    return f"""# C84F Protocol Timing Audit

## Identity

- C84C accepted engineering HEAD: `{C84C_FINAL_HEAD}`
- C84C result SHA-256: `{C84C_RESULT_SHA256}`
- C84C complete manifest SHA-256: `{C84C_MANIFEST_SHA256}`
- C84F execution/manifest protocol SHA-256: `{protocol_sha}`

## Prospective boundary

The C84F protocol was committed before the real full-field adapter, execution
lock, direct C84F authorization, remaining-subject access, remaining training,
complete-target forward instrumentation, or any C84 scientific computation.

| Protected event before protocol | Count |
|---|---:|
| Remaining-subject EEG access | 0 |
| Remaining C84F training phases | 0 |
| Remaining C84F model units | 0 |
| Complete-target instrumentation slices | 0 |
| Target construction/evaluation label reads | 0 |
| Selector/scientific outcome reads | 0 |
| C84F GPU jobs | 0 |

C84C's three engineering target views remain disclosed historical canary slices.
They cover 3/944 target contexts and 243/76,464 candidate-context slices. They
cannot drive model retention, retries, wave release, or scientific inference.

## Stop boundary

C84FL creates an implementation and execution lock only. It does not authorize
or execute C84F, and it does not create a C84S execution lock.
"""


def generate() -> dict[str, Any]:
    for stem, expected in (
        ("C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2", EXTERNAL_PROTOCOL_SHA256),
        ("C84_FIELD_GENERATION_PROTOCOL_V4", FIELD_PROTOCOL_V4_SHA256),
        ("C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2", SCIENCE_PROTOCOL_SHA256),
        ("C84R3_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL", NUMERICAL_REPAIR_SHA256),
    ):
        _assert_hash(REPORT_DIR / f"{stem}.json", expected)
    result, manifest = load_c84c()
    reusable = reusable_unit_rows(manifest)
    complete = complete_unit_rows()
    remaining = [row for row in complete if row["train_in_C84F"]]
    if len(remaining) != REMAINING_UNITS:
        raise RuntimeError("C84FL remaining unit registry drift")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(TABLE_DIR / "c84c_result_identity_replay.csv", [
        {"object": "C84C_final_HEAD", "identity": C84C_FINAL_HEAD, "replay_pass": 1, "C84F_reuse_role": "ancestry"},
        {"object": "C84C_result", "identity": C84C_RESULT_SHA256, "replay_pass": 1, "C84F_reuse_role": "engineering_evidence"},
        {"object": "C84C_complete_manifest", "identity": C84C_MANIFEST_SHA256, "replay_pass": 1, "C84F_reuse_role": "243_unit_source"},
        {"object": "C84C_valid_external_root", "identity": str(C84C_VALID_ROOT), "replay_pass": 1, "C84F_reuse_role": "read_only_reuse"},
        {"object": "failed_job_895366", "identity": str(C84C_FAILED_ROOT), "replay_pass": 1, "C84F_reuse_role": "rejected"},
    ])
    write_csv(TABLE_DIR / "c84c_reusable_unit_registry.csv", reusable)
    write_csv(TABLE_DIR / "c84c_canary_target_slice_registry.csv", canary_slice_rows(reusable))
    write_csv(TABLE_DIR / "complete_unit_registry.csv", complete)
    write_csv(TABLE_DIR / "remaining_training_registry.csv", remaining)
    write_csv(TABLE_DIR / "wave_registry.csv", wave_rows())
    write_csv(TABLE_DIR / "source_view_contract.csv", source_view_rows())
    write_csv(TABLE_DIR / "target_unlabeled_trial_registry_schema.csv", schema_rows(TARGET_TRIAL_REGISTRY_FIELDS, "target_trial_registry"))
    write_csv(TABLE_DIR / "raw_input_manifest_schema.csv", schema_rows(
        ("dataset", "source_path", "bytes", "sha256", "consumed_by_view", "warning_count", "warning_classes"),
        "raw_input_manifest",
    ))
    write_csv(TABLE_DIR / "model_field_manifest_schema.csv", schema_rows(MODEL_MANIFEST_FIELDS, "model_field_manifest"))
    write_csv(TABLE_DIR / "target_instrumentation_schema.csv", schema_rows(TARGET_ARTIFACT_FIELDS, "complete_target_unlabeled"))
    write_csv(TABLE_DIR / "field_unit_descriptor_schema.csv", schema_rows(FIELD_DESCRIPTOR_FIELDS, "field_unit_descriptor"))
    write_csv(TABLE_DIR / "complete_context_arithmetic.csv", context_rows())
    write_csv(TABLE_DIR / "canary_subset_replay_contract.csv", [{
        "check": check, "required": 1, "tolerance": tolerance, "failure_action": "BLOCK_COMPLETE_FIELD",
    } for check, tolerance in (
        ("dataset_unit_target_trial_ID_exact_match", "exact"),
        ("logits", "1e-6"), ("probabilities", "1e-6"), ("z", "1e-6"),
        ("Wz_plus_b", "1e-5"), ("classifier_weight", "exact_hash"),
        ("classifier_bias", "exact_hash"), ("canary_artifact_sha256", "manifest_exact"),
    )])
    write_csv(TABLE_DIR / "retry_policy.csv", retry_rows())
    write_csv(TABLE_DIR / "resource_estimate.csv", resource_rows())
    write_csv(TABLE_DIR / "risk_register.csv", risk_rows())
    write_csv(TABLE_DIR / "failure_reason_ledger.csv", [{
        "failure_id": "NONE", "stage": "C84FL_protocol", "blocking": 0,
        "reason": "no_failure", "real_data_access": 0, "scientific_outcome_access": 0,
        "repair_required": 0,
    }])

    protocol_path = REPORT_DIR / "C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL.json"
    write_json(protocol_path, build_protocol())
    protocol_sha = sha256_file(protocol_path)
    protocol_path.with_suffix(".sha256").write_text(
        f"{protocol_sha}  {protocol_path.name}\n", encoding="ascii",
    )
    (REPORT_DIR / "C84F_PROTOCOL_TIMING_AUDIT.md").write_text(
        timing_markdown(protocol_sha), encoding="utf-8",
    )
    return {
        "schema_version": "c84fl_protocol_generation_v1",
        "protocol_sha256": protocol_sha,
        "reusable_units": len(reusable), "remaining_units": len(remaining),
        "complete_units": len(complete), "target_contexts": TOTAL_CONTEXTS,
        "candidate_context_slices": TOTAL_SLICES,
        "real_data_access": 0, "training_forward_GPU": 0,
        "C84F_authorized": False, "C84S_lock_created": False,
        "C84C_gate": result["gate"],
    }


def main() -> int:
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
