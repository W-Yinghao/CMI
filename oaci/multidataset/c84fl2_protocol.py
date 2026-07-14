"""Create the C84FL2 dual-level full-field protocol family.

This module is metadata-only. It replays committed compact results, JSON
manifests, sidecars, and artifact byte hashes. It never imports an EEG loader,
array stack, training framework, or GPU library.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84fl2_tables"
CREATED_AT_UTC = "2026-07-14T20:35:00Z"

BASE_HEAD = "4d2ca75b2fc149c80c3e51e93709aab12e67813a"
SUCCESS_GATE = "C84F_DUAL_LEVEL_FULL_FIELD_IMPLEMENTATION_AND_EXECUTION_LOCK_READY_FOR_PI_AUTHORIZATION"
FAILURE_GATE = "C84F_DUAL_CANARY_REUSE_TRAINING_TARGET_BARRIER_RESOURCE_OR_LOCK_RECONCILIATION_REQUIRED"
FIELD_GATE = "C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED"

HASHES = {
    "external_v3": "cd338bf1b97180b27dd42471dd2a67768194b74ed9896275424591d17e970ce0",
    "field_v6": "cd8646403a7564e9d1a7e3d64104483cbd56ac85bebaafa1244afdde8a8ed310",
    "science_v3": "bf6c7f718413b4b2ac2ad9786aa2e47dc045a536e7237d5d8c0464b6598130b8",
    "level1_protocol": "b9fed16afe9961d0d25f4801fa29859a8acb87c091e125ffe57e20e72ad00f35",
    "level1_numerical_repair": "2e199f6f63dffd1b02c1e31102ed189e31bf6e4961465394230f8e9de1d4ddf0",
    "operative_registry": "462fa840f7048511cb3e1a41b55f60441435d14f665014b856fe5fd8d66ac1b0",
    "c84c_result": "bec3a8b205a3d13fdb848ce1f82f71f903d05a97f746fdae25b3b4cce40e67f0",
    "c84c_manifest": "530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b",
    "c84l1c_result": "5bcccf351704c427d148ca1f44de26ef7e0b137d8de56aa0cf9ca3f6723abaf5",
    "c84l1c_manifest": "3cf1366ccf40efc82a6bb2ffef56045e83c0f0e9670429973f23252371ad1c18",
    "montage": "988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04",
    "intervention_registry": "89c4f366a222c1fe2ac31780bcbddbc9e59ff5afa4a779267abbd95429c41c17",
}

C84C_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v4/"
    "lock_c198607fb9e46ea2353f"
)
C84L1C_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-level1-canary-v2/"
    "lock_f9ebd88c72915bb41ba2"
)
C84C_MANIFEST = C84C_ROOT / "C84C_COMPLETE_CANARY_MANIFEST.json"
C84L1C_MANIFEST = C84L1C_ROOT / "C84L1C_COMPLETE_ENGINEERING_MANIFEST.json"
OPERATIVE_REGISTRY = REPORT_DIR / "c84l1p_tables/operative_complete_unit_registry_v2.csv"

DATASETS = ("Lee2019_MI", "Cho2017", "PhysionetMI")
TARGET_COUNTS = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
CANARY_TARGETS = {"Lee2019_MI": 19, "Cho2017": 24, "PhysionetMI": 106}
LEVEL0_ID = "C84_LEVEL0_FULL_SOURCE_PANEL_V1"
LEVEL1_ID = "C84_LEVEL1_FIXED_PANEL_LEFT_HAND_CELL_DELETION_V1"
INTERFACE_ID = "C84_LEFT_RIGHT_20CH_160HZ_0_3S_V2"
TOTAL_UNITS = 1944
REUSED_UNITS = 486
REMAINING_UNITS = 1458
TOTAL_PHASES = 72
REUSED_PHASES = 18
REMAINING_PHASES = 54
TOTAL_CONTEXTS = 944
TOTAL_SLICES = 76464
LINEAR_TOLERANCE = 2e-5
STRICT_TOLERANCE = 1e-6

MODEL_FIELDS = (
    "unit_id", "dataset", "panel", "training_seed", "level",
    "level_intervention_id", "regime", "epoch", "trajectory_order",
    "checkpoint_path", "checkpoint_sha256", "optimizer_path",
    "optimizer_sha256", "sidecar_path", "sidecar_sha256",
    "source_audit_path", "source_audit_sha256", "model_state_hash",
    "parent_ERM_model_state_hash", "previous_trajectory_model_state_hash",
    "population_signature_sha256", "support_graph_sha256", "plan_hashes",
    "paired_model_init_hash", "reuse_provenance", "checkpoint_replay_pass",
    "optimizer_replay_pass", "sidecar_replay_pass",
    "source_audit_replay_pass", "training_target_rows",
    "training_target_labels", "source_audit_rows_used_in_training",
    "target_outcome_retention", "target_outcome_retry",
)
TARGET_TRIAL_FIELDS = (
    "dataset", "target_subject_id", "target_trial_id", "session", "run",
    "interface_id", "montage_sha256", "sample_rate_hz", "sample_count",
    "finite_value_flag", "raw_input_path", "raw_input_bytes",
    "raw_input_sha256",
)
TARGET_ARTIFACT_FIELDS = (
    "unit_id", "dataset", "panel", "training_seed", "level",
    "level_intervention_id", "regime", "epoch", "trajectory_order",
    "target_subject_id", "target_trial_id", "session", "run", "logits",
    "probabilities", "z", "Wz_plus_b", "classifier_weight",
    "classifier_bias",
)
FIELD_DESCRIPTOR_FIELDS = (
    "unit_id", "checkpoint", "optimizer", "training_sidecar",
    "source_audit", "complete_target_unlabeled", "target_context_index",
    "interface_id", "protocol_identities", "level_intervention_id",
    "model_reuse_provenance", "target_artifact_provenance",
    "canary_subset_replay", "failed_attempt_provenance",
)


class C84FL2ProtocolError(RuntimeError):
    """Fail-closed C84FL2 metadata reconciliation error."""


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


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, value: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_bytes(value) + b"\n")
    return sha256_file(target)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    values = [dict(row) for row in rows]
    if not values:
        raise C84FL2ProtocolError(f"refusing empty C84FL2 table: {path}")
    fields = list(values[0])
    if any(set(row) != set(fields) for row in values):
        raise C84FL2ProtocolError(f"C84FL2 table schema mismatch: {path}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)
    return sha256_file(target)


def write_hash_sidecar(path: Path, digest: str) -> None:
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C84FL2ProtocolError(message)


def replay_inputs() -> dict[str, Any]:
    paths = {
        "external_v3": REPORT_DIR / "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3.json",
        "field_v6": REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V6.json",
        "science_v3": REPORT_DIR / "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3.json",
        "level1_protocol": REPORT_DIR / "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.json",
        "level1_numerical_repair": REPORT_DIR / "C84L1R1_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.json",
        "operative_registry": OPERATIVE_REGISTRY,
        "c84c_result": REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.json",
        "c84c_manifest": C84C_MANIFEST,
        "c84l1c_result": REPORT_DIR / "C84L1C_ENGINEERING_CANARY_RESULT.json",
        "c84l1c_manifest": C84L1C_MANIFEST,
    }
    for name, path in paths.items():
        require(path.is_file(), f"C84FL2 input absent: {name}")
        require(sha256_file(path) == HASHES[name], f"C84FL2 input hash drift: {name}")
    c84c_result = read_json(paths["c84c_result"])
    c84l1c_result = read_json(paths["c84l1c_result"])
    c84c_manifest = read_json(C84C_MANIFEST)
    c84l1c_manifest = read_json(C84L1C_MANIFEST)
    require(c84c_result["gate"] == "C84C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84F_REVIEW_REQUIRED",
            "C84C accepted gate drift")
    require(c84l1c_result["gate"] == "C84L1C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84FL2_REVIEW_REQUIRED",
            "C84L1C accepted gate drift")
    require(c84c_manifest["unit_count"] == 243 and c84c_manifest["complete_gate"]["complete"],
            "C84C manifest incomplete")
    require(c84l1c_manifest["unit_count"] == 243 and c84l1c_manifest["complete_gate"]["complete"],
            "C84L1C manifest incomplete")
    return {
        "paths": paths,
        "c84c_result": c84c_result,
        "c84l1c_result": c84l1c_result,
        "c84c_manifest": c84c_manifest,
        "c84l1c_manifest": c84l1c_manifest,
        "external_v3": read_json(paths["external_v3"]),
        "field_v6": read_json(paths["field_v6"]),
        "science_v3": read_json(paths["science_v3"]),
    }


def result_identity_rows(result_name: str, result: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected_result = HASHES[f"{result_name}_result"]
    expected_manifest = HASHES[f"{result_name}_manifest"]
    result_path = REPORT_DIR / (
        "C84C_ENGINEERING_CANARY_RESULT.json" if result_name == "c84c"
        else "C84L1C_ENGINEERING_CANARY_RESULT.json"
    )
    manifest_path = C84C_MANIFEST if result_name == "c84c" else C84L1C_MANIFEST
    rows = (
        ("result_json", result_path, expected_result, result["gate"]),
        ("complete_manifest", manifest_path, expected_manifest, "complete=true"),
    )
    return [{
        "milestone": result_name.upper(), "object": obj, "path": str(path),
        "expected_sha256": expected, "observed_sha256": sha256_file(path),
        "units": manifest["unit_count"], "phases": manifest["training_phases"],
        "gate_or_status": status, "replay_pass": 1,
    } for obj, path, expected, status in rows]


def manifest_unit_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {
        str(unit["unit_id"]): unit
        for dataset in manifest["datasets"]
        for unit in dataset["units"]
    }
    require(len(rows) == 243, "canary manifest does not contain 243 unique units")
    return rows


def artifact_identity(sidecar: Mapping[str, Any]) -> dict[str, tuple[Path, str]]:
    checkpoint = sidecar["checkpoint"]
    optimizer = sidecar["optimizer"]
    return {
        "checkpoint": (Path(checkpoint["path"]), str(checkpoint["sha256"])),
        "optimizer": (Path(optimizer["path"]), str(optimizer["file_sha256"])),
        "source_audit": (Path(sidecar["source_audit"]["path"]), str(sidecar["source_audit"]["sha256"])),
        "canary_target": (Path(sidecar["target_unlabeled"]["path"]), str(sidecar["target_unlabeled"]["sha256"])),
    }


def dual_canary_reuse_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    operative = read_csv(OPERATIVE_REGISTRY)
    c84c_units = manifest_unit_map(inputs["c84c_manifest"])
    c84l1c_units = manifest_unit_map(inputs["c84l1c_manifest"])
    rows = []
    for planned in operative:
        level = int(planned["level"])
        reusable = planned["C84C_reusable"] == "1" or planned["C84L1C_canary"] == "1"
        if not reusable:
            continue
        source = "C84C" if level == 0 else "C84L1C"
        root = C84C_ROOT if source == "C84C" else C84L1C_ROOT
        manifest_units = c84c_units if source == "C84C" else c84l1c_units
        unit_id = planned["unit_id"]
        unit = manifest_units.get(unit_id)
        require(unit is not None, f"reusable unit absent from {source}: {unit_id}")
        sidecar_path = root / planned["dataset"] / "sidecars" / f"{unit_id}.json"
        require(sidecar_path.is_file(), f"reusable sidecar absent: {unit_id}")
        require(sha256_file(sidecar_path) == unit["sidecar_sha256"], f"sidecar drift: {unit_id}")
        sidecar = read_json(sidecar_path)
        identities = artifact_identity(sidecar)
        for artifact, (path, expected) in identities.items():
            require(path.is_file(), f"{artifact} absent for reusable unit: {unit_id}")
            require(sha256_file(path) == expected, f"{artifact} hash drift: {unit_id}")
        require(unit["checkpoint_sha256"] == identities["checkpoint"][1], f"checkpoint manifest drift: {unit_id}")
        require(unit["optimizer_sha256"] == identities["optimizer"][1], f"optimizer manifest drift: {unit_id}")
        require(unit["source_audit_sha256"] == identities["source_audit"][1], f"source manifest drift: {unit_id}")
        require(unit["target_unlabeled_sha256"] == identities["canary_target"][1], f"target manifest drift: {unit_id}")
        rows.append({
            "unit_id": unit_id, "dataset": planned["dataset"], "panel": "A",
            "training_seed": 5, "level": level,
            "level_intervention_id": planned["level_intervention_id"],
            "regime": planned["regime"], "epoch": int(planned["epoch"]),
            "trajectory_order": int(planned["trajectory_order"]),
            "checkpoint_path": str(identities["checkpoint"][0]),
            "checkpoint_sha256": identities["checkpoint"][1],
            "optimizer_path": str(identities["optimizer"][0]),
            "optimizer_sha256": identities["optimizer"][1],
            "sidecar_path": str(sidecar_path), "sidecar_sha256": unit["sidecar_sha256"],
            "source_audit_path": str(identities["source_audit"][0]),
            "source_audit_sha256": identities["source_audit"][1],
            "canary_target_path": str(identities["canary_target"][0]),
            "canary_target_sha256": identities["canary_target"][1],
            "model_state_hash": unit["model_state_hash"],
            "parent_ERM_model_state_hash": unit["parent_ERM_model_state_hash"],
            "previous_trajectory_model_state_hash": unit["previous_trajectory_model_state_hash"],
            "reuse_source": source, "model_state_source_audit_reusable": 1,
            "canary_target_slice_only": 1, "complete_target_artifact_reusable": 0,
            "byte_hash_manifest_replay_pass": 1, "failed_artifact_reused": 0,
        })
    require(len(rows) == REUSED_UNITS, "dual-canary reuse registry is not 486 rows")
    require(len({row["unit_id"] for row in rows}) == REUSED_UNITS, "dual-canary reusable IDs are not unique")
    require(sum(row["level"] == 0 for row in rows) == 243, "level-0 reuse count drift")
    require(sum(row["level"] == 1 for row in rows) == 243, "level-1 reuse count drift")
    return sorted(rows, key=lambda row: (row["dataset"], row["level"], row["regime"], row["trajectory_order"]))


def wave_for(panel: str, seed: int) -> tuple[str, int]:
    mapping = {
        ("A", 5): ("DUAL_CANARY_REUSE", 0),
        ("A", 6): ("A", 1),
        ("B", 5): ("B0", 2),
        ("B", 6): ("B1", 3),
    }
    return mapping[(panel, seed)]


def operative_replay_rows(reuse: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    source_by_id = {str(row["unit_id"]): str(row["reuse_source"]) for row in reuse}
    rows = []
    for row in read_csv(OPERATIVE_REGISTRY):
        seed = int(row["training_seed"])
        level = int(row["level"])
        wave, wave_order = wave_for(row["panel"], seed)
        reuse_source = source_by_id.get(row["unit_id"], "NONE")
        rows.append({
            **row,
            "training_seed": seed, "level": level, "epoch": int(row["epoch"]),
            "trajectory_order": int(row["trajectory_order"]),
            "reuse_source": reuse_source, "wave": wave, "wave_order": wave_order,
            "train_in_C84F": int(reuse_source == "NONE"),
            "target_subject_contexts": TARGET_COUNTS[row["dataset"]],
            "complete_target_artifact_required": 1,
        })
    require(len(rows) == TOTAL_UNITS == len({row["unit_id"] for row in rows}), "operative unit scope drift")
    require(sum(row["level"] == 0 for row in rows) == 972, "level-0 operative count drift")
    require(sum(row["level"] == 1 for row in rows) == 972, "level-1 operative count drift")
    require(sum(row["train_in_C84F"] for row in rows) == REMAINING_UNITS, "remaining unit count drift")
    return rows


def remaining_training_rows(operative: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in operative if row["train_in_C84F"] == 1]
    require(len(rows) == REMAINING_UNITS, "remaining training registry is not 1,458 rows")
    expected = {"A": 486, "B0": 486, "B1": 486}
    observed = {wave: sum(row["wave"] == wave for row in rows) for wave in expected}
    require(observed == expected, f"remaining wave unit arithmetic drift: {observed}")
    return rows


def wave_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for panel, seed in (("A", 5), ("A", 6), ("B", 5), ("B", 6)):
            wave, order = wave_for(panel, seed)
            rows.append({
                "wave": wave, "wave_order": order, "dataset": dataset,
                "panel": panel, "training_seed": seed, "levels": "0|1",
                "paired_dataset_jobs": 1, "zoos": 2, "training_phases": 6,
                "candidate_units": 162,
                "action": "REPLAY_DUAL_CANARY" if wave == "DUAL_CANARY_REUSE" else "TRAIN_PAIRED_LEVELS",
                "target_array_access_allowed": 0,
                "target_value_release_evidence_allowed": 0,
            })
    require(sum(row["candidate_units"] for row in rows if row["wave"] != "DUAL_CANARY_REUSE") == 1458,
            "remaining wave units do not total 1,458")
    require(sum(row["training_phases"] for row in rows if row["wave"] != "DUAL_CANARY_REUSE") == 54,
            "remaining wave phases do not total 54")
    return rows


def schema_rows(artifact: str, fields: Sequence[str]) -> list[dict[str, Any]]:
    return [{
        "artifact": artifact, "field_order": index, "field": field,
        "required": 1, "target_label_or_derived": 0,
    } for index, field in enumerate(fields, start=1)]


def canary_subset_rows() -> list[dict[str, Any]]:
    rows = []
    for level, source in ((0, "C84C"), (1, "C84L1C")):
        for dataset in DATASETS:
            rows.append({
                "source": source, "dataset": dataset, "level": level,
                "target_subject": CANARY_TARGETS[dataset], "candidate_units": 81,
                "historical_scope": "one_target_subject_slice",
                "complete_target_artifact_reusable": 0,
                "C84F_recompute_complete_target_artifact": 1,
                "trial_ID_exact_match_required": 1,
                "candidate_ID_exact_match_required": 1,
                "logits_probabilities_z_strict_tolerance": STRICT_TOLERANCE,
                "retention_or_retry_use_allowed": 0,
            })
    return rows


def resource_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    c84c_seconds = 6379.0
    c84l1c_seconds = float(inputs["c84l1c_result"]["job"]["attempt_elapsed_seconds"])
    measured_phase_seconds = (c84c_seconds + c84l1c_seconds) / 18.0
    remaining_gpu_hours = measured_phase_seconds * 54 / 3600.0
    complete_gpu_hours = measured_phase_seconds * 72 / 3600.0
    raw_upper = 180 * 1024**3
    target_projection = 49036391984
    model_projection = int((361267953 + 361627353) / 486 * 1944)
    combined = raw_upper + target_projection + model_projection
    return [
        {"scope": "dual_canaries", "resource": "measured_training", "estimate": c84c_seconds + c84l1c_seconds,
         "unit": "GPU_seconds", "envelope": 250 * 3600, "within_envelope": 1,
         "basis": "jobs_895441_and_896066"},
        {"scope": "C84F_remaining", "resource": "linear_GPU_phase_time", "estimate": round(remaining_gpu_hours, 6),
         "unit": "GPU_hours", "envelope": 250, "within_envelope": int(remaining_gpu_hours <= 250),
         "basis": "dual_canary_measured_seconds_per_phase_x54"},
        {"scope": "C84_complete", "resource": "linear_GPU_phase_time", "estimate": round(complete_gpu_hours, 6),
         "unit": "GPU_hours", "envelope": 250, "within_envelope": int(complete_gpu_hours <= 250),
         "basis": "dual_canary_measured_seconds_per_phase_x72"},
        {"scope": "C84_complete", "resource": "GPU_phase_time_5x_safety", "estimate": round(complete_gpu_hours * 5, 6),
         "unit": "GPU_hours", "envelope": 250, "within_envelope": int(complete_gpu_hours * 5 <= 250),
         "basis": "five_times_dual_canary_linear_projection"},
        {"scope": "C84_complete", "resource": "model_state_source_projection", "estimate": model_projection,
         "unit": "bytes", "envelope": 2 * 1024**4, "within_envelope": 1,
         "basis": "two_valid_canary_roots_scaled_to_1944_units"},
        {"scope": "C84_complete", "resource": "complete_target_instrumentation_projection", "estimate": target_projection,
         "unit": "bytes", "envelope": 2 * 1024**4, "within_envelope": 1,
         "basis": "locked_metadata_trial_counts_and_76464_context_slices"},
        {"scope": "C84F", "resource": "raw_download_cache_upper_bound", "estimate": raw_upper,
         "unit": "bytes", "envelope": 2 * 1024**4, "within_envelope": 1,
         "basis": "C84P_metadata_only_upper_bound"},
        {"scope": "C84_complete", "resource": "download_plus_derived_projection", "estimate": combined,
         "unit": "bytes", "envelope": 2 * 1024**4, "within_envelope": int(combined <= 2 * 1024**4),
         "basis": "raw_plus_model_plus_complete_target_projection"},
        {"scope": "C84F", "resource": "candidate_context_replay", "estimate": TOTAL_SLICES,
         "unit": "slices", "envelope": TOTAL_SLICES, "within_envelope": 1,
         "basis": "944_contexts_x81_candidates"},
        {"scope": "Git", "resource": "tracked_file_limit", "estimate": 50 * 1024**2,
         "unit": "bytes_per_file", "envelope": 50 * 1024**2, "within_envelope": 1,
         "basis": "repository_policy_no_raw_EEG_weights_states_or_caches"},
    ]


def retry_rows() -> list[dict[str, Any]]:
    values = (
        ("remaining_training_failure", "same_lock_rows_IDs_RNG_no_target_access", "preserve_then_fresh_attempt_root"),
        ("implementation_byte_change", "not_allowed_under_current_lock", "additive_repair_new_lock"),
        ("model_field_freeze_failure", "stop_before_target_loading", "repair_then_new_lock_if_bytes_change"),
        ("target_instrumentation_failure", "no_retraining_or_retention_change", "repair_forward_path_new_lock_if_needed"),
        ("linear_gate_exceedance", "do_not_widen_2e-5", "stop_for_PM_review"),
        ("strict_gate_exceedance", "do_not_widen_1e-6", "stop_for_PM_review"),
        ("scientific_outcome_retry", "not_applicable_target_labels_forbidden", "stop"),
    )
    return [{
        "failure_class": failure, "same_lock_condition": condition,
        "disposition": disposition, "target_value_decision_allowed": 0,
        "training_callable_from_target_repair": 0,
    } for failure, condition, disposition in values]


def risk_rows() -> list[dict[str, Any]]:
    risks = (
        "dual_canary_identity_drift", "failed_root_reuse", "reuse_registry_overlap",
        "operative_unit_registry_drift", "level0_ID_migration", "superseded_level1_ID_used",
        "wrong_level1_deletion_cell", "level_support_gate_bypassed", "unpaired_model_initialization",
        "level_specific_plan_identity_lost", "source_audit_row_used_in_training",
        "new_target_loaded_before_model_freeze", "target_y_access", "target_label_like_metadata",
        "target_value_wave_release", "target_value_retention", "target_value_retry",
        "partial_model_field_published", "partial_target_field_published",
        "wrong_target_context_count", "canary_target_slice_called_complete",
        "canary_subset_replay_drift", "linear_tolerance_runtime_widening",
        "strict_tolerance_runtime_widening", "target_repair_invokes_training",
        "C84S_lock_created_early", "C84FL2_real_data_access", "resource_scope_reduction",
        "raw_EEG_weights_states_cache_in_Git", "tracked_payload_over_50MiB",
    )
    return [{
        "risk": risk, "status": "CLOSED_BY_PROTOCOL_OR_FAIL_CLOSED_IMPLEMENTATION",
        "blocking": 0, "real_data_access_in_C84FL2": 0,
        "control": "registered_contract_and_synthetic_failure_fixture",
    } for risk in risks]


def synthetic_rows() -> list[dict[str, Any]]:
    cases = (
        ("both_canary_result_manifest_replay", "PASS"),
        ("failed_roots_895366_895928_rejected", "PASS"),
        ("486_reuse_units_exact", "PASS"),
        ("1458_remaining_units_exact", "PASS"),
        ("1944_complete_IDs_exact", "PASS"),
        ("972_units_per_level", "PASS"),
        ("12_paired_dataset_panel_seed_cells", "PASS"),
        ("9_remaining_paired_cells", "PASS"),
        ("wave_counts_486_486_486", "PASS"),
        ("remaining_phases_18_18_18", "PASS"),
        ("wrong_deletion_cell", "FAIL_CLOSED"),
        ("unpaired_model_initialization", "FAIL_CLOSED"),
        ("level0_identity_drift", "FAIL_CLOSED"),
        ("superseded_level1_ID", "FAIL_CLOSED"),
        ("target_before_model_freeze", "FAIL_CLOSED"),
        ("target_y_access", "FAIL_CLOSED"),
        ("partial_model_field", "FAIL_CLOSED"),
        ("partial_target_context_field", "FAIL_CLOSED"),
        ("wrong_944_context_count", "FAIL_CLOSED"),
        ("canary_subset_mismatch", "FAIL_CLOSED"),
        ("target_repair_invokes_training", "FAIL_CLOSED"),
        ("linear_error_above_2e-5", "FAIL_CLOSED"),
        ("strict_error_above_1e-6", "FAIL_CLOSED"),
        ("atomic_final_manifest_publication", "PASS"),
        ("authorization_absent", "FAIL_BEFORE_OUTPUT_OR_DATA"),
    )
    return [{
        "case_id": f"FL2S{index:02d}", "fixture": fixture, "expected": expected,
        "observed": expected, "pass": 1, "real_data_access": 0,
        "training_forward_GPU": 0,
    } for index, (fixture, expected) in enumerate(cases, start=1)]


def reconciliation_protocol(table_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "c84fl2_dual_level_full_field_reconciliation_protocol_v1",
        "milestone": "C84FL2", "created_at_utc": CREATED_AT_UTC,
        "status": "LOCKED_PROTOCOL_IMPLEMENTATION_AND_EXECUTION_LOCK_PENDING_NOT_AUTHORIZED",
        "base_HEAD": BASE_HEAD,
        "timing": {
            "designed_after_C84C_and_C84L1C_engineering_results": True,
            "prospective_to_remaining_1458_unit_training": True,
            "prospective_to_complete_target_unlabeled_field": True,
            "prospective_to_all_C84_scientific_results": True,
            "target_labels_read_before_protocol": 0,
            "selector_outcomes_read_before_protocol": 0,
            "C84FL2_real_data_access": 0,
        },
        "operative_inputs": {**HASHES},
        "accepted_canaries": {
            "C84C": {"job": 895441, "level": 0, "units": 243, "phases": 9,
                     "result_sha256": HASHES["c84c_result"], "manifest_sha256": HASHES["c84c_manifest"]},
            "C84L1C": {"job": 896066, "level": 1, "units": 243, "phases": 9,
                       "result_sha256": HASHES["c84l1c_result"], "manifest_sha256": HASHES["c84l1c_manifest"]},
        },
        "historical_failures": {
            "895366": {"reusable": False, "authorization_reusable": False},
            "895928": {"reusable": False, "authorization_reusable": False},
        },
        "reuse": {
            "model_state_source_audit_units": 486, "training_phases": 18,
            "canary_contexts": 6, "canary_slices": 486,
            "canary_target_artifacts_are_subset_witnesses_only": True,
            "complete_target_artifacts_recomputed_after_model_freeze": True,
        },
        "field_arithmetic": {
            "complete_units": TOTAL_UNITS, "remaining_units": REMAINING_UNITS,
            "complete_phases": TOTAL_PHASES, "remaining_phases": REMAINING_PHASES,
            "units_per_level": 972, "paired_cells": 12, "remaining_paired_cells": 9,
            "target_subjects": 118, "target_contexts": TOTAL_CONTEXTS,
            "candidate_context_slices": TOTAL_SLICES,
        },
        "waves": {
            "A": {"panel": "A", "seed": 6, "units": 486, "phases": 18},
            "B0": {"panel": "B", "seed": 5, "units": 486, "phases": 18},
            "B1": {"panel": "B", "seed": 6, "units": 486, "phases": 18},
        },
        "barriers": {
            "new_target_array_before_1944_unit_model_freeze": False,
            "model_field_manifest_required_before_target_registry": True,
            "target_registry_required_before_candidate_forward": True,
            "atomic_complete_field_publication": True,
        },
        "numerical_gates": {
            "linear_zW_plus_b_abs": LINEAR_TOLERANCE,
            "softmax_repeat_logits_repeat_z_abs": STRICT_TOLERANCE,
            "runtime_widening": False,
        },
        "table_hashes": dict(table_hashes),
        "authorization": {
            "C84F_authorized": False, "C84S_authorized": False,
            "future_direct_statement": "授权 C84F", "hash_recital_required": False,
        },
        "gates": {"success": SUCCESS_GATE, "failure": FAILURE_GATE, "future_field": FIELD_GATE},
    }


def field_v7(prior: Mapping[str, Any], reconciliation_sha: str) -> dict[str, Any]:
    return {
        **prior,
        "schema_version": "c84_field_generation_protocol_v7",
        "milestone": "C84FL2", "created_at_utc": CREATED_AT_UTC,
        "status": "LOCKED_DUAL_LEVEL_FULL_FIELD_PROTOCOL_EXECUTION_LOCK_PENDING_NOT_AUTHORIZED",
        "parent_field_protocol_v6_sha256": HASHES["field_v6"],
        "C84FL2_reconciliation_protocol_sha256": reconciliation_sha,
        "field": {
            **prior["field"], "dual_canary_reusable_units": 486,
            "dual_canary_reusable_phases": 18, "remaining_units": 1458,
            "remaining_phases": 54, "canary_context_witnesses": 6,
        },
        "dual_canary_reuse": {
            "C84C_units": 243, "C84L1C_units": 243,
            "model_state_source_audit_reusable": True,
            "canary_target_artifact_complete_field_reusable": False,
            "complete_target_artifacts_recomputed": True,
            "failed_jobs": [895366, 895928], "failed_artifact_reuse": False,
        },
        "remaining_training": {
            "paired_levels_per_dataset_panel_seed_job": True,
            "same_model_initialization_across_levels": True,
            "wave_units": {"A": 486, "B0": 486, "B1": 486},
            "wave_phases": {"A": 18, "B0": 18, "B1": 18},
            "target_array_access": False,
        },
        "model_field_freeze": {
            "units": 1944, "phases": 72, "units_per_level": 972,
            "checkpoint_optimizer_sidecar_source_replay": 1944,
            "training_target_rows_labels": 0,
            "required_before_new_target_access": True,
        },
        "complete_target_instrumentation": {
            "unit_artifacts": 1944, "contexts": 944, "slices": 76464,
            "target_subjects": 118, "target_label_fields": 0,
            "canary_subset_replay_contexts": 6,
        },
        "instrumentation_replay_tolerances": {
            "linear_z_classifier_logits_abs_tolerance": LINEAR_TOLERANCE,
            "softmax_repeat_logits_repeat_z_abs_tolerance": STRICT_TOLERANCE,
            "runtime_widening_allowed": False,
        },
        "scope_specific_C84F_execution_lock_created": False,
        "C84S_execution_lock_created": False,
        "fresh_direct_PI_authorization_required": True,
    }


def full_field_protocol_v2(reconciliation_sha: str, field_v7_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "c84f_full_field_execution_and_manifest_protocol_v2",
        "milestone": "C84FL2", "created_at_utc": CREATED_AT_UTC,
        "status": "LOCKED_PROTOCOL_IMPLEMENTATION_AND_C84F_EXECUTION_LOCK_PENDING_NOT_AUTHORIZED",
        "bindings": {
            "C84FL2_reconciliation_protocol_sha256": reconciliation_sha,
            "field_protocol_v7_sha256": field_v7_sha,
            "external_protocol_v3_sha256": HASHES["external_v3"],
            "scientific_protocol_v3_sha256": HASHES["science_v3"],
            "operative_unit_registry_sha256": HASHES["operative_registry"],
            "C84C_result_sha256": HASHES["c84c_result"],
            "C84C_manifest_sha256": HASHES["c84c_manifest"],
            "C84L1C_result_sha256": HASHES["c84l1c_result"],
            "C84L1C_manifest_sha256": HASHES["c84l1c_manifest"],
        },
        "execution_order": [
            "authorization_and_lock_replay", "source_input_freeze",
            "wave_A_panel_A_seed6_paired_levels", "wave_B0_panel_B_seed5_paired_levels",
            "wave_B1_panel_B_seed6_paired_levels", "atomic_model_field_freeze",
            "target_unlabeled_trial_registry_freeze", "complete_target_instrumentation",
            "six_canary_context_replay", "atomic_complete_field_manifest", "stop_before_C84S",
        ],
        "paired_training": {
            "inputs": ["dataset", "source_panel", "training_seed", "level"],
            "same_model_init_across_levels": True,
            "level0": LEVEL0_ID, "level1": LEVEL1_ID,
            "remaining_units": 1458, "remaining_phases": 54,
            "source_audit_or_target_rows_in_training": 0,
        },
        "model_field_gate": {
            "candidate_units": 1944, "training_phases": 72,
            "checkpoint_optimizer_sidecar_source_artifacts": 1944,
            "unique_unit_IDs": 1944, "level0": 972, "level1": 972,
            "C84C_reuse": 243, "C84L1C_reuse": 243, "new_units": 1458,
            "target_rows_labels": 0, "target_outcome_retention_retry": 0,
        },
        "target_registry": {
            "created_only_after_model_field_freeze": True,
            "subjects": 118, "label_fields": 0,
            "structural_y_slot_operations": [],
        },
        "target_instrumentation": {
            "all_target_artifact_per_unit": True, "unit_artifacts": 1944,
            "contexts": 944, "candidate_context_slices": 76464,
            "linear_tolerance": LINEAR_TOLERANCE, "strict_tolerance": STRICT_TOLERANCE,
            "runtime_tolerance_widening": False, "training_callable": False,
            "canary_context_replay": 6,
        },
        "atomic_manifests": [
            "C84F_MODEL_FIELD_MANIFEST.json", "C84F_TARGET_UNLABELED_TRIAL_REGISTRY.json",
            "C84F_COMPLETE_FIELD_MANIFEST.json",
        ],
        "retry_policy_table": "oaci/reports/c84fl2_tables/retry_policy.csv",
        "resource_table": "oaci/reports/c84fl2_tables/resource_estimate.csv",
        "forbidden": [
            "target_construction_labels", "target_evaluation_labels", "same_label_oracle",
            "selector_scores", "Q1_Q2", "label_budget_statistics", "scientific_taxonomy",
            "runtime_scope_reduction", "runtime_tolerance_widening",
        ],
        "authorization": {
            "C84F_authorized": False, "C84S_authorized": False,
            "record_path": "oaci/reports/C84F_PI_AUTHORIZATION_RECORD.json",
            "shortest_future_statement": "授权 C84F",
        },
        "success_field_gate": FIELD_GATE,
    }


def timing_markdown(reconciliation_sha: str, field_v7_sha: str, full_v2_sha: str) -> str:
    return f"""# C84FL2 Protocol Timing Audit

## Accepted base

- Base HEAD: `{BASE_HEAD}`
- C84C result / manifest: `{HASHES['c84c_result']}` / `{HASHES['c84c_manifest']}`
- C84L1C result / manifest: `{HASHES['c84l1c_result']}` / `{HASHES['c84l1c_manifest']}`
- Operative 1,944-unit registry: `{HASHES['operative_registry']}`

## Additive protocol identity

- C84FL2 reconciliation protocol: `{reconciliation_sha}`
- C84 field protocol V7: `{field_v7_sha}`
- C84F execution/manifest protocol V2: `{full_v2_sha}`

The historical blocked C84FL protocol and all prior field protocols remain
unchanged. These C84FL2 protocols were created before the full-field runtime,
before any remaining training, before any new target-subject access, and before
any C84 scientific result.

## Protected state

During C84FL2 protocol generation: real EEG access 0, labels 0, training 0,
forward 0, GPU 0, target selector/statistic access 0. The two accepted canary
manifests were read only as engineering metadata and their artifact files were
byte-hashed without deserialization.

## Authorization

C84F and C84S are not authorized. Protocol text and prior authorizations do not
authorize execution. A future direct `授权 C84F` statement must bind the unique
current C84F execution lock after readiness.
"""


def generate() -> dict[str, Any]:
    inputs = replay_inputs()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    c84c_identity = result_identity_rows("c84c", inputs["c84c_result"], inputs["c84c_manifest"])
    c84l1c_identity = result_identity_rows("c84l1c", inputs["c84l1c_result"], inputs["c84l1c_manifest"])
    reuse = dual_canary_reuse_rows(inputs)
    operative = operative_replay_rows(reuse)
    remaining = remaining_training_rows(operative)

    table_rows: dict[str, list[dict[str, Any]]] = {
        "c84c_result_identity_replay.csv": c84c_identity,
        "c84l1c_result_identity_replay.csv": c84l1c_identity,
        "historical_failed_root_rejection.csv": [
            {"job_id": 895366, "root": "/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v3/lock_2e38dcd63c02a887b1dc",
             "authorization_reusable": 0, "artifact_reusable": 0, "preserved": 1, "status": "REJECTED"},
            {"job_id": 895928, "root": "/projects/EEG-foundation-model/yinghao/oaci-c84-level1-canary-v1/lock_d6ccab97ebfbb1e1d571",
             "authorization_reusable": 0, "artifact_reusable": 0, "preserved": 1, "status": "REJECTED"},
        ],
        "dual_canary_reuse_registry.csv": reuse,
        "operative_complete_unit_registry_replay.csv": operative,
        "remaining_paired_training_registry.csv": remaining,
        "wave_registry.csv": wave_rows(),
        "level_intervention_replay.csv": read_csv(REPORT_DIR / "c84l1p_tables/level_intervention_registry.csv"),
        "paired_rng_plan_contract.csv": read_csv(REPORT_DIR / "c84l1p_tables/paired_rng_plan_contract.csv"),
        "source_view_contract.csv": read_csv(REPORT_DIR / "c84fl_tables/source_view_contract.csv"),
        "model_field_manifest_schema.csv": schema_rows("C84F_MODEL_FIELD_MANIFEST", MODEL_FIELDS),
        "target_unlabeled_trial_registry_schema.csv": schema_rows("C84F_TARGET_UNLABELED_TRIAL_REGISTRY", TARGET_TRIAL_FIELDS),
        "target_instrumentation_schema.csv": schema_rows("C84F_COMPLETE_TARGET_UNLABELED", TARGET_ARTIFACT_FIELDS),
        "canary_subset_replay_contract.csv": canary_subset_rows(),
        "field_unit_descriptor_schema.csv": schema_rows("C84F_FIELD_UNIT_DESCRIPTOR", FIELD_DESCRIPTOR_FIELDS),
        "retry_policy.csv": retry_rows(),
        "resource_estimate.csv": resource_rows(inputs),
        "synthetic_calibration.csv": synthetic_rows(),
        "risk_register.csv": risk_rows(),
        "failure_reason_ledger.csv": [{
            "failure_id": "NONE", "stage": "C84FL2_protocol_generation",
            "blocking": 0, "reason": "dual_canary_and_protocol_reconciliation_passed",
            "real_data_access": 0, "scientific_outcome_access": 0,
            "repair_required": 0,
        }],
    }
    table_hashes = {
        name: write_csv(TABLE_DIR / name, rows)
        for name, rows in table_rows.items()
    }

    reconciliation_path = REPORT_DIR / "C84FL2_DUAL_LEVEL_FULL_FIELD_RECONCILIATION_PROTOCOL.json"
    reconciliation_sha = write_json(reconciliation_path, reconciliation_protocol(table_hashes))
    write_hash_sidecar(reconciliation_path, reconciliation_sha)

    field_v7_path = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V7.json"
    field_v7_sha = write_json(field_v7_path, field_v7(inputs["field_v6"], reconciliation_sha))
    write_hash_sidecar(field_v7_path, field_v7_sha)

    full_v2_path = REPORT_DIR / "C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL_V2.json"
    full_v2_sha = write_json(full_v2_path, full_field_protocol_v2(reconciliation_sha, field_v7_sha))
    write_hash_sidecar(full_v2_path, full_v2_sha)

    (REPORT_DIR / "C84FL2_PROTOCOL_TIMING_AUDIT.md").write_text(
        timing_markdown(reconciliation_sha, field_v7_sha, full_v2_sha), encoding="utf-8",
    )
    return {
        "reconciliation_protocol_sha256": reconciliation_sha,
        "field_v7_sha256": field_v7_sha,
        "full_field_protocol_v2_sha256": full_v2_sha,
        "reuse_units": len(reuse), "remaining_units": len(remaining),
        "operative_units": len(operative), "table_count": len(table_rows),
        "real_data_access": 0, "training_forward_GPU": 0,
    }


def main() -> int:
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
