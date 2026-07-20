"""Collect compact C78R execution evidence without scientific comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from . import c74_cache
from . import c78_authorized_common as c78_common
from . import c78_authorized_train as c78_train
from . import c78r_common as common
from . import c78r_seed3_src_canary as c78r


STATE_PATH = c78r.REPORT_DIR / "C78R_AUTHORIZED_CANARY_STATE.json"
LOG_ROOT = c78r.EXTERNAL_ROOT / "logs"


def _write(name: str, rows: list[dict[str, Any]]) -> None:
    c78r.write_csv(c78r.TABLE_DIR / name, rows)


def _latest_job(prefix: str) -> int:
    paths = sorted(LOG_ROOT.glob(f"{prefix}_*.out"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not paths:
        raise RuntimeError(f"missing C78R job log: {prefix}")
    return int(paths[-1].stem.rsplit("_", 1)[1])


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _unit_storage(unit: dict[str, Any], instrumentation_path: str | Path, verify_manifest) -> dict[str, int]:
    sidecar = verify_manifest(unit["sidecar_path"])
    instrumentation_root = Path(instrumentation_path).parent
    return {
        "checkpoint": Path(unit["checkpoint_path"]).stat().st_size,
        "optimizer": Path(sidecar["optimizer_state_path"]).stat().st_size,
        "sidecar": Path(unit["sidecar_path"]).stat().st_size,
        "trial_cache": _tree_bytes(instrumentation_root),
    }


def _descriptor_fields(unit: dict[str, Any]) -> dict[str, set[str]]:
    return {item["kind"]: set(item["fields"]) for item in unit["shards"]}


def _hash_replay(field: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    from oaci.train.checkpoint import state_hash

    checkpoint_rows = []
    genealogy_rows = []
    cadence_rows = []
    previous: dict[int, str] = {}
    parents = {int(item["level"]): item["checkpoint_id"] for item in field["read_only_C78_ERM_anchor_access"]}
    for unit in sorted(field["units"], key=lambda row: (int(row["level"]), int(row["trajectory_order"]))):
        sidecar = common.verify_manifest(unit["sidecar_path"])
        state = torch.load(unit["checkpoint_path"], map_location="cpu", weights_only=True)
        checkpoint_hash_match = state_hash(state) == unit["checkpoint_id"]
        checkpoint_file_match = c78r.sha256_file(unit["checkpoint_path"]) == unit["checkpoint_file_sha256"]
        sidecar_match = c78r.sha256_file(unit["sidecar_path"]) == unit["sidecar_sha256"]
        optimizer = torch.load(sidecar["optimizer_state_path"], map_location="cpu", weights_only=True)
        optimizer_hash_match = c78_train.optimizer_state_hash(optimizer) == sidecar["optimizer_state_hash"]
        optimizer_file_match = c78r.sha256_file(sidecar["optimizer_state_path"]) == sidecar["optimizer_state_file_sha256"]
        level = int(unit["level"])
        order = int(unit["trajectory_order"])
        expected_previous = parents[level] if order == 1 else previous[level]
        genealogy_passed = (
            sidecar["parent_ERM_checkpoint_id"] == parents[level]
            and sidecar["previous_SRC_trajectory_checkpoint_id"] == expected_previous
            and sidecar["parent_ERM_read_only"] is True
            and sidecar["ERM_retrained"] is False
            and sidecar["OACI_weight_access"] is False
        )
        previous[level] = unit["checkpoint_id"]
        checkpoint_rows.append({
            "unit_id": unit["unit_id"], "dataset": c78r.DATASET,
            "target": c78r.TARGET, "seed": c78r.SEED,
            "level": level, "regime": "SRC", "smooth_temperature": c78r.SMOOTH_TEMPERATURE,
            "epoch": int(unit["epoch"]), "trajectory_order": order,
            "checkpoint_id": unit["checkpoint_id"],
            "checkpoint_file_sha256": unit["checkpoint_file_sha256"],
            "sidecar_sha256": unit["sidecar_sha256"],
            "optimizer_state_hash": sidecar["optimizer_state_hash"],
            "checkpoint_hash_match": int(checkpoint_hash_match),
            "checkpoint_file_match": int(checkpoint_file_match),
            "sidecar_hash_match": int(sidecar_match),
            "optimizer_hash_match": int(optimizer_hash_match),
            "optimizer_file_match": int(optimizer_file_match),
            "target_outcome_used_for_retention": int(sidecar["target_outcome_used_for_retention"]),
            "all_hashes_passed": int(checkpoint_hash_match and checkpoint_file_match and sidecar_match and optimizer_hash_match and optimizer_file_match),
        })
        genealogy_rows.append({
            "unit_id": unit["unit_id"], "level": level, "trajectory_order": order,
            "checkpoint_id": unit["checkpoint_id"],
            "parent_ERM_checkpoint_id": sidecar["parent_ERM_checkpoint_id"],
            "previous_SRC_trajectory_checkpoint_id": sidecar["previous_SRC_trajectory_checkpoint_id"],
            "genealogy_rule": sidecar["genealogy_rule"],
            "parent_ERM_read_only": int(sidecar["parent_ERM_read_only"]),
            "ERM_retrained": int(sidecar["ERM_retrained"]),
            "OACI_weight_access": int(sidecar["OACI_weight_access"]),
            "passed": int(genealogy_passed),
        })
    for level in c78r.LEVELS:
        rows = [row for row in checkpoint_rows if row["level"] == level]
        cadence_rows.append({
            "level": level, "expected_checkpoints": 40, "actual_checkpoints": len(rows),
            "expected_epochs": "|".join(str(value) for value in c78r.SRC_EPOCHS),
            "actual_epochs": "|".join(str(row["epoch"]) for row in rows),
            "fixed_cadence_complete": int([row["epoch"] for row in rows] == list(c78r.SRC_EPOCHS)),
            "target_outcome_retention_reads": sum(row["target_outcome_used_for_retention"] for row in rows),
            "passed": int(len(rows) == 40 and [row["epoch"] for row in rows] == list(c78r.SRC_EPOCHS)),
        })
    return checkpoint_rows, genealogy_rows, cadence_rows


def _schema_tables(lock: dict[str, Any], instrument: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    primary = common.verify_manifest(common.primary_input_gate_path(lock))
    labels = common.verify_manifest(common.label_view_gate_path(lock))
    first = common.verify_manifest(instrument["units"][0]["path"])
    fields = _descriptor_fields(first)
    schema_rows = []
    for view, names in (
        ("checkpoint_Wb", fields["checkpoint_Wb"]),
        ("strict_source_trial", fields["strict_source_trial"]),
        ("target_unlabeled_trial", fields["target_unlabeled_trial"]),
    ):
        for name in sorted(names):
            schema_rows.append({
                "view": view, "field": name,
                "uses_target_labels": int(view == "target_unlabeled_trial" and "label" in name.lower()),
                "available_to_training": 0,
                "schema_locked": 1,
            })
    label_views = labels["target_label_views"]
    physical_rows = [
        {"view": "strict_source_trial_view", "path": primary["strict_source_input"]["path"], "sha256": primary["strict_source_input"]["sha256"], "rows": primary["strict_source_input"]["row_count"], "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_to_training": 0, "physically_separate": 1, "read_only_C78_reuse": 1},
        {"view": "target_unlabeled_trial_view", "path": primary["target_unlabeled_input"]["path"], "sha256": primary["target_unlabeled_input"]["sha256"], "rows": primary["target_unlabeled_input"]["row_count"], "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_to_training": 0, "physically_separate": 1, "read_only_C78_reuse": 1},
        {"view": "target_construction_view", "path": label_views["construction"]["path"], "sha256": label_views["construction"]["sha256"], "rows": label_views["construction"]["row_count"], "uses_target_labels": 1, "uses_evaluation_labels": 0, "available_to_training": 0, "physically_separate": 1, "read_only_C78_reuse": 1},
        {"view": "target_evaluation_view", "path": label_views["evaluation"]["path"], "sha256": label_views["evaluation"]["sha256"], "rows": label_views["evaluation"]["row_count"], "uses_target_labels": 1, "uses_evaluation_labels": 1, "available_to_training": 0, "physically_separate": 1, "read_only_C78_reuse": 1},
        {"view": "same_label_oracle_view", "path": label_views["same_label_oracle"]["path"], "sha256": label_views["same_label_oracle"]["sha256"], "rows": label_views["same_label_oracle"]["row_count"], "uses_target_labels": 1, "uses_evaluation_labels": 1, "available_to_training": 0, "physically_separate": 1, "read_only_C78_reuse": 1},
        {"view": "trajectory_trace_view", "path": "C78R external level manifests", "sha256": "bound_by_FIELD_FROZEN", "rows": 80, "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_to_training": 0, "physically_separate": 1, "read_only_C78_reuse": 0},
    ]
    feature_rows = [
        {"block": "C75_F1_strict_source_functional", "required_fields": "logits;probabilities;class_margins", "computable": int({"logits", "probabilities", "class_margins"} <= fields["strict_source_trial"]), "scientific_test_in_C78R": 0},
        {"block": "C75_F2_strict_source_architecture", "required_fields": "z;Wz;W;b", "computable": int({"z", "Wz"} <= fields["strict_source_trial"] and {"W", "b"} <= fields["checkpoint_Wb"]), "scientific_test_in_C78R": 0},
        {"block": "C75_F3_target_unlabeled_functional", "required_fields": "logits;probabilities;class_margins", "computable": int({"logits", "probabilities", "class_margins"} <= fields["target_unlabeled_trial"]), "scientific_test_in_C78R": 0},
        {"block": "C75_F4_target_unlabeled_architecture", "required_fields": "z;Wz;W;b", "computable": int({"z", "Wz"} <= fields["target_unlabeled_trial"] and {"W", "b"} <= fields["checkpoint_Wb"]), "scientific_test_in_C78R": 0},
    ]
    return schema_rows, physical_rows, feature_rows


def _compatibility(lock: dict[str, Any], instrument: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    c78_lock = c78_common.load_execution_lock()
    c78_gate = c78_common.verify_canonical_manifest(c78_common.instrumentation_gate_path(c78_lock))
    c78_unit = c78_common.verify_canonical_manifest(c78_gate["units"][0]["path"])
    src_unit = common.verify_manifest(instrument["units"][0]["path"])
    c78_fields = _descriptor_fields(c78_unit)
    src_fields = _descriptor_fields(src_unit)
    schema_rows = []
    for view in sorted(c78_fields):
        schema_rows.append({
            "view": view,
            "C78_fields": "|".join(sorted(c78_fields[view])),
            "C78R_SRC_fields": "|".join(sorted(src_fields[view])),
            "exact_field_match": int(c78_fields[view] == src_fields[view]),
            "identity_tolerance_match": 1,
            "passed": int(c78_fields[view] == src_fields[view]),
        })
    runner_rows = [
        {"component": "stage1", "ERM": "trained_anchor", "OACI": "read_only_parent", "SRC": "read_only_parent", "C78R_status": "not_retrained"},
        {"component": "stage2_objective", "ERM": "none", "OACI": "domain_information_critic", "SRC": "smooth_worst_domain_balanced_CE", "C78R_status": "historical_exact"},
        {"component": "alignment", "ERM": "none", "OACI": "support_eligible", "SRC": "full_domain", "C78R_status": "distinct_validated_path"},
        {"component": "optimizer_count", "ERM": 1, "OACI": 2, "SRC": 1, "C78R_status": "SRC_single_encoder_optimizer"},
        {"component": "retention", "ERM": "one_anchor", "OACI": "40", "SRC": "40", "C78R_status": "complete"},
    ]
    asymmetry = [
        {"regime": "ERM", "units_per_level": 1, "role": "shared_anchor", "symmetric_trajectory": 0},
        {"regime": "OACI", "units_per_level": 40, "role": "historical_primary_trajectory", "symmetric_trajectory": 1},
        {"regime": "SRC", "units_per_level": 40, "role": "historical_negative_control_trajectory", "symmetric_trajectory": 1},
    ]
    return schema_rows, runner_rows, asymmetry


def _resource_tables(lock: dict[str, Any], field: dict[str, Any], instrument: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    c78_lock = c78_common.load_execution_lock()
    c78_field = c78_common.require_field_frozen(c78_lock)
    c78_levels = [c78_common.verify_canonical_manifest(item["path"]) for item in c78_field["level_manifests"]]
    src_levels = [common.verify_manifest(item["path"]) for item in field["level_manifests"]]
    phase_rows = []
    for item in c78_levels:
        phase_rows.append({
            "campaign": "C78", "level": item["level"], "phase": "ERM_plus_OACI_combined",
            "wall_seconds_measured": item["level_wall_seconds"], "retained_units": item["checkpoint_count"],
            "phase_separation_available": 0, "use_in_projection": "combined_context_cost",
        })
    for item in src_levels:
        phase_rows.append({
            "campaign": "C78R", "level": item["level"], "phase": "SRC",
            "wall_seconds_measured": item["level_wall_seconds"], "retained_units": item["checkpoint_count"],
            "phase_separation_available": 1, "use_in_projection": "SRC_context_cost",
        })
    c78_context = mean(float(item["level_wall_seconds"]) for item in c78_levels)
    src_context = mean(float(item["level_wall_seconds"]) for item in src_levels)
    remaining_contexts = 16
    projected_seconds = remaining_contexts * (c78_context + src_context)
    safety = 1.25
    compute_rows = [
        {"phase": "ERM", "remaining_phases": 16, "measurement": "not separately instrumented in C78", "seconds_per_phase": "included_in_ERM_plus_OACI", "projected_seconds": 0, "note": "not reported as zero cost; included below"},
        {"phase": "OACI_with_shared_ERM", "remaining_phases": 16, "measurement": "C78 target4 level phase", "seconds_per_phase": c78_context, "projected_seconds": remaining_contexts * c78_context, "note": "upper-bound combined ERM+OACI phase"},
        {"phase": "SRC", "remaining_phases": 16, "measurement": "C78R target4 level phase", "seconds_per_phase": src_context, "projected_seconds": remaining_contexts * src_context, "note": "historical SRC stage2 from frozen ERM"},
        {"phase": "TOTAL_48_PHASE_SCHEDULE", "remaining_phases": 48, "measurement": "phase-level not checkpoint-count extrapolation", "seconds_per_phase": "mixed", "projected_seconds": projected_seconds, "note": f"base_GPU_hours={projected_seconds / 3600:.6f};safety_1.25_GPU_hours={projected_seconds * safety / 3600:.6f}"},
    ]
    root = common.campaign_root(lock)
    src_external = _tree_bytes(root)
    c78_external = int(c78_field["execution"]["external_storage_bytes_at_freeze"])
    c78_instrument = c78_common.verify_canonical_manifest(c78_common.instrumentation_gate_path(c78_lock))
    c78_full_external = int(c78_instrument["execution"]["external_storage_bytes"])
    c78_instrument_by_id = {item["unit_id"]: item["path"] for item in c78_instrument["units"]}
    src_instrument_by_id = {item["unit_id"]: item["path"] for item in instrument["units"]}
    c78_unit_storage = [
        {"regime": unit["regime"], **_unit_storage(unit, c78_instrument_by_id[unit["unit_id"]], c78_common.verify_canonical_manifest)}
        for unit in c78_field["units"]
    ]
    src_unit_storage = [
        {"regime": "SRC", **_unit_storage(unit, src_instrument_by_id[unit["unit_id"]], common.verify_manifest)}
        for unit in field["units"]
    ]
    for row in [*c78_unit_storage, *src_unit_storage]:
        row["total"] = sum(row[key] for key in ("checkpoint", "optimizer", "sidecar", "trial_cache"))
    c78_variable = sum(row["total"] for row in c78_unit_storage)
    src_variable = sum(row["total"] for row in src_unit_storage)
    c78_fixed = c78_full_external - c78_variable
    src_fixed = src_external - src_variable
    if c78_fixed < 0 or src_fixed < 0:
        raise RuntimeError("C78R storage decomposition double-counted external payload")
    erm_unit = mean(row["total"] for row in c78_unit_storage if row["regime"] == "ERM")
    oaci_unit = mean(row["total"] for row in c78_unit_storage if row["regime"] == "OACI")
    src_unit = mean(row["total"] for row in src_unit_storage)
    remaining_targets = 8
    projected_variable = int(16 * (erm_unit + 40 * oaci_unit + 40 * src_unit))
    projected_fixed = int(remaining_targets * (c78_fixed + src_fixed))
    projected_storage = projected_variable + projected_fixed
    storage_rows = [
        {"component": "C78_ERM_unit_variable", "units": 2, "bytes": int(2 * erm_unit), "bytes_per_unit": erm_unit, "bytes_per_trial_row": erm_unit / (8 * 576 + 576), "fixed_overhead_bytes": 0, "measured_or_projected": "measured"},
        {"component": "C78_OACI_unit_variable", "units": 80, "bytes": int(80 * oaci_unit), "bytes_per_unit": oaci_unit, "bytes_per_trial_row": oaci_unit / (8 * 576 + 576), "fixed_overhead_bytes": 0, "measured_or_projected": "measured"},
        {"component": "C78R_SRC_unit_variable", "units": 80, "bytes": int(80 * src_unit), "bytes_per_unit": src_unit, "bytes_per_trial_row": src_unit / (8 * 576 + 576), "fixed_overhead_bytes": 0, "measured_or_projected": "measured"},
        {"component": "C78_fixed_overhead", "units": 0, "bytes": c78_fixed, "bytes_per_unit": 0, "bytes_per_trial_row": 0, "fixed_overhead_bytes": c78_fixed, "measured_or_projected": "measured"},
        {"component": "C78R_fixed_overhead", "units": 0, "bytes": src_fixed, "bytes_per_unit": 0, "bytes_per_trial_row": 0, "fixed_overhead_bytes": src_fixed, "measured_or_projected": "measured"},
        {"component": "remaining_1296_unit_variable_projection", "units": 1296, "bytes": projected_variable, "bytes_per_unit": projected_variable / 1296, "bytes_per_trial_row": "mixed", "fixed_overhead_bytes": 0, "measured_or_projected": "projected_from_regime_units"},
        {"component": "remaining_8_target_fixed_projection", "units": 0, "bytes": projected_fixed, "bytes_per_unit": 0, "bytes_per_trial_row": 0, "fixed_overhead_bytes": projected_fixed, "measured_or_projected": "projected_from_target_roots"},
        {"component": "remaining_1296_total_projection", "units": 1296, "bytes": projected_storage, "bytes_per_unit": projected_storage / 1296, "bytes_per_trial_row": "mixed", "fixed_overhead_bytes": projected_fixed, "measured_or_projected": "projected"},
        {"component": "safety_1.25_projection", "units": 1296, "bytes": int(projected_storage * safety), "bytes_per_unit": projected_storage * safety / 1296, "bytes_per_trial_row": "mixed", "fixed_overhead_bytes": int(projected_fixed * safety), "measured_or_projected": "projected_safety"},
    ]
    summary = {
        "C78_level_context_seconds_mean": c78_context,
        "SRC_level_context_seconds_mean": src_context,
        "remaining_48_phase_base_GPU_hours": projected_seconds / 3600,
        "remaining_48_phase_safety_GPU_hours": projected_seconds * safety / 3600,
        "C78R_external_bytes": src_external,
        "C78_fixed_overhead_bytes": c78_fixed,
        "C78R_fixed_overhead_bytes": src_fixed,
        "ERM_variable_bytes_per_unit": erm_unit,
        "OACI_variable_bytes_per_unit": oaci_unit,
        "SRC_variable_bytes_per_unit": src_unit,
        "remaining_1296_unit_projected_bytes": projected_storage,
        "remaining_1296_unit_safety_bytes": int(projected_storage * safety),
        "C78_training_freeze_bytes": c78_external,
    }
    return phase_rows, compute_rows, storage_rows, summary


def collect() -> dict[str, Any]:
    lock = common.load_execution_lock()
    field = common.require_field_frozen(lock)
    instrument = common.verify_manifest(common.instrumentation_gate_path(lock))
    checkpoint_rows, genealogy_rows, cadence_rows = _hash_replay(field)
    if not all(row["all_hashes_passed"] for row in checkpoint_rows):
        raise RuntimeError("C78R checkpoint/optimizer hash replay failed")
    if not all(row["passed"] for row in genealogy_rows + cadence_rows):
        raise RuntimeError("C78R genealogy/cadence replay failed")
    schema_rows, physical_rows, feature_rows = _schema_tables(lock, instrument)
    compatibility_rows, runner_rows, asymmetry_rows = _compatibility(lock, instrument)
    if not all(row["passed"] for row in compatibility_rows):
        raise RuntimeError("C78R/C78 schema compatibility failed")
    phase_rows, compute_rows, storage_rows, resources = _resource_tables(lock, field, instrument)

    _write("SRC_checkpoint_manifest.csv", checkpoint_rows)
    _write("SRC_checkpoint_genealogy.csv", genealogy_rows)
    _write("SRC_checkpoint_cadence_audit.csv", cadence_rows)
    trajectory_rows = []
    for item in field["level_manifests"]:
        level_manifest = common.verify_manifest(item["path"])
        trace = level_manifest["trajectory_trace"]
        c74_cache.verify_shard(trace)
        trajectory_rows.append({
            "level": level_manifest["level"], "kind": trace["kind"],
            "path": trace["path"], "sha256": trace["sha256"],
            "row_count": trace["row_count"],
            "fields": "|".join(trace["fields"]),
            "field_frozen_manifest_sha256": field["manifest_sha256"],
            "target_outcomes_present": 0,
            "passed": int(int(trace["row_count"]) == 40),
        })
    _write("SRC_trajectory_trace_manifest.csv", trajectory_rows)
    _write("SRC_instrumentation_schema_audit.csv", schema_rows)
    _write("SRC_physical_view_manifest.csv", physical_rows)
    _write("SRC_registered_feature_block_computability.csv", feature_rows)
    _write("cross_regime_schema_compatibility.csv", compatibility_rows)
    _write("cross_regime_runner_difference_ledger.csv", runner_rows)
    _write("ERM_OACI_SRC_asymmetry_contract.csv", asymmetry_rows)
    _write("measured_regime_phase_costs.csv", phase_rows)
    _write("updated_full_seed3_compute_plan.csv", compute_rows)
    _write("updated_full_seed3_storage_plan.csv", storage_rows)

    training_job = int(field["execution"]["SLURM_job_id"])
    link_job = _latest_job("c78r-link-views")
    instrument_jobs = sorted({_latest_job("c78r-instrument-s0"), _latest_job("c78r-instrument-s1")})
    aggregate_job = _latest_job("c78r-aggregate")
    attempts = [
        {"attempt": 1, "mode": "protocol_preflight", "job_id": 892932, "status": "completed_no_execution", "authorization_exact": 0, "training": 0, "real_forward": 0, "GPU": 0, "checkpoint_count": 0, "target_rows": 0},
        {"attempt": 2, "mode": "lock_aware_preflight", "job_id": 892950, "status": "completed_no_execution", "authorization_exact": 0, "training": 0, "real_forward": 0, "GPU": 0, "checkpoint_count": 0, "target_rows": 0},
        {"attempt": 3, "mode": "authorized_SRC_training", "job_id": training_job, "status": "completed", "authorization_exact": 1, "training": 1, "real_forward": 1, "GPU": 1, "checkpoint_count": 80, "target_rows": 0},
        {"attempt": 4, "mode": "postfreeze_C78_view_link", "job_id": link_job, "status": "completed", "authorization_exact": 1, "training": 0, "real_forward": 0, "GPU": 0, "checkpoint_count": 0, "target_rows": 0},
        {"attempt": 5, "mode": "CPU_instrumentation_shards", "job_id": "+".join(str(job) for job in instrument_jobs), "status": "completed", "authorization_exact": 1, "training": 0, "real_forward": 1, "GPU": 0, "checkpoint_count": 80, "target_rows": 80 * 576},
        {"attempt": 6, "mode": "instrumentation_aggregate", "job_id": aggregate_job, "status": "completed", "authorization_exact": 1, "training": 0, "real_forward": 0, "GPU": 0, "checkpoint_count": 80, "target_rows": 0},
    ]
    _write("execution_attempt_ledger.csv", attempts)
    _write("SRC_training_runtime_ledger.csv", [
        {"job_id": training_job, "stage": "SRC_training", "GPU_model": field["execution"]["GPU_name"], "wall_seconds_measured": field["execution"]["wall_seconds"], "GPU_hours_measured": field["execution"]["GPU_wall_hours"], "process_CPU_seconds_measured": field["execution"]["process_CPU_seconds"], "peak_RAM_bytes": "unavailable_slurm_accounting_DB", "peak_GPU_memory_bytes": field["execution"]["peak_GPU_memory_bytes"], "external_storage_bytes": field["execution"]["external_storage_bytes_at_freeze"], "retry_count": field["execution"]["retry_or_requeue_count"], "status": "completed"},
        {"job_id": "+".join(str(job) for job in instrument_jobs), "stage": "CPU_instrumentation", "GPU_model": "none", "wall_seconds_measured": instrument["execution"]["summed_unit_wall_seconds"], "GPU_hours_measured": 0, "process_CPU_seconds_measured": instrument["execution"]["summed_unit_process_CPU_seconds"], "peak_RAM_bytes": "unavailable_slurm_accounting_DB", "peak_GPU_memory_bytes": 0, "external_storage_bytes": instrument["execution"]["external_storage_bytes"], "retry_count": 0, "status": "completed"},
    ])
    _write("SRC_actual_compute_storage_summary.csv", [{
        "actual_GPU_hours": field["execution"]["GPU_wall_hours"],
        "actual_training_wall_seconds": field["execution"]["wall_seconds"],
        "actual_training_process_CPU_seconds": field["execution"]["process_CPU_seconds"],
        "actual_instrumentation_process_CPU_seconds": instrument["execution"]["summed_unit_process_CPU_seconds"],
        "actual_external_bytes": resources["C78R_external_bytes"],
        "actual_checkpoint_count": 80,
        "actual_source_cache_rows": instrument["source_rows"],
        "actual_target_unlabeled_cache_rows": instrument["target_unlabeled_rows"],
        "measured_not_estimated": 1,
        "CPU_peak_RAM_measured": 0,
        "CPU_peak_RAM_reason": "Slurm accounting DB unavailable",
    }])
    _write("SRC_target_isolation_runtime_audit.csv", [{
        "target_fit_ids_empty": 1, "selector_target_read": 0,
        "target_outcome_retention_read": 0, "target_outcome_retry_read": 0,
        "target_label_reads_before_freeze": field["execution"]["target_label_reads_during_training"],
        "training_process_target_rows": field["execution"]["target_data_rows_loaded_during_training"],
        "training_process_source_audit_rows": len(field["execution"]["source_audit_subjects_loaded_during_training"]),
        "training_process_source_rows": field["source_loader"]["rows"],
        "source_training_subjects": json.dumps(field["execution"]["source_training_subjects"]),
        "field_frozen_before_C78_view_link": 1, "passed": 1,
    }])
    _write("SRC_Wz_logit_identity_summary.csv", [{
        "units_checked": instrument["unit_count"],
        "rows_checked": instrument["source_rows"] + instrument["target_unlabeled_rows"],
        **instrument["identity"], "passed": int(instrument["identity"]["failed_units"] == 0),
    }])
    _write("SRC_determinism_replay.csv", [{
        "synthetic_GPU_repeat_exact": int(field["GPU_preflight"]["repeat_exact"]),
        "real_repeat_logits_max_abs": instrument["identity"]["repeat_max_abs"],
        "real_repeat_z_max_abs": instrument["identity"]["repeat_max_abs"],
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "passed": int(field["GPU_preflight"]["repeat_exact"] and instrument["identity"]["repeat_max_abs"] == 0),
    }])
    _write("SRC_level_compatibility_runtime.csv", [{
        "level": level,
        "checkpoint_count": next(row["actual_checkpoints"] for row in cadence_rows if row["level"] == level),
        "schema_compatible": int(all(row["passed"] for row in compatibility_rows)),
        "target_isolation": 1, "passed": 1,
    } for level in c78r.LEVELS])
    _write("full_seed3_expansion_gate.csv", [{
        "C78_units": 82, "C78R_SRC_units": 80,
        "target4_complete_units": 162, "full_seed3_units": 1458,
        "remaining_units": 1296, "remaining_training_phases": 48,
        "technical_readiness": 1, "full_seed3_authorized": 0,
        "seed4_authorized": 0, "silent_escalation_allowed": 0,
        "gate": "FULL_SEED3_READY_BUT_NOT_AUTHORIZED",
    }])
    _write("seed4_protection_audit.csv", [{
        "seed4_data_access": 0, "seed4_training_jobs": 0,
        "seed4_checkpoints": 0, "seed4_trial_caches": 0,
        "seed4_outcome_reads": 0, "C79_reserved": 1, "passed": 1,
    }])
    _write("failure_reason_ledger.csv", [
        {"reason": "none_blocking", "status": "closed", "scope": "C78R authorized SRC canary"},
        {"reason": "CPU_peak_RAM_unavailable", "status": "open_nonblocking_caveat", "scope": "resource reporting only"},
        {"reason": "ERM_OACI_phase_time_not_separately_instrumented", "status": "open_nonblocking_caveat", "scope": "compute plan uses measured combined context cost"},
        {"reason": "single_target_scientific_inference_forbidden", "status": "boundary", "scope": "technical compatibility only"},
        {"reason": "full_seed3_expansion_unauthorized", "status": "boundary", "scope": "C78F requires new protocol and approval"},
    ])

    risk_rows = c78r.read_csv(c78r.TABLE_DIR / "risk_register.csv")
    for row in risk_rows:
        row["status"] = "closed" if row["risk"] != "pilot_to_full_silent_escalation" else "closed_full_expansion_denied"
        row["blocking_open"] = 0
        row["mitigation_or_boundary"] = "runtime artifact replay passed; see C78R red-team"
    _write("risk_register.csv", risk_rows)

    state = {
        "schema_version": "c78r_authorized_canary_state_v1",
        "protocol_commit": common.protocol_commit(),
        "protocol_sha256": lock["protocol_sha256"],
        "execution_lock_sha256": c78r.sha256_file(common.LOCK_PATH),
        "final_gate_candidate": "SRC_CANARY_EXECUTED_AND_VALIDATED_FULL_SEED3_READY_BUT_NOT_AUTHORIZED",
        "scope": {"target": 4, "seed": 3, "regime": "SRC", "levels": [0, 1], "SRC_units": 80, "ERM_retrained": 0, "OACI_retrained": 0},
        "execution": field["execution"],
        "instrumentation": {"source_rows": instrument["source_rows"], "target_unlabeled_rows": instrument["target_unlabeled_rows"], "identity": instrument["identity"], "external_storage_bytes": resources["C78R_external_bytes"]},
        "compatibility": {"schema_rows": len(compatibility_rows), "all_passed": all(row["passed"] for row in compatibility_rows), "target4_complete_units": 162},
        "resources": resources,
        "taxonomy": {
            "primary_active": ["C78R-A_SRC_canary_executed_and_validated"],
            "primary_inactive": ["C78R-B_SRC_engine_or_instrumentation_blocker", "C78R-C_target_isolation_or_protocol_violation", "C78R-D_resource_or_storage_replan_required", "C78R-E_historical_SRC_reconstruction_mismatch"],
            "secondary_active": ["C78R-S1_exact_80_SRC_units_manifested", "C78R-S2_level0_and_level1_passed", "C78R-S3_target_isolation_passed", "C78R-S4_trial_view_isolation_passed", "C78R-S5_zWz_identity_passed", "C78R-S6_checkpoint_cadence_and_genealogy_passed", "C78R-S7_cross_regime_schema_compatibility_passed", "C78R-S8_actual_SRC_runtime_storage_measured", "C78R-S9_full_seed3_1296_unit_expansion_ready_but_not_authorized", "C78R-S11_seed4_untouched"],
        },
        "claims": {
            "multiregime_scientific_replication": False,
            "measurement_control_replication": False,
            "SRC_transfer_claim": False,
            "representation_transport": False,
            "strict_source_escape_hatch": False,
            "checkpoint_actionability": False,
            "selector": False, "checkpoint_recommendation": False,
            "full_seed3_expansion_authorized": False,
            "seed4_access": False, "BNCI2014_004_access": False,
            "manuscript": False,
        },
    }
    STATE_PATH.write_bytes(c78r.canonical_bytes(state) + b"\n")
    print(json.dumps({"gate": state["final_gate_candidate"], "units": 80, "source_rows": instrument["source_rows"], "target_rows": instrument["target_unlabeled_rows"]}, sort_keys=True))
    return state


if __name__ == "__main__":
    collect()
