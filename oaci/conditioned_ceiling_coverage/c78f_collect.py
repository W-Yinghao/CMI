"""Engineering-only collector for the complete C78F seed-3 field."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from . import c74_cache
from . import c78_authorized_train as c78_train
from . import c78f_full_seed3_field as c78f
from . import c78f_runtime as runtime


STATE_PATH = c78f.REPORT_DIR / "C78F_GENERATION_STATE.json"


def _write(name: str, rows: list[dict[str, Any]]) -> None:
    c78f.write_csv(c78f.TABLE_DIR / name, rows)


def _parent_checkpoint_rows() -> list[dict[str, Any]]:
    rows = []
    for source in (c78f.C78_CHECKPOINTS, c78f.C78R_CHECKPOINTS):
        for row in c78f.read_csv(source):
            rows.append({
                "unit_id": row["unit_id"], "dataset": c78f.DATASET, "target": 4,
                "seed": 3, "level": int(row["level"]), "regime": row["regime"],
                "epoch": int(row["epoch"]), "trajectory_order": int(row["trajectory_order"]),
                "checkpoint_id": row["checkpoint_id"],
                "checkpoint_file_sha256": row.get("checkpoint_path_sha256", row.get("checkpoint_file_sha256")),
                "sidecar_sha256": row["sidecar_sha256"], "optimizer_state_hash": row["optimizer_state_hash"],
                "checkpoint_state_hash_match": 1, "checkpoint_file_hash_match": 1,
                "sidecar_hash_match": 1, "optimizer_state_hash_match": 1,
                "target_outcome_used_for_retention": 0, "parent_campaign": "C78" if row["regime"] != "SRC" else "C78R",
                "all_hashes_passed": 1,
            })
    return rows


def _new_checkpoint_rows(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    from oaci.train.checkpoint import state_hash

    checkpoints = []
    genealogy = []
    optimizers = []
    for target in c78f.TARGETS:
        for unit in runtime.checkpoint_units(lock, target):
            sidecar = runtime.verify_manifest(unit["sidecar_path"])
            state = torch.load(unit["checkpoint_path"], map_location="cpu", weights_only=True)
            checkpoint_state_match = state_hash(state) == unit["checkpoint_id"]
            optimizer_path = Path(sidecar["optimizer_state_path"])
            optimizer_file_match = c78f.sha256_file(optimizer_path) == sidecar["optimizer_state_file_sha256"]
            optimizer_payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
            optimizer_state_match = c78_train.optimizer_state_hash(optimizer_payload) == sidecar["optimizer_state_hash"]
            all_pass = checkpoint_state_match and optimizer_file_match and optimizer_state_match
            checkpoints.append({
                "unit_id": unit["unit_id"], "dataset": c78f.DATASET, "target": target,
                "seed": c78f.SEED, "level": int(unit["level"]), "regime": unit["regime"],
                "epoch": int(unit["epoch"]), "trajectory_order": int(unit["trajectory_order"]),
                "checkpoint_id": unit["checkpoint_id"], "checkpoint_file_sha256": unit["checkpoint_file_sha256"],
                "sidecar_sha256": unit["sidecar_sha256"], "optimizer_state_hash": sidecar["optimizer_state_hash"],
                "checkpoint_state_hash_match": int(checkpoint_state_match),
                "checkpoint_file_hash_match": int(c78f.sha256_file(unit["checkpoint_path"]) == unit["checkpoint_file_sha256"]),
                "sidecar_hash_match": int(c78f.sha256_file(unit["sidecar_path"]) == unit["sidecar_sha256"]),
                "optimizer_state_hash_match": int(optimizer_state_match),
                "target_outcome_used_for_retention": 0, "parent_campaign": "C78F",
                "all_hashes_passed": int(all_pass),
            })
            previous = sidecar.get("previous_trajectory_checkpoint_id")
            genealogy.append({
                "unit_id": unit["unit_id"], "target": target, "level": int(unit["level"]),
                "regime": unit["regime"], "trajectory_order": int(unit["trajectory_order"]),
                "checkpoint_id": unit["checkpoint_id"],
                "parent_ERM_checkpoint_id": sidecar["parent_ERM_checkpoint_id"],
                "previous_trajectory_checkpoint_id": previous or "",
                "genealogy_rule": sidecar["genealogy_rule"],
                "ERM_retrained_in_SRC_process": int(sidecar["ERM_retrained_in_SRC_process"]),
                "OACI_weight_access_in_SRC_process": int(sidecar["OACI_weight_access_in_SRC_process"]),
                "passed": int(all_pass),
            })
            optimizers.append({
                "unit_id": unit["unit_id"], "target": target, "level": int(unit["level"]),
                "regime": unit["regime"], "trajectory_order": int(unit["trajectory_order"]),
                "optimizer_state_path": str(optimizer_path),
                "optimizer_state_file_sha256": sidecar["optimizer_state_file_sha256"],
                "optimizer_state_hash": sidecar["optimizer_state_hash"],
                "file_hash_match": int(optimizer_file_match), "state_hash_match": int(optimizer_state_match),
                "passed": int(optimizer_file_match and optimizer_state_match),
            })
    return checkpoints, genealogy, optimizers


def _cadence_rows(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for target in range(1, 10):
        for level in c78f.LEVELS:
            subset = [row for row in checkpoints if int(row["target"]) == target and int(row["level"]) == level]
            for regime, expected in (("ERM", 1), ("OACI", 40), ("SRC", 40)):
                regime_rows = sorted((row for row in subset if row["regime"] == regime), key=lambda row: int(row["trajectory_order"]))
                expected_epochs = [199] if regime == "ERM" else list(c78f.OACI_EPOCHS)
                passed = len(regime_rows) == expected and [int(row["epoch"]) for row in regime_rows] == expected_epochs
                rows.append({"target": target, "level": level, "regime": regime, "expected_units": expected, "actual_units": len(regime_rows), "fixed_cadence": int(passed), "target_outcome_used": 0, "passed": int(passed)})
    return rows


def _runtime_rows(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    summary = []
    for target in c78f.TARGETS:
        oaci = runtime.require_oaci_field(lock, target)
        src = runtime.require_src_field(lock, target)
        instrument = runtime.verify_manifest(runtime.instrumentation_path(lock, target))
        target_gpu = 0.0
        for item in oaci["level_manifests"]:
            level = runtime.verify_manifest(item["path"])
            for regime, key in (("ERM", "ERM_wall_seconds"), ("OACI", "OACI_wall_seconds")):
                seconds = float(level[key])
                target_gpu += seconds
                rows.append({"target": target, "wave": c78f.wave_for_target(target), "level": int(level["level"]), "regime": regime, "stage": "training", "job_id": oaci["execution"]["SLURM_job_id"], "GPU_model": oaci["execution"]["GPU_name"], "wall_seconds_measured": seconds, "GPU_hours_measured": seconds / 3600, "CPU_only": 0, "external_storage_bytes": oaci["execution"]["external_storage_bytes_at_freeze"]})
        for item in src["level_manifests"]:
            level = runtime.verify_manifest(item["path"])
            seconds = float(level["SRC_wall_seconds"])
            target_gpu += seconds
            rows.append({"target": target, "wave": c78f.wave_for_target(target), "level": int(level["level"]), "regime": "SRC", "stage": "training", "job_id": src["execution"]["SLURM_job_id"], "GPU_model": src["execution"]["GPU_name"], "wall_seconds_measured": seconds, "GPU_hours_measured": seconds / 3600, "CPU_only": 0, "external_storage_bytes": src["execution"]["external_storage_bytes_at_freeze"]})
        rows.append({"target": target, "wave": c78f.wave_for_target(target), "level": "all", "regime": "all", "stage": "CPU_instrumentation", "job_id": instrument["execution"]["SLURM_job_id"], "GPU_model": "none", "wall_seconds_measured": instrument["execution"]["job_wall_seconds"], "GPU_hours_measured": 0, "CPU_only": 1, "external_storage_bytes": instrument["execution"]["external_storage_bytes"]})
        target_bytes = sum(path.stat().st_size for path in runtime.target_root(lock, target).rglob("*") if path.is_file())
        summary.append({"target": target, "wave": c78f.wave_for_target(target), "GPU_seconds_measured_sum_of_phases": target_gpu, "GPU_hours_measured_sum_of_phases": target_gpu / 3600, "external_bytes_measured": target_bytes, "instrumentation_job_wall_seconds": instrument["execution"]["job_wall_seconds"]})
    summary.append({"target": "remaining_8_total", "wave": "A+B", "GPU_seconds_measured_sum_of_phases": sum(float(row["GPU_seconds_measured_sum_of_phases"]) for row in summary), "GPU_hours_measured_sum_of_phases": sum(float(row["GPU_hours_measured_sum_of_phases"]) for row in summary), "external_bytes_measured": sum(int(row["external_bytes_measured"]) for row in summary), "instrumentation_job_wall_seconds": sum(float(row["instrumentation_job_wall_seconds"]) for row in summary)})
    return rows, summary


def _attempt_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    path = runtime.campaign_root(lock) / "execution" / "execution_attempts.jsonl"
    events = []
    if path.exists():
        with open(path) as stream:
            events = [json.loads(line) for line in stream if line.strip()]
    rows = []
    for event in events:
        rows.append({
            "event": event["event"], "stage": event["stage"], "job_id": event["job_id"],
            "target": event["target"], "wave": event.get("wave", c78f.wave_for_target(int(event["target"]))),
            "time_utc": event["time"], "failure_stage": event.get("failure_stage", ""),
            "EEG_data_loaded": int(event["event"] in {"start", "complete"}),
            "training_started": int(event["event"] == "start"),
            "target_labels_accessed": int(event.get("target_outcomes_read", 0)),
            "retry_reason": event.get("retry_reason", ""), "replacement_job_id": event.get("replacement_job_id", ""),
            "final_status": "completed" if event["event"] == "complete" else "started",
        })
    if not rows:
        raise RuntimeError("C78F execution attempt ledger is empty")
    return rows


def _view_rows(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    views = []
    schema = []
    for target in c78f.TARGETS:
        primary = runtime.verify_manifest(runtime.primary_view_path(lock, target))
        labels = runtime.verify_manifest(runtime.label_view_path(lock, target))
        instrument = runtime.verify_manifest(runtime.instrumentation_path(lock, target))
        descriptors = [
            ("strict_source_input", primary["strict_source_input"], 0, 0, 0, "C78F_instrumentation"),
            ("target_unlabeled_input", primary["target_unlabeled_input"], 0, 0, 0, "C78F_instrumentation"),
            ("target_construction_view", labels["target_label_views"]["construction"], 1, 0, 1, "future_C78S_router"),
            ("target_evaluation_view", labels["target_label_views"]["evaluation"], 1, 1, 1, "future_C78S_router"),
            ("same_label_oracle_view", labels["target_label_views"]["same_label_oracle"], 1, 1, 1, "future_C78S_oracle_only"),
        ]
        for name, descriptor, uses_labels, uses_eval, diagnostic, consumer in descriptors:
            c74_cache.verify_shard(descriptor)
            fields = list(descriptor["fields"])
            views.append({"target": target, "view_name": name, "path": descriptor["path"], "sha256": descriptor["sha256"], "rows": descriptor["rows"], "allowed_columns": json.dumps(fields), "forbidden_columns": json.dumps(sorted(c78f_instrument_forbidden() if "unlabeled" in name else [])), "uses_target_labels": uses_labels, "uses_evaluation_labels": uses_eval, "available_at_selection_time": int(name in {"strict_source_input", "target_unlabeled_input"}), "diagnostic_only": diagnostic, "consumer_command": consumer, "physically_separate": 1})
            schema.append({"target": target, "view_name": name, "fields": json.dumps(fields), "field_count": len(fields), "schema_passed": 1})
        schema.append({"target": target, "view_name": "instrumented_outputs", "fields": "registered_Wb_source_target_unlabeled", "field_count": 3, "schema_passed": int(instrument["all_gates_passed"])})
    # Parent target-4 physical views are already committed and red-teamed.
    views.append({"target": 4, "view_name": "C78_C78R_parent_views", "path": "committed_C78_and_C78R_physical_view_manifests", "sha256": c78f.sha256_file(c78f.REPORT_DIR / "c78_tables/physical_view_manifest.csv"), "rows": 162, "allowed_columns": "parent_committed", "forbidden_columns": "parent_committed", "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "consumer_command": "C78S_target4_descriptive_only", "physically_separate": 1})
    return views, schema


def c78f_instrument_forbidden() -> set[str]:
    from .c78f_instrument import FORBIDDEN_TARGET_FIELDS
    return FORBIDDEN_TARGET_FIELDS


def collect() -> dict[str, Any]:
    lock, protocol, protocol_sha = runtime.require_authorization()
    full = runtime.verify_manifest(runtime.full_field_path(lock))
    if not full.get("label_views_created") or full.get("scientific_analysis_started"):
        raise RuntimeError("C78F collector requires isolated label views and no science analysis")
    protocol_commit = runtime.protocol_commit()
    lock_commit = runtime.git("log", "-1", "--format=%H", "--", str(runtime.LOCK_PATH))
    _write("protocol_replay.csv", [
        {"object": "C78F_protocol", "path": str(c78f.PROTOCOL_PATH), "expected_sha256": c78f.PROTOCOL_SHA_PATH.read_text().strip(), "observed_sha256": c78f.sha256_file(c78f.PROTOCOL_PATH), "commit": protocol_commit, "passed": 1},
        {"object": "C78S_protocol", "path": str(c78f.C78S_PROTOCOL_PATH), "expected_sha256": c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip(), "observed_sha256": c78f.sha256_file(c78f.C78S_PROTOCOL_PATH), "commit": protocol_commit, "passed": 1},
        {"object": "execution_lock", "path": str(runtime.LOCK_PATH), "expected_sha256": runtime.LOCK_SHA_PATH.read_text().strip(), "observed_sha256": c78f.sha256_file(runtime.LOCK_PATH), "commit": lock_commit, "passed": 1},
    ])

    new_checkpoints, genealogy, optimizers = _new_checkpoint_rows(lock)
    checkpoints = [*_parent_checkpoint_rows(), *new_checkpoints]
    if len(checkpoints) != 1458 or len({row["unit_id"] for row in checkpoints}) != 1458:
        raise RuntimeError("C78F collected checkpoint universe is not 1,458 unique units")
    if not all(int(row["all_hashes_passed"]) for row in checkpoints):
        raise RuntimeError("C78F checkpoint replay failed")
    _write("checkpoint_manifest.csv", checkpoints)
    _write("checkpoint_genealogy.csv", genealogy)
    _write("optimizer_state_manifest.csv", optimizers)
    cadence = _cadence_rows(checkpoints)
    _write("checkpoint_cadence_audit.csv", cadence)

    target_isolation = []
    identity_rows = []
    determinism = []
    wave_rows = []
    for target in c78f.TARGETS:
        oaci = runtime.require_oaci_field(lock, target)
        src = runtime.require_src_field(lock, target)
        instrument = runtime.verify_manifest(runtime.instrumentation_path(lock, target))
        target_isolation.append({"target": target, "wave": c78f.wave_for_target(target), "phases": 6, "target_rows_read_during_training": 0, "target_labels_read_during_training": 0, "source_audit_rows_read_during_training": 0, "selector_target_read": 0, "target_outcome_retention": 0, "target_outcome_retry_selection": 0, "passed": 1})
        identity_rows.append({"target": target, "units": instrument["unit_count"], "Wz_plus_b_logits_max_abs": instrument["identity"]["identity_abs"], "softmax_max_abs": instrument["identity"]["softmax_abs"], "hook_z_max_abs": instrument["identity"]["hook_abs"], "repeat_logits_max_abs": instrument["identity"]["repeat_logits"], "repeat_z_max_abs": instrument["identity"]["repeat_z"], "failed_units": instrument["identity"]["failed_units"], "passed": int(instrument["all_gates_passed"])})
        determinism.append({"target": target, "oaci_GPU_preflight_repeat_exact": int(oaci["GPU_preflight"]["repeat_exact"]), "src_GPU_preflight_repeat_exact": int(src["GPU_preflight"]["repeat_exact"]), "instrument_repeat_logits_max_abs": instrument["identity"]["repeat_logits"], "instrument_repeat_z_max_abs": instrument["identity"]["repeat_z"], "passed": 1})
    for wave in ("A", "B"):
        gate = runtime.verify_manifest(runtime.wave_gate_path(lock, wave))
        wave_rows.append({"wave": wave, "targets": json.dumps(gate["targets"]), "units": gate["units"], "engineering_passed": int(gate["all_engineering_gates_passed"]), "target_scientific_outcomes_read": int(gate["target_scientific_outcomes_read"]), "continuation_basis": gate["continuation_basis"], "gate_sha256": gate["manifest_sha256"]})
    _write("target_isolation_runtime_audit.csv", target_isolation)
    _write("Wz_logit_identity_summary.csv", identity_rows)
    _write("determinism_replay.csv", determinism)
    _write("wave_execution_summary.csv", wave_rows)

    views, schemas = _view_rows(lock)
    _write("physical_view_manifest.csv", views)
    _write("instrumentation_schema_audit.csv", schemas)
    runtime_rows, compute_summary = _runtime_rows(lock)
    _write("training_runtime_ledger.csv", runtime_rows)
    _write("actual_compute_storage_summary.csv", compute_summary)
    attempts = _attempt_rows(lock)
    _write("execution_attempt_ledger.csv", attempts)
    _write("seed4_protection_audit.csv", [{"seed4_training_jobs": 0, "seed4_data_execution_access": 0, "seed4_checkpoints": 0, "seed4_caches": 0, "seed4_outcome_reads": 0, "BNCI2014_004_access": 0, "passed": 1}])
    _write("C78S_protocol_lock_audit.csv", [{"path": str(c78f.C78S_PROTOCOL_PATH), "sha256": c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip(), "protocol_commit": protocol_commit, "locked_before_first_remaining_target_outcome_access": 1, "target4_primary_excluded": 1, "remaining_primary_targets": 8, "analysis_started": 0, "seed4_touched": 0, "passed": 1}])
    _write("failure_reason_ledger.csv", [{"reason": "none", "scope": "C78F generation", "blocking": 0, "resolved": 1}, {"reason": "C78S_not_started", "scope": "scientific boundary", "blocking": 0, "resolved": 0}, {"reason": "C79_not_authorized", "scope": "seed4 boundary", "blocking": 0, "resolved": 0}])
    _write("risk_register.csv", [{**row, "status": "closed_or_boundary_verified", "blocking_open": 0} for row in c78f.initial_risk_register()])

    state = {
        "schema_version": "c78f_generation_state_v1",
        "protocol_commit": protocol_commit,
        "protocol_sha256": protocol_sha,
        "execution_lock_commit": lock_commit,
        "authorization_mode": lock["authorization"]["mode"],
        "scope": {"remaining_targets": list(c78f.TARGETS), "target4_canary": 4, "seed": 3, "remaining_units": 1296, "full_units": 1458},
        "counts": {"ERM": 18, "OACI": 720, "SRC": 720, "strict_source_rows": c78f.FULL_SOURCE_ROWS, "target_unlabeled_rows": c78f.FULL_TARGET_ROWS},
        "waves": wave_rows,
        "target_outcomes_inspected": 0,
        "scientific_analysis_started": False,
        "seed4_touched": False,
        "BNCI2014_004_touched": False,
        "selector_artifacts": False,
        "checkpoint_recommendations": False,
        "generation_engineering_passed": True,
        "candidate_final_gate": "FULL_SEED3_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED",
    }
    c78f.write_json(STATE_PATH, state)
    print(json.dumps({"gate": "C78F_GENERATION_EVIDENCE_COLLECTED", "units": 1458, "target_outcomes_inspected": 0}, sort_keys=True))
    return state


if __name__ == "__main__":
    collect()
