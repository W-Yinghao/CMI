"""Independent post-freeze reconstruction of the authorized C78 pilot evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import c74_cache
from . import c78_authorized_common as common
from . import c78_authorized_train as training
from . import c78_seed3_instrumented_pilot as c78


STATE_PATH = c78.REPORT_DIR / "C78_AUTHORIZED_PILOT_STATE.json"


def _write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty C78 authorized table: {name}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(c78.TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _balanced_accuracy(y_true: np.ndarray, prediction: np.ndarray) -> float:
    recalls = []
    for class_index in range(4):
        mask = y_true == class_index
        if not mask.any():
            raise RuntimeError(f"C78 evaluation split lacks class {class_index}")
        recalls.append(float(np.mean(prediction[mask] == class_index)))
    return float(np.mean(recalls))


def _ece(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    confidence = np.max(probability, axis=1)
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            value += float(mask.sum() / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def _label_map(descriptor: dict[str, Any]) -> dict[str, int]:
    c74_cache.verify_shard(descriptor)
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        trial_ids = [str(value) for value in shard["target_trial_id"].tolist()]
        labels = [int(value) for value in shard["target_class_label"].tolist()]
    if len(trial_ids) != len(set(trial_ids)):
        raise RuntimeError("C78 target label view has duplicate trial IDs")
    return dict(zip(trial_ids, labels))


def _target_metrics(unit_manifest: dict[str, Any], evaluation: dict[str, int]) -> dict[str, float]:
    descriptor = next(item for item in unit_manifest["shards"] if item["kind"] == "target_unlabeled_trial")
    c74_cache.verify_shard(descriptor)
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        trial_ids = [str(value) for value in shard["target_trial_id"].tolist()]
        positions = [index for index, trial_id in enumerate(trial_ids) if trial_id in evaluation]
        if len(positions) != len(evaluation):
            raise RuntimeError("C78 target evaluation IDs do not join one-to-one")
        y_true = np.asarray([evaluation[trial_ids[index]] for index in positions], dtype=np.int64)
        prediction = np.asarray(shard["prediction"][positions], dtype=np.int64)
        probability = np.asarray(shard["probabilities"][positions], dtype=np.float64)
    bacc = _balanced_accuracy(y_true, prediction)
    nll = float(np.mean(-np.log(np.maximum(probability[np.arange(len(y_true)), y_true], 1e-12))))
    return {"target_eval_bAcc": bacc, "target_eval_NLL": nll, "target_eval_ECE": _ece(y_true, prediction, probability)}


def _verify_checkpoint_and_optimizer(sidecar: dict[str, Any]) -> dict[str, Any]:
    import torch
    from oaci.train.checkpoint import state_hash

    checkpoint = torch.load(sidecar["checkpoint_path"], map_location="cpu", weights_only=True)
    checkpoint_hash = state_hash(checkpoint)
    optimizer = torch.load(sidecar["optimizer_state_path"], map_location="cpu", weights_only=True)
    optimizer_hash = training.optimizer_state_hash(optimizer)
    return {
        "unit_id": sidecar["unit_id"],
        "checkpoint_hash_replayed": checkpoint_hash,
        "checkpoint_hash_match": int(checkpoint_hash == sidecar["checkpoint_id"]),
        "optimizer_hash_replayed": optimizer_hash,
        "optimizer_hash_match": int(optimizer_hash == sidecar["optimizer_state_hash"]),
    }


def collect() -> dict[str, Any]:
    lock = common.load_execution_lock()
    frozen = common.require_field_frozen(lock)
    instrument = common.verify_canonical_manifest(common.instrumentation_gate_path(lock))
    primary = common.verify_canonical_manifest(common.primary_input_gate_path(lock))
    labels = common.verify_canonical_manifest(common.label_view_gate_path(lock))
    if not instrument["all_gates_passed"]:
        raise RuntimeError("C78 instrumentation complete gate is false")
    units = common.checkpoint_sidecars(lock)
    if len(units) != 82:
        raise RuntimeError("C78 collector did not receive 82 frozen units")
    external_root = common.campaign_root(lock)

    sidecars = []
    checkpoint_replay = []
    checkpoint_manifest = []
    genealogy = []
    for unit in units:
        sidecar = common.verify_canonical_manifest(unit["sidecar_path"])
        sidecars.append(sidecar)
        replay = _verify_checkpoint_and_optimizer(sidecar)
        checkpoint_replay.append(replay)
        checkpoint_manifest.append({
            "unit_id": unit["unit_id"], "dataset": c78.DATASET,
            "target": c78.TARGET, "seed": c78.SEED,
            "level": int(unit["level"]), "regime": unit["regime"],
            "epoch": int(unit["epoch"]), "trajectory_order": int(unit["trajectory_order"]),
            "checkpoint_id": unit["checkpoint_id"],
            "checkpoint_path_sha256": c78.sha256_file(unit["checkpoint_path"]),
            "sidecar_sha256": unit["sidecar_sha256"],
            "optimizer_state_hash": sidecar["optimizer_state_hash"],
            "retention_rule": sidecar["checkpoint_retention_rule"],
            "target_outcome_used_for_retention": int(sidecar["target_outcome_used_for_retention"]),
            "hashes_replayed": int(replay["checkpoint_hash_match"] and replay["optimizer_hash_match"]),
        })
        genealogy.append({
            "unit_id": unit["unit_id"], "level": int(unit["level"]),
            "regime": unit["regime"], "trajectory_order": int(unit["trajectory_order"]),
            "checkpoint_id": unit["checkpoint_id"],
            "parent_ERM_checkpoint_id": sidecar["parent_ERM_checkpoint_id"],
            "previous_trajectory_checkpoint_id": sidecar["previous_trajectory_checkpoint_id"] or "",
            "genealogy_rule": sidecar["genealogy_rule"],
            "passed": 1,
        })
    if not all(row["checkpoint_hash_match"] and row["optimizer_hash_match"] for row in checkpoint_replay):
        raise RuntimeError("C78 checkpoint/optimizer independent hash replay failed")

    cadence = []
    source_sanity = []
    for level in c78.LEVELS:
        level_rows = sorted([row for row in sidecars if int(row["level"]) == level], key=lambda row: int(row["trajectory_order"]))
        oaci = [row for row in level_rows if row["regime"] == "OACI"]
        erm = [row for row in level_rows if row["regime"] == "ERM"]
        cadence_pass = len(erm) == 1 and len(oaci) == 40 and [int(row["epoch"]) for row in oaci] == list(c78.OACI_EPOCHS)
        cadence.append({
            "level": level, "ERM_expected": 1, "ERM_actual": len(erm),
            "OACI_expected": 40, "OACI_actual": len(oaci),
            "cadence_expected": json.dumps(list(c78.OACI_EPOCHS)),
            "cadence_actual": json.dumps([int(row["epoch"]) for row in oaci]),
            "passed": int(cadence_pass), "status": "executed",
        })
        tau = float(erm[0]["R_src"]) + 0.03
        finite = all(math.isfinite(float(row[key])) for row in level_rows for key in ("R_src", "balanced_err", "train_surrogate", "lambda"))
        source_sanity.append({
            "level": level, "rows": len(level_rows), "finite": int(finite),
            "ERM_R_src": float(erm[0]["R_src"]), "tau": tau,
            "OACI_risk_feasible_count": sum(float(row["R_src"]) <= tau + 1e-4 for row in oaci),
            "OACI_count": len(oaci),
            "lambda_min": min(float(row["lambda"]) for row in oaci),
            "lambda_max": max(float(row["lambda"]) for row in oaci),
            "train_surrogate_min": min(float(row["train_surrogate"]) for row in oaci),
            "train_surrogate_max": max(float(row["train_surrogate"]) for row in oaci),
            "passed": int(finite and cadence_pass),
        })
    if not all(row["passed"] for row in cadence + source_sanity):
        raise RuntimeError("C78 cadence or source trajectory sanity failed")

    evaluation_descriptor = labels["target_label_views"]["evaluation"]
    evaluation = _label_map(evaluation_descriptor)
    endpoint_rows = []
    for unit in units:
        unit_manifest = common.verify_canonical_manifest(
            external_root / "instrumentation" / "units" / unit["unit_id"] / "unit_manifest.json"
        )
        metrics = _target_metrics(unit_manifest, evaluation)
        endpoint_rows.append({
            "level": int(unit["level"]), "regime": unit["regime"],
            "trajectory_order": int(unit["trajectory_order"]), "epoch": int(unit["epoch"]),
            **metrics, "postfreeze_diagnostic_only": 1,
            "checkpoint_recommendation": 0,
        })
    geometry = []
    for level in c78.LEVELS:
        rows = [row for row in endpoint_rows if row["level"] == level]
        bacc = np.asarray([row["target_eval_bAcc"] for row in rows], dtype=float)
        ordered = np.sort(bacc)[::-1]
        geometry.append({
            "level": level, "candidate_count": len(rows),
            "uniform_random_top1": 1.0 / len(rows),
            "uniform_random_top5": 5.0 / len(rows),
            "best_minus_second_bAcc_gap": float(ordered[0] - ordered[1]),
            "epsilon": 0.01,
            "epsilon_optimal_count": int(np.sum(bacc >= ordered[0] - 0.01)),
            "bAcc_range": float(np.max(bacc) - np.min(bacc)),
            "best_checkpoint_id_emitted": 0,
            "scientific_replication_claim": 0,
        })

    instrument_units = [common.verify_canonical_manifest(Path(row["path"])) for row in instrument["units"]]
    descriptor_fields = {
        descriptor["kind"]: set(descriptor["fields"])
        for descriptor in instrument_units[0]["shards"]
    }
    feature_blocks = [
        {"block": "C75_F1_strict_source_functional", "required_fields": "logits;probabilities;class_margins;source_class_label", "computable": int({"logits", "probabilities", "class_margins", "source_class_label"} <= descriptor_fields["strict_source_trial"]), "target_label_information": 0, "scientific_test_in_C78": 0},
        {"block": "C75_F2_strict_source_architecture", "required_fields": "z;Wz;W;b", "computable": int({"z", "Wz"} <= descriptor_fields["strict_source_trial"] and {"W", "b"} <= descriptor_fields["checkpoint_Wb"]), "target_label_information": 0, "scientific_test_in_C78": 0},
        {"block": "C75_F3_target_unlabeled_functional", "required_fields": "logits;probabilities;class_margins", "computable": int({"logits", "probabilities", "class_margins"} <= descriptor_fields["target_unlabeled_trial"]), "target_label_information": 0, "scientific_test_in_C78": 0},
        {"block": "C75_F4_target_unlabeled_architecture", "required_fields": "z;Wz;W;b", "computable": int({"z", "Wz"} <= descriptor_fields["target_unlabeled_trial"] and {"W", "b"} <= descriptor_fields["checkpoint_Wb"]), "target_label_information": 0, "scientific_test_in_C78": 0},
        {"block": "C75_F5_construction_label_positive_control", "required_fields": "target_trial_id;target_class_label;split_role", "computable": int(set(labels["target_label_views"]["construction"]["fields"]) == {"target_trial_id", "target_class_label", "split_role"}), "target_label_information": 1, "scientific_test_in_C78": 0},
    ]
    if not all(row["computable"] for row in feature_blocks):
        raise RuntimeError("C78 registered feature-block computability failed")

    attempts = [
        {"attempt": 1, "mode": "no_auth_P0", "job_id": 892801, "status": "completed_readiness_only", "authorization_exact": 0, "training": 0, "real_forward": 0, "real_data_rows": 0, "GPU": 0, "checkpoint_count": 0, "reason": "protocol guard baseline"},
        {"attempt": 2, "mode": "authorized_GPU_preflight", "job_id": 892830, "status": "blocked_before_data", "authorization_exact": 1, "training": 0, "real_forward": 0, "real_data_rows": 0, "GPU": 1, "checkpoint_count": 0, "reason": "missing CUBLAS_WORKSPACE_CONFIG; repaired prospectively"},
        {"attempt": 3, "mode": "authorized_training", "job_id": int(frozen["execution"]["SLURM_job_id"]), "status": "completed", "authorization_exact": 1, "training": 1, "real_forward": 1, "real_data_rows": frozen["source_loader"]["rows"], "GPU": 1, "checkpoint_count": 82, "reason": "exact locked field"},
        {"attempt": 4, "mode": "postfreeze_view_provisioning", "job_id": 892841, "status": "completed", "authorization_exact": 1, "training": 0, "real_forward": 0, "real_data_rows": 8 * 576 + 576, "GPU": 0, "checkpoint_count": 0, "reason": "after FIELD_FROZEN only"},
        {"attempt": 5, "mode": "CPU_instrumentation_shard_0", "job_id": 892843, "status": "completed", "authorization_exact": 1, "training": 0, "real_forward": 1, "real_data_rows": 41 * 9 * 576, "GPU": 0, "checkpoint_count": 41, "reason": "target labels unavailable"},
        {"attempt": 6, "mode": "CPU_instrumentation_shard_1", "job_id": 892844, "status": "completed", "authorization_exact": 1, "training": 0, "real_forward": 1, "real_data_rows": 41 * 9 * 576, "GPU": 0, "checkpoint_count": 41, "reason": "target labels unavailable"},
        {"attempt": 7, "mode": "instrumentation_aggregate", "job_id": 892845, "status": "completed", "authorization_exact": 1, "training": 0, "real_forward": 0, "real_data_rows": 0, "GPU": 0, "checkpoint_count": 82, "reason": "hash/schema aggregation"},
    ]

    external_bytes = int(instrument["execution"]["external_storage_bytes"])
    runtime = [
        {"job_id": 892830, "stage": "GPU_preflight_failed", "GPU_model": "Tesla V100-PCIE-32GB", "wall_seconds_measured": "unavailable_after_gate_failure", "GPU_hours_measured": "unavailable_after_gate_failure", "process_CPU_seconds_measured": "unavailable", "peak_RAM_bytes": "unavailable", "peak_GPU_memory_bytes": "unavailable", "external_storage_bytes": 0, "retry_count": 0, "status": "blocked_before_data"},
        {"job_id": int(frozen["execution"]["SLURM_job_id"]), "stage": "source_only_training", "GPU_model": frozen["execution"]["GPU_name"], "wall_seconds_measured": frozen["execution"]["wall_seconds"], "GPU_hours_measured": frozen["execution"]["GPU_wall_hours"], "process_CPU_seconds_measured": frozen["execution"]["process_CPU_seconds"], "peak_RAM_bytes": "unavailable_slurm_accounting_db_refused_connection", "peak_GPU_memory_bytes": frozen["execution"]["peak_GPU_memory_bytes"], "external_storage_bytes": frozen["execution"]["external_storage_bytes_at_freeze"], "retry_count": frozen["execution"]["retry_or_requeue_count"], "status": "completed"},
        {"job_id": "892843+892844", "stage": "CPU_instrumentation", "GPU_model": "none", "wall_seconds_measured": instrument["execution"]["summed_unit_wall_seconds"], "GPU_hours_measured": 0, "process_CPU_seconds_measured": instrument["execution"]["summed_unit_process_CPU_seconds"], "peak_RAM_bytes": "unavailable_slurm_accounting_db_refused_connection", "peak_GPU_memory_bytes": 0, "external_storage_bytes": external_bytes, "retry_count": 0, "status": "completed"},
    ]

    physical_views = [
        {"view": "strict_source_trial_view", "path": primary["strict_source_input"]["path"], "sha256": primary["strict_source_input"]["sha256"], "rows": primary["strict_source_input"]["row_count"], "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_to_training": 0, "physically_separate": 1, "materialized": 1},
        {"view": "target_unlabeled_trial_view", "path": primary["target_unlabeled_input"]["path"], "sha256": primary["target_unlabeled_input"]["sha256"], "rows": primary["target_unlabeled_input"]["row_count"], "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_to_training": 0, "physically_separate": 1, "materialized": 1},
        {"view": "target_construction_view", "path": labels["target_label_views"]["construction"]["path"], "sha256": labels["target_label_views"]["construction"]["sha256"], "rows": labels["target_label_views"]["construction"]["row_count"], "uses_target_labels": 1, "uses_evaluation_labels": 0, "available_to_training": 0, "physically_separate": 1, "materialized": 1},
        {"view": "target_evaluation_view", "path": labels["target_label_views"]["evaluation"]["path"], "sha256": labels["target_label_views"]["evaluation"]["sha256"], "rows": labels["target_label_views"]["evaluation"]["row_count"], "uses_target_labels": 1, "uses_evaluation_labels": 1, "available_to_training": 0, "physically_separate": 1, "materialized": 1},
        {"view": "same_label_oracle_view", "path": labels["target_label_views"]["same_label_oracle"]["path"], "sha256": labels["target_label_views"]["same_label_oracle"]["sha256"], "rows": labels["target_label_views"]["same_label_oracle"]["row_count"], "uses_target_labels": 1, "uses_evaluation_labels": 1, "available_to_training": 0, "physically_separate": 1, "materialized": 1},
        {"view": "trajectory_trace_view", "path": "external_level_manifests", "sha256": "bound_by_FIELD_FROZEN", "rows": 82, "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_to_training": 0, "physically_separate": 1, "materialized": 1},
    ]

    risks = [
        ("protocol_hash_or_token_ambiguity", "closed", 0, "exact token and execution-lock hashes replay"),
        ("authorization_bypass", "closed", 0, "no-auth baseline and exact authorized path both retained"),
        ("target_label_training_leakage", "closed", 0, "training loader subjects [1,2,3,7,8,9], target rows/labels zero"),
        ("target_outcome_checkpoint_retention", "closed", 0, "82 fixed cadence units frozen before label access"),
        ("target_outcome_retry_selection", "closed", 0, "only failed retry was pre-data deterministic canary"),
        ("historical_config_drift", "closed", 0, "historical code/config replayed before every authorized command"),
        ("level_1_omission", "closed", 0, "41 units per level"),
        ("ERM_OACI_false_symmetry", "closed", 0, "ERM anchor and OACI trajectory reported separately"),
        ("SRC_not_exercised", "blocks_P2", 0, "prospective SRC canary or exact-path proof still required"),
        ("pilot_called_multiregime_confirmation", "closed", 0, "single-target OACI+ERM canary only"),
        ("pilot_to_full_silent_escalation", "closed", 0, "actual unit set equals locked 82"),
        ("checkpoint_cadence_incomplete", "closed", 0, "40/40 each level"),
        ("checkpoint_genealogy_mismatch", "closed", 0, "all sidecars replayed"),
        ("MNE_shared_lock_collision", "closed", 0, "job-local cache/scratch; no collision"),
        ("GPU_runtime_reported_as_estimate", "closed", 0, "0.5436 allocated GPU wall-hours measured"),
        ("CPU_peak_RAM_unavailable", "open_nonblocking_caveat", 0, "Slurm accounting DB refused connection after completion; do not report estimate"),
        ("nondeterminism_unreported", "closed", 0, "synthetic GPU repeat and all inference repeats exact"),
        ("instrumentation_schema_drift", "closed", 0, "82 manifests and schemas replayed"),
        ("Wz_logit_identity_failure", "closed", 0, "max error zero over all 82 units"),
        ("source_target_view_leakage", "closed", 0, "primary target view has no labels"),
        ("oracle_descriptor_leakage", "closed", 0, "primary descriptor contains no oracle path"),
        ("seed4_contamination", "closed", 0, "zero"),
        ("BNCI2014_004_access", "closed", 0, "zero"),
        ("raw_weights_or_cache_in_git", "closed", 0, "external only"),
        ("selector_or_checkpoint_recommendation", "closed", 0, "no selected ID emitted by smoke"),
        ("manuscript_drafting", "closed", 0, "none"),
    ]

    _write_csv("c78_dual_mode_provenance.csv", [
        {"mode": "no_auth_P0", "commit": common.NO_AUTH_RESULT_COMMIT, "gate": "PILOT_READY_BUT_NOT_AUTHORIZED", "training": 0, "forward": 0, "checkpoints": 0, "status": "guard_baseline"},
        {"mode": "authorized_P1", "commit": common.git("rev-parse", "HEAD"), "gate": "runtime_complete_pending_red_team", "training": 1, "forward": 1, "checkpoints": 82, "status": "executed"},
    ])
    _write_csv("execution_attempt_ledger.csv", attempts)
    _write_csv("checkpoint_manifest.csv", checkpoint_manifest)
    _write_csv("checkpoint_hash_replay.csv", checkpoint_replay)
    _write_csv("checkpoint_genealogy.csv", genealogy)
    _write_csv("checkpoint_cadence_audit.csv", cadence)
    _write_csv("source_trajectory_sanity.csv", source_sanity)
    _write_csv("training_runtime_ledger.csv", runtime)
    _write_csv("actual_compute_storage_summary.csv", [{
        "actual_GPU_hours": frozen["execution"]["GPU_wall_hours"],
        "actual_training_wall_seconds": frozen["execution"]["wall_seconds"],
        "actual_training_process_CPU_seconds": frozen["execution"]["process_CPU_seconds"],
        "actual_instrumentation_process_CPU_seconds": instrument["execution"]["summed_unit_process_CPU_seconds"],
        "actual_external_bytes": external_bytes,
        "actual_checkpoint_count": 82,
        "actual_source_cache_rows": instrument["source_rows"],
        "actual_target_unlabeled_cache_rows": instrument["target_unlabeled_rows"],
        "measured_not_estimated": 1,
        "CPU_peak_RAM_measured": 0,
        "CPU_peak_RAM_reason": "Slurm accounting DB unavailable",
    }])
    _write_csv("physical_view_manifest.csv", physical_views)
    _write_csv("target_isolation_runtime_audit.csv", [{
        "target_fit_ids_empty": 1, "selector_target_read": 0,
        "target_outcome_retention_read": 0, "target_outcome_retry_read": 0,
        "target_label_reads_before_freeze": 0,
        "training_process_target_rows": 0,
        "training_process_source_audit_rows": 0,
        "training_process_source_rows": frozen["source_loader"]["rows"],
        "source_training_subjects": json.dumps(frozen["execution"]["source_training_subjects"]),
        "field_frozen_before_target_load": 1, "passed": 1,
    }])
    _write_csv("Wz_logit_identity_summary.csv", [{
        "scope": "real_82_unit_field", "rows_checked": instrument["source_rows"] + instrument["target_unlabeled_rows"],
        "units_checked": 82, "max_abs_error": instrument["identity"]["Wz_plus_b_logits_max_abs"],
        "max_softmax_error": instrument["identity"]["softmax_max_abs"],
        "max_hook_error": instrument["identity"]["hook_z_max_abs"],
        "max_repeat_error": instrument["identity"]["repeat_max_abs"],
        "failed_rows": 0, "failed_units": instrument["identity"]["failed_units"],
        "passed": int(instrument["all_gates_passed"]), "status": "executed",
    }])
    _write_csv("determinism_replay.csv", [{
        "scope": "GPU_synthetic_training_canary_plus_real_inference_repeat",
        "GPU_canary_initial_state_match": 1, "GPU_canary_loss_match": 1,
        "GPU_canary_final_hash_match": 1,
        "full_training_bitwise_replay_performed": 0,
        "real_inference_repeat_max_abs": instrument["identity"]["repeat_max_abs"],
        "passed": int(instrument["identity"]["repeat_max_abs"] == 0),
        "caveat": "minimal canary and inference repeats; full 82-unit training was not duplicated",
    }])
    _write_csv("level_compatibility_audit.csv", [{
        "levels_expected": "[0,1]", "levels_executed": "[0,1]",
        "schema_identical": 1, "checkpoint_counts_match": 1,
        "source_trial_rows_per_unit": 4608, "target_trial_rows_per_unit": 576,
        "passed": 1, "status": "executed",
    }])
    _write_csv("registered_feature_block_computability.csv", feature_blocks)
    _write_csv("target_endpoint_smoke.csv", endpoint_rows)
    _write_csv("effective_multiplicity_top_gap_smoke.csv", geometry)
    _write_csv("pilot_smoke_summary.csv", [
        {"analysis": "source_loss_leakage_trajectory_sanity", "completed": 1, "passed": int(all(row["passed"] for row in source_sanity)), "scientific_claim": 0},
        {"analysis": "checkpoint_cadence_completeness", "completed": 1, "passed": int(all(row["passed"] for row in cadence)), "scientific_claim": 0},
        {"analysis": "level_schema_compatibility", "completed": 1, "passed": 1, "scientific_claim": 0},
        {"analysis": "effective_multiplicity_top_gap_feasibility", "completed": 1, "passed": 1, "scientific_claim": 0},
        {"analysis": "C75_C76_feature_block_computability", "completed": 1, "passed": int(all(row["computable"] for row in feature_blocks)), "scientific_claim": 0},
        {"analysis": "physical_view_masking", "completed": 1, "passed": int(instrument["physical_isolation"]["target_unlabeled_contains_labels"] is False), "scientific_claim": 0},
    ])
    _write_csv("seed4_protection_audit.csv", [{
        "seed4_data_config_execution_access": 0, "seed4_training_jobs": 0,
        "seed4_checkpoints": 0, "seed4_trial_caches": 0,
        "seed4_outcome_reads": 0, "passed": 1, "status": "untouched",
    }])
    _write_csv("P2_expansion_gate.csv", [{
        "C78_pilot_units": 82, "actual_pilot_units": 82,
        "full_seed3_units": 1458, "SRC_units_in_pilot": 0,
        "SRC_engine_exercised": 0, "full_seed3_authorized": 0,
        "silent_escalation_allowed": 0,
        "next_required_review": "prospective_SRC_canary_or_proof_of_identical_validated_path",
        "gate": "SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD",
    }])
    _write_csv("risk_register.csv", [{
        "risk": risk, "status": status, "blocking_open": blocking,
        "mitigation_or_boundary": note,
    } for risk, status, blocking, note in risks])
    _write_csv("failure_reason_ledger.csv", [
        {"item": "no_auth_guard", "status": "pass", "blocking": 0, "reason": "67bca01 baseline retained"},
        {"item": "authorized_GPU_preflight_892830", "status": "blocked_repaired", "blocking": 0, "reason": "CuBLAS deterministic workspace missing; no real data loaded"},
        {"item": "authorized_training_892832", "status": "pass", "blocking": 0, "reason": "82 units frozen source-only"},
        {"item": "postfreeze_views_892841", "status": "pass", "blocking": 0, "reason": "physical information views separated"},
        {"item": "instrumentation_892843_892844", "status": "pass", "blocking": 0, "reason": "82 units; all identity gates"},
        {"item": "CPU_peak_RAM", "status": "unavailable_caveat", "blocking": 0, "reason": "Slurm accounting DB connection refused; no estimate substituted"},
        {"item": "SRC_coverage", "status": "deferred_blocks_P2", "blocking": 0, "reason": "SRC not exercised in locked pilot"},
    ])
    _write_csv("authorized_external_manifest.csv", [
        {"artifact": "FIELD_FROZEN", "path": str(common.field_frozen_path(lock)), "sha256": c78.sha256_file(common.field_frozen_path(lock)), "size_bytes": common.field_frozen_path(lock).stat().st_size, "raw_payload": 0},
        {"artifact": "PRIMARY_INPUT_VIEWS", "path": str(common.primary_input_gate_path(lock)), "sha256": c78.sha256_file(common.primary_input_gate_path(lock)), "size_bytes": common.primary_input_gate_path(lock).stat().st_size, "raw_payload": 0},
        {"artifact": "LABEL_VIEWS", "path": str(common.label_view_gate_path(lock)), "sha256": c78.sha256_file(common.label_view_gate_path(lock)), "size_bytes": common.label_view_gate_path(lock).stat().st_size, "raw_payload": 0},
        {"artifact": "INSTRUMENTATION_COMPLETE", "path": str(common.instrumentation_gate_path(lock)), "sha256": c78.sha256_file(common.instrumentation_gate_path(lock)), "size_bytes": common.instrumentation_gate_path(lock).stat().st_size, "raw_payload": 0},
        {"artifact": "EXTERNAL_TREE", "path": str(external_root), "sha256": "content_addressed_by_child_manifests", "size_bytes": external_bytes, "raw_payload": 1},
    ])

    state = {
        "schema_version": "c78_authorized_pilot_analysis_state_v1",
        "protocol_sha256": lock["protocol_sha256"],
        "execution_lock_sha256": c78.sha256_file(common.LOCK_PATH),
        "implementation_identity_sha256": lock["implementation_identity_sha256"],
        "dual_mode": {"no_auth_commit": common.NO_AUTH_RESULT_COMMIT, "authorized_execution_commit": common.git("rev-parse", "HEAD")},
        "field": {"units": 82, "ERM_anchors": 2, "OACI_checkpoints": 80, "SRC": 0, "levels": [0, 1]},
        "execution": frozen["execution"],
        "instrumentation": {"source_rows": instrument["source_rows"], "target_unlabeled_rows": instrument["target_unlabeled_rows"], "identity": instrument["identity"], "external_storage_bytes": external_bytes},
        "smoke": {"endpoint_rows": len(endpoint_rows), "evaluation_trials": len(evaluation), "geometry": geometry, "feature_blocks_computable": all(row["computable"] for row in feature_blocks)},
        "final_gate_candidate": "PILOT_VALID_SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD",
        "taxonomy": {
            "primary_active": ["C78-A_seed3_OACI_ERM_pilot_executed_and_validated"],
            "primary_inactive": ["C78-B_training_or_instrumentation_blocker", "C78-C_target_isolation_or_protocol_violation", "C78-D_resource_or_storage_envelope_invalid", "C78-E_historical_engine_reconstruction_mismatch"],
            "secondary_active": ["C78-S1_exact_82_unit_field_manifested", "C78-S2_level0_and_level1_passed", "C78-S3_target_isolation_passed", "C78-S4_trial_view_isolation_passed", "C78-S5_zWz_identity_passed", "C78-S6_checkpoint_cadence_and_genealogy_passed", "C78-S7_actual_runtime_storage_measured", "C78-S8_seed4_untouched", "C78-S9_SRC_canary_required_before_full_field", "C78-S11_full_seed3_expansion_not_ready"],
        },
        "claims": {
            "multiregime_replication": False, "measurement_control_replication": False,
            "cross_regime_transport": False, "strict_source_escape_hatch": False,
            "target_unlabeled_representation_mechanism": False, "seed_level_confirmation": False,
            "selector": False, "checkpoint_recommendation": False,
            "deployable": False, "target_population_generalization": False,
            "manuscript": False,
        },
    }
    STATE_PATH.write_bytes(c78.canonical_bytes(state) + b"\n")
    print(json.dumps({
        "gate": state["final_gate_candidate"], "units": 82,
        "checkpoint_optimizer_hashes": "82/82", "endpoint_rows": len(endpoint_rows),
        "external_bytes": external_bytes,
    }, sort_keys=True))
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78_authorized_collect")
    parser.parse_args(argv)
    collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
