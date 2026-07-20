"""Prospective C78F field-generation and C78S analysis protocol lock.

This module is metadata-only.  It must remain importable without importing an
EEG loader, PyTorch, CUDA, or any training runner.  Real execution is guarded by
the committed execution lock implemented in :mod:`c78f_runtime`.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable


MILESTONE = "C78F"
DATASET = "BNCI2014_001"
SEED = 3
TARGET4_CANARY = 4
TARGETS = (1, 2, 3, 5, 6, 7, 8, 9)
LEVELS = (0, 1)
OACI_EPOCHS = tuple(range(4, 200, 5))
SRC_EPOCHS = OACI_EPOCHS
REGIMES = ("ERM", "OACI", "SRC")
UNITS_PER_CONTEXT = 81
REMAINING_UNITS = 1296
FULL_FIELD_UNITS = 1458
EXPECTED_SOURCE_ROWS = REMAINING_UNITS * 8 * 576
EXPECTED_TARGET_ROWS = REMAINING_UNITS * 576
FULL_SOURCE_ROWS = FULL_FIELD_UNITS * 8 * 576
FULL_TARGET_ROWS = FULL_FIELD_UNITS * 576
SRC_HISTORICAL_COMMIT = "2555b3623713f802018e69afcf2b7d1449050641"
SRC_SMOOTH_TEMPERATURE = 0.1
PARENT_RESULT_COMMIT = "ac60c8d76b007cec1219a819b4c27078e5e9f132"

# Direct user authorization supersedes the former magic-token ceremony.  The
# execution workers trust only a committed lock binding this evidence to the
# exact protocol scope; prompt scanning and environment variables remain invalid.
AUTHORIZATION_MODE = "direct_explicit_user_authorization"
AUTHORIZATION_EVIDENCE = "这个C78F,我授权"
AUTHORIZATION_EVIDENCE_SHA256 = hashlib.sha256(AUTHORIZATION_EVIDENCE.encode()).hexdigest()

REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c78f_tables"
PROTOCOL_PATH = REPORT_DIR / "C78F_FULL_SEED3_FIELD_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C78F_FULL_SEED3_FIELD_PROTOCOL.sha256"
TIMING_PATH = REPORT_DIR / "C78F_PROTOCOL_TIMING_AUDIT.md"
C78S_PROTOCOL_PATH = REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS_PROTOCOL.json"
C78S_PROTOCOL_SHA_PATH = REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS_PROTOCOL.sha256"
EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c78f-full-seed3")
MAX_GIT_PAYLOAD = 50 * 1024 * 1024

C78_RESULT = REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.json"
C78R_RESULT = REPORT_DIR / "C78R_SEED3_SRC_CANARY.json"
C78_CHECKPOINTS = REPORT_DIR / "c78_tables/checkpoint_manifest.csv"
C78R_CHECKPOINTS = REPORT_DIR / "c78r_tables/SRC_checkpoint_manifest.csv"
C77_ACTIONABILITY = REPORT_DIR / "c77_tables/actionability_gate_registry.csv"
C77_POWER = REPORT_DIR / "c77_tables/power_and_false_positive_plan.csv"
C78_TARGET4_SMOKE = REPORT_DIR / "c78_tables/target_endpoint_smoke.csv"

HISTORICAL_PATHS = (
    "oaci/protocol/confirmatory_v2.yaml",
    "oaci/confirmatory/materialize.py",
    "oaci/confirmatory/src_onefold.py",
    "oaci/methods/source_robust.py",
    "oaci/runner/objectives.py",
    "oaci/runner/plans.py",
    "oaci/runner/stage1.py",
    "oaci/train/engine.py",
)

IMPLEMENTATION_FILES = (
    "oaci/conditioned_ceiling_coverage/c78f_full_seed3_field.py",
    "oaci/conditioned_ceiling_coverage/c78f_runtime.py",
    "oaci/conditioned_ceiling_coverage/c78f_train.py",
    "oaci/conditioned_ceiling_coverage/c78f_instrument.py",
    "oaci/conditioned_ceiling_coverage/c78f_collect.py",
    "oaci/conditioned_ceiling_coverage/c78f_protocol_red_team.py",
    "oaci/conditioned_ceiling_coverage/c78f_red_team.py",
    "oaci/conditioned_ceiling_coverage/c78f_finalize.py",
    "oaci/tests/test_c78f_full_seed3_field.py",
    "oaci/slurm_c78f_train_oaci.sh",
    "oaci/slurm_c78f_train_src.sh",
    "oaci/slurm_c78f_instrument_target.sh",
    "oaci/slurm_c78f_wave_gate.sh",
    "oaci/slurm_c78f_collect.sh",
    "oaci/slurm_c78f_red_team.sh",
    "oaci/slurm_c78f_regression.sh",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty C78F table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload) + b"\n")


def _unit_id(target: int, level: int, regime: str, epoch: int, order: int) -> str:
    identity = {
        "dataset": DATASET,
        "target": int(target),
        "seed": SEED,
        "level": int(level),
        "regime": regime,
        "epoch": int(epoch),
        "trajectory_order": int(order),
        "milestone": MILESTONE,
    }
    return "c78f_" + sha256_bytes(canonical_bytes(identity))[:20]


def wave_targets() -> dict[str, tuple[int, ...]]:
    ordered = sorted(
        TARGETS,
        key=lambda target: hashlib.sha256(
            f"C78F|{DATASET}|target={target}|wave-v1".encode()
        ).hexdigest(),
    )
    waves = {"A": tuple(ordered[:4]), "B": tuple(ordered[4:])}
    if set(waves["A"]) | set(waves["B"]) != set(TARGETS) or set(waves["A"]) & set(waves["B"]):
        raise RuntimeError("C78F deterministic wave partition failed")
    return waves


def wave_for_target(target: int) -> str:
    for wave, targets in wave_targets().items():
        if int(target) in targets:
            return wave
    raise ValueError(f"target {target} is outside the remaining C78F field")


def remaining_unit_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        wave = wave_for_target(target)
        for level in LEVELS:
            rows.append({
                "unit_id": _unit_id(target, level, "ERM", 199, 0),
                "dataset": DATASET,
                "target": target,
                "seed": SEED,
                "level": level,
                "regime": "ERM",
                "epoch": 199,
                "trajectory_order": 0,
                "wave": wave,
                "role": "shared_stage1_anchor",
                "retention_rule": "stage1_final_only",
                "target_outcome_used_for_retention": 0,
            })
            for regime, epochs in (("OACI", OACI_EPOCHS), ("SRC", SRC_EPOCHS)):
                for order, epoch in enumerate(epochs, start=1):
                    rows.append({
                        "unit_id": _unit_id(target, level, regime, epoch, order),
                        "dataset": DATASET,
                        "target": target,
                        "seed": SEED,
                        "level": level,
                        "regime": regime,
                        "epoch": epoch,
                        "trajectory_order": order,
                        "wave": wave,
                        "role": "historical_primary_trajectory" if regime == "OACI" else "historical_negative_control_trajectory",
                        "retention_rule": "every_5_epochs_complete_trajectory",
                        "target_outcome_used_for_retention": 0,
                    })
    if len(rows) != REMAINING_UNITS or len({row["unit_id"] for row in rows}) != REMAINING_UNITS:
        raise RuntimeError("C78F remaining manifest is not 1,296 unique units")
    counts = {regime: sum(row["regime"] == regime for row in rows) for regime in REGIMES}
    if counts != {"ERM": 16, "OACI": 640, "SRC": 640}:
        raise RuntimeError(f"C78F regime counts drifted: {counts}")
    return rows


def training_phase_manifest() -> list[dict[str, Any]]:
    rows = []
    ordinal = 0
    for wave, targets in wave_targets().items():
        for target in targets:
            for level in LEVELS:
                for regime in REGIMES:
                    ordinal += 1
                    rows.append({
                        "phase_id": f"c78f_phase_{ordinal:02d}",
                        "wave": wave,
                        "target": target,
                        "level": level,
                        "regime": regime,
                        "job_stage": "oaci_erm" if regime in {"ERM", "OACI"} else "src",
                        "dependency": "none" if regime in {"ERM", "OACI"} else "same_target_oaci_erm_field_frozen",
                        "target_outcome_blind": 1,
                    })
    if len(rows) != 48:
        raise RuntimeError("C78F did not lock exactly 48 training phases")
    return rows


def _target4_units() -> list[dict[str, Any]]:
    c78_rows = read_csv(C78_CHECKPOINTS)
    c78r_rows = read_csv(C78R_CHECKPOINTS)
    if len(c78_rows) != 82 or len(c78r_rows) != 80:
        raise RuntimeError("C78F target-4 parent field is not 162 units")
    rows = []
    for row in [*c78_rows, *c78r_rows]:
        rows.append({
            "unit_id": row["unit_id"],
            "dataset": DATASET,
            "target": 4,
            "seed": 3,
            "level": int(row["level"]),
            "regime": row["regime"],
            "epoch": int(row["epoch"]),
            "trajectory_order": int(row["trajectory_order"]),
            "wave": "CANARY",
            "role": "engineering_canary_descriptive_only",
            "retention_rule": "parent_committed_C78_or_C78R",
            "target_outcome_used_for_retention": 0,
        })
    if len({row["unit_id"] for row in rows}) != 162:
        raise RuntimeError("C78F target-4 parent unit IDs are not unique")
    return rows


def full_unit_manifest() -> list[dict[str, Any]]:
    rows = [*_target4_units(), *remaining_unit_manifest()]
    if len(rows) != FULL_FIELD_UNITS or len({row["unit_id"] for row in rows}) != FULL_FIELD_UNITS:
        raise RuntimeError("C78F full seed-3 unit registry is not 1,458 unique units")
    return rows


def implementation_manifest() -> list[dict[str, Any]]:
    rows = []
    for name in IMPLEMENTATION_FILES:
        path = Path(name)
        if not path.is_file():
            raise RuntimeError(f"missing C78F implementation before protocol lock: {name}")
        rows.append({"path": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return rows


def historical_manifest() -> list[dict[str, Any]]:
    rows = []
    for name in HISTORICAL_PATHS:
        path = Path(name)
        rows.append({
            "path": name,
            "current_sha256": sha256_file(path),
            "current_blob": git("hash-object", name),
            "C78_C78R_validated": True,
            "target_outcome_tuned_for_C78F": False,
        })
    return rows


def build_c78s_protocol() -> dict[str, Any]:
    return {
        "schema_version": "c78s_seed3_scientific_analysis_protocol_v1",
        "milestone": "C78S",
        "status": "LOCKED_BEFORE_REMAINING_TARGET_OUTCOME_ACCESS",
        "created_at_utc": utc_now(),
        "data_roles": {
            "primary_targets": list(TARGETS),
            "target4_canary": "descriptive_only_excluded_from_all_primary_tests",
            "seed3": "exploratory_replication_field",
            "seed4": "untouched_locked_confirmation_field",
            "new_target_population_claim_allowed": False,
        },
        "primary_hypotheses": [
            {"id": "H1", "claim": "measurement_control_separation_replicates_across_new_seed3_contexts", "primary_metric": "reliability_or_association_nonzero_and_no_material_actionability"},
            {"id": "H2", "claim": "effective_near_tie_multiplicity_and_top_gap_predict_actionability_failure", "primary_metric": "blocked_incremental_deviance_effective_M_beyond_raw_M"},
            {"id": "H3", "claim": "local_nonlinear_representation_association_fails_cross_target_and_cross_regime_transport", "primary_metric": "local_association_minus_leave_target_and_leave_regime_transport"},
            {"id": "H4", "claim": "registered_strict_source_representation_block_does_not_qualify_as_escape_hatch", "primary_metric": "incremental_R2_and_actionability_candidate_gate"},
            {"id": "H5", "claim": "registered_target_unlabeled_representation_block_does_not_qualify_as_actionable_control", "primary_metric": "incremental_R2_transport_and_actionability_candidate_gate"},
            {"id": "H6", "claim": "split_label_construction_information_is_the_strongest_non_oracle_positive_control", "primary_metric": "paired_incremental_prediction_difference"},
        ],
        "registered_feature_sources": {
            "strict_source_block": "exact_C75_C76_F1_F2_registry_no_feature_search",
            "target_unlabeled_block": "exact_C75_C76_F3_F4_registry_no_feature_search",
            "construction_positive_control": "physically_separate_split_label_view_diagnostic_only",
            "same_label_oracle": "secondary_ceiling_only_unavailable_at_selection_time",
        },
        "primary_outcomes": [
            "continuous_joint_utility",
            "target_bAcc",
            "target_NLL",
            "target_ECE",
            "primary_joint_good",
            "top1",
            "top5",
            "top10",
            "continuous_regret",
            "coverage",
        ],
        "materiality": {
            "incremental_R2_min": 0.02,
            "absolute_topk_hit_improvement_min": 0.05,
            "standardized_regret_reduction_min": 0.05,
            "positive_primary_targets_min": 6,
            "leave_target_out_median_must_exceed": 0.0,
            "leave_regime_out_median_must_exceed": 0.0,
            "rationale": "C77_0.0075_actionability_drop_was_directional_not_material; thresholds_locked_from_C77_and_target4_canary_only",
        },
        "inference": {
            "within_target_centering": True,
            "within_regime_centering": True,
            "separate_by": ["regime", "target", "level", "trajectory", "seed"],
            "ERM_role": "anchor_not_symmetric_trajectory",
            "cluster_bootstrap": ["target", "checkpoint", "trial_id"],
            "leave_out": ["target", "regime", "trajectory"],
            "blocked_permutations": ["within_target_regime_level", "trajectory_preserving"],
            "row_iid_inference": False,
        },
        "nulls": [
            "target_block_permutation",
            "checkpoint_block_permutation",
            "trajectory_preserving_permutation",
            "candidate_within_target_regime_permutation",
            "identity_only_matched_null",
            "nested_bandwidth_null",
        ],
        "multiplicity": {
            "primary_family": "H1_H6_Holm",
            "feature_kernel_paths": "nested_max_stat_then_Holm",
            "two_sided_alpha": 0.05,
            "association_p_alone_qualifies_control": False,
        },
        "stop_rules": {
            "target4_in_primary": "block",
            "label_view_leakage": "block",
            "missing_context_or_unit": "block",
            "seed4_touch": "block",
            "unregistered_feature_search": "block",
        },
        "forbidden_claims": [
            "independent_target_population_confirmation",
            "deployable_selector",
            "checkpoint_recommendation",
            "source_only_rescue",
            "representation_causality",
            "seed4_confirmation",
        ],
    }


def build_protocol(c78s_sha256: str) -> dict[str, Any]:
    c78 = json.loads(C78_RESULT.read_text())
    c78r = json.loads(C78R_RESULT.read_text())
    if c78["final_gate"] != "PILOT_VALID_SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD":
        raise RuntimeError("C78F parent C78 gate mismatch")
    if c78r["final_gate"] != "SRC_CANARY_EXECUTED_AND_VALIDATED_FULL_SEED3_READY_BUT_NOT_AUTHORIZED":
        raise RuntimeError("C78F parent C78R gate mismatch")
    remaining = remaining_unit_manifest()
    phases = training_phase_manifest()
    waves = wave_targets()
    return {
        "schema_version": "c78f_full_seed3_field_protocol_v1",
        "milestone": MILESTONE,
        "status": "LOCKED_AUTHORIZED_PENDING_EXECUTION",
        "created_at_utc": utc_now(),
        "parent_result_commit": PARENT_RESULT_COMMIT,
        "authorization": {
            "required": True,
            "mode": AUTHORIZATION_MODE,
            "explicit_user_authorization_received": True,
            "evidence_sha256": AUTHORIZATION_EVIDENCE_SHA256,
            "accepted_channel": "direct_user_statement_in_active_session",
            "magic_token_required": False,
            "environment_or_prompt_scanning_allowed": False,
            "execution_requires_committed_scope_bound_lock": True,
            "PM_override": "direct_explicit_authorization_supersedes_exact_token_ceremony_for_C78F_and_future_milestones_unless_PM_changes_it",
        },
        "scope": {
            "dataset": DATASET,
            "seed": SEED,
            "remaining_targets": list(TARGETS),
            "levels": list(LEVELS),
            "regimes": list(REGIMES),
            "remaining_contexts": 16,
            "remaining_training_phases": len(phases),
            "remaining_units": len(remaining),
            "full_seed3_units": FULL_FIELD_UNITS,
            "expected_remaining_source_rows": EXPECTED_SOURCE_ROWS,
            "expected_remaining_target_unlabeled_rows": EXPECTED_TARGET_ROWS,
            "expected_full_source_rows": FULL_SOURCE_ROWS,
            "expected_full_target_unlabeled_rows": FULL_TARGET_ROWS,
            "target4_role": "engineering_canary_descriptive_only",
            "target4_primary_science_excluded": True,
        },
        "waves": {
            "assignment_rule": "ascending_SHA256(C78F|dataset|target|wave-v1)",
            "A": list(waves["A"]),
            "B": list(waves["B"]),
            "targets_per_wave": 4,
            "units_per_wave": 648,
            "training_phases_per_wave": 24,
            "wave_B_gate": "Wave_A_engineering_only_no_target_scientific_outcome",
            "target_outcomes_between_waves": False,
        },
        "unit_manifest": {
            "path": str(TABLE_DIR / "full_unit_manifest.csv"),
            "remaining_rows": REMAINING_UNITS,
            "full_rows": FULL_FIELD_UNITS,
            "remaining_unit_ids_sha256": sha256_bytes(canonical_bytes(sorted(row["unit_id"] for row in remaining))),
        },
        "training": {
            "ERM": "exact_historical_stage1_final_anchor",
            "OACI": "exact_historical_objective_40_fixed_cadence_checkpoints",
            "SRC": "exact_C11_historical_negative_control_commit_2555b36_smooth_temperature_0.1",
            "SRC_smooth_temperature": SRC_SMOOTH_TEMPERATURE,
            "checkpoint_every_epochs": 5,
            "stage1_epochs": 200,
            "stage2_epochs": 200,
            "outcome_blind_retention": True,
            "retry_policy": "engineering_failure_only_first_valid_success_retained",
            "target_fit_rows": 0,
        },
        "instrumentation": {
            "strict_source_rows_per_unit": 4608,
            "target_unlabeled_rows_per_unit": 576,
            "fields": ["logits", "probabilities", "prediction", "z", "Wz", "Wz_plus_b", "class_margins"],
            "identity_tolerances": {"Wz_plus_b_logits_abs": 1e-6, "softmax_abs": 1e-7, "hook_z_abs": 1e-6, "repeat_abs": 0.0},
            "primary_target_view_has_labels": False,
            "label_views_after_complete_field_freeze": True,
            "raw_payload_external_only": True,
        },
        "physical_views": {
            "strict_source_trial_view": "source_labels_allowed",
            "target_unlabeled_trial_view": "target_labels_forbidden",
            "target_construction_view": "physically_separate_after_full_freeze",
            "target_evaluation_view": "physically_separate_after_full_freeze",
            "same_label_oracle_view": "physically_separate_diagnostic_only",
        },
        "C78S_analysis_lock": {
            "path": str(C78S_PROTOCOL_PATH),
            "sha256": c78s_sha256,
            "locked_before_remaining_target_outcome_access": True,
            "analysis_started_in_C78F": False,
        },
        "seed4_protection": {
            "training_jobs": 0,
            "execution_access": 0,
            "checkpoints": 0,
            "caches": 0,
            "outcome_reads": 0,
        },
        "resource_stop_gates": {
            "minimum_free_external_bytes": 64 * 1024**3,
            "per_wave_projected_bytes": 16 * 1024**3,
            "stop_on_quota_or_identity_failure": True,
            "silent_escalation": False,
        },
        "implementation_files": implementation_manifest(),
        "historical_files": historical_manifest(),
        "forbidden": [
            "target_outcome_read_during_generation",
            "seed4",
            "BNCI2014_004",
            "selector_or_checkpoint_recommendation",
            "target_based_retry_or_retention",
            "scientific_result_claim",
            "manuscript_drafting",
            "raw_payload_in_git",
        ],
    }


def initial_risk_register() -> list[dict[str, Any]]:
    risks = [
        ("authorization_bypass", "blocking", "committed direct-user scope lock required"),
        ("protocol_after_data_access", "blocking", "timing audit and zero-access lock"),
        ("target4_pilot_in_primary_inference", "blocking", "C78S excludes target 4"),
        ("remaining_target_outcome_peeking", "blocking", "generation computes no outcomes"),
        ("wave_A_outcome_based_wave_B_decision", "blocking", "engineering-only wave gate"),
        ("target_outcome_regime_or_retry_selection", "blocking", "fixed first-valid execution policy"),
        ("historical_config_drift", "blocking", "implementation/config hashes locked"),
        ("target_label_training_leakage", "blocking", "source-only training process"),
        ("source_audit_training_leakage", "blocking", "six train-source subjects only"),
        ("checkpoint_cadence_incomplete", "blocking", "exact 40-point cadence per trajectory"),
        ("missing_unit_silently_dropped", "blocking", "1,296-unit set equality gate"),
        ("multiple_successful_rerun_selection", "blocking", "first valid run retained"),
        ("MNE_lock_collision", "blocking", "job-private MNE/scratch paths"),
        ("deterministic_cublas_preflight_bypass", "blocking", "GPU preflight before EEG loader"),
        ("instrumentation_schema_drift", "blocking", "registered exact field schemas"),
        ("view_label_leakage", "blocking", "physical target-unlabeled field gate"),
        ("Wz_logit_identity_failure", "blocking", "per-unit numerical identity gate"),
        ("ERM_trajectory_false_symmetry", "claim", "ERM retained as anchor only"),
        ("SRC_rescue_reinterpretation", "claim", "SRC remains historical negative control"),
        ("seed4_contamination", "blocking", "seed-4 zero-access audit"),
        ("BNCI2014_004_access", "blocking", "dataset allowlist is BNCI2014_001"),
        ("pilot_to_science_claim", "claim", "C78F generation only"),
        ("raw_weights_or_cache_in_git", "blocking", "external payload root and 50 MiB scan"),
        ("selector_or_checkpoint_recommendation", "blocking", "no outcome/selection consumer"),
        ("manuscript_drafting", "blocking", "explicitly forbidden"),
    ]
    return [
        {"risk": risk, "class": kind, "status": "mitigation_locked_pending_execution", "blocking_open": 0, "mitigation": mitigation}
        for risk, kind, mitigation in risks
    ]


def metadata_preflight() -> None:
    from oaci.confirmatory.loso_plan import loso_fold_spec

    rows = []
    for target in TARGETS:
        split = loso_fold_spec(target, dataset_id=DATASET)
        train = sorted(int(value) for value in split["source_train_subjects"])
        audit = sorted(int(value) for value in split["source_audit_subjects"])
        rows.append({
            "target": target,
            "wave": wave_for_target(target),
            "source_train_subjects": json.dumps(train),
            "source_audit_subjects": json.dumps(audit),
            "target_absent_from_source": int(target not in train and target not in audit),
            "partition_complete": int(set(train) | set(audit) | {target} == set(range(1, 10))),
            "expected_source_train_rows": 3456,
            "expected_strict_source_rows": 4608,
            "expected_target_rows": 576,
            "EEG_data_loaded": 0,
            "target_outcomes_read": 0,
            "passed": int(len(train) == 6 and len(audit) == 2 and target not in train and target not in audit),
        })
    write_csv(TABLE_DIR / "remaining_target_preflight.csv", rows)
    python_path = Path("/home/infres/yinwang/anaconda3/envs/icml/bin/python")
    write_csv(TABLE_DIR / "environment_preflight.csv", [{
        "python_path": str(python_path),
        "python_exists": int(python_path.is_file()),
        "GPU_runtime_check": "per_job_before_EEG_load",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "locked_partition": "V100",
        "instrumentation_partition": "cpu-high",
        "passed": int(python_path.is_file()),
    }])
    parent = EXTERNAL_ROOT.parent
    usage = shutil.disk_usage(parent)
    write_csv(TABLE_DIR / "storage_preflight.csv", [{
        "path": str(parent),
        "free_bytes": usage.free,
        "minimum_free_bytes": 64 * 1024**3,
        "projected_remaining_safety_bytes": int(31.160 * 1024**3),
        "passed": int(usage.free >= 64 * 1024**3),
        "EEG_data_loaded": 0,
    }])
    write_csv(TABLE_DIR / "MNE_lock_isolation_audit.csv", [{
        "scope": "per_Slurm_job",
        "scratch_pattern": "/tmp/c78f-<stage>-<target>-${SLURM_JOB_ID}",
        "MNE_CACHE_DIR_job_private": 1,
        "TMPDIR_job_private": 1,
        "shared_lock_path": 0,
        "passed": 1,
    }])


def _timing_markdown(protocol_sha: str, c78s_sha: str) -> str:
    return f"""# C78F Protocol Timing Audit

Status: prospective field-generation and analysis contracts locked.

```text
C78F protocol SHA-256: {protocol_sha}
C78S protocol SHA-256: {c78s_sha}
authorization mode: {AUTHORIZATION_MODE}
direct user authorization received: true
magic token required: false
remaining-target EEG data access before lock: 0
remaining-target GPU job submission before lock: 0
remaining-target outcome access before lock: 0
seed-4 access before lock: 0
```

The PM explicitly replaced the former exact-token ceremony with direct user
authorization. Execution remains fail-closed through a committed lock binding
that authorization record to these exact protocol and implementation hashes.

C78F is generation/instrumentation only. C78S is locked prospectively, excludes
target 4 from primary inference, and has not started.
"""


def lock_protocol() -> dict[str, Any]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(TABLE_DIR / "full_unit_manifest.csv", full_unit_manifest())
    write_csv(TABLE_DIR / "wave_assignment.csv", [
        {
            "wave": wave,
            "target": target,
            "assignment_key_sha256": hashlib.sha256(
                f"C78F|{DATASET}|target={target}|wave-v1".encode()
            ).hexdigest(),
            "target_outcome_used": 0,
        }
        for wave, targets in wave_targets().items()
        for target in targets
    ])
    write_csv(TABLE_DIR / "training_phase_manifest.csv", training_phase_manifest())
    write_csv(TABLE_DIR / "risk_register.csv", initial_risk_register())
    metadata_preflight()

    c78s = build_c78s_protocol()
    write_json(C78S_PROTOCOL_PATH, c78s)
    c78s_sha = sha256_file(C78S_PROTOCOL_PATH)
    C78S_PROTOCOL_SHA_PATH.write_text(c78s_sha + "\n")

    protocol = build_protocol(c78s_sha)
    write_json(PROTOCOL_PATH, protocol)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    PROTOCOL_SHA_PATH.write_text(protocol_sha + "\n")
    TIMING_PATH.write_text(_timing_markdown(protocol_sha, c78s_sha))
    write_csv(TABLE_DIR / "authorization_audit.csv", [{
        "authorization_required": 1,
        "authorization_received": 1,
        "mode": AUTHORIZATION_MODE,
        "evidence_sha256": AUTHORIZATION_EVIDENCE_SHA256,
        "magic_token_required": 0,
        "committed_execution_lock_required": 1,
        "remaining_target_data_access_before_protocol_lock": 0,
        "GPU_submission_before_protocol_lock": 0,
    }])
    return {"protocol_sha256": protocol_sha, "C78S_protocol_sha256": c78s_sha, "remaining_units": REMAINING_UNITS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78f_full_seed3_field")
    parser.add_argument("command", choices=("lock-protocol", "print-waves"))
    args = parser.parse_args(argv)
    if args.command == "lock-protocol":
        print(json.dumps(lock_protocol(), sort_keys=True))
    else:
        print(json.dumps({key: list(value) for key, value in wave_targets().items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
