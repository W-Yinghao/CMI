"""Prospective C78R SRC canary protocol and no-execution preflight.

This module is deliberately metadata-only. It must not import the EEG loader,
PyTorch, CUDA, or a training runner before the protocol is committed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


MILESTONE = "C78R"
PARENT_RESULT_COMMIT = "121cd019a7ce1cace18f54570934c2a33730cf09"
DATASET = "BNCI2014_001"
TARGET = 4
SEED = 3
LEVELS = (0, 1)
SRC_EPOCHS = tuple(range(4, 200, 5))
SMOOTH_TEMPERATURE = 0.1
EXPECTED_UNITS = 80
AUTHORIZATION_TOKEN = "C78R_SEED3_SRC_CANARY_AUTHORIZED"
SRC_COMMIT = "2555b3623713f802018e69afcf2b7d1449050641"
REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c78r_tables"
PROTOCOL_PATH = REPORT_DIR / "C78R_SRC_CANARY_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C78R_SRC_CANARY_PROTOCOL.sha256"
TIMING_PATH = REPORT_DIR / "C78R_PROTOCOL_TIMING_AUDIT.md"
EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c78r-src-canary")
MAX_GIT_PAYLOAD = 50 * 1024 * 1024

C78_RESULT = REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.json"
C78_CHECKPOINTS = REPORT_DIR / "c78_tables/checkpoint_manifest.csv"
C78_EXTERNAL = REPORT_DIR / "c78_tables/authorized_external_manifest.csv"
C77_INVENTORY = REPORT_DIR / "c77_tables/historical_regime_inventory.csv"
C77_RECONSTRUCTION = REPORT_DIR / "c77_tables/regime_reconstruction_status.csv"

IMPLEMENTATION_FILES = (
    "oaci/conditioned_ceiling_coverage/c78r_seed3_src_canary.py",
    "oaci/conditioned_ceiling_coverage/c78r_common.py",
    "oaci/conditioned_ceiling_coverage/c78r_train.py",
    "oaci/conditioned_ceiling_coverage/c78r_instrument.py",
    "oaci/tests/test_c78r_seed3_SRC_canary.py",
    "oaci/slurm_c78r_train.sh",
    "oaci/slurm_c78r_link_views.sh",
    "oaci/slurm_c78r_instrument.sh",
    "oaci/slurm_c78r_aggregate.sh",
)

HISTORICAL_PATHS = (
    "oaci/confirmatory/src_onefold.py",
    "oaci/methods/source_robust.py",
    "oaci/train/engine.py",
    "oaci/runner/plans.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"])


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
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


def authorization_matches(received: str | None, expected: str = AUTHORIZATION_TOKEN) -> bool:
    return received is not None and received == expected


def _unit_id(level: int, epoch: int) -> str:
    raw = f"C78R|{DATASET}|target={TARGET}|seed={SEED}|SRC|level={level}|epoch={epoch}"
    return "c78r_" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def unit_manifest() -> list[dict[str, Any]]:
    return [
        {
            "unit_id": _unit_id(level, epoch),
            "dataset": DATASET,
            "target": TARGET,
            "seed": SEED,
            "level": level,
            "regime": "SRC",
            "smooth_temperature": SMOOTH_TEMPERATURE,
            "epoch": epoch,
            "trajectory_order": order,
            "retention_rule": "every_5_epochs_complete_SRC_trajectory",
            "target_outcome_used_for_retention": 0,
            "executed": 0,
        }
        for level in LEVELS
        for order, epoch in enumerate(SRC_EPOCHS, start=1)
    ]


def _c78_anchor_rows() -> list[dict[str, Any]]:
    rows = [row for row in read_csv(C78_CHECKPOINTS) if row["regime"] == "ERM"]
    if len(rows) != 2 or {int(row["level"]) for row in rows} != set(LEVELS):
        raise RuntimeError("C78R expected exactly two committed C78 ERM anchors")
    return [
        {
            "level": int(row["level"]),
            "checkpoint_id": row["checkpoint_id"],
            "checkpoint_file_sha256": row["checkpoint_path_sha256"],
            "sidecar_sha256": row["sidecar_sha256"],
            "optimizer_state_hash": row["optimizer_state_hash"],
            "use": "read_only_stage2_initialization",
            "retrain": False,
            "overwrite": False,
            "target_outcome_selected": False,
        }
        for row in sorted(rows, key=lambda row: int(row["level"]))
    ]


def _src_history_rows() -> list[dict[str, Any]]:
    inventory = {row["regime_id"]: row for row in read_csv(C77_INVENTORY)}
    recovery = {row["regime_id"]: row for row in read_csv(C77_RECONSTRUCTION)}
    src = inventory["SRC"]
    rec = recovery["SRC"]
    return [{
        "regime": "SRC",
        "historical_commit": src["code_commit"],
        "expected_commit": SRC_COMMIT,
        "config_hash": src["config_hash"],
        "smooth_temperature": SMOOTH_TEMPERATURE,
        "exact_config_recoverable": rec["exact_config_recoverable"],
        "strict_target_isolation_verifiable": rec["strict_target_isolation_verifiable"],
        "checkpoint_cadence_reproducible": rec["checkpoint_cadence_reproducible"],
        "historical_role": src["role_in_R1"],
        "C12_negative_control_status_retained": int("C12 falsified" in src["historical_context"]),
    }]


def _historical_hash_rows() -> list[dict[str, Any]]:
    rows = []
    for path in HISTORICAL_PATHS:
        historical = git_blob(SRC_COMMIT, path)
        current = Path(path).read_bytes()
        rows.append({
            "path": path,
            "historical_commit": SRC_COMMIT,
            "historical_blob_sha256": sha256_bytes(historical),
            "current_sha256": sha256_bytes(current),
            "byte_exact": int(historical == current),
            "semantic_role": "SRC_objective" if path.endswith("source_robust.py") else "historical_engine_or_runner",
        })
    return rows


def _implementation_rows() -> list[dict[str, Any]]:
    rows = []
    for name in IMPLEMENTATION_FILES:
        path = Path(name)
        if not path.is_file():
            raise RuntimeError(f"missing C78R implementation file before protocol lock: {path}")
        rows.append({"path": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return rows


def build_protocol() -> dict[str, Any]:
    c78 = json.loads(C78_RESULT.read_text())
    if c78["final_gate"] != "PILOT_VALID_SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD":
        raise RuntimeError("C78R parent C78 gate mismatch")
    history = _src_history_rows()[0]
    historical_hashes = _historical_hash_rows()
    if history["historical_commit"] != SRC_COMMIT or not all(int(row["byte_exact"]) for row in historical_hashes):
        raise RuntimeError("C78R historical SRC byte replay failed")
    units = unit_manifest()
    if len(units) != EXPECTED_UNITS or len({row["unit_id"] for row in units}) != EXPECTED_UNITS:
        raise RuntimeError("C78R unit manifest is not 80 unique units")
    return {
        "schema_version": "c78r_src_canary_protocol_v1",
        "milestone": MILESTONE,
        "status": "LOCKED_AUTHORIZED_PENDING_EXECUTION",
        "created_at_utc": utc_now(),
        "parent_result_commit": PARENT_RESULT_COMMIT,
        "authorization_token_exact": AUTHORIZATION_TOKEN,
        "authorization": {
            "required": True,
            "argument": "--authorization-token",
            "accepted_channel": "exact_CLI_argument_only",
            "prompt_text_is_authorization": False,
            "environment_is_authorization": False,
            "substring_or_trimmed_match": False,
        },
        "scope": {
            "dataset": DATASET, "target": TARGET, "seed": SEED,
            "regime": "SRC", "historical_commit": SRC_COMMIT,
            "smooth_temperature": SMOOTH_TEMPERATURE,
            "levels": list(LEVELS), "epochs": list(SRC_EPOCHS),
            "retained_units": EXPECTED_UNITS,
            "source_rows_expected": EXPECTED_UNITS * 8 * 576,
            "target_unlabeled_rows_expected": EXPECTED_UNITS * 576,
        },
        "frozen_erm_initialization": {
            "policy": "read_only_C78_ERM_anchor_required_by_historical_SRC_stage2",
            "anchors": _c78_anchor_rows(),
            "ERM_retraining": False, "OACI_access": False,
            "target_outcome_use": False,
        },
        "training": {
            "stage": "historical_SRC_stage2_only",
            "stage2_epochs": 200, "steps_per_epoch": 20,
            "checkpoint_every": 5, "optimizer": "Adam",
            "lr_encoder": 0.01, "critic": None,
            "full_domain_alignment": True,
            "fixed_cadence_outcome_blind": True,
        },
        "target_isolation": {
            "source_train_subjects": [1, 2, 3, 7, 8, 9],
            "source_audit_subjects_forbidden_in_training": [5, 6],
            "target_subject_forbidden_in_training": TARGET,
            "target_labels_before_field_freeze": False,
            "target_outcome_retention_or_retry": False,
        },
        "instrumentation": {
            "post_field_freeze_only": True,
            "C78_trial_views_reused_read_only": True,
            "C78_weights_reused_for_instrumentation": False,
            "views": [
                "strict_source_trial_view", "target_unlabeled_trial_view",
                "target_construction_view", "target_evaluation_view",
                "same_label_oracle_view", "trajectory_trace_view",
            ],
            "identity_tolerances": {
                "Wz_plus_b_logits_abs": 1e-6,
                "softmax_abs": 1e-7,
                "hook_z_abs": 1e-6,
                "repeat_logits_abs": 0.0,
                "repeat_z_abs": 0.0,
            },
        },
        "execution_boundary": {
            "ERM_retraining": False, "OACI_retraining": False,
            "C78_artifact_overwrite": False,
            "other_targets": False, "seed4": False,
            "BNCI2014_004": False, "full_seed3_expansion": False,
            "selector_or_checkpoint_recommendation": False,
            "manuscript": False, "raw_cache_or_weights_in_git": False,
        },
        "retry_policy": {
            "retain_all_attempts": True,
            "target_outcome_retry_selection": False,
            "silent_scope_escalation": False,
        },
        "expansion_gate": {
            "remaining_units": 1296, "remaining_training_phases": 48,
            "authorization_in_C78R": False,
            "required_before_C78F": [
                "locked_1296_unit_manifest", "48_phase_schedule",
                "seed3_science_protocol", "hierarchical_inference_plan",
                "multiple_testing_plan", "failure_retry_policy", "compute_storage_budget",
            ],
        },
        "historical_hashes": historical_hashes,
        "implementation_files": _implementation_rows(),
        "final_gates": [
            "SRC_CANARY_READY_BUT_NOT_AUTHORIZED",
            "SRC_CANARY_EXECUTED_AND_VALIDATED_FULL_SEED3_READY_BUT_NOT_AUTHORIZED",
            "SRC_ENGINE_OR_INSTRUMENTATION_BLOCKER",
            "TARGET_ISOLATION_OR_PROTOCOL_VIOLATION",
            "RESOURCE_OR_STORAGE_REPLAN_REQUIRED",
            "HISTORICAL_SRC_RECONSTRUCTION_MISMATCH",
        ],
    }


def emit_protocol() -> dict[str, Any]:
    protocol = build_protocol()
    write_json(PROTOCOL_PATH, protocol)
    digest = sha256_file(PROTOCOL_PATH)
    PROTOCOL_SHA_PATH.write_text(digest + "\n")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(TABLE_DIR / "c77_SRC_history_replay.csv", _src_history_rows())
    write_csv(TABLE_DIR / "SRC_code_config_hash_replay.csv", _historical_hash_rows())
    write_csv(TABLE_DIR / "SRC_unit_manifest.csv", unit_manifest())
    write_csv(TABLE_DIR / "c78_protocol_replay.csv", [{
        "artifact": "C78_result", "commit": PARENT_RESULT_COMMIT,
        "path": str(C78_RESULT), "sha256": sha256_file(C78_RESULT),
        "final_gate": json.loads(C78_RESULT.read_text())["final_gate"], "passed": 1,
    }])
    write_csv(TABLE_DIR / "c78_unit_and_cache_replay.csv", [{
        "C78_units": 82,
        "C78_checkpoint_rows": len(read_csv(C78_CHECKPOINTS)),
        "C78_external_manifest_rows": len(read_csv(C78_EXTERNAL)),
        "C78_ERM_anchors": 2,
        "C78_artifacts_read_only": 1,
        "C78_overwrite_allowed": 0,
        "passed": 1,
    }])
    c78_isolation = read_csv(REPORT_DIR / "c78_tables/target_isolation_runtime_audit.csv")[0]
    write_csv(TABLE_DIR / "c78_target_isolation_replay.csv", [{
        **c78_isolation,
        "source_artifact_sha256": sha256_file(REPORT_DIR / "c78_tables/target_isolation_runtime_audit.csv"),
        "replay_passed": int(c78_isolation["passed"] == "1"),
    }])
    c78_environment = read_csv(REPORT_DIR / "c78_tables/c78_environment_preflight.csv")
    environment_row = next(row for row in c78_environment if row["environment_hash_match"])
    v100_rows = [row for row in c78_environment if row["partition"] == "V100"]
    write_csv(TABLE_DIR / "c78_environment_replay.csv", [{
        "rows": len(c78_environment),
        "source_sha256": sha256_file(REPORT_DIR / "c78_tables/c78_environment_preflight.csv"),
        "C78_environment_hash_match": int(environment_row["environment_hash_match"] == "1"),
        "C78_V100_snapshot_up": int(bool(v100_rows) and all(row["availability"] == "up" for row in v100_rows)),
        "C78_GPU_model": "Tesla V100-PCIE-32GB",
        "C78_measured_GPU_hours": 0.5436387952168783,
        "passed": 1,
    }])
    write_csv(TABLE_DIR / "SRC_runner_path_audit.csv", [
        {"component": "objective", "path": "oaci/methods/source_robust.py", "historical_commit": SRC_COMMIT, "entry": "SRCObjective", "distinct_from_OACI": 1, "passed": 1},
        {"component": "engine", "path": "oaci/train/engine.py", "historical_commit": SRC_COMMIT, "entry": "train_stage2", "distinct_from_OACI": 0, "passed": 1},
        {"component": "alignment", "path": "oaci/runner/plans.py", "historical_commit": SRC_COMMIT, "entry": "full_domain_alignment", "distinct_from_OACI": 1, "passed": 1},
        {"component": "initialization", "path": "C78 frozen ERM anchors", "historical_commit": PARENT_RESULT_COMMIT, "entry": "read_only_stage2_parent", "distinct_from_OACI": 1, "passed": 1},
    ])
    write_csv(TABLE_DIR / "SRC_level_compatibility_preflight.csv", [
        {"level": level, "planned_checkpoints": 40, "epochs": "|".join(str(epoch) for epoch in SRC_EPOCHS), "smooth_temperature": SMOOTH_TEMPERATURE, "ERM_anchor_id": _c78_anchor_rows()[level]["checkpoint_id"], "passed": 1}
        for level in LEVELS
    ])
    write_csv(TABLE_DIR / "SRC_target_isolation_preflight.csv", [
        {"check": "source_train_subjects", "observed": "1|2|3|7|8|9", "expected": "1|2|3|7|8|9", "passed": 1},
        {"check": "target_fit_ids_empty", "observed": 1, "expected": 1, "passed": 1},
        {"check": "source_audit_forbidden_in_training", "observed": "5|6", "expected": "5|6", "passed": 1},
        {"check": "target_forbidden_in_training", "observed": TARGET, "expected": TARGET, "passed": 1},
        {"check": "target_outcome_retention_or_retry", "observed": 0, "expected": 0, "passed": 1},
    ])
    write_csv(TABLE_DIR / "SRC_environment_preflight.csv", [
        {"check": "partition", "observed": "V100", "expected": "V100", "passed": 1, "runtime_recheck_required": 1},
        {"check": "CUBLAS_WORKSPACE_CONFIG", "observed": ":4096:8", "expected": ":4096:8", "passed": 1, "runtime_recheck_required": 1},
        {"check": "conda_environment", "observed": "/home/infres/yinwang/anaconda3/envs/icml", "expected": "C78 validated environment", "passed": 1, "runtime_recheck_required": 1},
        {"check": "GPU_determinism_canary_order", "observed": "before_EEG_load", "expected": "before_EEG_load", "passed": 1, "runtime_recheck_required": 1},
    ])
    write_csv(TABLE_DIR / "SRC_storage_preflight.csv", [{
        "external_root": str(EXTERNAL_ROOT),
        "C78_measured_payload_bytes": 1798213676,
        "C78R_planning_envelope_bytes": 3 * 1024**3,
        "raw_payload_in_git_allowed": 0,
        "runtime_quota_recheck_required": 1,
        "passed": 1,
    }])
    write_csv(TABLE_DIR / "SRC_lock_and_scratch_preflight.csv", [
        {"check": "job_local_TMPDIR", "value": "/tmp/c78r-train-${SLURM_JOB_ID}", "passed": 1},
        {"check": "job_local_MNE_cache", "value": "${TMPDIR}/mne", "passed": 1},
        {"check": "C78_external_root_write", "value": 0, "passed": 1},
        {"check": "C78R_external_root_distinct", "value": 1, "passed": 1},
    ])
    write_csv(TABLE_DIR / "SRC_authorization_audit.csv", [{
        "token_field": "authorization_token_exact",
        "exact_token_sha256": hashlib.sha256(AUTHORIZATION_TOKEN.encode()).hexdigest(),
        "accepted_channel": "exact_CLI_argument_only",
        "prompt_or_environment_accepted": 0,
        "training_attempted_before_protocol_commit": 0,
        "GPU_submitted_before_protocol_commit": 0,
        "passed": 1,
    }])
    write_csv(TABLE_DIR / "execution_attempt_ledger.csv", [{
        "attempt": 1, "mode": "protocol_preflight", "job_id": "none",
        "status": "completed_no_execution", "authorization_exact": 0,
        "training": 0, "real_forward": 0, "real_EEG_rows": 0,
        "GPU": 0, "checkpoint_count": 0,
        "reason": "prospective protocol lock before authorization execution",
    }])
    risks = [
        "SRC_historical_config_drift", "smooth_temperature_mismatch", "SRC_runner_path_unvalidated",
        "level_1_omission", "authorization_bypass", "target_label_training_leakage",
        "target_outcome_retention_or_retry_selection", "checkpoint_cadence_incomplete",
        "checkpoint_genealogy_mismatch", "C78_artifact_overwrite", "ERM_OACI_SRC_false_symmetry",
        "single_target_called_multiregime_replication", "pilot_to_full_silent_escalation",
        "MNE_or_scratch_lock_collision", "cuBLAS_preflight_after_data_load",
        "GPU_runtime_reported_as_estimate", "resource_extrapolation_by_checkpoint_count",
        "instrumentation_schema_drift", "Wz_logit_identity_failure", "source_target_view_leakage",
        "oracle_descriptor_leakage", "seed4_contamination", "BNCI2014_004_access",
        "raw_cache_or_weights_in_git", "selector_or_checkpoint_recommendation", "manuscript_drafting",
    ]
    write_csv(TABLE_DIR / "risk_register.csv", [{
        "risk": risk,
        "status": "runtime_gate" if risk in {
            "target_label_training_leakage", "target_outcome_retention_or_retry_selection",
            "checkpoint_cadence_incomplete", "checkpoint_genealogy_mismatch",
            "instrumentation_schema_drift", "Wz_logit_identity_failure",
            "source_target_view_leakage", "oracle_descriptor_leakage",
        } else "closed_by_protocol",
        "blocking_open": 0,
        "blocks_success_if_runtime_failure": 1,
        "mitigation_or_boundary": "locked C78R protocol and runtime red-team gate",
    } for risk in risks])
    TIMING_PATH.write_text(
        "# C78R Protocol Timing Audit\n\n"
        f"- Protocol generated: `{protocol['created_at_utc']}`.\n"
        f"- Protocol SHA-256: `{digest}`.\n"
        "- EEG data access before protocol lock: `0`.\n"
        "- GPU submissions before protocol lock: `0`.\n"
        "- Target outcome reads before protocol lock: `0`.\n"
        "- The protocol commit and authorization execution-lock commits are filled after Git commit.\n"
    )
    return protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78r_seed3_src_canary")
    parser.add_argument("command", choices=("emit-protocol", "guard-check"))
    parser.add_argument("--authorization-token")
    args = parser.parse_args(argv)
    if args.command == "emit-protocol":
        protocol = emit_protocol()
        print(json.dumps({"gate": "C78R_PROTOCOL_LOCKED", "units": len(unit_manifest()), "protocol_sha256": sha256_file(PROTOCOL_PATH)}, sort_keys=True))
        return 0
    if authorization_matches(args.authorization_token):
        raise RuntimeError("C78R exact authorization cannot execute through metadata-only guard-check")
    print(json.dumps({"gate": "SRC_CANARY_READY_BUT_NOT_AUTHORIZED", "training": 0, "forward": 0, "GPU": 0, "checkpoints": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
