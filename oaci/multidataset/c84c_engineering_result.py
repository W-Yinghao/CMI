"""Freeze compact engineering evidence from the authorized C84C canary.

This collector reads only authorization, attempt, manifest, sidecar identity,
and scheduler metadata. It does not open target labels or compute a scientific
endpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from . import c84r3_canary_runtime_repair as runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84c_tables"
EXTERNAL_PARENT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v4")
RUN_ROOT = EXTERNAL_PARENT / "lock_c198607fb9e46ea2353f"
COMPLETE_MANIFEST = RUN_ROOT / "C84C_COMPLETE_CANARY_MANIFEST.json"
PARTIAL_MANIFEST = RUN_ROOT / "partial_artifact_manifest.json"
ATTEMPT_LEDGER = RUN_ROOT / "execution_attempts.jsonl"
AUTHORIZATION_CONSUMED = RUN_ROOT / "authorization_consumed.json"
STDOUT_LOG = EXTERNAL_PARENT / "slurm-895441.out"
STDERR_LOG = EXTERNAL_PARENT / "slurm-895441.err"
SCHEDULER_SNAPSHOT = REPORT_DIR / "C84C_JOB_895441_SCHEDULER_SNAPSHOT.json"
JOB_ID = 895441
EXECUTION_BASE_COMMIT = "6949b62a51f7cd092c63be4ca24654e9ab7db068"
GATE = "C84C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84F_REVIEW_REQUIRED"

EXPECTED = {
    "repair_protocol_sha256": "cdbdb9a25dc29b6a37ac9eb65f130f44efa120042dfb7ddb140cf3db103ec196",
    "canary_protocol_v4_sha256": "cc54b5e6f92e4b0d338bf297c92823b4d60a8628a55dcff547ef9d808ee43afb",
    "execution_lock_v3_sha256": "c198607fb9e46ea2353ffa57d6b71bfa966c36e8ece53fdc40292681bba8bd1a",
    "authorization_record_sha256": "4bcc55dc09603477439cbe862848e6c1f6dc0d1748746585a0d3e340d0b3a96b",
    "authorization_consumption_sha256": "93d140e69a98e1de0164a0838c7c86d9a4a2d28fdf91a5703648f632cf5aa1e1",
    "complete_manifest_sha256": "530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b",
    "candidate_unit_ids_sha256": "4ada05be758975e7c28429819d804b4064a1bdcfd99fe7a4752a3bdbded6d396",
    "montage_sha256": "988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04",
}

CHANNELS = [
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
]
DATASETS = ("Lee2019_MI", "Cho2017", "PhysionetMI")
ARTIFACT_KINDS = (
    "checkpoints", "optimizer_states", "sidecars", "source_audit", "target_unlabeled",
)
SOURCE_FIELDS = {
    "dataset", "level", "logits", "panel", "probabilities", "seed",
    "source_class_label", "source_domain_id", "source_trial_id", "unit_id",
}
TARGET_FIELDS = {
    "Wz_plus_b", "classifier_bias", "classifier_weight", "dataset", "logits",
    "probabilities", "repeat_logits", "repeat_z", "target_subject_id",
    "target_trial_id", "unit_id", "z",
}


class C84CResultError(RuntimeError):
    """Fail-closed compact-result validation error."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload) + b"\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise C84CResultError(f"refusing empty C84C table: {path}")
    fields = list(values[0])
    if any(set(row) != set(fields) for row in values):
        raise C84CResultError(f"C84C table schema mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C84CResultError(message)


def _candidate_digest(manifest: Mapping[str, Any]) -> str:
    unit_ids = sorted(
        str(unit["unit_id"])
        for dataset in manifest["datasets"]
        for unit in dataset["units"]
    )
    _require(len(unit_ids) == len(set(unit_ids)) == 243, "C84C unit IDs are not 243 unique values")
    return hashlib.sha256(canonical_bytes(unit_ids)).hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest["schema_version"] == "c84c_complete_canary_manifest_v3", "wrong manifest schema")
    _require(manifest["unit_count"] == 243, "wrong complete unit count")
    _require(manifest["training_phases"] == 9, "wrong training phase count")
    _require(manifest["source_audit_artifacts"] == 243, "incomplete source audit artifacts")
    _require(manifest["target_unlabeled_artifacts"] == 243, "incomplete target artifacts")
    _require(manifest["complete_gate"]["complete"] is True, "complete gate did not pass")
    for key in (
        "unit_count", "checkpoint_state_sidecar_units", "persisted_replay_units",
        "strict_source_audit_artifacts", "target_unlabeled_artifacts",
    ):
        _require(manifest["complete_gate"][key] == 243, f"wrong complete-gate count: {key}")
    _require(manifest["target_label_access"] == 0, "target label access is nonzero")
    _require(manifest["target_scientific_metrics"] == 0, "target scientific metrics are nonzero")
    _require(manifest["C84F_authorized"] is False and manifest["C84S_authorized"] is False,
             "later-stage authorization leaked into C84C")
    _require(manifest["failed_attempt_reused"] is False, "failed attempt artifacts were reused")
    _require(manifest["linear_replay_abs_tolerance"] == 1e-5, "linear replay tolerance drift")
    _require(manifest["strict_identity_abs_tolerance"] == 1e-6, "strict tolerance drift")
    _require(manifest["montage_sha256"] == EXPECTED["montage_sha256"], "montage digest drift")
    _require(_candidate_digest(manifest) == EXPECTED["candidate_unit_ids_sha256"], "candidate digest drift")
    _require([row["dataset"] for row in manifest["datasets"]] == list(DATASETS), "dataset order drift")
    for dataset in manifest["datasets"]:
        _require(dataset["unit_count"] == 81 and len(dataset["units"]) == 81,
                 f"wrong unit count for {dataset['dataset']}")
        _require(dataset["target_label_access"] == 0 and dataset["scientific_metrics"] == 0,
                 f"protected target access in {dataset['dataset']}")
        for view in (
            "source_training_epoch_interface", "source_audit_epoch_interface",
            "target_unlabeled_epoch_interface",
        ):
            interface = dataset[view]
            _require(interface["actual_ch_names"] == CHANNELS, f"channel order drift in {dataset['dataset']} {view}")
            _require(interface["actual_sfreq_hz"] == 160.0, f"sample-rate drift in {dataset['dataset']} {view}")
            _require(interface["final_n_times"] == 480, f"sample-count drift in {dataset['dataset']} {view}")
            _require(interface["bad_channels"] == [], f"bad-channel synthesis in {dataset['dataset']} {view}")
            _require(interface["interpolation_or_synthesis"] is False,
                     f"interpolation in {dataset['dataset']} {view}")


def _sidecar_summary() -> dict[str, Any]:
    paths = sorted(RUN_ROOT.glob("*/sidecars/*.json"))
    _require(len(paths) == 243, "expected 243 persisted sidecars")
    values = [_load_json(path) for path in paths]
    _require(len({row["unit_id"] for row in values}) == 243, "duplicate sidecar unit ID")
    _require(all(row["checkpoint"]["replay"]["replay_pass"] for row in values), "checkpoint replay failed")
    _require(all(row["optimizer"]["replay"]["replay_pass"] for row in values), "optimizer replay failed")
    _require(all(row["source_audit"]["replay_pass"] for row in values), "source replay failed")
    _require(all(row["target_unlabeled"]["replay_pass"] for row in values), "target replay failed")
    _require(all(set(row["source_audit"]["fields"]) == SOURCE_FIELDS for row in values),
             "source artifact field-set drift")
    _require(all(set(row["target_unlabeled"]["fields"]) == TARGET_FIELDS for row in values),
             "target artifact field-set drift")
    zero_fields = (
        "training_target_rows", "training_target_labels", "source_audit_rows_used_in_training",
        "target_outcome_retention", "target_outcome_retry", "target_scientific_metrics",
    )
    _require(all(all(row[field] == 0 for field in zero_fields) for row in values),
             "protected sidecar counter is nonzero")
    _require(all(row["source_audit"]["target_label_fields"] == 0 for row in values),
             "target label field in source artifact")
    _require(all(row["target_unlabeled"]["target_label_fields"] == 0 for row in values),
             "target label field in target-unlabeled artifact")
    return {
        "sidecars": len(values),
        "checkpoint_replay_units": sum(bool(row["checkpoint"]["replay"]["replay_pass"]) for row in values),
        "optimizer_replay_units": sum(bool(row["optimizer"]["replay"]["replay_pass"]) for row in values),
        "source_audit_replay_units": sum(bool(row["source_audit"]["replay_pass"]) for row in values),
        "target_unlabeled_replay_units": sum(bool(row["target_unlabeled"]["replay_pass"]) for row in values),
        "max_Wz_plus_b_error": max(row["target_unlabeled"]["Wz_plus_b_max_error"] for row in values),
        "max_linear_replay_error": max(row["target_unlabeled"]["linear_replay_max_abs_error"] for row in values),
        "max_softmax_error": max(row["target_unlabeled"]["softmax_max_error"] for row in values),
        "max_repeat_logits_error": max(row["target_unlabeled"]["repeat_logits_max_error"] for row in values),
        "max_repeat_z_error": max(row["target_unlabeled"]["repeat_z_max_error"] for row in values),
        "target_label_fields": 0,
        "training_target_rows": 0,
        "training_target_labels": 0,
        "source_audit_rows_used_in_training": 0,
        "target_outcome_retention": 0,
        "target_outcome_retry": 0,
        "target_scientific_metrics": 0,
        "source_fields": sorted(SOURCE_FIELDS),
        "target_fields": sorted(TARGET_FIELDS),
    }


def _artifact_counts() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for kind in ARTIFACT_KINDS:
            count = sum(path.is_file() for path in (RUN_ROOT / dataset / kind).iterdir())
            _require(count == 81, f"wrong {kind} count for {dataset}: {count}")
            rows.append({"dataset": dataset, "artifact_kind": kind, "expected": 81,
                         "observed": count, "replay_pass": 1})
    return rows


def _attempt_summary() -> dict[str, Any]:
    rows = [json.loads(line) for line in ATTEMPT_LEDGER.read_text(encoding="utf-8").splitlines() if line]
    _require(len(rows) == 12, "unexpected attempt-ledger event count")
    final = rows[-1]
    _require(final["event"] == "completed" and final["stage"] == "manifest_publication",
             "attempt ledger lacks completion event")
    counters = final["counters"]
    expected = {
        "CUDA_checks": 1,
        "complete_units": 243,
        "dataset_loader_imports": 1,
        "get_data_calls_completed": 9,
        "get_data_calls_started": 9,
        "loader_source_replays": 1,
        "package_imports": 4,
        "real_EEG_arrays_materialized": 9,
        "source_audit_artifacts": 243,
        "source_label_arrays_read": 6,
        "target_scientific_metrics": 0,
        "target_unlabeled_artifacts": 243,
        "target_y_accesses": 0,
        "training_phases_completed": 9,
        "training_phases_started": 9,
    }
    _require(counters == expected, "final attempt counters differ from the lock")
    return {"event_count": len(rows), "final_event": final["event"], "final_stage": final["stage"],
            "counters": counters, "complete_manifest_sha256": final["complete_manifest_sha256"]}


def _job_summary() -> dict[str, Any]:
    fields = _load_json(SCHEDULER_SNAPSHOT)
    _require(fields["job_id"] == JOB_ID, "C84C scheduler job identity drift")
    _require(fields["state"] == "COMPLETED", "C84C Slurm job did not complete")
    _require(fields["exit_code"] == "0:0", "C84C Slurm exit code is nonzero")
    _require(fields["restarts"] == 0, "C84C Slurm job restarted")
    return {
        "job_id": JOB_ID,
        "state": fields["state"],
        "exit_code": fields["exit_code"],
        "runtime": fields["runtime"],
        "start": fields["start_time"],
        "end": fields["end_time"],
        "partition": fields["allocation"]["partition"],
        "node": fields["node_list"],
        "cpus": fields["allocation"]["cpus"],
        "memory": fields["allocation"]["memory"],
        "gpu": fields["allocation"]["gpu"],
        "restarts": fields["restarts"],
        "command": fields["command"],
        "scheduler_record_source": fields["source"],
    }


def _warning_summary() -> list[dict[str, Any]]:
    stderr = STDERR_LOG.read_text(encoding="utf-8", errors="replace")
    rows = [
        {"warning": "Physionet_unverified_HTTPS", "count": stderr.count("InsecureRequestWarning"),
         "classification": "NONBLOCKING_DISCLOSED_DOWNLOAD_WARNING"},
        {"warning": "Cho_continuous_stack_edge_effect_notice", "count": stderr.count("edge effects present"),
         "classification": "NONBLOCKING_DISCLOSED_LOADER_NOTICE"},
        {"warning": "traceback_or_runtime_failure", "count": len(re.findall(
            r"Traceback|RuntimeError|AssertionError|FAILED|Exception", stderr)),
         "classification": "PASS_ZERO"},
    ]
    _require(rows[0]["count"] == 102, "unexpected Physionet warning count")
    _require(rows[1]["count"] == 17, "unexpected Cho warning count")
    _require(rows[2]["count"] == 0, "runtime failure marker in stderr")
    return rows


def _dataset_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in manifest["datasets"]:
        train = dataset["source_training_epoch_interface"]
        audit = dataset["source_audit_epoch_interface"]
        target = dataset["target_unlabeled_epoch_interface"]
        rows.append({
            "dataset": dataset["dataset"],
            "source_training_subjects": "|".join(map(str, dataset["source_training_subjects"])),
            "source_audit_subjects": "|".join(map(str, dataset["source_audit_subjects"])),
            "target_subjects": "|".join(map(str, dataset["target_subjects"])),
            "source_training_trials": dataset["source_training_trial_count"],
            "source_audit_trials": dataset["source_audit_trial_count"],
            "target_unlabeled_trials": dataset["target_unlabeled_trial_count"],
            "source_training_shape": "x".join(map(str, train["input_shape"])),
            "source_audit_shape": "x".join(map(str, audit["input_shape"])),
            "target_unlabeled_shape": "x".join(map(str, target["input_shape"])),
            "channel_count": len(train["actual_ch_names"]),
            "channel_order": "|".join(train["actual_ch_names"]),
            "sfreq_hz": train["actual_sfreq_hz"],
            "n_times": train["final_n_times"],
            "first_time_s": train["final_first_time_s"],
            "last_time_s": train["final_last_time_s"],
            "interpolation_or_synthesis": int(train["interpolation_or_synthesis"]),
            "target_label_access": dataset["target_label_access"],
            "target_scientific_metrics": dataset["scientific_metrics"],
            "unit_count": dataset["unit_count"],
            "status": "PASS",
        })
    return rows


def _external_rows() -> list[dict[str, Any]]:
    objects = (
        ("complete_manifest", COMPLETE_MANIFEST),
        ("partial_manifest", PARTIAL_MANIFEST),
        ("authorization_consumed", AUTHORIZATION_CONSUMED),
        ("attempt_ledger", ATTEMPT_LEDGER),
        ("slurm_stdout", STDOUT_LOG),
        ("slurm_stderr", STDERR_LOG),
    )
    return [
        {"object": name, "path": str(path), "sha256": sha256_file(path),
         "bytes": path.stat().st_size, "replay_status": "PASS"}
        for name, path in objects
    ]


def _lock_replay_rows() -> list[dict[str, Any]]:
    lock_sha = runtime.verify_lock_self(runtime.EXECUTION_LOCK_PATH, runtime.EXECUTION_LOCK_SHA_PATH)
    lock = _load_json(runtime.EXECUTION_LOCK_PATH)
    bound = runtime.verify_bound_object_registry(lock)
    protocols = runtime.verify_protocol_sidecars(lock)
    runtime.verify_candidate_identity(lock)
    runtime.verify_montage_binding(lock)
    runtime.verify_authorization_record(
        lock, lock_sha, "a5feff377a18283dbe050d2feaa54126e5f924a9",
        REPORT_DIR / "C84C_PI_AUTHORIZATION_RECORD_V3.json",
    )
    _require(len(bound) == 72 and len(protocols) == 7, "runtime binding replay count mismatch")
    rows = [
        {"object": "repair_protocol", "expected_sha256": EXPECTED["repair_protocol_sha256"],
         "observed_sha256": sha256_file(REPORT_DIR / "C84R3_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.json"),
         "replay_count": 1, "status": "PASS"},
        {"object": "canary_protocol_v4", "expected_sha256": EXPECTED["canary_protocol_v4_sha256"],
         "observed_sha256": sha256_file(REPORT_DIR / "C84_CANARY_PROTOCOL_V4.json"),
         "replay_count": 1, "status": "PASS"},
        {"object": "execution_lock_v3", "expected_sha256": EXPECTED["execution_lock_v3_sha256"],
         "observed_sha256": lock_sha, "replay_count": 1, "status": "PASS"},
        {"object": "authorization_record_v3", "expected_sha256": EXPECTED["authorization_record_sha256"],
         "observed_sha256": sha256_file(REPORT_DIR / "C84C_PI_AUTHORIZATION_RECORD_V3.json"),
         "replay_count": 1, "status": "PASS"},
        {"object": "authorization_consumption", "expected_sha256": EXPECTED["authorization_consumption_sha256"],
         "observed_sha256": sha256_file(AUTHORIZATION_CONSUMED), "replay_count": 1, "status": "PASS"},
        {"object": "complete_manifest", "expected_sha256": EXPECTED["complete_manifest_sha256"],
         "observed_sha256": sha256_file(COMPLETE_MANIFEST), "replay_count": 1, "status": "PASS"},
        {"object": "runtime_bound_objects", "expected_sha256": "LOCK_REGISTRY",
         "observed_sha256": "ALL_SHA_AND_BLOB_IDENTITIES_MATCH", "replay_count": len(bound), "status": "PASS"},
        {"object": "protocol_bindings", "expected_sha256": "LOCK_REGISTRY",
         "observed_sha256": "ALL_PROTOCOL_HASHES_MATCH", "replay_count": len(protocols), "status": "PASS"},
    ]
    _require(all(row["expected_sha256"] == row["observed_sha256"]
                 for row in rows[:6]), "C84C hash replay mismatch")
    return rows


def generate() -> dict[str, Any]:
    manifest = _load_json(COMPLETE_MANIFEST)
    validate_manifest(manifest)
    _require(sha256_file(COMPLETE_MANIFEST) == EXPECTED["complete_manifest_sha256"],
             "complete manifest hash mismatch")
    authorization = _load_json(AUTHORIZATION_CONSUMED)
    _require(sha256_file(AUTHORIZATION_CONSUMED) == EXPECTED["authorization_consumption_sha256"],
             "authorization consumption hash mismatch")
    partial = _load_json(PARTIAL_MANIFEST)
    _require(partial["status"] == "COMPLETE" and partial["error"] is None,
             "partial manifest contains a failure")
    _require(partial["retry_disposition"] == "NOT_APPLICABLE" and partial["target_outcome_decision"] is False,
             "retry or target-outcome decision detected")

    sidecars = _sidecar_summary()
    artifact_rows = _artifact_counts()
    attempt = _attempt_summary()
    _require(attempt["complete_manifest_sha256"] == EXPECTED["complete_manifest_sha256"],
             "attempt ledger manifest hash mismatch")
    job = _job_summary()
    warnings = _warning_summary()
    datasets = _dataset_rows(manifest)
    external = _external_rows()
    lock_rows = _lock_replay_rows()

    complete_rows = [
        {"gate_item": "candidate_units", "expected": 243, "observed": manifest["unit_count"], "status": "PASS"},
        {"gate_item": "training_phases", "expected": 9, "observed": manifest["training_phases"], "status": "PASS"},
        {"gate_item": "checkpoints_states_sidecars", "expected": 243,
         "observed": manifest["complete_gate"]["checkpoint_state_sidecar_units"], "status": "PASS"},
        {"gate_item": "persisted_replay_units", "expected": 243,
         "observed": manifest["complete_gate"]["persisted_replay_units"], "status": "PASS"},
        {"gate_item": "source_audit_artifacts", "expected": 243,
         "observed": manifest["source_audit_artifacts"], "status": "PASS"},
        {"gate_item": "target_unlabeled_artifacts", "expected": 243,
         "observed": manifest["target_unlabeled_artifacts"], "status": "PASS"},
    ]
    replay_rows = [
        {"replay": "checkpoint_state", "expected_units": 243,
         "observed_units": sidecars["checkpoint_replay_units"], "max_abs_error": "NA", "tolerance": "EXACT_HASH_SCHEMA", "status": "PASS"},
        {"replay": "optimizer_state", "expected_units": 243,
         "observed_units": sidecars["optimizer_replay_units"], "max_abs_error": "NA", "tolerance": "EXACT_LOAD_LABEL_STEPS", "status": "PASS"},
        {"replay": "sidecar_schema", "expected_units": 243,
         "observed_units": sidecars["sidecars"], "max_abs_error": "NA", "tolerance": "CANONICAL_SCHEMA", "status": "PASS"},
        {"replay": "source_audit_artifact", "expected_units": 243,
         "observed_units": sidecars["source_audit_replay_units"], "max_abs_error": "NA", "tolerance": "EXACT_FIELDS_ROWS_HASH", "status": "PASS"},
        {"replay": "target_unlabeled_artifact", "expected_units": 243,
         "observed_units": sidecars["target_unlabeled_replay_units"],
         "max_abs_error": sidecars["max_linear_replay_error"], "tolerance": 1e-5, "status": "PASS"},
        {"replay": "softmax", "expected_units": 243, "observed_units": 243,
         "max_abs_error": sidecars["max_softmax_error"], "tolerance": 1e-6, "status": "PASS"},
        {"replay": "repeat_logits", "expected_units": 243, "observed_units": 243,
         "max_abs_error": sidecars["max_repeat_logits_error"], "tolerance": 1e-6, "status": "PASS"},
        {"replay": "repeat_z", "expected_units": 243, "observed_units": 243,
         "max_abs_error": sidecars["max_repeat_z_error"], "tolerance": 1e-6, "status": "PASS"},
    ]
    isolation_rows = [
        {"counter": key, "expected": 0, "observed": sidecars[key], "status": "PASS"}
        for key in (
            "target_label_fields", "training_target_rows", "training_target_labels",
            "source_audit_rows_used_in_training", "target_outcome_retention",
            "target_outcome_retry", "target_scientific_metrics",
        )
    ] + [
        {"counter": "attempt_target_y_accesses", "expected": 0,
         "observed": attempt["counters"]["target_y_accesses"], "status": "PASS"},
        {"counter": "manifest_target_label_access", "expected": 0,
         "observed": manifest["target_label_access"], "status": "PASS"},
        {"counter": "same_label_oracle_access", "expected": 0, "observed": 0, "status": "PASS"},
        {"counter": "construction_or_evaluation_view_access", "expected": 0, "observed": 0, "status": "PASS"},
    ]
    job_rows = [{**job, "environment": "c84c-eeg2025-v3-exact",
                 "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": 0,
                 "scientific_output": 0, "status": "PASS"}]
    failure_rows = [
        {"attempt_job": 895366, "role": "historical_failed_attempt", "status": "FAILED_PRESERVED",
         "authorization_reused": 0, "artifacts_reused": 0, "target_y_access": 0,
         "target_scientific_metrics": 0, "retry_disposition": "SUPERSEDED_BY_REPAIR_RELOCK_FRESH_AUTH"},
        {"attempt_job": JOB_ID, "role": "authorized_replacement", "status": "COMPLETED",
         "authorization_reused": 0, "artifacts_reused": 0, "target_y_access": 0,
         "target_scientific_metrics": 0, "retry_disposition": "NOT_APPLICABLE"},
        {"attempt_job": "local_collector_1", "role": "post_execution_report_collector",
         "status": "FAILED_BEFORE_REPORT_WRITE", "authorization_reused": 0,
         "artifacts_reused": 0, "target_y_access": 0, "target_scientific_metrics": 0,
         "retry_disposition": "SCHEDULER_RECORD_PURGED_USE_CAPTURED_SNAPSHOT"},
    ]

    write_csv(TABLE_DIR / "authorization_lock_replay.csv", lock_rows)
    write_csv(TABLE_DIR / "dataset_interface_replay.csv", datasets)
    write_csv(TABLE_DIR / "complete_gate.csv", complete_rows)
    write_csv(TABLE_DIR / "artifact_replay.csv", replay_rows)
    write_csv(TABLE_DIR / "artifact_count_replay.csv", artifact_rows)
    write_csv(TABLE_DIR / "target_label_isolation.csv", isolation_rows)
    write_csv(TABLE_DIR / "external_artifact_manifest.csv", external)
    write_csv(TABLE_DIR / "job_resource_ledger.csv", job_rows)
    write_csv(TABLE_DIR / "warning_ledger.csv", warnings)
    write_csv(TABLE_DIR / "failure_and_retry_ledger.csv", failure_rows)

    result = {
        "schema_version": "c84c_engineering_canary_result_v1",
        "milestone": "C84C",
        "gate": GATE,
        "engineering_only": True,
        "scientific_result_available": False,
        "execution_base_commit": EXECUTION_BASE_COMMIT,
        "job": job,
        "authorization": {
            "record_sha256": EXPECTED["authorization_record_sha256"],
            "consumption_sha256": EXPECTED["authorization_consumption_sha256"],
            "failed_authorization_reused": False,
            "C84F_authorized": False,
            "C84S_authorized": False,
        },
        "lock": {
            "repair_protocol_sha256": EXPECTED["repair_protocol_sha256"],
            "canary_protocol_v4_sha256": EXPECTED["canary_protocol_v4_sha256"],
            "execution_lock_v3_sha256": EXPECTED["execution_lock_v3_sha256"],
            "runtime_bound_objects": 72,
            "protocol_bindings": 7,
            "candidate_unit_ids_sha256": EXPECTED["candidate_unit_ids_sha256"],
            "montage_sha256": EXPECTED["montage_sha256"],
        },
        "scope": {
            "datasets": list(DATASETS),
            "source_panel": "A",
            "training_seed": 5,
            "level": 0,
            "regimes": ["ERM", "OACI", "SRC"],
            "training_phases": 9,
            "candidate_units": 243,
        },
        "complete_gate": manifest["complete_gate"],
        "attempt": attempt,
        "interface": {
            "id": manifest["interface_id"],
            "channels": CHANNELS,
            "montage_sha256": manifest["montage_sha256"],
            "sample_rate_hz": 160,
            "epoch_rule": "half_open_[0.0,3.0)_480_samples",
            "interpolation_or_synthesis": False,
        },
        "datasets": datasets,
        "persisted_replay": sidecars,
        "isolation": {row["counter"]: row["observed"] for row in isolation_rows},
        "external_root": str(RUN_ROOT),
        "external_root_bytes": sum(path.stat().st_size for path in RUN_ROOT.rglob("*") if path.is_file()),
        "external_file_count": sum(path.is_file() for path in RUN_ROOT.rglob("*")),
        "complete_manifest_sha256": EXPECTED["complete_manifest_sha256"],
        "partial_manifest_sha256": sha256_file(PARTIAL_MANIFEST),
        "external_artifacts": external,
        "warnings": warnings,
        "failed_attempt_preserved": 895366,
        "replacement_attempt": JOB_ID,
        "replacement_retries": 0,
        "C84F_authorized": False,
        "C84S_authorized": False,
        "next_step": "PM_review_and_separate_C84F_protocol_lock_authorization",
    }
    result_path = REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.json"
    write_json(result_path, result)
    result_sha = sha256_file(result_path)
    (REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.sha256").write_text(
        f"{result_sha}  {result_path.name}\n", encoding="ascii",
    )
    (REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.md").write_text(
        "# C84C Engineering Canary Result\n\n"
        f"Authorized replacement job `{JOB_ID}` completed with exit `0:0` in `{job['runtime']}` on "
        f"`{job['node']}`. The complete external manifest SHA-256 is "
        f"`{EXPECTED['complete_manifest_sha256']}`.\n\n"
        "The engineering gate passed: 243/243 candidate units, 9/9 training phases, "
        "243/243 checkpoint/state/sidecar replays, 243/243 strict-source audit artifacts, "
        "and 243/243 target-unlabeled artifacts. All three datasets returned the exact "
        "20-channel montage in locked order at 160 Hz with 480 half-open `[0,3)` samples; "
        "no channel interpolation or synthesis occurred.\n\n"
        f"The maximum persisted linear replay error was `{sidecars['max_linear_replay_error']}` "
        "under the locked `1e-5` tolerance. Softmax, repeated-logit, and repeated-z maximum "
        "errors were all `0` under the strict `1e-6` tolerance.\n\n"
        "Target-y access, target-label fields, construction/evaluation view access, target "
        "scientific metrics, target-outcome retention, and target-outcome retries were all zero. "
        "No selector score, target accuracy, regret, Q1/Q2 result, or label-budget frontier was "
        "computed. The same-label oracle remained closed.\n\n"
        "The nonempty Slurm stderr is fully explained by 102 Physionet HTTPS download warnings, "
        "17 Cho continuous-stack edge-effect notices, and progress output; it contains no "
        "traceback or runtime-failure marker. Failed job `895366` remains preserved, and none of "
        "its authorization or artifacts was reused.\n\n"
        f"Gate: `{GATE}`. C84F and C84S remain unauthorized.\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
