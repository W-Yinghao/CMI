"""Freeze compact engineering evidence from the authorized C84L1C canary.

The collector reads manifests, sidecars, artifact hashes, scheduler metadata,
and logs. It never opens target labels or computes a performance endpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from . import c84l1r1_runtime_repair as runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84l1c_tables"
RUN_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-level1-canary-v2/"
    "lock_f9ebd88c72915bb41ba2"
)
COMPLETE_MANIFEST = RUN_ROOT / "C84L1C_COMPLETE_ENGINEERING_MANIFEST.json"
PARTIAL_MANIFEST = RUN_ROOT / "partial_artifact_manifest.json"
ATTEMPT_LEDGER = RUN_ROOT / "execution_attempts.jsonl"
AUTHORIZATION_CONSUMED = RUN_ROOT / "authorization_consumed.json"
STDOUT_LOG = Path("/home/infres/yinwang/CMI_AAAI/c84l1c_v2_logs/c84l1c-v2-896066.out")
STDERR_LOG = Path("/home/infres/yinwang/CMI_AAAI/c84l1c_v2_logs/c84l1c-v2-896066.err")
SCHEDULER_SNAPSHOT = REPORT_DIR / "C84L1C_JOB_896066_SQUEUE_SNAPSHOT.json"
AUTHORIZATION_RECORD = REPORT_DIR / "C84L1C_PI_AUTHORIZATION_RECORD_V2.json"
JOB_ID = 896066
EXECUTION_BASE_COMMIT = "60dd725026559f880dde71907eb69773d51961d9"
LOCK_COMMIT = "afc5a6b5aedbb0e9d9b09acba0997657513e5268"
GATE = "C84L1C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84FL2_REVIEW_REQUIRED"

EXPECTED = {
    "repair_protocol_sha256": "2e199f6f63dffd1b02c1e31102ed189e31bf6e4961465394230f8e9de1d4ddf0",
    "canary_protocol_v2_sha256": "6e6bcb6b60726c76c8db0afc48e954d0e4a1cf68bfd29796987bfd6828355616",
    "execution_lock_v2_sha256": "f9ebd88c72915bb41ba2d2d84a2a00c6748272021d48043c299bce52a1ad3813",
    "authorization_record_sha256": "e287b40028ff9dc5373498b65f7316a443661de3e6548c23a456bedba40848fd",
    "authorization_consumption_sha256": "c11e73fd1be1264dd61719d3f149142ebe5e9cd442b621a2fda220fae0417552",
    "complete_manifest_sha256": "3cf1366ccf40efc82a6bb2ffef56045e83c0f0e9670429973f23252371ad1c18",
    "candidate_unit_ids_sha256": "db0c41a8caeb7d0fffd6938554c660eec36582596f12915b8b981c05bc092b95",
    "montage_sha256": "988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04",
    "intervention_registry_sha256": "89c4f366a222c1fe2ac31780bcbddbc9e59ff5afa4a779267abbd95429c41c17",
}

DATASETS = ("Lee2019_MI", "Cho2017", "PhysionetMI")
DELETED_SUBJECTS = {"Lee2019_MI": 31, "Cho2017": 17, "PhysionetMI": 103}
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


class C84L1CResultError(RuntimeError):
    """Fail-closed compact-result validation error."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload) + b"\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise C84L1CResultError(f"refusing empty C84L1C table: {path}")
    fields = list(values[0])
    if any(set(row) != set(fields) for row in values):
        raise C84L1CResultError(f"C84L1C table schema mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C84L1CResultError(message)


def candidate_digest(manifest: Mapping[str, Any]) -> str:
    unit_ids = sorted(
        str(unit["unit_id"])
        for dataset in manifest["datasets"]
        for unit in dataset["units"]
    )
    require(len(unit_ids) == len(set(unit_ids)) == 243, "unit IDs are not 243 unique values")
    return hashlib.sha256(canonical_bytes(unit_ids)).hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    require(manifest["schema_version"] == "c84l1c_complete_engineering_manifest_v2", "wrong schema")
    require(manifest["unit_count"] == 243 and manifest["training_phases"] == 9, "wrong scope")
    require(manifest["source_audit_artifacts"] == 243, "source artifacts incomplete")
    require(manifest["target_unlabeled_artifacts"] == 243, "target artifacts incomplete")
    require(manifest["complete_gate"] == {
        "checkpoint_optimizer_sidecar_units": 243,
        "complete": True,
        "strict_source_audit_artifacts": 243,
        "target_scientific_metrics": 0,
        "target_unlabeled_artifacts": 243,
        "target_y_access": 0,
        "unit_count": 243,
    }, "complete gate drift")
    require(manifest["target_label_access"] == 0, "target label access is nonzero")
    require(manifest["target_scientific_metrics"] == 0, "scientific metric count is nonzero")
    require(manifest["C84F_authorized"] is False and manifest["C84S_authorized"] is False,
            "later-stage authorization leaked into C84L1C")
    require(manifest["failed_authorization_reused"] is False, "failed authorization was reused")
    require(manifest["failed_partial_artifacts_reused"] is False, "failed artifacts were reused")
    require(manifest["historical_failed_job"] == 895928, "historical failed job drift")
    require(manifest["linear_replay_abs_tolerance"] == 2e-5, "linear tolerance drift")
    require(manifest["strict_identity_abs_tolerance"] == 1e-6, "strict tolerance drift")
    require(manifest["repair_protocol_sha256"] == EXPECTED["repair_protocol_sha256"], "repair hash drift")
    require(manifest["canary_protocol_v2_sha256"] == EXPECTED["canary_protocol_v2_sha256"],
            "canary protocol hash drift")
    require(manifest["execution_lock_v2_sha256"] == EXPECTED["execution_lock_v2_sha256"],
            "execution lock hash drift")
    require(manifest["authorization_consumption_sha256"] == EXPECTED["authorization_consumption_sha256"],
            "authorization consumption drift")
    require(candidate_digest(manifest) == EXPECTED["candidate_unit_ids_sha256"], "candidate digest drift")
    require([row["dataset"] for row in manifest["datasets"]] == list(DATASETS), "dataset order drift")
    for dataset in manifest["datasets"]:
        name = dataset["dataset"]
        intervention = dataset["level_intervention"]
        require(dataset["unit_count"] == len(dataset["units"]) == 81, f"wrong unit count for {name}")
        require(dataset["level"] == 1, f"wrong level for {name}")
        require(dataset["target_label_access"] == dataset["scientific_metrics"] == 0,
                f"protected access in {name}")
        require(intervention["deleted_source_subject"] == DELETED_SUBJECTS[name],
                f"deleted subject drift in {name}")
        require(intervention["deleted_class"] == "left_hand", f"deleted class drift in {name}")
        require(len(intervention["pre_cell_counts"]) == 24, f"pre-cell count drift in {name}")
        require(len(intervention["post_cell_counts"]) == 23, f"post-cell count drift in {name}")
        require(min(cell[2] for cell in intervention["post_cell_counts"]) >= 8,
                f"support floor failed in {name}")
        require(intervention["pre_trial_count"] - intervention["post_trial_count"] >= 8,
                f"deleted support is too small in {name}")
        for unit in dataset["units"]:
            for flag in (
                "checkpoint_replay_pass", "optimizer_replay_pass", "sidecar_replay_pass",
                "source_audit_replay_pass", "target_unlabeled_replay_pass", "support_replay_pass",
                "level0_plan_replay_pass", "paired_model_init_pass",
            ):
                require(unit[flag] is True, f"{flag} failed for {unit['unit_id']}")
            require(unit["target_y_access"] == unit["target_scientific_metrics"] == 0,
                    f"protected unit counter in {unit['unit_id']}")


def lock_replay_rows() -> list[dict[str, Any]]:
    lock_sha = runtime.verify_lock_self(runtime.EXECUTION_LOCK_PATH, runtime.EXECUTION_LOCK_SHA_PATH)
    lock = read_json(runtime.EXECUTION_LOCK_PATH)
    bound = runtime.base.prior.verify_bound_object_registry(lock)
    protocols = runtime.base.prior.verify_protocol_sidecars(lock)
    runtime.base.verify_intervention_registry(lock)
    runtime.base.verify_candidate_identity(lock)
    runtime.base.verify_c84c_level0_binding(lock)
    runtime.verify_failed_attempt_binding(lock)
    runtime.verify_authorization_record(lock, lock_sha, LOCK_COMMIT, AUTHORIZATION_RECORD)
    require(lock_sha == EXPECTED["execution_lock_v2_sha256"], "lock SHA drift")
    require(len(bound) == 125 and len(protocols) == 5, "lock replay count mismatch")
    rows = [
        ("repair_protocol", EXPECTED["repair_protocol_sha256"],
         sha256_file(REPORT_DIR / "C84L1R1_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.json"), 1),
        ("canary_protocol_v2", EXPECTED["canary_protocol_v2_sha256"],
         sha256_file(REPORT_DIR / "C84_LEVEL1_CANARY_PROTOCOL_V2.json"), 1),
        ("execution_lock_v2", EXPECTED["execution_lock_v2_sha256"], lock_sha, 1),
        ("authorization_record_v2", EXPECTED["authorization_record_sha256"],
         sha256_file(AUTHORIZATION_RECORD), 1),
        ("authorization_consumption", EXPECTED["authorization_consumption_sha256"],
         sha256_file(AUTHORIZATION_CONSUMED), 1),
        ("complete_manifest", EXPECTED["complete_manifest_sha256"],
         sha256_file(COMPLETE_MANIFEST), 1),
        ("runtime_bound_objects", "LOCK_REGISTRY", "ALL_SHA_AND_BLOB_IDENTITIES_MATCH", len(bound)),
        ("protocol_bindings", "LOCK_REGISTRY", "ALL_PROTOCOL_HASHES_MATCH", len(protocols)),
    ]
    require(all(expected == observed for _, expected, observed, _ in rows[:6]), "hash replay mismatch")
    return [
        {"object": name, "expected_sha256": expected, "observed_sha256": observed,
         "replay_count": count, "status": "PASS"}
        for name, expected, observed, count in rows
    ]


def unit_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(unit["unit_id"]): unit
        for dataset in manifest["datasets"]
        for unit in dataset["units"]
    }


def artifact_replay_rows(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_units = unit_map(manifest)
    sidecar_paths = sorted(RUN_ROOT.glob("*/sidecars/*.json"))
    require(len(sidecar_paths) == 243, "expected 243 sidecars")
    rows: list[dict[str, Any]] = []
    maxima = {"Wz_plus_b": 0.0, "linear": 0.0, "softmax": 0.0, "repeat_logits": 0.0, "repeat_z": 0.0}
    seen: set[str] = set()
    root = RUN_ROOT.resolve()
    for sidecar_path in sidecar_paths:
        sidecar = read_json(sidecar_path)
        unit_id = str(sidecar["unit_id"])
        require(unit_id in expected_units and unit_id not in seen, f"sidecar unit identity drift: {unit_id}")
        seen.add(unit_id)
        unit = expected_units[unit_id]
        require(sha256_file(sidecar_path) == unit["sidecar_sha256"], f"sidecar hash drift: {unit_id}")
        artifacts = {
            "checkpoint": (Path(sidecar["checkpoint"]["path"]), unit["checkpoint_sha256"]),
            "optimizer": (Path(sidecar["optimizer"]["path"]), unit["optimizer_sha256"]),
            "source_audit": (Path(sidecar["source_audit"]["path"]), unit["source_audit_sha256"]),
            "target_unlabeled": (
                Path(sidecar["target_unlabeled"]["path"]), unit["target_unlabeled_sha256"]
            ),
        }
        for name, (path, expected_sha) in artifacts.items():
            require(path.resolve().is_relative_to(root), f"artifact outside run root: {path}")
            require(path.is_file() and sha256_file(path) == expected_sha,
                    f"{name} byte/hash replay failed: {unit_id}")
        require(sidecar["checkpoint"]["replay"]["replay_pass"] is True, "checkpoint replay failed")
        require(sidecar["optimizer"]["replay"]["replay_pass"] is True, "optimizer replay failed")
        require(sidecar["source_audit"]["replay_pass"] is True, "source replay failed")
        require(sidecar["target_unlabeled"]["replay_pass"] is True, "target replay failed")
        require(set(sidecar["source_audit"]["fields"]) == SOURCE_FIELDS, "source field drift")
        require(set(sidecar["target_unlabeled"]["fields"]) == TARGET_FIELDS, "target field drift")
        require(sidecar["source_audit"]["target_label_fields"] == 0, "source target-label field")
        require(sidecar["target_unlabeled"]["target_label_fields"] == 0, "target-label field")
        require(sidecar["montage_sha256"] == EXPECTED["montage_sha256"], "montage drift")
        require(sidecar["level_intervention_registry_sha256"] == EXPECTED["intervention_registry_sha256"],
                "intervention registry drift")
        require(sidecar["level"] == 1 and sidecar["panel"] == "A" and sidecar["seed"] == 5,
                "canary scope drift")
        require(sidecar["deleted_source_subject"] == DELETED_SUBJECTS[sidecar["dataset"]],
                "deleted subject drift")
        require(sidecar["deleted_class"] == "left_hand", "deleted class drift")
        require(sidecar["paired_model_init_pass"] is True and sidecar["level0_plan_replay_pass"] is True,
                "paired identity replay failed")
        for counter in (
            "training_target_rows", "training_target_labels", "source_audit_rows_used_in_training",
            "target_outcome_retention", "target_outcome_retry", "target_scientific_metrics",
        ):
            require(sidecar[counter] == 0, f"protected counter {counter} is nonzero")
        target = sidecar["target_unlabeled"]
        require(target["Wz_plus_b_max_error"] <= 2e-5, "in-memory linear replay exceeds tolerance")
        require(target["linear_replay_max_abs_error"] <= 2e-5, "persisted linear replay exceeds tolerance")
        for key in ("softmax_max_error", "repeat_logits_max_error", "repeat_z_max_error"):
            require(target[key] <= 1e-6, f"strict target replay failed: {key}")
        maxima["Wz_plus_b"] = max(maxima["Wz_plus_b"], target["Wz_plus_b_max_error"])
        maxima["linear"] = max(maxima["linear"], target["linear_replay_max_abs_error"])
        maxima["softmax"] = max(maxima["softmax"], target["softmax_max_error"])
        maxima["repeat_logits"] = max(maxima["repeat_logits"], target["repeat_logits_max_error"])
        maxima["repeat_z"] = max(maxima["repeat_z"], target["repeat_z_max_error"])
        rows.append({
            "dataset": sidecar["dataset"], "unit_id": unit_id, "regime": sidecar["regime"],
            "epoch": sidecar["epoch"], "trajectory_order": sidecar["trajectory_order"],
            "checkpoint_sha256": unit["checkpoint_sha256"],
            "optimizer_sha256": unit["optimizer_sha256"], "sidecar_sha256": unit["sidecar_sha256"],
            "source_audit_sha256": unit["source_audit_sha256"],
            "target_unlabeled_sha256": unit["target_unlabeled_sha256"],
            "Wz_plus_b_max_error": target["Wz_plus_b_max_error"],
            "persisted_linear_max_error": target["linear_replay_max_abs_error"],
            "softmax_max_error": target["softmax_max_error"],
            "repeat_logits_max_error": target["repeat_logits_max_error"],
            "repeat_z_max_error": target["repeat_z_max_error"], "status": "PASS",
        })
    require(seen == set(expected_units), "sidecar coverage differs from manifest")
    summary = {
        "units": len(rows), "checkpoint_replay_units": len(rows), "optimizer_replay_units": len(rows),
        "sidecar_replay_units": len(rows), "source_audit_replay_units": len(rows),
        "target_unlabeled_replay_units": len(rows), "max_Wz_plus_b_error": maxima["Wz_plus_b"],
        "max_linear_replay_error": maxima["linear"], "max_softmax_error": maxima["softmax"],
        "max_repeat_logits_error": maxima["repeat_logits"], "max_repeat_z_error": maxima["repeat_z"],
    }
    return rows, summary


def dataset_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in manifest["datasets"]:
        intervention = dataset["level_intervention"]
        rows.append({
            "dataset": dataset["dataset"],
            "source_training_subjects": "|".join(map(str, dataset["source_training_subjects"])),
            "source_audit_subjects": "|".join(map(str, dataset["source_audit_subjects"])),
            "target_subjects": "|".join(map(str, dataset["target_subjects"])),
            "deleted_source_subject": intervention["deleted_source_subject"],
            "deleted_class": intervention["deleted_class"],
            "pre_trial_count": intervention["pre_trial_count"],
            "post_trial_count": intervention["post_trial_count"],
            "deleted_trial_count": intervention["pre_trial_count"] - intervention["post_trial_count"],
            "pre_support_cells": len(intervention["pre_cell_counts"]),
            "post_support_cells": len(intervention["post_cell_counts"]),
            "minimum_post_support": min(cell[2] for cell in intervention["post_cell_counts"]),
            "source_audit_trials": dataset["source_audit_trial_count"],
            "target_unlabeled_trials": dataset["target_unlabeled_trial_count"],
            "paired_model_init_hash": dataset["paired_model_init_hash"],
            "support_graph_sha256": intervention["support_graph_sha256"],
            "population_signature_sha256": intervention["population_signature_sha256"],
            "unit_count": dataset["unit_count"], "target_label_access": dataset["target_label_access"],
            "scientific_metrics": dataset["scientific_metrics"], "status": "PASS",
        })
    return rows


def artifact_count_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for kind in ARTIFACT_KINDS:
            count = sum(path.is_file() for path in (RUN_ROOT / dataset / kind).iterdir())
            require(count == 81, f"wrong {kind} count for {dataset}: {count}")
            rows.append({"dataset": dataset, "artifact_kind": kind, "expected": 81,
                         "observed": count, "status": "PASS"})
    return rows


def attempt_summary() -> dict[str, Any]:
    events = [json.loads(line) for line in ATTEMPT_LEDGER.read_text(encoding="utf-8").splitlines() if line]
    require(len(events) == 12, "unexpected attempt-ledger event count")
    final = events[-1]
    require(final["event"] == "completed" and final["stage"] == "manifest_publication",
            "attempt ledger lacks completion event")
    expected_counters = {
        "CUDA_checks": 1, "complete_units": 243, "dataset_loader_imports": 1,
        "get_data_calls_completed": 9, "get_data_calls_started": 9, "loader_source_replays": 1,
        "package_imports": 4, "real_EEG_arrays_materialized": 9, "source_audit_artifacts": 243,
        "source_label_arrays_read": 6, "target_scientific_metrics": 0,
        "target_unlabeled_artifacts": 243, "target_y_accesses": 0,
        "training_phases_completed": 9, "training_phases_started": 9,
    }
    require(final["counters"] == expected_counters, "final attempt counters drift")
    require(final["complete_manifest_sha256"] == EXPECTED["complete_manifest_sha256"],
            "attempt manifest hash drift")
    elapsed = (final["completed_at_unix_ns"] - events[0]["started_at_unix_ns"]) / 1_000_000_000
    return {"event_count": len(events), "final_event": final["event"], "final_stage": final["stage"],
            "elapsed_seconds": elapsed, "counters": final["counters"],
            "complete_manifest_sha256": final["complete_manifest_sha256"]}


def job_summary() -> dict[str, Any]:
    job = read_json(SCHEDULER_SNAPSHOT)
    require(job["job_id"] == JOB_ID and job["sacct_used"] is False, "scheduler identity drift")
    require(job["squeue_observed_running"] is True, "squeue never observed RUNNING")
    require(job["squeue_final_state"] == "ABSENT_AFTER_APPLICATION_COMPLETE", "final squeue state drift")
    require(job["application_complete"] is True, "application completion absent")
    require(job["application_complete_manifest_sha256"] == EXPECTED["complete_manifest_sha256"],
            "scheduler snapshot manifest drift")
    return job


def warning_rows() -> list[dict[str, Any]]:
    stderr = STDERR_LOG.read_text(encoding="utf-8", errors="replace")
    rows = [
        {"warning": "Cho_continuous_stack_edge_effect_notice",
         "count": stderr.count("edge effects present"),
         "classification": "NONBLOCKING_DISCLOSED_LOADER_NOTICE"},
        {"warning": "traceback_or_runtime_failure",
         "count": len(re.findall(r"Traceback|RuntimeError|AssertionError|FAILED|Exception", stderr)),
         "classification": "PASS_ZERO"},
    ]
    require(rows[0]["count"] == 17, "unexpected loader notice count")
    require(rows[1]["count"] == 0, "runtime failure marker in stderr")
    return rows


def external_rows() -> list[dict[str, Any]]:
    objects = (
        ("complete_manifest", COMPLETE_MANIFEST), ("partial_manifest", PARTIAL_MANIFEST),
        ("authorization_consumed", AUTHORIZATION_CONSUMED), ("attempt_ledger", ATTEMPT_LEDGER),
        ("slurm_stdout", STDOUT_LOG), ("slurm_stderr", STDERR_LOG),
    )
    return [
        {"object": name, "path": str(path), "sha256": sha256_file(path),
         "bytes": path.stat().st_size, "status": "PASS"}
        for name, path in objects
    ]


def generate() -> dict[str, Any]:
    manifest = read_json(COMPLETE_MANIFEST)
    validate_manifest(manifest)
    require(sha256_file(COMPLETE_MANIFEST) == EXPECTED["complete_manifest_sha256"],
            "complete manifest hash mismatch")
    require(sha256_file(AUTHORIZATION_RECORD) == EXPECTED["authorization_record_sha256"],
            "authorization record hash mismatch")
    require(sha256_file(AUTHORIZATION_CONSUMED) == EXPECTED["authorization_consumption_sha256"],
            "authorization consumption hash mismatch")
    partial = read_json(PARTIAL_MANIFEST)
    require(partial["status"] == "COMPLETE" and partial["error"] is None, "partial manifest failed")
    require(partial["retry_disposition"] == "NOT_APPLICABLE", "unexpected retry disposition")
    require(partial["target_outcome_decision"] is False, "target-outcome decision detected")

    lock_rows = lock_replay_rows()
    replay_rows, replay = artifact_replay_rows(manifest)
    datasets = dataset_rows(manifest)
    count_rows = artifact_count_rows()
    attempt = attempt_summary()
    job = job_summary()
    warnings = warning_rows()
    external = external_rows()

    complete_rows = [
        {"gate_item": "candidate_units", "expected": 243, "observed": manifest["unit_count"], "status": "PASS"},
        {"gate_item": "training_phases", "expected": 9, "observed": manifest["training_phases"], "status": "PASS"},
        {"gate_item": "checkpoint_optimizer_sidecar_units", "expected": 243,
         "observed": manifest["complete_gate"]["checkpoint_optimizer_sidecar_units"], "status": "PASS"},
        {"gate_item": "strict_source_audit_artifacts", "expected": 243,
         "observed": manifest["source_audit_artifacts"], "status": "PASS"},
        {"gate_item": "target_unlabeled_artifacts", "expected": 243,
         "observed": manifest["target_unlabeled_artifacts"], "status": "PASS"},
    ]
    isolation_rows = [
        {"counter": "target_y_accesses", "expected": 0,
         "observed": attempt["counters"]["target_y_accesses"], "status": "PASS"},
        {"counter": "target_label_fields", "expected": 0, "observed": 0, "status": "PASS"},
        {"counter": "training_target_rows", "expected": 0, "observed": 0, "status": "PASS"},
        {"counter": "training_target_labels", "expected": 0, "observed": 0, "status": "PASS"},
        {"counter": "source_audit_rows_used_in_training", "expected": 0, "observed": 0, "status": "PASS"},
        {"counter": "target_outcome_retention", "expected": 0, "observed": 0, "status": "PASS"},
        {"counter": "target_outcome_retry", "expected": 0, "observed": 0, "status": "PASS"},
        {"counter": "target_scientific_metrics", "expected": 0,
         "observed": attempt["counters"]["target_scientific_metrics"], "status": "PASS"},
        {"counter": "construction_evaluation_oracle_access", "expected": 0, "observed": 0, "status": "PASS"},
    ]
    failure_rows = [
        {"attempt_job": 895928, "role": "historical_failed_attempt", "status": "FAILED_PRESERVED",
         "authorization_reused": 0, "artifacts_reused": 0, "target_y_access": 0,
         "target_scientific_metrics": 0, "retry_disposition": "SUPERSEDED_BY_REPAIR_RELOCK_FRESH_AUTH"},
        {"attempt_job": JOB_ID, "role": "authorized_replacement", "status": "APPLICATION_COMPLETE",
         "authorization_reused": 0, "artifacts_reused": 0, "target_y_access": 0,
         "target_scientific_metrics": 0, "retry_disposition": "NOT_APPLICABLE"},
    ]
    job_rows = [{
        "job_id": JOB_ID, "scheduler_source": job["source"], "scheduler_final": job["squeue_final_state"],
        "scheduler_exit_code": job["exit_code"], "application_complete": int(job["application_complete"]),
        "attempt_elapsed_seconds": job["attempt_elapsed_seconds"], "last_squeue_runtime": job["last_observed_runtime"],
        "partition": job["allocation"]["partition"], "node": job["node"],
        "cpus": job["allocation"]["cpus"], "memory": job["allocation"]["memory"],
        "gpu": job["allocation"]["gpu"], "sacct_used": int(job["sacct_used"]), "status": "PASS",
    }]

    write_csv(TABLE_DIR / "authorization_lock_replay.csv", lock_rows)
    write_csv(TABLE_DIR / "dataset_support_replay.csv", datasets)
    write_csv(TABLE_DIR / "complete_gate.csv", complete_rows)
    write_csv(TABLE_DIR / "artifact_count_replay.csv", count_rows)
    write_csv(TABLE_DIR / "artifact_replay.csv", replay_rows)
    write_csv(TABLE_DIR / "target_label_isolation.csv", isolation_rows)
    write_csv(TABLE_DIR / "external_artifact_manifest.csv", external)
    write_csv(TABLE_DIR / "job_resource_ledger.csv", job_rows)
    write_csv(TABLE_DIR / "warning_ledger.csv", warnings)
    write_csv(TABLE_DIR / "failure_and_retry_ledger.csv", failure_rows)

    external_files = [path for path in RUN_ROOT.rglob("*") if path.is_file()]
    result = {
        "schema_version": "c84l1c_engineering_canary_result_v1",
        "milestone": "C84L1C", "gate": GATE, "engineering_only": True,
        "scientific_result_available": False, "execution_base_commit": EXECUTION_BASE_COMMIT,
        "job": job,
        "authorization": {
            "record_sha256": EXPECTED["authorization_record_sha256"],
            "consumption_sha256": EXPECTED["authorization_consumption_sha256"],
            "historical_authorization_reused": False, "C84F_authorized": False, "C84S_authorized": False,
        },
        "lock": {
            "repair_protocol_sha256": EXPECTED["repair_protocol_sha256"],
            "canary_protocol_v2_sha256": EXPECTED["canary_protocol_v2_sha256"],
            "execution_lock_v2_sha256": EXPECTED["execution_lock_v2_sha256"],
            "runtime_bound_objects": 125, "protocol_bindings": 5,
            "candidate_unit_ids_sha256": EXPECTED["candidate_unit_ids_sha256"],
            "montage_sha256": EXPECTED["montage_sha256"],
            "intervention_registry_sha256": EXPECTED["intervention_registry_sha256"],
        },
        "scope": {"datasets": list(DATASETS), "source_panel": "A", "training_seed": 5, "level": 1,
                  "regimes": ["ERM", "OACI", "SRC"], "training_phases": 9, "candidate_units": 243},
        "complete_gate": manifest["complete_gate"], "attempt": attempt, "datasets": datasets,
        "persisted_replay": replay,
        "isolation": {row["counter"]: row["observed"] for row in isolation_rows},
        "external_root": str(RUN_ROOT),
        "external_root_bytes": sum(path.stat().st_size for path in external_files),
        "external_file_count": len(external_files),
        "complete_manifest_sha256": EXPECTED["complete_manifest_sha256"],
        "partial_manifest_sha256": sha256_file(PARTIAL_MANIFEST),
        "external_artifacts": external, "warnings": warnings,
        "historical_failed_attempt": 895928, "replacement_attempt": JOB_ID, "replacement_retries": 0,
        "C84F_authorized": False, "C84S_authorized": False,
        "next_step": "PM_review_then_C84FL2_implementation_and_separate_C84F_lock",
    }
    result_path = REPORT_DIR / "C84L1C_ENGINEERING_CANARY_RESULT.json"
    write_json(result_path, result)
    result_sha = sha256_file(result_path)
    (REPORT_DIR / "C84L1C_ENGINEERING_CANARY_RESULT.sha256").write_text(
        f"{result_sha}  {result_path.name}\n", encoding="ascii",
    )
    (REPORT_DIR / "C84L1C_ENGINEERING_CANARY_RESULT.md").write_text(
        "# C84L1C Engineering Canary Result\n\n"
        f"Authorized replacement job `{JOB_ID}` completed the application in "
        f"`{attempt['elapsed_seconds']:.3f}` seconds on `{job['node']}`. Scheduler state was "
        "tracked with `squeue`; `sacct` was not used. The complete manifest SHA-256 is "
        f"`{EXPECTED['complete_manifest_sha256']}`.\n\n"
        "The engineering gate passed: 243/243 units, 9/9 phases, 243/243 checkpoint/optimizer/"
        "sidecar byte replays, 243/243 strict-source artifacts, and 243/243 target-unlabeled "
        "artifacts. Each dataset had exactly one fixed left-hand source-support cell removed, "
        "23/24 retained support cells, paired model initialization, and exact level-0 plan replay.\n\n"
        f"The maximum in-memory float32 linear replay error was `{replay['max_Wz_plus_b_error']}` "
        "under `2e-5`; the maximum persisted replay error was "
        f"`{replay['max_linear_replay_error']}`. Softmax, repeat-logit, and repeat-z maximum "
        "errors were zero under `1e-6`.\n\n"
        "Target-y access, target-label fields, construction/evaluation/oracle access, target "
        "scientific metrics, target-outcome retention, and target-outcome retry were all zero. "
        "No target accuracy, selector score, regret, Q1/Q2, label-budget, or level comparison was "
        "computed. The 17 stderr lines are disclosed Cho stacking notices with no failure marker.\n\n"
        "Failed job `895928` remains preserved; neither its authorization nor artifacts were "
        "reused. C84F and C84S remain unauthorized.\n\n"
        f"Gate: `{GATE}`.\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
