"""Generate the scope-specific, non-authorizing C84C execution lock."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from . import c84_dataset_registry_v2 as registry
from . import c84r_v2_protocols as protocols
from .c84r_montage_repair import (
    EPOCH_RULE,
    INTERFACE_ID,
    MONTAGE_SHA256,
    UNIT_ID_SALT,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84r_tables"
IMPLEMENTATION_COMMIT = "e91b71c5e0cd99d90c8ac9c44e2736a4cfc18f4f"
V2_PROTOCOL_COMMIT = "a5d9fd0a0e76a7e0c6a49b87048d642eb8c0da6a"
REPAIR_PROTOCOL_COMMIT = "482a725abc6bf1f0e5d33be76ea17d37bcfaa6c3"
REPAIR_PROTOCOL_SHA256 = "a6a1fd85ef1b7520a55ef8e075933d08bf6639cbf89bbcf761dec2a753ab1c91"
CREATED_AT_UTC = "2026-07-13T22:27:00Z"


IMPLEMENTATION_PATHS = (
    "oaci/multidataset/c84c_real_canary.py",
    "oaci/multidataset/c84_dataset_registry_v2.py",
    "oaci/multidataset/c84r_v2_protocols.py",
    "oaci/slurm_c84c_canary.sh",
    "oaci/models/factory.py",
    "oaci/models/shallow.py",
    "oaci/models/output.py",
    "oaci/support_graph.py",
    "oaci/config.py",
    "oaci/data/plan_materialize.py",
    "oaci/data/plan_sampler.py",
    "oaci/methods/oaci.py",
    "oaci/methods/source_robust.py",
    "oaci/train/batch_plan.py",
    "oaci/train/bn.py",
    "oaci/train/checkpoint.py",
    "oaci/train/data.py",
    "oaci/train/engine.py",
    "oaci/train/evaluate.py",
    "oaci/train/objective.py",
    "oaci/train/risk.py",
    "oaci/train/rng.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip()


def _sidecar_digest(path: Path) -> str:
    return path.read_text(encoding="ascii").split()[0]


def _blob(commit: str, path: str) -> str:
    return _git("rev-parse", f"{commit}:{path}")


def _file_binding(path: str, commit: str = IMPLEMENTATION_COMMIT) -> dict[str, Any]:
    current = REPO_ROOT / path
    committed_bytes = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout
    current_bytes = current.read_bytes()
    if committed_bytes != current_bytes:
        raise RuntimeError(f"C84C implementation path differs from {commit}: {path}")
    return {
        "path": path,
        "commit": commit,
        "blob": _blob(commit, path),
        "sha256": hashlib.sha256(current_bytes).hexdigest(),
        "bytes": len(current_bytes),
    }


def _protocol_binding(stem: str) -> dict[str, Any]:
    path = REPORT_DIR / f"{stem}.json"
    sha_path = REPORT_DIR / f"{stem}.sha256"
    digest = _sidecar_digest(sha_path)
    if sha256_file(path) != digest:
        raise RuntimeError(f"C84C protocol replay failed: {stem}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256_path": str(sha_path.relative_to(REPO_ROOT)),
        "sha256": digest,
        "commit": V2_PROTOCOL_COMMIT,
        "blob": _blob(V2_PROTOCOL_COMMIT, str(path.relative_to(REPO_ROOT))),
    }


def build_lock() -> dict[str, Any]:
    external = _protocol_binding("C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2")
    canary = _protocol_binding("C84_CANARY_PROTOCOL_V2")
    field = _protocol_binding("C84_FIELD_GENERATION_PROTOCOL_V2")
    science = _protocol_binding("C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2")
    units = protocols.candidate_units()
    canary_units = [row for row in units if row["canary_subset"]]
    if len(canary_units) != 243:
        raise RuntimeError("C84C lock did not resolve 243 canary units")
    partitions_path = REPORT_DIR / "c84p_tables/subject_partition_registry.csv"
    splits_path = REPORT_DIR / "c84r_tables/subject_partition_identity_replay.csv"
    selector_path = REPORT_DIR / "c84p_tables/selector_registry_replay.csv"
    resource_path = TABLE_DIR / "resource_estimate.csv"
    implementation = [_file_binding(path) for path in IMPLEMENTATION_PATHS]
    return {
        "schema_version": "c84c_execution_lock_v1",
        "milestone": "C84C",
        "status": "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED",
        "created_at_utc": CREATED_AT_UTC,
        "chronology": {
            "historical_C84P_HEAD": protocols.HISTORICAL_HEAD,
            "repair_protocol_commit": REPAIR_PROTOCOL_COMMIT,
            "V2_protocol_commit": V2_PROTOCOL_COMMIT,
            "final_canary_implementation_commit": IMPLEMENTATION_COMMIT,
            "repair_precedes_V2_protocols": True,
            "V2_protocols_precede_real_adapter": True,
            "real_data_access_before_lock": 0,
            "dataset_downloads_before_lock": 0,
            "real_labels_before_lock": 0,
            "training_forward_GPU_before_lock": 0,
        },
        "repair_protocol": {
            "path": "oaci/reports/C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.json",
            "commit": REPAIR_PROTOCOL_COMMIT,
            "sha256": REPAIR_PROTOCOL_SHA256,
        },
        "external_protocol": external,
        "canary_protocol": canary,
        "future_field_protocol": field,
        "future_scientific_protocol": science,
        "interface": {
            "id": INTERFACE_ID,
            "schema": registry.SCHEMA_VERSION,
            "channels": list(registry.PRIMARY_CHANNELS),
            "channel_count": 20,
            "montage_sha256": MONTAGE_SHA256,
            "epoch_rule": EPOCH_RULE,
            "sample_rate_hz": 160,
            "input_shape": [20, 480],
            "class_mapping_version": "C84_LEFT_RIGHT_CLASS_MAPPING_V1",
            "Fz_substitution": False,
            "FCz_interpolation": False,
            "zero_fill": False,
            "dataset_specific_mask": False,
        },
        "dataset_registry": {
            "path": "oaci/multidataset/c84_dataset_registry_v2.py",
            "blob": _blob(IMPLEMENTATION_COMMIT, "oaci/multidataset/c84_dataset_registry_v2.py"),
            "sha256": sha256_file(REPO_ROOT / "oaci/multidataset/c84_dataset_registry_v2.py"),
            "datasets": list(protocols.DATASET_ORDER),
            "moabb": "1.5.0",
            "mne": "1.11.0",
            "Lee_online_runs": "excluded",
            "Cho_bad_trials": "exact_MOABB_loader_return_no_extra_parse",
            "Physionet_subject_88": "excluded",
        },
        "subject_partition": {
            "path": str(partitions_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(partitions_path),
            "identity_replay_path": str(splits_path.relative_to(REPO_ROOT)),
            "identity_replay_sha256": sha256_file(splits_path),
            "salt": registry.SUBJECT_PARTITION_SALT,
            "panel": "A",
            "source_training_per_dataset": 12,
            "source_audit_per_dataset": 4,
            "canary_targets": protocols.CANARY_TARGETS,
            "subject_partition_changed_from_C84P": False,
        },
        "candidate_identity": {
            "function_path": "oaci/multidataset/c84r_v2_protocols.py",
            "function_blob": _blob(IMPLEMENTATION_COMMIT, "oaci/multidataset/c84r_v2_protocols.py"),
            "salt": UNIT_ID_SALT,
            "complete_unit_count": 1944,
            "canary_unit_count": len(canary_units),
            "canary_unit_ids_sha256": hashlib.sha256(canonical_bytes(sorted(row["unit_id"] for row in canary_units))).hexdigest(),
            "historical_blocked_unit_ID_reuse": 0,
            "C84C_C84F_identity_function_same": True,
        },
        "scope": {
            "datasets": list(protocols.DATASET_ORDER),
            "source_panel": "A", "training_seed": 5, "level": 0,
            "regimes": ["ERM", "OACI", "SRC"],
            "units_per_dataset": 81, "total_units": 243, "training_phases": 9,
            "engineering_only": True,
            "C84F": False, "C84S": False,
        },
        "implementation": {
            "commit": IMPLEMENTATION_COMMIT,
            "files": implementation,
            "entrypoint": "python -m oaci.multidataset.c84c_real_canary run-real",
            "slurm_entrypoint": "oaci/slurm_c84c_canary.sh",
            "loader_import_after_authorization_consumption": True,
            "historical_ERM_OACI_SRC_formulas_unchanged": True,
            "checkpoint_epochs": list(range(4, 200, 5)),
            "optimizer_state_retention": True,
            "instrumentation_identity": ["Wz_plus_b_equals_logits", "softmax_equals_probabilities",
                                         "repeat_logits", "repeat_z", "checkpoint_hash", "genealogy"],
        },
        "views": {
            "source_training_view": {"X": True, "y": True, "training": True},
            "source_audit_view": {"X": True, "y": True, "training": False, "scientific_output": False},
            "target_unlabeled_view": {"fields": ["X", "trial_id", "target_subject_id", "session", "run", "dataset_id"],
                                      "y": False, "scientific_output": False},
            "target_construction_view": {"provisioned": False},
            "target_evaluation_view": {"provisioned": False},
            "same_label_oracle_view": {"reachable": False},
            "target_y_operations": {"index": False, "hash": False, "summarize": False,
                                    "log": False, "retention": False, "retry": False},
        },
        "frozen_registries": {
            "selector_registry": {"path": str(selector_path.relative_to(REPO_ROOT)), "sha256": sha256_file(selector_path),
                                  "used_for_C84C_scientific_score": False},
            "budgets_used_in_C84C": False,
            "inference_used_in_C84C": False,
        },
        "environment": {
            "conda_prefix": "/home/infres/yinwang/anaconda3/envs/eeg2025",
            "python": "3.9.25", "torch": "2.6.0+cu124", "moabb": "1.5.0", "mne": "1.11.0",
            "CUDA_allocation": "Slurm_V100_one_GPU", "GPU_required_after_authorization": True,
        },
        "runtime": {
            "external_root": "/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v2",
            "content_addressed_subroot": "lock_<execution_lock_sha256_prefix20>",
            "empty_or_authorization_only_root_required": True,
            "resource_estimate_path": str(resource_path.relative_to(REPO_ROOT)),
            "resource_estimate_sha256": sha256_file(resource_path),
            "GPU_phase_hours_max": 250,
            "external_payload_TiB_max": 2.0,
            "Git_file_MiB_max": 50,
        },
        "authorization": {
            "record_path": "oaci/reports/C84C_PI_AUTHORIZATION_RECORD.json",
            "record_present_at_lock": False,
            "direct_PI_statement_required": "授权 C84C",
            "magic_token_required": False,
            "hash_recital_required": False,
            "server_record_must_bind_protocol_and_lock": True,
            "authorization_not_inherited_from_prior_C84_messages": True,
        },
        "retry_policy": {
            "all_attempts_preserved": True,
            "target_outcome_decisions": 0,
            "engineering_failure_requires_additive_repair_ledger_and_new_lock": True,
            "silent_rerun": False,
        },
        "report_schema": {
            "complete_manifest": "C84C_COMPLETE_CANARY_MANIFEST.json",
            "candidate_sidecars": "<dataset>/sidecars/<unit_id>.json",
            "checkpoints": "<dataset>/checkpoints/<unit_id>.pt",
            "optimizer_states": "<dataset>/optimizer_states/*.pt",
            "target_unlabeled_instrumentation": "<dataset>/instrumentation/<unit_id>.npz",
            "scientific_result_table": None,
        },
        "forbidden": {
            "target_accuracy_calibration_regret_or_label_counts": True,
            "selector_scores_Q1_Q2_budget_frontier": True,
            "construction_or_evaluation_labels": True,
            "same_label_oracle": True,
            "C84F_or_C84S_execution": True,
            "target_outcome_retention_or_retry": True,
            "Fz_FCz_interpolation_mask_zero_fill": True,
            "raw_EEG_weights_optimizer_states_or_caches_in_Git": True,
        },
        "protected_state_at_lock": {
            "real_EEG_arrays_loaded": 0,
            "real_labels_read": 0,
            "dataset_downloads": 0,
            "training_forward_GPU_jobs": 0,
            "candidate_units_created": 0,
            "target_scientific_metrics": 0,
            "authorization_record_present": False,
            "C84F_execution_lock_present": False,
            "C84S_execution_lock_present": False,
        },
    }


def generate() -> dict[str, Any]:
    lock = build_lock()
    path = REPORT_DIR / "C84C_EXECUTION_LOCK.json"
    path.write_bytes(canonical_bytes(lock) + b"\n")
    digest = sha256_file(path)
    sha_path = REPORT_DIR / "C84C_EXECUTION_LOCK.sha256"
    sha_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    rows = [
        {"object": "repair_protocol", "path": lock["repair_protocol"]["path"],
         "sha256": lock["repair_protocol"]["sha256"], "replay_pass": 1},
        {"object": "V2_external_protocol", "path": lock["external_protocol"]["path"],
         "sha256": lock["external_protocol"]["sha256"], "replay_pass": 1},
        {"object": "V2_canary_protocol", "path": lock["canary_protocol"]["path"],
         "sha256": lock["canary_protocol"]["sha256"], "replay_pass": 1},
        {"object": "C84C_execution_lock", "path": "oaci/reports/C84C_EXECUTION_LOCK.json",
         "sha256": digest, "replay_pass": int(sha256_file(path) == digest)},
        {"object": "adapter", "path": "oaci/multidataset/c84c_real_canary.py",
         "sha256": sha256_file(REPO_ROOT / "oaci/multidataset/c84c_real_canary.py"), "replay_pass": 1},
    ]
    table = TABLE_DIR / "canary_lock_replay.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "lock_sha256": digest,
        "lock_status": lock["status"],
        "canary_units": lock["scope"]["total_units"],
        "authorization_record_present": False,
        "real_EEG_arrays_loaded": 0,
        "C84F_or_C84S_locks_created": 0,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
