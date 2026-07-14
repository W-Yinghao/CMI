"""Generate the scope-specific C84L1C execution lock.

The generator is run only after its implementation bytes are committed. It
binds those exact Git blobs, the additive protocol family, and the accepted
C84C engineering manifest without loading EEG arrays or scientific outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from . import c84l1_protocols as protocol
from . import c84l1_runtime_guard as runtime
from .c84_dataset_registry_v2 import PRIMARY_CHANNELS
from .c84r_montage_repair import (
    CLASS_MAPPING_VERSION,
    EPOCH_RULE,
    INTERFACE_ID,
    MONTAGE_SHA256,
)


REPO_ROOT = protocol.REPO_ROOT
REPORT_DIR = protocol.REPORT_DIR
TABLE_DIR = protocol.TABLE_DIR
PRIOR_LOCK_PATH = REPORT_DIR / "C84C_EXECUTION_LOCK_V3.json"
C84C_RESULT_PATH = REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.json"
C84C_RESULT_SHA_PATH = REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.sha256"
LOCK_PATH = REPORT_DIR / "C84L1C_EXECUTION_LOCK.json"
LOCK_SHA_PATH = REPORT_DIR / "C84L1C_EXECUTION_LOCK.sha256"

NEW_IMPLEMENTATION_PATHS = (
    "oaci/multidataset/c84l1_protocols.py",
    "oaci/multidataset/c84l1_intervention.py",
    "oaci/multidataset/c84l1_runtime_guard.py",
    "oaci/multidataset/c84l1_canary.py",
    "oaci/multidataset/c84l1_lock.py",
    "oaci/slurm_c84l1c_canary.sh",
)
NEW_REGISTRY_PATHS = (
    "oaci/reports/C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.json",
    "oaci/reports/C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.sha256",
    "oaci/reports/C84L1_PROTOCOL_TIMING_AUDIT.md",
    "oaci/reports/C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3.json",
    "oaci/reports/C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3.sha256",
    "oaci/reports/C84_LEVEL1_CANARY_PROTOCOL_V1.json",
    "oaci/reports/C84_LEVEL1_CANARY_PROTOCOL_V1.sha256",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V5.json",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V5.sha256",
    "oaci/reports/C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3.json",
    "oaci/reports/C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3.sha256",
    "oaci/reports/C84C_ENGINEERING_CANARY_RESULT.json",
    "oaci/reports/C84C_ENGINEERING_CANARY_RESULT.sha256",
    "oaci/reports/C84FL_OVERALL_REPORT.json",
    "oaci/reports/C84FL_OVERALL_REPORT.md",
    "oaci/reports/C84FL_OVERALL_REPORT.sha256",
    "oaci/reports/c84fl_tables/source_view_contract.csv",
    "oaci/reports/c84l1p_tables/level_intervention_registry.csv",
    "oaci/reports/c84l1p_tables/level_support_contract.csv",
    "oaci/reports/c84l1p_tables/level1_fail_closed_support_cases.csv",
    "oaci/reports/c84l1p_tables/historical_level1_unit_id_supersession.csv",
    "oaci/reports/c84l1p_tables/operative_complete_unit_registry_v2.csv",
    "oaci/reports/c84l1p_tables/level1_candidate_id_registry.csv",
    "oaci/reports/c84l1p_tables/level1_candidate_id_digest.txt",
    "oaci/reports/c84l1p_tables/paired_rng_plan_contract.csv",
    "oaci/reports/c84l1p_tables/level0_identity_replay.csv",
    "oaci/reports/c84l1p_tables/level1_canary_scope.csv",
    "oaci/reports/c84l1p_tables/level1_canary_view_contract.csv",
    "oaci/reports/c84l1p_tables/level1_artifact_schema.csv",
    "oaci/tests/test_c84l1_intervention.py",
    "oaci/tests/test_c84l1_protocol_lock.py",
    "oaci/tests/test_c84l1_canary_contract.py",
)
PROTOCOL_STEMS = (
    "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL",
    "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3",
    "C84_LEVEL1_CANARY_PROTOCOL_V1",
    "C84_FIELD_GENERATION_PROTOCOL_V5",
    "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3",
)


def _git(*args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=check,
    ).stdout.strip()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def bind_path(path: str, implementation_commit: str) -> dict[str, Any]:
    current = REPO_ROOT / path
    if not current.is_file():
        raise RuntimeError(f"C84L1C lock path is absent: {path}")
    committed = subprocess.run(
        ["git", "show", f"{implementation_commit}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    current_bytes = current.read_bytes()
    if current_bytes != committed:
        raise RuntimeError(f"C84L1C path differs from implementation commit: {path}")
    return {
        "path": path,
        "sha256": hashlib.sha256(current_bytes).hexdigest(),
        "blob": _git("rev-parse", f"{implementation_commit}:{path}"),
        "bytes": len(current_bytes),
        "commit": implementation_commit,
    }


def protocol_binding(stem: str) -> dict[str, str]:
    path = REPORT_DIR / f"{stem}.json"
    sidecar = REPORT_DIR / f"{stem}.sha256"
    digest = sidecar.read_text(encoding="ascii").split()[0]
    if runtime.sha256_file(path) != digest:
        raise RuntimeError(f"C84L1C protocol hash replay failed: {stem}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256_path": str(sidecar.relative_to(REPO_ROOT)),
        "sha256": digest,
    }


def _accepted_c84c_binding(registry_sha256: str) -> dict[str, Any]:
    result_digest = C84C_RESULT_SHA_PATH.read_text(encoding="ascii").split()[0]
    if runtime.sha256_file(C84C_RESULT_PATH) != result_digest:
        raise RuntimeError("C84L1C accepted C84C result hash replay failed")
    result = json.loads(C84C_RESULT_PATH.read_text(encoding="utf-8"))
    manifest_path = Path(next(
        row["path"] for row in result["external_artifacts"] if row["object"] == "complete_manifest"
    ))
    if runtime.sha256_file(manifest_path) != protocol.C84C_MANIFEST_SHA256:
        raise RuntimeError("C84L1C accepted external manifest hash replay failed")
    summary = runtime.summarize_accepted_c84c_manifest(manifest_path)
    _, _, _, operative = protocol.identity_rows(registry_sha256)
    expected_ids = sorted(row["unit_id"] for row in operative if int(row["C84C_reusable"]) == 1)
    expected_digest = runtime.sha256_bytes(runtime.canonical_bytes(expected_ids))
    if expected_digest != summary["unit_ID_digest"]:
        raise RuntimeError("C84L1C accepted C84C IDs differ from unchanged level-0 IDs")
    return {
        "result_path": str(C84C_RESULT_PATH.relative_to(REPO_ROOT)),
        "result_sha256": result_digest,
        "result_commit": "2f541e526deb79091ad164b0d37419941e6f662b",
        "manifest_path": str(manifest_path),
        "manifest_sha256": protocol.C84C_MANIFEST_SHA256,
        "valid_job": 895441,
        "reusable_units": 243,
        "unit_ID_digest": summary["unit_ID_digest"],
        "model_unit_registry_sha256": summary["model_unit_registry_sha256"],
        "datasets": summary["datasets"],
        "level": 0,
        "scientific_outcomes": 0,
        "target_label_access": 0,
    }


def _candidate_binding(registry_sha256: str) -> dict[str, Any]:
    level0, level1, supersession, operative = protocol.identity_rows(registry_sha256)
    canary = sorted(row["unit_id"] for row in level1 if int(row["C84L1C_canary"]) == 1)
    level1_ids = sorted(row["unit_id"] for row in level1)
    operative_ids = sorted(row["unit_id"] for row in operative)
    blocked = sorted(row["historical_planned_level1_unit_id"] for row in supersession)
    return {
        "identity_salt": protocol.LEVEL1_UNIT_SALT,
        "level0_unit_count": len(level0),
        "level0_IDs_changed": False,
        "level1_unit_count": len(level1),
        "operative_unit_count": len(operative),
        "canary_unit_count": len(canary),
        "canary_unit_ID_digest": runtime.sha256_bytes(runtime.canonical_bytes(canary)),
        "level1_unit_ID_digest": runtime.sha256_bytes(runtime.canonical_bytes(level1_ids)),
        "operative_unit_ID_digest": runtime.sha256_bytes(runtime.canonical_bytes(operative_ids)),
        "blocked_level1_unit_ID_digest": runtime.sha256_bytes(runtime.canonical_bytes(blocked)),
        "historical_level1_IDs_operative": False,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise RuntimeError(f"refusing empty lock registry: {path}")
    fields = list(values[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def build_lock(implementation_commit: str) -> dict[str, Any]:
    if _git("rev-parse", "HEAD") != implementation_commit:
        raise RuntimeError("C84L1C lock generation requires HEAD at the implementation commit")
    if _git("status", "--porcelain"):
        raise RuntimeError("C84L1C lock generation requires a clean implementation worktree")
    prior_lock = json.loads(PRIOR_LOCK_PATH.read_text(encoding="utf-8"))
    prior_paths = [row["path"] for row in prior_lock["runtime_bound_objects"]]
    implementation_paths = _ordered_unique(
        [row["path"] for row in prior_lock["implementation"]["files"]] + list(NEW_IMPLEMENTATION_PATHS)
    )
    all_paths = _ordered_unique(prior_paths + list(NEW_IMPLEMENTATION_PATHS) + list(NEW_REGISTRY_PATHS))
    bound = [bind_path(path, implementation_commit) for path in all_paths]
    by_path = {row["path"]: row for row in bound}
    implementation = [by_path[path] for path in implementation_paths]
    protocols = [protocol_binding(stem) for stem in PROTOCOL_STEMS]
    registry_path = TABLE_DIR / "level_intervention_registry.csv"
    registry_sha = runtime.sha256_file(registry_path)
    accepted = _accepted_c84c_binding(registry_sha)
    candidates = _candidate_binding(registry_sha)

    return {
        "schema_version": "c84l1c_execution_lock_v1",
        "milestone": "C84L1C",
        "status": runtime.LOCK_READY_STATUS,
        "implementation_commit": implementation_commit,
        "chronology": {
            "C84FL_blocker_HEAD": protocol.C84FL_HEAD,
            "C84L1_protocol_commit": "a90f0051ed41937737ac7ac0258a882d45cefb33",
            "implementation_commit": implementation_commit,
            "protocol_precedes_implementation": True,
            "real_EEG_access_after_C84C": 0,
            "level1_real_EEG_access": 0,
            "level1_label_reads": 0,
            "level1_training_forward_GPU": 0,
        },
        "repair_protocol": protocols[0],
        "external_protocol": protocols[1],
        "canary_protocol": protocols[2],
        "future_field_protocol": protocols[3],
        "future_scientific_protocol": protocols[4],
        "protocol_bindings": protocols,
        "historical_C84FL": {
            "HEAD": protocol.C84FL_HEAD,
            "gate": "C84F_CANARY_REUSE_DATA_VIEW_IMPLEMENTATION_RESOURCE_OR_MANIFEST_RECONCILIATION_REQUIRED",
            "preserved": True,
            "level1_historical_IDs_operative": False,
        },
        "interface": {
            "id": INTERFACE_ID,
            "channels": list(PRIMARY_CHANNELS),
            "channel_count": 20,
            "montage_sha256": MONTAGE_SHA256,
            "epoch_rule": EPOCH_RULE,
            "sample_rate_hz": 160,
            "input_shape": [20, 480],
            "class_mapping_version": CLASS_MAPPING_VERSION,
            "Fz_substitution": False,
            "FCz_interpolation": False,
            "zero_fill": False,
            "dataset_specific_mask": False,
        },
        "scope": {
            "datasets": list(protocol.historical.DATASET_ORDER),
            "source_panel": "A",
            "training_seed": 5,
            "level": 1,
            "units_per_dataset": 81,
            "total_units": 243,
            "training_phases": 9,
            "targets": protocol.CANARY_TARGETS,
            "engineering_only": True,
            "C84F": False,
            "C84S": False,
        },
        "level_intervention": {
            "id": protocol.LEVEL1_ID,
            "registry_path": str(registry_path.relative_to(REPO_ROOT)),
            "registry_sha256": registry_sha,
            "deleted_class": protocol.DELETED_CLASS,
            "minimum_support": protocol.MIN_CELL_SUPPORT,
            "exact_post_cells": 23,
            "target_independent": True,
            "alternative_cell_allowed": False,
            "before_support_and_plan_materialization": True,
        },
        "candidate_identity": candidates,
        "accepted_C84C_level0": accepted,
        "paired_training": {
            "model_init_seed_rule": "derive_seed(training_seed,dataset,model_init)",
            "same_model_init_across_levels": True,
            "level0_plan_hashes_replayed_per_dataset": True,
            "architecture_optimizer_hyperparameters_epochs_cadence_same": True,
            "plans_materialized_from_level_specific_population_signature": True,
        },
        "environment": prior_lock["environment"],
        "loader_source_identity": prior_lock["loader_source_identity"],
        "implementation": {
            "commit": implementation_commit,
            "entrypoint": "python -m oaci.multidataset.c84l1_canary run-real",
            "slurm_entrypoint": "oaci/slurm_c84l1c_canary.sh",
            "files": implementation,
            "historical_ERM_OACI_SRC_formulas_unchanged": True,
        },
        "runtime_bound_objects": bound,
        "runtime_bound_object_count": len(bound),
        "runtime_replay": {
            "all_bound_SHA256_and_blobs_before_authorization_consumption": True,
            "protocol_sidecars": True,
            "accepted_C84C_manifest_plan_model_registry": True,
            "candidate_ID_universe": True,
            "environment_and_loader_identity": True,
            "clean_HEAD_equals_origin_oaci": True,
        },
        "views": {
            "source_training": {"X": True, "y": True, "level1_deletion_only_here": True},
            "source_audit": {"X": True, "y": True, "training": False, "metrics": False},
            "target_unlabeled": {"X": True, "y": False, "structural_y_slot_consumed": False},
            "target_construction": {"provisioned": False},
            "target_evaluation": {"provisioned": False},
            "same_label_oracle": {"reachable": False},
        },
        "instrumentation": {
            "checkpoint_optimizer_sidecar_units": 243,
            "strict_source_audit_artifacts": 243,
            "target_unlabeled_artifacts": 243,
            "linear_replay_abs_tolerance": runtime.LINEAR_REPLAY_ABS_TOLERANCE,
            "strict_identity_abs_tolerance": runtime.STRICT_IDENTITY_ABS_TOLERANCE,
            "persisted_reload_required": True,
            "support_and_intervention_replay_required": True,
            "paired_model_init_required": True,
            "accepted_level0_plan_replay_required": True,
        },
        "complete_gate": {
            "unique_units": 243,
            "checkpoint_replay_units": 243,
            "optimizer_replay_units": 243,
            "sidecar_replay_units": 243,
            "source_audit_replay_units": 243,
            "target_unlabeled_replay_units": 243,
            "target_y_access": 0,
            "target_scientific_metrics": 0,
            "partial_completion_reusable": False,
        },
        "runtime": {
            "external_root": str(runtime.DEFAULT_EXTERNAL_ROOT),
            "content_addressed_subroot": "lock_<execution_lock_sha256_prefix20>",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "resource_envelope_inherited_from_C84R3": True,
        },
        "authorization": {
            "record_path": str(runtime.AUTHORIZATION_RECORD_PATH.relative_to(REPO_ROOT)),
            "record_present_at_lock": False,
            "direct_PI_statement_required": "授权 C84L1C",
            "magic_token_required": False,
            "hash_recital_required": False,
            "fresh_binding_required": True,
            "C84C_authorization_reusable": False,
        },
        "attempt_ledger": {
            "created_after_authorization_consumption_before_protected_import": True,
            "all_stages_wrapped": True,
            "partial_manifest": True,
            "retry_requires_additive_repair_and_new_lock": True,
        },
        "retry_policy": {
            "same_lock_silent_rerun": False,
            "failed_attempts_preserved": True,
            "outcome_dependent_retry": False,
            "implementation_change_requires_new_lock": True,
        },
        "forbidden": {
            "real_execution_without_fresh_authorization": True,
            "target_y_or_label_like_metadata": True,
            "target_scientific_metrics": True,
            "level0_level1_target_performance_comparison": True,
            "alternative_deletion_cell": True,
            "target_specific_retraining": True,
            "C84F_or_C84S": True,
        },
    }


def generate(implementation_commit: str) -> dict[str, Any]:
    lock = build_lock(implementation_commit)
    LOCK_PATH.write_bytes(runtime.canonical_bytes(lock) + b"\n")
    digest = runtime.sha256_file(LOCK_PATH)
    LOCK_SHA_PATH.write_text(f"{digest}  {LOCK_PATH.name}\n", encoding="ascii")
    _write_csv(TABLE_DIR / "runtime_bound_object_registry.csv", [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "blob": row["blob"],
            "bytes": row["bytes"],
            "commit": row["commit"],
            "replay_before_data": 1,
        }
        for row in lock["runtime_bound_objects"]
    ])
    return {
        "execution_lock_sha256": digest,
        "runtime_bound_objects": lock["runtime_bound_object_count"],
        "implementation_files": len(lock["implementation"]["files"]),
        "canary_units": lock["scope"]["total_units"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate C84L1C execution lock")
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.implementation_commit), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
