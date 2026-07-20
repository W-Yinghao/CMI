"""Create the scope-specific C84F execution lock after implementation commit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from . import c84fl2_protocol as protocol


REPO_ROOT = protocol.REPO_ROOT
REPORT_DIR = protocol.REPORT_DIR
TABLE_DIR = protocol.TABLE_DIR
PROTOCOL_COMMIT = "24a21d795ccfaa8cfd816f77eaa9f41867fad847"
IMPLEMENTATION_COMMIT = "96af9f3751451f7def6d9b35b8ab395675e41394"
LOCK_PATH = REPORT_DIR / "C84F_EXECUTION_LOCK.json"
LOCK_SHA_PATH = REPORT_DIR / "C84F_EXECUTION_LOCK.sha256"
AUTHORIZATION_PATH = REPORT_DIR / "C84F_PI_AUTHORIZATION_RECORD.json"
LOCK_READY_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"


IMPLEMENTATION_PATHS = (
    "oaci/__init__.py",
    "oaci/config.py",
    "oaci/support_graph.py",
    "oaci/multidataset/__init__.py",
    "oaci/multidataset/c84_dataset_registry.py",
    "oaci/multidataset/c84_dataset_registry_v2.py",
    "oaci/multidataset/c84r_montage_repair.py",
    "oaci/multidataset/c84c_real_canary.py",
    "oaci/multidataset/c84c_real_canary_v2.py",
    "oaci/multidataset/c84r2_canary_runtime_repair.py",
    "oaci/multidataset/c84l1_protocols.py",
    "oaci/multidataset/c84l1_intervention.py",
    "oaci/multidataset/c84l1_canary.py",
    "oaci/multidataset/c84fl2_protocol.py",
    "oaci/multidataset/c84f_runtime_guard.py",
    "oaci/multidataset/c84f_dual_level_training.py",
    "oaci/multidataset/c84f_target_instrumentation.py",
    "oaci/multidataset/c84f_field_manifest.py",
    "oaci/models/__init__.py",
    "oaci/models/factory.py",
    "oaci/models/shallow.py",
    "oaci/models/output.py",
    "oaci/data/__init__.py",
    "oaci/data/plan_materialize.py",
    "oaci/data/plan_sampler.py",
    "oaci/methods/__init__.py",
    "oaci/methods/oaci.py",
    "oaci/methods/source_robust.py",
    "oaci/train/__init__.py",
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

PROTOCOL_PATHS = (
    "oaci/reports/C84FL2_DUAL_LEVEL_FULL_FIELD_RECONCILIATION_PROTOCOL.json",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V7.json",
    "oaci/reports/C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL_V2.json",
    "oaci/reports/C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3.json",
    "oaci/reports/C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3.json",
    "oaci/reports/C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.json",
    "oaci/reports/C84L1R1_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.json",
)

REGISTRY_PATHS = (
    "oaci/reports/C84FL2_PROTOCOL_TIMING_AUDIT.md",
    "oaci/reports/C84C_ENGINEERING_CANARY_RESULT.json",
    "oaci/reports/C84L1C_ENGINEERING_CANARY_RESULT.json",
    "oaci/reports/C84C_EXECUTION_LOCK_V3.json",
    "oaci/reports/C84L1C_EXECUTION_LOCK_V2.json",
    "oaci/reports/c84l1p_tables/operative_complete_unit_registry_v2.csv",
    "oaci/reports/c84l1p_tables/level_intervention_registry.csv",
    "oaci/reports/c84fl2_tables/c84c_result_identity_replay.csv",
    "oaci/reports/c84fl2_tables/c84l1c_result_identity_replay.csv",
    "oaci/reports/c84fl2_tables/historical_failed_root_rejection.csv",
    "oaci/reports/c84fl2_tables/dual_canary_reuse_registry.csv",
    "oaci/reports/c84fl2_tables/operative_complete_unit_registry_replay.csv",
    "oaci/reports/c84fl2_tables/remaining_paired_training_registry.csv",
    "oaci/reports/c84fl2_tables/wave_registry.csv",
    "oaci/reports/c84fl2_tables/level_intervention_replay.csv",
    "oaci/reports/c84fl2_tables/paired_rng_plan_contract.csv",
    "oaci/reports/c84fl2_tables/source_view_contract.csv",
    "oaci/reports/c84fl2_tables/model_field_manifest_schema.csv",
    "oaci/reports/c84fl2_tables/target_unlabeled_trial_registry_schema.csv",
    "oaci/reports/c84fl2_tables/target_instrumentation_schema.csv",
    "oaci/reports/c84fl2_tables/canary_subset_replay_contract.csv",
    "oaci/reports/c84fl2_tables/field_unit_descriptor_schema.csv",
    "oaci/reports/c84fl2_tables/retry_policy.csv",
    "oaci/reports/c84fl2_tables/resource_estimate.csv",
    "oaci/reports/c84fl2_tables/synthetic_calibration.csv",
    "oaci/reports/c84fl2_tables/risk_register.csv",
    "oaci/reports/c84fl2_tables/failure_reason_ledger.csv",
)


class C84FL2LockError(RuntimeError):
    """Raised when the C84F lock cannot bind an exact committed object."""


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments), cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


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


def _sidecar_digest(path: Path) -> str:
    values = path.read_text(encoding="ascii").split()
    if not values or len(values[0]) != 64:
        raise C84FL2LockError(f"malformed protocol sidecar: {path}")
    return values[0]


def bind_path(relative: str, commit: str = IMPLEMENTATION_COMMIT) -> dict[str, Any]:
    path = REPO_ROOT / relative
    if not path.is_file():
        raise C84FL2LockError(f"lock-bound file is absent: {relative}")
    blob = _git("rev-parse", f"{commit}:{relative}", check=False)
    if blob.returncode:
        raise C84FL2LockError(f"lock-bound file is not present at implementation commit: {relative}")
    blob_id = blob.stdout.strip()
    current_blob = _git("hash-object", str(path)).stdout.strip()
    if current_blob != blob_id:
        raise C84FL2LockError(f"current bytes differ from implementation commit: {relative}")
    return {
        "path": relative, "commit": commit, "blob": blob_id,
        "sha256": sha256_file(path), "bytes": path.stat().st_size,
    }


def protocol_binding(relative: str) -> dict[str, str]:
    path = REPO_ROOT / relative
    sidecar = path.with_suffix(".sha256")
    digest = sha256_file(path)
    if not sidecar.is_file() or _sidecar_digest(sidecar) != digest:
        raise C84FL2LockError(f"protocol hash sidecar does not replay: {relative}")
    return {
        "path": relative, "sha256_path": str(sidecar.relative_to(REPO_ROOT)), "sha256": digest,
    }


def _protocol_map(bindings: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    by_name = {Path(row["path"]).name: dict(row) for row in bindings}
    names = {
        "reconciliation": "C84FL2_DUAL_LEVEL_FULL_FIELD_RECONCILIATION_PROTOCOL.json",
        "field_v7": "C84_FIELD_GENERATION_PROTOCOL_V7.json",
        "full_field_v2": "C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL_V2.json",
        "external_v3": "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3.json",
        "science_v3": "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3.json",
        "level1_intervention": "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.json",
        "level1_numerical_repair": "C84L1R1_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.json",
    }
    return {key: by_name[value] for key, value in names.items()}


def build_lock() -> dict[str, Any]:
    if AUTHORIZATION_PATH.exists():
        raise C84FL2LockError("C84F authorization record must be absent at lock generation")
    if (REPORT_DIR / "C84S_EXECUTION_LOCK.json").exists():
        raise C84FL2LockError("C84S execution lock must not exist in C84FL2")
    if _git("merge-base", "--is-ancestor", PROTOCOL_COMMIT, IMPLEMENTATION_COMMIT, check=False).returncode:
        raise C84FL2LockError("protocol commit is not an ancestor of implementation commit")

    protocol_bindings = [protocol_binding(path) for path in PROTOCOL_PATHS]
    sidecars = [str((REPO_ROOT / path).with_suffix(".sha256").relative_to(REPO_ROOT)) for path in PROTOCOL_PATHS]
    runtime_paths = list(dict.fromkeys((*IMPLEMENTATION_PATHS, *PROTOCOL_PATHS, *sidecars, *REGISTRY_PATHS)))
    runtime_objects = [bind_path(path) for path in runtime_paths]
    implementation_files = [row for row in runtime_objects if row["path"] in IMPLEMENTATION_PATHS]

    operative_path = TABLE_DIR / "operative_complete_unit_registry_replay.csv"
    operative = read_csv(operative_path)
    unit_ids = sorted(row["unit_id"] for row in operative)
    if len(unit_ids) != 1944 or len(set(unit_ids)) != 1944:
        raise C84FL2LockError("operative registry is not 1,944 unique units")
    unit_digest = hashlib.sha256(canonical_bytes(unit_ids)).hexdigest()

    reuse_path = TABLE_DIR / "dual_canary_reuse_registry.csv"
    reuse = read_csv(reuse_path)
    if len(reuse) != 486 or len({row["unit_id"] for row in reuse}) != 486:
        raise C84FL2LockError("dual-canary reuse registry is not 486 unique units")
    artifact_files = sum(5 for _ in reuse)
    artifact_digest = hashlib.sha256(canonical_bytes(reuse)).hexdigest()

    prior_lock = read_json(REPORT_DIR / "C84L1C_EXECUTION_LOCK_V2.json")
    resources = read_csv(TABLE_DIR / "resource_estimate.csv")
    if not all(int(row["within_envelope"]) for row in resources):
        raise C84FL2LockError("resource estimate exceeds a hard envelope")
    tables = {
        path.name: sha256_file(path)
        for path in sorted(TABLE_DIR.glob("*.csv"))
        if path.name != "runtime_bound_object_registry.csv"
    }
    protocols = _protocol_map(protocol_bindings)
    return {
        "schema_version": "c84f_execution_lock_v1",
        "status": LOCK_READY_STATUS,
        "chronology": {
            "base_HEAD": protocol.BASE_HEAD,
            "protocol_commit": PROTOCOL_COMMIT,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "protocol_precedes_implementation": True,
            "lock_generated_after_implementation": True,
            "C84FL2_real_data_access": 0,
            "C84FL2_training_forward_GPU": 0,
            "target_labels_before_lock": 0,
            "scientific_outcomes_before_lock": 0,
        },
        "protocols": protocols,
        "protocol_bindings": protocol_bindings,
        "implementation": {
            "commit": IMPLEMENTATION_COMMIT,
            "entrypoint": "python -m oaci.multidataset.c84f_dual_level_training run-real",
            "files": implementation_files,
            "file_count": len(implementation_files),
            "parameterized_paired_cell_function": "train_paired_cell",
            "target_stage_training_callable": False,
        },
        "runtime_bound_objects": runtime_objects,
        "runtime_bound_object_count": len(runtime_objects),
        "environment": prior_lock["environment"],
        "loader_source_identity": prior_lock["loader_source_identity"],
        "interface": {
            "id": protocol.INTERFACE_ID,
            "channels": list(prior_lock["interface"]["channels"]),
            "montage_sha256": protocol.HASHES["montage"],
            "sample_rate_hz": 160, "sample_count": 480,
            "linear_replay_abs_tolerance": protocol.LINEAR_TOLERANCE,
            "strict_identity_abs_tolerance": protocol.STRICT_TOLERANCE,
            "Fz_substitution": False, "FCz_interpolation": False,
            "zero_fill": False, "dataset_specific_mask": False,
        },
        "candidate_identity": {
            "registry_path": str(operative_path.relative_to(REPO_ROOT)),
            "registry_sha256": sha256_file(operative_path),
            "unit_id_digest": unit_digest, "unit_count": 1944,
            "level0_units": 972, "level1_units": 972,
            "historical_superseded_level1_ids_operative": False,
        },
        "dual_canary_reuse": {
            "registry_path": str(reuse_path.relative_to(REPO_ROOT)),
            "registry_sha256": sha256_file(reuse_path),
            "registry_content_sha256": artifact_digest,
            "units": 486, "C84C_units": 243, "C84L1C_units": 243,
            "artifact_files_replayed_before_data": artifact_files,
            "C84C_manifest_path": str(protocol.C84C_MANIFEST),
            "C84C_manifest_sha256": protocol.HASHES["c84c_manifest"],
            "C84L1C_manifest_path": str(protocol.C84L1C_MANIFEST),
            "C84L1C_manifest_sha256": protocol.HASHES["c84l1c_manifest"],
            "canary_target_artifacts": "subset_replay_witnesses_only",
            "forbidden_failed_roots": [
                "/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v3/lock_2e38dcd63c02a887b1dc",
                "/projects/EEG-foundation-model/yinghao/oaci-c84-level1-canary-v1/lock_d6ccab97ebfbb1e1d571",
            ],
        },
        "scope": {
            "C84F": True, "C84S": False, "real_execution_at_lock": False,
            "candidate_units": 1944, "reused_units": 486, "new_units": 1458,
            "training_phases": 72, "target_contexts": 944,
            "candidate_context_slices": 76464,
        },
        "waves": {
            "A": {"panel": "A", "seed": 6, "levels": [0, 1], "dataset_jobs": 3,
                  "units": 486, "phases": 18, "order": 1},
            "B0": {"panel": "B", "seed": 5, "levels": [0, 1], "dataset_jobs": 3,
                   "units": 486, "phases": 18, "order": 2},
            "B1": {"panel": "B", "seed": 6, "levels": [0, 1], "dataset_jobs": 3,
                   "units": 486, "phases": 18, "order": 3},
            "release_evidence": "engineering_only_no_target_arrays_predictions_or_outcomes",
        },
        "barriers": {
            "source_views_only_before_model_freeze": True,
            "new_target_loader_call_before_model_manifest": False,
            "model_field_gate": {
                "units": 1944, "phases": 72, "checkpoints": 1944,
                "optimizers": 1944, "sidecars": 1944, "source_audit_artifacts": 1944,
                "training_target_rows": 0, "training_target_labels": 0,
            },
            "complete_target_gate": {
                "target_subjects": 118, "target_contexts": 944,
                "candidate_context_slices": 76464, "all_target_artifacts": 1944,
                "canary_contexts": 6, "canary_unit_witnesses": 486,
            },
        },
        "schemas": {
            "model_field": tables["model_field_manifest_schema.csv"],
            "target_trial_registry": tables["target_unlabeled_trial_registry_schema.csv"],
            "target_instrumentation": tables["target_instrumentation_schema.csv"],
            "field_descriptor": tables["field_unit_descriptor_schema.csv"],
            "complete_table_hashes": tables,
        },
        "numerical_gates": {
            "linear_in_memory_and_persisted_abs_max": 2e-5,
            "softmax_repeat_logits_repeat_z_abs_max": 1e-6,
            "runtime_widening_allowed": False,
        },
        "resources": {
            "table_path": "oaci/reports/c84fl2_tables/resource_estimate.csv",
            "table_sha256": tables["resource_estimate.csv"],
            "all_envelopes_pass": True,
            "GPU_phase_hours_max": 250, "external_payload_bytes_max": 2 * 1024**4,
            "Git_file_bytes_max": 50 * 1024**2,
        },
        "retry": {
            "policy_path": "oaci/reports/c84fl2_tables/retry_policy.csv",
            "policy_sha256": tables["retry_policy.csv"],
            "training_retry_requires_same_bytes_rows_IDs_RNG_and_no_target_access": True,
            "target_failure_can_invoke_training": False,
            "numerical_tolerance_widening": False,
        },
        "authorization": {
            "record_path": str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)),
            "record_present_at_lock": False,
            "fresh_direct_statement": "授权 C84F",
            "magic_token_required": False, "hash_recital_required": False,
            "C84S_authorized": False,
        },
        "external_roots": {
            "C84F_base": str(runtime_root()),
            "C84C_valid": str(protocol.C84C_ROOT),
            "C84L1C_valid": str(protocol.C84L1C_ROOT),
            "failed_roots_reusable": False,
        },
        "forbidden": {
            "target_labels": True, "target_construction_or_evaluation_view": True,
            "same_label_oracle": True, "selector_scores": True,
            "Q1_Q2_or_label_budget": True, "scientific_inference": True,
            "C84S": True, "training_after_target_instrumentation_failure": True,
            "runtime_scope_reduction": True, "runtime_tolerance_widening": True,
            "raw_EEG_weights_optimizer_states_or_caches_in_Git": True,
        },
        "field_completion_gate": protocol.FIELD_GATE,
    }


def runtime_root() -> Path:
    return Path("/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-v1")


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")
    return sha256_file(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise C84FL2LockError(f"refusing empty lock table: {path}")
    fields = list(values[0])
    if any(set(row) != set(fields) for row in values):
        raise C84FL2LockError(f"lock table schema drift: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def generate() -> dict[str, Any]:
    lock = build_lock()
    digest = write_json(LOCK_PATH, lock)
    LOCK_SHA_PATH.write_text(f"{digest}  {LOCK_PATH.name}\n", encoding="ascii")
    write_csv(TABLE_DIR / "runtime_bound_object_registry.csv", [{
        "path": row["path"], "object_class": "implementation" if row["path"] in IMPLEMENTATION_PATHS else "registry",
        "commit": row["commit"], "blob": row["blob"], "sha256": row["sha256"],
        "bytes": row["bytes"], "runtime_replay_required": 1,
    } for row in lock["runtime_bound_objects"]])
    return {
        "execution_lock_sha256": digest,
        "runtime_bound_objects": lock["runtime_bound_object_count"],
        "implementation_files": lock["implementation"]["file_count"],
        "candidate_units": lock["scope"]["candidate_units"],
        "reused_units": lock["scope"]["reused_units"],
        "authorization_record_present": AUTHORIZATION_PATH.exists(),
        "status": lock["status"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the C84F full-field execution lock")
    parser.parse_args(argv)
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
