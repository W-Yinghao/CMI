"""Generate the replacement C84L1C execution lock after C84L1R1."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import c84l1_lock as prior
from . import c84l1r1_runtime_repair as runtime


REPO_ROOT = prior.REPO_ROOT
REPORT_DIR = prior.REPORT_DIR
TABLE_DIR = REPORT_DIR / "c84l1r1_tables"
LOCK_PATH = REPORT_DIR / "C84L1C_EXECUTION_LOCK_V2.json"
LOCK_SHA_PATH = REPORT_DIR / "C84L1C_EXECUTION_LOCK_V2.sha256"
HISTORICAL_LOCK_PATH = REPORT_DIR / "C84L1C_EXECUTION_LOCK.json"
FAILED_REPORT_PATH = REPORT_DIR / "C84L1C_FAILED_ATTEMPT_895928.json"

NEW_IMPLEMENTATION_PATHS = (
    "oaci/multidataset/c84l1r1_runtime_repair.py",
    "oaci/multidataset/c84l1_canary_v2.py",
    "oaci/multidataset/c84l1r1_protocols.py",
    "oaci/multidataset/c84l1r1_lock.py",
    "oaci/slurm_c84l1c_canary_v2.sh",
)
NEW_REGISTRY_PATHS = (
    "oaci/reports/C84L1C_PI_AUTHORIZATION_RECORD.json",
    "oaci/reports/C84L1C_FAILED_ATTEMPT_895928.json",
    "oaci/reports/C84L1C_FAILED_ATTEMPT_895928.md",
    "oaci/reports/C84L1R1_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.json",
    "oaci/reports/C84L1R1_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.sha256",
    "oaci/reports/C84L1R1_PROTOCOL_TIMING_AUDIT.md",
    "oaci/reports/C84_LEVEL1_CANARY_PROTOCOL_V2.json",
    "oaci/reports/C84_LEVEL1_CANARY_PROTOCOL_V2.sha256",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V6.json",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V6.sha256",
    "oaci/reports/c84l1r1_tables/failed_attempt_ledger.csv",
    "oaci/reports/c84l1r1_tables/numerical_calibration.csv",
    "oaci/reports/c84l1r1_tables/repair_decision.csv",
)
PROTOCOL_STEMS = (
    "C84L1R1_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL",
    "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3",
    "C84_LEVEL1_CANARY_PROTOCOL_V2",
    "C84_FIELD_GENERATION_PROTOCOL_V6",
    "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3",
)


def _protocol_binding(stem: str) -> dict[str, str]:
    path = REPORT_DIR / f"{stem}.json"
    sidecar = REPORT_DIR / f"{stem}.sha256"
    digest = sidecar.read_text(encoding="ascii").split()[0]
    if runtime.sha256_file(path) != digest:
        raise RuntimeError(f"C84L1R1 protocol hash replay failed: {stem}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256_path": str(sidecar.relative_to(REPO_ROOT)),
        "sha256": digest,
    }


def _merge_bindings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        merged[str(row["path"])] = dict(row)
    return list(merged.values())


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise RuntimeError(f"refusing empty replacement lock table: {path}")
    fields = list(values[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def build_lock(implementation_commit: str) -> dict[str, Any]:
    if (REPORT_DIR / "C84L1C_PI_AUTHORIZATION_RECORD_V2.json").exists():
        raise RuntimeError("replacement authorization record must be absent at lock generation")
    lock = prior.build_lock(implementation_commit)
    extra_paths = list(NEW_IMPLEMENTATION_PATHS) + list(NEW_REGISTRY_PATHS)
    extras = [prior.bind_path(path, implementation_commit) for path in extra_paths]
    bound = _merge_bindings([*lock["runtime_bound_objects"], *extras])
    by_path = {row["path"]: row for row in bound}
    implementation_paths = list(dict.fromkeys([
        *[row["path"] for row in lock["implementation"]["files"]],
        *NEW_IMPLEMENTATION_PATHS,
    ]))
    protocols = [_protocol_binding(stem) for stem in PROTOCOL_STEMS]
    failed_report_sha = runtime.sha256_file(FAILED_REPORT_PATH)
    historical_lock_sha = runtime.sha256_file(HISTORICAL_LOCK_PATH)

    lock.update({
        "schema_version": "c84l1c_execution_lock_v2",
        "status": runtime.LOCK_READY_STATUS,
        "implementation_commit": implementation_commit,
        "repair_protocol": protocols[0],
        "external_protocol": protocols[1],
        "canary_protocol": protocols[2],
        "future_field_protocol": protocols[3],
        "future_scientific_protocol": protocols[4],
        "protocol_bindings": protocols,
        "runtime_bound_objects": bound,
        "runtime_bound_object_count": len(bound),
    })
    lock["chronology"].update({
        "C84L1C_failed_job": 895928,
        "C84L1R1_repair_protocol_commit": "e35ba0b",
        "C84L1R1_implementation_commit": implementation_commit,
        "replacement_real_EEG_access": 0,
        "replacement_target_y_access": 0,
        "replacement_scientific_metrics": 0,
    })
    lock["historical_C84L1C"] = {
        "execution_lock_path": str(HISTORICAL_LOCK_PATH.relative_to(REPO_ROOT)),
        "execution_lock_commit": "3eafd70795344c43e0c6326e5c190ecaea4c2934",
        "execution_lock_sha256": historical_lock_sha,
        "authorization_commit": "05bfca1",
        "authorization_consumed": True,
        "operative": False,
    }
    lock["historical_failed_attempt"] = {
        "job_id": 895928,
        "external_root": "/projects/EEG-foundation-model/yinghao/oaci-c84-level1-canary-v1/lock_d6ccab97ebfbb1e1d571",
        "partial_manifest_sha256": "ba67a4a0f8a516085b3eb020c353c401c2eafdd1981eb880c5c63587ac31b091",
        "report_path": str(FAILED_REPORT_PATH.relative_to(REPO_ROOT)),
        "report_sha256": failed_report_sha,
        "complete_units": 73,
        "partial_artifacts_reusable": False,
        "target_y_access": 0,
        "target_scientific_metrics": 0,
    }
    lock["implementation"] = {
        "commit": implementation_commit,
        "entrypoint": "python -m oaci.multidataset.c84l1_canary_v2 run-real",
        "slurm_entrypoint": "oaci/slurm_c84l1c_canary_v2.sh",
        "files": [by_path[path] for path in implementation_paths],
        "historical_ERM_OACI_SRC_formulas_unchanged": True,
    }
    lock["instrumentation"].update({
        "linear_replay_abs_tolerance": runtime.LINEAR_REPLAY_ABS_TOLERANCE,
        "strict_identity_abs_tolerance": runtime.STRICT_IDENTITY_ABS_TOLERANCE,
        "numerical_repair_scope": "float32_1040_term_CPU_GPU_classifier_reconstruction_only",
    })
    lock["runtime"].update({
        "external_root": str(runtime.DEFAULT_EXTERNAL_ROOT),
        "failed_external_root_reused": False,
    })
    lock["authorization"].update({
        "record_path": str(runtime.AUTHORIZATION_RECORD_PATH.relative_to(REPO_ROOT)),
        "record_present_at_lock": False,
        "fresh_binding_required": True,
        "historical_C84L1C_authorization_reusable": False,
    })
    lock["retry_policy"].update({
        "historical_failed_job": 895928,
        "failed_partial_artifacts_reusable": False,
        "replacement_retrain_units": 243,
        "new_content_addressed_root_required": True,
    })
    return lock


def generate(implementation_commit: str) -> dict[str, Any]:
    lock = build_lock(implementation_commit)
    LOCK_PATH.write_bytes(runtime.canonical_bytes(lock) + b"\n")
    digest = runtime.sha256_file(LOCK_PATH)
    LOCK_SHA_PATH.write_text(f"{digest}  {LOCK_PATH.name}\n", encoding="ascii")
    _write_csv(TABLE_DIR / "runtime_bound_object_registry.csv", [
        {
            "path": row["path"], "sha256": row["sha256"], "blob": row["blob"],
            "bytes": row["bytes"], "commit": row["commit"], "replay_before_data": 1,
        }
        for row in lock["runtime_bound_objects"]
    ])
    return {
        "execution_lock_v2_sha256": digest,
        "runtime_bound_objects": lock["runtime_bound_object_count"],
        "implementation_files": len(lock["implementation"]["files"]),
        "canary_units": lock["scope"]["total_units"],
        "failed_partial_artifacts_reusable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate repaired C84L1C execution lock")
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.implementation_commit), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
