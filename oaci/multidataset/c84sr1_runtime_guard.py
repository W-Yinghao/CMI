"""Fail-closed C84S V3 runtime guard and replacement-lock construction."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .c84s_common import (
    canonical_sha256, read_csv, read_json, require, sha256_file, write_csv,
    write_json,
)
from .c84sr1_common import (
    AUTHORIZATION_PATH, COMPLETE_FIELD_MANIFEST_PATH, LOCK_PATH, LOCK_SHA_PATH,
    PROTOCOL_PATH, PROTOCOL_SHA_PATH, REPORT_DIR, REPO_ROOT,
    TARGET_TRIAL_REGISTRY_PATH,
)


TABLE_DIR = REPORT_DIR / "c84sr1_tables"
RAW_MANIFEST_PATH = COMPLETE_FIELD_MANIFEST_PATH.parent / "C84F_TARGET_RAW_INPUT_MANIFEST.json"
MODEL_FIELD_MANIFEST_PATH = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-v1/"
    "lock_f9df9dcefea59b05bfea/C84F_MODEL_FIELD_MANIFEST.json"
)
DEFAULT_OUTPUT_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v3")
SYNTHETIC_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84sr1-production-synthetic-v1"
)
LOCK_READY_STATUS = "LOCKED_READY_FOR_FRESH_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
EXPECTED = {
    "repair": "3bdfbf67f1e1697a1488ccb5b7148494db06586ea9ff4318f16e030b88e7be2a",
    "complete_field": "cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8",
    "model_field": "d8931b81a3d68f4b1e098ac6e3ede3cd44cdb6c70cdef9f18a76e0a8c62ecdb2",
    "target_raw": "9539747e903dfe67295ee04a97441b85c0bb2179c9ef1bd2177788865e0ba5fd",
    "target_registry": "52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8",
    "science_v3": "bf6c7f718413b4b2ac2ad9786aa2e47dc045a536e7237d5d8c0464b6598130b8",
    "operationalization": "abf56676901bd7e5f484ffe4f4bb49de625d5c7b87cc34c0e3ab2bdb39361c5e",
    "end_to_end_repair": "b2d52a3bfdcb89b8a8db5d1a5501fb7b24a22ad860dbe1d6da2c5e6d77ca189c",
    "frontier": "020251ded9dbe4688ef08e9854875fc06b789120f4697c661b94f96eeef66fca",
    "method_registry": "ef48ecf7fcc55188b78b0878d86f07f6239fe4f6c88bbc854829b3a1c7a1a120",
    "selector_replay": "d589437e40812350eec44bdfbf1b75c52f10ef41e0e3ca5868e07844b0228e68",
    "historical_lock_v1": "e17e4da14b60ac77ca0ec8bec80a2ca249cda014baf5460cfd64627294f2047b",
    "historical_lock_v2": "94c896f0f00c53441095da6225f9ac574eb4a9baa904821a5dab3f11ea76f75c",
    "preflight_blocker": "2dc1ec7b5cc26c7abd605a03107497c0f8b81d2c1a179086c8f3f45b13d826d5",
}
PROTOCOL_INPUTS = {
    "repair": PROTOCOL_PATH,
    "science_v3": REPORT_DIR / "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3.json",
    "operationalization": REPORT_DIR / "C84S_ANALYSIS_OPERATIONALIZATION_PROTOCOL.json",
    "end_to_end_repair": REPORT_DIR / "C84SL_END_TO_END_RESULT_FREEZE_REPAIR_PROTOCOL.json",
    "frontier": REPORT_DIR / "C84SL_LABEL_FRONTIER_STABILITY_CLARIFICATION.json",
    "method_registry": REPORT_DIR / "C81_BASELINE_METHOD_REGISTRY.json",
    "selector_replay": REPORT_DIR / "c84p_tables/selector_registry_replay.csv",
    "historical_lock_v1": REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK.json",
    "historical_lock_v2": REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK_V2.json",
    "preflight_blocker": REPORT_DIR / "C84S_AUTHORIZED_PREFLIGHT_BLOCKER.json",
    "complete_field": COMPLETE_FIELD_MANIFEST_PATH,
    "model_field": MODEL_FIELD_MANIFEST_PATH,
    "target_raw": RAW_MANIFEST_PATH,
    "target_registry": TARGET_TRIAL_REGISTRY_PATH,
}
LOCKED_ENVIRONMENT = {
    "conda_prefix": "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact",
    "python": "3.13.7",
    "distributions": {
        "numpy": "2.3.3", "scipy": "1.16.2", "moabb": "1.5.0", "mne": "1.11.0",
    },
}
LOADER_SOURCE_IDENTITIES = (
    {"distribution": "moabb", "path": "moabb/datasets/Lee2019.py", "sha256": "a0234b81923fed15e4a221e011399f76a83873cd43d598ad5c8c71ba54678a6f"},
    {"distribution": "moabb", "path": "moabb/datasets/gigadb.py", "sha256": "42e2ef372762cb86aab11a886e1707675477ac776e0468448233de7a4ba71e32"},
    {"distribution": "moabb", "path": "moabb/datasets/physionet_mi.py", "sha256": "a8abe8097870d804a2d78f500f3c6820962c1c3402f53368e92e7a91068b84ba"},
    {"distribution": "moabb", "path": "moabb/paradigms/motor_imagery.py", "sha256": "f941a3f17c1bca4211045c28f7df3704c9d428ef689dff2410a478b5bf68651e"},
)
IMPLEMENTATION_PATHS = (
    "oaci/multidataset/c84s_common.py",
    "oaci/multidataset/c84s_label_views.py",
    "oaci/multidataset/c84s_selectors.py",
    "oaci/multidataset/c84s_q0_budget.py",
    "oaci/multidataset/c84s_evaluation.py",
    "oaci/multidataset/c84s_inference.py",
    "oaci/multidataset/c84s_taxonomy.py",
    "oaci/multidataset/c84s_analysis.py",
    "oaci/multidataset/c84sr1_common.py",
    "oaci/multidataset/c84sr1_context_enumerator.py",
    "oaci/multidataset/c84sr1_field_reader.py",
    "oaci/multidataset/c84sr1_q0_store.py",
    "oaci/multidataset/c84sr1_method_context_materialization.py",
    "oaci/multidataset/c84sr1_analysis.py",
    "oaci/multidataset/c84sr1_stage_a_labels.py",
    "oaci/multidataset/c84sr1_stage_b_selection.py",
    "oaci/multidataset/c84sr1_stage_c_evaluation.py",
    "oaci/multidataset/c84sr1_runtime_guard.py",
    "oaci/multidataset/c84sr1_execute.py",
    "oaci/multidataset/c84sr1_synthetic.py",
    "oaci/multidataset/c84sr1_readiness.py",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def verify_protocol_inputs() -> dict[str, str]:
    sidecar = PROTOCOL_SHA_PATH.read_text(encoding="ascii").split()[0]
    require(sidecar == EXPECTED["repair"] and sha256_file(PROTOCOL_PATH) == sidecar,
            "C84SR1 repair protocol identity drift")
    observed = {name: sha256_file(path) for name, path in PROTOCOL_INPUTS.items()}
    for name, expected in EXPECTED.items():
        require(observed[name] == expected, f"C84SR1 frozen input drift: {name}")
    manifest = read_json(COMPLETE_FIELD_MANIFEST_PATH)
    require(manifest["gate"] == "C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED",
            "C84F complete-field gate drift")
    require(len(manifest["field_descriptors"]) == 1944, "complete-field descriptor count drift")
    for field in ("target_construction_labels", "target_evaluation_labels", "same_label_oracle",
                  "selector_scores", "scientific_statistics"):
        require(manifest[field] == 0, f"protected complete-field counter is nonzero: {field}")
    blocker = read_json(PROTOCOL_INPUTS["preflight_blocker"])
    require(blocker["authorization"]["active_authorization_record_created"] is False and
            blocker["authorization"]["authorization_consumed"] is False and
            blocker["preflight"]["real_target_label_access"] == 0,
            "historical V2 blocker boundary drift")
    return observed


def external_artifact_rows(*, verify_bytes: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(COMPLETE_FIELD_MANIFEST_PATH)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    counts = {"target_artifact": 0, "target_sidecar": 0, "source_audit": 0, "training_sidecar": 0}
    keys = (
        ("target_artifact", "complete_target_unlabeled"),
        ("target_sidecar", "target_context_digest_index"),
        ("source_audit", "source_audit"),
        ("training_sidecar", "training_sidecar"),
    )
    total_bytes = 0
    seen: set[tuple[str, str]] = set()
    for descriptor in manifest["field_descriptors"]:
        unit_id = str(descriptor["unit_id"])
        for kind, key in keys:
            identity = descriptor[key]
            path = Path(identity["path"])
            require(path.is_file(), f"external analysis input absent: {unit_id}/{kind}")
            pair = (unit_id, kind)
            require(pair not in seen, "duplicate external analysis-input identity")
            seen.add(pair)
            expected = str(identity["sha256"])
            observed = sha256_file(path) if verify_bytes else expected
            require(observed == expected, f"external analysis-input SHA drift: {unit_id}/{kind}")
            size = path.stat().st_size
            total_bytes += size
            counts[kind] += 1
            rows.append({
                "unit_id": unit_id, "artifact_kind": kind, "path": str(path),
                "bytes": size, "expected_sha256": expected,
                "observed_sha256": observed, "replay_pass": 1,
                "verification_mode": "stream_sha256" if verify_bytes else "manifest_identity",
            })
    require(counts == {key: 1944 for key in counts} and len(rows) == 7776,
            "external analysis-input coverage drift")
    return rows, {
        "files": len(rows), "counts": counts, "bytes": total_bytes,
        "wall_seconds": time.monotonic() - started,
    }


def repository_object_registry(paths: Sequence[str] = IMPLEMENTATION_PATHS) -> list[dict[str, Any]]:
    head = _git("rev-parse", "HEAD")
    rows = []
    for relative in paths:
        path = REPO_ROOT / relative
        require(path.is_file(), f"C84SR1 implementation object absent: {relative}")
        rows.append({
            "path": relative, "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "blob": _git("rev-parse", f"HEAD:{relative}"),
            "implementation_commit": head,
        })
    return rows


def verify_bound_repository_objects(lock: Mapping[str, Any]) -> None:
    for identity in lock["runtime_bound_repository_objects"]:
        path = REPO_ROOT / identity["path"]
        require(path.is_file() and sha256_file(path) == identity["sha256"],
                f"bound implementation SHA drift: {identity['path']}")
        require(_git("rev-parse", f"HEAD:{identity['path']}") == identity["blob"],
                f"bound implementation Git blob drift: {identity['path']}")


def verify_lock_bound_readiness(lock: Mapping[str, Any]) -> dict[str, Any]:
    for relative, expected in lock["readiness_table_hashes"].items():
        path = REPO_ROOT / relative
        require(path.is_file() and sha256_file(path) == expected,
                f"C84SR1 readiness table drift: {relative}")
    synthetic = lock["production_path_synthetic_calibration"]
    path = Path(synthetic["summary_path"])
    require(path.is_file() and sha256_file(path) == synthetic["summary_sha256"],
            "C84SR1 full-scale synthetic summary drift")
    payload = read_json(path)
    require(payload["status"] == "PASS" and payload["full_scale_Q0_records"] == 9110448 and
            payload["full_scale_method_context_rows"] == 18608,
            "C84SR1 full-scale synthetic summary is incomplete")
    return {
        "readiness_tables": len(lock["readiness_table_hashes"]),
        "synthetic_summary_sha256": synthetic["summary_sha256"],
    }


def static_process_isolation_audit() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for relative in IMPLEMENTATION_PATHS:
        path = REPO_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        imported = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            str(node.module or "") for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        blocked = sorted(name for name in imported if name == "oaci.train" or name.startswith("oaci.train."))
        require(not blocked, f"training import in C84SR1 implementation: {relative}")
        checks.append({"path": relative, "check": "no_training_import", "pass": 1})
    stage_a = (REPO_ROOT / "oaci/multidataset/c84sr1_stage_a_labels.py").read_text(encoding="utf-8")
    stage_b = (REPO_ROOT / "oaci/multidataset/c84sr1_stage_b_selection.py").read_text(encoding="utf-8")
    stage_c = (REPO_ROOT / "oaci/multidataset/c84sr1_stage_c_evaluation.py").read_text(encoding="utf-8")
    require("c84s_selectors" not in stage_a and "complete_target_unlabeled" not in stage_a,
            "Stage A can reach candidate artifacts")
    require("evaluation_seal_path" not in stage_b and "c84s_evaluation" not in stage_b,
            "Stage B interface can reach evaluation")
    require("c84s_selectors" not in stage_c and "c84s_label_views" not in stage_c,
            "Stage C can mutate selection or reach construction provisioning")
    checks.extend([
        {"path": "c84sr1_stage_a_labels.py", "check": "candidate_artifacts_unavailable", "pass": 1},
        {"path": "c84sr1_stage_b_selection.py", "check": "evaluation_descriptor_unavailable", "pass": 1},
        {"path": "c84sr1_stage_c_evaluation.py", "check": "selector_and_construction_unavailable", "pass": 1},
    ])
    return checks


def verify_environment_and_loader_sources() -> dict[str, Any]:
    require(".".join(map(str, sys.version_info[:3])) == LOCKED_ENVIRONMENT["python"],
            "C84SR1 Python version drift")
    require(Path(sys.prefix).resolve() == Path(LOCKED_ENVIRONMENT["conda_prefix"]).resolve(),
            "C84SR1 environment prefix drift")
    versions = {}
    for distribution, expected in LOCKED_ENVIRONMENT["distributions"].items():
        observed = importlib.metadata.version(distribution)
        require(observed == expected, f"C84SR1 distribution version drift: {distribution}")
        versions[distribution] = observed
    sources = []
    for identity in LOADER_SOURCE_IDENTITIES:
        distribution = importlib.metadata.distribution(identity["distribution"])
        path = Path(distribution.locate_file(identity["path"]))
        require(path.is_file() and sha256_file(path) == identity["sha256"],
                f"C84SR1 loader source drift: {identity['path']}")
        sources.append({**identity, "observed_sha256": identity["sha256"]})
    return {"versions": versions, "loader_sources": sources}


def verify_clean_synced_branch() -> str:
    require(_git("status", "--porcelain") == "", "C84S V3 execution requires a clean worktree")
    require(_git("branch", "--show-current") == "oaci", "C84S V3 execution requires branch oaci")
    head = _git("rev-parse", "HEAD")
    require(head == _git("rev-parse", "origin/oaci"), "C84S V3 HEAD differs from origin/oaci")
    return head


def verify_lock_self() -> tuple[dict[str, Any], str]:
    require(LOCK_PATH.is_file() and LOCK_SHA_PATH.is_file(), "C84S V3 analysis lock absent")
    expected = LOCK_SHA_PATH.read_text(encoding="ascii").split()[0]
    observed = sha256_file(LOCK_PATH)
    require(observed == expected, "C84S V3 analysis-lock SHA drift")
    lock = read_json(LOCK_PATH)
    require(lock["status"] == LOCK_READY_STATUS, "C84S V3 lock is not authorization-ready")
    return lock, observed


def verify_authorization(lock: Mapping[str, Any], lock_sha: str, path: Path) -> dict[str, Any]:
    require(path.is_file(), "fresh C84S V3 PI authorization record absent")
    record = read_json(path)
    required = {
        "schema_version": "c84sr1_direct_pi_authorization_record_v1",
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84S",
        "repair_protocol_sha256": EXPECTED["repair"],
        "analysis_lock_sha256": lock_sha,
        "analysis_lock_commit": _git(
            "log", "-1", "--format=%H", "--", str(LOCK_PATH.relative_to(REPO_ROOT)),
        ),
        "complete_field_manifest_sha256": EXPECTED["complete_field"],
        "historical_V2_authorization_migrated": False,
        "training": False, "forward": False, "GPU": False,
        "same_label_oracle": False, "C85": False,
    }
    for key, expected in required.items():
        require(record.get(key) == expected, f"C84S V3 authorization binding drift: {key}")
    return {**record, "record_sha256": sha256_file(path)}


def pre_label_access_guard(
    *, authorization_path: Path, output_root: Path,
    verify_external_bytes: bool = True,
) -> dict[str, Any]:
    lock, lock_sha = verify_lock_self()
    head = verify_clean_synced_branch()
    protocols = verify_protocol_inputs()
    verify_bound_repository_objects(lock)
    readiness = verify_lock_bound_readiness(lock)
    environment = verify_environment_and_loader_sources()
    authorization = verify_authorization(lock, lock_sha, authorization_path)
    require(Path(output_root).resolve() == Path(lock["execution"]["output_root"]).resolve(),
            "C84S V3 output root differs from lock")
    require(not Path(output_root).exists(), "C84S V3 output root already exists")
    rows, external = external_artifact_rows(verify_bytes=verify_external_bytes)
    require(canonical_sha256([{key: row[key] for key in (
        "unit_id", "artifact_kind", "path", "bytes", "expected_sha256",
    )} for row in rows]) == lock["external_field"]["artifact_identity_sha256"],
            "C84S V3 external artifact registry identity drift")
    return {
        "head": head, "lock": lock, "lock_sha256": lock_sha,
        "authorization": authorization, "protocol_replay": protocols,
        "environment_replay": environment, "external_replay": external,
        "readiness_replay": readiness,
        "output_root": str(output_root),
    }


def consume_authorization(binding: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(binding["output_root"])
    require(not root.exists(), "C84S V3 run root exists before authorization consumption")
    root.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": "c84sr1_authorization_consumption_v1",
        "stage": "C84S_authorization_consumed",
        "authorized_stage": "C84S",
        "C84S_authorized": True,
        "analysis_lock_sha256": binding["lock_sha256"],
        "authorization_record_sha256": binding["authorization"]["record_sha256"],
        "consumed_at_unix_ns": time.time_ns(),
        "before_real_label_access": True,
        "before_stage_A": True,
        "target_construction_labels_accessed": 0,
        "target_evaluation_labels_accessed": 0,
        "selector_scores_computed": 0,
        "scientific_statistics_computed": 0,
    }
    from .c84sr1_common import write_stage_receipt
    path = root / "authorization_consumed.json"
    digest = write_stage_receipt(path, payload)
    return {**payload, "path": str(path), "sha256": digest}


def build_execution_lock(*, implementation_commit: str) -> dict[str, Any]:
    require(_git("rev-parse", "HEAD") == implementation_commit,
            "C84SR1 implementation commit is not current HEAD")
    require(not LOCK_PATH.exists() and not LOCK_SHA_PATH.exists(), "C84S V3 lock already exists")
    require(not AUTHORIZATION_PATH.exists(), "C84S V3 authorization exists before lock")
    verify_protocol_inputs()
    repository_rows = repository_object_registry()
    static_checks = static_process_isolation_audit()
    external_rows = read_csv(TABLE_DIR / "external_field_artifact_replay.csv")
    require(len(external_rows) == 7776 and all(row["replay_pass"] == "1" for row in external_rows),
            "C84SR1 external-field replay table incomplete")
    artifact_identity = canonical_sha256([{key: row[key] if key != "bytes" else int(row[key]) for key in (
        "unit_id", "artifact_kind", "path", "bytes", "expected_sha256",
    )} for row in external_rows])
    repository_sha = write_csv(TABLE_DIR / "runtime_bound_object_registry.csv", repository_rows)
    readiness_table_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in sorted(TABLE_DIR.glob("*.csv"))
    }
    synthetic_summary_path = SYNTHETIC_ROOT / "C84SR1_SYNTHETIC_CALIBRATION.json"
    require(synthetic_summary_path.is_file(), "C84SR1 full-scale synthetic summary absent")
    synthetic = read_json(synthetic_summary_path)
    require(synthetic["status"] == "PASS" and synthetic["full_scale_Q0_records"] == 9110448 and
            synthetic["full_scale_method_context_rows"] == 18608,
            "C84SR1 full-scale synthetic summary incomplete")
    lock = {
        "schema_version": "c84s_analysis_execution_lock_v3",
        "milestone": "C84SR1",
        "status": LOCK_READY_STATUS,
        "chronology": {
            "repair_protocol_commit": _git("log", "-1", "--format=%H", "--", str(PROTOCOL_PATH.relative_to(REPO_ROOT))),
            "implementation_commit": implementation_commit,
            "protocol_precedes_implementation": True,
            "target_label_access_at_lock": 0,
            "real_selector_scores_at_lock": 0,
            "scientific_statistics_at_lock": 0,
        },
        "protocol_hashes": verify_protocol_inputs(),
        "external_field": {
            "complete_manifest_path": str(COMPLETE_FIELD_MANIFEST_PATH),
            "complete_manifest_sha256": EXPECTED["complete_field"],
            "model_manifest_sha256": EXPECTED["model_field"],
            "target_raw_manifest_sha256": EXPECTED["target_raw"],
            "target_trial_registry_sha256": EXPECTED["target_registry"],
            "artifacts": 7776, "artifact_identity_sha256": artifact_identity,
            "readiness_byte_replay_sha256": sha256_file(TABLE_DIR / "external_field_artifact_replay.csv"),
        },
        "runtime_bound_repository_objects": repository_rows,
        "runtime_bound_object_registry": {
            "path": "oaci/reports/c84sr1_tables/runtime_bound_object_registry.csv",
            "sha256": repository_sha, "rows": len(repository_rows),
        },
        "readiness_table_hashes": readiness_table_hashes,
        "production_path_synthetic_calibration": {
            "root": str(SYNTHETIC_ROOT),
            "summary_path": str(synthetic_summary_path),
            "summary_sha256": sha256_file(synthetic_summary_path),
            "status": synthetic["status"],
            "contexts": synthetic["contexts"],
            "Q0_chains": synthetic["full_scale_Q0_chains"],
            "Q0_records": synthetic["full_scale_Q0_records"],
            "method_context_rows": synthetic["full_scale_method_context_rows"],
            "selection_freeze_sha256": synthetic["full_selection_manifest_sha256"],
            "result_sha256": synthetic["full_result_sha256"],
            "branch_results": synthetic["branch_results"],
            "precomputed_method_context_rows_injected": synthetic["precomputed_method_context_rows_injected"],
            "real_field_array_access": synthetic["real_field_array_access"],
            "real_target_label_access": synthetic["real_target_label_access"],
        },
        "environment": {**LOCKED_ENVIRONMENT, "GPU_required": False},
        "loader_sources": list(LOADER_SOURCE_IDENTITIES),
        "static_process_isolation_sha256": canonical_sha256(static_checks),
        "analysis_contract": {
            "contexts": 944, "candidate_score_rows": 535248,
            "candidate_rank_rows": 535248, "fixed_selection_rows": 4720,
            "Q0_records": 9110448, "Q0_shards": 944, "Q0_chains": 2048,
            "method_context_rows": 18608, "maxT_draws": 65536,
            "selection_freeze_before_evaluation": True,
            "measurement_applicability_flags": True,
            "context_catastrophic_field_removed": True,
            "atomic_stage_publication": True,
            "Q0_shard_schema": "c84sr1_q0_context_shard_v1",
            "selection_freeze_schema": "c84sr1_selection_freeze_manifest_v2",
            "method_context_schema": "c84sr1_method_context_v2",
            "result_schema": "c84sr1_result_v2",
            "result_table_registry_sha256": readiness_table_hashes[
                "oaci/reports/c84sr1_tables/result_table_registry.csv"
            ],
            "measurement_applicability_sha256": readiness_table_hashes[
                "oaci/reports/c84sr1_tables/measurement_applicability.csv"
            ],
        },
        "resource_envelope": {
            "RAM_GiB": 128, "CPU_workers": 32, "output_GiB": 40,
            "wall_hours": 48, "GPU": False,
        },
        "execution": {
            "module": "oaci.multidataset.c84sr1_execute",
            "output_root": str(DEFAULT_OUTPUT_ROOT),
            "fresh_root_required": True,
            "subprocess_stages": ["Stage_A", "Stage_B", "Stage_C"],
        },
        "authorization": {
            "record_path": str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)),
            "record_present_at_lock": False,
            "fresh_direct_statement": "授权 C84S",
            "historical_authorization_migrates": False,
        },
        "attempt_and_failure_policy": {
            "authorization_consumed_before_stage_A": True,
            "lifecycle_attempt_ledger": "C84S_LIFECYCLE_ATTEMPT.json",
            "per_stage_attempt_ledgers": True,
            "stage_B_failure_keeps_evaluation_sealed": True,
            "stage_C_failure_preserves_selection_freeze": True,
            "partial_final_root_publishable": False,
            "automatic_retry": False,
            "implementation_change_requires_additive_protocol_and_lock": True,
        },
        "historical_locks": {
            "V1_sha256": EXPECTED["historical_lock_v1"],
            "V2_sha256": EXPECTED["historical_lock_v2"],
            "V2_authorization_consumed": False,
            "operative": False,
        },
        "forbidden": {
            "training": True, "forward": True, "GPU": True,
            "same_label_oracle": True, "new_method": True, "retuning": True,
            "evaluation_before_selection_freeze": True, "C85": True,
        },
        "success_gate": "C84S_REAL_EXECUTION_ORCHESTRATION_Q0_INTEGRATION_REPAIRED_AND_LOCKED_READY_FOR_FRESH_PI_AUTHORIZATION",
        "failure_gate": "C84S_REAL_EXECUTION_SELECTION_AGGREGATION_RESOURCE_OR_PROVENANCE_RECONCILIATION_REQUIRED",
    }
    digest = write_json(LOCK_PATH, lock)
    LOCK_SHA_PATH.write_text(f"{digest}  {LOCK_PATH.name}\n", encoding="ascii")
    return {"lock": lock, "sha256": digest}
