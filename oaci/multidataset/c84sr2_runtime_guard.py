"""Fail-closed C84S V4 runtime guard and execution-lock builder."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .c84s_common import (
    canonical_sha256, read_csv, read_json, require, sha256_file, write_csv,
    write_json,
)
from .c84sr1_common import COMPLETE_FIELD_MANIFEST_PATH
from .c84sr1_context_enumerator import (
    LEVEL_INTERVENTION_IDS, enumerate_contexts, resolve_level_intervention_id,
)
from .c84sr1_runtime_guard import (
    IMPLEMENTATION_PATHS as SR1_IMPLEMENTATION_PATHS,
    LOCKED_ENVIRONMENT, LOADER_SOURCE_IDENTITIES, external_artifact_rows,
    verify_environment_and_loader_sources, verify_protocol_inputs as verify_sr1_protocol_inputs,
)
from .c84sr2_common import (
    AUTHORIZATION_PATH, DEFAULT_OUTPUT_ROOT, FAILURE_GATE,
    HISTORICAL_STAGE_A_ROOT, HISTORICAL_V3_AUTHORIZATION_SHA256,
    HISTORICAL_V3_LOCK_SHA256, HISTORICAL_V3_ROOT, LOCK_PATH, LOCK_READY_STATUS,
    LOCK_SHA_PATH, PROTOCOL_PATH, PROTOCOL_SHA_PATH, REPAIR_PROTOCOL_SHA256,
    REPO_ROOT, SUCCESS_GATE, SYNTHETIC_ROOT, TABLE_DIR,
)
from .c84sr2_stage_a_replay import replay_historical_stage_a


IMPLEMENTATION_PATHS = tuple(dict.fromkeys((*SR1_IMPLEMENTATION_PATHS,
    "oaci/multidataset/c84sr2_common.py",
    "oaci/multidataset/c84sr2_stage_a_replay.py",
    "oaci/multidataset/c84sr2_runtime_guard.py",
    "oaci/multidataset/c84sr2_execute.py",
    "oaci/multidataset/c84sr2_synthetic.py",
    "oaci/multidataset/c84sr2_readiness.py",
)))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def verify_protocol_inputs() -> dict[str, str]:
    sidecar = PROTOCOL_SHA_PATH.read_text(encoding="ascii").split()[0]
    require(sidecar == REPAIR_PROTOCOL_SHA256 and sha256_file(PROTOCOL_PATH) == sidecar,
            "C84SR2 repair protocol identity drift")
    sr1 = verify_sr1_protocol_inputs()
    historical_lock = REPO_ROOT / "oaci/reports/C84S_ANALYSIS_EXECUTION_LOCK_V3.json"
    historical_auth = REPO_ROOT / "oaci/reports/C84S_V3_PI_AUTHORIZATION_RECORD.json"
    require(sha256_file(historical_lock) == HISTORICAL_V3_LOCK_SHA256,
            "historical V3 lock identity drift")
    require(sha256_file(historical_auth) == HISTORICAL_V3_AUTHORIZATION_SHA256,
            "historical V3 authorization identity drift")
    stage_a = replay_historical_stage_a()
    return {
        "repair": sidecar, "historical_V3_lock": HISTORICAL_V3_LOCK_SHA256,
        "historical_V3_authorization": HISTORICAL_V3_AUTHORIZATION_SHA256,
        "historical_stage_A_complete": stage_a["file_identities"][
            "stage_a_labels/C84S_STAGE_A_COMPLETE.json"
        ],
        **{f"V3_{key}": value for key, value in sr1.items()},
    }


def field_descriptor_compatibility_rows() -> list[dict[str, Any]]:
    manifest = read_json(COMPLETE_FIELD_MANIFEST_PATH)
    require(len(manifest["field_descriptors"]) == 1944, "field descriptor count drift")
    rows: list[dict[str, Any]] = []
    for raw in manifest["field_descriptors"]:
        sidecar_path = Path(raw["training_sidecar"]["path"])
        require(sidecar_path.is_file() and
                sha256_file(sidecar_path) == raw["training_sidecar"]["sha256"],
                "training-sidecar identity drift during compatibility audit")
        sidecar = read_json(sidecar_path)
        intervention = resolve_level_intervention_id(raw, sidecar)
        field_present = "level_intervention_id" in sidecar
        rows.append({
            "unit_id": str(raw["unit_id"]), "dataset": str(sidecar["dataset"]),
            "panel": str(sidecar["panel"]), "training_seed": int(sidecar["seed"]),
            "level": int(sidecar["level"]), "regime": str(sidecar["regime"]),
            "trajectory_order": int(sidecar["trajectory_order"]),
            "field_descriptor_intervention_id": str(raw["level_intervention_id"]),
            "sidecar_intervention_present": int(field_present),
            "resolved_intervention_id": intervention,
            "model_reuse_provenance": str(raw.get("model_reuse_provenance", "")),
            "compatibility_scope": "EXACT_HISTORICAL_C84C" if not field_present else "NATIVE_SIDECAR",
            "pass": 1,
        })
    require(len(rows) == 1944 and len({row["unit_id"] for row in rows}) == 1944,
            "compatibility audit unit coverage drift")
    missing = [row for row in rows if row["sidecar_intervention_present"] == 0]
    require(len(missing) == 243, "historical sidecar omission count drift")
    require(all(
        row["model_reuse_provenance"] == "C84C" and row["panel"] == "A" and
        row["training_seed"] == 5 and row["level"] == 0 and
        row["resolved_intervention_id"] == LEVEL_INTERVENTION_IDS[0]
        for row in missing
    ), "historical compatibility scope expanded beyond exact C84C units")
    require(sum(row["level"] == 0 for row in rows) == 972 and
            sum(row["level"] == 1 for row in rows) == 972,
            "level unit arithmetic drift")
    contexts = enumerate_contexts()
    require(len(contexts) == 944 and all(len(context.candidates) == 81 for context in contexts),
            "real descriptor context enumeration drift")
    return rows


def repository_object_registry(paths: Sequence[str] = IMPLEMENTATION_PATHS) -> list[dict[str, Any]]:
    head = _git("rev-parse", "HEAD")
    rows = []
    for relative in paths:
        path = REPO_ROOT / relative
        require(path.is_file(), f"C84SR2 implementation object absent: {relative}")
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
                f"V4 bound implementation SHA drift: {identity['path']}")
        require(_git("rev-parse", f"HEAD:{identity['path']}") == identity["blob"],
                f"V4 bound implementation Git blob drift: {identity['path']}")


def static_process_isolation_audit() -> list[dict[str, Any]]:
    rows = []
    for relative in IMPLEMENTATION_PATHS:
        path = REPO_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        imports = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            str(node.module or "") for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        require(not any(name == "oaci.train" or name.startswith("oaci.train.") for name in imports),
                f"training import in C84SR2 implementation: {relative}")
        rows.append({"path": relative, "check": "no_training_import", "pass": 1})
    replay_source = (REPO_ROOT / "oaci/multidataset/c84sr2_stage_a_replay.py").read_text(encoding="utf-8")
    require("c84s_label_views" not in replay_source and "provision_real_label_views" not in replay_source,
            "Stage-A replay can reload labels")
    rows.append({
        "path": "oaci/multidataset/c84sr2_stage_a_replay.py",
        "check": "immutable_replay_has_no_label_loader", "pass": 1,
    })
    return rows


def verify_clean_synced_branch() -> str:
    require(_git("status", "--porcelain") == "", "C84S V4 execution requires a clean worktree")
    require(_git("branch", "--show-current") == "oaci", "C84S V4 requires branch oaci")
    head = _git("rev-parse", "HEAD")
    require(head == _git("rev-parse", "origin/oaci"), "C84S V4 HEAD differs from origin/oaci")
    return head


def verify_lock_self() -> tuple[dict[str, Any], str]:
    require(LOCK_PATH.is_file() and LOCK_SHA_PATH.is_file(), "C84S V4 lock absent")
    expected = LOCK_SHA_PATH.read_text(encoding="ascii").split()[0]
    require(sha256_file(LOCK_PATH) == expected, "C84S V4 lock SHA drift")
    lock = read_json(LOCK_PATH)
    require(lock["status"] == LOCK_READY_STATUS, "C84S V4 lock is not authorization-ready")
    return lock, expected


def verify_authorization(lock: Mapping[str, Any], lock_sha: str, path: Path) -> dict[str, Any]:
    require(path.is_file(), "fresh C84S V4 PI authorization record absent")
    record = read_json(path)
    required = {
        "schema_version": "c84sr2_direct_pi_authorization_record_v1",
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84S",
        "repair_protocol_sha256": REPAIR_PROTOCOL_SHA256,
        "analysis_lock_sha256": lock_sha,
        "analysis_lock_commit": _git(
            "log", "-1", "--format=%H", "--", str(LOCK_PATH.relative_to(REPO_ROOT)),
        ),
        "historical_V3_authorization_reused": False,
        "historical_stage_A_reused_without_label_reload": True,
        "training": False, "forward": False, "GPU": False,
        "same_label_oracle": False, "C85": False,
    }
    for key, expected in required.items():
        require(record.get(key) == expected, f"C84S V4 authorization binding drift: {key}")
    return {**record, "record_sha256": sha256_file(path)}


def verify_lock_bound_readiness(lock: Mapping[str, Any]) -> dict[str, Any]:
    for relative, expected in lock["readiness_table_hashes"].items():
        path = REPO_ROOT / relative
        require(path.is_file() and sha256_file(path) == expected,
                f"C84SR2 readiness table drift: {relative}")
    synthetic = lock["production_path_synthetic_calibration"]
    path = Path(synthetic["summary_path"])
    require(path.is_file() and sha256_file(path) == synthetic["summary_sha256"],
            "C84SR2 synthetic summary drift")
    payload = read_json(path)
    require(payload["status"] == "PASS" and payload["full_scale_Q0_records"] == 9110448 and
            payload["full_scale_method_context_rows"] == 18608,
            "C84SR2 full-scale synthetic incomplete")
    return {"tables": len(lock["readiness_table_hashes"]), "synthetic": synthetic["summary_sha256"]}


def pre_label_access_guard(
    *, authorization_path: Path, output_root: Path, verify_external_bytes: bool = True,
) -> dict[str, Any]:
    lock, lock_sha = verify_lock_self()
    head = verify_clean_synced_branch()
    protocols = verify_protocol_inputs()
    verify_bound_repository_objects(lock)
    readiness = verify_lock_bound_readiness(lock)
    environment = verify_environment_and_loader_sources()
    authorization = verify_authorization(lock, lock_sha, authorization_path)
    require(output_root.resolve() == Path(lock["execution"]["output_root"]).resolve(),
            "C84S V4 output root differs from lock")
    require(not output_root.exists(), "C84S V4 output root already exists")
    rows, external = external_artifact_rows(verify_bytes=verify_external_bytes)
    identity = canonical_sha256([{key: row[key] for key in (
        "unit_id", "artifact_kind", "path", "bytes", "expected_sha256",
    )} for row in rows])
    require(identity == lock["external_field"]["artifact_identity_sha256"],
            "C84S V4 external artifact identity drift")
    return {
        "head": head, "lock": lock, "lock_sha256": lock_sha,
        "protocol_replay": protocols, "readiness_replay": readiness,
        "environment_replay": environment, "authorization": authorization,
        "external_replay": external, "output_root": str(output_root),
    }


def consume_authorization(binding: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(binding["output_root"])
    require(not root.exists(), "C84S V4 root exists before authorization consumption")
    root.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": "c84sr2_authorization_consumption_v1",
        "stage": "C84S_V4_authorization_consumed", "authorized_stage": "C84S",
        "C84S_authorized": True, "analysis_lock_sha256": binding["lock_sha256"],
        "authorization_record_sha256": binding["authorization"]["record_sha256"],
        "consumed_at_unix_ns": time.time_ns(), "before_stage_A_replay": True,
        "historical_V3_authorization_reused": False,
        "target_label_rows_reloaded": 0, "target_evaluation_labels_accessed": 0,
        "selector_scores_computed": 0, "scientific_statistics_computed": 0,
    }
    from .c84sr1_common import write_stage_receipt
    path = root / "authorization_consumed.json"
    digest = write_stage_receipt(path, payload)
    return {**payload, "path": str(path), "sha256": digest}


def build_execution_lock(*, implementation_commit: str) -> dict[str, Any]:
    require(_git("rev-parse", "HEAD") == implementation_commit,
            "C84SR2 implementation commit is not current HEAD")
    require(not LOCK_PATH.exists() and not LOCK_SHA_PATH.exists(), "C84S V4 lock already exists")
    require(not AUTHORIZATION_PATH.exists(), "C84S V4 authorization exists before lock")
    protocols = verify_protocol_inputs()
    rows = repository_object_registry()
    static = static_process_isolation_audit()
    compatibility = read_csv(TABLE_DIR / "field_descriptor_compatibility_audit.csv")
    require(len(compatibility) == 1944 and
            sum(row["sidecar_intervention_present"] == "0" for row in compatibility) == 243,
            "C84SR2 compatibility readiness table incomplete")
    external = read_csv(TABLE_DIR / "external_field_artifact_replay.csv")
    require(len(external) == 7776 and all(row["replay_pass"] == "1" for row in external),
            "C84SR2 external-field table incomplete")
    artifact_identity = canonical_sha256([{key: row[key] if key != "bytes" else int(row[key]) for key in (
        "unit_id", "artifact_kind", "path", "bytes", "expected_sha256",
    )} for row in external])
    registry_sha = write_csv(TABLE_DIR / "runtime_bound_object_registry.csv", rows)
    table_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in sorted(TABLE_DIR.glob("*.csv"))
    }
    summary_path = SYNTHETIC_ROOT / "C84SR2_SYNTHETIC_CALIBRATION.json"
    require(summary_path.is_file(), "C84SR2 synthetic summary absent")
    synthetic = read_json(summary_path)
    require(synthetic["status"] == "PASS" and synthetic["full_scale_Q0_records"] == 9110448 and
            synthetic["full_scale_method_context_rows"] == 18608,
            "C84SR2 full-scale synthetic incomplete")
    stage_a = replay_historical_stage_a()
    lock = {
        "schema_version": "c84s_analysis_execution_lock_v4", "milestone": "C84SR2",
        "status": LOCK_READY_STATUS,
        "chronology": {
            "repair_protocol_commit": _git("log", "-1", "--format=%H", "--", str(PROTOCOL_PATH.relative_to(REPO_ROOT))),
            "implementation_commit": implementation_commit,
            "protocol_precedes_implementation": True,
            "historical_V3_authorization_consumed": True,
            "historical_V3_authorization_reusable": False,
            "evaluation_label_access_at_lock": 0, "selector_scores_at_lock": 0,
            "scientific_statistics_at_lock": 0,
        },
        "protocol_hashes": protocols,
        "historical_stage_A": {
            **stage_a, "reuse_mode": "IMMUTABLE_REPLAY_ONLY",
            "label_reload_allowed": False,
        },
        "field_descriptor_compatibility": {
            "units": 1944, "native_sidecars": 1701,
            "historical_C84C_omissions": 243, "contexts": 944,
            "table_sha256": table_hashes[
                "oaci/reports/c84sr2_tables/field_descriptor_compatibility_audit.csv"
            ],
        },
        "external_field": {
            "artifacts": 7776, "artifact_identity_sha256": artifact_identity,
            "complete_manifest_sha256": protocols["V3_complete_field"],
        },
        "runtime_bound_repository_objects": rows,
        "runtime_bound_object_registry": {
            "path": "oaci/reports/c84sr2_tables/runtime_bound_object_registry.csv",
            "sha256": registry_sha, "rows": len(rows),
        },
        "readiness_table_hashes": table_hashes,
        "production_path_synthetic_calibration": {
            "root": str(SYNTHETIC_ROOT), "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path), "status": synthetic["status"],
            "contexts": synthetic["contexts"], "Q0_chains": synthetic["full_scale_Q0_chains"],
            "Q0_records": synthetic["full_scale_Q0_records"],
            "method_context_rows": synthetic["full_scale_method_context_rows"],
        },
        "environment": {**LOCKED_ENVIRONMENT, "GPU_required": False},
        "loader_sources": list(LOADER_SOURCE_IDENTITIES),
        "static_process_isolation_sha256": canonical_sha256(static),
        "analysis_contract": {
            "contexts": 944, "candidates_per_context": 81,
            "candidate_score_rows": 535248, "candidate_rank_rows": 535248,
            "fixed_selection_rows": 4720, "Q0_records": 9110448,
            "Q0_shards": 944, "Q0_chains": 2048, "method_context_rows": 18608,
            "selection_freeze_before_evaluation": True,
            "scientific_contract_changed": False,
        },
        "execution": {
            "module": "oaci.multidataset.c84sr2_execute",
            "output_root": str(DEFAULT_OUTPUT_ROOT), "fresh_root_required": True,
            "subprocess_stages": ["Stage_A_immutable_replay", "Stage_B", "Stage_C"],
        },
        "authorization": {
            "record_path": str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)),
            "record_present_at_lock": False, "fresh_direct_statement": "授权 C84S",
            "historical_V3_authorization_migrates": False,
        },
        "failure_policy": {
            "automatic_retry": False, "partial_stage_B_publishable": False,
            "stage_B_failure_keeps_evaluation_sealed": True,
            "implementation_change_requires_new_protocol_and_lock": True,
        },
        "forbidden": {
            "label_reload_in_stage_A_replay": True, "training": True, "forward": True,
            "GPU": True, "same_label_oracle": True, "new_method": True,
            "threshold_change": True, "evaluation_before_selection_freeze": True,
            "C85": True,
        },
        "success_gate": SUCCESS_GATE, "failure_gate": FAILURE_GATE,
    }
    digest = write_json(LOCK_PATH, lock)
    LOCK_SHA_PATH.write_text(f"{digest}  {LOCK_PATH.name}\n", encoding="ascii")
    return {"lock": lock, "sha256": digest}
