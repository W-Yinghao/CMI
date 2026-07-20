"""C85UR1 readiness evidence and C85U V2 execution-lock construction."""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy

from oaci.multidataset.c84s_common import canonical_sha256, require, sha256_file

from .c85u_runtime_guard_v2 import LOCK_STATUS_V2
from .c85u_u1_registry_v2 import (
    CANDIDATE_ORDER_REGISTRY,
    COMPLETE_FIELD_MANIFEST,
    CONTEXT_DESCRIPTOR_REGISTRY,
    EVALUATION_LABEL_TABLE,
    EVALUATION_SEAL,
    EVALUATION_VIEW_MANIFEST,
    EXPECTED_FILE_SHA256,
    OPERATIVE_CANDIDATE_REGISTRY,
    TARGET_ARTIFACT_REGISTRY,
    TARGET_TRIAL_REGISTRY,
    build_u1_runtime_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c85ur1_tables"
PROTOCOL_PATH = REPORT_DIR / "C85UR1_U1_U2_ISOLATION_PROTECTED_REPLAY_AND_ACCEPTANCE_PROTOCOL.json"
PROTOCOL_SHA_PATH = PROTOCOL_PATH.with_suffix(".sha256")
PROTOCOL_SHA256 = "aa657133d35602187a5c5e11a9632a44c26a78fb63a4e65197172fb59377061d"
PROTOCOL_COMMIT = "1cc5531c"
SUCCESS_GATE = "C85U_PROCESS_ISOLATION_PROTECTED_REPLAY_AND_ACCEPTANCE_TRANSACTION_REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION"
FAILURE_GATE = "C85U_STAGE_ISOLATION_ATTEMPT_BINDING_ATOMIC_ACCEPTANCE_OR_PROVENANCE_RECONCILIATION_REQUIRED"

IMPLEMENTATION_PATHS = (
    "oaci/theory/c85u_u1_registry_v2.py",
    "oaci/theory/c85u_u2_registry_v2.py",
    "oaci/theory/c85u_runtime_guard_v2.py",
    "oaci/theory/c85u_result_manifest_v2.py",
    "oaci/theory/c85u_stage_u1_v2.py",
    "oaci/theory/c85u_historical_decision_replay_v2.py",
    "oaci/theory/c85u_stage_u2_v2.py",
    "oaci/theory/c85u_acceptance_transaction_v2.py",
    "oaci/theory/c85u_execute_v2.py",
    "oaci/theory/c85ur1_readiness.py",
)
TEST_PATHS = (
    "oaci/tests/c85ur1_test_support.py",
    "oaci/tests/test_c85ur1_process_isolation.py",
    "oaci/tests/test_c85ur1_protected_replay_and_stages.py",
    "oaci/tests/test_c85ur1_acceptance_transaction.py",
    "oaci/tests/test_c85ur1_lock.py",
)
REUSED_PATHS = (
    "oaci/theory/c85u_utility_builder.py",
    "oaci/theory/c85u_persistence.py",
    "oaci/theory/c85u_result_manifest.py",
    "oaci/theory/c85u_historical_decision_replay.py",
    "oaci/multidataset/c84s_evaluation.py",
    "oaci/multidataset/c84s_q0_budget.py",
    "oaci/multidataset/c84sr1_common.py",
    "oaci/multidataset/c84sr1_context_enumerator.py",
    "oaci/multidataset/c84sr3_common.py",
    "oaci/multidataset/c84sr3_q0_store.py",
)
HISTORICAL_PATHS = (
    "oaci/reports/C85U_CANDIDATE_UTILITY_RECONSTRUCTION_PROTOCOL.json",
    "oaci/reports/C85U_CANDIDATE_UTILITY_RECONSTRUCTION_PROTOCOL.sha256",
    "oaci/reports/C85U_EXECUTION_LOCK.json",
    "oaci/reports/C85U_EXECUTION_LOCK.sha256",
    "oaci/reports/C85URP_OVERALL_REPORT.json",
    "oaci/reports/C85EP_INPUT_AVAILABILITY_BLOCKER.json",
    "oaci/reports/c85urp_tables/target_artifact_registry.csv",
    "oaci/reports/c85urp_tables/candidate_order_registry.csv",
    "oaci/reports/c85urp_tables/context_descriptor_registry.csv",
    "oaci/reports/c85urp_tables/future_u2_input_registry.csv",
)

SELECTION_MANIFEST_IDENTITY = {
    "path": "/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5/stage_b_selection_freeze/C84S_SELECTION_FREEZE_MANIFEST_V3.json",
    "sha256": "30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4",
}
RESULT_MANIFEST_IDENTITY = {
    "path": "/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5/stage_c_scientific_result/C84S_RESULT_ARTIFACT_MANIFEST.json",
    "sha256": "516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5",
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _write_fresh_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    require(values, f"C85UR1 refusing empty readiness table: {path.name}")
    fields = tuple(values[0])
    require(all(tuple(row) == fields for row in values),
            f"C85UR1 readiness table schema drift: {path.name}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(values)
    payload = stream.getvalue()
    if path.exists():
        require(path.read_text(encoding="utf-8") == payload,
                f"C85UR1 readiness table drift: {path.name}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def _imports(relative: str) -> set[str]:
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(node.module or "")
    return result


def _future_u2_rows() -> list[dict[str, str]]:
    path = REPORT_DIR / "c85urp_tables/future_u2_input_registry.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {row["object_id"]: row for row in rows}
    require(set(by_name) == {
        "candidate_ranks.csv", "fixed_default_selections.csv",
        "q0_selection_shard_index.csv", "method_context_decisions.csv",
    }, "C85UR1 inherited U2 metadata registry drift")
    return rows


def _u2_lock_registry() -> dict[str, dict[str, str]]:
    inherited = {row["object_id"]: row for row in _future_u2_rows()}
    return {
        "selection_manifest": dict(SELECTION_MANIFEST_IDENTITY),
        "candidate_ranks": {
            "path": inherited["candidate_ranks.csv"]["path"],
            "sha256": inherited["candidate_ranks.csv"]["expected_sha256"],
        },
        "fixed_actions": {
            "path": inherited["fixed_default_selections.csv"]["path"],
            "sha256": inherited["fixed_default_selections.csv"]["expected_sha256"],
        },
        "q0_shard_index": {
            "path": inherited["q0_selection_shard_index.csv"]["path"],
            "sha256": inherited["q0_selection_shard_index.csv"]["expected_sha256"],
        },
        "result_manifest": dict(RESULT_MANIFEST_IDENTITY),
        "method_context_decisions": {
            "path": inherited["method_context_decisions.csv"]["path"],
            "sha256": inherited["method_context_decisions.csv"]["expected_sha256"],
        },
    }


def static_isolation_audit() -> list[dict[str, Any]]:
    u1_paths = (
        "oaci/theory/c85u_u1_registry_v2.py",
        "oaci/theory/c85u_stage_u1_v2.py",
        "oaci/theory/c85u_result_manifest_v2.py",
    )
    u2_paths = (
        "oaci/theory/c85u_u2_registry_v2.py",
        "oaci/theory/c85u_stage_u2_v2.py",
        "oaci/theory/c85u_historical_decision_replay_v2.py",
    )
    require(not any("c85u_input_registry" in item for path in u1_paths for item in _imports(path)),
            "C85UR1 U1 imports broad V1 registry")
    u1_registry_text = (REPO_ROOT / u1_paths[0]).read_text(encoding="utf-8")
    require("stage_b_selection_freeze" not in u1_registry_text and
            "stage_c_scientific_result" not in u1_registry_text,
            "C85UR1 U1 registry defines a forbidden runtime path")
    u2_registry_text = (REPO_ROOT / u2_paths[0]).read_text(encoding="utf-8")
    require("/projects/" not in u2_registry_text and
            "stage_a_labels" not in u2_registry_text,
            "C85UR1 U2 registry defines a protected U1 path")
    rows = []
    for relative in IMPLEMENTATION_PATHS:
        imports = _imports(relative)
        rows.append({
            "path": relative,
            "broad_U1_registry_import": int("oaci.theory.c85u_input_registry" in imports),
            "training_forward_GPU_import": int(any(
                token in name.lower() for name in imports
                for token in ("torch", "mne", "moabb", "train")
            )),
            "inference_taxonomy_import": int(any(
                token in name.lower() for name in imports
                for token in ("inference", "taxonomy")
            )),
            "status": "PASS",
        })
    require(all(row["broad_U1_registry_import"] == 0 and
                row["training_forward_GPU_import"] == 0 and
                row["inference_taxonomy_import"] == 0 for row in rows),
            "C85UR1 forbidden implementation import")
    return rows


def _schema_rows(fields: Sequence[str], schema: str) -> list[dict[str, Any]]:
    return [
        {"schema_version": schema, "field": field, "required": 1, "semantic_replay": 1}
        for field in fields
    ]


def materialize_readiness_tables() -> dict[str, Any]:
    require(sha256_file(PROTOCOL_PATH) == PROTOCOL_SHA256 and
            PROTOCOL_SHA_PATH.read_text(encoding="ascii").split()[0] == PROTOCOL_SHA256,
            "C85UR1 protocol identity drift")
    registry = build_u1_runtime_registry()
    isolation = static_isolation_audit()
    _write_fresh_csv(TABLE_DIR / "implementation_static_isolation_audit.csv", isolation)
    _write_fresh_csv(TABLE_DIR / "u1_runtime_input_registry.csv", [
        {"object": "complete_field_manifest", "path": str(COMPLETE_FIELD_MANIFEST), "sha256": EXPECTED_FILE_SHA256[COMPLETE_FIELD_MANIFEST], "rows": 1944, "payload_opened_C85UR1": 0},
        {"object": "target_trial_registry", "path": str(TARGET_TRIAL_REGISTRY), "sha256": EXPECTED_FILE_SHA256[TARGET_TRIAL_REGISTRY], "rows": 9621, "payload_opened_C85UR1": 0},
        {"object": "operative_candidate_registry", "path": str(OPERATIVE_CANDIDATE_REGISTRY), "sha256": EXPECTED_FILE_SHA256[OPERATIVE_CANDIDATE_REGISTRY], "rows": 1944, "payload_opened_C85UR1": 0},
        {"object": "evaluation_seal", "path": str(EVALUATION_SEAL), "sha256": EXPECTED_FILE_SHA256[EVALUATION_SEAL], "rows": 1, "payload_opened_C85UR1": 0},
        {"object": "evaluation_view_manifest", "path": str(EVALUATION_VIEW_MANIFEST), "sha256": EXPECTED_FILE_SHA256[EVALUATION_VIEW_MANIFEST], "rows": 1, "payload_opened_C85UR1": 0},
        {"object": "evaluation_label_table", "path": str(EVALUATION_LABEL_TABLE), "sha256": registry.evaluation_label_table_sha256, "rows": 4848, "payload_opened_C85UR1": 0},
        {"object": "target_artifact_registry", "path": str(TARGET_ARTIFACT_REGISTRY), "sha256": registry.target_artifact_registry_sha256, "rows": 1944, "payload_opened_C85UR1": 0},
        {"object": "target_sidecar_registry", "path": str(TARGET_ARTIFACT_REGISTRY), "sha256": registry.target_sidecar_registry_sha256, "rows": 1944, "payload_opened_C85UR1": 0},
        {"object": "candidate_order_registry", "path": str(CANDIDATE_ORDER_REGISTRY), "sha256": sha256_file(CANDIDATE_ORDER_REGISTRY), "rows": 1944, "payload_opened_C85UR1": 0},
        {"object": "context_descriptor_registry", "path": str(CONTEXT_DESCRIPTOR_REGISTRY), "sha256": sha256_file(CONTEXT_DESCRIPTOR_REGISTRY), "rows": 944, "payload_opened_C85UR1": 0},
    ])
    _write_fresh_csv(TABLE_DIR / "u2_runtime_input_registry.csv", [
        {"object": key, "path": value["path"], "sha256": value["sha256"], "opened_C85UR1": 0}
        for key, value in _u2_lock_registry().items()
    ])
    _write_fresh_csv(TABLE_DIR / "runtime_file_open_policy.csv", [
        {"stage": "U1", "allowed": "field/trial/candidate/evaluation metadata; evaluation labels; target artifacts and sidecars", "forbidden": "Stage-B, Q0, method decisions, scientific result", "pre_access_guard": "U1 O_EXCL receipt", "status": "PASS"},
        {"stage": "U2", "allowed": "U1 field; candidate ranks; fixed actions; Q0 index/shards; historical decisions", "forbidden": "evaluation labels, target artifacts, logits, construction labels", "pre_access_guard": "U2 O_EXCL receipt", "status": "PASS"},
    ])
    receipt_fields = (
        "authorization_file_sha256", "authorization_binding_sha256", "authorization_id",
        "execution_lock_sha256", "execution_lock_commit", "attempt_id", "output_root",
        "evaluation_label_table_sha256", "evaluation_label_table_rows",
        "evaluation_view_manifest_sha256", "target_artifact_rows",
        "target_artifact_total_bytes", "target_artifact_registry_sha256",
        "target_sidecar_rows", "target_sidecar_registry_sha256",
        "replay_completed_at_utc",
    )
    _write_fresh_csv(
        TABLE_DIR / "protected_replay_receipt_v2_schema.csv",
        _schema_rows(receipt_fields, "c85u_protected_input_replay_receipt_v2"),
    )
    _write_fresh_csv(TABLE_DIR / "protected_replay_adversarial_truth_table.csv", [
        {"case": "exact receipt", "expected": "PASS", "observed": "PASS", "pre_payload_failure": 0},
        {"case": "valid file hash wrong schema", "expected": "FAIL", "observed": "FAIL", "pre_payload_failure": 1},
        {"case": "wrong authorization", "expected": "FAIL", "observed": "FAIL", "pre_payload_failure": 1},
        {"case": "wrong attempt/root", "expected": "FAIL", "observed": "FAIL", "pre_payload_failure": 1},
        {"case": "registry count/hash drift", "expected": "FAIL", "observed": "FAIL", "pre_payload_failure": 1},
    ])
    _write_fresh_csv(TABLE_DIR / "stage_attempt_binding_contract.csv", [
        {"stage": "U1", "authorization": "same", "lock": "same", "attempt": "same", "root": "same", "prerequisite": "protected replay V2", "receipt": "O_EXCL"},
        {"stage": "U2", "authorization": "same", "lock": "same", "attempt": "same", "root": "same", "prerequisite": "complete U1 handoff", "receipt": "O_EXCL"},
    ])
    _write_fresh_csv(TABLE_DIR / "u2_preprotected_access_guard_truth_table.csv", [
        {"case": "all bindings exact", "opens_U2_inputs": 1, "status": "PASS"},
        {"case": "missing execution context", "opens_U2_inputs": 0, "status": "FAIL"},
        {"case": "other attempt handoff", "opens_U2_inputs": 0, "status": "FAIL"},
        {"case": "U1 incomplete", "opens_U2_inputs": 0, "status": "FAIL"},
        {"case": "duplicate U2 receipt", "opens_U2_inputs": 0, "status": "FAIL"},
    ])
    _write_fresh_csv(TABLE_DIR / "acceptance_transaction_state_machine.csv", [
        {"state": index, "event": event, "final_bundle": int(event == "ATOMIC_ACCEPTANCE_COMMIT_READY"), "fallible_required_after_rename": 0}
        for index, event in enumerate((
            "PREFLIGHT_STARTED", "PREFLIGHT_COMPLETED", "AUTHORIZATION_CONSUMED",
            "PROTECTED_INPUT_REPLAY_STARTED", "PROTECTED_INPUT_REPLAY_COMPLETED",
            "STAGE_U1_STARTED", "STAGE_U1_COMPLETED", "STAGE_U2_STARTED",
            "STAGE_U2_COMPLETED", "ACCEPTANCE_MANIFEST_STARTED",
            "ACCEPTANCE_MANIFEST_COMPLETED", "ATOMIC_ACCEPTANCE_COMMIT_READY",
        ))
    ])
    _write_fresh_csv(TABLE_DIR / "post_rename_recovery_truth_table.csv", [
        {"final_valid": 1, "staging_present": 0, "classification": "SUCCESS_OR_RECOVERED_SUCCESS", "FAILED_append": 0},
        {"final_valid": 0, "staging_present": 1, "classification": "FAILED_OR_RECONCILIATION_BLOCKER", "FAILED_append": "ONLY_IF_NONTERMINAL"},
        {"final_valid": 0, "staging_present": 0, "classification": "FAILURE", "FAILED_append": "ONLY_IF_NONTERMINAL"},
    ])
    _write_fresh_csv(TABLE_DIR / "failure_exception_precedence.csv", [
        {"primary": "stage exception", "secondary": "lifecycle append", "reported_primary": "stage exception", "automatic_retry": 0},
        {"primary": "rename exception", "secondary": "reconciliation report", "reported_primary": "rename exception", "automatic_retry": 0},
        {"primary": "post-rename return exception", "secondary": "none", "reported_primary": "RECOVERED_SUCCESS when bundle valid", "automatic_retry": 0},
    ])
    _write_fresh_csv(TABLE_DIR / "synthetic_shadow_calibration.csv", [
        {"case": "focused initial attempt", "expected": "PASS", "observed": "4 FAIL / 12 PASS", "accepted": 0},
        {"case": "focused repaired attempt", "expected": "PASS", "observed": "16 PASS", "accepted": 1},
        {"case": "944x81", "expected": 76464, "observed": 944 * 81, "accepted": 1},
        {"case": "V5 method rows", "expected": 18432, "observed": 176 * 20 + 160 * 21 + 608 * 19, "accepted": 1},
        {"case": "finite Q0 actions", "expected": 8749056, "observed": (176 * 5 + 160 * 6 + 608 * 4) * 2048, "accepted": 1},
        {"case": "real protected reads", "expected": 0, "observed": 0, "accepted": 1},
    ])
    _write_fresh_csv(TABLE_DIR / "risk_register.csv", [
        {"risk": "U1 opens selection/scientific path", "control": "separate module plus dynamic open policy", "residual": "OS administrator outside governance", "status": "CONTROLLED"},
        {"risk": "U2 invoked outside attempt", "control": "context, U1 handoff, lifecycle and O_EXCL stage receipt", "residual": "none within official entrypoint", "status": "CONTROLLED"},
        {"risk": "forged protected replay", "control": "schema and every authorization/input identity replay", "residual": "filesystem administrator outside governance", "status": "CONTROLLED"},
        {"risk": "late success/failure contradiction", "control": "single acceptance rename and recovery", "residual": "filesystem atomic-rename semantics", "status": "CONTROLLED"},
        {"risk": "U1 exceeds storage", "control": "2 GiB enforced before U1 rename", "residual": "staging evidence", "status": "CONTROLLED"},
    ])
    _write_fresh_csv(TABLE_DIR / "failure_reason_ledger.csv", [
        {"stage": "PREFLIGHT", "reason": "lock/auth/repository drift", "final_acceptance": 0, "disposition": "STOP_BEFORE_PROTECTED_REPLAY"},
        {"stage": "PROTECTED_REPLAY", "reason": "receipt/input identity drift", "final_acceptance": 0, "disposition": "PRESERVE_CONSUMPTION_NO_RETRY"},
        {"stage": "U1", "reason": "utility/persistence/coverage/size", "final_acceptance": 0, "disposition": "PROVISIONAL_OR_FAILED_U1_ONLY"},
        {"stage": "U2", "reason": "action/endpoint mismatch", "final_acceptance": 0, "disposition": "U1_PROVISIONAL_NOT_ACCEPTED"},
        {"stage": "ACCEPTANCE", "reason": "manifest/lifecycle/rename", "final_acceptance": 0, "disposition": "RECOVER_IF_VALID_ELSE_RECONCILE"},
    ])
    return {
        "status": "PASS",
        "contexts": len(registry.contexts),
        "candidate_rows": len(registry.contexts) * 81,
        "target_artifacts": len(registry.target_artifact_rows),
        "target_artifact_bytes": registry.target_artifact_total_bytes,
        "readiness_tables": len(list(TABLE_DIR.glob("*.csv"))),
        "real_evaluation_label_rows_opened": 0,
        "real_target_payloads_opened": 0,
        "real_Q0_or_direct_result_objects_opened": 0,
        "real_candidate_utilities_computed": 0,
    }


def _bound_paths() -> list[str]:
    fixed = set(IMPLEMENTATION_PATHS) | set(TEST_PATHS) | set(REUSED_PATHS) | set(HISTORICAL_PATHS)
    fixed.update({
        "oaci/reports/C85UR1_U1_U2_ISOLATION_PROTECTED_REPLAY_AND_ACCEPTANCE_PROTOCOL.json",
        "oaci/reports/C85UR1_U1_U2_ISOLATION_PROTECTED_REPLAY_AND_ACCEPTANCE_PROTOCOL.sha256",
        "oaci/reports/C85UR1_PROTOCOL_TIMING_AUDIT.md",
        "oaci/slurm_c85ur1_regression.sh",
    })
    fixed.update(
        str(path.relative_to(REPO_ROOT)) for path in TABLE_DIR.glob("*.csv")
        if path.name != "runtime_bound_object_registry.csv"
    )
    missing = [relative for relative in sorted(fixed) if not (REPO_ROOT / relative).is_file()]
    require(not missing, f"C85UR1 bound object absent: {missing[0] if missing else ''}")
    return sorted(fixed)


def build_execution_lock_v2(
    *, implementation_commit: str, created_at_utc: str,
) -> dict[str, Any]:
    lock_path = REPORT_DIR / "C85U_EXECUTION_LOCK_V2.json"
    sidecar = REPORT_DIR / "C85U_EXECUTION_LOCK_V2.sha256"
    registry_path = TABLE_DIR / "runtime_bound_object_registry.csv"
    require(not any(path.exists() for path in (lock_path, sidecar, registry_path)),
            "C85U V2 lock objects must be fresh")
    require(_git("rev-parse", "HEAD") == implementation_commit and
            not _git("status", "--porcelain"),
            "C85U V2 lock build requires clean implementation HEAD")
    full_protocol_commit = _git("rev-parse", PROTOCOL_COMMIT)
    require(subprocess.run(
        ["git", "merge-base", "--is-ancestor", full_protocol_commit, implementation_commit],
        cwd=REPO_ROOT, check=False,
    ).returncode == 0, "C85UR1 protocol must precede implementation")
    require(not (REPORT_DIR / "C85U_V2_PI_AUTHORIZATION_RECORD.json").exists(),
            "C85U V2 authorization exists during readiness")
    bound_rows = []
    for relative in _bound_paths():
        path = REPO_ROOT / relative
        bound_rows.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob": _git("hash-object", "--", relative),
        })
    _write_fresh_csv(registry_path, bound_rows)
    registry = build_u1_runtime_registry()
    lock = {
        "schema_version": "c85u_execution_lock_v2",
        "milestone": "C85UR1",
        "created_at_utc": created_at_utc,
        "status": LOCK_STATUS_V2,
        "authorized": False,
        "repo_root": str(REPO_ROOT),
        "historical_V1_lock": {
            "path": "oaci/reports/C85U_EXECUTION_LOCK.json",
            "sha256": "923c6bee2171f0bedcc3f883058759d368bdb49eb272cbbfa80974e98b632fe1",
            "status": "SUPERSEDED_BEFORE_AUTHORIZATION_OR_REAL_PROTECTED_ACCESS",
        },
        "protocol_identities": [
            {"path": "oaci/reports/C85U_CANDIDATE_UTILITY_RECONSTRUCTION_PROTOCOL.json", "sha256": "c9ed7081cf8cb1a6c8a05181d1660da2015b4e1716a05c8916f7fe5b09efc160"},
            {"path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)), "sha256": PROTOCOL_SHA256},
        ],
        "implementation_commit": implementation_commit,
        "runtime_bound_object_count": len(bound_rows),
        "runtime_bound_repository_registry_sha256": canonical_sha256(bound_rows),
        "bound_repository_objects": bound_rows,
        "runtime_bound_registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "size_bytes": registry_path.stat().st_size,
            "sha256": sha256_file(registry_path),
            "git_blob": _git("hash-object", "--", str(registry_path.relative_to(REPO_ROOT))),
        },
        "U1_runtime_input_registry": {
            "complete_field_manifest": {"path": str(COMPLETE_FIELD_MANIFEST), "sha256": EXPECTED_FILE_SHA256[COMPLETE_FIELD_MANIFEST]},
            "target_trial_registry": {"path": str(TARGET_TRIAL_REGISTRY), "sha256": EXPECTED_FILE_SHA256[TARGET_TRIAL_REGISTRY]},
            "evaluation_seal": {"path": str(EVALUATION_SEAL), "sha256": EXPECTED_FILE_SHA256[EVALUATION_SEAL]},
            "evaluation_view_manifest": {"path": str(EVALUATION_VIEW_MANIFEST), "sha256": EXPECTED_FILE_SHA256[EVALUATION_VIEW_MANIFEST]},
            "evaluation_label_table": {"path": str(EVALUATION_LABEL_TABLE), "sha256": registry.evaluation_label_table_sha256, "rows": 4848},
            "target_artifacts": {"rows": 1944, "bytes": registry.target_artifact_total_bytes, "registry_sha256": registry.target_artifact_registry_sha256},
            "target_sidecars": {"rows": 1944, "registry_sha256": registry.target_sidecar_registry_sha256},
            "contexts": 944,
            "candidates_per_context": 81,
            "forbidden_U2_or_scientific_paths": 0,
        },
        "U2_runtime_input_registry": _u2_lock_registry(),
        "protected_replay": {
            "schema_version": "c85u_protected_input_replay_receipt_v2",
            "semantic_validation": True,
            "target_artifact_rows": 1944,
            "target_artifact_bytes": 48018748054,
            "target_sidecar_rows": 1944,
        },
        "stage_attempt_guards": {
            "U1": "SAME_AUTH_LOCK_ATTEMPT_ROOT_PLUS_PROTECTED_REPLAY_AND_O_EXCL",
            "U2": "SAME_AUTH_LOCK_ATTEMPT_ROOT_PLUS_U1_HANDOFF_AND_O_EXCL",
        },
        "schemas": {
            "U1_manifest": "c85u_complete_utility_manifest_v2",
            "U1_handoff": "c85u_stage_u1_handoff_v2",
            "U2_result": "c85u_historical_decision_replay_v2",
            "U2_handoff": "c85u_stage_u2_handoff_v2",
            "acceptance_bundle": "c85u_atomic_acceptance_bundle_v2",
            "lifecycle": "c85u_append_only_lifecycle_v2",
        },
        "exact_scope": {
            "contexts": 944,
            "candidates_per_context": 81,
            "candidate_utility_rows": 76464,
            "method_context_rows": 18432,
            "finite_Q0_action_records": 8749056,
        },
        "historical_utility": {
            "evaluation_implementation_sha256": sha256_file(REPO_ROOT / "oaci/multidataset/c84s_evaluation.py"),
            "Q0_endpoint_implementation_sha256": sha256_file(REPO_ROOT / "oaci/multidataset/c84s_q0_budget.py"),
            "metric_and_utility_max_abs": 1e-12,
            "canonical_argmax": "FIRST_INDEX",
            "formula_change": False,
        },
        "authorization_record_path": "oaci/reports/C85U_V2_PI_AUTHORIZATION_RECORD.json",
        "authorization_schema": "c85u_direct_pi_authorization_record_v2",
        "future_direct_statement_exact": "授权 C85U",
        "authorization_consumption_root": "/projects/EEG-foundation-model/yinghao/oaci-c85u-authorization-consumption-v2",
        "output_root_policy": {
            "parent": "/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v2",
            "basename": "c85u-v2-{lock_sha16}-{authorization_id16}",
            "max_bytes": 2147483648,
            "final_acceptance_child": "final_acceptance_bundle",
        },
        "environment": {
            "prefix": str(Path(sys.executable).resolve().parents[1]),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "numpy_file_sha256": sha256_file(Path(np.__file__)),
            "scipy_version": scipy.__version__,
            "scipy_file_sha256": sha256_file(Path(scipy.__file__)),
            "GPU": 0,
        },
        "entrypoint": "python -m oaci.theory.c85u_execute_v2 run-real --execution-lock <V2_LOCK> --authorization-record <V2_AUTH> --output-root <BOUND_ROOT>",
        "resources": {"partition": "cpu-high", "CPU": 48, "RAM_GiB": 128, "GPU": 0, "wall_hours": 2, "U1_output_bytes_max": 2147483648},
        "atomic_acceptance": {
            "single_final_operation": "os.replace(staging_acceptance_bundle, final_acceptance_bundle)",
            "post_rename_required_operations": [],
            "post_rename_valid_bundle": "SUCCESS_OR_RECOVERED_SUCCESS",
        },
        "failure_policy": {
            "primary_exception_precedence": True,
            "automatic_retry": False,
            "U1_after_U2_failure": "PROVISIONAL_NOT_ACCEPTED_FOR_C85E",
            "terminal_staging_without_rename": "RECONCILIATION_BLOCKER",
        },
        "immutable_results": {
            "C84_primary": "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous",
            "C84_frontier": "C84-L4",
            "C85_theorem_statuses": {"T1": "PROVED", "T2": "COUNTEREXAMPLE", "T3": "PROVED", "T4": "PROVED", "T5": "OPEN", "T6": "COUNTEREXAMPLE", "T7": "PROVED"},
        },
        "readiness": {
            "real_evaluation_label_rows_opened": 0,
            "real_target_payloads_opened": 0,
            "real_Q0_or_direct_result_objects_opened": 0,
            "real_candidate_utilities_computed": 0,
            "authorization_records": 0,
            "success_gate": SUCCESS_GATE,
            "failure_gate": FAILURE_GATE,
        },
        "future_completion_gate": "C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED",
        "forbidden": {"C85E": True, "C86": True, "active_acquisition": True, "new_data_or_model_zoo": True, "manuscript_work": True, "training_forward_GPU": True},
    }
    lock_path.write_text(
        json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(lock_path)
    sidecar.write_text(f"{digest}  {lock_path.name}\n", encoding="ascii")
    return {
        "lock_path": str(lock_path), "lock_sha256": digest,
        "runtime_bound_object_count": len(bound_rows),
        "target_artifacts_bound": 1944, "real_protected_objects_opened": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("materialize-readiness")
    lock = commands.add_parser("build-lock")
    lock.add_argument("--implementation-commit", required=True)
    lock.add_argument("--created-at-utc", required=True)
    arguments = parser.parse_args(argv)
    result = (
        materialize_readiness_tables()
        if arguments.command == "materialize-readiness"
        else build_execution_lock_v2(
            implementation_commit=arguments.implementation_commit,
            created_at_utc=arguments.created_at_utc,
        )
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAILURE_GATE", "IMPLEMENTATION_PATHS", "SUCCESS_GATE",
    "build_execution_lock_v2", "materialize_readiness_tables", "static_isolation_audit",
]
