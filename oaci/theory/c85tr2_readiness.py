"""C85TR2 contract tables and prospective C85T V3 lock builder."""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence

from .c85_decision_experiments import DecisionContractError
from .c85t_execution_context_v3 import canonical_json_bytes


PROTOCOL_SHA256 = "f9a1db908f34818b7551c0d4f8de65fa7a11e71c41b8e5fe28824f042904a844"
PROTOCOL_COMMIT = "2e79f304202faffb857610e273ec5510a608080a"
HISTORICAL_V2_LOCK_SHA256 = "0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719"
LOCK_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
SUCCESS_GATE = (
    "C85T_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_"
    "REPAIRED_V3_LOCK_READY_FOR_PI_AUTHORIZATION"
)
FAILURE_GATE = (
    "C85T_EXECUTION_CONTEXT_ATOMIC_PUBLICATION_OR_RESULT_SEMANTIC_REPLAY_"
    "RECONCILIATION_REQUIRED"
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise DecisionContractError(f"refusing empty C85TR2 table: {path.name}")
    fields = tuple(values[0])
    if any(tuple(row) != fields for row in values):
        raise DecisionContractError(f"C85TR2 table schema drifted: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _static_transaction_audit(repo_root: Path) -> dict[str, int]:
    path = repo_root / "oaci/theory/c85t_transaction_v3.py"
    source = path.read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_commit_prepared"
    )
    replace_nodes = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
    ]
    replace_index = next(
        index
        for index, node in enumerate(method.body)
        if isinstance(node, ast.Expr) and node.value in replace_nodes
    )
    if len(replace_nodes) != 1 or len(method.body) != replace_index + 2:
        raise DecisionContractError("C85T V3 commit has work after final rename")
    if not isinstance(method.body[-1], ast.Return):
        raise DecisionContractError("C85T V3 commit does not return in-memory receipt")
    forbidden = (
        "manifest_completed_callback",
        "atomic_publish_callback",
        "_CAPABILITY_SENTINEL",
        "_ISSUED_CAPABILITIES",
        "_issue_capability",
        "consume_authorization_once",
    )
    operative = "\n".join(
        (repo_root / "oaci/theory" / name).read_text()
        for name in (
            "c85t_execution_context_v3.py",
            "c85t_registered_v3.py",
            "c85t_semantic_replay_v3.py",
            "c85t_transaction_v3.py",
            "c85t_execute_v3.py",
        )
    )
    if any(value in operative for value in forbidden):
        raise DecisionContractError("C85T V3 operative source retained a superseded API")
    return {"os_replace_calls": 1, "required_operations_after_rename": 0}


def materialize_contract_tables(repo_root: Path) -> dict[str, Any]:
    reports = repo_root / "oaci/reports"
    tables = reports / "c85tr2_tables"
    if tables.exists():
        raise DecisionContractError("C85TR2 table directory must be fresh")
    if _sha(reports / "C85T_EXECUTION_LOCK_V2.json") != HISTORICAL_V2_LOCK_SHA256:
        raise DecisionContractError("historical C85T V2 lock drifted")
    static = _static_transaction_audit(repo_root)
    tables.mkdir(parents=True)
    _write_csv(
        tables / "historical_v2_lock_supersession.csv",
        [
            {
                "object": "C85T_EXECUTION_LOCK_V2.json",
                "commit": "920c5540a6ae157b77f2acb36f227bfdc172110b",
                "sha256": HISTORICAL_V2_LOCK_SHA256,
                "authorization_records": 0,
                "authorization_consumed": 0,
                "registered_S0_S10_draws": 0,
                "status": "SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION",
                "historical_bytes_modified": 0,
            }
        ],
    )
    _write_csv(
        tables / "validated_execution_context_contract.csv",
        [
            {"step": 1, "operation": "replay committed V3 lock sidecar/schema/status", "inseparable_factory": 1, "blocking": 1},
            {"step": 2, "operation": "replay all bound bytes and Git blobs", "inseparable_factory": 1, "blocking": 1},
            {"step": 3, "operation": "replay branch clean HEAD origin and ancestry", "inseparable_factory": 1, "blocking": 1},
            {"step": 4, "operation": "validate committed authorization path/content/chronology", "inseparable_factory": 1, "blocking": 1},
            {"step": 5, "operation": "validate exact output and consumption-root policies", "inseparable_factory": 1, "blocking": 1},
            {"step": 6, "operation": "create O_EXCL receipt and fsync file/directory", "inseparable_factory": 1, "blocking": 1},
            {"step": 7, "operation": "append matching AUTHORIZATION_CONSUMED", "inseparable_factory": 1, "blocking": 1},
            {"step": 8, "operation": "return receipt-validated context", "inseparable_factory": 1, "blocking": 1},
        ],
    )
    _write_csv(
        tables / "unauthorized_internal_api_adversarial_tests.csv",
        [
            {"case": "direct constructor", "expected": "FAIL", "observed_shadow": "PASS", "security_claim": "official-result governance"},
            {"case": "generic mapping", "expected": "FAIL", "observed_shadow": "PASS", "security_claim": "official-result governance"},
            {"case": "copy/deepcopy/pickle", "expected": "FAIL", "observed_shadow": "PASS", "security_claim": "official-result governance"},
            {"case": "different attempt/root binding", "expected": "FAIL", "observed_shadow": "PASS", "security_claim": "official-result governance"},
            {"case": "receipt deletion/tamper", "expected": "FAIL", "observed_shadow": "PASS", "security_claim": "official-result governance"},
            {"case": "second O_EXCL consumption", "expected": "FAIL", "observed_shadow": "PASS", "security_claim": "official-result governance"},
        ],
    )
    transaction_stages = (
        "PREFLIGHT_STARTED",
        "PREFLIGHT_COMPLETED",
        "AUTHORIZATION_CONSUMED",
        "EXACT_SCENARIOS_STARTED",
        "EXACT_SCENARIOS_COMPLETED",
        "MONTE_CARLO_STARTED",
        "MONTE_CARLO_COMPLETED",
        "PROOF_CANDIDATES_STARTED",
        "PROOF_CANDIDATES_COMPLETED",
        "MANIFEST_STARTED",
        "MANIFEST_COMPLETED",
        "ATOMIC_PUBLISH_COMMIT_READY",
    )
    _write_csv(
        tables / "atomic_transaction_state_machine.csv",
        [
            {
                "sequence": sequence,
                "stage": stage,
                "inside_staging_before_rename": 1,
                "terminal": int(stage == "ATOMIC_PUBLISH_COMMIT_READY"),
                "fallible_required_operation_after_rename": 0,
            }
            for sequence, stage in enumerate(transaction_stages)
        ],
    )
    _write_csv(
        tables / "post_rename_recovery_truth_table.csv",
        [
            {"final_bundle": "valid", "staging": "absent", "terminal_event": "present", "classification": "RECOVERED_SUCCESS_AFTER_FINAL_RENAME", "append_FAILED": 0},
            {"final_bundle": "absent", "staging": "nonterminal", "terminal_event": "absent", "classification": "FAILED_STAGING_RETAINED", "append_FAILED": 1},
            {"final_bundle": "absent", "staging": "terminal", "terminal_event": "present", "classification": "RECONCILIATION_BLOCKER", "append_FAILED": 0},
            {"final_bundle": "invalid", "staging": "any", "terminal_event": "any", "classification": "RECONCILIATION_BLOCKER", "append_FAILED": 0},
        ],
    )
    _write_csv(
        tables / "failure_exception_precedence.csv",
        [
            {"secondary_operation": "terminal-ledger append", "may_replace_primary": 0, "storage": "secondary_errors"},
            {"secondary_operation": "cleanup", "may_replace_primary": 0, "storage": "secondary_errors"},
            {"secondary_operation": "quarantine rename", "may_replace_primary": 0, "storage": "secondary_errors"},
            {"secondary_operation": "failure reporting", "may_replace_primary": 0, "storage": "secondary_errors"},
            {"secondary_operation": "post-rename recovery", "may_replace_primary": 0, "storage": "recovered-success or secondary_errors"},
        ],
    )
    semantic_checks = (
        ("exact scenario key set", "S0..S10 exactly", "blocking"),
        ("S10 exact arithmetic", "11/40|0|3/5|13/40", "blocking"),
        ("S8 rational certificate", "all required LP fields", "blocking"),
        ("S6/S7 replicate IDs", "4096 unique canonical IDs each", "blocking"),
        ("S6/S7 selected action", "within contract action set", "blocking"),
        ("S6/S7 indicators/regret", "rederived from selected action", "blocking"),
        ("S9 selected action", "0..3", "blocking"),
        ("S9 raw dtype/count", "<i8|51|46", "blocking"),
        ("S9 digest fields", "lowercase 64-hex and deterministic consumed-stream replay", "blocking"),
        ("S9 aggregates", "recomputed from reloaded arrays", "blocking"),
        ("proof candidate coverage", "T1..T7 exactly", "blocking"),
        ("proof file/CSV hash", "exact", "blocking"),
        ("proof statement SHA", "bound statement", "blocking"),
        ("formal theorem status", "OPEN for T1..T7", "blocking"),
        ("authorization/result identity", "lock|auth|attempt|root|HEAD exact", "blocking"),
        ("protected counters", "zero", "blocking"),
    )
    _write_csv(
        tables / "result_semantic_replay_v3.csv",
        [
            {"object": obj, "required_identity": identity, "disposition": disposition}
            for obj, identity, disposition in semantic_checks
        ],
    )
    _write_csv(
        tables / "atomic_bundle_schema_v3.csv",
        [
            {"object": "scientific/synthetic artifacts", "inside_staging": 1, "manifest_bound": 1, "written_before_rename": 1},
            {"object": "proof candidates", "inside_staging": 1, "manifest_bound": 1, "written_before_rename": 1},
            {"object": "result manifest", "inside_staging": 1, "manifest_bound": 0, "written_before_rename": 1},
            {"object": "lifecycle ledger", "inside_staging": 1, "manifest_bound": 0, "written_before_rename": 1},
            {"object": "completion receipt", "inside_staging": 1, "manifest_bound": 0, "written_before_rename": 1},
        ],
    )
    _write_csv(
        tables / "risk_register.csv",
        [
            {"risk": "Python helper treated as authorization", "control": "committed path replay plus external receipt validation", "residual": "not OS/adversarial Python security", "status": "CONTROLLED_GOVERNANCE"},
            {"risk": "valid-looking partial final root", "control": "terminal staging bundle then one rename", "residual": "filesystem atomic-rename semantics", "status": "CONTROLLED"},
            {"risk": "post-rename process crash", "control": "hash-chain recovery classifies valid final as success", "residual": "external receipt filesystem availability", "status": "CONTROLLED"},
            {"risk": "semantic manifest false positive", "control": "artifact-derived counts and deterministic replay", "residual": "future environment must replay exactly", "status": "CONTROLLED"},
            {"risk": "proof status leakage", "control": "C85T candidates only; all formal statuses OPEN", "residual": "future C85V review quality", "status": "DEFERRED_TO_C85V"},
        ],
    )
    _write_csv(
        tables / "failure_reason_ledger.csv",
        [
            {"reason": "V2 importable capability minting", "historical_blocker": 1, "V3_control": "receipt-validated committed-path context", "readiness_status": "REPAIRED"},
            {"reason": "V2 callback after final rename", "historical_blocker": 1, "V3_control": "terminal bundle before one rename", "readiness_status": "REPAIRED"},
            {"reason": "V2 terminal FAILED masking", "historical_blocker": 1, "V3_control": "primary/secondary exception precedence", "readiness_status": "REPAIRED"},
            {"reason": "V2 weak semantic replay", "historical_blocker": 1, "V3_control": "artifact-derived semantic replay", "readiness_status": "REPAIRED"},
        ],
    )
    return {
        "table_count": len(list(tables.glob("*.csv"))),
        **static,
        "registered_S0_S10_draws": 0,
        "canonical_proof_candidates": 0,
        "theorem_status_transitions": 0,
        "authorization_records": 0,
    }


def _bound_paths(repo_root: Path) -> list[str]:
    reports = repo_root / "oaci/reports"
    historical = json.loads((reports / "C85T_EXECUTION_LOCK_V2.json").read_text())
    paths = {row["path"] for row in historical["bound_repository_objects"]}
    paths.update(
        {
            "oaci/reports/C85T_EXECUTION_LOCK_V2.json",
            "oaci/reports/C85T_EXECUTION_LOCK_V2.sha256",
            "oaci/reports/C85TR2_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_PROTOCOL.json",
            "oaci/reports/C85TR2_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_PROTOCOL.sha256",
            "oaci/reports/C85TR2_PROTOCOL_TIMING_AUDIT.md",
            "oaci/theory/c85t_execution_context_v3.py",
            "oaci/theory/c85t_registered_v3.py",
            "oaci/theory/c85t_semantic_replay_v3.py",
            "oaci/theory/c85t_transaction_v3.py",
            "oaci/theory/c85t_execute_v3.py",
            "oaci/theory/c85tr2_readiness.py",
            "oaci/tests/c85tr2_test_support.py",
            "oaci/tests/test_c85tr2_execution_context.py",
            "oaci/tests/test_c85tr2_atomic_transaction.py",
            "oaci/tests/test_c85tr2_semantic_replay.py",
            "oaci/tests/test_c85tr2_lock.py",
            "oaci/slurm_c85tr2_regression.sh",
        }
    )
    for path in (reports / "c85tr2_tables").glob("*.csv"):
        if path.name != "runtime_bound_object_registry.csv":
            paths.add(str(path.relative_to(repo_root)))
    missing = [relative for relative in sorted(paths) if not (repo_root / relative).is_file()]
    if missing:
        raise DecisionContractError(f"C85TR2 bound object is absent: {missing[0]}")
    return sorted(paths)


def build_execution_lock_v3(
    repo_root: Path, *, implementation_commit: str, created_at_utc: str
) -> dict[str, Any]:
    reports = repo_root / "oaci/reports"
    tables = reports / "c85tr2_tables"
    registry_path = tables / "runtime_bound_object_registry.csv"
    lock_path = reports / "C85T_EXECUTION_LOCK_V3.json"
    sidecar = reports / "C85T_EXECUTION_LOCK_V3.sha256"
    if any(path.exists() for path in (registry_path, lock_path, sidecar)):
        raise DecisionContractError("C85T V3 lock objects must be fresh")
    if _git(repo_root, "rev-parse", "HEAD") != implementation_commit:
        raise DecisionContractError("C85T V3 implementation commit must equal HEAD")
    if _git(repo_root, "status", "--porcelain"):
        raise DecisionContractError("C85T V3 lock build requires a clean worktree")
    protocol = reports / "C85TR2_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_PROTOCOL.json"
    if _sha(protocol) != PROTOCOL_SHA256:
        raise DecisionContractError("C85TR2 protocol hash drifted")
    if _sha(reports / "C85T_EXECUTION_LOCK_V2.json") != HISTORICAL_V2_LOCK_SHA256:
        raise DecisionContractError("historical C85T V2 lock drifted")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, implementation_commit],
            cwd=repo_root,
            check=False,
        ).returncode
        != 0
    ):
        raise DecisionContractError("C85TR2 protocol does not precede implementation")
    forbidden = (
        reports / "C85T_V3_PI_AUTHORIZATION_RECORD.json",
        reports / "C85T_RESULT.json",
        reports / "c85t_proof_candidates",
    )
    if any(path.exists() for path in forbidden):
        raise DecisionContractError("C85T V3 authorization or result exists at lock build")
    rows: list[dict[str, Any]] = []
    for relative in _bound_paths(repo_root):
        path = repo_root / relative
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
                "git_blob": _git(repo_root, "hash-object", "--", relative),
            }
        )
    _write_csv(registry_path, rows)
    registry_identity = {
        "path": str(registry_path.relative_to(repo_root)),
        "size_bytes": registry_path.stat().st_size,
        "sha256": _sha(registry_path),
        "git_blob": _git(
            repo_root,
            "hash-object",
            "--",
            str(registry_path.relative_to(repo_root)),
        ),
    }
    protocol_paths = (
        "oaci/reports/C85_TPAMI_DECISION_THEORY_PROTOCOL.json",
        "oaci/reports/C85R_SYNTHETIC_CONTRACT_SEMANTIC_REPAIR_PROTOCOL.json",
        "oaci/reports/c85r_tables/synthetic_generator_contract_v2.json",
        "oaci/reports/C85T_PROOF_AND_SYNTHETIC_EXECUTION_OPERATIONALIZATION_PROTOCOL.json",
        "oaci/reports/C85TR1_EXECUTION_GUARD_RNG_PERSISTENCE_AND_PROOF_REVIEW_PROTOCOL.json",
        "oaci/reports/C85TR2_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_PROTOCOL.json",
    )
    protocol_identities = [
        {"path": relative, "sha256": _sha(repo_root / relative)}
        for relative in protocol_paths
    ]
    lock = {
        "schema_version": "c85t_execution_lock_v3",
        "milestone": "C85TR2",
        "created_at_utc": created_at_utc,
        "status": LOCK_STATUS,
        "authorized": False,
        "execution_scope": "REGISTERED_C85T",
        "implementation_commit": implementation_commit,
        "execution_lock_commit_binding": "DISCOVER_FROM_GIT_PATH_AND_BIND_IN_FUTURE_AUTHORIZATION",
        "protocol_identities": protocol_identities,
        "c85p_protocol_path": protocol_paths[0],
        "c85r_repair_protocol_path": protocol_paths[1],
        "v2_generator_path": protocol_paths[2],
        "c85tl_operationalization_path": protocol_paths[3],
        "c85tr1_repair_protocol_path": protocol_paths[4],
        "c85tr2_repair_protocol_path": protocol_paths[5],
        "historical_execution_locks": [
            {"path": "oaci/reports/C85T_EXECUTION_LOCK.json", "status": "SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION"},
            {"path": "oaci/reports/C85T_EXECUTION_LOCK_V2.json", "sha256": HISTORICAL_V2_LOCK_SHA256, "status": "SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION"},
        ],
        "runtime_bound_object_count": len(rows),
        "runtime_bound_registry": registry_identity,
        "bound_repository_objects": rows,
        "environment": {
            "enforce_exact": True,
            "prefix": "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact",
            "python": "3.13.7",
            "numpy_runtime": "2.4.4",
            "numpy_metadata_first_match": "2.3.3",
            "bit_generator": "PCG64DXSM",
            "GPU": 0,
        },
        "authorization_record_path": "oaci/reports/C85T_V3_PI_AUTHORIZATION_RECORD.json",
        "authorization_schema": "c85t_direct_pi_authorization_record_v3",
        "authorization_consumption_root": "/projects/EEG-foundation-model/yinghao/oaci-c85t-authorization-consumption-v3",
        "output_root_policy": {
            "parent": "/projects/EEG-foundation-model/yinghao/oaci-c85t-synthetic-v3",
            "basename": "c85t-v3-{lock_sha16}-{authorization_id16}",
            "exact_absolute_binding_required": True,
        },
        "execution_context": {
            "class": "ValidatedC85TExecutionContext",
            "factory": "create_validated_c85t_execution_context",
            "preparsed_mapping_sufficient": False,
            "external_O_EXCL_receipt": True,
            "receipt_file_and_directory_fsync": True,
            "security_claim": "OFFICIAL_RESULT_GOVERNANCE_NOT_OS_SECURITY",
        },
        "transaction": {
            "bundle_schema": "c85t_atomic_execution_bundle_v3",
            "manifest_schema": "c85t_atomic_result_manifest_v3",
            "lifecycle_schema": "c85t_append_only_lifecycle_ledger_v3",
            "completion_schema": "c85t_execution_completion_receipt_v3",
            "final_rename_count": 1,
            "required_operations_after_rename": 0,
            "post_rename_recovery": True,
        },
        "result": {
            "schema": "c85t_synthetic_validation_and_proof_candidates_result_v3",
            "success_gate": "C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED",
            "semantic_replay": "ARTIFACT_DERIVED_AND_DETERMINISTIC_S9_DIGEST_REPLAY",
            "scenario_count": 11,
            "S6_S7_rows": 8192,
            "S9_design_rows": 8192,
            "S9_digest_rows": 4096,
            "proof_candidates": 7,
            "formal_status": "OPEN",
        },
        "proof_governance": {
            "C85T_candidates_only": True,
            "formal_status": "OPEN",
            "automatic_transition": False,
            "C85V_authorized": False,
        },
        "rng": {
            "namespace": "C85_SYNTHETIC_V1",
            "raw_S9_dtype": "<i8",
            "draw_order": "51_L_then_46_H",
            "replicates": 4096,
        },
        "entrypoint": "python -m oaci.theory.c85t_execute_v3 run-locked --execution-lock <V3_LOCK> --authorization-record <COMMITTED_AUTHORIZATION> --output-root <EXACT_AUTHORIZED_ROOT>",
        "resources": {"CPU": 1, "GPU": 0, "RAM_GiB": 8, "wall_minutes": 30, "storage_MiB": 64},
        "readiness": {
            "registered_S0_S10_draws": 0,
            "canonical_proof_candidates": 0,
            "theorem_status_transitions": 0,
            "authorization_records": 0,
            "success_gate": SUCCESS_GATE,
            "failure_gate": FAILURE_GATE,
        },
        "forbidden": {
            "real_project_data": True,
            "active_acquisition": True,
            "C85V": True,
            "C85E": True,
            "new_data_or_model_zoo": True,
            "manuscript_work": True,
        },
    }
    lock_path.write_bytes(canonical_json_bytes(lock))
    digest = _sha(lock_path)
    sidecar.write_text(f"{digest}  {lock_path.name}\n")
    return {
        "lock_path": str(lock_path),
        "lock_sha256": digest,
        "runtime_bound_object_count": len(rows),
        "registered_S0_S10_draws": 0,
        "canonical_proof_candidates": 0,
        "theorem_status_transitions": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    tables = commands.add_parser("build-contract-tables")
    tables.add_argument("--repo-root", type=Path, required=True)
    lock = commands.add_parser("build-execution-lock-v3")
    lock.add_argument("--repo-root", type=Path, required=True)
    lock.add_argument("--implementation-commit", required=True)
    lock.add_argument("--created-at-utc", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-contract-tables":
        result = materialize_contract_tables(args.repo_root.resolve())
    else:
        result = build_execution_lock_v3(
            args.repo_root.resolve(),
            implementation_commit=args.implementation_commit,
            created_at_utc=args.created_at_utc,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
