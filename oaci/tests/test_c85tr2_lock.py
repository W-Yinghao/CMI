"""Chronology, byte replay, and no-execution checks for the C85T V3 lock."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from oaci.theory.c85t_execution_context_v3 import replay_execution_lock_v3


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c85tr2_tables"
PROTOCOL = REPORTS / "C85TR2_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_PROTOCOL.json"
LOCK = REPORTS / "C85T_EXECUTION_LOCK_V3.json"
EXPECTED_PROTOCOL_SHA = "f9a1db908f34818b7551c0d4f8de65fa7a11e71c41b8e5fe28824f042904a844"
EXPECTED_PROTOCOL_COMMIT = "2e79f304202faffb857610e273ec5510a608080a"
HISTORICAL_V2_LOCK_SHA = "0f6907f9b997634b48062e97e789d50ae189dcc3c01f03cb5cee8b105c379719"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_c85tr2_protocol_hash_and_chronology() -> None:
    assert _sha(PROTOCOL) == EXPECTED_PROTOCOL_SHA
    assert PROTOCOL.with_suffix(".sha256").read_text().split()[0] == EXPECTED_PROTOCOL_SHA
    assert _git("log", "-1", "--format=%H", "--", str(PROTOCOL.relative_to(ROOT))) == EXPECTED_PROTOCOL_COMMIT
    lock_commit = _git("log", "-1", "--format=%H", "--", str(LOCK.relative_to(ROOT)))
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_PROTOCOL_COMMIT, lock_commit],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def test_historical_v2_lock_is_exact_and_superseded_without_execution() -> None:
    historical = REPORTS / "C85T_EXECUTION_LOCK_V2.json"
    assert _sha(historical) == HISTORICAL_V2_LOCK_SHA
    rows = _rows("historical_v2_lock_supersession.csv")
    assert len(rows) == 1
    assert rows[0]["status"] == "SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION"
    assert rows[0]["authorization_consumed"] == "0"
    assert rows[0]["registered_S0_S10_draws"] == "0"
    assert rows[0]["historical_bytes_modified"] == "0"


def test_v3_lock_and_every_bound_byte_and_git_blob_replay() -> None:
    lock, lock_sha, repo_root = replay_execution_lock_v3(LOCK)
    assert repo_root == ROOT
    assert lock_sha == LOCK.with_suffix(".sha256").read_text().split()[0]
    assert lock["status"] == "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
    assert lock["authorized"] is False
    assert lock["execution_scope"] == "REGISTERED_C85T"
    assert lock["runtime_bound_object_count"] == len(lock["bound_repository_objects"])
    assert lock["historical_execution_locks"][1]["sha256"] == HISTORICAL_V2_LOCK_SHA


def test_runtime_registry_matches_v3_lock_exactly() -> None:
    lock = json.loads(LOCK.read_text())
    registry = _rows("runtime_bound_object_registry.csv")
    assert len(registry) == lock["runtime_bound_object_count"]
    by_path = {row["path"]: row for row in registry}
    assert set(by_path) == {row["path"] for row in lock["bound_repository_objects"]}
    for row in lock["bound_repository_objects"]:
        assert by_path[row["path"]]["sha256"] == row["sha256"]
        assert by_path[row["path"]]["git_blob"] == row["git_blob"]


def test_required_c85tr2_contract_tables_are_complete() -> None:
    required = {
        "historical_v2_lock_supersession.csv",
        "validated_execution_context_contract.csv",
        "unauthorized_internal_api_adversarial_tests.csv",
        "atomic_transaction_state_machine.csv",
        "post_rename_recovery_truth_table.csv",
        "failure_exception_precedence.csv",
        "result_semantic_replay_v3.csv",
        "atomic_bundle_schema_v3.csv",
        "runtime_bound_object_registry.csv",
        "risk_register.csv",
        "failure_reason_ledger.csv",
    }
    assert {path.name for path in TABLES.glob("*.csv")} == required
    assert all(_rows(name) for name in required)


def test_v3_lock_binds_receipt_transaction_and_semantic_replay() -> None:
    lock = json.loads(LOCK.read_text())
    context = lock["execution_context"]
    assert context["class"] == "ValidatedC85TExecutionContext"
    assert context["preparsed_mapping_sufficient"] is False
    assert context["external_O_EXCL_receipt"] is True
    assert context["receipt_file_and_directory_fsync"] is True
    transaction = lock["transaction"]
    assert transaction["final_rename_count"] == 1
    assert transaction["required_operations_after_rename"] == 0
    assert transaction["post_rename_recovery"] is True
    assert lock["result"]["scenario_count"] == 11
    assert lock["result"]["S6_S7_rows"] == 8192
    assert lock["result"]["S9_design_rows"] == 8192
    assert lock["result"]["S9_digest_rows"] == 4096
    assert lock["result"]["formal_status"] == "OPEN"


def test_no_authorization_result_proof_or_status_transition_exists() -> None:
    lock = json.loads(LOCK.read_text())
    assert not (ROOT / lock["authorization_record_path"]).exists()
    assert not (REPORTS / "C85T_RESULT.json").exists()
    assert not (REPORTS / "c85t_proof_candidates").exists()
    assert lock["readiness"]["registered_S0_S10_draws"] == 0
    assert lock["readiness"]["canonical_proof_candidates"] == 0
    assert lock["readiness"]["theorem_status_transitions"] == 0
    assert lock["readiness"]["authorization_records"] == 0
    assert lock["proof_governance"]["formal_status"] == "OPEN"
    assert lock["proof_governance"]["automatic_transition"] is False
    assert lock["proof_governance"]["C85V_authorized"] is False


def test_v3_modules_have_no_real_data_import_or_superseded_mint_api() -> None:
    paths = [
        ROOT / "oaci/theory/c85t_execution_context_v3.py",
        ROOT / "oaci/theory/c85t_registered_v3.py",
        ROOT / "oaci/theory/c85t_semantic_replay_v3.py",
        ROOT / "oaci/theory/c85t_transaction_v3.py",
        ROOT / "oaci/theory/c85t_execute_v3.py",
    ]
    source = "\n".join(path.read_text() for path in paths)
    for forbidden in (
        "_CAPABILITY_SENTINEL",
        "_ISSUED_CAPABILITIES",
        "_issue_capability",
        "consume_authorization_once",
        "INDEPENDENT_PROOF_RED_TEAM",
    ):
        assert forbidden not in source
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in {"torch", "mne", "moabb"} for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"torch", "mne", "moabb"}


def test_c85tr2_tests_are_in_required_regression_paths() -> None:
    from oaci.multidataset import c84r_regression_suite as suites

    expected = {
        "test_c85tr2_execution_context.py",
        "test_c85tr2_atomic_transaction.py",
        "test_c85tr2_semantic_replay.py",
        "test_c85tr2_lock.py",
    }
    for suite in ("c65", "c23"):
        assert expected <= {path.name for path in suites.suite_files(suite)}
    wrapper = (ROOT / "oaci/slurm_c85tr2_regression.sh").read_text()
    assert all(name in wrapper for name in expected)
    assert suites.suite_files("full") == [suites.TEST_DIR]
