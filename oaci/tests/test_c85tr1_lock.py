"""Chronology, byte replay, and no-execution checks for the C85T V2 lock."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from oaci.theory import c85t_execute_v2 as execute


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci" / "reports"
TABLES = REPORTS / "c85tr1_tables"
PROTOCOL = REPORTS / "C85TR1_EXECUTION_GUARD_RNG_PERSISTENCE_AND_PROOF_REVIEW_PROTOCOL.json"
LOCK = REPORTS / "C85T_EXECUTION_LOCK_V2.json"
EXPECTED_PROTOCOL_SHA = "9c0a7084a7ddd83ef96b8d7f95faf89138829729c0acc5c3d6baeb0ef87ab13d"
EXPECTED_PROTOCOL_COMMIT = "46442b281d61d00a575fae17685648b749659263"
HISTORICAL_LOCK_SHA = "4a289a46040b10855c6f23def53c328bdce0a8b1c71b7e90523887b6c1db7991"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_repair_protocol_hash_and_chronology() -> None:
    assert _sha(PROTOCOL) == EXPECTED_PROTOCOL_SHA
    assert PROTOCOL.with_suffix(".sha256").read_text().split()[0] == EXPECTED_PROTOCOL_SHA
    assert _git("log", "-1", "--format=%H", "--", str(PROTOCOL.relative_to(ROOT))) == EXPECTED_PROTOCOL_COMMIT
    lock_commit = _git("log", "-1", "--format=%H", "--", str(LOCK.relative_to(ROOT)))
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_PROTOCOL_COMMIT, lock_commit],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def test_historical_lock_is_immutable_and_superseded_before_execution() -> None:
    historical = REPORTS / "C85T_EXECUTION_LOCK.json"
    assert _sha(historical) == HISTORICAL_LOCK_SHA
    rows = _rows("historical_lock_supersession.csv")
    assert len(rows) == 1
    assert rows[0]["status"] == "SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION"
    assert rows[0]["authorization_consumed"] == rows[0]["registered_execution"] == "0"


def test_v2_lock_and_all_current_bound_bytes_replay() -> None:
    lock, lock_sha = execute.replay_execution_lock_v2(LOCK)
    replay = execute.replay_bound_repository_objects_v2(lock)
    assert lock_sha == LOCK.with_suffix(".sha256").read_text().split()[0]
    assert lock["status"] == execute.LOCK_STATUS
    assert lock["authorized"] is False
    assert replay["object_count"] == lock["runtime_bound_object_count"]
    assert lock["historical_execution_lock"]["sha256"] == HISTORICAL_LOCK_SHA


def test_runtime_registry_matches_lock_exactly() -> None:
    lock = json.loads(LOCK.read_text())
    registry = _rows("runtime_bound_object_registry.csv")
    assert len(registry) == lock["runtime_bound_object_count"]
    by_path = {row["path"]: row for row in registry}
    assert set(by_path) == {row["path"] for row in lock["bound_repository_objects"]}
    for row in lock["bound_repository_objects"]:
        assert by_path[row["path"]]["sha256"] == row["sha256"]
        assert by_path[row["path"]]["git_blob"] == row["git_blob"]


def test_required_contract_table_set_is_complete() -> None:
    required = {
        "historical_lock_supersession.csv",
        "S9_rng_dtype_reconciliation.csv",
        "shadow_rademacher_int64_replay.csv",
        "monte_carlo_interval_contract_v2.csv",
        "replicate_artifact_schema_v2.csv",
        "aggregate_from_saved_array_replay.csv",
        "authorization_single_use_contract.csv",
        "authorization_failure_truth_table.csv",
        "runtime_capability_contract.csv",
        "C85T_C85V_stage_separation.csv",
        "proof_candidate_disposition_schema.csv",
        "lifecycle_event_schema_v2.csv",
        "result_manifest_v2_contract.csv",
        "runtime_bound_object_registry.csv",
        "risk_register.csv",
        "failure_reason_ledger.csv",
    }
    assert {path.name for path in TABLES.glob("*.csv")} == required
    assert all(_rows(name) for name in required)


def test_no_v2_authorization_result_proof_or_status_transition_exists() -> None:
    lock = json.loads(LOCK.read_text())
    assert not (ROOT / lock["authorization_record_path"]).exists()
    assert not (REPORTS / "C85T_RESULT.json").exists()
    assert not (REPORTS / "c85t_proof_candidates").exists()
    assert lock["readiness"]["registered_S0_S10_draws"] == 0
    assert lock["readiness"]["canonical_proof_artifacts"] == 0
    assert lock["readiness"]["theorem_status_transitions"] == 0
    assert lock["proof_governance"]["C85T_formal_status"] == "OPEN"
    assert lock["proof_governance"]["automatic_transition"] is False


def test_operative_modules_have_no_static_token_or_real_data_import() -> None:
    paths = [
        ROOT / "oaci/theory/c85t_rng.py",
        ROOT / "oaci/theory/c85t_execution_guard.py",
        ROOT / "oaci/theory/c85t_exact_scenarios.py",
        ROOT / "oaci/theory/c85t_monte_carlo.py",
        ROOT / "oaci/theory/c85t_proofs.py",
        ROOT / "oaci/theory/c85t_result_manifest.py",
        ROOT / "oaci/theory/c85t_execute_v2.py",
    ]
    for path in paths:
        source = path.read_text()
        tree = ast.parse(source)
        assert "REGISTERED_EXECUTION_TOKEN" not in source
        assert "C85T_LOCKED_EXECUTION_AUTHORIZATION_REPLAYED" not in source
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in {"torch", "mne", "moabb"} for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"torch", "mne", "moabb"}


def test_c85tr1_tests_are_in_all_required_suites() -> None:
    from oaci.multidataset import c84r_regression_suite as suites

    expected = {
        "test_c85tr1_execution_guard.py",
        "test_c85tr1_replicate_persistence.py",
        "test_c85tr1_lock.py",
    }
    for suite in ("focused", "c65", "c23"):
        assert expected <= {path.name for path in suites.suite_files(suite)}
    assert suites.suite_files("full") == [suites.TEST_DIR]
