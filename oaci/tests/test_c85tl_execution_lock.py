"""Chronology, isolation, and execution-lock tests for C85TL readiness."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from oaci.theory.c85_decision_experiments import DecisionContractError
from oaci.theory import c85t_execute as execute
from oaci.theory import c85t_rng as rng


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci" / "reports"
TABLES = REPORTS / "c85tl_tables"
PROTOCOL = REPORTS / "C85T_PROOF_AND_SYNTHETIC_EXECUTION_OPERATIONALIZATION_PROTOCOL.json"
PROTOCOL_SIDECAR = PROTOCOL.with_suffix(".sha256")
LOCK = REPORTS / "C85T_EXECUTION_LOCK.json"
EXPECTED_PROTOCOL_SHA256 = "6543d6ebbfccb8158f8f48a4fe6409c6243a708bbb0358d350932dd249e6b7c2"
EXPECTED_PROTOCOL_COMMIT = "7e8ffdffcbd8aef5a59e6bfa9a2fe0c5aa20a28f"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_operationalization_protocol_hash_and_chronology_replay() -> None:
    assert _sha(PROTOCOL) == EXPECTED_PROTOCOL_SHA256
    assert PROTOCOL_SIDECAR.read_text().split()[0] == EXPECTED_PROTOCOL_SHA256
    assert _git("log", "-1", "--format=%H", "--", str(PROTOCOL.relative_to(ROOT))) == EXPECTED_PROTOCOL_COMMIT
    for name in (
        "c85t_rng.py",
        "c85t_exact_scenarios.py",
        "c85t_monte_carlo.py",
        "c85t_proofs.py",
        "c85t_result_manifest.py",
        "c85t_execute.py",
    ):
        commit = _git("log", "-1", "--format=%H", "--", f"oaci/theory/{name}")
        assert subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_PROTOCOL_COMMIT, commit],
            cwd=ROOT,
            check=False,
        ).returncode == 0
        assert commit != EXPECTED_PROTOCOL_COMMIT


def test_exact_numpy_rng_environment_and_dual_metadata_replay() -> None:
    observed = rng.validate_environment(strict_prefix=True)
    assert observed["python"] == "3.13.7"
    assert observed["numpy_runtime"] == "2.4.4"
    assert observed["numpy_metadata_first_match"] == "2.3.3"
    assert observed["bit_generator"] == "PCG64DXSM"
    assert len(observed["bound_files"]) == 11


def test_scenario_modes_and_statuses_remain_prospective() -> None:
    modes = _csv("scenario_execution_mode_registry.csv")
    transitions = _csv("theorem_transition_contract.csv")
    assert [row["scenario_id"] for row in modes] == [f"S{i}" for i in range(11)]
    assert all(row["executed_in_C85TL"] == "0" for row in modes)
    assert len(transitions) == 7
    assert all(row["historical_status"] == "OPEN" for row in transitions)
    assert all(row["simulation_can_prove"] == row["citation_alone_can_prove"] == "0" for row in transitions)


def test_required_readiness_table_set_is_complete() -> None:
    expected = {
        "scenario_execution_mode_registry.csv",
        "rng_and_draw_order_contract.csv",
        "deterministic_seed_and_raw_draw_replay.csv",
        "S9_estimator_selection_topk_contract.csv",
        "S9_simulation_output_schema.csv",
        "S6_S7_execution_output_schema.csv",
        "S8_exact_LP_output_schema.csv",
        "T6_candidate_alpha_region.csv",
        "theorem_transition_contract.csv",
        "proof_artifact_schema.csv",
        "proof_red_team_contract.csv",
        "result_table_registry.csv",
        "runtime_bound_object_registry.csv",
        "risk_register.csv",
        "failure_reason_ledger.csv",
    }
    assert {path.name for path in TABLES.glob("*.csv")} == expected
    assert all(_csv(name) for name in expected)


def test_c85t_modules_have_no_real_project_or_active_imports() -> None:
    forbidden_roots = {"torch", "mne", "moabb", "pandas"}
    forbidden_oaci = {"multidataset", "train", "models", "methods", "data"}
    for path in sorted((ROOT / "oaci" / "theory").glob("c85t_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in forbidden_roots for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module.split(".")[0] not in forbidden_roots
                if module.startswith("oaci."):
                    assert module.split(".")[1] not in forbidden_oaci
        source = path.read_text()
        assert "/projects/EEG-foundation-model" not in source
        assert "target_construction_label_view" not in source
        assert "active_acquisition_execute" not in source


def test_execution_lock_self_hash_and_bound_bytes_replay() -> None:
    lock, lock_sha = execute.replay_execution_lock(LOCK)
    observed = execute.replay_bound_repository_objects(lock)
    assert lock_sha == LOCK.with_suffix(".sha256").read_text().split()[0]
    assert observed["object_count"] == lock["runtime_bound_object_count"]
    assert lock["status"] == execute.LOCK_STATUS
    assert lock["authorized"] is False


def test_runtime_registry_exactly_matches_lock_bound_objects() -> None:
    lock = json.loads(LOCK.read_text())
    registry = _csv("runtime_bound_object_registry.csv")
    by_path = {row["path"]: row for row in registry}
    assert len(by_path) == len(registry) == lock["runtime_bound_object_count"]
    assert {row["path"] for row in lock["bound_repository_objects"]} == set(by_path)
    for row in lock["bound_repository_objects"]:
        registered = by_path[row["path"]]
        assert registered["sha256"] == row["sha256"]
        assert registered["git_blob"] == row["git_blob"]
        assert int(registered["size_bytes"]) == row["size_bytes"]


def test_lock_commit_is_after_implementation_and_current_head() -> None:
    lock = json.loads(LOCK.read_text())
    lock_commit = _git("log", "-1", "--format=%H", "--", str(LOCK.relative_to(ROOT)))
    assert lock_commit
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock["implementation_commit"], lock_commit],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock_commit, _git("rev-parse", "HEAD")],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def test_lock_is_not_authorized_and_no_result_or_proof_artifact_exists() -> None:
    lock = json.loads(LOCK.read_text())
    assert lock["authorization_record_path"] == "oaci/reports/C85T_PI_AUTHORIZATION_RECORD.json"
    assert not (REPORTS / "C85T_PI_AUTHORIZATION_RECORD.json").exists()
    assert not (REPORTS / "C85T_RESULT.json").exists()
    assert not (REPORTS / "C85_SYNTHETIC_SCIENTIFIC_RESULT.json").exists()
    assert not (REPORTS / "c85t_proofs").exists()
    assert lock["forbidden"]["C85E"] is True
    assert lock["forbidden"]["active_acquisition"] is True
    assert lock["forbidden"]["manuscript_work"] is True


def test_missing_future_authorization_fails_before_execution(tmp_path: Path) -> None:
    lock, lock_sha = execute.replay_execution_lock(LOCK)
    lock_commit = _git("log", "-1", "--format=%H", "--", str(LOCK.relative_to(ROOT)))
    with pytest.raises(DecisionContractError, match="authorization record is absent"):
        execute.replay_authorization(
            REPORTS / "C85T_PI_AUTHORIZATION_RECORD.json",
            lock,
            lock_sha,
            lock_commit,
        )
    assert not (tmp_path / "result").exists()


def test_c85tl_tests_are_in_focused_and_cumulative_suites() -> None:
    from oaci.multidataset import c84r_regression_suite as suites

    names = {"test_c85t_shadow_execution.py", "test_c85tl_execution_lock.py"}
    for suite in ("focused", "c65", "c23"):
        assert names <= {path.name for path in suites.suite_files(suite)}
    assert suites.suite_files("full") == [suites.TEST_DIR]

