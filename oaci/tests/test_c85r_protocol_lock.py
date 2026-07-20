"""Chronology, isolation, and fail-closed tests for C85R readiness."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from oaci.theory import c85r_synthetic_semantic_repair as repair


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci" / "reports"
THEORY = ROOT / "oaci" / "theory"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_c85p_objects_remain_byte_identical() -> None:
    assert _sha(REPORTS / "C85_TPAMI_DECISION_THEORY_PROTOCOL.json") == repair.c85p.EXPECTED_PROTOCOL_SHA256
    assert _sha(REPORTS / "c85p_tables" / "synthetic_generator_contract.json") == repair.EXPECTED_HISTORICAL_SHA256
    assert sum(repair.c85p.validate_materialized_tables().values()) == 193


def test_repair_protocol_precedes_v2_and_implementation() -> None:
    protocol_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(repair.REPAIR_PROTOCOL_PATH.relative_to(ROOT))],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert protocol_commit == repair.EXPECTED_REPAIR_PROTOCOL_COMMIT
    for path in (repair.V2_CONTRACT_PATH, Path(repair.__file__)):
        implementation_commit = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        if implementation_commit:
            assert subprocess.run(
                ["git", "merge-base", "--is-ancestor", protocol_commit, implementation_commit],
                cwd=ROOT, check=False,
            ).returncode == 0


def test_repair_protocol_and_v2_hashes_replay() -> None:
    locked = repair.validate_locked_contracts()
    assert locked["repair_protocol_sha256"] == repair.EXPECTED_REPAIR_PROTOCOL_SHA256
    assert locked["v2_sha256"] == repair.EXPECTED_V2_CONTRACT_SHA256
    assert locked["historical_sha256"] == repair.EXPECTED_HISTORICAL_SHA256


def test_every_t1_t7_status_remains_open() -> None:
    locked = repair.validate_locked_contracts()
    assert locked["v2"]["theorem_statuses"] == {f"T{i}": "OPEN" for i in range(1, 8)}
    with (repair.C85R_TABLE_DIR / "theorem_status_replay.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert all(row["historical_status"] == row["V2_status"] == "OPEN" for row in rows)
    assert all(row["proof_executed_C85R"] == row["status_transition"] == "0" for row in rows)


def test_all_s0_s10_seed_identities_remain_historical() -> None:
    with (repair.C85R_TABLE_DIR / "deterministic_seed_replay.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["scenario_id"] for row in rows] == [f"S{i}" for i in range(11)]
    for row in rows:
        scenario_id = row["scenario_id"]
        assert int(row["replicate_0_seed"]) == repair.c85p.deterministic_seed(scenario_id, 0)
        assert int(row["replicate_1_seed"]) == repair.c85p.deterministic_seed(scenario_id, 1)
        assert int(row["replicate_4095_seed"]) == repair.c85p.deterministic_seed(scenario_id, 4095)
        assert row["seed_rule_changed"] == row["scientific_draw_generated"] == "0"


def test_no_c85t_result_or_authorized_execution_exists() -> None:
    forbidden = [
        "C85T_RESULT.json",
        "C85_SYNTHETIC_SCIENTIFIC_RESULT.json",
        "C85E_EXECUTION_LOCK.json",
        "C85_ACTIVE_ACQUISITION_LOCK.json",
        "C85_REAL_DATA_AUTHORIZATION.json",
    ]
    assert not any((REPORTS / name).exists() for name in forbidden)
    theory_lock = REPORTS / "C85T_EXECUTION_LOCK.json"
    if theory_lock.exists():
        value = json.loads(theory_lock.read_text())
        assert value["status"] == "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
        assert value["authorized"] is False


def test_no_active_acquisition_is_authorized() -> None:
    with (REPORTS / "c85p_tables" / "active_testing_method_registry.csv").open(newline="") as handle:
        methods = list(csv.DictReader(handle))
    assert all(row["authorized"] == "0" for row in methods)
    protocol = json.loads(repair.REPAIR_PROTOCOL_PATH.read_text())
    assert protocol["authorization_boundary"]["active_acquisition_authorized"] is False
    assert protocol["authorization_boundary"]["C85T_authorized"] is False
    assert protocol["authorization_boundary"]["C85E_authorized"] is False


def test_repair_module_has_no_real_project_or_empirical_import() -> None:
    path = Path(repair.__file__)
    tree = ast.parse(path.read_text(), filename=str(path))
    forbidden_roots = {"torch", "mne", "moabb", "numpy", "scipy"}
    forbidden_oaci = {"multidataset", "train", "methods", "models"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.split(".")[0] in forbidden_roots for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[0] not in forbidden_roots
            if module.startswith("oaci."):
                assert module.split(".")[1] not in forbidden_oaci
    text = path.read_text()
    assert "/projects/EEG-foundation-model" not in text
    assert "target_construction_label_view" not in text
    assert "C84S_RESULT" not in text


def test_semantic_status_never_claims_scientific_execution() -> None:
    value = json.loads(repair.V2_CONTRACT_PATH.read_text())
    semantic = value["semantic_validation"]
    assert semantic["status"] == repair.SEMANTIC_STATUS
    assert semantic["scientific_simulation_executed"] is False
    assert semantic["proof_executed"] is False


def test_c85r_tests_are_in_focused_and_cumulative_suites() -> None:
    from oaci.multidataset import c84r_regression_suite as suites

    names = {"test_c85r_synthetic_semantic_repair.py", "test_c85r_protocol_lock.py"}
    for suite in ("focused", "c65", "c23"):
        assert names <= {path.name for path in suites.suite_files(suite)}
    assert suites.suite_files("full") == [suites.TEST_DIR]
