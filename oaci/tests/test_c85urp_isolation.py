from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from oaci.multidataset.c84s_common import C84SContractError
from oaci.theory import c85u_input_registry
from oaci.theory import c85u_runtime_guard
from oaci.theory.c85u_utility_builder import ProtectedTargetZooReader


REPO_ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(node.module or "")
    return result


def test_stage_u1_and_u2_static_process_isolation() -> None:
    theory = REPO_ROOT / "oaci/theory"
    u1_paths = [
        theory / "c85u_stage_u1.py", theory / "c85u_utility_builder.py",
        theory / "c85u_persistence.py", theory / "c85u_result_manifest.py",
    ]
    u2_paths = [
        theory / "c85u_stage_u2.py",
        theory / "c85u_historical_decision_replay.py",
    ]
    u1_imports = set().union(*(_imports(path) for path in u1_paths))
    assert not any(
        token in imported
        for imported in u1_imports
        for token in (
            "stage_b", "q0_store", "method_context_materialization",
            "analysis", "inference", "taxonomy",
        )
    )
    u2_imports = set().union(*(_imports(path) for path in u2_paths))
    assert not any(
        token in imported
        for imported in u2_imports
        for token in (
            "label_views", "field_reader", "utility_builder", "input_registry",
            "analysis", "inference", "taxonomy", "train",
        )
    )
    stage_u2 = ast.parse((theory / "c85u_stage_u2.py").read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(stage_u2) if isinstance(node, ast.FunctionDef)]
    argument_names = {
        argument.arg
        for function in functions
        for argument in (*function.args.args, *function.args.kwonlyargs)
    }
    assert not any("label" in name or "logit" in name for name in argument_names)


def test_protected_reader_requires_authorized_replay_context() -> None:
    with pytest.raises((C84SContractError, AttributeError)):
        ProtectedTargetZooReader(None)


def test_metadata_registry_has_zero_protected_access() -> None:
    registry = c85u_input_registry.build_frozen_input_registry()
    assert len(registry.contexts) == 944
    assert len(registry.target_artifact_rows) == 1944
    assert set(registry.access_counters.values()) == {0}
    assert all(row["target_artifact_opened"] == 0 for row in registry.target_artifact_rows)


def test_confirmatory_results_and_theorem_statuses_are_immutable() -> None:
    protocol = json.loads(
        (REPO_ROOT / "oaci/reports/C85U_CANDIDATE_UTILITY_RECONSTRUCTION_PROTOCOL.json")
        .read_text(encoding="utf-8")
    )
    assert protocol["immutable_results"]["C84_primary"].startswith("C84-D_")
    assert protocol["immutable_results"]["C84_frontier"] == "C84-L4"
    assert protocol["immutable_results"]["C85_theorem_statuses"] == {
        "T1": "PROVED", "T2": "COUNTEREXAMPLE", "T3": "PROVED",
        "T4": "PROVED", "T5": "OPEN", "T6": "COUNTEREXAMPLE",
        "T7": "PROVED",
    }
    assert not (REPO_ROOT / "oaci/reports/C85U_PI_AUTHORIZATION_RECORD.json").exists()
    assert not (REPO_ROOT / "oaci/reports/C85E_EXECUTION_LOCK.json").exists()


def test_single_use_receipt_uses_exclusive_create(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    c85u_runtime_guard._write_exclusive_fsynced(path, b"{}\n")
    with pytest.raises(RuntimeError, match="already consumed"):
        c85u_runtime_guard._write_exclusive_fsynced(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
