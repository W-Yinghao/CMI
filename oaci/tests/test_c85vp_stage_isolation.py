"""Shadow Stage-A/Stage-B isolation and immutable-input tests for C85VP."""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from oaci.tests.c85vp_test_support import (
    shadow_obligations,
    shadow_statements,
    write_shadow_candidates,
    write_shadow_exact_results,
)
from oaci.theory.c85_decision_experiments import DecisionContractError
from oaci.theory.c85v_stage_a_derivation import freeze_stage_a_derivations
from oaci.theory.c85v_stage_b_candidate_audit import freeze_stage_b_comparisons
from oaci.theory.c85vp_readiness import IMPLEMENTATION_PATHS, static_isolation_audit


ROOT = Path(__file__).resolve().parents[2]


def _shadow_stage_a(tmp_path: Path):
    statements = shadow_statements()
    root = tmp_path / "stage_a"
    manifest = freeze_stage_a_derivations(
        statements=statements,
        obligations=shadow_obligations(),
        available_source_ids=frozenset({f"V{index:02d}" for index in range(1, 7)}),
        output_root=root,
        review_mode="SHADOW_C85VP",
    )
    return statements, root, manifest


def test_stage_a_has_no_candidate_argument_or_forbidden_import() -> None:
    rows = static_isolation_audit(ROOT)
    assert len(rows) == len(IMPLEMENTATION_PATHS)
    assert all(row["status"] == "PASS" for row in rows)
    path = ROOT / "oaci/theory/c85v_stage_a_derivation.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "freeze_stage_a_derivations"
    )
    arguments = [node.arg for node in (*function.args.args, *function.args.kwonlyargs)]
    assert not any("candidate" in name.lower() for name in arguments)


def test_stage_a_freezes_seven_candidate_blind_derivations(tmp_path: Path) -> None:
    _, root, manifest = _shadow_stage_a(tmp_path)
    assert manifest["derivation_count"] == 7
    assert manifest["candidate_text_access"] == 0
    assert manifest["monte_carlo_access"] == 0
    assert manifest["formal_status_transitions"] == 0
    assert len(list(root.glob("T*_independent_derivation.json"))) == 7


def test_stage_b_cannot_run_before_complete_stage_a_freeze(tmp_path: Path) -> None:
    statements = shadow_statements()
    candidates = tmp_path / "candidates"
    identities = write_shadow_candidates(candidates, statements)
    exact = tmp_path / "exact.json"
    write_shadow_exact_results(exact)
    with pytest.raises(DecisionContractError, match="Stage A freeze"):
        freeze_stage_b_comparisons(
            stage_a_root=tmp_path / "missing_stage_a",
            candidate_bundle_root=candidates,
            exact_results_path=exact,
            output_root=tmp_path / "stage_b",
            statements=statements,
            identities=identities,
            review_mode="SHADOW_C85VP",
        )


def test_stage_b_shadow_comparison_has_no_monte_carlo_or_status_change(tmp_path: Path) -> None:
    statements, stage_a, _ = _shadow_stage_a(tmp_path)
    candidates = tmp_path / "candidates"
    identities = write_shadow_candidates(candidates, statements)
    exact = tmp_path / "exact.json"
    write_shadow_exact_results(exact)
    manifest = freeze_stage_b_comparisons(
        stage_a_root=stage_a,
        candidate_bundle_root=candidates,
        exact_results_path=exact,
        output_root=tmp_path / "stage_b",
        statements=statements,
        identities=identities,
        review_mode="SHADOW_C85VP",
    )
    assert manifest["comparison_count"] == 7
    assert manifest["adversarial_audit_count"] == 7
    assert manifest["candidate_text_files_accessed"] == 7
    assert manifest["monte_carlo_reruns"] == 0
    assert manifest["formal_status_transitions"] == 0


def test_proof_file_hash_drift_fails(tmp_path: Path) -> None:
    statements, stage_a, _ = _shadow_stage_a(tmp_path)
    candidates = tmp_path / "candidates"
    identities = write_shadow_candidates(candidates, statements)
    exact = tmp_path / "exact.json"
    write_shadow_exact_results(exact)
    (candidates / identities["T1"].relative_path).write_text("tampered")
    with pytest.raises(DecisionContractError, match="candidate hash drifted"):
        freeze_stage_b_comparisons(
            stage_a_root=stage_a,
            candidate_bundle_root=candidates,
            exact_results_path=exact,
            output_root=tmp_path / "stage_b",
            statements=statements,
            identities=identities,
            review_mode="SHADOW_C85VP",
        )


def test_statement_hash_drift_fails(tmp_path: Path) -> None:
    statements, stage_a, _ = _shadow_stage_a(tmp_path)
    candidates = tmp_path / "candidates"
    identities = write_shadow_candidates(candidates, statements)
    exact = tmp_path / "exact.json"
    write_shadow_exact_results(exact)
    drifted = dict(statements)
    drifted["T1"] = replace(statements["T1"], sha256="0" * 64)
    with pytest.raises(DecisionContractError, match="statement identity drifted"):
        freeze_stage_b_comparisons(
            stage_a_root=stage_a,
            candidate_bundle_root=candidates,
            exact_results_path=exact,
            output_root=tmp_path / "stage_b",
            statements=drifted,
            identities=identities,
            review_mode="SHADOW_C85VP",
        )
