"""C85EP manifest-only availability and fail-closed blocker tests."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from oaci.theory import c85e_input_replay as replay


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c85ep_tables"
PROTOCOL = REPORTS / "C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_PROTOCOL.json"
ADDENDUM = REPORTS / "C85V_PM_THEOREM_SCOPE_AND_REVIEW_INDEPENDENCE_ADDENDUM.json"
PROTOCOL_SHA = "a42cc71498971ee6eeb75ef53e62744e73e91b92e444ef78c9e4c856d61ac052"
PROTOCOL_COMMIT = "0af9f286c31e70beded08ae6143a01e2dd2430ee"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_protocol_is_committed_before_availability_implementation() -> None:
    assert _sha(PROTOCOL) == PROTOCOL_SHA
    assert PROTOCOL.with_suffix(".sha256").read_text().split()[0] == PROTOCOL_SHA
    observed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(PROTOCOL.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert observed == PROTOCOL_COMMIT
    source_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "oaci/theory/c85e_input_replay.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert source_commit == "" or subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, source_commit],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def test_pm_addendum_preserves_exact_status_and_review_scope() -> None:
    value = json.loads(ADDENDUM.read_text())
    assert value["formal_statuses"] == {
        "T1": "PROVED",
        "T2": "COUNTEREXAMPLE",
        "T3": "PROVED",
        "T4": "PROVED",
        "T5": "OPEN",
        "T6": "COUNTEREXAMPLE",
        "T7": "PROVED",
    }
    assert value["review_independence"]["external_human_peer_review"] is False
    assert value["scope_contract"]["C85_theorem_establishes_C84_mechanism"] is False
    assert value["C85E_authorized"] is False


def test_manifest_only_availability_gate_fails_on_missing_candidate_utility() -> None:
    audit = replay.audit_frozen_input_availability()
    assert audit["status"] == "BLOCKED"
    assert audit["gate"] == replay.FAILURE_GATE
    assert audit["execution_lock_permitted"] is False
    assert audit["failures"] == [
        "A1_COMPLETE_CANDIDATE_UTILITY",
        "A6_NO_LABEL_ROOT_OR_STAGE_C_REOPEN",
    ]
    assert audit["candidate_level_objects_opened"] == 0
    assert audit["chain_level_objects_opened"] == 0
    assert audit["direct_result_tables_opened"] == 0
    assert audit["direct_label_or_field_arrays_opened"] == 0


def test_availability_evidence_records_exact_missing_object() -> None:
    rows = {row["requirement_id"]: row for row in _rows(TABLES / "frozen_input_availability_audit.csv")}
    assert set(rows) == {
        "A1_COMPLETE_CANDIDATE_UTILITY",
        "A2_CANDIDATE_ID_AND_ORDER_METADATA",
        "A3_IMMUTABLE_SELECTION_ACTIONS",
        "A4_CONTEXT_IDENTITY",
        "A5_FROZEN_TARGET_INFERENCE_COMPONENTS",
        "A6_NO_LABEL_ROOT_OR_STAGE_C_REOPEN",
    }
    assert rows["A1_COMPLETE_CANDIDATE_UTILITY"]["status"] == "FAIL_ABSENT_FROM_FROZEN_MANIFEST"
    assert "76464" in rows["A1_COMPLETE_CANDIDATE_UTILITY"]["expected"]
    assert rows["A1_COMPLETE_CANDIDATE_UTILITY"]["full_object_opened"] == "0"
    assert rows["A1_COMPLETE_CANDIDATE_UTILITY"]["reconstruction_forbidden"] == "1"
    assert rows["A2_CANDIDATE_ID_AND_ORDER_METADATA"]["status"].startswith("PASS")
    assert rows["A3_IMMUTABLE_SELECTION_ACTIONS"]["status"].startswith("PASS")


def test_input_identity_registry_uses_metadata_only_access_classes() -> None:
    rows = _rows(TABLES / "frozen_input_identity_registry.csv")
    assert len(rows) == 9
    assert {row["status"] for row in rows} == {
        "PASS",
        "BOUND_IDENTITY_ONLY_NOT_REHASHED",
    }
    identity_only = [row for row in rows if row["access_class"].startswith("IDENTITY_ONLY")]
    assert len(identity_only) == 2
    assert all(row["observed_sha256"] == "NOT_REHASHED_C85EP" for row in identity_only)
    assert {row["access_class"] for row in rows}.issubset(
        {
            "IDENTITY_ONLY_NOT_OPENED",
            "MANIFEST_METADATA_OPENED",
            "COMMITTED_COMPACT_REPORT_OPENED",
            "COMMITTED_SCHEMA_METADATA_OPENED",
            "COMMITTED_PROTOCOL_OPENED",
        }
    )
    assert not any("method_context_decisions.csv" in row["path"] for row in rows)
    assert not any("q0_shards" in row["path"] for row in rows)


def test_protocol_locks_grids_aggregation_and_theorem_guards() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["geometry_contract"]["epsilon_grid"] == [0.005, 0.01, 0.02, 0.05]
    assert protocol["geometry_contract"]["tau_grid"] == [0.005, 0.01, 0.02, 0.05, 0.1]
    assert protocol["robust_risk_contract"]["CVaR_alpha_grid"] == [0.5, 0.75, 0.9]
    assert protocol["aggregation_contract"]["target_equal_weight"] is True
    assert protocol["aggregation_contract"]["dataset_pooling"] is False
    assert protocol["theorem_applicability_contract"]["minimum_classification"]["T5"] == "OPEN_THEOREM"
    assert protocol["theorem_applicability_contract"]["theorem_status_transitions"] is False


def test_availability_module_has_no_forbidden_runtime_import_or_path() -> None:
    source_path = ROOT / "oaci/theory/c85e_input_replay.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    forbidden = ("torch", "mne", "moabb", "oaci.train", "c84s_label_views", "c84sr3_stage_c_evaluation")
    assert not any(name.startswith(forbidden) for name in imports)
    assert "target_construction_label_view" not in source
    assert "target_evaluation_label_view" not in source
    assert "np.load" not in source
    assert "read_context_shard" not in source


def test_no_c85e_lock_authorization_or_analysis_implementation_exists() -> None:
    assert not (REPORTS / "C85E_EXECUTION_LOCK.json").exists()
    assert not (REPORTS / "C85E_PI_AUTHORIZATION_RECORD.json").exists()
    for name in (
        "c85e_policy_use.py",
        "c85e_action_geometry.py",
        "c85e_robust_risk.py",
        "c85e_theorem_bridge.py",
        "c85e_result_manifest.py",
        "c85e_execute.py",
    ):
        assert not (ROOT / "oaci/theory" / name).exists()


def test_c84_and_c85_formal_results_remain_immutable() -> None:
    c84s = json.loads((REPORTS / "C84S_OVERALL_REPORT.json").read_text())
    c85v = json.loads((REPORTS / "C85V_OVERALL_REPORT.json").read_text())
    assert c84s["primary_gate"] == "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous"
    assert c84s["label_frontier_tag"] == "C84-L4"
    assert c85v["theorem_statuses"]["T5"] == "OPEN"
    assert c85v["protected_boundaries"]["C85E_authorized"] is False
