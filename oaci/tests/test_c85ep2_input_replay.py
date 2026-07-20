"""C85EP2 chronology, accepted C85U replay, and runtime-input isolation."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from oaci.theory.c85ep2_input_acceptance import EXPECTED


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c85ep2_tables"
PROTOCOL = REPORTS / "C85EP2_EXECUTABLE_SEMANTICS_AND_INPUT_REPLAY_PROTOCOL.json"
PROTOCOL_SHA = "abbb110de2ad651534f115937198987248f719ba8059d4cc300344db1b784516"
PROTOCOL_COMMIT = "29dcf67e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_executable_semantics_protocol_precedes_replay_and_implementation() -> None:
    assert _sha(PROTOCOL) == PROTOCOL_SHA
    assert PROTOCOL.with_suffix(".sha256").read_text().split()[0] == PROTOCOL_SHA
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(PROTOCOL.relative_to(ROOT))],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert commit.startswith(PROTOCOL_COMMIT)
    replay_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "oaci/theory/c85ep2_input_acceptance.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, replay_commit],
        cwd=ROOT, check=False,
    ).returncode == 0


def test_c85u_acceptance_certificate_has_exact_identity_and_coverage_only() -> None:
    path = REPORTS / "C85EP2_C85U_INPUT_ACCEPTANCE_CERTIFICATE.json"
    certificate = json.loads(path.read_text())
    assert certificate["status"] == "PASS_C85U_INPUT_ACCEPTED_FOR_C85E_LOCK_READINESS"
    assert certificate["public_output_excludes_scientific_values"] is True
    assert certificate["C85E_executed"] is False
    assert certificate["U1_utility_field_replay"]["contexts"] == 944
    assert certificate["U1_utility_field_replay"]["candidate_rows"] == 76_464
    assert certificate["U1_utility_field_replay"]["context_artifacts"] == 944
    assert certificate["U2_historical_endpoint_replay"]["method_context_rows"] == 18_432
    assert certificate["U2_historical_endpoint_replay"]["finite_Q0_action_records"] == 8_749_056
    assert certificate["U2_historical_endpoint_replay"]["Q0_shards"] == 944
    assert set(certificate["U2_historical_endpoint_replay"]["maximum_absolute_differences"].values()) == {0.0}
    assert certificate["U2_historical_endpoint_replay"]["selected_regime_mismatches"] == 0
    encoded = json.dumps(certificate, sort_keys=True).lower()
    for forbidden in (
        "mean_regret", "median_regret", "cvar_", "near_optimal_set",
        "effective_multiplicity", "action_divergence_rate", "theorem_applicability",
    ):
        assert forbidden not in encoded


def test_replay_tables_bind_all_authorization_lifecycle_u1_u2_and_acceptance() -> None:
    required = {
        "c85u_authorization_lifecycle_replay.csv",
        "c85u_u1_artifact_replay.csv",
        "c85u_u2_endpoint_replay.csv",
        "c85u_acceptance_bundle_replay.csv",
    }
    assert required.issubset({path.name for path in TABLES.glob("*.csv")})
    assert all(_rows(TABLES / name)[0]["status"] == "PASS" for name in required)
    assert _rows(TABLES / "c85u_u1_artifact_replay.csv")[0]["candidate_rows"] == "76464"
    assert _rows(TABLES / "c85u_u2_endpoint_replay.csv")[0]["finite_Q0_action_records"] == "8749056"


def test_frozen_input_registry_is_complete_and_has_no_direct_data_path() -> None:
    rows = _rows(TABLES / "c85e_frozen_input_registry.csv")
    assert len(rows) == 1_955
    assert len({row["object_id"] for row in rows}) == len(rows)
    assert len({row["path"] for row in rows}) == len(rows)
    assert sum(row["semantic_role"] == "C85U_HELD_EVALUATION_UTILITY_CONTEXT" for row in rows) == 944
    assert sum(row["semantic_role"] == "C84S_FROZEN_Q0_ACTION_SHARD" for row in rows) == 944
    assert all(row["runtime_access"] == "READ_ONLY" for row in rows)
    lowered = "\n".join(row["path"].lower() for row in rows)
    for forbidden in (
        "target_evaluation_label_view", "target_construction_label_view",
        "target_logits", "eeg_arrays", "source_arrays", "checkpoints",
    ):
        assert forbidden not in lowered


def test_runtime_modules_have_no_forbidden_import_or_embedded_direct_path() -> None:
    modules = (
        "c85e_policy_use.py", "c85e_action_geometry.py", "c85e_rank_topk_regret.py",
        "c85e_robust_risk.py", "c85e_theorem_bridge.py", "c85e_result_manifest.py",
        "c85e_runtime_guard.py", "c85e_execute.py",
    )
    forbidden_imports = (
        "torch", "mne", "moabb", "c84s_label_views", "c84s_q0_budget",
        "c84s_inference", "c84s_selectors", "c84sr3_stage_c_evaluation",
    )
    for name in modules:
        source = (ROOT / "oaci/theory" / name).read_text()
        tree = ast.parse(source)
        imports = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
        assert not any(value.startswith(forbidden_imports) for value in imports)
        if name != "c85e_runtime_guard.py":
            assert "target_evaluation_label_view" not in source
            assert "target_construction_label_view" not in source


def test_input_identities_match_accepted_c85u_and_c84s_values() -> None:
    registry = {row["object_id"]: row for row in _rows(TABLES / "c85e_frozen_input_registry.csv")}
    assert registry["C85U_U1_C85U_CANDIDATE_UTILITY_MANIFEST_V2.json"]["sha256"] == EXPECTED["u1_manifest_sha256"]
    assert registry["C85U_U2_C85U_HISTORICAL_DECISION_REPLAY_V2.json"]["sha256"] == EXPECTED["u2_result_sha256"]
    assert registry["C84S_STAGE_B_C84S_SELECTION_FREEZE_MANIFEST_V3.json"]["sha256"] == EXPECTED["selection_manifest_sha256"]
    assert registry["C84S_RESULT_C84S_RESULT_ARTIFACT_MANIFEST.json"]["sha256"] == EXPECTED["result_manifest_sha256"]
