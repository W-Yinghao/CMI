"""C85VP chronology, lock replay, and shadow atomic-publication tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from oaci.tests.c85vp_test_support import (
    shadow_obligations,
    shadow_statements,
    write_shadow_candidates,
    write_shadow_exact_results,
)
from oaci.theory.c85v_adjudication import freeze_adjudication
from oaci.theory.c85v_execute import replay_execution_lock
from oaci.theory.c85v_result_manifest import (
    AtomicC85VResultBundle,
    RESULT_SCHEMA,
    SUCCESS_GATE,
    validate_complete_bundle,
)
from oaci.theory.c85v_stage_a_derivation import freeze_stage_a_derivations
from oaci.theory.c85v_stage_b_candidate_audit import freeze_stage_b_comparisons
from oaci.theory.c85v_statement_registry import PROTOCOL_COMMIT, PROTOCOL_SHA256, THEOREM_IDS


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c85vp_tables"
PROTOCOL = REPORTS / "C85V_INDEPENDENT_PROOF_REVIEW_PROTOCOL.json"
LOCK = REPORTS / "C85V_EXECUTION_LOCK.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_protocol_hash_chronology_and_candidate_blind_statement() -> None:
    assert _sha(PROTOCOL) == PROTOCOL_SHA256
    assert PROTOCOL.with_suffix(".sha256").read_text().split()[0] == PROTOCOL_SHA256
    assert _git("log", "-1", "--format=%H", "--", str(PROTOCOL.relative_to(ROOT))) == PROTOCOL_COMMIT
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["chronology"]["proof_candidate_text_opened_for_review_before_protocol"] is False
    assert protocol["chronology"]["formal_status_transitions_before_protocol"] == 0
    lock_commit = _git("log", "-1", "--format=%H", "--", str(LOCK.relative_to(ROOT)))
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, lock_commit],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def test_execution_lock_replays_every_bound_object() -> None:
    lock, lock_sha, repo_root = replay_execution_lock(LOCK)
    assert repo_root == ROOT
    assert lock_sha == LOCK.with_suffix(".sha256").read_text().split()[0]
    assert lock["status"] == "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
    assert lock["authorized"] is False
    assert lock["runtime_bound_object_count"] == len(lock["bound_repository_objects"])
    assert len(lock["bound_external_objects"]) == 13
    assert lock["review_process"]["candidate_text_withheld_until_stage_A_freeze"] is True
    assert lock["review_process"]["monte_carlo_rerun"] is False


def test_runtime_registry_and_readiness_tables_are_complete() -> None:
    lock = json.loads(LOCK.read_text())
    registry = _rows(TABLES / "runtime_bound_object_registry.csv")
    assert len(registry) == lock["runtime_bound_object_count"]
    assert {row["path"] for row in registry} == {
        row["path"] for row in lock["bound_repository_objects"]
    }
    required = {
        "theorem_statement_registry.csv",
        "proof_candidate_identity_registry.csv",
        "primary_literature_registry.csv",
        "review_stage_contract.csv",
        "verdict_contract.csv",
        "theorem_review_obligations.csv",
        "c85t_identity_replay.csv",
        "implementation_isolation_audit.csv",
        "exact_adversarial_fixture_registry.csv",
        "result_bundle_schema.csv",
        "shadow_validation.csv",
        "runtime_bound_object_registry.csv",
        "risk_register.csv",
        "failure_reason_ledger.csv",
    }
    assert {path.name for path in TABLES.glob("*.csv")} == required
    assert all(_rows(TABLES / name) for name in required)


def test_c85vp_has_no_authorization_result_or_status_transition() -> None:
    lock = json.loads(LOCK.read_text())
    assert not (REPORTS / "C85V_PI_AUTHORIZATION_RECORD.json").exists()
    assert not (REPORTS / "C85V_RESULT.json").exists()
    assert lock["readiness"]["registered_review_executions"] == 0
    assert lock["readiness"]["monte_carlo_reruns"] == 0
    assert lock["readiness"]["formal_status_transitions"] == 0
    assert lock["readiness"]["authorization_records"] == 0
    assert lock["frozen_c85t"]["formal_statuses"] == {
        theorem_id: "OPEN" for theorem_id in THEOREM_IDS
    }


def _prepare_shadow_bundle(tmp_path: Path, monkeypatch=None):
    output = tmp_path / "final"
    identity = {
        "execution_lock_sha256": "1" * 64,
        "execution_lock_commit": "2" * 40,
        "authorization_file_sha256": "3" * 64,
        "attempt_id": "shadow-attempt",
        "output_root": str(output.resolve()),
    }
    bundle = AtomicC85VResultBundle(
        output_root=output,
        attempt_id="shadow-attempt",
        identity=identity,
        review_mode="SHADOW_C85VP",
    )
    bundle.append_event("PREFLIGHT_STARTED")
    bundle.append_event("PREFLIGHT_COMPLETED")
    bundle.write_json("authorization_consumed.json", {"shadow": True})
    bundle.append_event("AUTHORIZATION_CONSUMED", "4" * 64)
    statements = shadow_statements()
    bundle.append_event("STAGE_A_STARTED")
    stage_a = bundle.staging_root / "stage_a"
    freeze_stage_a_derivations(
        statements=statements,
        obligations=shadow_obligations(),
        available_source_ids=frozenset({f"V{index:02d}" for index in range(1, 7)}),
        output_root=stage_a,
        review_mode="SHADOW_C85VP",
    )
    bundle.append_event(
        "STAGE_A_COMPLETED", _sha(stage_a / "C85V_STAGE_A_DERIVATION_MANIFEST.json")
    )
    candidates = tmp_path / "candidates"
    identities = write_shadow_candidates(candidates, statements)
    exact = tmp_path / "exact.json"
    write_shadow_exact_results(exact)
    bundle.append_event("STAGE_B_STARTED")
    stage_b = bundle.staging_root / "stage_b"
    freeze_stage_b_comparisons(
        stage_a_root=stage_a,
        candidate_bundle_root=candidates,
        exact_results_path=exact,
        output_root=stage_b,
        statements=statements,
        identities=identities,
        review_mode="SHADOW_C85VP",
    )
    bundle.append_event(
        "STAGE_B_COMPLETED", _sha(stage_b / "C85V_STAGE_B_COMPARISON_MANIFEST.json")
    )
    bundle.append_event("ADJUDICATION_STARTED")
    adjudication = bundle.staging_root / "adjudication"
    freeze_adjudication(
        stage_a_root=stage_a,
        stage_b_root=stage_b,
        output_root=adjudication,
        review_mode="SHADOW_C85VP",
        authorization_receipt=None,
    )
    bundle.append_event(
        "ADJUDICATION_COMPLETED", _sha(adjudication / "C85V_ADJUDICATION_MANIFEST.json")
    )
    (bundle.staging_root / "primary_literature_registry.csv").write_text(
        "source_id,verified\nSHADOW,1\n"
    )
    with (bundle.staging_root / "proof_candidate_retention_ledger.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("theorem_id", "external_path", "sha256", "retained", "overwritten"),
        )
        writer.writeheader()
        for theorem_id in THEOREM_IDS:
            writer.writerow({
                "theorem_id": theorem_id,
                "external_path": identities[theorem_id].relative_path,
                "sha256": identities[theorem_id].sha256,
                "retained": 1,
                "overwritten": 0,
            })
    statuses = {
        row["theorem_id"]: row["formal_status"]
        for row in _rows(adjudication / "formal_theorem_status_registry.csv")
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        **identity,
        "theorem_count": 7,
        "formal_theorem_statuses": statuses,
        "monte_carlo_reruns": 0,
        "real_data_access": 0,
        "active_acquisition": 0,
        "C85E_authorized": False,
        "manuscript_modified": False,
        "final_gate": SUCCESS_GATE,
    }
    if monkeypatch is not None:
        monkeypatch.setattr(os, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("shadow rename failure")))
    return bundle, result, output, identity


def test_shadow_atomic_result_publication(tmp_path: Path) -> None:
    bundle, result, output, identity = _prepare_shadow_bundle(tmp_path)
    bundle.prepare_and_publish(result)
    assert output.is_dir()
    assert not bundle.staging_root.exists()
    observed = validate_complete_bundle(output, expected_identity=identity)
    assert observed["review_mode"] == "SHADOW_C85VP"


def test_atomic_rename_failure_leaves_no_final_root(tmp_path: Path, monkeypatch) -> None:
    bundle, result, output, _ = _prepare_shadow_bundle(tmp_path, monkeypatch)
    with pytest.raises(OSError, match="shadow rename failure"):
        bundle.prepare_and_publish(result)
    assert not output.exists()
    assert bundle.staging_root.exists()


def test_c85vp_tests_are_in_cumulative_and_focused_regression_paths() -> None:
    from oaci.multidataset import c84r_regression_suite as suites

    expected = {
        "test_c85vp_stage_isolation.py",
        "test_c85vp_theorem_contracts.py",
        "test_c85vp_execution_lock.py",
    }
    for suite in ("c65", "c23"):
        assert expected <= {path.name for path in suites.suite_files(suite)}
    wrapper = (ROOT / "oaci/slurm_c85vp_regression.sh").read_text()
    assert all(name in wrapper for name in expected)
