from __future__ import annotations

import json
from pathlib import Path
import subprocess

from oaci.multidataset.c84s_common import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
LOCK_PATH = REPORT_DIR / "C85U_EXECUTION_LOCK_V2.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def test_v2_lock_self_hash_schema_and_status() -> None:
    sidecar = LOCK_PATH.with_suffix(".sha256")
    assert LOCK_PATH.is_file() and sidecar.is_file()
    digest = sha256_file(LOCK_PATH)
    assert sidecar.read_text(encoding="ascii").split() == [digest, LOCK_PATH.name]
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["schema_version"] == "c85u_execution_lock_v2"
    assert lock["status"] == "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
    assert lock["authorized"] is False
    assert lock["historical_V1_lock"]["status"] == (
        "SUPERSEDED_BEFORE_AUTHORIZATION_OR_REAL_PROTECTED_ACCESS"
    )


def test_v2_lock_replays_all_bound_repository_bytes_and_git_blobs() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    rows = lock["bound_repository_objects"]
    assert len(rows) == lock["runtime_bound_object_count"]
    assert len({row["path"] for row in rows}) == len(rows)
    for row in rows:
        path = REPO_ROOT / row["path"]
        assert path.is_file() and path.stat().st_size == row["size_bytes"]
        assert sha256_file(path) == row["sha256"]
        assert _git("hash-object", "--", row["path"]) == row["git_blob"]


def test_v2_lock_exact_stage_scope_and_atomic_acceptance() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["exact_scope"] == {
        "contexts": 944,
        "candidates_per_context": 81,
        "candidate_utility_rows": 76_464,
        "method_context_rows": 18_432,
        "finite_Q0_action_records": 8_749_056,
    }
    assert lock["U1_runtime_input_registry"]["forbidden_U2_or_scientific_paths"] == 0
    assert lock["protected_replay"]["target_artifact_bytes"] == 48_018_748_054
    assert lock["atomic_acceptance"]["post_rename_required_operations"] == []
    assert "c85u_execute_v2" in lock["entrypoint"]
    assert lock["future_completion_gate"] == (
        "C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED"
    )


def test_protocol_precedes_implementation_and_lock() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    protocol_commit = _git(
        "log", "-1", "--format=%H", "--",
        "oaci/reports/C85UR1_U1_U2_ISOLATION_PROTECTED_REPLAY_AND_ACCEPTANCE_PROTOCOL.json",
    )
    implementation = lock["implementation_commit"]
    lock_commit = _git("log", "-1", "--format=%H", "--", str(LOCK_PATH.relative_to(REPO_ROOT)))
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, implementation],
        cwd=REPO_ROOT, check=False,
    ).returncode == 0
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation, lock_commit],
        cwd=REPO_ROOT, check=False,
    ).returncode == 0


def test_no_authorization_real_execution_or_downstream_lock() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["readiness"] == {
        "real_evaluation_label_rows_opened": 0,
        "real_target_payloads_opened": 0,
        "real_Q0_or_direct_result_objects_opened": 0,
        "real_candidate_utilities_computed": 0,
        "authorization_records": 0,
        "success_gate": "C85U_PROCESS_ISOLATION_PROTECTED_REPLAY_AND_ACCEPTANCE_TRANSACTION_REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION",
        "failure_gate": "C85U_STAGE_ISOLATION_ATTEMPT_BINDING_ATOMIC_ACCEPTANCE_OR_PROVENANCE_RECONCILIATION_REQUIRED",
    }
    assert not (REPORT_DIR / "C85U_V2_PI_AUTHORIZATION_RECORD.json").exists()
    assert not (REPORT_DIR / "C85E_EXECUTION_LOCK.json").exists()
