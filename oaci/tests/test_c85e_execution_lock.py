"""C85E V1 execution-lock, runtime guard, and atomic publication tests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from oaci.theory.c85e_result_manifest import (
    REGISTERED_TABLES, SUCCESS_GATE, publish_result_bundle, replay_result_bundle,
)
from oaci.theory.c85e_runtime_guard import (
    ValidatedC85EExecutionContext, expected_output_root, require_registered_path,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
LOCK = REPORTS / "C85E_EXECUTION_LOCK.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _context(tmp_path: Path) -> ValidatedC85EExecutionContext:
    output = tmp_path / "final"
    receipt_path = tmp_path / "authorization_consumed.json"
    receipt = {
        "schema_version": "c85e_authorization_consumption_receipt_v1",
        "authorization_file_sha256": "1" * 64,
        "authorization_binding_sha256": "2" * 64,
        "authorization_id": "00000000-0000-4000-8000-000000000001",
        "execution_lock_sha256": "3" * 64,
        "execution_lock_commit": "4" * 40,
        "attempt_id": "shadow-attempt",
        "output_root": str(output.resolve()),
        "HEAD": "5" * 40,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    return ValidatedC85EExecutionContext(
        authorization_file_sha256="1" * 64,
        authorization_binding_sha256="2" * 64,
        authorization_id=receipt["authorization_id"],
        execution_lock_sha256="3" * 64,
        execution_lock_commit="4" * 40,
        attempt_id="shadow-attempt", output_root=output.resolve(),
        consumption_receipt_path=receipt_path.resolve(),
        consumption_receipt_sha256=sha256_file(receipt_path),
        head="5" * 40, inputs=(),
    )


def _tables() -> dict[str, list[dict[str, object]]]:
    return {
        name: [{"shadow_row": 1, "result_tag": "POST_C84S_EXPLORATORY"}]
        for name in REGISTERED_TABLES
    }


def test_execution_lock_self_hash_status_scope_and_resources() -> None:
    sidecar = LOCK.with_suffix(".sha256")
    assert LOCK.is_file() and sidecar.is_file()
    digest = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    assert sidecar.read_text().split() == [digest, LOCK.name]
    lock = json.loads(LOCK.read_text())
    assert lock["schema_version"] == "c85e_execution_lock_v1"
    assert lock["status"] == "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
    assert lock["authorized"] is False
    assert lock["exact_scope"] == {
        "utility_contexts": 944, "candidates_per_context": 81,
        "candidate_utility_rows": 76_464, "method_context_rows": 18_432,
        "finite_Q0_action_records": 8_749_056, "Q0_shards": 944,
    }
    assert lock["resources"] == {
        "partition": "cpu-high", "CPU": 32, "RAM_GiB": 128,
        "GPU": 0, "wall_hours": 2, "result_output_bytes_max": 2_147_483_648,
    }
    assert lock["future_direct_statement_exact"] == "授权 C85E"


def test_lock_replays_bound_repository_objects_and_git_blobs() -> None:
    lock = json.loads(LOCK.read_text())
    rows = lock["bound_repository_objects"]
    assert len(rows) == lock["runtime_bound_object_count"]
    assert len({row["path"] for row in rows}) == len(rows)
    for row in rows:
        path = ROOT / row["path"]
        assert path.is_file() and path.stat().st_size == row["size_bytes"]
        assert sha256_file(path) == row["sha256"]
        assert _git("hash-object", "--", row["path"]) == row["git_blob"]


def test_lock_chronology_and_no_authorization_or_real_execution() -> None:
    lock = json.loads(LOCK.read_text())
    lock_commit = _git("log", "-1", "--format=%H", "--", str(LOCK.relative_to(ROOT)))
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock["implementation_commit"], lock_commit],
        cwd=ROOT, check=False,
    ).returncode == 0
    assert not (REPORTS / "C85E_PI_AUTHORIZATION_RECORD.json").exists()
    assert lock["readiness"]["real_C85E_executions"] == 0
    assert lock["readiness"]["authorization_records"] == 0
    assert lock["readiness"]["theorem_status_transitions"] == 0


def test_content_addressed_output_and_unregistered_open_fail_closed(tmp_path: Path) -> None:
    value = expected_output_root(
        "a" * 64, "00000000-0000-4000-8000-000000000001",
    )
    assert value.name == "c85e-v1-aaaaaaaaaaaaaaaa-0000000000004000"
    context = _context(tmp_path)
    unregistered = tmp_path / "direct-label-like.csv"
    unregistered.write_text("forbidden\n")
    with pytest.raises(Exception, match="unregistered path"):
        require_registered_path(context, unregistered)


def test_shadow_result_publication_is_atomic_and_semantically_replayed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    result = publish_result_bundle(
        context=context, tables=_tables(), synthesis_markdown="# Shadow\n",
        input_replay_sha256="6" * 64,
    )
    assert result["gate"] == SUCCESS_GATE
    assert context.output_root.is_dir()
    replay = replay_result_bundle(context.output_root)
    assert replay == {
        "status": "PASS", "artifacts": len(REGISTERED_TABLES) + 3,
        "registered_tables": len(REGISTERED_TABLES), "gate": SUCCESS_GATE,
    }


def test_atomic_rename_failure_leaves_no_success_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(
        os, "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("shadow rename failure")),
    )
    with pytest.raises(OSError, match="shadow rename failure"):
        publish_result_bundle(
            context=context, tables=_tables(), synthesis_markdown="# Shadow\n",
            input_replay_sha256="6" * 64,
        )
    assert not context.output_root.exists()


def test_result_writer_rejects_untagged_rows(tmp_path: Path) -> None:
    context = _context(tmp_path)
    tables = _tables()
    tables[REGISTERED_TABLES[0]][0]["result_tag"] = "CONFIRMATORY"
    with pytest.raises(Exception, match="exploratory result tag drift"):
        publish_result_bundle(
            context=context, tables=tables, synthesis_markdown="# Shadow\n",
            input_replay_sha256="6" * 64,
        )
    assert not context.output_root.exists()
