"""Single-use authorization, capability, lifecycle, and proof-stage tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from oaci.theory.c85_decision_experiments import DecisionContractError
from oaci.theory.c85_lower_bound_contracts import TheoremStatus
from oaci.theory import c85t_execution_guard as guard
from oaci.theory import c85t_proofs as proofs
from oaci.theory import c85t_rng as rng


LOCK_SHA = "a" * 64
LOCK_COMMIT = "b" * 40
AUTH_ID = "01234567-89ab-4cde-8fab-0123456789ab"


def _record(tmp_path: Path) -> tuple[dict[str, object], str, Path]:
    output_parent = tmp_path / "outputs"
    consumption_root = tmp_path / "consumption"
    output = guard.expected_output_root(output_parent.resolve(), LOCK_SHA, AUTH_ID)
    record: dict[str, object] = {
        "schema_version": guard.AUTHORIZATION_SCHEMA,
        "direct_explicit_PI_authorization": True,
        "direct_statement_exact": guard.DIRECT_STATEMENT,
        "authorized_stage": "C85T",
        "authorization_id": AUTH_ID,
        "execution_lock_sha256": LOCK_SHA,
        "execution_lock_commit": LOCK_COMMIT,
        "output_root": str(output),
        "consumption_ledger_path": "NORMALIZED_BEFORE_DERIVATION",
        "C85E": False,
        "active_acquisition": False,
        "real_data": False,
        "manuscript": False,
    }
    authorization_sha = guard.authorization_binding_sha256(record)
    record["consumption_ledger_path"] = str(
        guard.expected_consumption_path(consumption_root.resolve(), authorization_sha)
    )
    guard.validate_authorization_record(
        record,
        authorization_sha256=authorization_sha,
        lock_sha256=LOCK_SHA,
        lock_commit=LOCK_COMMIT,
        output_parent=output_parent.resolve(),
        consumption_root=consumption_root.resolve(),
    )
    return record, authorization_sha, output


def test_authorization_consumption_is_globally_single_use(tmp_path: Path) -> None:
    record, authorization_sha, output = _record(tmp_path)
    receipt, capability = guard.consume_authorization_once(
        record=record,
        authorization_sha256=authorization_sha,
        lock_sha256=LOCK_SHA,
        lock_commit=LOCK_COMMIT,
        output_root=output,
        attempt_id="attempt-a",
        head="c" * 40,
    )
    assert receipt["output_root"] == str(output)
    guard.require_registered_capability(
        capability,
        authorization_sha256=authorization_sha,
        execution_lock_sha256=LOCK_SHA,
        attempt_id="attempt-a",
        output_root=output,
    )
    with pytest.raises(DecisionContractError, match="already consumed"):
        guard.consume_authorization_once(
            record=record,
            authorization_sha256=authorization_sha,
            lock_sha256=LOCK_SHA,
            lock_commit=LOCK_COMMIT,
            output_root=output,
            attempt_id="attempt-b",
            head="c" * 40,
        )


def test_authorization_cannot_move_to_a_different_output_root(tmp_path: Path) -> None:
    record, authorization_sha, output = _record(tmp_path)
    with pytest.raises(DecisionContractError, match="different output root"):
        guard.consume_authorization_once(
            record=record,
            authorization_sha256=authorization_sha,
            lock_sha256=LOCK_SHA,
            lock_commit=LOCK_COMMIT,
            output_root=output.with_name("different"),
            attempt_id="attempt-a",
            head="c" * 40,
        )
    assert not Path(record["consumption_ledger_path"]).exists()


def test_static_string_and_unissued_object_cannot_unlock_registered_rng() -> None:
    assert not hasattr(rng, "REGISTERED_EXECUTION_TOKEN")
    for fake in (
        "C85T_LOCKED_EXECUTION_AUTHORIZATION_REPLAYED",
        object(),
        None,
    ):
        with pytest.raises(DecisionContractError, match="consumed C85T authorization"):
            rng.deterministic_seed("S6", 0, capability=fake)


def test_capability_is_bound_to_one_attempt(tmp_path: Path) -> None:
    record, authorization_sha, output = _record(tmp_path)
    _, capability = guard.consume_authorization_once(
        record=record,
        authorization_sha256=authorization_sha,
        lock_sha256=LOCK_SHA,
        lock_commit=LOCK_COMMIT,
        output_root=output,
        attempt_id="attempt-a",
        head="c" * 40,
    )
    with pytest.raises(DecisionContractError, match="binding drifted"):
        guard.require_registered_capability(capability, attempt_id="attempt-b")
    with pytest.raises(DecisionContractError, match="binding drifted"):
        guard.require_registered_capability(
            capability, output_root=output.with_name("different")
        )


def test_lifecycle_is_append_only_and_replays_failure_location(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.jsonl"
    ledger = guard.AppendOnlyLifecycleLedger(
        path,
        authorization_sha256="d" * 64,
        lock_sha256=LOCK_SHA,
        attempt_id="attempt-a",
    )
    ledger.append("PREFLIGHT_STARTED")
    ledger.append("PREFLIGHT_COMPLETED", artifact_or_receipt_sha256="e" * 64)
    ledger.append(
        "FAILED",
        failure={
            "primary_exception_type": "RuntimeError",
            "primary_exception_message": "shadow failure",
        },
    )
    events = guard.replay_lifecycle(path)
    assert [row["sequence_number"] for row in events] == [0, 1, 2]
    assert events[-1]["last_completed_stage"] == "PREFLIGHT_COMPLETED"
    assert events[-1]["primary_exception_message"] == "shadow failure"
    with pytest.raises(DecisionContractError, match="already exists"):
        guard.AppendOnlyLifecycleLedger(
            path,
            authorization_sha256="d" * 64,
            lock_sha256=LOCK_SHA,
            attempt_id="attempt-b",
        )


def test_lifecycle_rejects_skipped_stage(tmp_path: Path) -> None:
    ledger = guard.AppendOnlyLifecycleLedger(
        tmp_path / "out-of-order.jsonl",
        authorization_sha256="d" * 64,
        lock_sha256=LOCK_SHA,
        attempt_id="attempt-a",
    )
    with pytest.raises(DecisionContractError, match="stage order"):
        ledger.append("MONTE_CARLO_STARTED")


def test_c85t_cannot_transition_a_theorem_even_with_fabricated_pass() -> None:
    candidate = proofs.ProofCandidate(
        theorem_id="T1",
        exact_statement="shadow statement",
        assumptions=("shadow assumption",),
        proof_or_counterexample="shadow argument",
        boundary_cases=("shadow boundary",),
        proposed_status=TheoremStatus.PROVED,
    )
    audit = proofs.ProofAudit("T1", "PASS", ("fabricated same-process check",))
    with pytest.raises(DecisionContractError, match="reserved for future C85V"):
        proofs.apply_status_transition(candidate, audit)


def test_normalized_authorization_hash_avoids_path_self_reference(tmp_path: Path) -> None:
    record, authorization_sha, _ = _record(tmp_path)
    assert guard.authorization_binding_sha256(record) == authorization_sha
    raw_file_sha = __import__("hashlib").sha256(
        guard.canonical_json_bytes(record)
    ).hexdigest()
    assert raw_file_sha != authorization_sha
    assert Path(record["consumption_ledger_path"]).name == f"{authorization_sha}.json"
