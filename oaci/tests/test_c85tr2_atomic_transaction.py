from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from oaci.theory.c85_decision_experiments import DecisionContractError
from oaci.theory.c85t_execution_context_v3 import (
    AppendOnlyLifecycleLedgerV3,
    SUCCESS_EVENTS_V3,
    canonical_json_bytes,
    create_validated_c85t_execution_context,
    replay_lifecycle_v3,
)
from oaci.theory.c85t_transaction_v3 import (
    AtomicExecutionBundleV3,
    preserve_primary_exception_v3,
    recover_post_rename_bundle_v3,
    replay_artifact_manifest_v3,
)
from oaci.tests.c85tr2_test_support import (
    create_shadow_authorized_repository,
    populate_shadow_bundle,
)


def _prepared(tmp_path: Path):
    fixture = create_shadow_authorized_repository(tmp_path)
    context = create_validated_c85t_execution_context(
        fixture["lock_path"], fixture["authorization_path"], fixture["output_root"]
    )
    bundle = AtomicExecutionBundleV3(context)
    shadow = populate_shadow_bundle(bundle, context, fixture["statements"])
    completion = bundle._prepare_commit(
        shadow["result"],
        contract=shadow["contract"],
        statements=fixture["statements"],
        shadow_expected_exact=shadow["exact"],
        shadow_expected_s9_arrays=shadow["s9_arrays"],
        shadow_expected_s9_digest_rows=shadow["digest_rows"],
    )
    return fixture, context, bundle, completion


def test_complete_bundle_is_terminal_before_single_rename(tmp_path: Path) -> None:
    fixture, context, bundle, expected = _prepared(tmp_path)
    staging_events = replay_lifecycle_v3(context.lifecycle_path)
    assert [row["stage"] for row in staging_events] == list(SUCCESS_EVENTS_V3)
    observed = bundle._commit_prepared()
    assert observed == expected
    assert fixture["output_root"].is_dir()
    assert not context.staging_bundle_root.exists()
    manifest = replay_artifact_manifest_v3(fixture["output_root"])
    assert manifest["derived_counts"]["scenario_results"] == 11
    assert manifest["derived_counts"]["S6_S7_logical_replicate_rows"] == 8192
    final_events = replay_lifecycle_v3(
        fixture["output_root"] / "C85T_V3_LIFECYCLE.jsonl"
    )
    assert final_events[-1]["stage"] == "ATOMIC_PUBLISH_COMMIT_READY"


def test_crash_after_rename_recovers_valid_bundle_as_success(tmp_path: Path) -> None:
    fixture, context, bundle, _ = _prepared(tmp_path)
    os.replace(context.staging_bundle_root, context.output_root)
    recovered = recover_post_rename_bundle_v3(
        context.output_root,
        external_consumption_receipt_path=fixture["external_receipt_path"],
    )
    assert recovered["classification"] == "RECOVERED_SUCCESS_AFTER_FINAL_RENAME"
    assert recovered["output_root"] == str(context.output_root)


def test_terminal_staging_failure_preserves_primary_and_does_not_append_failed(
    tmp_path: Path,
) -> None:
    _, context, _, _ = _prepared(tmp_path)
    primary = RuntimeError("PRIMARY_COMMIT_EXCEPTION")
    assert preserve_primary_exception_v3(context, primary) is None
    events = replay_lifecycle_v3(context.lifecycle_path)
    assert events[-1]["stage"] == "ATOMIC_PUBLISH_COMMIT_READY"
    assert all(row["stage"] != "FAILED" for row in events)
    blocker = json.loads(
        (context.staging_bundle_root / "C85T_V3_RECONCILIATION_BLOCKER.json").read_text()
    )
    assert blocker["primary_exception_type"] == "RuntimeError"
    assert blocker["primary_exception_message"] == "PRIMARY_COMMIT_EXCEPTION"


def test_nonterminal_failure_appends_failed_once_without_final_root(tmp_path: Path) -> None:
    fixture = create_shadow_authorized_repository(tmp_path)
    context = create_validated_c85t_execution_context(
        fixture["lock_path"], fixture["authorization_path"], fixture["output_root"]
    )
    primary = ValueError("PRIMARY_BEFORE_EXECUTION")
    assert preserve_primary_exception_v3(context, primary) is None
    events = replay_lifecycle_v3(context.lifecycle_path)
    assert [row["stage"] for row in events][-1] == "FAILED"
    assert events[-1]["primary_exception_type"] == "ValueError"
    assert events[-1]["primary_exception_message"] == "PRIMARY_BEFORE_EXECUTION"
    assert not fixture["output_root"].exists()
    context.close()


def test_lifecycle_append_error_does_not_replace_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = create_shadow_authorized_repository(tmp_path)
    context = create_validated_c85t_execution_context(
        fixture["lock_path"], fixture["authorization_path"], fixture["output_root"]
    )

    def fail_append(*args, **kwargs):
        raise OSError("SECONDARY_LEDGER_FAILURE")

    monkeypatch.setattr(AppendOnlyLifecycleLedgerV3, "append", fail_append)
    primary = RuntimeError("PRIMARY_EXCEPTION_MUST_SURVIVE")
    assert preserve_primary_exception_v3(context, primary) is None
    failure = json.loads(
        (context.staging_bundle_root / "C85T_V3_FAILURE_RECEIPT.json").read_text()
    )
    assert failure["primary_exception_type"] == "RuntimeError"
    assert failure["primary_exception_message"] == "PRIMARY_EXCEPTION_MUST_SURVIVE"
    assert any("SECONDARY_LEDGER_FAILURE" in row for row in failure["secondary_errors"])
    context.close()


def test_commit_has_no_required_operation_or_callback_after_rename() -> None:
    source_path = (
        Path(__file__).resolve().parents[1] / "theory" / "c85t_transaction_v3.py"
    )
    source = source_path.read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_commit_prepared"
    )
    replace_index = next(
        index
        for index, node in enumerate(method.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "replace"
    )
    assert len(method.body) == replace_index + 2
    assert isinstance(method.body[-1], ast.Return)
    assert "manifest_completed_callback" not in source
    assert "atomic_publish_callback" not in source


def test_failure_before_manifest_cannot_leave_final_bundle(tmp_path: Path) -> None:
    fixture = create_shadow_authorized_repository(tmp_path)
    context = create_validated_c85t_execution_context(
        fixture["lock_path"], fixture["authorization_path"], fixture["output_root"]
    )
    bundle = AtomicExecutionBundleV3(context)
    shadow = populate_shadow_bundle(bundle, context, fixture["statements"])
    exact_path = context.staging_bundle_root / "exact_scenario_results.json"
    exact = json.loads(exact_path.read_text())
    exact.pop("S10")
    exact_path.write_bytes(canonical_json_bytes(exact))
    with pytest.raises(DecisionContractError, match="exact scenario keys") as caught:
        bundle._prepare_commit(
            shadow["result"],
            contract=shadow["contract"],
            statements=fixture["statements"],
            shadow_expected_exact=shadow["exact"],
            shadow_expected_s9_arrays=shadow["s9_arrays"],
            shadow_expected_s9_digest_rows=shadow["digest_rows"],
        )
    preserve_primary_exception_v3(context, caught.value)
    assert not fixture["output_root"].exists()
    assert replay_lifecycle_v3(context.lifecycle_path)[-1]["stage"] == "FAILED"
    context.close()
