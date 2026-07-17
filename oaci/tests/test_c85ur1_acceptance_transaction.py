from __future__ import annotations

import ast
from pathlib import Path

import pytest

from oaci.theory import c85u_acceptance_transaction_v2 as transaction
from oaci.theory.c85u_runtime_guard_v2 import AppendOnlyLifecycleV2

from .c85ur1_test_support import (
    attach_shadow_protected_replay,
    make_shadow_context,
    make_shadow_registry,
)


def _context_through_u2(tmp_path: Path):
    context = make_shadow_context(tmp_path)
    context = attach_shadow_protected_replay(context, make_shadow_registry(tmp_path))
    lifecycle = AppendOnlyLifecycleV2(context.lifecycle_path)
    lifecycle.append("STAGE_U1_STARTED", context=context)
    lifecycle.append("STAGE_U1_COMPLETED", context=context, artifact_or_receipt_sha256="1" * 64)
    lifecycle.append("STAGE_U2_STARTED", context=context)
    lifecycle.append("STAGE_U2_COMPLETED", context=context, artifact_or_receipt_sha256="2" * 64)
    return context


def _patch_stage_replays(monkeypatch) -> None:
    monkeypatch.setattr(transaction, "validate_u1_manifest_v2", lambda *args, **kwargs: {
        "manifest_sha256": "3" * 64, "handoff_sha256": "1" * 64,
        "contexts": 944, "candidate_rows": 76_464,
    })
    monkeypatch.setattr(transaction, "validate_u2_result_v2", lambda *args, **kwargs: {
        "result_sha256": "4" * 64, "handoff_sha256": "2" * 64,
        "method_context_rows": 18_432,
        "finite_Q0_action_records_replayed": 8_749_056,
    })


def test_atomic_acceptance_and_post_rename_recovery(monkeypatch, tmp_path: Path) -> None:
    context = _context_through_u2(tmp_path)
    _patch_stage_replays(monkeypatch)
    completion = transaction.AtomicC85UAcceptanceTransactionV2(context).publish(
        u1_root=context.output_root / "u1",
        u2_root=context.output_root / "u2",
        u1_handoff_sha256="1" * 64,
        u2_handoff_sha256="2" * 64,
    )
    final = context.output_root / transaction.FINAL_ACCEPTANCE_NAME
    assert final.is_dir() and not context.acceptance_staging_root.exists()
    assert completion["final_gate"] == transaction.SUCCESS_GATE
    recovered = transaction.recover_post_rename_acceptance_v2(
        final, external_receipt_path=context.receipt_path,
    )
    assert recovered["classification"] == "RECOVERED_SUCCESS_AFTER_FINAL_ACCEPTANCE_RENAME"


def test_u1_success_u2_failure_has_no_acceptance_bundle(tmp_path: Path) -> None:
    context = make_shadow_context(tmp_path)
    context = attach_shadow_protected_replay(context, make_shadow_registry(tmp_path))
    lifecycle = AppendOnlyLifecycleV2(context.lifecycle_path)
    lifecycle.append("STAGE_U1_STARTED", context=context)
    lifecycle.append("STAGE_U1_COMPLETED", context=context, artifact_or_receipt_sha256="1" * 64)
    (context.output_root / "stage_u1_candidate_utility_v2").mkdir()
    primary = RuntimeError("shadow U2 failure")
    assert transaction.preserve_primary_exception_v2(context, primary) is None
    assert not (context.output_root / transaction.FINAL_ACCEPTANCE_NAME).exists()
    assert (context.output_root / "C85U_FAILURE_RECEIPT_V2.json").is_file()


def test_terminal_staging_rename_failure_cannot_publish_success(monkeypatch, tmp_path: Path) -> None:
    context = _context_through_u2(tmp_path)
    _patch_stage_replays(monkeypatch)
    bundle = transaction.AtomicC85UAcceptanceTransactionV2(context)
    bundle.prepare(
        u1_root=context.output_root / "u1",
        u2_root=context.output_root / "u2",
        u1_handoff_sha256="1" * 64,
        u2_handoff_sha256="2" * 64,
    )
    primary = OSError("injected rename failure")
    monkeypatch.setattr(transaction.os, "replace", lambda *args: (_ for _ in ()).throw(primary))
    with pytest.raises(OSError, match="injected rename"):
        bundle.commit()
    assert not (context.output_root / transaction.FINAL_ACCEPTANCE_NAME).exists()
    assert transaction.preserve_primary_exception_v2(context, primary) is None
    assert (context.acceptance_staging_root / "C85U_RECONCILIATION_BLOCKER.json").is_file()


def test_primary_exception_survives_secondary_lifecycle_failure(monkeypatch, tmp_path: Path) -> None:
    context = make_shadow_context(tmp_path)
    primary = ValueError("primary failure")
    monkeypatch.setattr(
        AppendOnlyLifecycleV2, "append_failed",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("secondary ledger failure")),
    )
    assert transaction.preserve_primary_exception_v2(context, primary) is None
    failure = (context.output_root / "C85U_FAILURE_RECEIPT_V2.json").read_text()
    assert "ValueError" in failure and "primary failure" in failure
    assert "secondary ledger failure" in failure


def test_commit_has_no_required_operation_after_single_replace() -> None:
    source = Path(transaction.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "commit"
    )
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    replaces = [
        node for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
    ]
    assert len(replaces) == 1
    replace_line = replaces[0].lineno
    later_calls = [node for node in calls if node.lineno > replace_line]
    assert later_calls == []
