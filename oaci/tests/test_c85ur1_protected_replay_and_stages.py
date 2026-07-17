from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from oaci.multidataset.c84s_common import C84SContractError, sha256_file
from oaci.theory.c85u_result_manifest_v2 import (
    MAX_U1_BYTES,
    publish_utility_field_v2,
    validate_u1_manifest_v2,
)
from oaci.theory.c85u_runtime_guard_v2 import (
    AppendOnlyLifecycleV2,
    create_stage_receipt_v2,
    validate_protected_replay_receipt_v2,
    validate_stage_receipt_v2,
)

from .c85ur1_test_support import (
    attach_shadow_protected_replay,
    make_shadow_context,
    make_shadow_registry,
)
from .c85urp_test_support import shadow_payload


def test_semantic_protected_replay_and_valid_hash_wrong_schema(tmp_path: Path) -> None:
    context = make_shadow_context(tmp_path)
    registry = make_shadow_registry(tmp_path)
    context = attach_shadow_protected_replay(context, registry)
    assert validate_protected_replay_receipt_v2(context, registry)["status"].startswith("PASS")

    value = json.loads(context.protected_replay_path.read_text(encoding="utf-8"))
    value["schema_version"] = "forged_schema_with_valid_file_hash"
    context.protected_replay_path.write_text(json.dumps(value), encoding="utf-8")
    forged = replace(context, protected_replay_sha256=sha256_file(context.protected_replay_path))
    with pytest.raises(C84SContractError, match="semantic linkage"):
        validate_protected_replay_receipt_v2(forged, registry)


def test_protected_replay_from_another_attempt_fails(tmp_path: Path) -> None:
    first = make_shadow_context(tmp_path / "first", attempt_id="first")
    registry = make_shadow_registry(tmp_path / "first")
    first = attach_shadow_protected_replay(first, registry)
    second = make_shadow_context(tmp_path / "second", attempt_id="second")
    copied = replace(
        second,
        protected_replay_path=first.protected_replay_path,
        protected_replay_sha256=first.protected_replay_sha256,
    )
    with pytest.raises(C84SContractError, match="semantic linkage"):
        validate_protected_replay_receipt_v2(copied, registry)


def test_duplicate_u1_and_u2_stage_receipts_fail(tmp_path: Path) -> None:
    context = make_shadow_context(tmp_path)
    registry = make_shadow_registry(tmp_path)
    context = attach_shadow_protected_replay(context, registry)
    lifecycle = AppendOnlyLifecycleV2(context.lifecycle_path)
    lifecycle.append("STAGE_U1_STARTED", context=context)
    _, u1_sha = create_stage_receipt_v2(
        context, "U1", prerequisite_sha256=context.protected_replay_sha256,
    )
    validate_stage_receipt_v2(
        context, "U1", prerequisite_sha256=context.protected_replay_sha256,
    )
    with pytest.raises(RuntimeError, match="already exists"):
        create_stage_receipt_v2(
            context, "U1", prerequisite_sha256=context.protected_replay_sha256,
        )
    lifecycle.append("STAGE_U1_COMPLETED", context=context, artifact_or_receipt_sha256=u1_sha)
    lifecycle.append("STAGE_U2_STARTED", context=context)
    _, u2_sha = create_stage_receipt_v2(context, "U2", prerequisite_sha256="2" * 64)
    assert len(u2_sha) == 64
    with pytest.raises(RuntimeError, match="already exists"):
        create_stage_receipt_v2(context, "U2", prerequisite_sha256="2" * 64)


def test_shadow_u1_v2_manifest_attempt_binding_and_size_envelope(tmp_path: Path) -> None:
    context = make_shadow_context(tmp_path)
    registry = make_shadow_registry(tmp_path)
    context = attach_shadow_protected_replay(context, registry)
    AppendOnlyLifecycleV2(context.lifecycle_path).append("STAGE_U1_STARTED", context=context)
    _, stage_sha = create_stage_receipt_v2(
        context, "U1", prerequisite_sha256=context.protected_replay_sha256,
    )
    final = context.output_root / "stage_u1_candidate_utility_v2"
    result = publish_utility_field_v2(
        payloads=[shadow_payload()], final_root=final,
        context=context, registry=registry,
        stage_receipt_sha256=stage_sha,
        allowed_access_counters={"evaluation_label_rows_read": 20},
        forbidden_access_counters={"selection_objects": 0},
        expected_contexts=1, expected_candidate_rows=81,
    )
    replay = validate_u1_manifest_v2(
        final, context=context, expected_handoff_sha256=result["handoff_sha256"],
        expected_contexts=1, expected_candidate_rows=81,
    )
    assert replay["actual_total_output_bytes"] <= MAX_U1_BYTES
    assert replay["contexts"] == 1 and replay["candidate_rows"] == 81


def test_registered_production_equivalent_arithmetic() -> None:
    contexts = 22 * 8 + 20 * 8 + 76 * 8
    finite_budget_contexts = 176 * 5 + 160 * 6 + 608 * 4
    assert contexts == 944
    assert contexts * 81 == 76_464
    assert 176 * 20 + 160 * 21 + 608 * 19 == 18_432
    assert finite_budget_contexts * 2_048 == 8_749_056
