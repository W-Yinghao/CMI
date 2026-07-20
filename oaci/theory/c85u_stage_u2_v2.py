"""Authorized C85U V2 Stage U2 with pre-access attempt guards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from oaci.multidataset.c84s_common import require, sha256_file

from .c85u_historical_decision_replay_v2 import run_historical_decision_replay_v2
from .c85u_result_manifest_v2 import U1_HANDOFF_NAME, validate_u1_manifest_v2
from .c85u_runtime_guard_v2 import (
    AppendOnlyLifecycleV2,
    create_stage_receipt_v2,
    load_execution_context_record_v2,
    validate_context_lock_v2,
    validate_stage_receipt_v2,
)
from .c85u_u2_registry_v2 import (
    replay_u2_runtime_registry,
    resolve_u2_runtime_registry,
)


def _validate_u1_handoff_before_u2(
    *, context: Any, handoff_path: Path, utility_root: Path,
) -> tuple[dict[str, Any], str]:
    require(handoff_path.resolve() == utility_root / U1_HANDOFF_NAME and
            handoff_path.is_file(), "C85U V2 U2 U1-handoff path drift")
    sidecar = handoff_path.with_suffix(".sha256")
    digest = sha256_file(handoff_path)
    require(sidecar.is_file() and sidecar.read_text(encoding="ascii").split()
            == [digest, handoff_path.name], "C85U V2 U2 U1-handoff sidecar drift")
    value = json.loads(handoff_path.read_text(encoding="utf-8"))
    identity = {
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_id": context.authorization_id,
        "attempt_id": context.attempt_id,
        "parent_output_root": str(context.output_root),
        "U1_output_root": str(utility_root),
        "protected_replay_sha256": context.protected_replay_sha256,
    }
    require(all(value.get(key) == expected for key, expected in identity.items()),
            "C85U V2 U2 U1-handoff attempt linkage drift")
    replay = validate_u1_manifest_v2(
        utility_root, context=context, expected_handoff_sha256=digest,
    )
    require(replay["manifest_sha256"] == value["U1_manifest_sha256"],
            "C85U V2 U2 U1 manifest/handoff linkage drift")
    return value, digest


def run_stage_u2_v2(
    *, execution_context_path: str | Path, u1_handoff: str | Path,
    utility_root: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    context = load_execution_context_record_v2(execution_context_path)
    lock = validate_context_lock_v2(context)
    utility = Path(utility_root).resolve()
    output = Path(output_root).resolve()
    require(utility.parent == context.output_root and
            utility.name == "stage_u1_candidate_utility_v2",
            "C85U V2 U2 utility-root binding drift")
    require(output.parent == context.output_root and
            output.name == "stage_u2_historical_replay_v2",
            "C85U V2 U2 output-root binding drift")
    _, u1_handoff_sha = _validate_u1_handoff_before_u2(
        context=context, handoff_path=Path(u1_handoff), utility_root=utility,
    )
    events = AppendOnlyLifecycleV2(context.lifecycle_path).replay()
    require("STAGE_U1_COMPLETED" in [row["stage"] for row in events] and
            events[-1]["stage"] == "STAGE_U2_STARTED",
            "C85U V2 U2 lifecycle prerequisite drift")
    stage_path, stage_sha = create_stage_receipt_v2(
        context, "U2", prerequisite_sha256=u1_handoff_sha,
    )
    validate_stage_receipt_v2(
        context, "U2", prerequisite_sha256=u1_handoff_sha,
    )
    registry = resolve_u2_runtime_registry(lock)
    replay_u2_runtime_registry(registry)
    result = run_historical_decision_replay_v2(
        utility_root=utility,
        registry=registry,
        final_root=output,
        context=context,
        u1_handoff_sha256=u1_handoff_sha,
        stage_receipt_sha256=stage_sha,
    )
    AppendOnlyLifecycleV2(context.lifecycle_path).append(
        "STAGE_U2_COMPLETED", context=context,
        artifact_or_receipt_sha256=result["handoff_sha256"],
        details={
            "result_sha256": result["result_sha256"],
            "stage_receipt_sha256": stage_sha,
            "stage_receipt_path": str(stage_path),
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--execution-context", required=True)
    parser.add_argument("--u1-handoff", required=True)
    parser.add_argument("--utility-root", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args(argv)
    result = run_stage_u2_v2(
        execution_context_path=arguments.execution_context,
        u1_handoff=arguments.u1_handoff,
        utility_root=arguments.utility_root,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_stage_u2_v2"]
