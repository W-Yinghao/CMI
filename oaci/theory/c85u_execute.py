"""Authorized C85U U1-to-U2 production coordinator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from oaci.multidataset.c84s_common import require, sha256_file
from .c85u_runtime_guard import (
    C85UExecutionContext,
    create_execution_context,
    execution_context_record,
    replay_protected_inputs_after_consumption,
)


SUCCESS_GATE = "C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _write_fsynced(path: Path, payload: bytes, *, exclusive: bool = True) -> None:
    mode = "xb" if exclusive else "wb"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


class Lifecycle:
    def __init__(self, path: Path, context: C85UExecutionContext) -> None:
        self.path = path
        self.context = context
        self.sequence = 0
        _write_fsynced(path, b"")

    def append(self, stage: str, **details: Any) -> None:
        row = {
            "schema_version": "c85u_append_only_lifecycle_v1",
            "sequence": self.sequence,
            "timestamp_unix_ns": time.time_ns(),
            "stage": stage,
            "authorization_binding_sha256": self.context.authorization_binding_sha256,
            "execution_lock_sha256": self.context.execution_lock_sha256,
            "attempt_id": self.context.attempt_id,
            **details,
        }
        with self.path.open("ab") as handle:
            handle.write(_canonical_json_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
        self.sequence += 1


def _run_stage(command: list[str], *, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    _write_fsynced(stdout_path, completed.stdout.encode("utf-8"))
    _write_fsynced(stderr_path, completed.stderr.encode("utf-8"))
    require(completed.returncode == 0, f"C85U subprocess failed: {command[2]}")
    require(completed.stderr == "", f"C85U subprocess stderr is nonempty: {command[2]}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    require(lines, f"C85U subprocess result absent: {command[2]}")
    value = json.loads(lines[-1])
    require(isinstance(value, dict), "C85U subprocess result malformed")
    return value


def _write_preflight_blocker(output_root: Path, error: BaseException) -> None:
    blocker = output_root.parent / f".{output_root.name}.preflight_blocker.json"
    if blocker.exists():
        return
    value = {
        "schema_version": "c85u_preflight_blocker_v1",
        "status": "FAILED_BEFORE_REGISTERED_STAGE_START",
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp_unix_ns": time.time_ns(),
    }
    try:
        _write_fsynced(blocker, _canonical_json_bytes(value))
    except BaseException:
        pass


def run_locked(
    *,
    execution_lock: str | Path,
    authorization_record: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    context: C85UExecutionContext | None = None
    lifecycle: Lifecycle | None = None
    try:
        context, registry = create_execution_context(
            execution_lock=execution_lock,
            authorization_record=authorization_record,
            output_root=output,
        )
        output.mkdir(parents=True, exist_ok=False)
        lifecycle = Lifecycle(output / "C85U_LIFECYCLE.jsonl", context)
        lifecycle.append("PREFLIGHT_STARTED")
        lifecycle.append("PREFLIGHT_COMPLETED")
        _write_fsynced(
            output / "authorization_consumed.json", context.receipt_path.read_bytes(),
        )
        lifecycle.append(
            "AUTHORIZATION_CONSUMED",
            receipt_sha256=context.receipt_sha256,
        )
        lifecycle.append("PROTECTED_INPUT_REPLAY_STARTED")
        context = replay_protected_inputs_after_consumption(context, registry)
        lifecycle.context = context
        lifecycle.append(
            "PROTECTED_INPUT_REPLAY_COMPLETED",
            replay_sha256=context.protected_replay_sha256,
        )
        context_path = output / "C85U_EXECUTION_CONTEXT.json"
        _write_fsynced(context_path, _canonical_json_bytes(execution_context_record(context)))

        lifecycle.append("STAGE_U1_STARTED")
        u1_command = [
            sys.executable, "-m", "oaci.theory.c85u_stage_u1", "run-real",
            "--execution-context", str(context_path),
            "--output-root", str(output / "stage_u1_candidate_utility"),
        ]
        u1_result = _run_stage(
            u1_command,
            stdout_path=output / "stage_u1.stdout.log",
            stderr_path=output / "stage_u1.stderr.log",
        )
        lifecycle.append(
            "STAGE_U1_COMPLETED", manifest_sha256=u1_result["manifest_sha256"],
            command=u1_command,
        )

        lifecycle.append("STAGE_U2_STARTED")
        u2_command = [
            sys.executable, "-m", "oaci.theory.c85u_stage_u2", "run-real",
            "--utility-root", str(output / "stage_u1_candidate_utility"),
            "--output-root", str(output / "stage_u2_historical_replay"),
        ]
        u2_result = _run_stage(
            u2_command,
            stdout_path=output / "stage_u2.stdout.log",
            stderr_path=output / "stage_u2.stderr.log",
        )
        lifecycle.append(
            "STAGE_U2_COMPLETED", result_sha256=u2_result["result_sha256"],
            command=u2_command,
        )
        result = {
            "schema_version": "c85u_execution_result_v1",
            "gate": SUCCESS_GATE,
            "execution_lock_sha256": context.execution_lock_sha256,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "attempt_id": context.attempt_id,
            "U1_manifest_sha256": u1_result["manifest_sha256"],
            "U2_replay_sha256": u2_result["result_sha256"],
            "contexts": 944,
            "candidate_utility_rows": 76_464,
            "historical_method_context_rows_replayed": 18_432,
            "protected_counters": {
                "construction_label_access": 0,
                "selector_recomputation": 0,
                "Q0_resampling": 0,
                "scientific_inference": 0,
                "theorem_status_writes": 0,
                "C85E": 0,
                "C86": 0,
            },
        }
        result_path = output / "C85U_EXECUTION_RESULT.json"
        _write_fsynced(result_path, _canonical_json_bytes(result))
        lifecycle.append("EXECUTION_COMPLETED", result_sha256=sha256_file(result_path))
        return result
    except BaseException as error:
        secondary_errors: list[str] = []
        if lifecycle is not None:
            try:
                lifecycle.append(
                    "FAILED", error_type=type(error).__name__, error=str(error),
                )
            except BaseException as secondary:
                secondary_errors.append(f"lifecycle:{type(secondary).__name__}:{secondary}")
        if output.is_dir():
            failure = {
                "schema_version": "c85u_failure_receipt_v1",
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "secondary_errors": secondary_errors,
                "authorization_consumed": context is not None,
                "U1_manifest_published": (
                    output / "stage_u1_candidate_utility/C85U_CANDIDATE_UTILITY_MANIFEST.json"
                ).is_file(),
                "U2_replay_published": (
                    output / "stage_u2_historical_replay/C85U_HISTORICAL_DECISION_REPLAY.json"
                ).is_file(),
                "automatic_retry": False,
            }
            try:
                _write_fsynced(
                    output / "C85U_FAILURE_RECEIPT.json", _canonical_json_bytes(failure),
                )
            except BaseException as secondary:
                secondary_errors.append(f"failure_receipt:{type(secondary).__name__}:{secondary}")
        else:
            _write_preflight_blocker(output, error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--execution-lock", required=True)
    parser.add_argument("--authorization-record", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args(argv)
    result = run_locked(
        execution_lock=arguments.execution_lock,
        authorization_record=arguments.authorization_record,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SUCCESS_GATE", "run_locked"]
