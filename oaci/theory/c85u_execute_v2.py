"""Lock-bound C85U V2 coordinator for U1, U2, and atomic acceptance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from oaci.multidataset.c84s_common import require

from .c85u_acceptance_transaction_v2 import (
    AtomicC85UAcceptanceTransactionV2,
    SUCCESS_GATE,
    preserve_primary_exception_v2,
)
from .c85u_result_manifest_v2 import U1_HANDOFF_NAME
from .c85u_runtime_guard_v2 import (
    AppendOnlyLifecycleV2,
    C85UExecutionContextV2,
    canonical_json_bytes,
    create_execution_context_v2,
    execution_context_record_v2,
    replay_protected_inputs_v2,
)
from .c85u_u1_registry_v2 import build_u1_runtime_registry


def _write_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _run_stage(
    command: list[str], *, stdout_path: Path, stderr_path: Path,
) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    _write_fsynced(stdout_path, completed.stdout.encode("utf-8"))
    _write_fsynced(stderr_path, completed.stderr.encode("utf-8"))
    require(completed.returncode == 0, f"C85U V2 subprocess failed: {command[2]}")
    require(completed.stderr == "", f"C85U V2 subprocess stderr nonempty: {command[2]}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    require(lines, f"C85U V2 subprocess result absent: {command[2]}")
    value = json.loads(lines[-1])
    require(isinstance(value, dict), "C85U V2 subprocess result malformed")
    return value


def _write_preflight_blocker(output: Path, error: BaseException) -> None:
    blocker = output.parent / f".{output.name}.c85u_v2_preflight_blocker.json"
    if blocker.exists():
        return
    try:
        _write_fsynced(blocker, canonical_json_bytes({
            "schema_version": "c85u_v2_preflight_blocker_v1",
            "status": "FAILED_BEFORE_PROTECTED_INPUT_REPLAY",
            "error_type": type(error).__name__,
            "error": str(error),
            "timestamp_unix_ns": time.time_ns(),
        }))
    except BaseException:
        pass


def run_locked_v2(
    *, execution_lock: str | Path, authorization_record: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    context: C85UExecutionContextV2 | None = None
    try:
        context, _ = create_execution_context_v2(
            execution_lock=execution_lock,
            authorization_record=authorization_record,
            output_root=output,
        )
        registry = build_u1_runtime_registry()
        context = replay_protected_inputs_v2(context, registry)
        context_path = output / "C85U_EXECUTION_CONTEXT_V2.json"
        _write_fsynced(context_path, canonical_json_bytes(execution_context_record_v2(context)))
        lifecycle = AppendOnlyLifecycleV2(context.lifecycle_path)

        lifecycle.append("STAGE_U1_STARTED", context=context)
        u1_root = output / "stage_u1_candidate_utility_v2"
        u1_command = [
            sys.executable, "-m", "oaci.theory.c85u_stage_u1_v2", "run-real",
            "--execution-context", str(context_path),
            "--output-root", str(u1_root),
        ]
        u1 = _run_stage(
            u1_command,
            stdout_path=output / "stage_u1_v2.stdout.log",
            stderr_path=output / "stage_u1_v2.stderr.log",
        )
        require(AppendOnlyLifecycleV2(context.lifecycle_path).replay()[-1]["stage"]
                == "STAGE_U1_COMPLETED",
                "C85U V2 U1 subprocess did not freeze lifecycle completion")

        lifecycle.append("STAGE_U2_STARTED", context=context)
        u2_root = output / "stage_u2_historical_replay_v2"
        u2_command = [
            sys.executable, "-m", "oaci.theory.c85u_stage_u2_v2", "run-real",
            "--execution-context", str(context_path),
            "--u1-handoff", str(u1_root / U1_HANDOFF_NAME),
            "--utility-root", str(u1_root),
            "--output-root", str(u2_root),
        ]
        u2 = _run_stage(
            u2_command,
            stdout_path=output / "stage_u2_v2.stdout.log",
            stderr_path=output / "stage_u2_v2.stderr.log",
        )
        require(AppendOnlyLifecycleV2(context.lifecycle_path).replay()[-1]["stage"]
                == "STAGE_U2_COMPLETED",
                "C85U V2 U2 subprocess did not freeze lifecycle completion")

        completion = AtomicC85UAcceptanceTransactionV2(context).publish(
            u1_root=u1_root,
            u2_root=u2_root,
            u1_handoff_sha256=u1["handoff_sha256"],
            u2_handoff_sha256=u2["handoff_sha256"],
        )
        # No required write, replay, callback, or lifecycle append follows the
        # final acceptance rename. This return uses the pre-rename receipt data.
        return {
            "gate": SUCCESS_GATE,
            "classification": "SUCCESS",
            "attempt_id": completion["attempt_id"],
            "manifest_sha256": completion["manifest_sha256"],
            "output_root": completion["output_root"],
        }
    except BaseException as error:
        if context is None:
            _write_preflight_blocker(output, error)
        else:
            recovered = preserve_primary_exception_v2(context, error)
            if recovered is not None:
                return recovered
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--execution-lock", required=True)
    parser.add_argument("--authorization-record", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args(argv)
    result = run_locked_v2(
        execution_lock=arguments.execution_lock,
        authorization_record=arguments.authorization_record,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SUCCESS_GATE", "run_locked_v2"]
