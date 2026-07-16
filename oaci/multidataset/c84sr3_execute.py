"""Single fail-closed real coordinator for C84S V5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from .c84s_common import read_json, require, sha256_file, write_json
from .c84sr3_common import DEFAULT_OUTPUT_ROOT, METHOD_CONTEXT_ROWS, Q0_RECORDS
from .c84sr3_runtime_guard import consume_authorization, pre_label_access_guard


def _run_stage(
    *, name: str, module: str, arguments: Sequence[str], run_root: Path,
) -> dict[str, Any]:
    command = [sys.executable, "-m", module, *arguments]
    stdout_path = run_root / f"{name.lower()}_stdout.log"
    stderr_path = run_root / f"{name.lower()}_stderr.log"
    started = time.time_ns()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    receipt = {
        "schema_version": "c84sr3_subprocess_receipt_v1", "stage": name,
        "module": module, "command": command, "exit_code": completed.returncode,
        "started_at_unix_ns": started, "finished_at_unix_ns": time.time_ns(),
        "stdout": {
            "path": str(stdout_path), "bytes": stdout_path.stat().st_size,
            "sha256": sha256_file(stdout_path),
        },
        "stderr": {
            "path": str(stderr_path), "bytes": stderr_path.stat().st_size,
            "sha256": sha256_file(stderr_path),
        },
    }
    receipt_path = run_root / f"{name.lower()}_subprocess_receipt.json"
    write_json(receipt_path, receipt)
    require(completed.returncode == 0,
            f"{name} subprocess failed with exit code {completed.returncode}")
    return {**receipt, "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path)}


def run_real(*, authorization_path: Path, output_root: Path) -> dict[str, Any]:
    binding = pre_label_access_guard(
        authorization_path=authorization_path, output_root=output_root,
        verify_external_bytes=True,
    )
    consumption = consume_authorization(binding)
    run_root = Path(binding["output_root"])
    lifecycle_path = run_root / "C84S_V5_LIFECYCLE_ATTEMPT.json"
    lifecycle: dict[str, Any] = {
        "schema_version": "c84sr3_lifecycle_attempt_v1", "status": "STARTED",
        "stage": "authorization_consumed", "started_at_unix_ns": time.time_ns(),
        "analysis_lock_sha256": binding["lock_sha256"],
        "authorization_consumption_sha256": consumption["sha256"],
        "historical_V4_authorization_reused": False,
        "protected_counters": {
            "stage_A_label_loader_calls": 0, "target_label_rows_reloaded": 0,
            "construction_label_access": 0, "evaluation_label_access": 0,
            "selector_score_contexts": 0, "scientific_result_rows": 0,
            "training": 0, "forward": 0, "GPU": 0, "same_label_oracle": 0,
        },
        "stage_receipts": [], "retry_disposition": "NOT_APPLICABLE",
    }
    write_json(lifecycle_path, lifecycle)
    write_json(run_root / "C84S_V5_PRE_LABEL_ACCESS_REPLAY.json", {
        "schema_version": "c84sr3_pre_label_access_replay_v1",
        "head": binding["head"], "lock_sha256": binding["lock_sha256"],
        "protocol_replay": binding["protocol_replay"],
        "environment_replay": binding["environment_replay"],
        "readiness_replay": binding["readiness_replay"],
        "external_replay": binding["external_replay"],
        "completed_before_authorization_consumption": True,
    })
    try:
        replay_root = run_root / "stage_a_immutable_replay"
        lifecycle["stage"] = "Stage_A_immutable_replay"
        write_json(lifecycle_path, lifecycle)
        stage_a = _run_stage(
            name="Stage_A_Replay", module="oaci.multidataset.c84sr3_stage_a_replay",
            arguments=(
                "run-replay", "--receipt", consumption["path"],
                "--output-root", str(replay_root),
            ), run_root=run_root,
        )
        lifecycle["stage_receipts"].append(stage_a)
        replay = read_json(replay_root / "C84S_STAGE_A_REPLAY.json")
        require(replay["status"] == "PASS" and replay["label_loader_calls"] == 0 and
                replay["target_label_rows_reloaded"] == 0 and
                replay["evaluation_descriptor_released_to_Stage_B"] is False,
                "C84SR3 immutable Stage-A replay contract failed")
        construction_handoff = Path(replay["construction_handoff_path"])
        evaluation_seal = Path(replay["evaluation_seal_path"])
        require(sha256_file(construction_handoff) == replay["construction_handoff_sha256"],
                "C84SR3 construction handoff drift")
        require(sha256_file(evaluation_seal) == replay["evaluation_seal_sha256"],
                "C84SR3 evaluation seal drift")

        stage_b_root = run_root / "stage_b_selection_freeze"
        lifecycle["stage"] = "Stage_B_V3_selection_evaluation_sealed"
        lifecycle["protected_counters"]["construction_label_access"] = 1
        write_json(lifecycle_path, lifecycle)
        stage_b = _run_stage(
            name="Stage_B_V3", module="oaci.multidataset.c84sr3_stage_b_selection",
            arguments=(
                "run-real", "--stage-a-handoff", str(construction_handoff),
                "--output-root", str(stage_b_root),
            ), run_root=run_root,
        )
        lifecycle["stage_receipts"].append(stage_b)
        freeze_path = stage_b_root / "C84S_SELECTION_FREEZE_MANIFEST_V3.json"
        freeze = read_json(freeze_path)
        require(freeze["status"] ==
                "SELECTION_FROZEN_EVALUATION_DESCRIPTOR_NOT_YET_AVAILABLE" and
                freeze["evaluation_label_descriptor_received"] is False and
                freeze["contexts"] == 944 and freeze["Q0_records"] == Q0_RECORDS,
                "C84SR3 Stage-B exact freeze failed before evaluation release")
        lifecycle["protected_counters"]["selector_score_contexts"] = 944

        stage_c_root = run_root / "stage_c_scientific_result"
        lifecycle["stage"] = "Stage_C_V3_evaluation_released_after_selection_freeze"
        lifecycle["protected_counters"]["evaluation_label_access"] = 1
        write_json(lifecycle_path, lifecycle)
        stage_c = _run_stage(
            name="Stage_C_V3", module="oaci.multidataset.c84sr3_stage_c_evaluation",
            arguments=(
                "run-real", "--selection-root", str(stage_b_root),
                "--evaluation-seal", str(evaluation_seal),
                "--output-root", str(stage_c_root),
            ), run_root=run_root,
        )
        lifecycle["stage_receipts"].append(stage_c)
        result_path = stage_c_root / "C84S_RESULT.json"
        result = read_json(result_path)
        require(result["method_context_rows"] == METHOD_CONTEXT_ROWS,
                "C84SR3 final method-context arithmetic drift")
        lifecycle["protected_counters"]["scientific_result_rows"] = METHOD_CONTEXT_ROWS
        lifecycle.update({
            "status": "COMPLETE", "stage": "complete",
            "finished_at_unix_ns": time.time_ns(),
            "selection_freeze_sha256": sha256_file(freeze_path),
            "scientific_result_sha256": sha256_file(result_path),
            "final_gate": result["primary_gate"],
            "label_frontier_tag": result["label_frontier_tag"],
        })
        write_json(lifecycle_path, lifecycle)
        final = {
            "schema_version": "c84sr3_real_execution_result_v1",
            "status": "COMPLETE", "final_gate": result["primary_gate"],
            "label_frontier_tag": result["label_frontier_tag"],
            "selection_freeze_sha256": lifecycle["selection_freeze_sha256"],
            "scientific_result_sha256": lifecycle["scientific_result_sha256"],
            "authorization_consumption_sha256": consumption["sha256"],
            "stage_A_reused_without_label_reload": True,
            "Lee_Q0_B32_status": "INPUT_UNAVAILABLE_NO_SELECTION_OR_RESULT_ROW",
            "training": 0, "forward": 0, "GPU": 0,
            "same_label_oracle": 0, "C85_authorized": False,
        }
        write_json(run_root / "C84S_V5_EXECUTION_RESULT.json", final)
        return final
    except BaseException as error:
        lifecycle.update({
            "status": "FAILED", "finished_at_unix_ns": time.time_ns(),
            "error_type": type(error).__name__, "error": str(error),
            "error_notes": list(getattr(error, "__notes__", ())),
            "retry_disposition": "NO_AUTOMATIC_RETRY_PM_REVIEW_REQUIRED",
            "evaluation_descriptor_sealed":
                lifecycle["protected_counters"]["evaluation_label_access"] == 0,
        })
        write_json(lifecycle_path, lifecycle)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--authorization-record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = run_real(
        authorization_path=args.authorization_record, output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
