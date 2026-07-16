"""Replay the immutable C84S Stage-A views under a V5 authorization receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

from .c84s_common import atomic_publish_directory, require, sha256_file, write_json
from .c84sr1_common import replay_stage_receipt
from .c84sr2_stage_a_replay import replay_historical_stage_a


def run_replay(*, receipt_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    receipt = replay_stage_receipt(receipt_path, stage="C84S_V5_authorization_consumed")
    require(receipt.get("C84S_authorized") is True,
            "C84SR3 Stage-A replay lacks V5 authorization")
    require(receipt.get("historical_V4_authorization_reused") is False,
            "C84SR3 Stage-A replay received a migrated V4 authorization")
    output_root = Path(output_root)
    attempt_path = output_root.parent / f"{output_root.name}.stage_a_replay_attempt.json"
    require(not output_root.exists() and not attempt_path.exists(),
            "C84SR3 Stage-A replay output already exists")
    attempt = {
        "schema_version": "c84sr3_stage_attempt_v1",
        "stage": "Stage_A_immutable_replay", "status": "STARTED",
        "started_at_unix_ns": time.time_ns(),
        "authorization_receipt_sha256": sha256_file(receipt_path),
        "label_loader_calls": 0, "target_label_rows_reloaded": 0,
    }
    write_json(attempt_path, attempt)
    try:
        result = replay_historical_stage_a()
        result = {
            **result, "schema_version": "c84sr3_immutable_stage_a_replay_v1",
            "V5_authorization_receipt_sha256": sha256_file(receipt_path),
        }
        atomic_publish_directory(
            output_root,
            lambda staging: write_json(staging / "C84S_STAGE_A_REPLAY.json", result),
        )
        attempt.update({
            "status": "COMPLETE", "finished_at_unix_ns": time.time_ns(),
            "result_sha256": sha256_file(output_root / "C84S_STAGE_A_REPLAY.json"),
        })
        write_json(attempt_path, attempt)
        return result
    except BaseException as error:
        attempt.update({
            "status": "FAILED", "finished_at_unix_ns": time.time_ns(),
            "error_type": type(error).__name__, "error": str(error),
            "error_notes": list(getattr(error, "__notes__", ())),
        })
        write_json(attempt_path, attempt)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-replay",))
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = run_replay(receipt_path=args.receipt, output_root=args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
