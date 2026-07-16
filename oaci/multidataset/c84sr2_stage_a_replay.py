"""Replay the immutable authorized V3 Stage-A views without reloading labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .c84s_common import atomic_publish_directory, read_json, require, sha256_file, write_json
from .c84sr1_common import replay_stage_receipt
from .c84sr2_common import (
    HISTORICAL_ATTEMPT_IDENTITIES, HISTORICAL_STAGE_A_ROOT, HISTORICAL_V3_ROOT,
    STAGE_A_IDENTITIES,
)


def replay_historical_stage_a() -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative, expected in HISTORICAL_ATTEMPT_IDENTITIES.items():
        path = HISTORICAL_V3_ROOT / relative
        require(path.is_file() and sha256_file(path) == expected,
                f"historical V3 attempt identity drift: {relative}")
        observed[relative] = expected
    for relative, expected in STAGE_A_IDENTITIES.items():
        path = HISTORICAL_STAGE_A_ROOT / relative
        require(path.is_file() and sha256_file(path) == expected,
                f"immutable Stage-A identity drift: {relative}")
        observed[f"stage_a_labels/{relative}"] = expected

    lifecycle = read_json(HISTORICAL_V3_ROOT / "C84S_LIFECYCLE_ATTEMPT.json")
    counters = lifecycle["protected_counters"]
    require(lifecycle["status"] == "FAILED" and lifecycle["stage"] == "Stage_B_selection_evaluation_sealed",
            "historical V3 lifecycle disposition drift")
    require(counters == {
        "construction_label_access": 1, "evaluation_label_access": 0,
        "selector_score_contexts": 0, "scientific_result_rows": 0,
        "training": 0, "forward": 0, "GPU": 0, "same_label_oracle": 0,
    }, "historical V3 protected counters drift")
    require(lifecycle.get("evaluation_descriptor_sealed") is True,
            "historical evaluation descriptor was not sealed")

    complete = replay_stage_receipt(
        HISTORICAL_STAGE_A_ROOT / "C84S_STAGE_A_COMPLETE.json", stage="Stage_A",
    )
    handoff = replay_stage_receipt(
        HISTORICAL_STAGE_A_ROOT / "C84S_STAGE_A_HANDOFF.json",
        stage="Stage_A_to_Stage_B",
    )
    seal = replay_stage_receipt(
        HISTORICAL_STAGE_A_ROOT / "C84S_STAGE_A_EVALUATION_SEAL.json",
        stage="Stage_A_evaluation_seal",
    )
    require(complete["construction_handoff_sha256"] == STAGE_A_IDENTITIES["C84S_STAGE_A_HANDOFF.json"] and
            complete["evaluation_seal_sha256"] == STAGE_A_IDENTITIES["C84S_STAGE_A_EVALUATION_SEAL.json"],
            "Stage-A complete receipt linkage drift")
    require(complete["split_audit"]["construction_rows"] == 4773 and
            complete["split_audit"]["evaluation_rows"] == 4848 and
            complete["split_audit"]["overlap"] == 0,
            "Stage-A split arithmetic drift")
    require(handoff["candidate_artifact_access"] == 0,
            "Stage-A construction handoff reached candidate artifacts")
    require(seal["released_to_Stage_B"] is False and seal["released_to_Stage_C"] is False,
            "historical evaluation seal release state drift")

    construction = handoff["construction_descriptor"]
    evaluation = seal["evaluation_descriptor"]
    require(construction["kind"] == "construction" and construction["row_count"] == 4773,
            "construction descriptor drift")
    require(evaluation["kind"] == "evaluation" and evaluation["row_count"] == 4848,
            "evaluation descriptor drift")
    require(Path(construction["root"]).resolve() ==
            (HISTORICAL_STAGE_A_ROOT / "target_construction_label_view").resolve(),
            "construction root drift")
    require(Path(evaluation["root"]).resolve() ==
            (HISTORICAL_STAGE_A_ROOT / "target_evaluation_label_view").resolve(),
            "evaluation root drift")
    require(construction["manifest_sha256"] ==
            STAGE_A_IDENTITIES["target_construction_label_view/manifest.json"],
            "construction manifest linkage drift")
    require(evaluation["manifest_sha256"] ==
            STAGE_A_IDENTITIES["target_evaluation_label_view/manifest.json"],
            "evaluation manifest linkage drift")
    return {
        "schema_version": "c84sr2_immutable_stage_a_replay_v1",
        "status": "PASS", "historical_job": 897843,
        "historical_root": str(HISTORICAL_V3_ROOT),
        "construction_handoff_path": str(HISTORICAL_STAGE_A_ROOT / "C84S_STAGE_A_HANDOFF.json"),
        "construction_handoff_sha256": STAGE_A_IDENTITIES["C84S_STAGE_A_HANDOFF.json"],
        "evaluation_seal_path": str(HISTORICAL_STAGE_A_ROOT / "C84S_STAGE_A_EVALUATION_SEAL.json"),
        "evaluation_seal_sha256": STAGE_A_IDENTITIES["C84S_STAGE_A_EVALUATION_SEAL.json"],
        "files_replayed": len(observed), "file_identities": observed,
        "label_loader_calls": 0, "target_label_rows_reloaded": 0,
        "selector_scores": 0, "scientific_statistics": 0,
        "evaluation_descriptor_released_to_Stage_B": False,
    }


def run_replay(*, receipt_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    receipt = replay_stage_receipt(receipt_path, stage="C84S_V4_authorization_consumed")
    require(receipt.get("C84S_authorized") is True, "Stage-A replay lacks V4 authorization")
    output_root = Path(output_root)
    attempt_path = output_root.parent / f"{output_root.name}.stage_a_replay_attempt.json"
    require(not output_root.exists() and not attempt_path.exists(), "Stage-A replay output already exists")
    attempt = {
        "schema_version": "c84sr2_stage_attempt_v1", "stage": "Stage_A_immutable_replay",
        "status": "STARTED", "started_at_unix_ns": time.time_ns(),
        "authorization_receipt_sha256": sha256_file(receipt_path),
        "label_loader_calls": 0, "target_label_rows_reloaded": 0,
    }
    write_json(attempt_path, attempt)
    try:
        result = replay_historical_stage_a()
        atomic_publish_directory(output_root, lambda staging: write_json(
            staging / "C84S_STAGE_A_REPLAY.json", result,
        ))
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
