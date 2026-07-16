"""Standalone C84SR1 Stage-A label provisioning command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from . import c84s_label_views as label_views
from .c84s_common import canonical_sha256, read_json, require, sha256_file, write_json
from .c84sr1_common import replay_stage_receipt, write_stage_receipt


def _attempt_path(output_root: Path) -> Path:
    return output_root.parent / f"{output_root.name}.stage_a_attempt.json"


def _publish_handoffs(
    output_root: Path,
    construction: label_views.LabelViewDescriptor,
    evaluation: label_views.LabelViewDescriptor,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    construction_payload = {
        "schema_version": "c84sr1_stage_a_to_b_handoff_v1",
        "stage": "Stage_A_to_Stage_B",
        "construction_descriptor": {
            "kind": construction.kind, "root": construction.root,
            "manifest_sha256": construction.manifest_sha256,
            "row_count": construction.row_count,
            "descriptor_sha256": construction.descriptor_sha256,
        },
        "candidate_artifact_access": 0,
    }
    handoff_sha = write_stage_receipt(output_root / "C84S_STAGE_A_HANDOFF.json", construction_payload)
    (output_root / "C84S_STAGE_A_HANDOFF.sha256").write_text(
        f"{handoff_sha}  C84S_STAGE_A_HANDOFF.json\n", encoding="ascii",
    )
    evaluation_payload = {
        "schema_version": "c84sr1_stage_a_evaluation_seal_v1",
        "stage": "Stage_A_evaluation_seal",
        "evaluation_descriptor": {
            "kind": evaluation.kind, "root": evaluation.root,
            "manifest_sha256": evaluation.manifest_sha256,
            "row_count": evaluation.row_count,
            "descriptor_sha256": evaluation.descriptor_sha256,
        },
        "released_to_Stage_B": False,
        "released_to_Stage_C": False,
    }
    evaluation_sha = write_stage_receipt(
        output_root / "C84S_STAGE_A_EVALUATION_SEAL.json", evaluation_payload,
    )
    (output_root / "C84S_STAGE_A_EVALUATION_SEAL.sha256").write_text(
        f"{evaluation_sha}  C84S_STAGE_A_EVALUATION_SEAL.json\n", encoding="ascii",
    )
    summary = {
        "schema_version": "c84sr1_stage_a_complete_v1",
        "stage": "Stage_A",
        "construction_handoff_sha256": handoff_sha,
        "evaluation_seal_sha256": evaluation_sha,
        "split_audit": dict(audit),
        "protected": {
            "candidate_artifact_access": 0, "selector_scores": 0,
            "scientific_statistics": 0, "same_label_oracle": 0,
        },
    }
    summary_sha = write_stage_receipt(output_root / "C84S_STAGE_A_COMPLETE.json", summary)
    return {**summary, "sha256": summary_sha}


def run_stage_a_from_rows(
    *,
    guard_receipt: Mapping[str, Any],
    frozen_registry_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    output_root: str | Path,
) -> dict[str, Any]:
    require(guard_receipt.get("C84S_authorized") is True, "Stage A lacks C84S authorization receipt")
    output_root = Path(output_root)
    require(not output_root.exists(), "Stage-A output root exists")
    output_root.mkdir(parents=True)
    construction_rows, evaluation_rows, audit = label_views.align_and_split_labels(
        frozen_registry_rows, label_rows,
    )
    construction, evaluation = label_views.publish_physical_label_views(
        output_root, construction_rows, evaluation_rows,
    )
    return _publish_handoffs(output_root, construction, evaluation, audit)


def run_stage_a_real(
    *, receipt_path: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    receipt = replay_stage_receipt(receipt_path, stage="C84S_authorization_consumed")
    require(receipt.get("C84S_authorized") is True, "Stage-A authorization receipt is inactive")
    output_root = Path(output_root)
    attempt_path = _attempt_path(output_root)
    require(not output_root.exists() and not attempt_path.exists(), "Stage-A root/attempt already exists")
    started = time.time_ns()
    attempt = {
        "schema_version": "c84sr1_stage_attempt_v1", "stage": "Stage_A",
        "status": "STARTED", "started_at_unix_ns": started,
        "authorization_receipt_sha256": sha256_file(receipt_path),
        "label_loader_calls": 0, "candidate_artifact_access": 0,
        "selector_scores": 0, "scientific_statistics": 0,
    }
    write_json(attempt_path, attempt)
    try:
        construction, evaluation, audit = label_views.provision_real_label_views(
            guard_receipt=receipt, output_root=output_root,
        )
        result = _publish_handoffs(output_root, construction, evaluation, audit)
        attempt.update({
            "status": "COMPLETE", "finished_at_unix_ns": time.time_ns(),
            "label_loader_calls": int(audit["loader_calls"]),
            "stage_output_identity": canonical_sha256(result),
        })
        write_json(attempt_path, attempt)
        return result
    except BaseException as error:
        attempt.update({
            "status": "FAILED", "finished_at_unix_ns": time.time_ns(),
            "error_type": type(error).__name__, "error": str(error),
            "partial_root_exists": output_root.exists(),
        })
        write_json(attempt_path, attempt)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = run_stage_a_real(receipt_path=args.receipt, output_root=args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
