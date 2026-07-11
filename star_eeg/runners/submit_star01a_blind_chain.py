#!/usr/bin/env python
"""Submit the approval-locked six-cell array and its afterok closure once."""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict

from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.training.approval_lock import (
    APPROVED_CELLS,
    validate_approval_lock,
)
from star_eeg.training.persistence import (
    atomic_write_json_no_overwrite,
    sha256_file,
)


def _fresh_or_empty(path: Path) -> bool:
    return not path.exists() or (
        path.is_dir() and not path.is_symlink() and not any(path.iterdir())
    )


def _sbatch(repo_root: Path, *arguments: str) -> str:
    output = subprocess.check_output(
        ["sbatch", "--parsable", *arguments],
        cwd=str(repo_root),
        text=True,
    ).strip()
    job_id = output.split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unexpected sbatch job identifier: {output}")
    return job_id


def submit_blind_chain(
    repo_root: Path,
    approval_path: Path,
    attempt_id: str,
    runtime_root: Path,
    closure_root: Path,
    control_dir: Path,
) -> Dict[str, object]:
    repo_root = Path(repo_root).resolve()
    approval_path = Path(approval_path).resolve()
    approval = validate_approval_lock(approval_path, repo_root)
    if approval["approved_attempt_id"] != attempt_id:
        raise PermissionError("attempt_id differs from the single-use approval lock")
    gate_path = repo_root / "results/star/star00c_preflight/star00c_go_nogo.json"
    gate = json.loads(gate_path.read_text())
    gate_core = {
        key: value for key, value in gate.items()
        if key != "star00c_go_nogo_hash"
    }
    if canonical_hash(gate_core) != gate.get("star00c_go_nogo_hash"):
        raise RuntimeError("STAR_00C go/no-go hash mismatch")
    if gate.get("status") != "PASS" or gate.get(
        "STAR_01A_BLIND_SIX_CELL_TRAINING"
    ) != "CONDITIONALLY_APPROVED_AFTER_APPROVAL_LOCK":
        raise PermissionError("STAR_00C machine gate has not passed")
    attempt_paths = {
        cell: Path(runtime_root) / cell / attempt_id for cell in APPROVED_CELLS
    }
    blocked = [cell for cell, path in attempt_paths.items() if not _fresh_or_empty(path)]
    closure_output = Path(closure_root) / attempt_id
    if blocked or not _fresh_or_empty(closure_output):
        raise FileExistsError(
            f"attempt output collision: cells={blocked}, closure={closure_output}"
        )
    control_dir = Path(control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = control_dir / f"launch_receipt.{attempt_id}.json"
    array_marker_path = control_dir / f"array_submission.{attempt_id}.json"
    if receipt_path.exists() or array_marker_path.exists():
        raise FileExistsError(
            f"launch or array receipt already exists for {attempt_id}"
        )
    Path("/home/infres/yinwang/CMI_AAAI_star_runtime/logs").mkdir(
        parents=True, exist_ok=True
    )
    export = (
        f"ALL,STAR_APPROVAL={approval_path},STAR_ATTEMPT_ID={attempt_id},"
        f"STAR_REPO_ROOT={repo_root}"
    )
    array_job_id = _sbatch(
        repo_root,
        f"--export={export}",
        "star_eeg/slurm/star01_train_array.sbatch",
    )
    array_marker_core = {
        "phase": "STAR_01A_ARRAY_SUBMISSION",
        "attempt_id": attempt_id,
        "array_job_id": array_job_id,
        "array_task_ids": [f"{array_job_id}_{index}" for index in range(6)],
        "cells": list(APPROVED_CELLS),
        "approval_manifest_hash": approval["approval_manifest_hash"],
        "target_scoring_submitted": False,
    }
    atomic_write_json_no_overwrite(array_marker_path, {
        **array_marker_core,
        "array_submission_hash": canonical_hash(array_marker_core),
    })
    array_marker_path.chmod(0o444)
    closure_export = (
        f"{export},STAR_ARRAY_JOB_ID={array_job_id}"
    )
    closure_job_id = _sbatch(
        repo_root,
        f"--dependency=afterok:{array_job_id}",
        f"--export={closure_export}",
        "star_eeg/slurm/star01a_final_closure.sbatch",
    )
    core = {
        "schema_version": 1,
        "phase": "STAR_01A_BLIND_SIX_CELL_TRAINING_LAUNCH",
        "approved_execution_commit": approval["approved_execution_commit"],
        "approval_manifest": str(approval_path),
        "approval_manifest_hash": approval["approval_manifest_hash"],
        "approval_file_sha256": sha256_file(approval_path),
        "attempt_id": attempt_id,
        "array_job_id": array_job_id,
        "array_task_ids": [f"{array_job_id}_{index}" for index in range(6)],
        "cells": list(APPROVED_CELLS),
        "all_six_submitted_together": True,
        "closure_job_id": closure_job_id,
        "closure_dependency": f"afterok:{array_job_id}",
        "target_scoring_submitted": False,
        "target_scoring": "BLOCKED",
        "array_submission_marker": str(array_marker_path.resolve()),
        "array_submission_marker_sha256": sha256_file(array_marker_path),
    }
    receipt = {**core, "launch_receipt_hash": canonical_hash(core)}
    atomic_write_json_no_overwrite(receipt_path, receipt)
    receipt_path.chmod(0o444)
    return {**receipt, "launch_receipt_path": str(receipt_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--approval-manifest", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--runtime-root",
        default="/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01",
    )
    parser.add_argument(
        "--closure-root",
        default="/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01_closure",
    )
    parser.add_argument(
        "--control-dir",
        default="/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01_control",
    )
    args = parser.parse_args()
    result = submit_blind_chain(
        repo_root=Path(args.repo_root),
        approval_path=Path(args.approval_manifest),
        attempt_id=args.attempt_id,
        runtime_root=Path(args.runtime_root),
        closure_root=Path(args.closure_root),
        control_dir=Path(args.control_dir),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
