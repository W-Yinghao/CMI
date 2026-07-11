#!/usr/bin/env python
"""Immutable afterok closure for all six STAR_01A final-step checkpoints."""

import argparse
import csv
import io
import json
import stat
from pathlib import Path
from typing import Dict, List, Mapping

from star_eeg.config import STAR01
from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.training.approval_lock import (
    APPROVED_CELLS,
    validate_approval_lock,
)
from star_eeg.training.persistence import (
    atomic_copy_file_no_overwrite,
    atomic_write_bytes_no_overwrite,
    atomic_write_json_no_overwrite,
    claim_attempt_directory,
    freeze_completed_attempt,
    no_temporary_files,
    sha256_file,
    tree_is_read_only,
    validate_telemetry_file,
)


def _load_hashed_json(path: Path, hash_field: str) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text())
    core = {key: value for key, value in payload.items() if key != hash_field}
    if canonical_hash(core) != payload.get(hash_field):
        raise RuntimeError(f"JSON hash mismatch: {path}")
    return payload


def _attempt_record(
    attempt_dir: Path,
    cell: str,
    approval_hash: str,
) -> Dict[str, object]:
    from star_eeg.training.real_star_runner import strict_reload_training_checkpoint

    attempt_dir = Path(attempt_dir)
    if not attempt_dir.is_dir() or attempt_dir.is_symlink():
        raise RuntimeError(f"training attempt missing: {attempt_dir}")
    completion = _load_hashed_json(attempt_dir / "completion.json", "completion_hash")
    run_summary = _load_hashed_json(
        attempt_dir / "run_summary.json", "run_summary_hash"
    )
    run_manifest = _load_hashed_json(
        attempt_dir / "run_manifest.json", "run_manifest_hash"
    )
    execution = _load_hashed_json(
        attempt_dir / "execution_manifest.json", "execution_manifest_hash"
    )
    if completion.get("status") != "COMPLETE" or not all(completion.get("checks", {}).values()):
        raise RuntimeError(f"cell completion gate failed: {cell}")
    if completion.get("cell") != cell or run_summary.get("cell") != cell:
        raise RuntimeError(f"cell identity mismatch: {cell}")
    if int(run_summary.get("optimizer_steps", -1)) != STAR01.continuation_optimizer_steps:
        raise RuntimeError(f"cell did not complete 3750 steps: {cell}")
    if execution.get("approval_manifest_hash") != approval_hash:
        raise RuntimeError(f"approval hash mismatch: {cell}")
    telemetry_path = attempt_dir / "telemetry.jsonl"
    telemetry = validate_telemetry_file(
        telemetry_path, STAR01.continuation_optimizer_steps
    )
    if telemetry["sha256"] != run_summary.get("telemetry_sha256"):
        raise RuntimeError(f"telemetry hash mismatch: {cell}")
    checkpoint_path = attempt_dir / "checkpoints" / "step_3750.pth"
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise RuntimeError(f"final checkpoint missing: {cell}")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != run_summary.get("checkpoint_sha256"):
        raise RuntimeError(f"final checkpoint SHA mismatch: {cell}")
    if not strict_reload_training_checkpoint(checkpoint_path, "cpu"):
        raise RuntimeError(f"final checkpoint strict reload failed: {cell}")
    if not tree_is_read_only(attempt_dir):
        raise RuntimeError(f"completed attempt is not frozen read-only: {cell}")
    return {
        "cell": cell,
        "variant": run_summary["variant"],
        "model_seed": int(run_summary["model_seed"]),
        "attempt_id": run_summary["attempt_id"],
        "optimizer_steps": int(run_summary["optimizer_steps"]),
        "telemetry_rows": int(telemetry["rows"]),
        "telemetry_sha256": telemetry["sha256"],
        "run_manifest_hash": run_manifest["run_manifest_hash"],
        "execution_manifest_hash": execution["execution_manifest_hash"],
        "run_summary_hash": run_summary["run_summary_hash"],
        "completion_hash": completion["completion_hash"],
        "source_checkpoint_sha_before": run_summary["source_checkpoint_sha_before"],
        "source_checkpoint_sha_after": run_summary["source_checkpoint_sha_after"],
        "source_checkpoint_unchanged": run_summary["source_checkpoint_unchanged"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_strict_reload_pass": True,
        "wall_seconds": run_summary["wall_seconds"],
        "peak_gpu_memory_bytes": run_summary["peak_gpu_memory_bytes"],
        "losses_finite": run_summary["losses_finite"],
        "gradients_finite": run_summary["gradients_finite"],
        "parameter_deltas_finite": run_summary["parameter_deltas_finite"],
        "target_data_used": False,
        "source_only_task_gate_status": "NOT_RUN_SEPARATE_AFTER_CLOSURE",
    }


def _write_completion_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes_no_overwrite(path, buffer.getvalue().encode("utf-8"))


def close_star01a_finals(
    repo_root: Path,
    runtime_root: Path,
    attempt_id: str,
    approval_path: Path,
    closure_output_dir: Path,
) -> Dict[str, object]:
    repo_root = Path(repo_root).resolve()
    approval = validate_approval_lock(approval_path, repo_root)
    closure_dir = claim_attempt_directory(closure_output_dir, attempt_id)
    checkpoint_root = closure_dir / "checkpoints"
    checkpoint_root.mkdir()
    records = []
    for task_id, cell in enumerate(APPROVED_CELLS):
        attempt_dir = Path(runtime_root) / cell / attempt_id
        record = _attempt_record(
            attempt_dir,
            cell=cell,
            approval_hash=approval["approval_manifest_hash"],
        )
        source = Path(record["checkpoint_path"])
        destination_dir = checkpoint_root / cell
        destination_dir.mkdir()
        destination = destination_dir / f"step_3750.{record['checkpoint_sha256']}.pth"
        copy_audit = atomic_copy_file_no_overwrite(source, destination)
        destination.chmod(0o444)
        from star_eeg.training.real_star_runner import strict_reload_training_checkpoint

        strict_reload_training_checkpoint(destination, "cpu")
        records.append({
            "task_id": task_id,
            **record,
            "immutable_checkpoint": str(destination.resolve()),
            "immutable_mode_octal": oct(stat.S_IMODE(destination.stat().st_mode)),
            "immutable_sha256": sha256_file(destination),
            "copy_source_sha_stable": (
                copy_audit["source_sha_before"]
                == copy_audit["source_sha_after"]
                == copy_audit["destination_sha"]
            ),
        })

    all_checks = {
        "six_cells_present": tuple(row["cell"] for row in records) == APPROVED_CELLS,
        "all_3750_steps": all(row["optimizer_steps"] == 3750 for row in records),
        "all_3750_telemetry_rows": all(row["telemetry_rows"] == 3750 for row in records),
        "all_final_checkpoints_strict_reload": all(
            row["checkpoint_strict_reload_pass"] for row in records
        ),
        "all_sources_unchanged": all(row["source_checkpoint_unchanged"] for row in records),
        "all_integrity_finite": all(
            row["losses_finite"]
            and row["gradients_finite"]
            and row["parameter_deltas_finite"]
            for row in records
        ),
        "all_immutable_shas_match": all(
            row["immutable_sha256"] == row["checkpoint_sha256"] for row in records
        ),
        "all_copy_sources_stable": all(row["copy_source_sha_stable"] for row in records),
        "no_target_data_used": all(row["target_data_used"] is False for row in records),
        "no_temporary_files": no_temporary_files(closure_dir),
    }
    if not all(all_checks.values()):
        raise RuntimeError(f"STAR_01A immutable closure checks failed: {all_checks}")
    manifest_core = {
        "schema_version": 1,
        "phase": "STAR_01A_FINAL_CHECKPOINT_IMMUTABLE_CLOSURE",
        "approval_manifest_hash": approval["approval_manifest_hash"],
        "attempt_id": attempt_id,
        "checkpoints": records,
        "target_scoring_allowed": False,
        "source_only_task_gate_controls_mechanism_interpretation_only": True,
        "all_cells_remain_required_for_future_target_scoring": True,
    }
    manifest = {
        **manifest_core,
        "star01_final_checkpoint_manifest_hash": canonical_hash(manifest_core),
    }
    atomic_write_json_no_overwrite(
        closure_dir / "star01_final_checkpoint_manifest.json", manifest
    )
    _write_completion_csv(
        closure_dir / "star01_training_completion_matrix.csv", records
    )
    gate_core = {
        "schema_version": 1,
        "phase": "STAR_01A_FINAL_CHECKPOINT_IMMUTABLE_CLOSURE",
        "status": "PASS",
        "approval_manifest_hash": approval["approval_manifest_hash"],
        "attempt_id": attempt_id,
        "checks": all_checks,
        "closed_cells": list(APPROVED_CELLS),
        "target_metrics_computed": False,
        "target_scoring_allowed": False,
        "source_only_task_audit_next": True,
    }
    gate = {**gate_core, "star01_closure_go_nogo_hash": canonical_hash(gate_core)}
    atomic_write_json_no_overwrite(
        closure_dir / "star01_closure_go_nogo.json", gate
    )
    closure_completion_core = {
        "status": "COMPLETE",
        "attempt_id": attempt_id,
        "manifest_hash": manifest["star01_final_checkpoint_manifest_hash"],
        "go_nogo_hash": gate["star01_closure_go_nogo_hash"],
        "completion_matrix_sha256": sha256_file(
            closure_dir / "star01_training_completion_matrix.csv"
        ),
        "target_metrics_computed": False,
    }
    atomic_write_json_no_overwrite(
        closure_dir / "completion.json",
        {
            **closure_completion_core,
            "completion_hash": canonical_hash(closure_completion_core),
        },
    )
    freeze_completed_attempt(closure_dir)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--approval-manifest", required=True)
    parser.add_argument("--closure-output-dir", required=True)
    args = parser.parse_args()
    result = close_star01a_finals(
        repo_root=Path(args.repo_root),
        runtime_root=Path(args.runtime_root),
        attempt_id=args.attempt_id,
        approval_path=Path(args.approval_manifest),
        closure_output_dir=Path(args.closure_output_dir),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
