"""Commit- and artifact-bound PM approval lock for STAR_01A training."""

import csv
import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from star_eeg.config import PROJECT_NAME, STAR01, STAR_BRANCH
from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.training.persistence import (
    atomic_write_json_no_overwrite,
    sha256_file,
)


APPROVED_CELLS: Tuple[str, ...] = (
    "H200_SSL_CONT_s0",
    "H200_SSL_CONT_s1",
    "H200_STAR_TRUE_s0",
    "H200_STAR_TRUE_s1",
    "H200_STAR_SHUFFLED_s0",
    "H200_STAR_SHUFFLED_s1",
)

ARTIFACT_BINDINGS = {
    "h200_immutable_manifest": "results/star/star00b_preflight/h200_immutable_manifest.json",
    "h200_immutable_go_nogo": "results/star/star00b_preflight/h200_immutable_go_nogo.json",
    "anchor_manifest": "results/star/star00b_preflight/anchor_manifest.json",
    "shuffled_manifest": "results/star/star00b_preflight/shuffled_label_manifest.json",
    "compute_match_contract": "results/star/star00b_preflight/compute_match_contract.json",
    "training_tasks_csv": "results/star/star00b_preflight/star01_training_tasks.csv",
    "runner_source": "star_eeg/training/real_star_runner.py",
    "persistence_source": "star_eeg/training/persistence.py",
    "approval_lock_source": "star_eeg/training/approval_lock.py",
    "slurm_source": "star_eeg/slurm/star01_train_array.sbatch",
    "submitter_source": "star_eeg/runners/submit_star01a_blind_chain.py",
    "closure_source": "star_eeg/artifacts/close_star01a_finals.py",
    "closure_slurm_source": "star_eeg/slurm/star01a_final_closure.sbatch",
}

APPROVAL_HASH_FIELD = "star01a_approval_manifest_hash"


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(repo_root), text=True
    ).strip()


def cell_name(variant: str, model_seed: int) -> str:
    return f"{variant}_s{int(model_seed)}"


def validate_six_cell_array_environment(
    variant: str,
    model_seed: int,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    environment = dict(os.environ if environ is None else environ)
    required = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_ARRAY_TASK_MAX",
        "STAR_ATTEMPT_ID",
    )
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise PermissionError(f"formal launch is not a six-cell Slurm array: {missing}")
    task_id = int(environment["SLURM_ARRAY_TASK_ID"])
    if (
        int(environment["SLURM_ARRAY_TASK_COUNT"]) != 6
        or int(environment["SLURM_ARRAY_TASK_MIN"]) != 0
        or int(environment["SLURM_ARRAY_TASK_MAX"]) != 5
        or task_id not in range(6)
    ):
        raise PermissionError("formal Slurm array geometry differs from 0-5")
    expected_cell = APPROVED_CELLS[task_id]
    observed_cell = cell_name(variant, model_seed)
    if observed_cell != expected_cell:
        raise PermissionError(
            f"array task {task_id} expects {expected_cell}, not {observed_cell}"
        )
    return {
        "status": "PASS",
        "array_job_id": environment["SLURM_ARRAY_JOB_ID"],
        "array_task_id": task_id,
        "array_task_count": 6,
        "expected_cell": expected_cell,
        "attempt_id": environment["STAR_ATTEMPT_ID"],
    }


def expected_artifact_hashes(repo_root: Path) -> Dict[str, str]:
    output = {}
    for label, relative_path in ARTIFACT_BINDINGS.items():
        path = Path(repo_root) / relative_path
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"approval-bound artifact missing or not regular: {relative_path}")
        output[label] = sha256_file(path)
    return output


def read_training_cells(repo_root: Path) -> Tuple[str, ...]:
    path = Path(repo_root) / ARTIFACT_BINDINGS["training_tasks_csv"]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    cells = tuple(f"{row['variant']}_s{int(row['seed'])}" for row in rows)
    task_ids = tuple(int(row["task_id"]) for row in rows)
    if cells != APPROVED_CELLS or task_ids != tuple(range(6)):
        raise RuntimeError("training task CSV differs from the approved six-cell universe")
    return cells


def build_approval_payload(
    repo_root: Path,
    approved_execution_commit: str,
    approved_attempt_id: str,
    issued_at_utc: Optional[str] = None,
) -> Dict[str, object]:
    repo_root = Path(repo_root).resolve()
    current_commit = _git(repo_root, "rev-parse", "HEAD")
    if current_commit != approved_execution_commit:
        raise RuntimeError("approval commit must equal current HEAD")
    if _git(repo_root, "branch", "--show-current") != STAR_BRANCH:
        raise RuntimeError("approval can only be issued on the frozen STAR branch")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("approval requires a clean tracked worktree")
    if approved_attempt_id != "attempt_01":
        raise RuntimeError("the initial PM gate authorizes attempt_01 only")
    cells = read_training_cells(repo_root)
    core = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "phase": "STAR_01A_BLIND_TRAINING",
        "STAR_01_SCIENTIFIC_TRAINING": "APPROVED",
        "STAR_TARGET_SCORING": "BLOCKED",
        "approved_execution_commit": approved_execution_commit,
        "approved_branch": STAR_BRANCH,
        "approved_cells": list(cells),
        "approved_attempt_id": approved_attempt_id,
        "single_array_launch_only": True,
        "optimizer_steps": STAR01.continuation_optimizer_steps,
        "primary_checkpoint_step": STAR01.primary_checkpoint_step,
        "all_six_submit_together": True,
        "partial_target_scoring_forbidden": True,
        "diagnostic_checkpoint_substitution_forbidden": True,
        "artifact_hash_algorithm": "sha256_file_bytes",
        "artifact_hashes": expected_artifact_hashes(repo_root),
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "pm_gate": "STAR_00C_PASS_CONDITIONAL_STAR_01A_APPROVAL",
    }
    return {**core, APPROVAL_HASH_FIELD: canonical_hash(core)}


def write_approval_lock(
    repo_root: Path,
    output_dir: Path,
    approved_execution_commit: str,
    approved_attempt_id: str,
    issued_at_utc: Optional[str] = None,
) -> Path:
    payload = build_approval_payload(
        repo_root,
        approved_execution_commit=approved_execution_commit,
        approved_attempt_id=approved_attempt_id,
        issued_at_utc=issued_at_utc,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = str(payload[APPROVAL_HASH_FIELD])
    path = output_dir / f"star01a_approval.{manifest_hash}.json"
    atomic_write_json_no_overwrite(path, payload)
    path.chmod(0o444)
    return path


def validate_approval_lock(
    approval_path: Path,
    repo_root: Path,
    variant: Optional[str] = None,
    model_seed: Optional[int] = None,
) -> Dict[str, object]:
    path = Path(approval_path).resolve()
    repo_root = Path(repo_root).resolve()
    if path.is_symlink() or not path.is_file():
        raise PermissionError("approval lock must be a regular file")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise PermissionError("approval lock must be read-only")
    payload = json.loads(path.read_text())
    manifest_hash = payload.get(APPROVAL_HASH_FIELD)
    core = {key: value for key, value in payload.items() if key != APPROVAL_HASH_FIELD}
    if canonical_hash(core) != manifest_hash:
        raise PermissionError("approval manifest canonical hash mismatch")
    if path.name != f"star01a_approval.{manifest_hash}.json":
        raise PermissionError("approval filename does not embed its canonical hash")
    required = {
        "project": PROJECT_NAME,
        "phase": "STAR_01A_BLIND_TRAINING",
        "STAR_01_SCIENTIFIC_TRAINING": "APPROVED",
        "STAR_TARGET_SCORING": "BLOCKED",
        "approved_branch": STAR_BRANCH,
        "approved_attempt_id": "attempt_01",
        "single_array_launch_only": True,
        "optimizer_steps": STAR01.continuation_optimizer_steps,
        "primary_checkpoint_step": STAR01.primary_checkpoint_step,
        "all_six_submit_together": True,
        "partial_target_scoring_forbidden": True,
        "diagnostic_checkpoint_substitution_forbidden": True,
        "artifact_hash_algorithm": "sha256_file_bytes",
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise PermissionError(f"approval field {field} is not frozen")
    if tuple(payload.get("approved_cells", [])) != APPROVED_CELLS:
        raise PermissionError("approval does not bind the exact six cells")
    read_training_cells(repo_root)
    current_commit = _git(repo_root, "rev-parse", "HEAD")
    if payload.get("approved_execution_commit") != current_commit:
        raise PermissionError("approval execution commit differs from HEAD")
    if _git(repo_root, "branch", "--show-current") != STAR_BRANCH:
        raise PermissionError("current branch differs from approval")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=no"):
        raise PermissionError("formal launch refuses a dirty tracked worktree")
    observed_hashes = expected_artifact_hashes(repo_root)
    if payload.get("artifact_hashes") != observed_hashes:
        raise PermissionError("approval-bound artifact hashes differ from execution tree")
    approved_cell = None
    if variant is not None or model_seed is not None:
        if variant is None or model_seed is None:
            raise ValueError("variant and model_seed must be supplied together")
        approved_cell = cell_name(variant, int(model_seed))
        if approved_cell not in APPROVED_CELLS:
            raise PermissionError(f"cell is not approved: {approved_cell}")
    return {
        "status": "PASS",
        "approval_manifest_hash": manifest_hash,
        "approval_file_sha256": sha256_file(path),
        "approved_execution_commit": current_commit,
        "approved_attempt_id": payload["approved_attempt_id"],
        "approved_cell": approved_cell,
        "approved_cells": list(APPROVED_CELLS),
        "target_scoring": "BLOCKED",
        "artifact_hashes": observed_hashes,
    }
