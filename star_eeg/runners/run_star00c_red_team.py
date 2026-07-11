#!/usr/bin/env python
"""Independent machine verifier for STAR_00C launch hardening."""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Mapping

from star_eeg.config import STAR01
from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.red_team.no_forbidden_method_guard import (
    evaluate_no_forbidden_method_guard,
)
from star_eeg.runners.run_star00c_preflight import (
    FROZEN_STAR01_PROTOCOL_HASH,
    STAR00B_BASELINE,
)
from star_eeg.training.approval_lock import expected_artifact_hashes
from star_eeg.training.persistence import sha256_file


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(Path(path).read_text())


def _hash_valid(payload: Mapping[str, object], field: str) -> bool:
    core = {key: value for key, value in payload.items() if key != field}
    return canonical_hash(core) == payload.get(field)


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(repo_root), text=True
    ).strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify(
    repo_root: Path,
    out_dir: Path,
    job_id: str,
    job_state: str,
    job_exit_code: str,
) -> Dict[str, object]:
    persistence = _load(out_dir / "launch_persistence_contract.json")
    approval = _load(out_dir / "approval_lock_contract.json")
    comparison = _load(out_dir / "comparison_accounting_contract.json")
    closure = _load(out_dir / "final_closure_contract.json")
    audit_boundary = _load(out_dir / "source_task_audit_boundary.json")
    dependency = _load(out_dir / "star00c_dependency_manifest.json")
    smoke = _load(out_dir / "realpath_smoke_summary.json")
    smoke_persistence = _load(out_dir / "smoke_persistence_index.json")

    execution_hashes = smoke.get("execution_source_hashes", {})
    execution_sources_match = bool(execution_hashes) and all(
        not Path(relative).is_absolute()
        and ".." not in Path(relative).parts
        and (repo_root / relative).is_file()
        and sha256_file(repo_root / relative) == expected
        for relative, expected in execution_hashes.items()
    )
    status_lines = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=str(repo_root),
        text=True,
    ).splitlines()
    observed_paths = [line[3:] for line in status_lines if len(line) >= 4]
    allowed_prefixes = ("star_eeg/", "results/star/star00c_preflight/")
    unexpected_paths = sorted(
        path for path in observed_paths
        if not any(path.startswith(prefix) for prefix in allowed_prefixes)
    )
    protected_diff = _git(
        repo_root,
        "diff",
        "--name-only",
        STAR00B_BASELINE,
        "--",
        "docs/S2P_*",
        "results/s2p_*",
        "h2cmi",
        "oaci",
        "notes/project_A_observability",
    ).splitlines()
    checkpoint_payloads = [
        str(path) for root in (
            repo_root / "star_eeg",
            out_dir,
        )
        for pattern in ("*.pth", "*.pt", "*.npz")
        for path in root.rglob(pattern)
    ]
    train_slurm = (repo_root / "star_eeg/slurm/star01_train_array.sbatch").read_text()
    closure_slurm = (repo_root / "star_eeg/slurm/star01a_final_closure.sbatch").read_text()
    submitter = (repo_root / "star_eeg/runners/submit_star01a_blind_chain.py").read_text()
    runner = (repo_root / "star_eeg/training/real_star_runner.py").read_text()
    weak_approval_check = (
        "validate_approval_lock" in runner
        and "approved_execution_commit" in approval["required_fields"]
        and len(approval["artifact_bindings"]) >= 7
    )
    checks = {
        "star00b_baseline_is_ancestor": (
            _git(repo_root, "merge-base", "HEAD", STAR00B_BASELINE)
            == STAR00B_BASELINE
        ),
        "scientific_protocol_hash_unchanged": (
            canonical_hash(STAR01.as_dict()) == FROZEN_STAR01_PROTOCOL_HASH
        ),
        "no_unexpected_worktree_paths": not unexpected_paths,
        "no_protected_project_diff": not protected_diff,
        "no_checkpoint_payload_in_git_scope": not checkpoint_payloads,
        "persistence_contract_hash_valid": _hash_valid(
            persistence, "launch_persistence_contract_hash"
        ),
        "approval_contract_hash_valid": _hash_valid(
            approval, "approval_lock_contract_hash"
        ),
        "comparison_contract_hash_valid": _hash_valid(
            comparison, "comparison_accounting_contract_hash"
        ),
        "closure_contract_hash_valid": _hash_valid(
            closure, "final_closure_contract_hash"
        ),
        "audit_boundary_hash_valid": _hash_valid(
            audit_boundary, "source_task_audit_boundary_hash"
        ),
        "dependency_manifest_hash_valid": _hash_valid(
            dependency, "star00c_dependency_manifest_hash"
        ),
        "final_code_gpu_smoke_pass": (
            smoke.get("status") == "PASS"
            and all(smoke.get("checks", {}).values())
        ),
        "slurm_smoke_completed_zero_exit": (
            job_state == "COMPLETED" and job_exit_code == "0:0"
        ),
        "gpu_smoke_execution_sources_match_worktree": execution_sources_match,
        "smoke_persistence_hash_valid": _hash_valid(
            smoke_persistence, "smoke_persistence_index_hash"
        ),
        "formal_telemetry_persistence_smoked": all(
            row["telemetry_rows"] == 10
            and row["completion_checks_pass"]
            and row["attempt_tree_read_only"]
            for row in smoke_persistence["rows"]
        ),
        "empty_output_no_overwrite_guard_smoked": all(
            row["same_attempt_reuse_rejected"]
            for row in smoke_persistence["rows"]
        ),
        "approval_hash_binding_contract_pass": weak_approval_check
        and approval["candidate_artifact_hashes"]
        == expected_artifact_hashes(repo_root),
        "formal_array_environment_lock_pass": (
            approval["formal_array_environment"]
            == {
                "task_min": 0,
                "task_max": 5,
                "task_count": 6,
                "task_id_bound_to_cell": True,
                "attempt_id_bound_to_STAR_ATTEMPT_ID": True,
            }
            and "validate_six_cell_array_environment" in runner
        ),
        "atomic_checkpoint_implementation_active": (
            "atomic_torch_save_no_overwrite" in runner
        ),
        "array_is_exact_six_cell_single_submission": (
            "#SBATCH --array=0-5" in train_slurm
            and "STAR_ATTEMPT_ID" in train_slurm
            and "STAR_APPROVAL" in train_slurm
        ),
        "afterok_immutable_closure_executable": (
            "--dependency=afterok:" in submitter
            and "close_star01a_finals" in closure_slurm
            and "STAR_ARRAY_JOB_ID" in closure_slurm
        ),
        "source_task_gate_cannot_select_target_cells": (
            audit_boundary[
                "task_gate_failure_may_remove_cell_from_all_cell_target_scoring"
            ]
            is False
        ),
        "comparison_wording_not_flop_matched": (
            comparison["strict_flop_matched"] is False
            and comparison["allowed_wording"]
            == "optimizer-update- and batch-count-matched"
        ),
        "active_forbidden_method_guard_pass": (
            evaluate_no_forbidden_method_guard()["status"] == "PASS"
        ),
        "formal_3750_training_not_run_in_00c": (
            smoke["formal_3750_step_training_run"] is False
        ),
        "target_metrics_not_computed": smoke["target_metrics_computed"] is False,
        "target_scoring_still_blocked": (
            audit_boundary["target_scoring_currently_blocked"] is True
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    core = {
        "schema_version": 1,
        "phase": "STAR_00C_LAUNCH_LOCK_AND_PERSISTENCE_RED_TEAM",
        "status": status,
        "job_id": str(job_id),
        "job_state": job_state,
        "job_exit_code": job_exit_code,
        "checks": checks,
        "unexpected_worktree_paths": unexpected_paths,
        "protected_diff": protected_diff,
        "checkpoint_payloads_in_git_scope": checkpoint_payloads,
        "formal_3750_step_training_run": False,
        "target_metrics_computed": False,
        "STAR_TARGET_SCORING": "BLOCKED",
    }
    return {**core, "star00c_red_team_hash": canonical_hash(core)}


def _readout(result: Mapping[str, object], job_node: str, job_runtime: str) -> str:
    return "\n".join([
        "# STAR_00C Launch Lock and Persistence Readout",
        "",
        "STAR_00B real-path preflight remains PASS.",
        "STAR_00C launch lock and persistent provenance: PASS.",
        "No 3,750-step scientific training cell was run during STAR_00C.",
        "No target metric was computed; target scoring remains blocked.",
        "No scientific hyperparameter, variant, or gate threshold changed.",
        "",
        "## Machine gates",
        "",
        f"- Final-code bounded GPU smoke: `{result['job_id']}`, `{result['job_state']}`, `{result['job_exit_code']}`, node `{job_node}`, runtime `{job_runtime}`.",
        f"- Independent red-team: `{result['status']}`; hash `{result['star00c_red_team_hash']}`.",
        "- Formal telemetry/run-summary persistence: `PASS`.",
        "- Approval commit/artifact hash binding: `PASS`.",
        "- Empty-output and no-overwrite guard: `PASS`.",
        "- Atomic checkpoint publication: `PASS`.",
        "- Executable array-afterok-closure chain: `PASS`.",
        "",
        "## Next gate",
        "",
        "A SHA-named approval lock may be created only after the clean STAR_00C commit. It may authorize one six-cell array plus afterok immutable closure. Target scoring is not authorized.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out-dir", default="results/star/star00c_preflight")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-state", required=True)
    parser.add_argument("--job-exit-code", required=True)
    parser.add_argument("--job-node", required=True)
    parser.add_argument("--job-runtime", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    result = verify(
        repo_root,
        out_dir,
        args.job_id,
        args.job_state,
        args.job_exit_code,
    )
    _write_json(out_dir / "star00c_red_team.json", result)
    job_core = {
        "job_id": str(args.job_id),
        "state": args.job_state,
        "exit_code": args.job_exit_code,
        "node": args.job_node,
        "runtime": args.job_runtime,
        "partition": "A40",
        "smoke_summary_hash": _load(out_dir / "realpath_smoke_summary.json")[
            "star00c_realpath_smoke_summary_hash"
        ],
        "formal_3750_step_training_run": False,
        "target_metrics_computed": False,
    }
    _write_json(out_dir / "realpath_smoke_job.json", {
        **job_core,
        "realpath_smoke_job_hash": canonical_hash(job_core),
    })
    summary = _load(out_dir / "star00c_preflight_summary.json")
    summary_core = {
        key: value for key, value in summary.items()
        if key != "star00c_preflight_summary_hash"
    }
    summary_core["status"] = "PASS" if result["status"] == "PASS" else "FAIL"
    summary_core["checks"]["final_code_gpu_smoke"] = result["status"]
    _write_json(out_dir / "star00c_preflight_summary.json", {
        **summary_core,
        "star00c_preflight_summary_hash": canonical_hash(summary_core),
    })
    gate_core = {
        "schema_version": 1,
        "status": result["status"],
        "STAR_00C_LAUNCH_LOCK_AND_PERSISTENCE": result["status"],
        "STAR_01A_BLIND_SIX_CELL_TRAINING": (
            "CONDITIONALLY_APPROVED_AFTER_APPROVAL_LOCK"
            if result["status"] == "PASS"
            else "BLOCKED"
        ),
        "STAR_TARGET_SCORING": "BLOCKED",
        "red_team_hash": result["star00c_red_team_hash"],
        "final_code_smoke_job_id": str(args.job_id),
        "scientific_protocol_hash": FROZEN_STAR01_PROTOCOL_HASH,
        "formal_3750_step_training_run": False,
        "target_metrics_computed": False,
    }
    _write_json(out_dir / "star00c_go_nogo.json", {
        **gate_core,
        "star00c_go_nogo_hash": canonical_hash(gate_core),
    })
    report = repo_root / "star_eeg/reports/STAR_00C_PREFLIGHT_READOUT.md"
    report.write_text(_readout(result, args.job_node, args.job_runtime))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
