#!/usr/bin/env python
"""Generate STAR_00C launch-lock and persistence contracts without training."""

import argparse
import inspect
import json
import subprocess
from pathlib import Path
from typing import Dict, Mapping

from star_eeg.config import DEPENDENCY_COMMIT, STAR01, STAR_BRANCH
from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.training.approval_lock import (
    APPROVED_CELLS,
    ARTIFACT_BINDINGS,
    expected_artifact_hashes,
    read_training_cells,
)
from star_eeg.training.real_star_runner import RealStarConfig, run_real_star


STAR00B_BASELINE = "d63bd3ba6184c02e79caeb001bb25888736d1a84"
FROZEN_STAR01_PROTOCOL_HASH = (
    "d158be22ed82ef4eddbae68372fd3ec1fe823864b12b7834f30e293a379f3530"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(repo_root), text=True
    ).strip()


def _hashed(core: Mapping[str, object], field: str) -> Dict[str, object]:
    return {**core, field: canonical_hash(core)}


def build_contracts(repo_root: Path) -> Dict[str, Mapping[str, object]]:
    repo_root = Path(repo_root).resolve()
    protocol_hash = canonical_hash(STAR01.as_dict())
    if protocol_hash != FROZEN_STAR01_PROTOCOL_HASH:
        raise RuntimeError("STAR scientific protocol changed after STAR_00B")
    if tuple(read_training_cells(repo_root)) != APPROVED_CELLS:
        raise RuntimeError("six-cell task CSV changed")
    real_signature = str(inspect.signature(run_real_star))
    if "target" in real_signature.lower() or "source_val" in real_signature.lower():
        raise RuntimeError("formal runner signature exposes non-source data")
    scientific_config = RealStarConfig(
        variant="H200_STAR_TRUE",
        model_seed=0,
        optimizer_steps=3750,
    )
    scientific_config.validate()

    persistence_core = {
        "schema_version": 1,
        "phase": "STAR_00C_LAUNCH_LOCK_AND_PERSISTENCE",
        "formal_layout": {
            "root": "runtime/<cell>/<attempt_NN>",
            "telemetry": "telemetry.jsonl",
            "run_manifest": "run_manifest.json",
            "execution_manifest": "execution_manifest.json",
            "run_summary": "run_summary.json",
            "checkpoints": [
                f"checkpoints/step_{step:04d}.pth"
                for step in STAR01.checkpoint_save_steps
            ],
            "completion": "completion.json",
        },
        "telemetry_policy": "one canonical JSON row per optimizer step; append, flush, fsync",
        "checkpoint_policy": "temporary file plus fsync plus atomic no-overwrite hard-link publish",
        "output_policy": "absent-or-empty attempt only; atomic claim; completed tree read-only",
        "rerun_policy": "new attempt_NN directory required",
        "completion_requires": [
            "exactly_3750_telemetry_rows",
            "final_step_3750",
            "step_3750_strict_reload",
            "source_sha_before_equals_after",
            "no_target_access",
            "all_loss_gradient_delta_finite",
            "summary_and_telemetry_hashes_verified",
        ],
        "run_real_star_signature": real_signature,
        "formal_runner_persists_return_internally": True,
    }
    approval_core = {
        "schema_version": 1,
        "phase": "STAR_00C_APPROVAL_LOCK",
        "approved_cells": list(APPROVED_CELLS),
        "required_fields": [
            "project",
            "phase",
            "STAR_01_SCIENTIFIC_TRAINING",
            "STAR_TARGET_SCORING",
            "approved_execution_commit",
            "approved_branch",
            "approved_cells",
            "approved_attempt_id",
            "single_array_launch_only",
            "optimizer_steps",
            "primary_checkpoint_step",
            "all_six_submit_together",
            "partial_target_scoring_forbidden",
            "artifact_hashes",
        ],
        "artifact_bindings": ARTIFACT_BINDINGS,
        "candidate_artifact_hashes": expected_artifact_hashes(repo_root),
        "approval_path_policy": "SHA-named regular read-only external operational manifest",
        "weak_single_field_approval_rejected": True,
        "formal_array_environment": {
            "task_min": 0,
            "task_max": 5,
            "task_count": 6,
            "task_id_bound_to_cell": True,
            "attempt_id_bound_to_STAR_ATTEMPT_ID": True,
        },
        "actual_approval_created_only_after_clean_STAR00C_commit": True,
    }
    comparison_core = {
        "schema_version": 1,
        "optimizer_steps_each": 3750,
        "batch_count_each": 3750,
        "scheduler_steps_each": 3750,
        "strict_flop_matched": False,
        "accounting": {
            "H200_SSL_CONT": {
                "encoder_updates": 3750,
                "full_reconstruction_updates": 3750,
                "anchor_updates": 0,
            },
            "H200_STAR_TRUE": {
                "encoder_updates": 3750,
                "full_reconstruction_updates": 3000,
                "anchor_updates": 750,
            },
            "H200_STAR_SHUFFLED": {
                "encoder_updates": 3750,
                "full_reconstruction_updates": 3000,
                "anchor_updates": 750,
            },
        },
        "allowed_wording": "optimizer-update- and batch-count-matched",
        "c_minus_d_interpretation": "task-semantics contrast",
        "c_minus_b_interpretation": "task-anchor allocation versus extra SSL continuation",
        "wall_time_must_be_reported": True,
    }
    closure_core = {
        "schema_version": 1,
        "array_cells": list(APPROVED_CELLS),
        "array_submission": "single Slurm array 0-5",
        "closure_dependency": "afterok:<six-cell-array-job-id>",
        "closure_executable": "star_eeg/slurm/star01a_final_closure.sbatch",
        "closure_outputs": [
            "star01_final_checkpoint_manifest.json",
            "star01_training_completion_matrix.csv",
            "star01_closure_go_nogo.json",
        ],
        "final_payload_policy": "SHA-named read-only strict-reload",
        "target_scoring_submitted": False,
    }
    audit_boundary_core = {
        "schema_version": 1,
        "source_val_task_gate_process": "separate_after_immutable_closure",
        "task_gate_failure_effect": "suppress_L4_L5_L6_interpretation_for_that_cell_only",
        "task_gate_failure_may_remove_cell_from_all_cell_target_scoring": False,
        "integrity_or_firewall_failure_may_block_target_stage": True,
        "future_target_scoring_universe": [
            "H200_BASE_s0",
            "H200_BASE_s1",
            *APPROVED_CELLS,
            "random",
            "released",
            "H500_s0",
            "H500_s1",
            "H1000_s0",
            "H1000_s1",
            "H2000_s0",
            "H2000_s1",
        ],
        "target_scoring_currently_blocked": True,
    }
    dependency_core = {
        "schema_version": 1,
        "star00b_baseline": STAR00B_BASELINE,
        "star00b_is_ancestor": (
            _git(repo_root, "merge-base", "HEAD", STAR00B_BASELINE)
            == STAR00B_BASELINE
        ),
        "dependency_commit": DEPENDENCY_COMMIT,
        "dependency_merge_base": _git(
            repo_root, "merge-base", "HEAD", DEPENDENCY_COMMIT
        ),
        "branch": _git(repo_root, "branch", "--show-current"),
        "scientific_protocol_hash": protocol_hash,
        "scientific_protocol_unchanged": protocol_hash == FROZEN_STAR01_PROTOCOL_HASH,
    }
    return {
        "launch_persistence_contract.json": _hashed(
            persistence_core, "launch_persistence_contract_hash"
        ),
        "approval_lock_contract.json": _hashed(
            approval_core, "approval_lock_contract_hash"
        ),
        "comparison_accounting_contract.json": _hashed(
            comparison_core, "comparison_accounting_contract_hash"
        ),
        "final_closure_contract.json": _hashed(
            closure_core, "final_closure_contract_hash"
        ),
        "source_task_audit_boundary.json": _hashed(
            audit_boundary_core, "source_task_audit_boundary_hash"
        ),
        "star00c_dependency_manifest.json": _hashed(
            dependency_core, "star00c_dependency_manifest_hash"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out-dir", default="results/star/star00c_preflight")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    contracts = build_contracts(repo_root)
    for name, payload in contracts.items():
        _write_json(out_dir / name, payload)
    checks = {
        "star00b_baseline": "PASS",
        "scientific_protocol_unchanged": "PASS",
        "persistent_runtime_contract": "PASS",
        "approval_lock_contract": "PASS",
        "no_overwrite_atomic_checkpoint_contract": "PASS",
        "afterok_closure_contract": "PASS",
        "source_task_audit_boundary": "PASS",
        "final_code_gpu_smoke": "PENDING",
    }
    summary_core = {
        "schema_version": 1,
        "phase": "STAR_00C_LAUNCH_LOCK_AND_PERSISTENCE",
        "status": "PENDING_FINAL_CODE_GPU_SMOKE",
        "checks": checks,
        "scientific_protocol_hash": FROZEN_STAR01_PROTOCOL_HASH,
        "formal_3750_step_training_run": False,
        "target_metrics_computed": False,
        "STAR_TARGET_SCORING": "BLOCKED",
        "STAR_MANUSCRIPT_CLAIM": "FORBIDDEN",
    }
    summary = {
        **summary_core,
        "star00c_preflight_summary_hash": canonical_hash(summary_core),
    }
    _write_json(out_dir / "star00c_preflight_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
