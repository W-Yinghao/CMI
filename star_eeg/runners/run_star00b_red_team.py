#!/usr/bin/env python
"""Independent STAR_00B artifact and launch-boundary verifier."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Mapping

from star_eeg.config import DEPENDENCY_COMMIT
from star_eeg.data.checkpoint_registry import sha256_file
from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.red_team.no_forbidden_method_guard import evaluate_no_forbidden_method_guard


STAR00A_BASELINE = "26c3fca009d0ecbde7e92e7c759c0256caf1361d"


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(repo_root), text=True).strip()


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text())


def _hash_valid(payload: Mapping[str, object], field: str) -> bool:
    core = {key: value for key, value in payload.items() if key != field}
    return canonical_hash(core) == payload.get(field)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify(
    repo_root: Path,
    out_dir: Path,
    job_id: str,
    job_state: str,
    job_exit_code: str,
) -> Dict[str, object]:
    closure = _load(out_dir / "h200_immutable_manifest.json")
    closure_gate = _load(out_dir / "h200_immutable_go_nogo.json")
    source = _load(out_dir / "faced_source_train_inventory.json")
    anchor = _load(out_dir / "anchor_manifest.json")
    shuffled = _load(out_dir / "shuffled_label_manifest.json")
    anchor_stream = _load(out_dir / "anchor_batch_stream_hashes.json")
    ssl_stream = _load(out_dir / "ssl_batch_stream_hashes.json")
    firewall = _load(out_dir / "source_loader_firewall.json")
    runner_contract = _load(out_dir / "realpath_runner_contract.json")
    compute = _load(out_dir / "compute_match_contract.json")
    blind = _load(out_dir / "blind_evaluation_chain.json")
    smoke = _load(out_dir / "realpath_smoke_summary.json")
    telemetry = _load(out_dir / "realpath_smoke_telemetry.json")
    launch = _load(out_dir / "star00b_launch_manifest.json")
    summary = _load(out_dir / "star00b_preflight_summary.json")
    dependency = _load(out_dir / "star00b_dependency_manifest.json")

    immutable_rows = closure["checkpoints"]
    immutable_paths_ok = all(
        Path(row["launcher_accepted_path"]).is_file()
        and not Path(row["launcher_accepted_path"]).is_symlink()
        and Path(row["launcher_accepted_path"]).name == f"best.{row['sha256']}.pth"
        and Path(row["launcher_accepted_path"]).stat().st_mode & 0o222 == 0
        and sha256_file(Path(row["launcher_accepted_path"])) == row["sha256"]
        for row in immutable_rows
    )
    source_shas_stable = all(
        sha256_file(Path(row["source_checkpoint"])) == row["source_sha256_before"] == row["source_sha256_after"]
        for row in immutable_rows
    )

    status_lines = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=str(repo_root), text=True,
    ).splitlines()
    observed_paths = [line[3:] for line in status_lines if len(line) >= 4]
    allowed_prefixes = ("star_eeg/", "results/star/star00b_preflight/")
    unexpected_paths = sorted(
        path for path in observed_paths if not any(path.startswith(prefix) for prefix in allowed_prefixes)
    )
    protected_diff = _git(
        repo_root,
        "diff",
        "--name-only",
        STAR00A_BASELINE,
        "--",
        "docs/S2P_*",
        "results/s2p_*",
        "h2cmi",
        "oaci",
        "notes/project_A_observability",
    ).splitlines()
    tracked_pth = list(repo_root.glob("star_eeg/**/*.pth")) + list(out_dir.glob("**/*.pth"))

    smoke_checks = smoke.get("checks", {})
    execution_source_hashes = telemetry.get("execution_source_hashes", {})
    execution_source_paths_safe = all(
        not Path(relative_path).is_absolute()
        and ".." not in Path(relative_path).parts
        for relative_path in execution_source_hashes
    )
    execution_source_hashes_match = bool(execution_source_hashes) and execution_source_paths_safe and all(
        (repo_root / relative_path).is_file()
        and sha256_file(repo_root / relative_path) == expected_sha
        for relative_path, expected_sha in execution_source_hashes.items()
    )
    forbidden_guard = evaluate_no_forbidden_method_guard()
    checks = {
        "star00a_baseline_is_ancestor": _git(repo_root, "merge-base", "HEAD", STAR00A_BASELINE) == STAR00A_BASELINE,
        "dependency_merge_base_pinned": _git(repo_root, "merge-base", "HEAD", DEPENDENCY_COMMIT) == DEPENDENCY_COMMIT,
        "no_unexpected_worktree_paths": not unexpected_paths,
        "no_protected_project_diff": not protected_diff,
        "no_checkpoint_payload_in_git_scope": not tracked_pth,
        "closure_manifest_hash_valid": _hash_valid(closure, "h200_immutable_manifest_hash"),
        "closure_gate_hash_valid": _hash_valid(closure_gate, "h200_immutable_go_nogo_hash"),
        "closure_gate_pass": closure_gate.get("status") == "PASS",
        "immutable_paths_sha_named_read_only": immutable_paths_ok,
        "source_h200_shas_unchanged": source_shas_stable,
        "source_inventory_hash_valid": _hash_valid(source, "faced_source_train_inventory_hash"),
        "source_inventory_exact_6720_80": source.get("n_records") == 6720 and source.get("n_subjects") == 80,
        "source_inventory_non_source_reads_zero": source["access_audit"]["source_val_sample_reads"] == 0 and source["access_audit"]["test_sample_reads"] == 0,
        "anchor_hash_valid": _hash_valid(anchor, "anchor_manifest_hash"),
        "shuffled_hash_valid": _hash_valid(shuffled, "shuffled_manifest_hash"),
        "shuffled_non_source_participation_zero": shuffled.get("source_val_participated") is False and shuffled.get("test_participated") is False,
        "anchor_stream_hash_valid": _hash_valid(anchor_stream, "anchor_batch_stream_hashes_hash"),
        "anchor_stream_exact_counts": anchor_stream.get("anchor_batches") == 750 and anchor_stream.get("total_exposures") == 48000,
        "anchor_stream_c_d_and_marginals_pass": all(
            row["c_d_x_stream_identical"]
            and row["subject_exposures_exact_600"]
            and row["true_shuffled_exposure_marginals_equal"]
            for row in anchor_stream["streams"].values()
        ),
        "ssl_stream_hash_valid": _hash_valid(ssl_stream, "ssl_batch_stream_hashes_hash"),
        "ssl_stream_exact_counts": all(
            row["common_batches"] == 3000 and row["replacement_batches"] == 750
            for row in ssl_stream["streams"].values()
        ),
        "source_loader_firewall_pass": firewall.get("status") == "PASS",
        "source_loader_firewall_hash_valid": _hash_valid(
            firewall, "source_loader_firewall_hash"
        ),
        "actual_runner_contract_pass": runner_contract.get("status") == "PASS",
        "actual_runner_contract_hash_valid": _hash_valid(
            runner_contract, "realpath_runner_contract_hash"
        ),
        "active_method_and_import_guard_pass": forbidden_guard.get("status") == "PASS",
        "compute_match_pass": _hash_valid(compute, "compute_match_hash")
        and all(compute.get("checks", {}).values()),
        "real_gpu_smoke_pass": smoke.get("status") == "PASS" and all(smoke_checks.values()),
        "gpu_smoke_execution_sources_match_worktree": execution_source_hashes_match
        and smoke.get("execution_source_hashes") == execution_source_hashes,
        "slurm_smoke_completed_zero_exit": job_state == "COMPLETED" and job_exit_code == "0:0",
        "smoke_telemetry_hash_valid": _hash_valid(telemetry, "realpath_smoke_telemetry_hash"),
        "formal_training_not_run": smoke.get("formal_3750_step_training_run") is False and summary.get("formal_3750_step_training_run") is False,
        "target_metrics_not_computed": smoke.get("target_metrics_computed") is False and summary.get("target_metrics_computed") is False,
        "blind_chain_hash_valid": _hash_valid(blind, "blind_evaluation_chain_hash"),
        "blind_chain_training_before_scoring": blind["stages"][3]["dependency"] == "afterok_source_only_audit",
        "blind_chain_source_val_gate_is_separate_and_target_blind":
        blind["stages"][2].get("separate_from_training_process") is True
        and blind["stages"][2].get("source_val_labels_for_task_gate_only") is True
        and blind["stages"][2].get("target_test_samples_or_labels_read") is False,
        "blind_chain_not_submitted": blind.get("target_scoring_submitted") is False,
        "star01_execution_still_blocked": launch.get("star01_runner_execution_allowed") is False and not launch.get("approved_new_training_cells"),
        "star01_approval_manifest_absent": not (repo_root / "results/star/star01_approval.json").exists(),
        "target_scoring_still_blocked": launch.get("target_scoring_allowed") is False,
        "star00b_summary_pass": summary.get("STAR_00B_REAL_PATH_PREFLIGHT") == "PASS",
        "dependency_artifact_hash_valid": _hash_valid(
            dependency, "star00b_dependency_hash"
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    core = {
        "phase": "STAR_00B_independent_red_team",
        "status": status,
        "slurm_smoke_job_id": str(job_id),
        "slurm_smoke_job_state": job_state,
        "slurm_smoke_exit_code": job_exit_code,
        "checks": checks,
        "unexpected_worktree_paths": unexpected_paths,
        "protected_diff": protected_diff,
        "tracked_checkpoint_payloads": [str(path) for path in tracked_pth],
        "STAR_01_SCIENTIFIC_TRAINING": "BLOCKED_PENDING_PM_REVIEW",
        "STAR_TARGET_SCORING": "BLOCKED",
        "formal_3750_step_training_run": False,
        "target_metrics_computed": False,
    }
    return {**core, "star00b_red_team_hash": canonical_hash(core)}


def _readout(result: Mapping[str, object], out_dir: Path) -> str:
    summary = _load(out_dir / "star00b_preflight_summary.json")
    smoke = _load(out_dir / "realpath_smoke_summary.json")
    closure = _load(out_dir / "h200_immutable_manifest.json")
    source = _load(out_dir / "faced_source_train_inventory.json")
    anchor_stream = _load(out_dir / "anchor_batch_stream_hashes.json")
    lines = [
        "# STAR_00B Real-Path Preflight Readout",
        "",
        "This is a real-path launch preflight with bounded ten-step CUDA smoke updates.",
        "No 3,750-step STAR scientific training cell was run.",
        "No FACED source_val or target sample was deserialized.",
        "No target metric was computed.",
        "STAR_01 scientific training remains blocked pending PM review.",
        "S2P Phase B, H2CMI, and OACI remain independent and unchanged.",
        "",
        "## Gate state",
        "",
        f"- `STAR_00_PROJECT_CHARTER`: `{summary['STAR_00_PROJECT_CHARTER']}`",
        f"- `STAR_00A_DESIGN_AND_RED_TEAM_PREFLIGHT`: `{summary['STAR_00A_DESIGN_AND_RED_TEAM_PREFLIGHT']}`",
        f"- `STAR_H200_ARTIFACT_SUPPLY`: `{summary['STAR_H200_ARTIFACT_SUPPLY']}`",
        f"- `STAR_00B_REAL_PATH_PREFLIGHT`: `{summary['STAR_00B_REAL_PATH_PREFLIGHT']}`",
        f"- `STAR_01_SCIENTIFIC_TRAINING`: `{summary['STAR_01_SCIENTIFIC_TRAINING']}`",
        f"- `STAR_TARGET_SCORING`: `{summary['STAR_TARGET_SCORING']}`",
        f"- `STAR_MANUSCRIPT_CLAIM`: `{summary['STAR_MANUSCRIPT_CLAIM']}`",
        "",
        "## Load-bearing results",
        "",
        f"- H200 immutable closure: 2/2 SHA-named read-only strict-load payloads; manifest `{closure['h200_immutable_manifest_hash']}`.",
        f"- FACED source-only inventory: {source['n_records']} records, {source['n_subjects']} subjects; source_val/test reads both zero.",
        f"- Anchor stream: {anchor_stream['total_exposures']} exposures, 600 per subject, C/D X and exposure marginals matched.",
        f"- CUDA smoke job: `{result['slurm_smoke_job_id']}`; status `{smoke['status']}`; GPU {', '.join(smoke['gpu_names'])}.",
        f"- Independent red-team: `{result['status']}`; hash `{result['star00b_red_team_hash']}`.",
        "",
        "The bounded smoke updated real CBraMod parameters on real H200 Route-B TUEG and FACED source_train batches only. Its telemetry is integrity evidence, not a checkpoint-selection or scientific endpoint.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out-dir", default="results/star/star00b_preflight")
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
        repo_root, out_dir, args.job_id, args.job_state, args.job_exit_code
    )
    smoke_job_core = {
        "job_id": str(args.job_id),
        "state": args.job_state,
        "exit_code": args.job_exit_code,
        "node": args.job_node,
        "runtime": args.job_runtime,
        "partition": "A40",
        "submission_head": STAR00A_BASELINE,
        "stdout": str(out_dir / f"logs/realpath-smoke-{args.job_id}.out"),
        "stderr": str(out_dir / f"logs/realpath-smoke-{args.job_id}.err"),
        "smoke_summary_hash": _load(out_dir / "realpath_smoke_summary.json")["realpath_smoke_summary_hash"],
        "formal_3750_step_training_run": False,
        "target_metrics_computed": False,
    }
    _write_json(out_dir / "realpath_smoke_job.json", {
        **smoke_job_core,
        "realpath_smoke_job_hash": canonical_hash(smoke_job_core),
    })
    _write_json(out_dir / "star00b_red_team.json", result)
    readout = repo_root / "star_eeg/reports/STAR_00B_PREFLIGHT_READOUT.md"
    readout.write_text(_readout(result, out_dir))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
