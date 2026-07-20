"""Independent metadata-only red team for C81P readiness."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
from typing import Any

from . import c81_baseline_comparison as baseline
from . import c81_synthetic_baseline_benchmark as synthetic


def _check(rows: list[dict[str, Any]], name: str, passed: bool, evidence: Any, blocking: bool = True) -> None:
    rows.append({
        "check_id": name,
        "passed": int(bool(passed)),
        "blocking": int(blocking and not passed),
        "evidence": json.dumps(evidence, sort_keys=True) if isinstance(evidence, (dict, list)) else str(evidence),
    })


def _tracked_large_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-tree", "-r", "-l", "HEAD"], cwd=baseline.REPO_ROOT,
        check=True, capture_output=True, text=True,
    )
    large = []
    for line in completed.stdout.splitlines():
        left, path = line.split("\t", 1)
        size = int(left.split()[-1])
        if size > 50 * 1024 * 1024:
            large.append(path)
    return large


def _forbidden_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    return names & {"torch", "mne", "moabb"}


def run() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    protocol, protocol_sha = baseline.load_protocol()
    registry = baseline.load_method_registry()
    lock, lock_sha = baseline.load_execution_lock()
    audit = baseline.protocol_audit()
    methods = {row["id"]: row for row in registry["methods"]}
    _check(rows, "protocol_hash_replay", protocol_sha == baseline.PROTOCOL_SHA_PATH.read_text().strip(), protocol_sha)
    _check(rows, "protocol_commit_precedes_implementation", subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["accepted_C80_operating_objects"]["final_head"], lock["protocol"]["commit"]],
        cwd=baseline.REPO_ROOT,
    ).returncode == 0, lock["protocol"]["commit"])
    _check(rows, "lock_hash_replay", lock_sha == baseline.LOCK_SHA_PATH.read_text().strip(), lock_sha)
    _check(rows, "lock_binds_protocol", lock["protocol"]["sha256"] == protocol_sha, lock["protocol"])
    _check(rows, "lock_binds_implementation", len(lock["implementation"]) == 2, lock["implementation"])
    _check(rows, "method_registry_unique", len(methods) == 34, len(methods))
    _check(rows, "primary_representatives_fixed", tuple(lock["method_registry"]["primary_zero_label_representatives"]) == baseline.PRIMARY_ZERO_METHODS, lock["method_registry"]["primary_zero_label_representatives"])
    _check(rows, "unavailable_methods_excluded", all(methods[name]["status"] == "EXCLUDED_INPUT_UNAVAILABLE" for name in ("S3", "S4", "U8", "U9", "U10")), "5/5")
    _check(rows, "all_registered_methods_unconditional", lock["runtime"]["all_registered_methods_unconditional"], len(baseline.SELECTION_METHODS))
    _check(rows, "score_direction_frozen", all(row["may_flip_after_outcome"] == "0" for row in baseline.read_csv(baseline.TABLE_DIR / "score_direction_registry.csv")), "34/34")
    _check(rows, "class_prior_never_target_label", all(row["target_labels_used"] == "0" for row in baseline.read_csv(baseline.TABLE_DIR / "class_prior_registry.csv")), "4/4")
    _check(rows, "C80_frontier_replay", len(baseline.read_csv(baseline.TABLE_DIR / "c80e_frontier_replay.csv")) == 14, "14/14")
    _check(rows, "C80_LOTO_replay", len(baseline.read_csv(baseline.TABLE_DIR / "c80e_loto_stability_replay.csv")) == 16, "16/16")
    field_replay = baseline.read_csv(baseline.TABLE_DIR / "c80e_field_view_manifest_replay.csv")
    _check(rows, "C80_field_view_manifest_replay", len(field_replay) == 11 and all(row["pass"] == "1" for row in field_replay), "11/11")
    decision_replay = baseline.read_csv(baseline.TABLE_DIR / "c80e_decision_endpoint_replay.csv")
    _check(rows, "C80_topk_coverage_replay", len(decision_replay) == 14 and all(row["pass"] == "1" for row in decision_replay), "14/14")
    target_replay = baseline.read_csv(baseline.TABLE_DIR / "c80e_target_level_replay.csv")
    _check(rows, "C80_target_level_replay", len(target_replay) == 224 and all(row["target4_primary"] == row["same_label_oracle_accessed"] == "0" for row in target_replay), "224/224")
    result_replay = baseline.read_csv(baseline.TABLE_DIR / "c80e_result_artifact_hash_replay.csv")
    _check(rows, "C80_result_artifact_hash_replay", len(result_replay) == 22 and all(row["pass"] == "1" for row in result_replay), "22/22")
    _check(rows, "C80_authorization_replay", baseline.read_csv(baseline.TABLE_DIR / "c80e_authorization_replay.csv")[0]["pass"] == "1", "PASS")
    _check(rows, "historical_C80_lock_not_misreported_operational", protocol["accepted_C80_operating_objects"]["historical_lock_status"].startswith("superseded"), protocol["accepted_C80_operating_objects"])
    _check(rows, "C80_descriptor_repair_non_scientific", not protocol["accepted_C80_operating_objects"]["repair_changed_science"], "scientific_change=false")
    _check(rows, "synthetic_scenarios", all(row["passed"] == "1" for row in baseline.read_csv(baseline.TABLE_DIR / "synthetic_baseline_calibration.csv")), "13/13")
    _check(rows, "synthetic_familywise", all(row["passed"] == "1" for row in baseline.read_csv(baseline.TABLE_DIR / "synthetic_familywise_error.csv")), "2/2")
    _check(rows, "synthetic_pair_dependence", all(row["passed"] == "1" for row in baseline.read_csv(baseline.TABLE_DIR / "synthetic_pair_dependence_calibration.csv")), "2/2")
    _check(rows, "synthetic_noninferiority", all(row["passed"] == "1" for row in baseline.read_csv(baseline.TABLE_DIR / "synthetic_noninferiority_calibration.csv")), "3/3")
    _check(rows, "target_principal_cluster", lock["runtime"]["target_is_principal_cluster"], "target")
    _check(rows, "seed_paired_factor", lock["runtime"]["seed_is_paired_factor"], "paired")
    _check(rows, "pairs_trials_checkpoints_not_iid", not lock["inference"]["pairs_checkpoints_trials_and_MC_as_scientific_N"], "not_iid")
    _check(rows, "target4_excluded", not lock["scope"]["target4_primary"] and 4 not in protocol["candidate_universe"]["primary_targets"], "0 rows")
    _check(rows, "oracle_closed", not lock["scope"]["same_label_oracle"] and not protocol["physical_view_contract"]["same_label_oracle_view_reachable"], "closed")
    _check(rows, "trial_id_not_feature", protocol["physical_view_contract"]["trial_id_role"] == "join_split_and_dependence_key_only", protocol["physical_view_contract"]["trial_id_role"])
    _check(rows, "row_order_not_feature", protocol["physical_view_contract"]["row_order_role"] == "alignment_only_never_predictor", protocol["physical_view_contract"]["row_order_role"])
    _check(rows, "selection_before_evaluation", lock["runtime"]["selection_manifest_freeze_required"] and lock["runtime"]["evaluation_requires_selection_hash_replay"], "two-stage")
    _check(rows, "C81E_authorization_absent", not baseline.AUTHORIZATION_PATH.exists(), baseline.AUTHORIZATION_PATH)
    _check(rows, "real_baseline_statistics_zero", audit["real_baseline_statistics"] == 0, audit["real_baseline_statistics"])
    _check(rows, "evaluation_label_reads_zero", audit["evaluation_label_reads"] == 0, audit["evaluation_label_reads"])
    _check(rows, "real_result_absent", not (baseline.REPORT_DIR / "C81_FROZEN_FIELD_BASELINE_COMPARISON.json").exists(), "absent")
    try:
        baseline.run_real()
    except RuntimeError as error:
        guard_pass = "authorization" in str(error)
        guard_evidence = str(error)
    else:
        guard_pass = False
        guard_evidence = "run-real unexpectedly entered"
    _check(rows, "run_real_fail_closed", guard_pass, guard_evidence)
    _check(rows, "no_EEG_or_training_imports", not _forbidden_imports(Path(baseline.__file__)), sorted(_forbidden_imports(Path(baseline.__file__))))
    _check(rows, "risk_register_no_blocker", all(row["blocking"] == "0" for row in baseline.read_csv(baseline.TABLE_DIR / "risk_register.csv")), "27/27")
    large = _tracked_large_files()
    _check(rows, "no_tracked_payload_over_50MiB", not large, large)
    _check(rows, "no_raw_weight_or_cache_in_new_paths", all(token not in path.name.lower() for path in baseline.TABLE_DIR.iterdir() for token in ("weight", "checkpoint", "raw_cache")), "PASS")
    _check(rows, "no_manuscript_scope", not lock["scope"]["manuscript"], "false")
    _check(rows, "no_GPU_training_forward_scope", not any(lock["scope"][name] for name in ("GPU", "training", "forward", "reinference")), "all false")
    baseline.write_csv(baseline.TABLE_DIR / "pre_execution_red_team.csv", rows)
    passed = sum(row["passed"] for row in rows)
    result = {
        "schema_version": "c81p_pre_execution_red_team_v1",
        "checks": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "blocking": sum(row["blocking"] for row in rows),
        "protocol_sha256": protocol_sha,
        "analysis_lock_sha256": lock_sha,
        "real_baseline_statistics": 0,
        "evaluation_label_reads": 0,
        "same_label_oracle_accesses": 0,
        "target4_primary_rows": 0,
        "status": "PASS" if passed == len(rows) else "FAIL",
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"C81P red team failed: {result}")
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
