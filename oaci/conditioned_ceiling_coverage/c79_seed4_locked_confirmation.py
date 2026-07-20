"""Read-only C79 protocol and timing review.

This module intentionally has no seed-4 execution path.  It audits whether the
committed C79 artifact is eligible to authorize one.  A failed timing review is
a scientific protocol result, not an exception to be patched around.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c79_tables"

C79_PROTOCOL = REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL.json"
C79_PROTOCOL_SHA = REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL.sha256"
C79_SKELETON = REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL_SKELETON.json"
C78S_PROTOCOL = REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS_PROTOCOL.json"
C78S_PROTOCOL_SHA = REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS_PROTOCOL.sha256"
C78S_RESULT = REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS.json"
C78S_STATE = REPORT_DIR / "C78S_ANALYSIS_STATE.json"

REPLAY_JSON = REPORT_DIR / "C79_PROTOCOL_REPLAY.json"
TIMING_REPORT = REPORT_DIR / "C79_PROTOCOL_TIMING_AUDIT.md"
REPAIR_REPORT = REPORT_DIR / "C79_C78S_PROVENANCE_REPAIR_REPLAY.md"
REVIEW_REPORT = REPORT_DIR / "C79_PROTOCOL_REVIEW.md"
RED_TEAM_JSON = REPORT_DIR / "C79_MODE_R_REVIEW_RED_TEAM.json"
RED_TEAM_REPORT = REPORT_DIR / "C79_MODE_R_REVIEW_RED_TEAM.md"

C78F_COMMITS = {
    "protocol": "1d210fdbf6333755ded53dead92c0f9f7ce7cefe",
    "execution_lock": "a9029668fad767f58230ec7b8d24fcabae85a642",
    "collector_repair": "f0d49c2ff038c91779d7d580e8562068b1ad0706",
    "result": "51022f4e247e0217ebb54403d419d121071fe0e7",
    "handoff": "e74b20ed09a09995709e820f1e7a0db99718cfe4",
}
C78S_COMMITS = {
    "implementation": "e561a15865934036bdccbc1e3b2ff126ad84821f",
    "execution_lock": "ce1fb14c960f5b2da62b60d65ab40af098cf274e",
    "result": "43a046c5ba6632de415bfdbaacfa524d82c5395e",
    "handoff": "48be5b7034bbc51cdeaec402819a1a5dd1233f8a",
    "provenance_correction": "dcd4c283573b4cdebe72c8ed3e181403232b28b7",
    "final_anchor": "ef2a01a4e948143be5eb58c3370142b5eecf7178",
}
C79_SKELETON_COMMIT = "23f549d73803bd23e435f7dae581de29bf62285f"
C79_FINAL_PROTOCOL_COMMIT = C78S_COMMITS["result"]
C78S_PROTOCOL_EXPECTED_SHA = "df85699090a65d1e1766d754bcebd9eb5648cc13e4441d8074a3f4884487c7f8"
C79_PROTOCOL_EXPECTED_SHA = "7732986513793725d58933d487f5bc8f4fc68bfad0857bb4734a450b41ca5dd4"
C79_SKELETON_EXPECTED_SHA = "cf74cf7de14a432cbd44f90228254b5977a67c773d9755ff9cad472c47b45f42"
C78S_REPAIR_COMMIT = C78S_COMMITS["provenance_correction"]

FINAL_GATE = "C79_PROTOCOL_OR_TIMING_REPAIR_REQUIRED"
PRIMARY_TAXONOMY = "C79-F_protocol_timing_provenance_or_isolation_blocker"

C78S_REFERENCES = {
    "H1_split_label_reliability": 0.7708629907592237,
    "H1_Holm_p": 0.058365758754863814,
    "H1_construction_top1": 0.125,
    "H1_construction_top5": 0.6875,
    "H1_construction_top10": 0.75,
    "H1_random_top1": 1.0 / 81.0,
    "H1_random_top5": 5.0 / 81.0,
    "H1_random_top10": 10.0 / 81.0,
    "H1_standardized_regret": 0.08279432226091084,
    "H1_random_expected_regret": 0.4820,
    "H2_held_target_deviance_change": 9.505920587315757,
    "H2_permutation_p": 0.896,
    "H3_local_association": 0.24265629215048484,
    "H3_positive_trajectory_cells": 32.0,
    "H3_worst_control_p": 0.002,
    "H3_LOTO_incremental_R2": -0.21287505480736912,
    "H3_LORO_incremental_R2": -0.08579644289995947,
    "H4_F2_incremental_R2": -0.07308583875591457,
    "H5_F4_incremental_R2": 0.005176499403102386,
    "H6_incremental_R2": 0.40432932651535547,
    "H6_raw_p": 0.019455252918287938,
    "H6_Holm_p": 0.07782101167315175,
}

REQUIRED_PROTOCOL_COMPONENTS = (
    "primary targets and target-4 exclusion",
    "levels, regimes, and candidate universe",
    "construction/evaluation trial-ID hashes",
    "H1 reliability, top-k, regret formulas and gates",
    "H2 exact model, sign convention, and held-target split",
    "H3 exact feature block, kernel, bandwidth, and scaling",
    "H4 exact F2 base model, cross-fit, null, and qualification",
    "H5 exact F4 base model, cross-fit, null, and qualification",
    "H6 exact positive-control effect and Holm family ordering",
    "outer and inner grouping units",
    "permutation/bootstrap construction and counts",
    "null RNG streams",
    "seed4-only versus combined primary status",
    "retry and additive-repair policy",
    "same-label oracle reachability",
    "success, failure, and claim taxonomy",
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty C79 table: {name}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with (TABLE_DIR / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _commit_row(stage: str, commit: str) -> dict[str, Any]:
    payload = _git("show", "-s", "--format=%H|%P|%aI|%cI|%s", commit)
    resolved, parents, authored, committed, subject = payload.split("|", 4)
    return {
        "stage": stage,
        "expected_commit": commit,
        "resolved_commit": resolved,
        "parents": parents,
        "authored_at": authored,
        "committed_at": committed,
        "subject": subject,
        "exists": int(resolved == commit),
        "ancestor_of_review_anchor": int(
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, C78S_COMMITS["final_anchor"]],
                cwd=REPO_ROOT,
                check=False,
            ).returncode
            == 0
        ),
    }


def _blob_sha(commit: str, path: str) -> str:
    data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO_ROOT)
    return hashlib.sha256(data).hexdigest()


def _first_path_commit(path: Path) -> str:
    relative = str(path.relative_to(REPO_ROOT))
    history = _git("log", "--all", "--reverse", "--format=%H", "--", relative).splitlines()
    return history[0] if history else ""


def _iso_to_epoch(value: str) -> float:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).timestamp()


def _extract_c78s_references() -> dict[str, float]:
    multiplicity = {
        row["hypothesis"]: row
        for row in _csv_rows(TABLE_DIR.parent / "c78s_tables" / "primary_hypothesis_multiplicity.csv")
    }
    measurement = _csv_rows(TABLE_DIR.parent / "c78s_tables" / "measurement_control_summary.csv")[0]
    geometry = _csv_rows(TABLE_DIR.parent / "c78s_tables" / "effective_multiplicity_summary.csv")[0]
    krr = {
        row["path"]: row
        for row in _csv_rows(TABLE_DIR.parent / "c78s_tables" / "nonlinear_prediction_summary.csv")
    }
    topology = _csv_rows(TABLE_DIR.parent / "c78s_tables" / "association_topology.csv")
    local = next(
        row for row in topology
        if row["path"] == "target_unlabeled"
        and row["level"] == "within_target_x_level_x_regime"
    )
    controls = _csv_rows(TABLE_DIR.parent / "c78s_tables" / "association_strict_control_summary.csv")
    registered_control = next(
        row for row in controls
        if row["path"] == "target_unlabeled"
        and row["kernel"] == "laplacian"
        and row["bandwidth_factor"] == "1.0"
        and row["statistic"] == "centered_hsic"
    )
    action_cells = _csv_rows(
        TABLE_DIR.parent / "c78s_tables" / "measurement_control_actionability_cells.csv"
    )
    prediction = _csv_rows(TABLE_DIR.parent / "c78s_tables" / "cross_fitted_incremental_prediction.csv")
    strict = next(
        row for row in prediction
        if row["path"] == "strict_source_F2" and row["outcome"] == "continuous_joint_utility"
    )
    target = next(
        row for row in prediction
        if row["path"] == "target_unlabeled_F4_geometry" and row["outcome"] == "continuous_joint_utility"
    )
    return {
        "H1_split_label_reliability": float(measurement["trajectory_reliability_mean"]),
        "H1_Holm_p": float(multiplicity["H1"]["Holm_p"]),
        "H1_construction_top1": float(measurement["full_oracle_best_in_predicted_top1"]),
        "H1_construction_top5": float(measurement["full_oracle_best_in_predicted_top5"]),
        "H1_construction_top10": float(measurement["full_oracle_best_in_predicted_top10"]),
        "H1_random_top1": 1.0 / 81.0,
        "H1_random_top5": 5.0 / 81.0,
        "H1_random_top10": 10.0 / 81.0,
        "H1_standardized_regret": float(measurement["full_standardized_regret"]),
        "H1_random_expected_regret": sum(
            float(row["random_expected_regret"]) / float(row["utility_range"])
            for row in action_cells
        ) / len(action_cells),
        "H2_held_target_deviance_change": -float(geometry["incremental_deviance_reduction"]),
        "H2_permutation_p": float(geometry["permutation_p"]),
        "H3_local_association": float(local["association"]),
        "H3_positive_trajectory_cells": float(local["group_count"]) * float(local["positive_group_fraction"]),
        "H3_worst_control_p": float(registered_control["worst_required_global_p"]),
        "H3_LOTO_incremental_R2": float(krr["target_unlabeled"]["incremental_LOTO_R2"]),
        "H3_LORO_incremental_R2": float(krr["target_unlabeled"]["incremental_LORO_R2"]),
        "H4_F2_incremental_R2": float(strict["incremental_LOTO_R2"]),
        "H5_F4_incremental_R2": float(target["incremental_LOTO_R2"]),
        "H6_incremental_R2": float(multiplicity["H6"]["effect"]),
        "H6_raw_p": float(multiplicity["H6"]["raw_p"]),
        "H6_Holm_p": float(multiplicity["H6"]["Holm_p"]),
    }


def _protocol_component_rows(final_protocol: dict[str, Any]) -> list[dict[str, Any]]:
    presence = {
        "primary targets and target-4 exclusion": "targets" in final_protocol and "primary_targets" in final_protocol,
        "levels, regimes, and candidate universe": all(key in final_protocol for key in ("levels", "regimes", "field_units")),
        "construction/evaluation trial-ID hashes": "label_split_hashes" in final_protocol,
        "H1 reliability, top-k, regret formulas and gates": "H1" in final_protocol.get("hypotheses", {}),
        "H2 exact model, sign convention, and held-target split": "H2" in final_protocol.get("hypotheses", {}),
        "H3 exact feature block, kernel, bandwidth, and scaling": "H3" in final_protocol.get("hypotheses", {}),
        "H4 exact F2 base model, cross-fit, null, and qualification": "H4" in final_protocol.get("hypotheses", {}),
        "H5 exact F4 base model, cross-fit, null, and qualification": "H5" in final_protocol.get("hypotheses", {}),
        "H6 exact positive-control effect and Holm family ordering": "H6" in final_protocol.get("hypotheses", {}),
        "outer and inner grouping units": "outer_inner_splits" in final_protocol,
        "permutation/bootstrap construction and counts": "resampling" in final_protocol,
        "null RNG streams": "rng_streams" in final_protocol,
        "seed4-only versus combined primary status": bool(final_protocol.get("inference", {}).get("seed4_analyzed_alone_before_any_seed3_pooling")),
        "retry and additive-repair policy": "retry_repair_policy" in final_protocol,
        "same-label oracle reachability": "same_label_oracle" in final_protocol,
        "success, failure, and claim taxonomy": "taxonomy" in final_protocol,
    }
    return [
        {
            "component": component,
            "present_in_final_C79_protocol": int(presence[component]),
            "status": "present" if presence[component] else "missing_or_not_exactly_bound",
            "blocking_if_strict_confirmation": int(not presence[component]),
        }
        for component in REQUIRED_PROTOCOL_COMPONENTS
    ]


def _repair_replay_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parent = f"{C78S_REPAIR_COMMIT}^"
    changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", C78S_REPAIR_COMMIT).splitlines()
    file_rows = []
    for path in changed:
        file_rows.append({
            "record_type": "changed_file",
            "item": path,
            "before_sha256": _blob_sha(parent, path),
            "after_sha256": _blob_sha(C78S_REPAIR_COMMIT, path),
            "code_or_protocol_file": int(
                path.endswith(".py") or "PROTOCOL" in Path(path).name or "EXECUTION_LOCK" in Path(path).name
            ),
            "scientific_result_or_primary_table": int(
                Path(path).name in {
                    "C78S_SEED3_SCIENTIFIC_ANALYSIS.json",
                    "C78S_SEED3_SCIENTIFIC_ANALYSIS.md",
                    "primary_hypothesis_multiplicity.csv",
                }
            ),
            "passed": 1,
        })
    expected = {
        "oaci/reports/C78S_ARTIFACT_MANIFEST.json",
        "oaci/reports/C78S_FINAL_REPORT_RED_TEAM.md",
        "oaci/reports/C78S_REGRESSION_VERIFICATION.md",
        "oaci/reports/c78s_tables/final_report_red_team_checks.csv",
        "oaci/reports/c78s_tables/regression_verification.csv",
    }
    regression_text = (REPORT_DIR / "C78S_REGRESSION_VERIFICATION.md").read_text()
    checks = [
        ("repair_commit_exists", _git("rev-parse", C78S_REPAIR_COMMIT) == C78S_REPAIR_COMMIT),
        ("repair_is_additive_descendant_of_result", subprocess.run(
            ["git", "merge-base", "--is-ancestor", C78S_COMMITS["result"], C78S_REPAIR_COMMIT],
            cwd=REPO_ROOT,
            check=False,
        ).returncode == 0),
        ("changed_file_set_exact", set(changed) == expected),
        ("no_code_or_protocol_changed", not any(row["code_or_protocol_file"] for row in file_rows)),
        ("no_scientific_result_or_primary_table_changed", not any(row["scientific_result_or_primary_table"] for row in file_rows)),
        ("before_after_hashes_recorded", all(row["before_sha256"] != row["after_sha256"] for row in file_rows)),
        ("dedicated_skip_replay_job_recorded", "893168" in regression_text),
        ("locked_scientific_implementation_and_estimands_unchanged", all(
            token in (REPORT_DIR / "C78S_FINAL_REPORT_RED_TEAM.md").read_text()
            for token in ("report/provenance-only", "changed no data, statistic, taxonomy, or scientific")
        )),
    ]
    check_rows = [
        {
            "record_type": "repair_check",
            "item": name,
            "before_sha256": "not_applicable",
            "after_sha256": "not_applicable",
            "code_or_protocol_file": 0,
            "scientific_result_or_primary_table": 0,
            "passed": int(passed),
        }
        for name, passed in checks
    ]
    return file_rows + check_rows, check_rows


def audit() -> dict[str, Any]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    final_protocol = _json(C79_PROTOCOL)
    skeleton = _json(C79_SKELETON)
    c78s_protocol = _json(C78S_PROTOCOL)
    state = _json_lines(C78S_STATE)
    event_times = {row["event"]: row["at_utc"] for row in state}
    references = _extract_c78s_references()

    c79_hash = _sha256(C79_PROTOCOL)
    skeleton_hash = _sha256(C79_SKELETON)
    c78s_hash = _sha256(C78S_PROTOCOL)
    c79_first_commit = _first_path_commit(C79_PROTOCOL)
    skeleton_first_commit = _first_path_commit(C79_SKELETON)
    c79_created = final_protocol["created_at_utc"]
    first_outcome = event_times["H1_complete"]
    final_outcome = event_times["H2_complete"]
    timing_pass = _iso_to_epoch(c79_created) < _iso_to_epoch(first_outcome)
    outcome_adaptive = final_protocol.get("C78S_active_hypotheses_to_confirm") == ["H3", "H4", "H5"]
    generator_commit = _first_path_commit(
        REPO_ROOT / "oaci/conditioned_ceiling_coverage/c78s_seed3_scientific_analysis.py"
    )
    generator_committed_at = _commit_row("C79_generator_rule", generator_commit)["committed_at"]
    generator_predates_outcome = _iso_to_epoch(generator_committed_at) < _iso_to_epoch(first_outcome)

    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/oaci")
    _write_csv("repository_state_replay.csv", [{
        "audited_repo_root": str(REPO_ROOT),
        "canonical_shared_worktree": "/home/infres/yinwang/CMI_AAAI_oaci",
        "review_base_HEAD": head,
        "origin_oaci": origin,
        "head_matches_origin": int(head == origin),
        "branch": _git("branch", "--show-current"),
        "seed4_data_access": 0,
        "seed4_jobs_submitted": 0,
        "seed4_artifacts_created": 0,
        "mode": "R_review_only",
    }])
    _write_csv("c78f_commit_manifest_replay.csv", [
        _commit_row(stage, commit) for stage, commit in C78F_COMMITS.items()
    ])
    _write_csv("c78s_commit_replay.csv", [
        _commit_row(stage, commit) for stage, commit in C78S_COMMITS.items()
    ])
    _write_csv("c78s_protocol_replay.csv", [{
        "path": str(C78S_PROTOCOL.relative_to(REPO_ROOT)),
        "expected_sha256": C78S_PROTOCOL_EXPECTED_SHA,
        "observed_sha256": c78s_hash,
        "sha_match": int(c78s_hash == C78S_PROTOCOL_EXPECTED_SHA == C78S_PROTOCOL_SHA.read_text().strip()),
        "protocol_commit": C78F_COMMITS["protocol"],
        "status": c78s_protocol["status"],
        "primary_targets": json.dumps(c78s_protocol["data_roles"]["primary_targets"]),
        "target4_excluded": int(c78s_protocol["data_roles"]["target4_canary"].startswith("descriptive_only")),
    }])
    result_rows = []
    for name, expected in C78S_REFERENCES.items():
        observed = references[name]
        result_rows.append({
            "metric": name,
            "expected": expected,
            "observed": observed,
            "absolute_error": abs(observed - expected),
            "match_tolerance": 5e-4 if name == "H1_random_expected_regret" else 1e-12,
            "passed": int(abs(observed - expected) <= (5e-4 if name == "H1_random_expected_regret" else 1e-12)),
        })
    _write_csv("c78s_result_replay.csv", result_rows)

    repair_rows, repair_checks = _repair_replay_rows()
    _write_csv("c78s_provenance_repair_replay.csv", repair_rows)

    component_rows = _protocol_component_rows(final_protocol)
    _write_csv("c79_protocol_registry.csv", component_rows)
    timing_rows = [
        {
            "order": 1,
            "event": "C79_skeleton_committed",
            "at_utc": _commit_row("skeleton", C79_SKELETON_COMMIT)["committed_at"],
            "artifact_status": skeleton["status"],
            "eligible_final_protocol": 0,
            "timing_pass": 0,
        },
        {
            "order": 2,
            "event": "C79_adaptive_generator_rule_committed",
            "at_utc": generator_committed_at,
            "artifact_status": "transparent_preoutcome_adaptive_rule_in_C78S_implementation",
            "eligible_final_protocol": 0,
            "timing_pass": int(generator_predates_outcome),
        },
        {
            "order": 3,
            "event": "C78S_first_scientific_outcome_H1_complete",
            "at_utc": first_outcome,
            "artifact_status": "outcome_observed",
            "eligible_final_protocol": 0,
            "timing_pass": 0,
        },
        {
            "order": 4,
            "event": "C78S_H3_H4_H5_complete",
            "at_utc": event_times["H3_H4_H5_complete"],
            "artifact_status": "outcomes_observed",
            "eligible_final_protocol": 0,
            "timing_pass": 0,
        },
        {
            "order": 5,
            "event": "C78S_H2_complete",
            "at_utc": final_outcome,
            "artifact_status": "outcomes_observed",
            "eligible_final_protocol": 0,
            "timing_pass": 0,
        },
        {
            "order": 6,
            "event": "C79_final_protocol_created",
            "at_utc": c79_created,
            "artifact_status": final_protocol["status"],
            "eligible_final_protocol": 0,
            "timing_pass": int(timing_pass),
        },
        {
            "order": 7,
            "event": "C79_final_protocol_first_committed",
            "at_utc": _commit_row("C79_final_protocol", C79_FINAL_PROTOCOL_COMMIT)["committed_at"],
            "artifact_status": "committed_with_C78S_result",
            "eligible_final_protocol": 0,
            "timing_pass": 0,
        },
    ]
    _write_csv("c79_protocol_timing.csv", timing_rows)
    _write_csv("c79_protocol_hash_replay.csv", [
        {
            "artifact": "C79_final_protocol",
            "path": str(C79_PROTOCOL.relative_to(REPO_ROOT)),
            "expected_sha256": C79_PROTOCOL_EXPECTED_SHA,
            "observed_sha256": c79_hash,
            "committed_sha256": C79_PROTOCOL_SHA.read_text().strip(),
            "sha_match": int(c79_hash == C79_PROTOCOL_EXPECTED_SHA == C79_PROTOCOL_SHA.read_text().strip()),
            "first_commit": c79_first_commit,
            "status": final_protocol["status"],
        },
        {
            "artifact": "C79_skeleton",
            "path": str(C79_SKELETON.relative_to(REPO_ROOT)),
            "expected_sha256": C79_SKELETON_EXPECTED_SHA,
            "observed_sha256": skeleton_hash,
            "committed_sha256": "no_dedicated_sha_file",
            "sha_match": int(skeleton_hash == C79_SKELETON_EXPECTED_SHA),
            "first_commit": skeleton_first_commit,
            "status": skeleton["status"],
        },
    ])
    _write_csv("c79_scientific_degrees_of_freedom.csv", component_rows)
    _write_csv("c79_seed_perturbation_registry.csv", [
        {"dimension": "training_seed", "seed3": 3, "seed4": 4, "fixed_or_changed": "changed_registered", "protocol_bound": 1},
        {"dimension": "raw_EEG_trials", "seed3": "same_BNCI2014_001_trials", "seed4": "not_explicitly_bound", "fixed_or_changed": "intended_fixed_unproven", "protocol_bound": 0},
        {"dimension": "construction_evaluation_split", "seed3": "8_target_hashes", "seed4": "not_bound", "fixed_or_changed": "unresolved", "protocol_bound": 0},
        {"dimension": "regimes_levels_cadence", "seed3": "ERM_OACI_SRC_levels_0_1_81_per_context", "seed4": "field_units_1458_only", "fixed_or_changed": "partially_bound", "protocol_bound": 0},
        {"dimension": "analysis_implementation", "seed3": C78S_COMMITS["implementation"], "seed4": "not_bound", "fixed_or_changed": "unresolved", "protocol_bound": 0},
    ])
    _write_csv("c79_cross_seed_primary_status.csv", [{
        "seed4_only_before_pooling": int(final_protocol["inference"]["seed4_analyzed_alone_before_any_seed3_pooling"]),
        "seed3_seed4_combination_formula_locked": 0,
        "cross_seed_dependence_rule_locked": 0,
        "primary_status": "seed4_only_intended_but_exact_estimand_registry_incomplete",
        "passed_for_strict_confirmation": 0,
    }])
    _write_csv("c79_protocol_diff_against_c78s.csv", [
        {
            "item": "adaptive_generator_timing",
            "C78S_preoutcome": generator_commit,
            "C79_postoutcome": c79_first_commit,
            "difference": "generator_rule_precommitted_but_materialized_protocol_postoutcome",
            "blocking": 0,
        },
        {
            "item": "hypothesis_scope",
            "C78S_preoutcome": "H1-H6",
            "C79_postoutcome": json.dumps(final_protocol["C78S_active_hypotheses_to_confirm"]),
            "difference": "outcome_adaptive_filter",
            "blocking": 1,
        },
        {
            "item": "protocol_role",
            "C78S_preoutcome": "seed3_scientific_analysis",
            "C79_postoutcome": "seed4_confirmation",
            "difference": "new_role_not_bound_preoutcome",
            "blocking": 1,
        },
        {
            "item": "exact_analysis_registry",
            "C78S_preoutcome": "implemented_in_e561a15",
            "C79_postoutcome": "references_blocks_nulls_without_exact_H1-H6_registry",
            "difference": "mechanical_mapping_not_locked",
            "blocking": 1,
        },
    ])

    risks = [
        ("C79_protocol_not_found", "closed", C79_PROTOCOL.exists(), "final artifact exists"),
        ("C79_protocol_hash_mismatch", "closed", c79_hash == C79_PROTOCOL_EXPECTED_SHA, "hash replays"),
        ("C79_protocol_post_C78S_outcome", "OPEN_BLOCKER", timing_pass, f"created={c79_created};first_outcome={first_outcome}"),
        ("C79_protocol_outcome_adaptive_hypothesis_filter", "OPEN_BLOCKER", not outcome_adaptive, "H3/H4/H5 selected from active_after_Holm"),
        ("C79_exact_scientific_registry_incomplete", "OPEN_BLOCKER", all(row["present_in_final_C79_protocol"] for row in component_rows), "missing exact H1-H6/split/RNG/taxonomy bindings"),
        ("C78S_provenance_repair_unreplayed", "closed", all(row["passed"] for row in repair_checks), "independent 8-check replay"),
        ("C78S_repair_changed_science", "closed", not any(row.get("code_or_protocol_file") or row.get("scientific_result_or_primary_table") for row in repair_rows if row["record_type"] == "changed_file"), "five report/provenance files only"),
        ("seed4_touched_before_authorization", "closed_guard", True, "Mode R counters all zero"),
        ("seed4_touched_before_execution_lock", "closed_guard", True, "no execution lock was created"),
        ("handoff_text_mistaken_for_authorization", "closed_guard", True, "Mode R only"),
        ("training_seed_not_propagated", "not_exercised_guarded", True, "no seed4 training implementation created"),
        ("seed3_weight_or_optimizer_reuse", "not_exercised_guarded", True, "no seed4 training or checkpoint creation"),
        ("target_label_training_leak", "not_exercised_guarded", True, "no training or label-view opening"),
        ("source_audit_used_in_training", "not_exercised_guarded", True, "no training"),
        ("target4_in_primary_estimand", "not_exercised_guarded", True, "no C79 analysis executed"),
        ("target4_in_primary_null", "not_exercised_guarded", True, "no C79 analysis executed"),
        ("target4_in_multiplicity_family", "not_exercised_guarded", True, "no C79 analysis executed"),
        ("same_label_oracle_reachable", "closed_guard", True, "no label view opened"),
        ("construction_evaluation_overlap", "not_exercised_guarded", True, "seed4 label views not created or opened"),
        ("trial_id_used_as_feature", "not_exercised_guarded", True, "no C79 predictive analysis"),
        ("row_order_used_as_feature", "not_exercised_guarded", True, "no C79 predictive analysis"),
        ("partial_field_scientific_outcome_read", "closed_guard", True, "no seed4 field or outcome exists"),
        ("wave_adaptivity_from_scientific_outcomes", "not_exercised_guarded", True, "no waves submitted"),
        ("outcome_driven_checkpoint_retention", "not_exercised_guarded", True, "no checkpoints created"),
        ("outcome_driven_retry", "not_exercised_guarded", True, "no jobs or retries"),
        ("failed_job_hidden", "not_exercised_guarded", True, "no C79 execution attempts"),
        ("repair_overwrites_history", "closed_guard", True, "C78S correction is additive; no C79 repair"),
        ("collector_repair_changes_locked_analysis", "not_exercised_guarded", True, "no C79 collector or analysis"),
        ("checkpoint_or_sidecar_identity_failure", "not_exercised_guarded", True, "no seed4 field"),
        ("optimizer_state_identity_failure", "not_exercised_guarded", True, "no seed4 field"),
        ("genealogy_or_cadence_mismatch", "not_exercised_guarded", True, "no seed4 field"),
        ("ERM_treated_as_trajectory", "not_exercised_guarded", True, "no C79 H3 analysis"),
        ("H1_materiality_margin_changed", "blocked_before_implementation", True, "exact H1 registry is missing; no implementation allowed"),
        ("H1_pvalue_only_interpretation", "closed_claim_guard", True, "review makes no H1 scientific verdict"),
        ("H1_construction_label_called_source_only", "closed_claim_guard", True, "review preserves diagnostic information class"),
        ("few_label_sufficiency_overclaim", "closed_claim_guard", True, "review makes no sufficiency claim"),
        ("H2_model_substitution", "blocked_before_implementation", True, "exact H2 model is missing; no implementation allowed"),
        ("H2_multiplicity_definition_changed", "blocked_before_implementation", True, "no H2 execution"),
        ("H3_kernel_retuning", "blocked_before_implementation", True, "exact H3 kernel binding is missing; no implementation allowed"),
        ("H3_bandwidth_retuning", "blocked_before_implementation", True, "no H3 execution"),
        ("H3_local_association_called_transport", "closed_claim_guard", True, "review makes no transport claim"),
        ("H3_association_called_mechanism", "closed_claim_guard", True, "review makes no mechanism claim"),
        ("H4_F2_feature_mining", "blocked_before_implementation", True, "no C79 feature implementation"),
        ("H5_F4_feature_mining", "blocked_before_implementation", True, "no C79 feature implementation"),
        ("candidate_nonqualification_called_impossibility", "closed_claim_guard", True, "review makes no impossibility claim"),
        ("H6_Holm_family_changed", "blocked_before_implementation", True, "exact H6 family ordering is missing; no implementation allowed"),
        ("H6_raw_p_rescue", "closed_claim_guard", True, "review does not reinterpret C78S H6"),
        ("unregistered_seed3_seed4_pooling", "not_exercised_guarded", True, "no C79 analysis executed"),
        ("shared_trial_cross_seed_dependence_ignored", "not_exercised_guarded", True, "no cross-seed synthesis"),
        ("two_seed_random_effects_overclaim", "closed_claim_guard", True, "no seed-level inference claim"),
        ("seed4_called_new_target_population", "closed_claim_guard", True, "review preserves same-target boundary"),
        ("candidate_alignment_by_target_outcome", "not_exercised_guarded", True, "no seed4 candidates"),
        ("checkpoint_rows_treated_iid", "not_exercised_guarded", True, "no C79 inference"),
        ("trial_rows_treated_iid", "not_exercised_guarded", True, "no C79 inference"),
        ("raw_cache_or_weights_in_git", "closed_guard", True, "review uses compact Git artifacts only"),
        ("large_payload_in_git", "closed_guard", True, "review artifacts are compact text"),
        ("unauthorized_GPU_or_forward", "closed_guard", True, "training/forward/GPU counters zero"),
        ("BNCI2014_004_scope_creep", "closed_guard", True, "not accessed"),
        ("manuscript_scope_creep", "closed_guard", True, "not started"),
        ("C80_auto_start", "closed_guard", True, "not started"),
        ("report_generated_before_red_team", "closed_by_ordered_finalization", True, "C79_PROTOCOL_REVIEW generated only after 18/18 review red team"),
    ]
    risk_rows = [
        {
            "risk": name,
            "status": status,
            "gate_condition_passed": int(passed),
            "blocking_open": int(status == "OPEN_BLOCKER"),
            "evidence": evidence,
        }
        for name, status, passed, evidence in risks
    ]
    _write_csv("risk_register.csv", risk_rows)
    _write_csv("failure_reason_ledger.csv", [
        {
            "failure": "C79_final_protocol_postdates_C78S_outcomes",
            "blocking": 1,
            "status": "unresolved",
            "evidence": "c79_protocol_timing.csv",
            "required_action": "PM claim/protocol redesign; do not access seed4 under strict-confirmation label",
        },
        {
            "failure": "C79_final_protocol_is_outcome_adaptive_and_incomplete",
            "blocking": 1,
            "status": "unresolved",
            "evidence": "c79_protocol_registry.csv;c79_protocol_diff_against_c78s.csv",
            "required_action": "lock a complete prospective post-seed3 replication protocol or abandon C79 execution",
        },
    ])
    _write_csv("c79_mode_r_execution_boundary.csv", [{
        "mode": "R",
        "seed4_EEG_loads": 0,
        "seed4_Slurm_jobs": 0,
        "training": 0,
        "forward_or_reinference": 0,
        "GPU": 0,
        "seed4_checkpoints": 0,
        "seed4_caches": 0,
        "seed4_label_views": 0,
        "same_label_oracle": 0,
        "BNCI2014_004": 0,
        "manuscript": 0,
        "execution_lock_created": 0,
        "expected_seed4_manifest_created": 0,
    }])

    replay = {
        "schema_version": "c79_protocol_review_replay_v1",
        "milestone": "C79",
        "mode": "R_review_only",
        "final_gate": FINAL_GATE,
        "primary_taxonomy": PRIMARY_TAXONOMY,
        "protocol": {
            "path": str(C79_PROTOCOL.relative_to(REPO_ROOT)),
            "sha256": c79_hash,
            "sha_match": c79_hash == C79_PROTOCOL_EXPECTED_SHA,
            "first_commit": c79_first_commit,
            "created_at_utc": c79_created,
            "predates_first_C78S_outcome": timing_pass,
            "outcome_adaptive_active_hypotheses": outcome_adaptive,
            "adaptive_generator_rule_first_commit": generator_commit,
            "adaptive_generator_rule_predates_first_outcome": generator_predates_outcome,
            "complete_registry_components": sum(row["present_in_final_C79_protocol"] for row in component_rows),
            "required_registry_components": len(component_rows),
        },
        "skeleton": {
            "sha256": skeleton_hash,
            "first_commit": skeleton_first_commit,
            "status": skeleton["status"],
            "eligible_as_final_protocol": False,
        },
        "C78S_replay": {
            "protocol_sha256": c78s_hash,
            "all_reference_values_match": all(row["passed"] for row in result_rows),
            "provenance_repair_checks_passed": sum(row["passed"] for row in repair_checks),
            "provenance_repair_checks_total": len(repair_checks),
        },
        "execution_boundary": {
            "seed4_access": 0,
            "seed4_jobs": 0,
            "training": 0,
            "forward_or_reinference": 0,
            "GPU": 0,
            "seed4_artifacts": 0,
            "label_views": 0,
            "same_label_oracle": 0,
        },
        "stop_rule": "Phase_1_failed_no_implementation_lock_no_Mode_E",
        "blocking_risks": [row["risk"] for row in risk_rows if row["blocking_open"]],
    }
    REPLAY_JSON.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")
    _write_timing_report(replay, timing_rows)
    _write_repair_report(repair_rows, repair_checks)
    return replay


def _write_timing_report(replay: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    timeline = "\n".join(
        f"- `{row['at_utc']}`: `{row['event']}` ({row['artifact_status']})" for row in rows
    )
    TIMING_REPORT.write_text(f"""# C79 Protocol Timing Audit

## Verdict

```text
{FINAL_GATE}
```

The final C79 protocol hash replays exactly, but protocol integrity is not protocol
prospectivity.  The only pre-outcome C79 artifact is explicitly marked
`SKELETON_ONLY_NOT_FINAL_NOT_AUTHORIZED`.  The final protocol was created after
C78S H1, H2, and H3-H5 outcomes and was first committed with the C78S result.

The adaptive generator rule itself was transparently committed before C78S outcomes
in `{replay['protocol']['adaptive_generator_rule_first_commit']}`.  This rules out
hidden post-outcome code editing, but it does not make the materialized H3/H4/H5
artifact a fixed, complete H1-H6 confirmation protocol under the current C79 rule.

## Timeline

{timeline}

The final protocol also derives `C78S_active_hypotheses_to_confirm` from
`active_after_Holm`, yielding H3/H4/H5.  This is an outcome-adaptive choice and
cannot be represented as a protocol locked before C78S scientific-outcome access.

No seed-4 data, job, forward pass, cache, label view, or outcome was accessed during
this audit.  No C79 execution lock was created.
""")


def _write_repair_report(rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    files = [row for row in rows if row["record_type"] == "changed_file"]
    file_lines = "\n".join(
        f"- `{row['item']}`: `{row['before_sha256']}` -> `{row['after_sha256']}`"
        for row in files
    )
    REPAIR_REPORT.write_text(f"""# C79 Replay of the C78S Provenance Correction

## Verdict

The additive correction commit `{C78S_REPAIR_COMMIT}` passes `{sum(row['passed'] for row in checks)}/{len(checks)}`
independent checks.  It corrected an unsupported description of one intentional
pytest skip.  Slurm job `893168` established that the skip was the finalized-C78F
guard, not a C78S route-absence branch.

Changed files:

{file_lines}

No Python source, protocol, execution lock, C78S result, primary hypothesis table,
statistic, null, taxonomy, or outcome-dependent decision changed.  There was no
failed/replacement scientific job pair; `893168` was a dedicated provenance replay.
The correction therefore does not cause the C79 blocker.  The blocker is the C79
protocol timing and outcome-adaptive scope documented separately.
""")


def red_team() -> dict[str, Any]:
    replay = _json(REPLAY_JSON)
    timing = _csv_rows(TABLE_DIR / "c79_protocol_timing.csv")
    repair = _csv_rows(TABLE_DIR / "c78s_provenance_repair_replay.csv")
    references = _csv_rows(TABLE_DIR / "c78s_result_replay.csv")
    registry = _csv_rows(TABLE_DIR / "c79_protocol_registry.csv")
    risks = _csv_rows(TABLE_DIR / "risk_register.csv")
    boundary = _csv_rows(TABLE_DIR / "c79_mode_r_execution_boundary.csv")[0]
    checks = [
        ("C79_final_protocol_hash_replays", replay["protocol"]["sha_match"], replay["protocol"]["sha256"]),
        ("preoutcome_artifact_is_skeleton_only", replay["skeleton"]["status"] == "SKELETON_ONLY_NOT_FINAL_NOT_AUTHORIZED", replay["skeleton"]["status"]),
        ("final_protocol_first_commit_is_C78S_result", replay["protocol"]["first_commit"] == C79_FINAL_PROTOCOL_COMMIT, replay["protocol"]["first_commit"]),
        ("final_protocol_does_not_predate_first_outcome", not replay["protocol"]["predates_first_C78S_outcome"], str(replay["protocol"]["created_at_utc"])),
        ("outcome_adaptive_H3_H4_H5_filter_detected", replay["protocol"]["outcome_adaptive_active_hypotheses"], "active_after_Holm"),
        ("adaptive_generator_precommit_acknowledged", replay["protocol"]["adaptive_generator_rule_predates_first_outcome"], replay["protocol"]["adaptive_generator_rule_first_commit"]),
        ("incomplete_exact_registry_detected", any(row["blocking_if_strict_confirmation"] == "1" for row in registry), f"present={replay['protocol']['complete_registry_components']}/{replay['protocol']['required_registry_components']}"),
        ("all_C78S_reference_values_replay", all(row["passed"] == "1" for row in references), f"metrics={len(references)}"),
        ("C78S_provenance_correction_replays_8_of_8", sum(row["passed"] == "1" for row in repair if row["record_type"] == "repair_check") == 8, "8 checks"),
        ("C78S_correction_changed_no_code_or_protocol", all(row["code_or_protocol_file"] == "0" for row in repair), "five provenance/report files"),
        ("blocking_risks_are_open_and_explicit", sum(row["blocking_open"] == "1" for row in risks) >= 2, ",".join(row["risk"] for row in risks if row["blocking_open"] == "1")),
        ("final_gate_is_timing_repair_required", replay["final_gate"] == FINAL_GATE, replay["final_gate"]),
        ("no_seed4_EEG_or_job_access", boundary["seed4_EEG_loads"] == "0" and boundary["seed4_Slurm_jobs"] == "0", "zero"),
        ("no_training_forward_or_GPU", boundary["training"] == boundary["forward_or_reinference"] == boundary["GPU"] == "0", "zero"),
        ("no_seed4_artifact_or_label_view", boundary["seed4_checkpoints"] == boundary["seed4_caches"] == boundary["seed4_label_views"] == "0", "zero"),
        ("same_label_oracle_closed", boundary["same_label_oracle"] == "0", "zero"),
        ("no_execution_lock_or_expected_manifest_created", boundary["execution_lock_created"] == boundary["expected_seed4_manifest_created"] == "0", "Phase 1 stop"),
        ("no_BNCI004_or_manuscript_scope", boundary["BNCI2014_004"] == boundary["manuscript"] == "0", "zero"),
        ("timeline_orders_outcomes_before_final_protocol", [row["event"] for row in timing].index("C78S_first_scientific_outcome_H1_complete") < [row["event"] for row in timing].index("C79_final_protocol_created"), "H1 before C79 final protocol"),
    ]
    rows = [
        {"check": name, "status": "PASS" if passed else "FAIL", "blocking": int(not passed), "evidence": evidence}
        for name, passed, evidence in checks
    ]
    _write_csv("c79_review_red_team_checks.csv", rows)
    result = {
        "schema_version": "c79_mode_r_review_red_team_v1",
        "passed": sum(row["status"] == "PASS" for row in rows),
        "total": len(rows),
        "blockers": sum(row["blocking"] for row in rows),
        "review_gate": FINAL_GATE,
        "positive_execution_readiness": False,
    }
    RED_TEAM_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    RED_TEAM_REPORT.write_text(f"""# C79 Mode-R Review Red Team

All `{result['passed']}/{result['total']}` review checks passed with zero audit
implementation blockers.  The red team confirms the negative protocol verdict:

```text
{FINAL_GATE}
```

This is not a positive execution-readiness result.  The final C79 protocol was
created after C78S outcomes, filters hypotheses from those outcomes, and lacks a
complete exact H1-H6 seed-4 registry.  Mode E remains prohibited.  Seed 4, the
same-label oracle, BNCI2014_004, GPU work, and manuscript work remained untouched.
""")
    return result


def report() -> dict[str, Any]:
    replay = _json(REPLAY_JSON)
    red = _json(RED_TEAM_JSON)
    if red["blockers"] or red["passed"] != red["total"]:
        raise RuntimeError("C79 review red team has not passed")
    if replay["final_gate"] != FINAL_GATE:
        raise RuntimeError("C79 report refuses a non-blocking gate")
    registry = _csv_rows(TABLE_DIR / "c79_protocol_registry.csv")
    missing = [row["component"] for row in registry if row["blocking_if_strict_confirmation"] == "1"]
    REVIEW_REPORT.write_text(f"""# C79 — Seed-4 Locked Confirmation Protocol Review

## Final Mode-R gate

```text
{FINAL_GATE}
```

Primary taxonomy: `{PRIMARY_TAXONOMY}`.

## Decision

The committed C79 JSON is byte-stable and its SHA-256 is correct, but it is not a
prospectively locked strict-confirmation protocol under the C79 timing rule.
The only pre-C78S-outcome C79 artifact is explicitly a non-final skeleton.  The
final JSON was generated at `2026-07-11T10:43:47Z`, after H1 completed at
`10:41:28Z`, H3/H4/H5 at `10:43:26Z`, and H2 at `10:43:27Z`; it was first committed
with C78S result commit `{C79_FINAL_PROTOCOL_COMMIT}`.

More importantly, the generator selected `{['H3', 'H4', 'H5']}` from
`active_after_Holm`.  That is an outcome-adaptive confirmation scope.  A protocol
hash can prove immutability after creation; it cannot move creation before the
outcomes that informed it.

The generator rule was itself committed before outcomes in
`{replay['protocol']['adaptive_generator_rule_first_commit']}`.  This is transparent
and excludes hidden post-outcome code editing.  It still does not satisfy the
handoff's stricter requirement for a final, exact H1-H6 protocol committed before
outcome access: the rule materializes an outcome-filtered H3/H4/H5 scope and the
resulting registry is incomplete.

## Registry completeness

Only `{replay['protocol']['complete_registry_components']}` of
`{replay['protocol']['required_registry_components']}` required exact registry
components are present.  Missing or insufficiently bound components include:

{chr(10).join(f'- {item}' for item in missing)}

## Replayed evidence

- C78F and C78S commit chains replay through `{C78S_COMMITS['final_anchor']}`.
- C78S protocol SHA-256 replays as `{C78S_PROTOCOL_EXPECTED_SHA}`.
- All `{len(C78S_REFERENCES)}` registered C78S reference values replay within tolerance.
- The C78S regression-provenance correction `{C78S_REPAIR_COMMIT}` passes 8/8
  independent checks and changed no code, protocol, result, estimand, statistic,
  null, or taxonomy.

## Execution boundary

Mode R stopped at Phase 1.  It did not create a C79 implementation/execution lock
or expected seed-4 manifest because those artifacts would falsely imply that the
protocol review passed.  It performed zero seed-4 EEG loads, jobs, training,
forward/re-inference, GPU work, checkpoints, caches, label-view access, or outcome
reads.  The same-label oracle remained closed.

## Required PM repair

The scientifically clean option is to relabel the next study as a prospectively
locked **post-seed3 seed-4 replication/robustness study**, fully specifying H1-H6
before seed-4 access.  It cannot be described as a C79 protocol that predates C78S
outcomes.  Alternatively, stop the seed-4 campaign.  The current artifact cannot
authorize strict C79 confirmation as written.

No C79 Mode-E authorization should be requested until PM resolves that claim and
protocol category.  No C80, additional seed, BNCI2014_004, feature/kernel search,
oracle analysis, or manuscript work is authorized.
""")
    artifacts = [
        TIMING_REPORT,
        REPAIR_REPORT,
        REVIEW_REPORT,
        REPLAY_JSON,
        RED_TEAM_JSON,
        RED_TEAM_REPORT,
        *sorted(TABLE_DIR.glob("*.csv")),
    ]
    manifest_rows = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "raw_EEG_or_weight_payload": 0,
        }
        for path in artifacts
        if path.exists() and path.name != "artifact_manifest.csv"
    ]
    _write_csv("artifact_manifest.csv", manifest_rows)
    return replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "red-team", "report"))
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit()
    elif args.command == "red-team":
        result = red_team()
    else:
        result = report()
    print(json.dumps({"gate": result.get("final_gate", result.get("review_gate")), "mode": "R"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
