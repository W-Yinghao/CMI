"""Independent pre-execution and result red-team checks for C78S."""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any, Callable

from . import c78s_protocol as protocol


PRE_JSON = protocol.REPORT_DIR / "C78S_PRE_EXECUTION_RED_TEAM.json"
PRE_MD = protocol.REPORT_DIR / "C78S_PRE_EXECUTION_RED_TEAM.md"
PRE_TABLE = protocol.TABLE_DIR / "pre_execution_red_team_checks.csv"
RESULT_JSON = protocol.REPORT_DIR / "C78S_AUTHORIZED_RED_TEAM_VERIFICATION.json"
RESULT_MD = protocol.REPORT_DIR / "C78S_AUTHORIZED_RED_TEAM_VERIFICATION.md"
RESULT_TABLE = protocol.TABLE_DIR / "authorized_red_team_checks.csv"
RESULT_PATH = protocol.REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS.json"
ARTIFACT_MANIFEST = protocol.REPORT_DIR / "C78S_ARTIFACT_MANIFEST.json"


def _check(rows: list[dict[str, Any]], name: str, condition: bool, evidence: Any) -> None:
    rows.append({
        "check": name,
        "status": "PASS" if condition else "FAIL",
        "blocking": int(not condition),
        "evidence": json.dumps(evidence, sort_keys=True, ensure_ascii=True) if not isinstance(evidence, str) else evidence,
    })


def _source_imports(path: str | Path) -> set[str]:
    tree = ast.parse(Path(path).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _raw_lock() -> tuple[dict[str, Any], str]:
    expected = protocol.LOCK_SHA_PATH.read_text().strip()
    observed = protocol.sha256_file(protocol.LOCK_PATH)
    if expected != observed:
        raise RuntimeError("C78S pre-execution lock hash mismatch")
    return json.loads(protocol.LOCK_PATH.read_text()), observed


def preflight() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    locked_protocol, protocol_sha = protocol.load_protocol()
    route, route_sha = protocol.load_primary_route()
    lock, lock_sha = _raw_lock()
    _check(rows, "protocol_sha_exact", protocol_sha == protocol.PROTOCOL_SHA_PATH.read_text().strip(), protocol_sha)
    _check(rows, "protocol_authoritative_sha", protocol_sha == "df85699090a65d1e1766d754bcebd9eb5648cc13e4441d8074a3f4884487c7f8", protocol_sha)
    _check(rows, "route_sha_exact", route_sha == protocol.ROUTE_SHA_PATH.read_text().strip(), route_sha)
    _check(rows, "lock_sha_exact", lock_sha == protocol.LOCK_SHA_PATH.read_text().strip(), lock_sha)
    _check(rows, "direct_authorization_received", lock["authorization"]["received"] is True, lock["authorization"])
    _check(rows, "direct_authorization_mode", lock["authorization"]["mode"] == protocol.AUTHORIZATION_MODE, lock["authorization"])
    _check(rows, "authorization_evidence_hash", lock["authorization"]["evidence_sha256"] == protocol.AUTHORIZATION_EVIDENCE_SHA256, lock["authorization"]["evidence_sha256"])
    _check(rows, "scope_bound", lock["authorization"]["scope_bound"] is True, lock["scope"])
    _check(rows, "primary_targets_exact", tuple(lock["scope"]["primary_targets"]) == protocol.PRIMARY_TARGETS, lock["scope"]["primary_targets"])
    _check(rows, "target4_excluded", lock["scope"]["target4_primary"] is False and route["target4_included"] is False, route["primary_targets"])
    _check(rows, "seed3_only", lock["scope"]["seed"] == 3, lock["scope"]["seed"])
    _check(rows, "seed4_blocked", lock["scope"]["seed4"] is False, lock["scope"])
    _check(rows, "C79_blocked", lock["scope"]["C79"] is False, lock["scope"])
    _check(rows, "BNCI004_blocked", lock["scope"]["BNCI2014_004"] is False, lock["scope"])
    _check(rows, "training_blocked", lock["scope"]["training"] is False, lock["scope"])
    _check(rows, "forward_blocked", lock["scope"]["forward"] is False, lock["scope"])
    _check(rows, "reinference_blocked", lock["scope"]["reinference"] is False, lock["scope"])
    _check(rows, "GPU_blocked", lock["scope"]["GPU"] is False, lock["scope"])
    _check(rows, "manuscript_blocked", lock["scope"]["manuscript"] is False, lock["scope"])
    _check(rows, "same_label_oracle_blocked", lock["scope"]["same_label_oracle"] is False, lock["scope"])
    route_text = protocol.ROUTE_PATH.read_text()
    _check(rows, "oracle_descriptor_absent", "same_label_oracle_view" not in route_text and "/oracle/" not in route_text, "oracle tokens absent")
    _check(rows, "four_views_per_target", all(len(item) == 5 for item in route["views"].values()), {target: sorted(item) for target, item in route["views"].items()})
    _check(rows, "route_targets_exact", set(map(int, route["views"])) == set(protocol.PRIMARY_TARGETS), sorted(route["views"]))
    _check(rows, "trial_id_not_predictor", route["trial_id_role"].endswith("never_predictor"), route["trial_id_role"])
    _check(rows, "row_order_not_predictor", route["row_order_role"].endswith("never_predictor"), route["row_order_role"])
    implementation = lock["implementation_files"]
    _check(rows, "implementation_file_count", len(implementation) == len(protocol.IMPLEMENTATION_FILES), len(implementation))
    for item in implementation:
        _check(rows, f"implementation_hash::{Path(item['path']).name}", protocol.sha256_file(item["path"]) == item["sha256"], item["sha256"])
    analysis_imports = _source_imports("oaci/conditioned_ceiling_coverage/c78s_seed3_scientific_analysis.py")
    _check(rows, "analysis_no_torch_import", "torch" not in analysis_imports, sorted(analysis_imports))
    _check(rows, "analysis_no_EEG_loader_import", "moabb" not in analysis_imports and "mne" not in analysis_imports, sorted(analysis_imports))
    _check(rows, "H1_H6_exact", [item["id"] for item in locked_protocol["primary_hypotheses"]] == [f"H{i}" for i in range(1, 7)], [item["id"] for item in locked_protocol["primary_hypotheses"]])
    _check(rows, "null_replicates_locked", lock["locked_analysis"]["null_replicates"] == 499, lock["locked_analysis"]["null_replicates"])
    _check(rows, "bootstrap_replicates_locked", lock["locked_analysis"]["bootstrap_replicates"] == 2000, lock["locked_analysis"]["bootstrap_replicates"])
    _check(rows, "feature_paths_locked", len(lock["locked_analysis"]["primary_feature_paths"]) == 4, lock["locked_analysis"]["primary_feature_paths"])
    _check(rows, "geometry_prefixes_locked", lock["locked_analysis"]["candidate_prefix_sizes"] == [5, 10, 20, 40], lock["locked_analysis"]["candidate_prefix_sizes"])
    _check(rows, "oracle_stage_not_run", lock["locked_analysis"]["same_label_oracle_stage"] == "not_run_in_C78S", lock["locked_analysis"]["same_label_oracle_stage"])
    _check(rows, "no_label_payload_read_before_lock", lock["before_lock"]["quarantined_label_payload_reads_by_C78S"] == 0, lock["before_lock"])
    _check(rows, "no_outcomes_before_lock", lock["before_lock"]["scientific_outcomes_computed_by_C78S"] == 0, lock["before_lock"])
    failures = sum(row["blocking"] for row in rows)
    result = {
        "schema_version": "c78s_pre_execution_red_team_v1",
        "created_at_utc": protocol.utc_now(),
        "checks": len(rows),
        "blocking_failures": failures,
        "passed": failures == 0,
        "protocol_sha256": protocol_sha,
        "execution_lock_sha256": lock_sha,
    }
    protocol.write_csv(PRE_TABLE, rows)
    protocol.write_json(PRE_JSON, result)
    PRE_MD.write_text(
        "# C78S Pre-Execution Red Team\n\n"
        f"Result: **{'PASS' if result['passed'] else 'FAIL'}** ({len(rows) - failures}/{len(rows)} checks, {failures} blockers).\n\n"
        "This audit ran before any C78S label payload read or scientific outcome computation.\n"
    )
    if failures:
        raise RuntimeError(f"C78S pre-execution red team failed {failures} checks")
    return result


def _table(name: str) -> list[dict[str, str]]:
    return protocol.read_csv(protocol.TABLE_DIR / name)


def _float(value: Any) -> float:
    return float(value)


def result_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    lock, lock_sha = protocol.load_execution_lock()
    result = json.loads(RESULT_PATH.read_text())
    artifacts = json.loads(ARTIFACT_MANIFEST.read_text())
    _check(rows, "result_schema", result["schema_version"] == "c78s_seed3_scientific_analysis_result_v1", result["schema_version"])
    _check(rows, "protocol_hash_replayed", result["protocol_sha256"] == lock["protocol_sha256"], result["protocol_sha256"])
    _check(rows, "execution_lock_replayed", result["execution_lock_sha256"] == lock_sha, result["execution_lock_sha256"])
    _check(rows, "authorization_scope", result["authorization"]["scope"] == "C78S_locked_seed3_analysis_only", result["authorization"])
    _check(rows, "primary_unit_count", result["field"]["units"] == 1296, result["field"])
    _check(rows, "primary_targets_exact", result["field"]["primary_targets"] == list(protocol.PRIMARY_TARGETS), result["field"]["primary_targets"])
    _check(rows, "target4_primary_false", result["field"]["target4_primary"] is False, result["field"])
    _check(rows, "seed3_exploratory", result["field"]["seed3_role"] == "exploratory_replication_not_seed_confirmation", result["field"]["seed3_role"])
    field_replay = _table("field_replay.csv")
    complete = next(row for row in field_replay if row["registry"] == "complete_seed3")
    primary = next(row for row in field_replay if row["registry"] == "C78S_primary")
    _check(rows, "complete_1458_replay", int(complete["observed_units"]) == 1458 and int(complete["unique_units"]) == 1458, complete)
    _check(rows, "primary_1296_replay", int(primary["observed_units"]) == 1296 and int(primary["unique_units"]) == 1296, primary)
    _check(rows, "primary_target4_zero", int(primary["target4_units"]) == 0, primary)
    exclusion = _table("target4_exclusion_audit.csv")[0]
    _check(rows, "target4_estimand_zero", int(exclusion["target4_primary_estimands"]) == 0, exclusion)
    _check(rows, "target4_null_zero", int(exclusion["target4_primary_null_members"]) == 0, exclusion)
    _check(rows, "target4_multiplicity_zero", int(exclusion["target4_primary_multiplicity_members"]) == 0, exclusion)
    split = _table("label_split_isolation.csv")
    _check(rows, "split_target_count", len(split) == 8, len(split))
    _check(rows, "split_disjoint_all", all(int(row["overlap_rows"]) == 0 for row in split), split)
    _check(rows, "split_union_576_all", all(int(row["union_rows"]) == 576 for row in split), split)
    _check(rows, "split_oracle_zero", all(int(row["same_label_oracle_accessed"]) == 0 for row in split), split)
    _check(rows, "split_trial_id_not_predictor", all(int(row["trial_id_used_as_predictor"]) == 0 for row in split), split)
    oracle = _table("oracle_access_audit.csv")[0]
    _check(rows, "oracle_descriptor_absent", int(oracle["primary_route_contains_oracle_descriptor"]) == 0, oracle)
    _check(rows, "oracle_not_opened", int(oracle["same_label_oracle_view_opened"]) == 0, oracle)
    _check(rows, "terminal_oracle_not_run", int(oracle["terminal_oracle_stage_run"]) == 0, oracle)
    features = _table("feature_block_registry.csv")
    _check(rows, "feature_registry_F0_F5", [row["block"] for row in features] == [f"F{i}" for i in range(6)], [row["block"] for row in features])
    _check(rows, "feature_dimensions", [int(row["dimension"]) for row in features] == [9, 25, 25, 18, 35, 15], [row["dimension"] for row in features])
    _check(rows, "trial_id_never_feature", all(int(row["predictor_trial_id"]) == 0 for row in features), features)
    primary_rows = _table("primary_hypothesis_multiplicity.csv")
    _check(rows, "H1_H6_exact_once", [row["hypothesis"] for row in primary_rows] == [f"H{i}" for i in range(1, 7)], [row["hypothesis"] for row in primary_rows])
    _check(rows, "result_H1_H6_match", [row["hypothesis"] for row in result["primary_hypotheses"]] == [f"H{i}" for i in range(1, 7)], result["primary_hypotheses"])
    recomputed = modeling_holm([float(row["raw_p"]) for row in primary_rows])
    _check(rows, "Holm_recomputed", all(abs(float(row["Holm_p"]) - value) < 1e-12 for row, value in zip(primary_rows, recomputed)), recomputed)
    measurement = _table("measurement_control_summary.csv")[0]
    _check(rows, "headline_reliability_match", abs(_float(measurement["trajectory_reliability_mean"]) - result["headline"]["trajectory_reliability_mean"]) < 1e-12, measurement["trajectory_reliability_mean"])
    prediction = _table("cross_fitted_incremental_prediction.csv")
    strict = next(row for row in prediction if row["path"] == "strict_source_F2" and row["outcome"] == "continuous_joint_utility")
    target = next(row for row in prediction if row["path"] == "target_unlabeled_F4_geometry" and row["outcome"] == "continuous_joint_utility")
    _check(rows, "strict_headline_match", abs(_float(strict["incremental_LOTO_R2"]) - result["headline"]["strict_source_incremental_R2"]) < 1e-12, strict["incremental_LOTO_R2"])
    _check(rows, "target_headline_match", abs(_float(target["incremental_LOTO_R2"]) - result["headline"]["target_unlabeled_incremental_R2"]) < 1e-12, target["incremental_LOTO_R2"])
    candidate = _table("registered_candidate_gate.csv")
    _check(rows, "candidate_gate_two_paths", {row["path"] for row in candidate} == {"strict_source_F2", "target_unlabeled_F4_geometry"}, candidate)
    _check(rows, "candidate_gate_materiality_explicit", all("material_actionability" in row for row in candidate), candidate)
    association_sep = _table("association_prediction_separation.csv")
    _check(rows, "association_prediction_separated", all(int(row["association_is_not_prediction"]) == 1 for row in association_sep), association_sep)
    _check(rows, "prediction_actionability_separated", all(int(row["prediction_is_not_actionability"]) == 1 for row in association_sep), association_sep)
    geometry = _table("effective_multiplicity_summary.csv")[0]
    _check(rows, "geometry_diagnostic_only", int(geometry["endpoint_geometry_is_diagnostic_not_selector"]) == 1, geometry)
    _check(rows, "geometry_headline_match", abs(_float(geometry["incremental_deviance_reduction"]) - result["headline"]["geometry_deviance_reduction"]) < 1e-12, geometry["incremental_deviance_reduction"])
    bootstrap = _table("hierarchical_bootstrap_summary.csv")
    _check(rows, "target_bootstrap_present", any(row["bootstrap_level"] == "target" for row in bootstrap), bootstrap)
    _check(rows, "checkpoint_bootstrap_present", any(row["bootstrap_level"] == "checkpoint_within_trajectory_then_target" for row in bootstrap), bootstrap)
    trial = _table("trial_cluster_bootstrap_by_target.csv")
    _check(rows, "trial_cluster_eight_targets", len(trial) == 8, len(trial))
    _check(rows, "trial_cluster_499_replicates", all(int(row["trial_bootstrap_replicates"]) == 499 for row in trial), trial)
    crossed = _table("crossed_target_trial_bootstrap.csv")
    _check(rows, "crossed_target_trial_present", len(crossed) == 3 and all(int(row["replicates"]) == 499 for row in crossed), crossed)
    regime = _table("regime_summary.csv")
    erm = next(row for row in regime if row["regime"] == "ERM")
    _check(rows, "ERM_anchor_semantics", erm["trajectory_role"] == "anchor", erm)
    attempts = _table("execution_attempt_ledger.csv")
    _check(rows, "one_registered_execution", len(attempts) == 1, attempts)
    _check(rows, "no_outcome_adaptive_retry", all(int(row["outcome_adaptive_retry"]) == 0 for row in attempts), attempts)
    _check(rows, "no_training_forward_reinference_GPU", all(sum(int(row[key]) for key in ("training", "forward", "reinference", "GPU")) == 0 for row in attempts), attempts)
    boundaries = result["information_boundaries"]
    _check(rows, "result_oracle_false", boundaries["same_label_oracle_accessed"] is False, boundaries)
    _check(rows, "result_seed4_false", boundaries["seed4"] is False and boundaries["C79_execution"] is False, boundaries)
    _check(rows, "result_BNCI004_false", boundaries["BNCI2014_004"] is False, boundaries)
    _check(rows, "result_selector_false", boundaries["selector_or_checkpoint_recommendation"] is False, boundaries)
    _check(rows, "result_manuscript_false", boundaries["manuscript_drafting"] is False, boundaries)
    seed4 = _table("seed4_protection_audit.csv")[0]
    _check(rows, "seed4_all_zero", all(int(value) == 0 for key, value in seed4.items() if key != "passed"), seed4)
    c79 = json.loads((protocol.REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL.json").read_text())
    _check(rows, "C79_protocol_not_execution", c79["execution_requires_future_direct_PI_authorization"] is True and c79["authorization_received"] is False, c79["status"])
    risk = _table("risk_register.csv")
    _check(rows, "risk_registry_complete", len(risk) == len(protocol.risk_registry()), len(risk))
    _check(rows, "no_open_blocking_risk", all(row["status"] == "CLOSED" and int(row["blocking"]) == 0 for row in risk), risk)
    failures = _table("failure_reason_ledger.csv")
    _check(rows, "no_analysis_blocker", all(int(row["blocking"]) == 0 for row in failures), failures)
    tables = artifacts["tables"]
    _check(rows, "artifact_manifest_table_hashes", all(protocol.sha256_file(row["path"]) == row["sha256"] for row in tables), len(tables))
    _check(rows, "git_payload_under_50MiB", all(Path(row["path"]).stat().st_size < 50 * 1024 * 1024 for row in tables), max(Path(row["path"]).stat().st_size for row in tables))
    _check(rows, "nonoracle_output_frozen", result["nonoracle_output_manifest"]["manifest_sha256"] and result["primary_freeze_manifest_sha256"], result["nonoracle_output_manifest"])
    allowed_gates = {
        "SEED3_MEASUREMENT_CONTROL_SEPARATION_REPLICATED_C79_READY_BUT_NOT_AUTHORIZED",
        "SEED3_SOURCE_OR_TARGET_UNLABELED_ESCAPE_HATCH_REQUIRES_FORENSICS",
        "SEED3_MIXED_RESULTS_C79_PROTOCOL_REVIEW_REQUIRED",
    }
    _check(rows, "registered_final_gate", result["final_gate"] in allowed_gates, result["final_gate"])
    failures_count = sum(row["blocking"] for row in rows)
    audit = {
        "schema_version": "c78s_authorized_result_red_team_v1",
        "created_at_utc": protocol.utc_now(),
        "checks": len(rows),
        "blocking_failures": failures_count,
        "passed": failures_count == 0,
        "final_gate_audited": result["final_gate"],
        "target4_primary": False,
        "same_label_oracle_accessed": False,
        "seed4_accessed": False,
    }
    protocol.write_csv(RESULT_TABLE, rows)
    protocol.write_json(RESULT_JSON, audit)
    RESULT_MD.write_text(
        "# C78S Authorized Result Red Team\n\n"
        f"Result: **{'PASS' if audit['passed'] else 'FAIL'}** "
        f"({len(rows) - failures_count}/{len(rows)} checks, {failures_count} blockers).\n\n"
        "The audit independently replayed protocol/lock identity, 1,458/1,296 field coverage, "
        "target-4 exclusion, split-label isolation, H1-H6 multiplicity, hierarchical dependence, "
        "candidate materiality gates, artifact hashes, and all scope boundaries.\n"
    )
    if failures_count:
        raise RuntimeError(f"C78S result red team failed {failures_count} checks")
    return audit


def modeling_holm(p_values: list[float]) -> list[float]:
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    result = [math.nan] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        result[index] = running
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78s_red_team")
    parser.add_argument("command", choices=("preflight", "result"))
    args = parser.parse_args(argv)
    payload = preflight() if args.command == "preflight" else result_audit()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
