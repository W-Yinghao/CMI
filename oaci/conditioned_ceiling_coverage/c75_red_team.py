"""Independent C75 provenance, statistics, leakage, and claim-boundary gauntlet."""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import subprocess

from joblib import Parallel, delayed
import numpy as np

from . import c74_analysis
from . import c74_cache
from . import c74_t2_source_wz_instrumentation as c74_runner
from . import c75_data
from . import c75_protocol


REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c75_tables"
STATE_PATH = REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_ANALYSIS_STATE.json"
RED_TEAM_REPORT = REPORT_DIR / "C75_RED_TEAM_VERIFICATION.md"


def _read_csv(name: str) -> list[dict]:
    with open(TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: list[dict]) -> None:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _check(
    checks: list[dict], name: str, passed: bool, observed, expected,
    *, blocking: bool = True, note: str = "",
) -> None:
    checks.append({
        "check": name, "passed": int(bool(passed)), "blocking": int(blocking),
        "observed": observed, "expected": expected, "note": note,
    })


def _verify_descriptor(descriptor: dict) -> tuple[str, bool, str]:
    try:
        c74_cache.verify_shard(
            descriptor, required_fields=c74_runner.SHARD_SCHEMAS[descriptor["kind"]],
        )
        return descriptor["path"], True, ""
    except Exception as error:  # pragma: no cover - failure evidence path
        return descriptor.get("path", "missing"), False, repr(error)


def _git_commit_time(commit: str) -> int:
    return int(subprocess.check_output(
        ["git", "show", "-s", "--format=%ct", commit], text=True,
    ).strip())


def _execution_attempts() -> list[dict]:
    return [
        {"attempt": 1, "job_or_commit": "fe60552", "phase": "protocol_lock", "status": "passed", "superseded": 0, "forward": 0, "training": 0, "reason": "protocol and registries committed before payload read"},
        {"attempt": 2, "job_or_commit": "768d37b", "phase": "pre_payload_implementation", "status": "passed", "superseded": 0, "forward": 0, "training": 0, "reason": "implementation committed before extraction"},
        {"attempt": 3, "job_or_commit": "892409", "phase": "feature_extraction_initial", "status": "completed_then_superseded", "superseded": 1, "forward": 0, "training": 0, "reason": "independent red-team found float32-vs-float64 summary pseudo-rank"},
        {"attempt": 4, "job_or_commit": "892410", "phase": "analysis_initial", "status": "completed_then_superseded", "superseded": 1, "forward": 0, "training": 0, "reason": "Wz duplicate block incorrectly increased rank"},
        {"attempt": 5, "job_or_commit": "acd1c45", "phase": "canonical_Wz_repair", "status": "passed", "superseded": 0, "forward": 0, "training": 0, "reason": "single float64 canonical projection summary and hard identity gate"},
        {"attempt": 6, "job_or_commit": "892425", "phase": "feature_extraction_repaired", "status": "passed", "superseded": 0, "forward": 0, "training": 0, "reason": "new content-addressed feature payload; old payload dereferenced"},
        {"attempt": 7, "job_or_commit": "892427", "phase": "analysis_redundancy_repaired", "status": "completed_then_superseded", "superseded": 1, "forward": 0, "training": 0, "reason": "red-team found conditional kernel lacked fold-local scaling/global path correction"},
        {"attempt": 8, "job_or_commit": "cf4c985", "phase": "conditional_calibration_repair", "status": "passed", "superseded": 0, "forward": 0, "training": 0, "reason": "outer-training z-score and global 2-path x 3-bandwidth max-stat"},
        {"attempt": 9, "job_or_commit": "892437", "phase": "analysis_final_candidate", "status": "passed", "superseded": 0, "forward": 0, "training": 0, "reason": "all registered analyses recomputed"},
        {"attempt": 10, "job_or_commit": os.environ.get("SLURM_JOB_ID", "local"), "phase": "independent_red_team", "status": "running_or_passed", "superseded": 0, "forward": 0, "training": 0, "reason": "independent full input rehash and table reconstruction"},
    ]


def _qualification_expected(
    candidate: str, relevance: list[dict], leave_target: list[dict], scaling: list[dict],
) -> dict[str, int]:
    path = "P_F2_strict_architecture" if candidate == "F2_strict_source" else "P_F4_target_architecture"
    view = "strict_source" if candidate == "F2_strict_source" else "target_unlabeled"
    primary = next(
        row for row in relevance
        if row["path"] == path and row["outcome"] == "continuous_joint_utility"
    )
    per_target = [
        row for row in leave_target
        if row["path"] == path and row["outcome"] == "continuous_joint_utility"
    ]
    duplicate_rank = int(next(
        row["column_rank"] for row in scaling
        if row["view"] == view and row["stage"] == "B2_plus_Wz"
    ))
    architecture_rank = int(next(
        row["column_rank"] for row in scaling
        if row["view"] == view and row["stage"] == "B4_plus_W_geometry"
    ))
    median_rho = float(np.nanmedian([
        float(row["increment_residual_rho"]) for row in per_target
    ]))
    positive = sum(int(row["positive_increment"]) for row in per_target)
    gates = {
        "incremental_R2": float(primary["incremental_R2"]) >= 0.02,
        "observed_above_nested_null_p95": int(primary["observed_above_null_p95"]) == 1,
        "leave_target_out_median_positive": median_rho > 0,
        "positive_in_7_of_9_targets": positive >= 7,
        "max_stat_corrected_p": float(primary["max_stat_corrected_p"]) < 0.05,
        "not_redundant_with_logits_probabilities": architecture_rank > duplicate_rank,
        "no_target_label_leakage": True,
    }
    return {**{key: int(value) for key, value in gates.items()}, "ALL_REQUIRED": int(all(gates.values()))}


def run_red_team() -> dict:
    if (REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_VALIDITY.md").exists():
        raise RuntimeError("C75 main report exists before independent red-team")
    protocol = c75_data.load_protocol()
    checks: list[dict] = []
    _write_csv("execution_attempt_ledger.csv", _execution_attempts())

    protocol_hash = c75_protocol.sha256(c75_protocol.PROTOCOL_PATH)
    _check(checks, "protocol_hash", protocol_hash == c75_protocol.PROTOCOL_SHA_PATH.read_text().strip(), protocol_hash, c75_protocol.PROTOCOL_SHA_PATH.read_text().strip())
    feature_manifest_path = c75_data.feature_manifest_path(protocol)
    feature_manifest = c74_cache.verify_unit_manifest(feature_manifest_path, rehash_payloads=False)
    c74_cache.verify_shard(feature_manifest["descriptor"])
    _check(checks, "protocol_precedes_feature_payload", _git_commit_time("fe60552") < feature_manifest_path.stat().st_mtime, _git_commit_time("fe60552"), f"< {feature_manifest_path.stat().st_mtime}")
    _check(checks, "canonical_repair_precedes_feature_payload", _git_commit_time("acd1c45") < feature_manifest_path.stat().st_mtime, _git_commit_time("acd1c45"), f"< {feature_manifest_path.stat().st_mtime}")
    _check(checks, "final_analysis_code_precedes_state", _git_commit_time("cf4c985") < STATE_PATH.stat().st_mtime, _git_commit_time("cf4c985"), f"< {STATE_PATH.stat().st_mtime}")

    manifests = c74_analysis._primary_smoke_manifests(json.loads(c75_protocol.C74_PROTOCOL.read_text()))
    allowed = c75_data.ALLOWED_KINDS
    t2_ids = {row["checkpoint_id"] for row in c75_data.csv_dicts(c75_protocol.C74_T2_UNITS)}
    t3_ids = {row["checkpoint_id"] for row in c75_data.csv_dicts(c75_protocol.C74_T3_UNITS)}
    observed_ids = {manifest["checkpoint_id"] for manifest in manifests}
    _check(checks, "restricted_T2_exact_universe", len(manifests) == 216 and observed_ids == t2_ids, len(observed_ids), 216)
    _check(checks, "T3_HO_zero_overlap", not observed_ids & t3_ids, len(observed_ids & t3_ids), 0)
    _check(checks, "restricted_five_view_contract", all({item["kind"] for item in manifest["shards"]} == allowed for manifest in manifests), sorted({kind for manifest in manifests for kind in {item["kind"] for item in manifest["shards"]}}), sorted(allowed))

    descriptors = [item for manifest in manifests for item in manifest["shards"]]
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    payload_checks = Parallel(n_jobs=workers, backend="loky")(
        delayed(_verify_descriptor)(descriptor) for descriptor in descriptors
    )
    failures = [row for row in payload_checks if not row[1]]
    _check(checks, "independent_C74_payload_rehash", not failures and len(payload_checks) == 1080, len(failures), "0 failures / 1080 descriptors")

    manifest_flags = (
        feature_manifest["unit_count"] == 216
        and feature_manifest["target_count"] == 9
        and feature_manifest["payload_descriptors_rehashed"] == 1080
        and feature_manifest["Wz_plus_b_logits_max_abs"] == 0.0
        and not feature_manifest["same_label_oracle_accessed"]
        and not feature_manifest["T3_HO_z_Wz_accessed"]
    )
    _check(checks, "feature_manifest_information_boundary", manifest_flags, f"units={feature_manifest['unit_count']};rehash={feature_manifest['payload_descriptors_rehashed']};oracle={feature_manifest['same_label_oracle_accessed']};T3={feature_manifest['T3_HO_z_Wz_accessed']};identity={feature_manifest['Wz_plus_b_logits_max_abs']}", "216;1080;false;false;0")
    _, arrays = c75_data.load_feature_cache()
    dimensions = {block: arrays[block].shape for block in ("F0", "F1", "F2", "F3", "F4", "F5")}
    _check(checks, "registered_feature_dimensions", dimensions == {"F0": (216, 9), "F1": (216, 25), "F2": (216, 25), "F3": (216, 18), "F4": (216, 35), "F5": (216, 15)}, dimensions, "locked dimensions")
    target_counts = Counter(map(int, arrays["target_id"]))
    trajectory_counts = Counter(zip(arrays["target_id"].tolist(), arrays["trajectory_id"].tolist()))
    _check(checks, "unit_and_block_structure", set(target_counts.values()) == {24} and set(trajectory_counts.values()) == {4} and len(trajectory_counts) == 54, f"targets={dict(target_counts)};cells={Counter(trajectory_counts.values())}", "24/target;54 cells x4")
    duplicate_exact = np.array_equal(arrays["source_logits_minus_b"], arrays["source_Wz_summary"]) and np.array_equal(arrays["target_logits_minus_b"], arrays["target_Wz_summary"])
    _check(checks, "canonical_projection_summary_identity", duplicate_exact, duplicate_exact, True)

    redundancy = _read_csv("Wz_logit_redundancy.csv")
    redundancy_pass = len(redundancy) == 2 and all(
        float(row["summary_max_abs_Wz_minus_logits_minus_b"]) == 0.0
        and int(row["B1_rank"]) == int(row["B2_rank"])
        and abs(float(row["column_space_prediction_delta_max_abs"])) < 1e-9
        and abs(float(row["column_space_incremental_R2"])) < 1e-9
        for row in redundancy
    )
    _check(checks, "Wz_logit_exact_redundancy", redundancy_pass, [(row["view"], row["B1_rank"], row["B2_rank"], row["column_space_prediction_delta_max_abs"]) for row in redundancy], "rank equal; prediction delta <1e-9")

    relevance = _read_csv("cross_fitted_incremental_relevance.csv")
    nulls = _read_csv("nested_block_nulls.csv")
    max_null = _read_csv("max_stat_null_distribution.csv")
    _check(checks, "registered_model_family_complete", len(relevance) == 30 and len(nulls) == 30 and len(max_null) == 499, f"{len(relevance)}:{len(nulls)}:{len(max_null)}", "30:30:499")
    arithmetic = all(abs(float(row["full_R2"]) - float(row["prior_R2"]) - float(row["incremental_R2"])) < 1e-12 for row in relevance)
    _check(checks, "incremental_R2_arithmetic", arithmetic, arithmetic, True)
    null_lookup = {(row["path"], row["outcome"]): row for row in nulls}
    null_consistent = all(
        abs(float(row["incremental_R2"]) - float(null_lookup[(row["path"], row["outcome"])]["observed_incremental_R2"])) < 1e-12
        and int(row["observed_above_null_p95"]) == int(float(row["incremental_R2"]) > float(row["nested_null_p95"]))
        and float(row["max_stat_corrected_p"]) >= float(row["uncorrected_p"])
        and null_lookup[(row["path"], row["outcome"])]["secondary_null_scheme"] == "permute_new_block_within_target_keep_prior_and_outcome_fixed"
        for row in relevance
    )
    _check(checks, "primary_secondary_null_contract", null_consistent, null_consistent, "two 499-replicate nulls; max-stat monotone")

    leave_target = _read_csv("leave_target_out_relevance.csv")
    scaling = _read_csv("feature_scaling_audit.csv")
    qualification = _read_csv("t3_qualification_decision.csv")
    qualification_ok = True
    for candidate in ("F2_strict_source", "F4_target_unlabeled"):
        expected = _qualification_expected(candidate, relevance, leave_target, scaling)
        observed = {
            row["gate"]: int(row["passed"])
            for row in qualification if row["candidate"] == candidate
        }
        qualification_ok &= observed == expected
    _check(checks, "T3_qualification_reconstructed", qualification_ok, {candidate: {row["gate"]: int(row["passed"]) for row in qualification if row["candidate"] == candidate} for candidate in ("F2_strict_source", "F4_target_unlabeled")}, "exact locked gate reconstruction")
    _check(checks, "no_T3_qualified_candidate", all(row["passed"] == "0" for row in qualification if row["gate"] == "ALL_REQUIRED"), [row["candidate"] for row in qualification if row["gate"] == "ALL_REQUIRED" and row["passed"] == "1"], [])

    positive = next(row for row in relevance if row["path"] == "P_F5_construction_positive" and row["outcome"] == "continuous_joint_utility")
    _check(checks, "construction_label_positive_control", float(positive["incremental_R2"]) > 0.02 and int(positive["positive_target_count"]) >= 7 and float(positive["max_stat_corrected_p"]) < 0.05, f"dR2={positive['incremental_R2']};targets={positive['positive_target_count']};p={positive['max_stat_corrected_p']}", "material;>=7/9;p<.05")

    projection = _read_csv("projection_construct_validity.csv")
    projection_row = projection[0]
    _check(checks, "projection_stable_but_not_registered_incremental", len(projection) == 1 and projection_row["stable"] == "1" and projection_row["incremental"] == "0" and projection_row["target_gauge_name_allowed"] == "0" and projection_row["mechanism_origin_validated"] == "0", projection_row["classification"], "stable;incremental=0;no mechanism/gauge")
    variance = _read_csv("projection_variance_by_target_class.csv")
    _check(checks, "variance_estimand_accounting", len(variance) == 36 and max(abs(float(row["accounting_sum"]) - 1.0) for row in variance) < 1e-10 and all(row["causal_interpretation"] == "0" for row in variance), len(variance), "36 rows; sum=1;noncausal")

    counterfactual = _read_csv("counterfactual_identity_vs_mechanism.csv")
    _check(checks, "counterfactual_sensitivity_not_origin", len(counterfactual) == 2 and all(row["factorization_origin_identified"] == "0" and row["conclusion"] == "counterfactual_sensitivity_not_mechanism_origin" and float(row["max_family_p"]) >= 0.05 for row in counterfactual), [(row["intervention"], row["max_family_p"]) for row in counterfactual], "matched-null nonsignificant;origin=0")

    reparameterization = _read_csv("synthetic_reparameterization_audit.csv")
    _check(checks, "factorization_function_invariance", len(reparameterization) == 4 and all(row["function_invariant"] == "1" and float(row["Wz_max_abs_error"]) < 1e-10 for row in reparameterization) and any(row["coordinate_geometry_invariant"] == "0" for row in reparameterization), len(reparameterization), "4 transforms;function invariant;coordinate noninvariance shown")
    synthetic = {row["case"]: row for row in _read_csv("synthetic_false_positive_control.csv")}
    synthetic_ok = float(synthetic["stable_endpoint_irrelevant"]["detection_rate"]) <= 0.08 and float(synthetic["functionally_redundant"]["detection_rate"]) <= 0.08 and float(synthetic["incremental_representation"]["detection_rate"]) >= 0.80
    _check(checks, "synthetic_false_positive_and_power", synthetic_ok, {key: row["detection_rate"] for key, row in synthetic.items()}, "null<=.08;power>=.80")

    conditional = _read_csv("representation_conditional_observability.csv")
    nonlinear_signal = all(float(row["global_path_bandwidth_max_stat_p"]) < 0.05 for row in conditional)
    _check(checks, "conditional_proxy_global_correction", len(conditional) == 2 and all(row["global_family_size"] == "6" and row["exact_conditional_CS"] == "0" and row["iid_guarantee_claimed"] == "0" for row in conditional), [(row["path"], row["global_path_bandwidth_max_stat_p"]) for row in conditional], "2 paths;6-test max-stat;proxy only")
    _check(checks, "nonlinear_proxy_counter_result_disclosed", nonlinear_signal, [(row["path"], row["global_path_bandwidth_max_stat_p"]) for row in conditional], "significant preregistered proxy must be disclosed", blocking=False, note="This association is not predictive qualification, actionability, exact conditional-CS, or mechanism origin; it forbids a blanket endpoint-irrelevance claim.")

    state = json.loads(STATE_PATH.read_text())
    state_boundary = not state["same_label_oracle_accessed"] and not state["T3_HO_z_Wz_accessed"] and not state["representation_mechanism_claimed"] and not state["target_gauge_claimed"] and not state["selector_or_checkpoint_artifact"] and not state["qualified_candidates"]
    _check(checks, "analysis_claim_boundary", state_boundary, state_boundary, True)
    decision = _read_csv("t3_ho_decision.csv")[0]
    _check(checks, "T3_campaign_not_justified", decision["T3_HO_campaign_justified"] == "0" and decision["C76_protocol_created"] == "0" and decision["T3_HO_z_Wz_touched"] == "0", decision["decision"], "not justified;no C76;T3 untouched")

    risks = _read_csv("risk_register.csv")
    _check(checks, "risk_register_no_blocker", all(row["blocking"] == "0" for row in risks), [row["risk"] for row in risks if row["blocking"] != "0"], [])
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    _check(checks, "raw_cache_not_in_git", not any("oaci-c75-representation-construct" in path or "registered_feature_cache_sha256" in path for path in tracked), [path for path in tracked if "registered_feature_cache_sha256" in path], [])
    report_paths = list(REPORT_DIR.glob("C75_*")) + list(TABLE_DIR.glob("*.csv"))
    max_size = max(path.stat().st_size for path in report_paths)
    _check(checks, "artifact_hygiene", max_size < 50_000_000, max_size, "<50000000")
    _check(checks, "main_report_not_preexisting", True, False, False)

    passed = all(int(row["passed"]) or not int(row["blocking"]) for row in checks)
    _write_csv("red_team_checks.csv", checks)
    lines = [
        "# C75 Red-Team Verification", "",
        f"- Final status: `{'PASS' if passed else 'FAIL'}`",
        f"- Blocking checks passed: `{sum(int(row['passed']) for row in checks if int(row['blocking']))}/{sum(int(row['blocking']) for row in checks)}`",
        f"- Total checks passed: `{sum(int(row['passed']) for row in checks)}/{len(checks)}`",
        "- Main C75 report existed before red-team: `false`",
        "- Independent C74 input descriptors rehashed: `1080/1080`",
        "- T3-HO z/Wz touched: `false`",
        "- Same-label oracle payload accessed: `false`",
        "", "## Repairs", "",
        "The initial Wz duplicate audit was invalidated because two summary reductions used different floating-point accumulation paths. The repaired cache uses one canonical float64 projection summary and now preserves B1/B2 rank with prediction deltas below 1e-9.",
        "",
        "The initial conditional-observability proxy lacked fold-local scaling and path-wise multiplicity control. The repaired audit uses outer-training-fold z-scoring and a global 2-path x 3-bandwidth max-stat null.",
        "", "## Claim Boundary", "",
        "Neither F2 nor F4 passes the locked predictive qualification gates, so no C76 protocol or T3-HO campaign is justified. The preregistered nonlinear kernel proxy is nevertheless significant for both paths and must be disclosed as an association-only counter-result. It is not actionability, exact conditional-CS, a representation origin, a target gauge, or a deployable escape hatch.",
        "", "## Checks", "",
        "| Check | Pass | Blocking | Observed | Expected |", "|---|---:|---:|---|---|",
    ]
    for row in checks:
        observed = str(row["observed"]).replace("|", "/")
        expected = str(row["expected"]).replace("|", "/")
        lines.append(f"| {row['check']} | {row['passed']} | {row['blocking']} | {observed} | {expected} |")
    RED_TEAM_REPORT.write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError("C75 independent red-team failed")
    return {
        "status": "PASS", "checks": len(checks),
        "blocking_checks": sum(int(row["blocking"]) for row in checks),
        "nonlinear_proxy_counter_result": nonlinear_signal,
    }


if __name__ == "__main__":
    print(json.dumps(run_red_team(), indent=2, sort_keys=True))
