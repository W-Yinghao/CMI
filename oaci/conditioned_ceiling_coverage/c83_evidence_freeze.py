"""Build and validate the C83P compact evidence freeze without new outcomes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c83p_tables"
C80_TABLE_DIR = REPORT_DIR / "c80e_tables"
C81_TABLE_DIR = REPORT_DIR / "c81p_tables"
C82_TABLE_DIR = REPORT_DIR / "c82e_tables"

C82_BASE_HEAD = "34ae76a4a588059ab5a6d82b8116088a14af4ad5"
C82_PROTOCOL_COMMIT = "8b0df50b3707dbb3af4a459b6dc6de36c97d562f"
C82_PROTOCOL_SHA256 = "9f58c7a8e6b495a6d8f510c0d72d24ede4485908ef94bc078abe8f124b03a8f3"
C82_LOCK_COMMIT = "6c6739c61d362bc33df6d8b016e4cda724772a62"
C82_LOCK_SHA256 = "d5de6d6ff242b9f3d7f9c318cbdd6e1e16c509060bc14cca59292b738a75f5ce"
C82_RESULT_SHA256 = "d8060e6636adf7fcca7a0ace0e47bb7043676b7681569e09fb8705dcb8d5a8b7"
C82_MANIFEST_SHA256 = "910e2ff1d8445dae262be82d417140cd44fc48be1306f2bbe5a439ec3549f0a2"
C82_GATE = "C82-D_zero_label_comparison_training_seed_method_identity_or_target_heterogeneous"
C81_GATE = "C81-E_protocol_input_implementation_or_provenance_blocker"
C83_GATE = "C83_AAAI_EVIDENCE_CLAIM_FIGURE_TABLE_FREEZE_READY_FOR_MANUSCRIPT_AUTHORIZATION"

PRIMARY_ZERO_METHODS = ("U7", "U5", "U11", "U13", "U14", "U15")
FIGURE3_CONTEXT_METHODS = (
    "B0", "B1", "B2", "B3", "B4O", "B4S", "S1",
    *PRIMARY_ZERO_METHODS, "L1", "L7", "B5",
)
SEEDS = (3, 4)


class C83EvidenceError(RuntimeError):
    """Raised when a committed evidence identity or claim contract drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows:
        raise C83EvidenceError(f"refusing to write empty registry: {path}")
    fieldnames = list(fields or rows[0].keys())
    for index, row in enumerate(rows):
        if set(row) != set(fieldnames):
            raise C83EvidenceError(f"schema mismatch in {path.name} row {index}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def csv_lookup(path: Path, key: Mapping[str, Any]) -> dict[str, str]:
    matches = [
        row for row in read_csv(path)
        if all(row.get(field) == str(value) for field, value in key.items())
    ]
    if len(matches) != 1:
        raise C83EvidenceError(f"expected one row in {relative(path)} for {key}, got {len(matches)}")
    return matches[0]


def row_key(key: Mapping[str, Any]) -> str:
    return "|".join(f"{field}={value}" for field, value in key.items())


def json_pointer(value: Any, pointer: str) -> Any:
    current = value
    for token in pointer.strip("/").split("/") if pointer != "/" else []:
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=check, capture_output=True, text=True,
    )


def _number_base(
    number_id: str,
    value: Any,
    *,
    units: str,
    seed: str = "ALL",
    method: str = "NA",
    endpoint: str,
    aggregation: str,
    cluster: str,
    source_artifact: str,
    source_row_key: str,
    source_field: str,
    rounding_rule: str,
    allowed_interpretation: str,
) -> dict[str, str]:
    return {
        "number_id": number_id,
        "value": str(value),
        "units": units,
        "seed": str(seed),
        "method": method,
        "endpoint": endpoint,
        "aggregation": aggregation,
        "cluster": cluster,
        "source_artifact": source_artifact,
        "source_row_key": source_row_key,
        "source_field": source_field,
        "rounding_rule": rounding_rule,
        "allowed_interpretation": allowed_interpretation,
    }


def number_from_csv(
    rows: list[dict[str, str]],
    number_id: str,
    path: Path,
    key: Mapping[str, Any],
    field: str,
    **metadata: str,
) -> None:
    source = csv_lookup(path, key)
    rows.append(_number_base(
        number_id, source[field], source_artifact=relative(path),
        source_row_key=row_key(key), source_field=field, **metadata,
    ))


def number_from_json(
    rows: list[dict[str, str]],
    number_id: str,
    path: Path,
    pointer: str,
    **metadata: str,
) -> None:
    source = read_json(path)
    value = json_pointer(source, pointer)
    if isinstance(value, bool):
        value = int(value)
    rows.append(_number_base(
        number_id, value, source_artifact=relative(path),
        source_row_key=pointer, source_field="@json_pointer", **metadata,
    ))


def number_from_row_count(
    rows: list[dict[str, str]],
    number_id: str,
    path: Path,
    **metadata: str,
) -> None:
    rows.append(_number_base(
        number_id, len(read_csv(path)), source_artifact=relative(path),
        source_row_key="ALL_ROWS", source_field="@row_count", **metadata,
    ))


def build_replay_tables() -> None:
    commit_chain = [
        ("C82_protocol", C82_PROTOCOL_COMMIT),
        ("C82_analysis_lock", C82_LOCK_COMMIT),
        ("C82_authorization", "5644157ff20d519db37e5061f773875131453041"),
        ("C82_pre_adapter_attempt", "24f5ee1717a5c896d697a2079a33fc7a3cb15268"),
        ("C82_result_freeze", "ce0564d7a5f56a78d7fd210ef6c3be8edb351cfd"),
        ("C82_scientific_red_team", "61b3fe24ddb502d963b77e5468fe80ad97bb26d7"),
        ("C82_report", "d4c035d80de8c1ed1892f83d470296e89de74a06"),
        ("C82_regression", "b88c757606099b2018bdfed279c7824281489872"),
        ("C82_final_red_team", "35f87a2ed5211bcecef93110fec1dc8a4cbdbc08"),
        ("C82_final_handoff", C82_BASE_HEAD),
        ("C82_PM_addendum", "5cee693132bc950d7e0ad9c3c9028e7cb1fcfcf7"),
    ]
    commit_rows = []
    previous = None
    for object_id, commit in commit_chain:
        reachable = git("cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode == 0
        ancestor = git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode == 0
        chronological = True if previous is None else (
            git("merge-base", "--is-ancestor", previous, commit, check=False).returncode == 0
        )
        commit_rows.append({
            "object_id": object_id,
            "expected_commit": commit,
            "observed_commit": git("rev-parse", commit).stdout.strip() if reachable else "UNREACHABLE",
            "reachable": int(reachable),
            "ancestor_of_current_HEAD": int(ancestor),
            "chronology_from_previous": int(chronological),
            "status": "PASS" if reachable and ancestor and chronological else "FAIL",
        })
        previous = commit
    write_csv(TABLE_DIR / "c82_commit_chain_replay.csv", commit_rows)

    identity_checks = [
        ("protocol_sha256", C82_PROTOCOL_SHA256, sha256_file(REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY_PROTOCOL.json")),
        ("analysis_lock_sha256", C82_LOCK_SHA256, sha256_file(REPORT_DIR / "C82_ANALYSIS_EXECUTION_LOCK.json")),
        ("result_sha256", C82_RESULT_SHA256, sha256_file(REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY.json")),
        ("artifact_manifest_sha256", C82_MANIFEST_SHA256, sha256_file(C82_TABLE_DIR / "result_artifact_manifest.json")),
        ("authorization_record_sha256", "8e55a9e2b091208598a3b2d87e4e0acf9867b73561046039dac8a550a629ba3a", sha256_file(REPORT_DIR / "C82E_PI_AUTHORIZATION_RECORD.json")),
        ("authorization_consumption_sha256", "e336d8afb38140242108b60fe43846c7ff4b374227403bd602ac5079b31f93e0", sha256_file(REPORT_DIR / "C82E_AUTHORIZATION_CONSUMPTION_RECORD.json")),
        ("PM_addendum_sha256", "168f125a4316e1d15303bc689e18405988b104e7eb61a867d1c7c5ddac522b69", sha256_file(REPORT_DIR / "C82E_PM_GITHUB_AUDIT_ADDENDUM.json")),
    ]
    identity_rows = [{
        "object_id": object_id,
        "expected": expected,
        "observed": observed,
        "pass": int(expected == observed),
    } for object_id, expected, observed in identity_checks]
    result = read_json(REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY.json")
    for object_id, expected, observed in (
        ("primary_taxonomy", C82_GATE, result["primary_taxonomy"]),
        ("historical_C81_gate", C81_GATE, result["C81_gate_unchanged"]),
        ("method_context_rows", "672", str(result["method_context_rows"])),
        ("registered_tables", "23", str(result["artifact_manifest_table_count"])),
        ("LOTO_preserved", "7", str(result["LOTO_preserved"])),
        ("LOTO_total", "16", str(result["LOTO_total"])),
    ):
        identity_rows.append({
            "object_id": object_id, "expected": expected, "observed": observed,
            "pass": int(expected == observed),
        })
    write_csv(TABLE_DIR / "c82_result_identity_replay.csv", identity_rows)

    manifest = read_json(C82_TABLE_DIR / "result_artifact_manifest.json")
    manifest_rows = []
    for artifact in manifest["artifacts"]:
        path = C82_TABLE_DIR / artifact["path"]
        observed_rows = len(read_csv(path))
        observed_sha = sha256_file(path)
        manifest_rows.append({
            "path": relative(path),
            "expected_sha256": artifact["sha256"],
            "observed_sha256": observed_sha,
            "expected_rows": artifact["rows"],
            "observed_rows": observed_rows,
            "hash_pass": int(observed_sha == artifact["sha256"]),
            "row_count_pass": int(observed_rows == artifact["rows"]),
        })
    write_csv(TABLE_DIR / "c82_table_manifest_replay.csv", manifest_rows)

    accepted = [row for row in read_csv(C82_TABLE_DIR / "regression_attempt_ledger.csv") if row["accepted"] == "1"]
    regression_rows = [{
        "job_id": row["job_id"], "suite": row["suite"],
        "commit_under_test": row["commit_under_test"], "environment": row["environment"],
        "CPUs": row["CPUs"], "GPU": row["GPU"], "passed": row["passed"],
        "failed": row["failed"], "skipped": row["skipped"], "deselected": row["deselected"],
        "stderr_bytes": row["stderr_bytes"], "status": "PASS" if row["status"] == "PASS" and row["stderr_bytes"] == "0" else "FAIL",
    } for row in accepted]
    write_csv(TABLE_DIR / "c82_regression_replay.csv", regression_rows)


def claim_rows() -> list[dict[str, str]]:
    claims = [
        ("C1", "Strict-source S1 has high regret on the frozen field.", "L5_actionability", "IS", "standardized_regret", "frozen_seed3_seed4_field", 1, "oaci/reports/c82e_tables/seed_specific_method_results.csv", "Field-specific comparison against registered controls and Q0 only.", "Universal source-only failure; intrinsic weakness of balanced accuracy.", "main"),
        ("C2", "COTT materially improves S1 on seed 3 under the registered Q1 gate.", "L5_actionability", "ISU", "standardized_regret", "seed3_full_panel", 1, "oaci/reports/c82e_tables/seed_method_Q1_Q2.csv", "Exact C82 max-T gate; COTT/U13; one training seed.", "Cross-seed success; deployability; COTT superiority in general.", "main"),
        ("C3", "COTT retains positive Q1 direction on seed 4 but the exact Q1 gate is inactive.", "L5_actionability", "ISU", "standardized_regret", "seed4_full_panel", 1, "oaci/reports/c82e_tables/seed_method_Q1_Q2.csv", "Positive material mean, 7/8 favorable targets, max-T p=0.101167; no sign reversal claim.", "Seed4 Q1 pass; negative directional effect; general COTT failure.", "main"),
        ("C4", "No primary zero-label method passes Q0 B=1 noninferiority on either seed.", "L5_actionability", "IU_ISU_vs_ILc", "standardized_regret_noninferiority", "two_seed_full_panels", 1, "oaci/reports/c82e_tables/seed_method_Q1_Q2.csv", "Applies to six locked primary zero-label representatives and margin 0.05.", "Universal zero-label impossibility; universal one-label sufficiency.", "main"),
        ("C5", "No fixed zero-label method supports a common cross-seed A or B result.", "L5_actionability", "IU_ISU", "same_method_cross_seed_stability", "paired_training_seeds", 1, "oaci/reports/c82e_tables/cross_seed_qualifying_method_intersection.csv", "Seed categories are B/C; both A and B intersections are empty.", "Independent population replication; method-family interchangeability.", "main"),
        ("C6", "Measurement association, regret, top-k localization, and target robustness yield distinct conclusions.", "evidence_ladder", "mixed", "measurement_vs_decision", "frozen_field", 1, "oaci/reports/c82e_tables/measurement_vs_decision_separation.csv;oaci/reports/c82e_tables/primary_method_topk_table.csv", "No endpoint substitutes for another; target is the principal cluster.", "Mechanism; causality; high association implies actionability.", "main"),
        ("C7", "Construction-label information yields a lower-regret frontier on this frozen field.", "L5_actionability", "ILc", "standardized_regret", "C80_existing_field", 1, "oaci/reports/c80e_tables/seed3_budget_frontier.csv;oaci/reports/c80e_tables/seed4_budget_frontier.csv", "Exact Q0 passive policy, frozen split, source-relative materiality.", "Universal minimal budget; deployment-free labels; active-policy optimality.", "main"),
        ("C8", "C80 B*=1 is a full-panel source-relative result with a 2-4 LOTO envelope.", "L5_actionability", "ILc", "Bstar_and_LOTO", "C80_existing_field", 1, "oaci/reports/c80e_tables/seed_specific_bstar.csv;oaci/reports/c80e_tables/leave_one_target_out_sensitivity.csv", "All 16 LOTO panels move B* to 2 or 4; FULL remains cell-specific.", "Universal one-label sufficiency; target-population stability.", "main"),
        ("C9", "C81 is blocked and C82 is a post-C81-outcome-access recovery.", "provenance", "mixed", "result_identity", "C81_C82_lifecycle", 1, "oaci/reports/C81_OVERALL_REPORT_GITHUB_AUDIT_ADDENDUM.json;oaci/reports/C82_POST_C81_BASELINE_RECOVERY.json", "C82 does not retroactively change C81-E.", "C81 scientific result; untouched C82 confirmation.", "main"),
        ("C10", "All C80/C82 conclusions are dataset-, field-, split-, and policy-specific.", "external_validity", "mixed", "scope", "BNCI2014_001_existing_field", 1, "oaci/reports/C82_POST_C81_BASELINE_RECOVERY.json;oaci/reports/C80_LABEL_BUDGET_FRONTIER.json", "Same targets/trials and two paired training seeds; no external dataset.", "External validity; new-subject generalization; deployment claim.", "main"),
    ]
    forbidden = [
        ("F1", "Universal zero-label impossibility"),
        ("F2", "Universal one-label sufficiency"),
        ("F3", "External validity or new-subject generalization"),
        ("F4", "Deployability"),
        ("F5", "Causal representation mechanism"),
        ("F6", "COTT failure in general"),
        ("F7", "OACI or SRC rescue"),
        ("F8", "Cross-regime selector transport"),
        ("F9", "Information-theoretic minimum label budget"),
        ("F10", "Training seed as independent population replication"),
    ]
    fields = (
        "claim_id", "claim_text_short", "evidence_level", "information_class",
        "decision_objective", "data_scope", "supported", "supporting_artifacts",
        "required_qualifiers", "forbidden_expansion", "main_text_or_supplement",
    )
    rows = [dict(zip(fields, claim)) for claim in claims]
    rows.extend({
        "claim_id": claim_id,
        "claim_text_short": text,
        "evidence_level": "forbidden_expansion",
        "information_class": "ALL",
        "decision_objective": "claim_boundary",
        "data_scope": "outside_supported_scope",
        "supported": "0",
        "supporting_artifacts": "oaci/reports/C82_POST_C81_BASELINE_RECOVERY.json",
        "required_qualifiers": "Must remain explicitly unsupported.",
        "forbidden_expansion": text,
        "main_text_or_supplement": "claim_scanner",
    } for claim_id, text in forbidden)
    return rows


def build_number_registry() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    integer = "exact_integer_from_committed_artifact"
    exact = "preserve_full_artifact_precision; display_six_decimals_half_even"
    rate = "preserve_full_artifact_precision; display_four_decimals_half_even"
    common = dict(cluster="target", allowed_interpretation="Frozen-field, target-cluster summary only.")

    c81_protocol = REPORT_DIR / "C81_AAAI_BASELINE_COMPARISON_PROTOCOL.json"
    c81_ready = REPORT_DIR / "C81P_PROTOCOL_READINESS.json"
    c82_result = REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY.json"
    c82_identity = REPORT_DIR / "C82E_EXECUTION_RESULT_IDENTITY.json"
    for number_id, path, pointer, endpoint, units in (
        ("universe_primary_candidates", c81_protocol, "/candidate_universe/primary_candidates", "candidate_universe", "candidate_units"),
        ("universe_contexts", c81_protocol, "/candidate_universe/contexts", "contexts", "contexts"),
        ("universe_candidates_per_context", c81_protocol, "/candidate_universe/candidates_per_context", "candidates_per_context", "candidates"),
        ("registered_methods", c81_ready, "/registered_methods", "method_registry_size", "methods"),
        ("primary_zero_label_methods", c81_ready, "/primary_zero_label_representatives", "primary_zero_label_methods", "methods"),
        ("selection_methods", c82_result, "/selection_methods", "selection_methods", "methods"),
        ("canonical_result_rows", c82_result, "/method_context_rows", "canonical_rows", "rows"),
        ("registered_result_tables", c82_result, "/artifact_manifest_table_count", "result_tables", "tables"),
        ("c82_loto_preserved", c82_result, "/LOTO_preserved", "global_method_aware_LOTO", "panels"),
        ("c82_loto_total", c82_result, "/LOTO_total", "global_method_aware_LOTO", "panels"),
        ("c82_evaluation_views", c82_identity, "/view_access/target_evaluation_views", "view_access", "views"),
        ("c82_evaluation_label_rows", c82_identity, "/view_access/target_evaluation_label_rows", "view_access", "rows"),
    ):
        number_from_json(rows, number_id, path, pointer, units=units, seed="ALL", method="NA",
                         endpoint=endpoint, aggregation="committed_identity", rounding_rule=integer, **common)

    for seed in SEEDS:
        key = {"seed": seed}
        number_from_csv(rows, f"c80_seed{seed}_Bstar", C80_TABLE_DIR / "seed_specific_bstar.csv", key, "Bstar",
                        units="ordinal_budget", seed=str(seed), method="Q0", endpoint="Bstar",
                        aggregation="full_eight_target_panel", rounding_rule=integer, **common)
    for source in read_csv(C80_TABLE_DIR / "leave_one_target_out_sensitivity.csv"):
        seed, target = source["seed"], source["left_out_target"]
        key = {"seed": seed, "left_out_target": target}
        number_from_csv(rows, f"c80_loto_seed{seed}_target{target}_Bstar", C80_TABLE_DIR / "leave_one_target_out_sensitivity.csv", key, "Bstar",
                        units="ordinal_budget", seed=seed, method="Q0", endpoint="LOTO_Bstar",
                        aggregation=f"leave_target_{target}_out", rounding_rule=integer, **common)

    frontier_fields = (
        ("expected_standardized_regret", "standardized_regret", exact),
        ("mean_regret_reduction_vs_source", "source_relative_gain", exact),
        ("positive_targets", "positive_targets", integer),
        ("catastrophic_target", "catastrophic_target", integer),
        ("maxT_p", "maxT_p", exact),
        ("direct_qualification", "direct_qualification", integer),
        ("closure_qualification", "closure_qualification", integer),
        ("simultaneous_band_low", "simultaneous_band_low", exact),
        ("simultaneous_band_high", "simultaneous_band_high", exact),
    )
    topk_fields = tuple((field, field, rate) for field in (
        "top1", "top5", "top10", "coverage_top1", "coverage_top5", "coverage_top10",
    ))
    for seed in SEEDS:
        frontier_path = C80_TABLE_DIR / f"seed{seed}_budget_frontier.csv"
        for source in read_csv(frontier_path):
            budget = source["budget"]
            key = {"seed": seed, "budget": budget}
            for field, endpoint, rounding in frontier_fields:
                number_from_csv(rows, f"c80_seed{seed}_budget{budget}_{field}", frontier_path, key, field,
                                units="proportion" if field != "positive_targets" and field != "catastrophic_target" else "targets",
                                seed=str(seed), method="Q0", endpoint=endpoint,
                                aggregation=f"budget_{budget}_full_panel", rounding_rule=rounding, **common)
            topk_path = C80_TABLE_DIR / "topk_coverage_summary.csv"
            for field, endpoint, rounding in topk_fields:
                number_from_csv(rows, f"c80_seed{seed}_budget{budget}_{field}", topk_path, key, field,
                                units="proportion", seed=str(seed), method="Q0", endpoint=endpoint,
                                aggregation=f"budget_{budget}_full_panel", rounding_rule=rounding, **common)

    method_path = C82_TABLE_DIR / "seed_specific_method_results.csv"
    method_fields = (
        ("mean_standardized_regret", exact), ("mean_selected_utility", exact),
        ("source_relative_regret_gain", exact), ("top1", rate), ("top5", rate),
        ("top10", rate), ("coverage_top1", rate), ("coverage_top5", rate),
        ("coverage_top10", rate), ("target_count", integer), ("context_count", integer),
    )
    for source in read_csv(method_path):
        seed, method = source["seed"], source["method_id"]
        key = {"seed": seed, "method_id": method}
        for field, rounding in method_fields:
            units = "targets" if field == "target_count" else "contexts" if field == "context_count" else "proportion"
            number_from_csv(rows, f"c82_seed{seed}_{method}_{field}", method_path, key, field,
                            units=units, seed=seed, method=method, endpoint=field,
                            aggregation="full_eight_target_panel", rounding_rule=rounding, **common)

    q_path = C82_TABLE_DIR / "seed_method_Q1_Q2.csv"
    q_fields = (
        ("target_count", integer, "targets"),
        ("mean_regret_improvement_vs_source", exact, "proportion"),
        ("Q1_simultaneous_lower", exact, "proportion"),
        ("Q1_maxT_p", exact, "p_value"),
        ("Q1_favorable_targets", integer, "targets"),
        ("Q1_worst_target", exact, "proportion"),
        ("Q1_pass", integer, "binary"),
        ("mean_regret_difference_vs_Q0_B1", exact, "proportion"),
        ("Q2_simultaneous_upper", exact, "proportion"),
        ("Q2_maxT_p", exact, "p_value"),
        ("Q2_favorable_targets", integer, "targets"),
        ("Q2_worst_target", exact, "proportion"),
        ("Q2_pass", integer, "binary"),
    )
    for source in read_csv(q_path):
        seed, method = source["seed"], source["method_id"]
        key = {"seed": seed, "method_id": method}
        for field, rounding, units in q_fields:
            number_from_csv(rows, f"c82_seed{seed}_{method}_q1q2_{field}", q_path, key, field,
                            units=units, seed=seed, method=method, endpoint=field,
                            aggregation="target_cluster_maxT_family", rounding_rule=rounding, **common)

    measurement_path = C82_TABLE_DIR / "measurement_vs_decision_separation.csv"
    for source in read_csv(measurement_path):
        seed, method = source["seed"], source["method_id"]
        key = {"seed": seed, "method_id": method}
        for field in ("mean_spearman", "mean_pairwise_order_accuracy"):
            number_from_csv(rows, f"c82_seed{seed}_{method}_{field}", measurement_path, key, field,
                            units="coefficient" if field == "mean_spearman" else "proportion",
                            seed=seed, method=method, endpoint=field, aggregation="mean_within_context",
                            rounding_rule=exact, **common)

    q5_path = C82_TABLE_DIR / "information_class_summary_Q5.csv"
    for source in read_csv(q5_path):
        seed, info = source["seed"], source["information_class"]
        key = {"seed": seed, "information_class": info}
        number_from_csv(rows, f"c82_seed{seed}_Q5_{info}_descriptive_regret", q5_path, key,
                        "best_mean_standardized_regret", units="proportion", seed=seed,
                        method=source["best_registered_method"], endpoint="descriptive_best_within_fixed_class_regret",
                        aggregation="outcome_selected_descriptive_minimum", rounding_rule=exact, **common)

    regression_path = C82_TABLE_DIR / "regression_attempt_ledger.csv"
    for source in read_csv(regression_path):
        if source["accepted"] != "1":
            continue
        key = {"job_id": source["job_id"]}
        number_from_csv(rows, f"c82_regression_{source['suite']}_passed", regression_path, key, "passed",
                        units="tests", endpoint="verification_count", seed="ALL", method="NA",
                        aggregation=source["suite"], rounding_rule=integer, **common)

    for number_id, path, aggregation in (
        ("c82_scientific_red_team_checks", C82_TABLE_DIR / "scientific_result_red_team.csv", "scientific_red_team_59_of_59_pass"),
        ("c82_final_report_red_team_checks", C82_TABLE_DIR / "final_report_red_team.csv", "final_report_red_team_50_of_50_pass"),
    ):
        number_from_row_count(rows, number_id, path, units="checks", seed="ALL", method="NA",
                              endpoint="verification_count", aggregation=aggregation,
                              rounding_rule=integer, **common)

    if len({row["number_id"] for row in rows}) != len(rows):
        raise C83EvidenceError("authoritative number IDs are not unique")
    return rows


def build_figure_contracts(number_index: Mapping[str, Mapping[str, str]]) -> None:
    figure1 = [
        {"element_id": "I0", "axis": "information", "label": "no target information", "predecessor": "NONE", "empirical_scale": "0", "allowed_claim": "information-class label", "forbidden_claim": "ordinal performance guarantee"},
        {"element_id": "IS", "axis": "information", "label": "strict-source information", "predecessor": "I0", "empirical_scale": "0", "allowed_claim": "source-only selection class", "forbidden_claim": "target transport"},
        {"element_id": "IU_ISU", "axis": "information", "label": "target-unlabeled or source-calibrated unlabeled", "predecessor": "IS", "empirical_scale": "0", "allowed_claim": "zero-target-label classes", "forbidden_claim": "homogeneous method family"},
        {"element_id": "ILc", "axis": "information", "label": "independent target-construction labels", "predecessor": "IU_ISU", "empirical_scale": "0", "allowed_claim": "construction-label class", "forbidden_claim": "deployment-free information"},
        {"element_id": "L1", "axis": "evidence", "label": "reliability", "predecessor": "NONE", "empirical_scale": "0", "allowed_claim": "distinct evidence level", "forbidden_claim": "prediction"},
        {"element_id": "L2", "axis": "evidence", "label": "association", "predecessor": "L1", "empirical_scale": "0", "allowed_claim": "distinct evidence level", "forbidden_claim": "causality"},
        {"element_id": "L3", "axis": "evidence", "label": "prediction", "predecessor": "L2", "empirical_scale": "0", "allowed_claim": "distinct evidence level", "forbidden_claim": "transport"},
        {"element_id": "L4", "axis": "evidence", "label": "transport", "predecessor": "L3", "empirical_scale": "0", "allowed_claim": "distinct evidence level", "forbidden_claim": "actionability"},
        {"element_id": "L5", "axis": "evidence", "label": "actionability", "predecessor": "L4", "empirical_scale": "0", "allowed_claim": "decision endpoint", "forbidden_claim": "deployability"},
    ]
    write_csv(TABLE_DIR / "figure_1_contract.csv", figure1)

    figure2 = [
        ("training_seeds", "2", "seeds", "paired training factor", "oaci/reports/C81_AAAI_BASELINE_COMPARISON_PROTOCOL.json", "not independent populations"),
        ("primary_targets", "8", "targets", "principal scientific clusters", "oaci/reports/C81_AAAI_BASELINE_COMPARISON_PROTOCOL.json", "target4 excluded"),
        ("levels", "2", "levels", "repeated within target", "oaci/reports/C81_AAAI_BASELINE_COMPARISON_PROTOCOL.json", "not independent subjects"),
        ("candidates_per_context", "81", "candidates", "fixed candidate universe", "oaci/reports/C81_AAAI_BASELINE_COMPARISON_PROTOCOL.json", "1 ERM plus 40 OACI plus 40 SRC"),
        ("contexts", "32", "contexts", "seed-target-level contexts", "oaci/reports/C82_POST_C81_BASELINE_RECOVERY.json", "not scientific N"),
        ("selection_freeze", "1", "barrier", "selection precedes evaluation", "oaci/reports/C82E_EXECUTION_RESULT_IDENTITY.json", "selection recomputation forbidden"),
        ("physical_disjointness", "1", "contract", "construction/evaluation split", "oaci/reports/C82_ANALYSIS_EXECUTION_LOCK.json", "evaluation never enters selector"),
        ("target4_primary", "0", "rows", "engineering-only exclusion", "oaci/reports/C82E_EXECUTION_RESULT_IDENTITY.json", "no primary use"),
        ("oracle_access", "0", "events", "same-label oracle closure", "oaci/reports/C82E_EXECUTION_RESULT_IDENTITY.json", "descriptive ceiling B5 is evaluation denominator, not oracle selector"),
    ]
    fields = ("element_id", "value", "units", "role", "source_artifact", "qualifier")
    write_csv(TABLE_DIR / "figure_2_contract.csv", [dict(zip(fields, row)) for row in figure2])

    methods = {row["id"]: row for row in read_json(REPORT_DIR / "C81_BASELINE_METHOD_REGISTRY.json")["methods"]}
    q5_best = {(row["seed"], row["best_registered_method"]): row["information_class"] for row in read_csv(C82_TABLE_DIR / "information_class_summary_Q5.csv")}
    fig3: list[dict[str, str]] = []
    for seed in SEEDS:
        for method in FIGURE3_CONTEXT_METHODS:
            if method in ("L1", "L7"):
                budget = "1" if method == "L1" else "FULL"
                number_id = f"c80_seed{seed}_budget{budget}_expected_standardized_regret"
                value = number_index[number_id]["value"]
            else:
                number_id = f"c82_seed{seed}_{method}_mean_standardized_regret"
                value = number_index[number_id]["value"]
            fig3.append({
                "figure_id": "3", "panel": "A_regret", "seed": str(seed),
                "method_id": method, "method_label": methods[method]["name"],
                "information_class": methods[method]["family"], "metric": "standardized_regret",
                "value": value, "number_id": number_id,
                "display_role": "descriptive_best_within_fixed_class" if (str(seed), method) in q5_best else "fixed_registered_method",
                "claim_boundary": "lower_is_better; B5_ceiling_only; descriptive_best_is_not_inferential_winner",
            })
        for method in ("S1", "U13", "L1"):
            for metric in ("top1", "top5", "top10"):
                if method == "L1":
                    number_id = f"c80_seed{seed}_budget1_{metric}"
                else:
                    number_id = f"c82_seed{seed}_{method}_{metric}"
                fig3.append({
                    "figure_id": "3", "panel": "B_topk", "seed": str(seed),
                    "method_id": method, "method_label": methods[method]["name"],
                    "information_class": methods[method]["family"], "metric": metric,
                    "value": number_index[number_id]["value"], "number_id": number_id,
                    "display_role": "fixed_registered_method",
                    "claim_boundary": "localization_endpoint_does_not_override_regret_or_stability",
                })
    write_csv(TABLE_DIR / "figure_3_data.csv", fig3)

    fig4: list[dict[str, str]] = []
    for seed in SEEDS:
        for budget in ("1", "2", "4", "8", "16", "32", "FULL"):
            for metric, suffix in (("standardized_regret", "expected_standardized_regret"), ("closure_qualification", "closure_qualification")):
                number_id = f"c80_seed{seed}_budget{budget}_{suffix}"
                fig4.append({
                    "figure_id": "4", "panel": "A_budget_frontier", "object_id": f"seed{seed}_budget{budget}",
                    "seed": str(seed), "target": "ALL", "budget": budget, "metric": metric,
                    "value": number_index[number_id]["value"], "number_id": number_id,
                    "qualifier": "FULL_is_cell_specific; no_interpolation_between_32_and_FULL",
                })
        number_id = f"c80_seed{seed}_Bstar"
        fig4.append({
            "figure_id": "4", "panel": "B_full_panel_Bstar", "object_id": f"seed{seed}_Bstar",
            "seed": str(seed), "target": "ALL", "budget": number_index[number_id]["value"],
            "metric": "Bstar", "value": number_index[number_id]["value"], "number_id": number_id,
            "qualifier": "full_panel_source_relative_closure_rule",
        })
    for source in read_csv(C80_TABLE_DIR / "leave_one_target_out_sensitivity.csv"):
        number_id = f"c80_loto_seed{source['seed']}_target{source['left_out_target']}_Bstar"
        fig4.append({
            "figure_id": "4", "panel": "C_C80_LOTO", "object_id": f"seed{source['seed']}_leave{source['left_out_target']}",
            "seed": source["seed"], "target": source["left_out_target"], "budget": source["Bstar"],
            "metric": "LOTO_Bstar", "value": source["Bstar"], "number_id": number_id,
            "qualifier": "descriptive_sensitivity; envelope_2_to_4",
        })
    stability = csv_lookup(C82_TABLE_DIR / "cross_seed_method_identity_stability.csv", {})
    for seed, category in ((3, stability["seed3_category"]), (4, stability["seed4_category"])):
        fig4.append({
            "figure_id": "4", "panel": "D_C82_stability", "object_id": f"seed{seed}_category",
            "seed": str(seed), "target": "ALL", "budget": "NA", "metric": "seed_category",
            "value": category, "number_id": "NA",
            "qualifier": "B=Q1_any_and_no_Q2; C=no_Q1",
        })
    for set_id in ("A_intersection", "B_intersection"):
        source = csv_lookup(C82_TABLE_DIR / "cross_seed_qualifying_method_intersection.csv", {"set_id": set_id})
        fig4.append({
            "figure_id": "4", "panel": "D_C82_stability", "object_id": set_id,
            "seed": "PAIRED", "target": "ALL", "budget": "NA", "metric": "method_intersection_count",
            "value": source["count"], "number_id": "NA",
            "qualifier": f"methods={source['methods']}",
        })
    for metric, number_id, value in (
        ("global_method_aware_LOTO_preserved", "c82_loto_preserved", stability["LOTO_preserved"]),
        ("global_method_aware_LOTO_total", "c82_loto_total", stability["LOTO_total"]),
    ):
        fig4.append({
            "figure_id": "4", "panel": "D_C82_stability", "object_id": metric,
            "seed": "PAIRED", "target": "ALL", "budget": "NA", "metric": metric,
            "value": value, "number_id": number_id,
            "qualifier": "global_cross_seed_common_method_rule; not_per_panel_COTT_Q1_ledger",
        })
    write_csv(TABLE_DIR / "figure_4_data.csv", fig4)


def build_main_tables(number_index: Mapping[str, Mapping[str, str]]) -> None:
    methods = read_json(REPORT_DIR / "C81_BASELINE_METHOD_REGISTRY.json")["methods"]
    table1 = []
    for method in methods:
        if method["id"] in ("B0", "B1", "S1", *PRIMARY_ZERO_METHODS, "L1", "L7", "B5"):
            table1.append({
                "method_id": method["id"], "method": method["name"],
                "information_class": method["family"], "required_views": "|".join(method["views"]) or "NONE",
                "role": method["status"], "registered_deviation": method["deviation"],
                "claim_boundary": "fixed_method_not_best_of_many" if method["id"] != "B5" else "denominator_ceiling_only",
            })
    write_csv(TABLE_DIR / "main_table_1.csv", table1)

    table2 = []
    registry = {row["id"]: row for row in methods}
    for seed in SEEDS:
        for method in ("S1", *PRIMARY_ZERO_METHODS, "L1", "L7"):
            if method in ("L1", "L7"):
                budget = "1" if method == "L1" else "FULL"
                regret_id = f"c80_seed{seed}_budget{budget}_expected_standardized_regret"
                gain_id = f"c80_seed{seed}_budget{budget}_mean_regret_reduction_vs_source"
                top_ids = [f"c80_seed{seed}_budget{budget}_{metric}" for metric in ("top1", "top5", "top10")]
                q1_pass = q2_pass = q1_p = q2_upper = "NA"
            else:
                regret_id = f"c82_seed{seed}_{method}_mean_standardized_regret"
                gain_id = f"c82_seed{seed}_{method}_source_relative_regret_gain"
                top_ids = [f"c82_seed{seed}_{method}_{metric}" for metric in ("top1", "top5", "top10")]
                if method in PRIMARY_ZERO_METHODS:
                    q1_pass = number_index[f"c82_seed{seed}_{method}_q1q2_Q1_pass"]["value"]
                    q2_pass = number_index[f"c82_seed{seed}_{method}_q1q2_Q2_pass"]["value"]
                    q1_p = number_index[f"c82_seed{seed}_{method}_q1q2_Q1_maxT_p"]["value"]
                    q2_upper = number_index[f"c82_seed{seed}_{method}_q1q2_Q2_simultaneous_upper"]["value"]
                else:
                    q1_pass = q2_pass = q1_p = q2_upper = "NA"
            table2.append({
                "seed": seed, "method_id": method, "method": registry[method]["name"],
                "information_class": registry[method]["family"],
                "standardized_regret": number_index[regret_id]["value"], "regret_number_id": regret_id,
                "source_relative_gain": number_index[gain_id]["value"], "gain_number_id": gain_id,
                "Q1_pass": q1_pass, "Q1_maxT_p": q1_p,
                "Q2_pass": q2_pass, "Q2_simultaneous_upper": q2_upper,
                "top1": number_index[top_ids[0]]["value"], "top5": number_index[top_ids[1]]["value"],
                "top10": number_index[top_ids[2]]["value"], "topk_number_ids": "|".join(top_ids),
                "allowed_interpretation": "registered_comparator" if method in ("S1", "L1", "L7") else "fixed_primary_zero_label_method",
            })
    write_csv(TABLE_DIR / "main_table_2.csv", table2)

    table3 = [
        {"milestone": "C78S", "result_commit_or_head": "43a046c", "gate": "SEED3_MIXED_RESULTS_C79_PROTOCOL_REVIEW_REQUIRED", "timing_identity": "prospective_seed3_analysis", "operative_science": "mixed_seed3_H1_to_H6", "claim_boundary": "seed3_exploratory_replication"},
        {"milestone": "C79E", "result_commit_or_head": "dadd166", "gate": "C79-E_seed4_does_not_replicate_either_core_pattern", "timing_identity": "post_seed3_pre_seed4_locked", "operative_science": "training_seed_robustness", "claim_boundary": "directional_concordance_with_compound_gate_failure"},
        {"milestone": "C80E", "result_commit_or_head": "111df25", "gate": "C80-A_stable_low_regret_label_budget_frontier_across_training_seeds", "timing_identity": "post_C79_existing_field", "operative_science": "Q0_label_budget_frontier", "claim_boundary": "full_panel_source_relative_LOTO_sensitive"},
        {"milestone": "C81", "result_commit_or_head": "d64f16b", "gate": C81_GATE, "timing_identity": "blocked_after_evaluation_access", "operative_science": "no_valid_baseline_result", "claim_boundary": "C81_A_B_C_D_unavailable"},
        {"milestone": "C82E", "result_commit_or_head": C82_BASE_HEAD, "gate": C82_GATE, "timing_identity": "post_C81_outcome_access_recovery", "operative_science": "frozen_selection_baseline_comparison", "claim_boundary": "same_field_no_external_validity"},
    ]
    write_csv(TABLE_DIR / "main_table_3.csv", table3)

    supplements = [
        ("S01", "all_34_methods_and_availability", "oaci/reports/C81_BASELINE_METHOD_REGISTRY.json", "34-method fixed registry and feasibility"),
        ("S02", "all_21_context_methods", "oaci/reports/c82e_tables/seed_specific_method_results.csv", "seed-level results for all emitted context methods"),
        ("S03", "target_level_effects", "oaci/reports/c82e_tables/target_level_catastrophic_failures.csv", "target-cluster heterogeneity"),
        ("S04", "all_Q0_budgets_seed3", "oaci/reports/c80e_tables/seed3_budget_frontier.csv", "seven-point C80 curve"),
        ("S05", "all_Q0_budgets_seed4", "oaci/reports/c80e_tables/seed4_budget_frontier.csv", "seven-point C80 curve"),
        ("S06", "measurement_metrics", "oaci/reports/c82e_tables/method_measurement_metrics.csv", "applicability-constrained measurement outputs"),
        ("S07", "selected_regimes", "oaci/reports/c82e_tables/selected_regime_distribution.csv", "descriptive selected-regime distribution"),
        ("S08", "coverage", "oaci/reports/c82e_tables/coverage_summary.csv", "registered top-k coverage"),
        ("S09", "C82_red_team", "oaci/reports/c82e_tables/scientific_result_red_team.csv", "59 scientific checks"),
        ("S10", "C82_provenance", "oaci/reports/C82E_EXECUTION_RESULT_IDENTITY.json", "authorization, views, result identity"),
        ("S11", "repair_history", "oaci/reports/C81_OVERALL_REPORT_GITHUB_AUDIT_ADDENDUM.json", "C81 blocker audit"),
        ("S12", "C82_interpretation_addendum", "oaci/reports/C82E_PM_GITHUB_AUDIT_ADDENDUM.json", "LOTO and Q5 wording limits"),
    ]
    fields = ("table_id", "content_class", "source_artifact", "role")
    write_csv(TABLE_DIR / "supplement_table_registry.csv", [dict(zip(fields, row)) for row in supplements])

    table_registry = [
        {"table_id": "T1", "placement": "main", "data_path": "oaci/reports/c83p_tables/main_table_1.csv", "row_count": len(table1), "scientific_role": "information_classes_views_representatives", "source_identity": "C81_method_registry", "claim_boundary": "method_access_contract_not_outcome_ranking"},
        {"table_id": "T2", "placement": "main", "data_path": "oaci/reports/c83p_tables/main_table_2.csv", "row_count": len(table2), "scientific_role": "seed_specific_primary_baseline_results", "source_identity": "C80_C82_compact_results", "claim_boundary": "fixed_methods_and_exact_gates"},
        {"table_id": "T3", "placement": "main", "data_path": "oaci/reports/c83p_tables/main_table_3.csv", "row_count": len(table3), "scientific_role": "protocol_timing_and_evidence_identity", "source_identity": "committed_milestone_chain", "claim_boundary": "C81_E_and_post_outcome_C82_explicit"},
        {"table_id": "S_registry", "placement": "supplement", "data_path": "oaci/reports/c83p_tables/supplement_table_registry.csv", "row_count": len(supplements), "scientific_role": "supplement_source_map", "source_identity": "C80_C82_tables", "claim_boundary": "no_new_aggregation"},
        {"table_id": "S_fidelity", "placement": "supplement", "data_path": "oaci/reports/c83p_tables/baseline_reference_fidelity_appendix.csv", "row_count": 34, "scientific_role": "method_reference_fidelity", "source_identity": "C81_fixed_registry", "claim_boundary": "declared_adaptations_and_input_unavailability"},
    ]
    write_csv(TABLE_DIR / "table_data_registry.csv", table_registry)

    figure_registry = [
        {"figure_id": "F1", "data_path": "oaci/reports/c83p_tables/figure_1_contract.csv", "row_count": 9, "empirical": 0, "scientific_role": "information_and_evidence_ladders", "source_identity": "claim_contract", "claim_boundary": "conceptual_no_empirical_scale"},
        {"figure_id": "F2", "data_path": "oaci/reports/c83p_tables/figure_2_contract.csv", "row_count": 9, "empirical": 0, "scientific_role": "field_and_view_design", "source_identity": "C81_C82_protocols", "claim_boundary": "physical_design_only"},
        {"figure_id": "F3", "data_path": "oaci/reports/c83p_tables/figure_3_data.csv", "row_count": 50, "empirical": 1, "scientific_role": "regret_and_topk", "source_identity": "authoritative_number_registry", "claim_boundary": "descriptive_I0_best_and_B5_ceiling_flagged"},
        {"figure_id": "F4", "data_path": "oaci/reports/c83p_tables/figure_4_data.csv", "row_count": 52, "empirical": 1, "scientific_role": "budget_and_stability", "source_identity": "authoritative_number_registry", "claim_boundary": "global_LOTO_not_per_panel_COTT_ledger"},
    ]
    write_csv(TABLE_DIR / "figure_data_registry.csv", figure_registry)


def build_baseline_fidelity_appendix() -> None:
    fidelity = {row["method_id"]: row for row in read_csv(C81_TABLE_DIR / "baseline_reference_fidelity.csv")}
    info = {row["method_id"]: row for row in read_csv(C81_TABLE_DIR / "baseline_information_class_registry.csv")}
    views = {row["method_id"]: row for row in read_csv(C81_TABLE_DIR / "baseline_input_view_matrix.csv")}
    registry = read_json(REPORT_DIR / "C81_BASELINE_METHOD_REGISTRY.json")["methods"]
    rows = []
    for method in registry:
        method_id = method["id"]
        rows.append({
            "method_id": method_id, "method": method["name"],
            "information_class": info[method_id]["information_class"],
            "status": method["status"], "canonical_reference": fidelity[method_id]["canonical_url"] or "internal_control",
            "implementation_source": fidelity[method_id]["implementation_source"],
            "fidelity_status": fidelity[method_id]["fidelity_status"],
            "declared_adaptation_or_exclusion": fidelity[method_id]["declared_deviation"],
            "selection_views": "|".join(method["views"]) or "NONE",
            "evaluation_view_for_selection": views[method_id]["target_evaluation_view"],
            "oracle_reachable": views[method_id]["oracle_reachable"],
            "outcome_tuned": fidelity[method_id]["outcome_tuned"],
        })
    write_csv(TABLE_DIR / "baseline_reference_fidelity_appendix.csv", rows)


def build_timing_and_risk() -> None:
    timing = [
        (1, "C78S", "43a046c", "seed3_scientific_result", "seed3_outcomes_available", "prospective_seed3"),
        (2, "C79P", "ec4834c", "post_seed3_seed4_protocol", "before_seed4_checkpoint_outcomes", "outcome_informed_pre_seed4"),
        (3, "C79E", "dadd166", "seed4_training_seed_result", "seed4_outcomes_available", "training_seed_robustness_only"),
        (4, "C80E", "111df25", "existing_field_budget_result", "budget_outcomes_available", "post_C79_existing_field"),
        (5, "C81", "d64f16b", "blocked_baseline_attempt", "evaluation_access_without_result_freeze", "historical_C81_E"),
        (6, "C82P", C82_PROTOCOL_COMMIT, "post_C81_recovery_protocol", "before_new_C82_scoring", "post_outcome_access_recovery"),
        (7, "C82E", C82_BASE_HEAD, "valid_recovery_result", "C82_D_frozen", "existing_field_no_external_validity"),
        (8, "C83P", "5cee693", "evidence_freeze", "no_new_outcomes", "manuscript_readiness_only"),
    ]
    fields = ("order", "milestone", "commit", "event", "outcome_access_boundary", "epistemic_status")
    write_csv(TABLE_DIR / "protocol_timing_diagram.csv", [dict(zip(fields, row)) for row in timing])

    risks = [
        ("R01", "one_EEG_dataset_only", "Main scope sentence", "field/view manifest", "Dataset-specific claim only"),
        ("R02", "two_training_seeds_not_population_replication", "Methods and limitations", "paired-seed registry", "Training-seed robustness only"),
        ("R03", "eight_target_clusters_limit_inference", "Results uncertainty note", "target-level tables", "Target-cluster inference and explicit small-N limit"),
        ("R04", "C82_post_C81_outcome_access_recovery", "Provenance statement", "C81/C82 addenda", "Never call C82 untouched confirmation"),
        ("R05", "C80_Bstar1_full_panel_source_relative_LOTO_sensitive", "C80 result qualifier", "C80 Bstar and LOTO tables", "Report 2-4 LOTO envelope with B*=1"),
        ("R06", "COTT_seed4_direction_positive_gate_inactive", "C82 component result", "seed_method_Q1_Q2", "No sign-reversal wording"),
        ("R07", "S1_weaker_than_some_fixed_no_information_defaults", "Comparator disclosure", "Q5 and method results", "Do not generalize Q1 beyond S1 comparator"),
        ("R08", "Q5_best_within_class_is_descriptive", "Q5 label", "C82 PM addendum", "No inferential winner language"),
        ("R09", "LOTO_7_of_16_is_global_common_method_rule", "Stability definition", "C82 PM addendum", "No per-panel COTT failure claim"),
        ("R10", "five_literature_methods_input_unavailable", "Baseline availability", "C81 method registry", "Do not call registry exhaustive of literature"),
        ("R11", "SND_softmax_output_adaptation", "Method appendix", "reference fidelity appendix", "Disclose output-layer adaptation"),
        ("R12", "MaNo_COTT_ATC_adaptations", "Method appendix", "reference fidelity appendix", "Disclose every registered deviation"),
        ("R13", "no_cross_regime_selector_transport_claim", "Claim boundary", "loro_status.csv", "No LORO or transport claim"),
        ("R14", "no_active_acquisition_comparison", "Scope", "C80/C82 protocols", "Q0 passive policy only"),
        ("R15", "no_external_dataset", "Limitations", "field manifest", "No external validation"),
        ("R16", "regret_improvement_not_high_top1", "Results component table", "primary_method_topk_table", "Show top-k alongside regret"),
        ("R17", "construction_labels_independent_not_deployment_free", "Information-class definition", "physical view contract", "No deployment-free or zero-label label"),
    ]
    risk_fields = ("risk_id", "risk", "main_text_disclosure", "supplement_evidence", "claim_narrowing")
    risk_rows = []
    for row in risks:
        item = dict(zip(risk_fields, row))
        item["new_experiment_required"] = "no_for_C83P"
        item["status"] = "CLOSED_BY_DISCLOSURE_AND_CLAIM_NARROWING"
        risk_rows.append(item)
    write_csv(TABLE_DIR / "reviewer_risk_ledger.csv", risk_rows)


def build_reproducibility_index() -> dict[str, Any]:
    local_items = [
        ("C80_protocol", "protocol", REPORT_DIR / "C80R_ADDITIVE_REPAIR_PROTOCOL.json"),
        ("C80_operational_lock", "execution_lock", REPORT_DIR / "C80R_REPAIRED_ANALYSIS_EXECUTION_LOCK.json"),
        ("C80_result", "result", REPORT_DIR / "C80_LABEL_BUDGET_FRONTIER.json"),
        ("C81_protocol", "protocol", REPORT_DIR / "C81_AAAI_BASELINE_COMPARISON_PROTOCOL.json"),
        ("C81_method_registry", "method_registry", REPORT_DIR / "C81_BASELINE_METHOD_REGISTRY.json"),
        ("C81_audit_addendum", "repair_ledger", REPORT_DIR / "C81_OVERALL_REPORT_GITHUB_AUDIT_ADDENDUM.json"),
        ("C82_protocol", "protocol", REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY_PROTOCOL.json"),
        ("C82_execution_lock", "execution_lock", REPORT_DIR / "C82_ANALYSIS_EXECUTION_LOCK.json"),
        ("C82_authorization", "authorization", REPORT_DIR / "C82E_PI_AUTHORIZATION_RECORD.json"),
        ("C82_result", "result", REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY.json"),
        ("C82_result_manifest", "artifact_manifest", C82_TABLE_DIR / "result_artifact_manifest.json"),
        ("C82_scientific_red_team", "red_team", C82_TABLE_DIR / "scientific_result_red_team.csv"),
        ("C82_final_red_team", "red_team", C82_TABLE_DIR / "final_report_red_team.csv"),
        ("C82_regression", "regression", C82_TABLE_DIR / "regression_attempt_ledger.csv"),
        ("C82_PM_addendum", "interpretation_addendum", REPORT_DIR / "C82E_PM_GITHUB_AUDIT_ADDENDUM.json"),
    ]
    items = [{
        "object_id": object_id, "category": category, "path": relative(path),
        "sha256": sha256_file(path), "bytes": path.stat().st_size,
        "raw_payload_copied": False,
    } for object_id, category, path in local_items]
    identity = read_json(REPORT_DIR / "C82E_EXECUTION_RESULT_IDENTITY.json")
    items.append({
        "object_id": "C82_external_result_root", "category": "external_payload_location",
        "path": identity["external_result"]["directory"],
        "sha256": identity["external_result"]["artifact_manifest_sha256"],
        "bytes": identity["external_result"]["total_bytes"], "raw_payload_copied": False,
    })
    return {
        "schema_version": "c83_aaai_reproducibility_index_v1",
        "milestone": "C83P",
        "no_new_scientific_computation": True,
        "raw_EEG_or_label_arrays_in_git": False,
        "model_weights_or_caches_in_Git": False,
        "items": items,
    }


def build_contract_reports(claims: Sequence[Mapping[str, str]], reproducibility: Mapping[str, Any]) -> None:
    supported = [row for row in claims if row["supported"] == "1"]
    forbidden = [row for row in claims if row["supported"] == "0"]
    write_json(REPORT_DIR / "C83_AAAI_CLAIM_CONTRACT.json", {
        "schema_version": "c83_aaai_claim_contract_v1",
        "milestone": "C83P",
        "scientific_gate": C82_GATE,
        "historical_C81_gate": C81_GATE,
        "manuscript_drafting_authorized": False,
        "new_real_data_statistics": 0,
        "supported_claims": supported,
        "forbidden_claims": forbidden,
    })

    (REPORT_DIR / "C83_AAAI_FIGURE_CONTRACT.md").write_text("""# C83 AAAI Figure Contract

This document freezes data identity and caption boundaries. It is not a
manuscript caption draft and contains no new scientific computation.

## Figure 1

Conceptual information and evidence ladders only. Arrow placement carries no
empirical distance, monotonicity, causality, or sufficiency claim. Data contract:
`c83p_tables/figure_1_contract.csv`.

## Figure 2

Frozen candidate-field and physical-view design: two paired training seeds,
eight primary target clusters, two levels, 81 candidates per context, selection
freeze before evaluation, target4 exclusion, and oracle closure. Data contract:
`c83p_tables/figure_2_contract.csv`.

## Figure 3

Panel A shows committed seed-specific standardized regret for fixed controls,
S1, six primary zero-label methods, Q0 B=1, Q0 FULL, and the ceiling. Panel B
shows top-1/top-5/top-10 for S1, COTT, and Q0 B=1. Any displayed best
no-information default is descriptive, not an inferential winner. Data contract:
`c83p_tables/figure_3_data.csv`.

## Figure 4

Shows the committed C80 curves, full-panel B*=1 values, 2-4 LOTO envelope,
C82 seed categories B/C, empty common A/B method sets, and global method-aware
LOTO preservation 7/16. The 7/16 rule is not a per-panel COTT Q1 ledger. Data
contract: `c83p_tables/figure_4_data.csv`.

No publication figure is rendered in C83P.
""")

    (REPORT_DIR / "C83_AAAI_BASELINE_REFERENCE_FIDELITY_APPENDIX.md").write_text("""# C83 Baseline Reference-Fidelity Appendix Contract

The authoritative 34-method availability, implementation-source, adaptation,
view-access, and reference-fidelity rows are frozen in
`c83p_tables/baseline_reference_fidelity_appendix.csv`. The appendix distinguishes
faithful implementations, declared adaptations, diagnostic-only objects, and
five input-unavailable methods. It does not rank fidelity or add methods.
""")

    (REPORT_DIR / "C83_AAAI_PROTOCOL_TIMING_DIAGRAM_SPEC.md").write_text("""# C83 Protocol-Timing Diagram Specification

The diagram must use only the ordered nodes in
`c83p_tables/protocol_timing_diagram.csv`. It must distinguish prospective
seed-3 analysis, post-seed-3/pre-seed-4 locking, post-C79 existing-field C80,
historical C81-E, post-C81-outcome-access C82 recovery, and no-new-outcome C83P.
It must not depict C82 as untouched confirmation or C83P as scientific execution.
""")

    limitations = {
        "schema_version": "c83_aaai_limitations_external_validity_contract_v1",
        "milestone": "C83P",
        "independent_datasets": 1,
        "training_seeds": 2,
        "primary_target_clusters": 8,
        "external_validation": False,
        "new_subject_generalization": False,
        "deployability": False,
        "universal_zero_label_impossibility": False,
        "universal_one_label_sufficiency": False,
        "C80_Bstar1_scope": "full_panel_source_relative_Q0_policy_with_LOTO_envelope_2_to_4",
        "C82_scope": "post_C81_outcome_access_same_field_recovery",
        "required_disclosures": [row["risk"] for row in read_csv(TABLE_DIR / "reviewer_risk_ledger.csv")],
        "new_experiment_required_for_C83P": False,
    }
    write_json(REPORT_DIR / "C83_AAAI_LIMITATIONS_AND_EXTERNAL_VALIDITY_CONTRACT.json", limitations)
    (REPORT_DIR / "C83_AAAI_LIMITATIONS_AND_EXTERNAL_VALIDITY_CONTRACT.md").write_text("""# C83 Limitations and External-Validity Contract

This evidence freeze covers one EEG dataset, eight target clusters, two paired
training seeds, one frozen candidate field, and fixed selection policies. It
does not support external validation, new-subject generalization, deployability,
universal zero-label impossibility, or universal one-label sufficiency.

C80 B*=1 is full-panel and source-relative under Q0; all 16 leave-one-target
analyses move the frontier to 2 or 4. C82 is a post-C81-outcome-access recovery.
The complete disclosure and claim-narrowing actions are in
`c83p_tables/reviewer_risk_ledger.csv`. C83P requires no new experiment.
""")

    write_json(REPORT_DIR / "C83_AAAI_REPRODUCIBILITY_INDEX.json", reproducibility)
    lines = [
        "# C83 AAAI Reproducibility Index", "",
        "This index references committed compact artifacts and external payload locations; it copies no raw data.", "",
        "| Object | Category | Path | SHA-256 | Bytes |",
        "|---|---|---|---|---:|",
    ]
    for item in reproducibility["items"]:
        lines.append(f"| {item['object_id']} | {item['category']} | `{item['path']}` | `{item['sha256']}` | {item['bytes']} |")
    lines.extend(["", "Raw EEG, target-label arrays, caches, and model weights are not part of this index.", ""])
    (REPORT_DIR / "C83_AAAI_REPRODUCIBILITY_INDEX.md").write_text("\n".join(lines))


def build() -> dict[str, Any]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    build_replay_tables()
    claims = claim_rows()
    write_csv(TABLE_DIR / "claim_contract.csv", claims)
    numbers = build_number_registry()
    write_csv(TABLE_DIR / "authoritative_number_registry.csv", numbers)
    number_index = {row["number_id"]: row for row in numbers}
    build_figure_contracts(number_index)
    build_main_tables(number_index)
    build_baseline_fidelity_appendix()
    build_timing_and_risk()
    reproducibility = build_reproducibility_index()
    build_contract_reports(claims, reproducibility)
    validation = validate()
    write_csv(TABLE_DIR / "evidence_freeze_validation.csv", validation["checks"])
    write_json(REPORT_DIR / "C83P_EVIDENCE_FREEZE_VALIDATION.json", validation["result"])
    return validation


def _validate_number(row: Mapping[str, str]) -> bool:
    path = REPO_ROOT / row["source_artifact"]
    if row["source_field"] == "@row_count":
        return str(len(read_csv(path))) == row["value"]
    if row["source_field"] == "@json_pointer":
        value = json_pointer(read_json(path), row["source_row_key"])
        value = int(value) if isinstance(value, bool) else value
        return str(value) == row["value"]
    key = dict(part.split("=", 1) for part in row["source_row_key"].split("|") if part)
    return csv_lookup(path, key)[row["source_field"]] == row["value"]


def validate() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, evidence: Any) -> None:
        checks.append({"check_id": check_id, "passed": int(bool(passed)), "blocking": int(not passed), "evidence": str(evidence)})

    replay = read_csv(TABLE_DIR / "c82_commit_chain_replay.csv")
    check("C82_commit_chain", all(row["status"] == "PASS" for row in replay), f"{len(replay)}/{len(replay)}")
    identity = read_csv(TABLE_DIR / "c82_result_identity_replay.csv")
    check("C82_result_identity", all(row["pass"] == "1" for row in identity), f"{len(identity)}/{len(identity)}")
    manifests = read_csv(TABLE_DIR / "c82_table_manifest_replay.csv")
    check("C82_23_table_hashes_and_rows", len(manifests) == 23 and all(row["hash_pass"] == row["row_count_pass"] == "1" for row in manifests), f"{len(manifests)}/23")
    regressions = read_csv(TABLE_DIR / "c82_regression_replay.csv")
    check("C82_regression_replay", len(regressions) == 4 and all(row["status"] == "PASS" for row in regressions), f"{len(regressions)}/4")

    claims = read_csv(TABLE_DIR / "claim_contract.csv")
    check("claim_contract_C1_C10", {f"C{i}" for i in range(1, 11)} <= {row["claim_id"] for row in claims}, len(claims))
    check("forbidden_claim_contract", sum(row["supported"] == "0" for row in claims) >= 10, "10 minimum")
    artifacts_exist = all(
        (REPO_ROOT / path).exists()
        for row in claims for path in row["supporting_artifacts"].split(";") if path
    )
    check("claim_artifact_referential_integrity", artifacts_exist, "all paths exist")

    numbers = read_csv(TABLE_DIR / "authoritative_number_registry.csv")
    number_ids = [row["number_id"] for row in numbers]
    check("number_ids_unique", len(number_ids) == len(set(number_ids)), len(number_ids))
    number_replay = [_validate_number(row) for row in numbers]
    check("number_registry_exact_replay", all(number_replay), f"{sum(number_replay)}/{len(number_replay)}")
    number_index = {row["number_id"]: row for row in numbers}

    referenced = []
    for name in ("figure_3_data.csv", "figure_4_data.csv", "main_table_2.csv"):
        for row in read_csv(TABLE_DIR / name):
            for field in ("number_id", "regret_number_id", "gain_number_id"):
                if row.get(field) and row[field] != "NA":
                    referenced.append((name, row, row[field]))
            if row.get("topk_number_ids"):
                referenced.extend((name, row, item) for item in row["topk_number_ids"].split("|"))
    check("figure_table_number_ids_resolve", all(number_id in number_index for _, _, number_id in referenced), len(referenced))
    exact_values = True
    for name, row, number_id in referenced:
        if "value" in row and row.get("number_id") == number_id:
            exact_values &= row["value"] == number_index[number_id]["value"]
        if row.get("regret_number_id") == number_id:
            exact_values &= row["standardized_regret"] == number_index[number_id]["value"]
        if row.get("gain_number_id") == number_id:
            exact_values &= row["source_relative_gain"] == number_index[number_id]["value"]
    check("figure_table_values_exact", exact_values, "source value identity")

    q5_text = (REPORT_DIR / "C82E_PM_GITHUB_AUDIT_ADDENDUM.md").read_text()
    check("LOTO_wording_contract", "global method-aware stability rule preserved 7/16" in q5_text and "per-panel, per-method Q1 ledger" in q5_text, "narrow wording present")
    check("Q5_descriptive_wording_contract", "descriptive best registered method within a fixed class" in q5_text, "required wording present")

    method_context = read_csv(C82_TABLE_DIR / "method_context_results.csv")
    false_values = {"0", "False", "false"}
    check("target4_primary_zero", len(method_context) == 672 and all(row["target4_primary"] in false_values for row in method_context), "672/672")
    check("same_label_oracle_zero", all(row["same_label_oracle_accessed"] in false_values for row in method_context), "672/672")
    check("C81_gate_unchanged", read_json(REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY.json")["C81_gate_unchanged"] == C81_GATE, C81_GATE)
    check("C82_gate_exact", read_json(REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY.json")["primary_taxonomy"] == C82_GATE, C82_GATE)
    check("no_new_scientific_result_file", not (REPORT_DIR / "C83_SCIENTIFIC_RESULT.json").exists(), "absent")

    large = []
    for line in git("ls-tree", "-r", "-l", "HEAD").stdout.splitlines():
        left, path = line.split("\t", 1)
        if int(left.split()[-1]) > 50 * 1024 * 1024:
            large.append(path)
    check("no_tracked_payload_over_50MiB", not large, large)

    forbidden_phrases = (
        "C82 is an untouched confirmation",
        "universal zero-label impossibility",
        "universal one-label sufficiency",
        "every seed-3 LOTO panel independently demonstrated",
        "prospectively fixed class representative",
        "inferential winner across methods",
    )
    scan_paths = [
        REPORT_DIR / "C83_AAAI_FIGURE_CONTRACT.md",
        REPORT_DIR / "C83_AAAI_REPRODUCIBILITY_INDEX.md",
        TABLE_DIR / "main_table_1.csv", TABLE_DIR / "main_table_2.csv", TABLE_DIR / "main_table_3.csv",
    ]
    supported_claim_text = "\n".join(row["claim_text_short"] for row in claims if row["supported"] == "1")
    found = {
        phrase: [relative(path) for path in scan_paths if phrase.lower() in path.read_text().lower()]
        + (["supported_claim_text"] if phrase.lower() in supported_claim_text.lower() else [])
        for phrase in forbidden_phrases
    }
    # Explicit prohibition fields are intentionally excluded from this scanner.
    unexpected = {phrase: paths for phrase, paths in found.items() if paths}
    check("forbidden_claim_scanner", not unexpected, unexpected)

    passed = sum(row["passed"] for row in checks)
    result = {
        "schema_version": "c83p_evidence_freeze_validation_v1",
        "checks": len(checks), "passed": passed, "failed": len(checks) - passed,
        "blocking": sum(row["blocking"] for row in checks),
        "new_real_data_statistics": 0, "EEG_or_label_view_accesses": 0,
        "manuscript_prose_created": False,
        "gate": C83_GATE if passed == len(checks) else "C83_EVIDENCE_IDENTITY_CLAIM_OR_TABLE_RECONCILIATION_REQUIRED",
    }
    if result["blocking"]:
        failed = [row for row in checks if not row["passed"]]
        raise C83EvidenceError(f"C83P validation failed: {failed}")
    return {"result": result, "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the deterministic compact evidence freeze")
    args = parser.parse_args(argv)
    output = build() if args.write else validate()
    print(json.dumps(output["result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
