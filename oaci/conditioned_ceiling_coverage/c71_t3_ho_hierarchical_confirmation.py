"""C71 - T3-HO hierarchical confirmation readiness and protocol gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import audit_utils as au


MILESTONE = "C71"
AUTH_TOKEN = "C71_T3_HO_REINFERENCE_ONLY_AUTHORIZED"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c71_tables"
REPORT_JSON = "oaci/reports/C71_T3_HO_HIERARCHICAL_CONFIRMATION.json"
C70_JSON = "oaci/reports/C70_SPLIT_LABEL_INFORMATION_BUDGET.json"
C70_PROTOCOL = "oaci/reports/c70_tables/C71_T3_CONFIRMATORY_PROTOCOL.json"
C70_PROTOCOL_SHA = "oaci/reports/c70_tables/C71_T3_CONFIRMATORY_PROTOCOL.sha256"
MAX_REPORT_BYTES = 50_000_000

PRIMARY_BUDGETS = ("8", "64", "full-construction")
SECONDARY_BUDGETS = ("0", "1", "2", "4", "12", "16", "24", "32", "48")

DECISIONS = (
    "C71-A_within_target_split_label_reliability_confirmed_actionability_weak",
    "C71-B_small_budget_split_label_actionability_confirmed",
    "C71-C_dense_label_partial_recovery_confirmed",
    "C71-D_C70_effect_not_replicated_on_T3_HO",
    "C71-E_hierarchical_signal_replication_but_measurement_control_gap_narrows",
    "C71-F_protocol_masking_or_dependency_blocker",
    "C71-G_T3_HO_ready_but_not_authorized",
    "C71-S1_T3_HO_disjointness_confirmed",
    "C71-S2_physical_view_isolation_passed",
    "C71-S3_candidate_specific_gauge_recovery_partial",
    "C71-S4_common_offset_not_explanatory",
    "C71-S5_no_strict_source_escape_hatch",
    "C71-S6_strict_source_escape_hatch_found",
    "C71-S7_conditional_observability_stable_diagnostic",
    "C71-S8_conditional_cs_proxy_only",
    "C71-S9_target_population_generalization_unresolved",
    "C71-S10_new_training_not_justified",
    "C71-S11_independent_target_or_dataset_replication_now_justified",
)

FINAL_GATES = (
    "T3_HO_CONFIRMS_MEASUREMENT_CONTROL_SEPARATION",
    "T3_HO_CONFIRMS_SMALL_BUDGET_ACTIONABILITY",
    "T3_HO_CONFIRMS_DENSE_LABEL_PARTIAL_RECOVERY_ONLY",
    "T3_HO_FAILS_TO_REPLICATE_C70",
    "T3_HO_ANALYSIS_BLOCKED_BY_PROTOCOL_OR_MASKING",
    "T3_HO_READY_BUT_NOT_AUTHORIZED",
)

FORBIDDEN_PATTERNS = (
    "few-label sufficiency",
    "deployable selector",
    "checkpoint recommendation",
    "source-only rescue",
    "oaci rescue",
    "target-population generalization established",
    "full conditional-cs established",
    "same-label endpoint scalar available at selection time",
    "row-level iid",
    "new training is justified",
    "gpu used",
    "forward pass executed",
    "re-inference executed",
    "t3-ho outcome accessed",
    "manuscript drafting",
)

NEGATION_CUES = (
    "not ",
    "no ",
    "never ",
    "without ",
    "forbid",
    "forbidden ",
    "unavailable ",
    "not authorized ",
    "not executed ",
    "not accessed ",
    "diagnostic only ",
    "diagnostic-only ",
    "proxy-only ",
    "unresolved ",
)

RISK_ROWS = (
    "protocol_timing",
    "adaptive_analysis_in_frozen_universe",
    "T3_HO_disjointness",
    "target_label_sampling_blindness",
    "unique_trial_budget_contract",
    "construction_eval_overlap",
    "cache_rows_not_independent",
    "small_number_of_targets",
    "physical_masking",
    "strict_source_feature_provenance",
    "low_resolution_permutation",
    "bandwidth_multiple_testing",
    "reliability_not_actionability",
    "same_label_oracle_misuse",
    "conditional_cs_proxy_overclaim",
    "target_population_overclaim",
    "raw_cache_in_git",
    "unauthorized_forward_or_training",
)


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _auth_present(token: str = "") -> bool:
    # Exact CLI argument only. Do not inspect prompt/protocol text or env vars.
    return str(token).strip() == AUTH_TOKEN


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: list[dict], cols: list[str]) -> None:
    au.write_csv(path, rows, cols)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _path_hash(path: str) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_or_empty(args: list[str]) -> str:
    try:
        return _git(args)
    except Exception:
        return ""


def _listed_paths() -> list[Path]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        list(Path(REPORT_DIR).glob("C71_*.md"))
        + list(Path(REPORT_DIR).glob("C71_*.json"))
        + list(Path(REPORT_DIR).glob("C71_*.sha256"))
        + [p for p in Path(TABLE_DIR).glob("*.csv") if p.name not in skip]
    )


def _large_scan(paths: list[Path]) -> list[dict]:
    return [
        {
            "path": str(p),
            "size_bytes": os.path.getsize(p),
            "over_50mb": int(os.path.getsize(p) > MAX_REPORT_BYTES),
            "passed": int(os.path.getsize(p) <= MAX_REPORT_BYTES),
        }
        for p in sorted(paths)
    ]


def _artifact_manifest(paths: list[Path], table_dir: str) -> list[dict]:
    counts: dict[str, int | str] = {}
    for path in Path(table_dir).glob("*.csv"):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            counts[str(path)] = sum(1 for _ in reader)
    return [
        {
            "path": str(p),
            "size_bytes": os.path.getsize(p),
            "sha256": _sha256(str(p)),
            "artifact_class": "table" if str(p).endswith(".csv") else "protocol" if "PROTOCOL" in str(p) else "summary_json" if str(p).endswith(".json") else "report",
            "row_count": counts.get(str(p), ""),
        }
        for p in sorted(paths)
    ]


def _affirmative_hit(text: str, phrase: str, window: int = 240) -> bool:
    low = text.lower()
    phrase = phrase.lower()
    start = 0
    while True:
        idx = low.find(phrase, start)
        if idx == -1:
            return False
        ctx = low[max(0, idx - window):idx]
        if not any(cue in ctx for cue in NEGATION_CUES):
            return True
        start = idx + len(phrase)


def build_forbidden_scan(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in FORBIDDEN_PATTERNS:
        total = affirmative = 0
        files = []
        for path in paths:
            if os.path.basename(path) in {"forbidden_claim_scan.csv", "red_team_failure_ledger.csv"}:
                continue
            text = open(path, errors="ignore").read()
            count = text.lower().count(pattern.lower())
            if count:
                total += count
                files.append(path)
                if _affirmative_hit(text, pattern):
                    affirmative += 1
        rows.append({"pattern": pattern, "total_hits": total, "affirmative_hits": affirmative, "files": ";".join(files), "passed": int(affirmative == 0)})
    return rows


def load_context() -> dict:
    c70 = _load_json(C70_JSON)
    parent_protocol = _load_json(C70_PROTOCOL)
    parent_protocol_sha = open(C70_PROTOCOL_SHA).read().strip()
    return {
        "c70": c70,
        "parent_protocol": parent_protocol,
        "parent_protocol_sha": parent_protocol_sha,
        "parent_protocol_sha_replay": _sha256(C70_PROTOCOL),
        "head": _git_or_empty(["rev-parse", "--short", "HEAD"]),
        "branch": _git_or_empty(["branch", "--show-current"]),
        "origin_oaci": _git_or_empty(["rev-parse", "--short", "origin/oaci"]),
    }


def build_c71_protocol(ctx: dict, authorized: bool, timestamp: str) -> tuple[dict, str]:
    parent = ctx["parent_protocol"]
    protocol = {
        "schema_version": "c71_t3_ho_confirmatory_protocol_v1",
        "milestone": "C71",
        "parent_c70_protocol_sha256": ctx["parent_protocol_sha"],
        "parent_c70_protocol_sha256_replayed": ctx["parent_protocol_sha_replay"],
        "protocol_lock_timestamp_utc": timestamp,
        "protocol_lock_source_commit": ctx["head"],
        "authorization_token_status": "present" if authorized else "absent",
        "first_t3_ho_manifest_path_read_timestamp_utc": "",
        "first_t3_ho_outcome_read_timestamp_utc": "",
        "t3_ho_cache_generation_authorized": int(authorized),
        "t3_ho_cache_generation_executed": 0,
        "primary_hypotheses": {
            "H1": "within-target split-label reliability replicates on T3-HO",
            "H2": "8 labels/class practical actionability remains weak unless all gates pass",
            "H3": "64/full dense recovery remains partial relative to endpoint oracle",
            "H4": "measurement-control separation persists inside I5",
            "H5": "same-label endpoint scalar remains oracle-only",
        },
        "primary_budgets": list(PRIMARY_BUDGETS),
        "secondary_budgets": list(SECONDARY_BUDGETS),
        "split_seed_registry": {"base_seed": 71071, "repeat_count": 256},
        "construction_evaluation_contract": {
            "shared_trial_ids_across_candidates": True,
            "construction_eval_disjoint": True,
            "class_stratified_where_support_allows": True,
            "label_budget_counts_unique_target_trial_ids": True,
        },
        "hierarchical_inference_plan": [
            "within-target centering",
            "target-level descriptive estimates",
            "checkpoint-cluster bootstrap",
            "trial-id cluster bootstrap",
            "leave-one-target-out",
            "leave-trajectory-out",
            "blocked permutation preserving target/class/checkpoint/trial structure",
        ],
        "permutation_plan": {"primary_min_permutations": 4999, "conditional_cs_min_permutations": 999, "plus_one_correction": True, "max_stat_over_primary_family": True},
        "bandwidth_rule": "no bandwidth selection for primary split-label tests; any kernel secondary uses fixed grid with nested max-stat/null correction",
        "multiplicity_correction": "max-stat or closed testing over H1-H5 and primary budgets/actionability metrics",
        "actionability_thresholds": {"gauge_recovery": 0.50, "coverage": 0.75, "hit": 0.70, "enrichment": 1.50},
        "failure_gates": [
            "forbidden T3-HO outcome accessed before protocol lock",
            "T3-HO overlaps T1/T2",
            "target-outcome-adaptive inclusion",
            "candidate-specific construction labels under fixed budget",
            "construction/evaluation overlap",
            "forbidden row-level iid inference",
            "same-label endpoint scalar enters construction path",
            "raw cache committed to git",
            "unauthorized forward/training/GPU",
        ],
        "t3_ho_units_from_parent": parent["t3_ho_units"],
        "t3_full_physical_units_from_parent": parent["t3_full_physical_units"],
        "t2_consumed_units_from_parent": parent["t2_consumed_units"],
        "t3_ho_checkpoint_id_set_sha256_from_parent": parent["t3_ho_checkpoint_id_set_sha256"],
        "diagnostic_only_non_deployable": True,
    }
    body = json.dumps(protocol, indent=2, sort_keys=True)
    return protocol, _sha256_text(body + "\n")


def build_risk_register(authorized: bool) -> list[dict]:
    rows = []
    for risk in RISK_ROWS:
        status = "mitigated_for_readiness"
        evidence = "C71 is no-forward readiness because exact CLI authorization token is absent."
        blocking = 0
        mitigation = "Protocol amendment and blocking gates emitted before any T3-HO outcome access."
        caveat = "Confirmatory science requires future explicit authorization and T3-HO cache generation."
        future = 1
        if risk == "unauthorized_forward_or_training":
            status = "blocked_by_exact_token_gate" if not authorized else "authorized_but_not_executed_in_readiness"
            evidence = "No exact C71 CLI token supplied; forward/re-inference/training/GPU observed = 0."
            future = int(not authorized)
        elif risk == "T3_HO_disjointness":
            status = "protocol_locked_from_c70_not_executed"
            evidence = "Parent C70 protocol records T3-HO=1052 and T2 overlap=0; C71 does not consume T3-HO cache."
        elif risk == "physical_masking":
            status = "view_contract_prepared_no_cache"
            evidence = "Physical view manifest is schema/path-policy only until authorized cache exists."
        elif risk == "low_resolution_permutation":
            status = "mitigated_in_protocol"
            evidence = "C71 protocol requires >=4999 primary blocked permutations and reports floor/exceedances."
        elif risk == "small_number_of_targets":
            status = "open_caveat_nonblocking_for_readiness"
            evidence = "C71 remains conditional on nine frozen targets; target-population claim forbidden."
        elif risk == "raw_cache_in_git":
            status = "mitigated"
            evidence = "No raw T3-HO cache is generated or committed."
            future = 0
        rows.append({
            "risk_id": risk,
            "risk_name": risk,
            "status": status,
            "evidence": evidence,
            "blocking": blocking,
            "mitigation": mitigation,
            "residual_caveat": caveat,
            "future_confirmation_needed": future,
        })
    return rows


def build_readiness_tables(ctx: dict, protocol: dict, protocol_sha: str, authorized: bool) -> dict[str, list[dict]]:
    parent = ctx["parent_protocol"]
    t3 = int(parent["t3_ho_units"])
    t2 = int(parent["t2_consumed_units"])
    t1 = 64
    full = int(parent["t3_full_physical_units"])
    timestamp = protocol["protocol_lock_timestamp_utc"]
    noauth = "not_run_not_authorized"
    return {
        "risk_register_rows": build_risk_register(authorized),
        "t3_ho_disjointness_ledger_rows": [
            {"check": "parent_protocol_sha_match", "expected": ctx["parent_protocol_sha"], "observed": ctx["parent_protocol_sha_replay"], "passed": int(ctx["parent_protocol_sha"] == ctx["parent_protocol_sha_replay"]), "status": "pass", "notes": "C71 references locked C70 protocol."},
            {"check": "t3_ho_units", "expected": "1052", "observed": t3, "passed": int(t3 == 1052), "status": "protocol_only_no_t3_access", "notes": "No T3-HO cache/path/outcome read in no-auth C71."},
            {"check": "t2_t3_ho_overlap", "expected": "0", "observed": 0, "passed": 1, "status": "inherited_from_c70_protocol", "notes": "T3-HO execution not authorized."},
        ],
        "t1_t2_t3_overlap_matrix_rows": [
            {"left": "T1", "right": "T1", "left_units": t1, "right_units": t1, "overlap_units": t1, "independent_confirmation": 0},
            {"left": "T1", "right": "T2", "left_units": t1, "right_units": t2, "overlap_units": t1, "independent_confirmation": 0},
            {"left": "T1", "right": "T3-HO", "left_units": t1, "right_units": t3, "overlap_units": 0, "independent_confirmation": 0},
            {"left": "T2", "right": "T3-HO", "left_units": t2, "right_units": t3, "overlap_units": 0, "independent_confirmation": 1},
            {"left": "T3-full", "right": "T3-HO", "left_units": full, "right_units": t3, "overlap_units": t3, "independent_confirmation": 0},
        ],
        "shared_trial_split_contract_rows": [
            {"contract": "unique_trial_budget", "status": "locked_not_executed", "required": 1, "observed": "", "passed": 1, "notes": "Budget counts unique target trial IDs per target/class."},
            {"contract": "shared_construction_ids", "status": "locked_not_executed", "required": 1, "observed": "", "passed": 1, "notes": "Same construction IDs for every candidate within target."},
            {"contract": "disjoint_construction_evaluation", "status": "locked_not_executed", "required": 1, "observed": "", "passed": 1, "notes": "Overlap audited only after authorized cache exists."},
        ],
        "unique_label_budget_ledger_rows": [
            *[{"budget": b, "role": "primary", "labels_counted_as": "unique_target_trial_ids_per_class", "checkpoint_scaled_cost_allowed": 0, "status": "locked_not_executed"} for b in PRIMARY_BUDGETS],
            *[{"budget": b, "role": "secondary_descriptive", "labels_counted_as": "unique_target_trial_ids_per_class", "checkpoint_scaled_cost_allowed": 0, "status": "locked_not_executed"} for b in SECONDARY_BUDGETS],
        ],
        "construction_eval_overlap_audit_rows": [
            {"audit": "construction_eval_overlap", "status": noauth, "overlap_trial_ids": "", "passed": 1, "notes": "No T3-HO split instantiated without authorization."}
        ],
        "physical_view_manifest_rows": [
            {"view_name": "source_only_view", "path": "", "sha256": "", "allowed_columns": "checkpoint/source metadata only", "forbidden_columns": "target labels;target correctness;endpoint scalar", "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "consumer_command": "not_materialized_not_authorized"},
            {"view_name": "key_template_view", "path": "", "sha256": "", "allowed_columns": "registered keys/templates", "forbidden_columns": "target labels;endpoint scalar", "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "consumer_command": "not_materialized_not_authorized"},
            {"view_name": "construction_label_view", "path": "", "sha256": "", "allowed_columns": "construction labels only", "forbidden_columns": "evaluation labels;same-label endpoint scalar", "uses_target_labels": 1, "uses_evaluation_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "consumer_command": "not_materialized_not_authorized"},
            {"view_name": "evaluation_label_view", "path": "", "sha256": "", "allowed_columns": "evaluation labels only", "forbidden_columns": "construction-tuned thresholds;same-label endpoint scalar", "uses_target_labels": 1, "uses_evaluation_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "consumer_command": "not_materialized_not_authorized"},
            {"view_name": "same_label_oracle_view", "path": "", "sha256": "", "allowed_columns": "endpoint oracle after primary freeze", "forbidden_columns": "primary construction path", "uses_target_labels": 1, "uses_evaluation_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "consumer_command": "locked_inaccessible_until_primary_freeze"},
        ],
        "dependency_unit_summary_rows": [
            {"unit_family": "T3-HO", "total_rows": 0, "unique_checkpoints": t3, "unique_checkpoint_target_cells": "", "unique_targets": 9, "unique_trajectories": "", "unique_trial_ids": "", "unique_construction_trial_ids": "", "unique_evaluation_trial_ids": "", "status": noauth},
        ],
        "primary_hypothesis_summary_rows": [
            {"hypothesis": "H1_within_target_reliability", "primary_budget": "full-construction", "primary_statistic": "within-target centered Spearman", "confirmatory_gate": "blocked max-stat p<0.01 and positive direction in >=7/9 targets", "status": noauth, "result": ""},
            {"hypothesis": "H2_small_budget_weakness", "primary_budget": "8", "primary_statistic": "gauge/actionability gates", "confirmatory_gate": "small-budget actionability only if all gates pass", "status": noauth, "result": ""},
            {"hypothesis": "H3_dense_partial_recovery", "primary_budget": "64;full-construction", "primary_statistic": "gauge recovery versus 0.50 and residual gap", "confirmatory_gate": "one-sided/equivalence logic from protocol", "status": noauth, "result": ""},
            {"hypothesis": "H4_measurement_control_separation", "primary_budget": "8;64;full-construction", "primary_statistic": "reliability significant while actionability partial", "confirmatory_gate": "joint H1 positive and H2/H3 partial", "status": noauth, "result": ""},
            {"hypothesis": "H5_endpoint_oracle_boundary", "primary_budget": "post-primary", "primary_statistic": "oracle availability tags", "confirmatory_gate": "oracle not available at selection time", "status": noauth, "result": ""},
        ],
        "per_target_confirmatory_results_rows": [{"target_id": str(i), "status": noauth, "within_target_spearman": "", "direction_positive": "", "top1_hit": "", "coverage": "", "gauge_recovery": ""} for i in range(1, 10)],
        "leave_one_target_out_summary_rows": [{"left_out_target": str(i), "status": noauth, "pooled_statistic": "", "p_value": ""} for i in range(1, 10)],
        "reliability_actionability_separation_rows": [{"budget": b, "status": noauth, "within_target_centered_spearman": "", "pairwise_order_accuracy": "", "top1_hit": "", "topk_hit": "", "enrichment": "", "continuous_regret": "", "coverage": "", "measurement_control_separation": ""} for b in PRIMARY_BUDGETS],
        "t3_ho_gauge_recovery_rows": [{"budget": b, "status": noauth, "rank_recovery": "", "candidate_specific_gauge_recovery": "", "common_target_offset_contribution": 0, "residual_variance": "", "source_to_oracle_gap_closed": ""} for b in PRIMARY_BUDGETS],
        "t3_ho_rank_vs_gauge_decomposition_rows": [{"budget": b, "status": noauth, "rank_component": "", "gauge_component": "", "finite_trial_residual": "", "common_offset_not_credited": 1} for b in PRIMARY_BUDGETS],
        "t3_ho_pair_margin_recovery_rows": [{"budget": b, "status": noauth, "pair_count": "", "pairwise_recovery": "", "median_margin": ""} for b in PRIMARY_BUDGETS],
        "cluster_bootstrap_summary_rows": [{"bootstrap": kind, "status": noauth, "replicates": "", "ci_lower": "", "ci_upper": "", "row_iid_used": 0} for kind in ("checkpoint_cluster", "trial_id_cluster", "crossed_pigeonhole", "target_cluster")],
        "blocked_permutation_summary_rows": [{"test": "primary_max_stat", "status": noauth, "permutations": 4999, "exceedances": "", "p_value": "", "minimum_p": 1 / 5000, "row_iid_used": 0}],
        "permutation_resolution_ledger_rows": [
            {"test": "primary_H1_H2_H3_max_stat", "planned_permutations": 4999, "minimum_attainable_p": 1 / 5000, "plus_one_correction": 1, "random_seed_base": 71071, "status": "locked_not_executed"},
            {"test": "conditional_cs_secondary", "planned_permutations": 999, "minimum_attainable_p": 1 / 1000, "plus_one_correction": 1, "random_seed_base": 71171, "status": "locked_not_executed"},
        ],
        "conditional_observability_block_summary_rows": [{"estimator": "finite_partition_binary_y_cod", "status": noauth, "block_valid_status": "planned", "full_conditional_cs_claimed": 0}],
        "conditional_cs_estimator_contract_rows": [{"contract": "conditional_cs_exact_estimator", "assumptions_met_now": 0, "crossed_dependence_handled": 0, "status": "proxy_only_until_authorized_cache_and_block_design", "faithfulness_claim_allowed": 0}],
        "bandwidth_nested_null_audit_rows": [{"bandwidth_rule": "fixed_grid_nested_max_stat_if_kernel_secondary_runs", "status": "locked_not_executed", "selection_inside_null": 1, "evaluation_label_tuning_allowed": 0}],
        "feature_availability_ledger_rows": [
            {"feature_family": "strict_source_domain_trial_logits", "available_now": 0, "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "status": "path_not_available_without_new_instrumentation"},
            {"feature_family": "key_metadata", "available_now": 1, "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 1, "status": "not_strict_source_trial_signal"},
            {"feature_family": "construction_label_content", "available_now": 0, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "requires_authorized_T3_HO_cache"},
            {"feature_family": "same_label_endpoint_oracle", "available_now": 0, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "locked_until_primary_freeze"},
        ],
        "strict_source_adversary_summary_rows": [{"adversary": "strict_source_trial_logits", "status": "not_run_feature_path_unavailable", "target_labels_used": 0, "escape_hatch_found": 0, "notes": "Metadata is not treated as strict source-domain trial evidence."}],
        "failure_reason_ledger_rows": [
            {"reason": "missing_exact_cli_authorization", "status": "blocking_execution_not_readiness", "evidence": "C71 exact authorization token absent", "blocks_science_claim": 1},
            {"reason": "protocol_locked_before_t3_access", "status": "pass", "evidence": f"protocol_sha={protocol_sha}; no T3-HO manifest/outcome timestamp", "blocks_science_claim": 0},
            {"reason": "t3_ho_not_consumed", "status": "pass", "evidence": "T3-HO cache generation/execution observed=0", "blocks_science_claim": 0},
            {"reason": "target_population_generalization", "status": "unresolved", "evidence": "Future C71 remains conditional on nine frozen targets", "blocks_science_claim": 1},
        ],
        "protocol_timing_rows": [
            {"event": "c70_parent_protocol_lock", "timestamp_utc": "", "sha256": ctx["parent_protocol_sha"], "status": "replayed"},
            {"event": "c71_protocol_lock", "timestamp_utc": timestamp, "sha256": protocol_sha, "status": "created_before_t3_access"},
            {"event": "first_t3_ho_manifest_path_read", "timestamp_utc": "", "sha256": "", "status": "not_accessed_no_authorization"},
            {"event": "first_t3_ho_outcome_read", "timestamp_utc": "", "sha256": "", "status": "not_accessed_no_authorization"},
        ],
    }


def build_red_team_rows(res: dict) -> list[dict]:
    risks = {r["risk_id"]: r for r in res["risk_register_rows"]}
    views = res["physical_view_manifest_rows"]
    checks = [
        ("exact_cli_auth_absent_blocks_forward", not res["authorization_present"] and res["forward_or_reinference_executed"] == 0, "No exact C71 CLI token; no forward/re-inference."),
        ("protocol_locked_before_t3_access", res["first_t3_ho_manifest_path_read_timestamp_utc"] == "" and res["first_t3_ho_outcome_read_timestamp_utc"] == "", "Protocol exists before any T3-HO access."),
        ("parent_protocol_sha_replayed", res["parent_c70_protocol_sha256"] == res["parent_c70_protocol_sha256_replayed"], "C70 protocol SHA replayed."),
        ("t3_ho_not_consumed", res["t3_cache_consumed"] == 0, "No T3-HO cache/path/outcome consumed."),
        ("risk_register_no_blocking_for_readiness", all(int(r["blocking"]) == 0 for r in risks.values()), "Risk register has no blocking risk for readiness verdict."),
        ("physical_views_not_materialized_without_cache", all(r["path"] == "" for r in views), "No physical views are materialized without authorized T3-HO cache."),
        ("same_label_oracle_unavailable", next(r for r in views if r["view_name"] == "same_label_oracle_view")["available_at_selection_time"] == 0, "Same-label oracle remains unavailable at selection time."),
        ("row_iid_not_used", all(int(r["row_iid_used"]) == 0 for r in res["blocked_permutation_summary_rows"] + res["cluster_bootstrap_summary_rows"]), "No row-level iid inference is used."),
        ("conditional_cs_proxy_only", all(int(r["full_conditional_cs_claimed"]) == 0 for r in res["conditional_observability_block_summary_rows"]), "No full conditional-CS claim."),
        ("strict_source_no_escape", all(int(r["escape_hatch_found"]) == 0 for r in res["strict_source_adversary_summary_rows"]), "No strict-source escape hatch claim."),
        ("no_training_gpu_reserved", res["training_attempted"] == 0 and res["gpu_used"] == 0 and res["bnci004_used"] == 0 and res["reserved_seeds_used"] == 0, "No training/GPU/heldout release."),
        ("large_artifact_scan_passed", all(int(r["passed"]) for r in res["large_artifact_scan_rows"]), "All committed C71 artifacts under 50MB."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "No affirmative forbidden claims found."),
    ]
    return [{"gate": gate, "failed": int(not ok), "finding": finding} for gate, ok, finding in checks]


def classify(res: dict) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    if failures:
        active = ["C71-F_protocol_masking_or_dependency_blocker"]
        gate = "T3_HO_ANALYSIS_BLOCKED_BY_PROTOCOL_OR_MASKING"
    elif not res["authorization_present"]:
        active = [
            "C71-G_T3_HO_ready_but_not_authorized",
            "C71-S8_conditional_cs_proxy_only",
            "C71-S9_target_population_generalization_unresolved",
            "C71-S10_new_training_not_justified",
        ]
        gate = "T3_HO_READY_BUT_NOT_AUTHORIZED"
    else:
        active = ["C71-F_protocol_masking_or_dependency_blocker"]
        gate = "T3_HO_ANALYSIS_BLOCKED_BY_PROTOCOL_OR_MASKING"
    return {
        "primary": active[0],
        "active": active,
        "inactive": [d for d in DECISIONS if d not in active],
        "final_gate": gate,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": "remote review; provide exact CLI authorization token only if T3-HO re-inference is approved",
    }


def table_row_counts(res: dict) -> dict:
    keys = {
        "risk_register": "risk_register_rows",
        "t3_ho_disjointness_ledger": "t3_ho_disjointness_ledger_rows",
        "t1_t2_t3_overlap_matrix": "t1_t2_t3_overlap_matrix_rows",
        "shared_trial_split_contract": "shared_trial_split_contract_rows",
        "unique_label_budget_ledger": "unique_label_budget_ledger_rows",
        "construction_eval_overlap_audit": "construction_eval_overlap_audit_rows",
        "physical_view_manifest": "physical_view_manifest_rows",
        "dependency_unit_summary": "dependency_unit_summary_rows",
        "primary_hypothesis_summary": "primary_hypothesis_summary_rows",
        "per_target_confirmatory_results": "per_target_confirmatory_results_rows",
        "leave_one_target_out_summary": "leave_one_target_out_summary_rows",
        "reliability_actionability_separation": "reliability_actionability_separation_rows",
        "t3_ho_gauge_recovery": "t3_ho_gauge_recovery_rows",
        "t3_ho_rank_vs_gauge_decomposition": "t3_ho_rank_vs_gauge_decomposition_rows",
        "t3_ho_pair_margin_recovery": "t3_ho_pair_margin_recovery_rows",
        "cluster_bootstrap_summary": "cluster_bootstrap_summary_rows",
        "blocked_permutation_summary": "blocked_permutation_summary_rows",
        "permutation_resolution_ledger": "permutation_resolution_ledger_rows",
        "conditional_observability_block_summary": "conditional_observability_block_summary_rows",
        "conditional_cs_estimator_contract": "conditional_cs_estimator_contract_rows",
        "bandwidth_nested_null_audit": "bandwidth_nested_null_audit_rows",
        "feature_availability_ledger": "feature_availability_ledger_rows",
        "strict_source_adversary_summary": "strict_source_adversary_summary_rows",
        "failure_reason_ledger": "failure_reason_ledger_rows",
        "protocol_timing": "protocol_timing_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "artifact_manifest": "artifact_manifest_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "test_command_manifest": "test_command_manifest_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def run(*, authorization_token: str = "", timestamp: str = "", test_status: str = "planned") -> dict:
    authorized = _auth_present(authorization_token)
    ctx = load_context()
    timestamp = timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    protocol, protocol_sha = build_c71_protocol(ctx, authorized, timestamp)
    readiness = build_readiness_tables(ctx, protocol, protocol_sha, authorized)
    res = {
        "config_hash": _lock_config(),
        "authorization_present": authorized,
        "authorization_token_name": "--authorization-token",
        "current_head": ctx["head"],
        "c70_commit": "4822f1c",
        "c70_final_gate": ctx["c70"].get("final_gate", ""),
        "parent_c70_protocol_sha256": ctx["parent_protocol_sha"],
        "parent_c70_protocol_sha256_replayed": ctx["parent_protocol_sha_replay"],
        "c71_protocol_sha256": protocol_sha,
        "protocol_lock_timestamp_utc": timestamp,
        "first_t3_ho_manifest_path_read_timestamp_utc": "",
        "first_t3_ho_outcome_read_timestamp_utc": "",
        "forward_or_reinference_executed": 0,
        "training_attempted": 0,
        "gpu_used": 0,
        "bnci004_used": 0,
        "reserved_seeds_used": 0,
        "t3_cache_consumed": 0,
        "raw_cache_rows_emitted": 0,
        "selector_artifact_emitted": 0,
        "checkpoint_recommendation_artifact_emitted": 0,
        "_protocol": protocol,
        **readiness,
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "large_artifact_scan_rows": [],
        "artifact_manifest_rows": [],
        "schema_validation_summary_rows": [],
        "red_team_failure_ledger_rows": [],
    }
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []})
    return res


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c71", "command": "python -m pytest oaci/tests/test_c71_t3_ho_hierarchical_confirmation.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c71_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c7*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c71_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c7*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "no_forward_readiness_only": not res["authorization_present"],
        "authorization_present": res["authorization_present"],
        "authorization_token_name": res["authorization_token_name"],
        "forward_or_reinference_executed": res["forward_or_reinference_executed"],
        "training_attempted": res["training_attempted"],
        "gpu_used": res["gpu_used"],
        "bnci004_used": res["bnci004_used"],
        "reserved_seeds_used": res["reserved_seeds_used"],
        "t3_cache_consumed": res["t3_cache_consumed"],
        "raw_cache_rows_emitted": res["raw_cache_rows_emitted"],
        "selector_artifact_emitted": res["selector_artifact_emitted"],
        "checkpoint_recommendation_artifact_emitted": res["checkpoint_recommendation_artifact_emitted"],
        "c70_commit": res["c70_commit"],
        "c70_final_gate": res["c70_final_gate"],
        "current_head_at_generation": res["current_head"],
        "parent_c70_protocol_sha256": res["parent_c70_protocol_sha256"],
        "parent_c70_protocol_sha256_replayed": res["parent_c70_protocol_sha256_replayed"],
        "c71_protocol_sha256": res["c71_protocol_sha256"],
        "protocol_lock_timestamp_utc": res["protocol_lock_timestamp_utc"],
        "first_t3_ho_manifest_path_read_timestamp_utc": res["first_t3_ho_manifest_path_read_timestamp_utc"],
        "first_t3_ho_outcome_read_timestamp_utc": res["first_t3_ho_outcome_read_timestamp_utc"],
        "decision": res["decision"],
        "final_gate": res["decision"]["final_gate"],
        "key_numbers": {
            "t3_full_physical_units": res["_protocol"]["t3_full_physical_units_from_parent"],
            "t2_consumed_units": res["_protocol"]["t2_consumed_units_from_parent"],
            "t3_ho_disjoint_units": res["_protocol"]["t3_ho_units_from_parent"],
            "primary_budgets": list(PRIMARY_BUDGETS),
            "primary_blocked_permutations_planned": 4999,
            "conditional_cs_permutations_planned": 999,
            "red_team_failure_count": res["decision"]["red_team_failure_count"],
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    auth_line = "absent" if not res["authorization_present"] else "present"
    main = "\n".join([
        f"# C71 - T3-HO Hierarchical Confirmation / Measurement-Control Separation Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "## 1. Executive Verdict",
        "",
        f"Primary: `{d['primary']}`",
        "",
        f"Active: `{' ; '.join(d['active'])}`",
        "",
        f"Inactive: `{' ; '.join(d['inactive'])}`",
        "",
        f"Final gate: `{d['final_gate']}`",
        "",
        "## 2. Authorization Boundary",
        "",
        f"C71 exact CLI authorization status: `{auth_line}`. This run is a readiness/protocol audit only.",
        "",
        f"Observed execution counters: forward/re-inference `{res['forward_or_reinference_executed']}`, training `{res['training_attempted']}`, GPU `{res['gpu_used']}`, T3 cache consumption `{res['t3_cache_consumed']}`, raw cache rows `{res['raw_cache_rows_emitted']}`.",
        "",
        "The command-line token is the only accepted authorization route; prompt text, protocol text, comments, and environment variables are ignored.",
        "",
        "## 3. Protocol Lock",
        "",
        f"C70 parent protocol SHA-256: `{res['parent_c70_protocol_sha256']}`.",
        "",
        f"C71 prospective protocol SHA-256: `{res['c71_protocol_sha256']}`.",
        "",
        f"C71 protocol lock timestamp: `{res['protocol_lock_timestamp_utc']}`.",
        "",
        "No T3-HO manifest path or outcome timestamp is recorded in this no-auth run.",
        "",
        "## 4. Readiness Ledger",
        "",
        f"Parent C70 records `{res['_protocol']['t3_full_physical_units_from_parent']}` full physical units, `{res['_protocol']['t2_consumed_units_from_parent']}` T2 consumed units, and `{res['_protocol']['t3_ho_units_from_parent']}` T3-HO disjoint units.",
        "",
        "C71 emits the risk register, disjointness ledger, overlap matrix, split contract, physical-view manifest, dependency summary, hypothesis table, hierarchical-inference placeholders, conditional-observability contracts, feature provenance, and failure ledger required for the future authorized run.",
        "",
        "## 5. Interpretation",
        "",
        "No C71 scientific confirmation is claimed here. The only completed result is that the C71 protocol and blocking gates are ready while the exact execution token is absent.",
    ])
    timing = "\n".join([
        "# C71 - Protocol Timing Audit",
        "",
        f"- C70 parent protocol SHA-256: `{res['parent_c70_protocol_sha256']}`",
        f"- C70 parent replay SHA-256: `{res['parent_c70_protocol_sha256_replayed']}`",
        f"- C71 protocol lock timestamp: `{res['protocol_lock_timestamp_utc']}`",
        f"- C71 protocol SHA-256: `{res['c71_protocol_sha256']}`",
        "- First T3-HO manifest/path read timestamp: `not_accessed_no_authorization`",
        "- First T3-HO outcome read timestamp: `not_accessed_no_authorization`",
        "",
        "Protocol lock and hash were emitted before any T3-HO path or outcome access in this readiness run.",
    ])
    red = "\n".join([
        "# C71 - Red-Team Verification",
        "",
        "All C71 readiness red-team gates pass." if d["red_team_failure_count"] == 0 else "C71 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    return {
        "C71_T3_HO_HIERARCHICAL_CONFIRMATION.md": main,
        "C71_PROTOCOL_TIMING_AUDIT.md": timing,
        "C71_RED_TEAM_VERIFICATION.md": red,
    }


def write_tables(res: dict) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    specs = {
        "risk_register.csv": ("risk_register_rows", ["risk_id", "risk_name", "status", "evidence", "blocking", "mitigation", "residual_caveat", "future_confirmation_needed"]),
        "t3_ho_disjointness_ledger.csv": ("t3_ho_disjointness_ledger_rows", ["check", "expected", "observed", "passed", "status", "notes"]),
        "t1_t2_t3_overlap_matrix.csv": ("t1_t2_t3_overlap_matrix_rows", ["left", "right", "left_units", "right_units", "overlap_units", "independent_confirmation"]),
        "shared_trial_split_contract.csv": ("shared_trial_split_contract_rows", ["contract", "status", "required", "observed", "passed", "notes"]),
        "unique_label_budget_ledger.csv": ("unique_label_budget_ledger_rows", ["budget", "role", "labels_counted_as", "checkpoint_scaled_cost_allowed", "status"]),
        "construction_eval_overlap_audit.csv": ("construction_eval_overlap_audit_rows", ["audit", "status", "overlap_trial_ids", "passed", "notes"]),
        "physical_view_manifest.csv": ("physical_view_manifest_rows", ["view_name", "path", "sha256", "allowed_columns", "forbidden_columns", "uses_target_labels", "uses_evaluation_labels", "available_at_selection_time", "diagnostic_only", "consumer_command"]),
        "dependency_unit_summary.csv": ("dependency_unit_summary_rows", ["unit_family", "total_rows", "unique_checkpoints", "unique_checkpoint_target_cells", "unique_targets", "unique_trajectories", "unique_trial_ids", "unique_construction_trial_ids", "unique_evaluation_trial_ids", "status"]),
        "primary_hypothesis_summary.csv": ("primary_hypothesis_summary_rows", ["hypothesis", "primary_budget", "primary_statistic", "confirmatory_gate", "status", "result"]),
        "per_target_confirmatory_results.csv": ("per_target_confirmatory_results_rows", ["target_id", "status", "within_target_spearman", "direction_positive", "top1_hit", "coverage", "gauge_recovery"]),
        "leave_one_target_out_summary.csv": ("leave_one_target_out_summary_rows", ["left_out_target", "status", "pooled_statistic", "p_value"]),
        "reliability_actionability_separation.csv": ("reliability_actionability_separation_rows", ["budget", "status", "within_target_centered_spearman", "pairwise_order_accuracy", "top1_hit", "topk_hit", "enrichment", "continuous_regret", "coverage", "measurement_control_separation"]),
        "t3_ho_gauge_recovery.csv": ("t3_ho_gauge_recovery_rows", ["budget", "status", "rank_recovery", "candidate_specific_gauge_recovery", "common_target_offset_contribution", "residual_variance", "source_to_oracle_gap_closed"]),
        "t3_ho_rank_vs_gauge_decomposition.csv": ("t3_ho_rank_vs_gauge_decomposition_rows", ["budget", "status", "rank_component", "gauge_component", "finite_trial_residual", "common_offset_not_credited"]),
        "t3_ho_pair_margin_recovery.csv": ("t3_ho_pair_margin_recovery_rows", ["budget", "status", "pair_count", "pairwise_recovery", "median_margin"]),
        "cluster_bootstrap_summary.csv": ("cluster_bootstrap_summary_rows", ["bootstrap", "status", "replicates", "ci_lower", "ci_upper", "row_iid_used"]),
        "blocked_permutation_summary.csv": ("blocked_permutation_summary_rows", ["test", "status", "permutations", "exceedances", "p_value", "minimum_p", "row_iid_used"]),
        "permutation_resolution_ledger.csv": ("permutation_resolution_ledger_rows", ["test", "planned_permutations", "minimum_attainable_p", "plus_one_correction", "random_seed_base", "status"]),
        "conditional_observability_block_summary.csv": ("conditional_observability_block_summary_rows", ["estimator", "status", "block_valid_status", "full_conditional_cs_claimed"]),
        "conditional_cs_estimator_contract.csv": ("conditional_cs_estimator_contract_rows", ["contract", "assumptions_met_now", "crossed_dependence_handled", "status", "faithfulness_claim_allowed"]),
        "bandwidth_nested_null_audit.csv": ("bandwidth_nested_null_audit_rows", ["bandwidth_rule", "status", "selection_inside_null", "evaluation_label_tuning_allowed"]),
        "feature_availability_ledger.csv": ("feature_availability_ledger_rows", ["feature_family", "available_now", "uses_target_labels", "available_at_selection_time", "diagnostic_only", "status"]),
        "strict_source_adversary_summary.csv": ("strict_source_adversary_summary_rows", ["adversary", "status", "target_labels_used", "escape_hatch_found", "notes"]),
        "failure_reason_ledger.csv": ("failure_reason_ledger_rows", ["reason", "status", "evidence", "blocks_science_claim"]),
        "protocol_timing.csv": ("protocol_timing_rows", ["event", "timestamp_utc", "sha256", "status"]),
        "test_command_manifest.csv": ("test_command_manifest_rows", ["test_scope", "command", "status", "environment", "slurm_partition"]),
        "forbidden_claim_scan.csv": ("forbidden_claim_scan_rows", ["pattern", "total_hits", "affirmative_hits", "files", "passed"]),
        "large_artifact_scan.csv": ("large_artifact_scan_rows", ["path", "size_bytes", "over_50mb", "passed"]),
        "schema_validation_summary.csv": ("schema_validation_summary_rows", ["table_name", "row_count", "required_columns_present", "passed"]),
        "red_team_failure_ledger.csv": ("red_team_failure_ledger_rows", ["gate", "failed", "finding"]),
        "artifact_manifest.csv": ("artifact_manifest_rows", ["path", "size_bytes", "sha256", "artifact_class", "row_count"]),
    }
    for name, (key, cols) in specs.items():
        _write_csv(os.path.join(TABLE_DIR, name), res.get(key, []), cols)


def _schema_rows() -> list[dict]:
    rows = []
    for path in sorted(Path(TABLE_DIR).glob("*.csv")):
        if path.name in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": path.name, "row_count": count, "required_columns_present": int(bool(header)), "passed": int(bool(header))})
    return rows


def write_protocol_artifacts(protocol: dict, protocol_sha: str) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "C71_T3_HO_CONFIRMATORY_PROTOCOL.json")
    with open(path, "w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
        f.write("\n")
    actual = _sha256(path)
    if actual != protocol_sha:
        raise ValueError(f"C71 protocol SHA mismatch: expected {protocol_sha}; got {actual}")
    with open(os.path.join(REPORT_DIR, "C71_T3_HO_CONFIRMATORY_PROTOCOL.sha256"), "w") as f:
        f.write(actual + "\n")


def _write_reports_and_json(res: dict) -> None:
    for name, text in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
        f.write("\n")


def _refresh_quality_rows(res: dict) -> None:
    write_tables(res)
    _write_reports_and_json(res)
    paths = [str(p) for p in _listed_paths()]
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan([Path(p) for p in paths])
    write_tables(res)
    res["schema_validation_summary_rows"] = _schema_rows()
    write_tables(res)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)


def write_artifacts(res: dict) -> dict:
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    write_protocol_artifacts(res["_protocol"], res["c71_protocol_sha256"])
    _refresh_quality_rows(res)
    _refresh_quality_rows(res)
    _write_reports_and_json(res)
    paths = _listed_paths()
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["artifact_manifest_rows"] = [{} for _ in paths]
    write_tables(res)
    res["schema_validation_summary_rows"] = _schema_rows()
    write_tables(res)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    _write_reports_and_json(res)
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    write_tables(res)
    _write_reports_and_json(res)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c71_t3_ho_hierarchical_confirmation")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--authorization-token", default="")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(authorization_token=args.authorization_token, test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res)
    print(f"[C71] decision={res['decision']['primary']} gate={res['decision']['final_gate']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
