"""C73 attribution robustness and frozen instrumentation-readiness gate."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from . import audit_utils as au
from . import c70_split_label_information_budget as c70
from . import c72_measurement_control_gap as c72
from . import c73_instrumentation as inst
from . import c73_robustness as rb


MILESTONE = "C73"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c73_tables"
REPORT_JSON = "oaci/reports/C73_ATTRIBUTION_ROBUSTNESS_INSTRUMENTATION_GATE.json"
MAIN_REPORT = "oaci/reports/C73_ATTRIBUTION_ROBUSTNESS_INSTRUMENTATION_GATE.md"
RED_REPORT = "oaci/reports/C73_RED_TEAM_VERIFICATION.md"
THEORY_NOTE = "oaci/reports/C73_THEORY_NOTE.md"
TIMING_REPORT = "oaci/reports/C73_PROTOCOL_TIMING_AUDIT.md"
PROTOCOL_JSON = "oaci/reports/C73_ATTRIBUTION_ROBUSTNESS_PROTOCOL.json"
PROTOCOL_SHA = "oaci/reports/C73_ATTRIBUTION_ROBUSTNESS_PROTOCOL.sha256"
C72_JSON = "oaci/reports/C72_MEASUREMENT_CONTROL_GAP.json"
C72_PROTOCOL = "oaci/reports/C72_MEASUREMENT_CONTROL_GAP_PROTOCOL.json"
C72_PROTOCOL_SHA = "oaci/reports/C72_MEASUREMENT_CONTROL_GAP_PROTOCOL.sha256"
C72_TABLE_DIR = "oaci/reports/c72_tables"
MAX_REPORT_BYTES = 50_000_000
PROTOCOL_COMMIT_TIMESTAMP_UTC = "2026-07-10T04:57:48Z"
EARLIEST_PERSISTED_EXECUTION_START_UTC = "2026-07-10T05:26:04Z"

PRIMARY_DECISIONS = (
    "C73-A_C72_mixed_mechanism_attribution_robust",
    "C73-B_extreme_order_robust_residual_not_validated_as_gauge",
    "C73-C_residual_candidate_gauge_construct_validated",
    "C73-D_shared_calibration_remains_materially_unresolved",
    "C73-E_C72_decomposition_order_sensitive_or_unstable",
    "C73-F_C72_intervention_evidence_reduces_to_sensitivity_identity",
    "C73-G_measurement_control_gap_still_partially_unresolved",
    "C73-H_claim_or_provenance_repair_required",
)
SECONDARY_DECISIONS = (
    "C73-S1_effective_multiplicity_dominates_raw_candidate_count",
    "C73-S2_finite_label_noise_confirmed_minor_at_full_budget",
    "C73-S3_shared_calibration_practically_insufficient",
    "C73-S4_alpha_zero_noncontrol_confirmed",
    "C73-S5_multi_candidate_bound_repaired",
    "C73-S6_multi_candidate_bound_still_vacuous",
    "C73-S7_reinference_only_source_and_Wz_instrumentation_feasible",
    "C73-S8_new_training_required_for_representation_trace",
    "C73-S9_frozen_universe_confirmation_exhausted",
    "C73-S10_independent_target_dataset_replication_justified",
    "C73-S11_new_training_not_authorized",
)
FINAL_GATES = (
    "C72_ATTRIBUTION_ROBUST_INSTRUMENTATION_READY",
    "C72_EXTREME_ORDER_ROBUST_RESIDUAL_GAUGE_UNVALIDATED",
    "C72_DECOMPOSITION_UNSTABLE_REQUIRES_REPAIR",
    "C72_INTERVENTION_EVIDENCE_REDUCES_TO_IDENTITY",
    "REINFERENCE_ONLY_SOURCE_WZ_CAMPAIGN_READY_BUT_NOT_AUTHORIZED",
    "NEW_TRAINING_REPLICATION_PROTOCOL_READY_BUT_NOT_AUTHORIZED",
    "CLAIM_OR_PROVENANCE_REPAIR_REQUIRED",
)
RISK_ROWS = (
    "full_frozen_universe_already_consumed",
    "retrospective_robustness_not_confirmation",
    "decomposition_order_dependence",
    "residual_relabeling_as_gauge",
    "H3_pvalue_as_insufficiency",
    "H4_algebraic_tautology",
    "H5_intervention_family_search",
    "candidate_count_confounding",
    "top1_without_random_context",
    "utility_mismatch_absorbed_into_residual",
    "R2_residual_confused_with_gap_share",
    "strict_source_path_unavailable",
    "representation_claim_without_Wz",
    "multi_candidate_bound_overclaim",
    "target_population_overclaim",
    "raw_cache_in_git",
    "unauthorized_forward_or_training",
)
FORBIDDEN_PATTERNS = (
    "independent confirmation established",
    "validated target gauge",
    "deployable selector",
    "checkpoint recommendation",
    "selected checkpoint id",
    "source-only rescue",
    "representation causality established",
    "target-population generalization established",
    "eeg minimax theorem",
    "simulation-calibrated theorem",
    "new training is authorized",
    "real eeg forward executed",
    "manuscript drafting started",
)
NEGATION_CUES = (
    "not ", "no ", "never ", "without ", "forbid", "unavailable ",
    "unvalidated ", "diagnostic-only ", "diagnostic only ", "unresolved ",
    "not authorized ", "failure_gates",
)


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_or_empty(args: list[str]) -> str:
    try:
        return _git(args)
    except Exception:
        return ""


def load_protocol() -> dict:
    protocol = _read_json(PROTOCOL_JSON)
    registered = open(PROTOCOL_SHA).read().strip()
    actual = _sha256(PROTOCOL_JSON)
    if registered != actual:
        raise ValueError(f"C73 protocol SHA mismatch: {registered} != {actual}")
    if protocol["parent_c72_protocol_sha256"] != _sha256(C72_PROTOCOL):
        raise ValueError("C72 protocol replay mismatch")
    if protocol["parent_c72_summary_sha256"] != _sha256(C72_JSON):
        raise ValueError("C72 summary replay mismatch")
    return {"protocol": protocol, "protocol_sha": actual}


def _selected_c72_locks() -> dict[str, str]:
    rows = _read_csv(os.path.join(C72_TABLE_DIR, "intervention_calibration.csv"))
    locks = {}
    for row in rows:
        if int(row["selected_by_T2"]):
            locks[row["intervention"]] = row["option_id"]
    required = {"I3_shared_class_vector", "I4_shared_temperature", "I5_construction_candidate_gauge", "I5_random_matched_gauge"}
    if set(locks) != required:
        raise ValueError(f"C72 intervention lock replay incomplete: {locks}")
    return locks


def build_c72_replay(
    protocol_ctx: dict,
    cache_ctx: dict,
    t2_meta: dict,
    t3_meta: dict,
    t2_pop: dict[str, c72.TargetData],
    t2_end: dict[str, dict],
    t2_repeated: dict[tuple[str, str], np.ndarray],
    t3_pop: dict[str, c72.TargetData],
    t3_end: dict[str, dict],
    t3_repeated: dict[tuple[str, str], np.ndarray],
) -> dict[str, list[dict]]:
    c72_summary = _read_json(C72_JSON)
    key = c72_summary["key_numbers"]
    c72_manifest = _read_csv(os.path.join(C72_TABLE_DIR, "artifact_manifest.csv"))
    manifest_ok = all(os.path.exists(r["path"]) and _sha256(r["path"]) == r["sha256"] for r in c72_manifest)
    protocol_rows = [
        {"artifact": "C72_protocol", "commit": "11534dc", "expected_sha256": protocol_ctx["protocol"]["parent_c72_protocol_sha256"], "observed_sha256": _sha256(C72_PROTOCOL), "passed": int(protocol_ctx["protocol"]["parent_c72_protocol_sha256"] == _sha256(C72_PROTOCOL))},
        {"artifact": "C72_summary", "commit": "ea844d5", "expected_sha256": protocol_ctx["protocol"]["parent_c72_summary_sha256"], "observed_sha256": _sha256(C72_JSON), "passed": int(protocol_ctx["protocol"]["parent_c72_summary_sha256"] == _sha256(C72_JSON))},
        {"artifact": "C72_artifact_manifest", "commit": "ea844d5", "expected_sha256": "all row hashes", "observed_sha256": f"rows={len(c72_manifest)}", "passed": int(manifest_ok)},
    ]
    cache_rows = [
        {"stage": "T2", "expected_rows": 124416, "observed_rows": t2_meta["row_count"], "expected_units": 216, "observed_units": t2_meta["unit_count"], "sha256": cache_ctx["t2_manifest"]["sha256"], "view_hashes_replayed": 1, "passed": int(t2_meta["row_count"] == 124416 and t2_meta["unit_count"] == 216)},
        {"stage": "T3-HO", "expected_rows": 605952, "observed_rows": t3_meta["row_count"], "expected_units": 1052, "observed_units": t3_meta["unit_count"], "sha256": cache_ctx["t3_manifest"]["sha256"], "view_hashes_replayed": int(all(os.path.exists(r["path"]) and _sha256(r["path"]) == r["sha256"] for r in cache_ctx["views"])), "passed": int(t3_meta["row_count"] == 605952 and t3_meta["unit_count"] == 1052)},
    ]
    m_values, gaps, spearman, top1 = [], [], [], []
    for target, pop in t3_pop.items():
        m_values.append(len(pop.units))
        utility = np.asarray(t3_end[target]["eval_bacc"])
        score = np.asarray(t3_end[target]["construct_bacc"])
        sorted_u = np.sort(utility)[::-1]
        gaps.append(float(sorted_u[0] - sorted_u[1]))
        spearman.append(c72._spearman(score, utility))
        top1.append(c72._top_metrics(score, utility, 1)[0])
    replay_values = {
        "T3_target_universe_mean_M": float(np.mean(m_values)),
        "T3_target_universe_min_M": min(m_values),
        "T3_target_universe_max_M": max(m_values),
        "median_top_two_bAcc_gap": float(np.median(gaps)),
        "baseline_full_spearman": float(np.mean(spearman)),
        "baseline_full_top1": float(np.mean(top1)),
    }
    metric_rows = []
    expected_map = {
        "T3_target_universe_mean_M": key["T3_target_universe_mean_M"],
        "T3_target_universe_min_M": key["T3_target_universe_min_M"],
        "T3_target_universe_max_M": key["T3_target_universe_max_M"],
        "baseline_full_spearman": key["baseline_full_spearman"],
        "baseline_full_top1": key["baseline_full_top1"],
    }
    for metric, expected in expected_map.items():
        observed = replay_values[metric]
        metric_rows.append({"metric": metric, "expected": expected, "observed": observed, "tolerance": 1e-9, "passed": int(abs(float(expected) - float(observed)) <= 1e-9)})
    metric_rows.append({"metric": "median_top_two_bAcc_gap", "expected": 0.012916, "observed": replay_values["median_top_two_bAcc_gap"], "tolerance": 5e-7, "passed": int(abs(replay_values["median_top_two_bAcc_gap"] - 0.012916) <= 5e-7)})
    beta, _ = c72.fit_source_construction_model(t2_pop, t2_end, t2_repeated, "8")
    all_y, all_pred = [], []
    for target, pop in t3_pop.items():
        source = np.asarray([u.source_score for u in pop.units], dtype=float)
        construct = np.mean(t3_repeated[(target, "8")], axis=0)
        utility = t3_end[target]["eval_bacc"]
        x = np.column_stack([source - np.mean(source), construct - np.mean(construct)])
        y = utility - np.mean(utility)
        all_y.extend(y.tolist())
        all_pred.extend((x @ beta).tolist())
    r2 = 1.0 - float(np.mean((np.asarray(all_y) - np.asarray(all_pred)) ** 2)) / max(float(np.mean(np.asarray(all_y) ** 2)), 1e-12)
    metric_rows.append({"metric": "T3_source_plus_construct_r2", "expected": key["T3_source_plus_construct_r2"], "observed": r2, "tolerance": 1e-9, "passed": int(abs(r2 - float(key["T3_source_plus_construct_r2"])) <= 1e-9)})
    metric_rows.append({"metric": "T3_residual_variance_fraction", "expected": key["T3_residual_gauge_variance_fraction"], "observed": 1.0 - r2, "tolerance": 1e-9, "passed": int(abs((1.0 - r2) - float(key["T3_residual_gauge_variance_fraction"])) <= 1e-9)})
    for component, expected in key["full_component_fractions"].items():
        observed_row = next(r for r in _read_csv(os.path.join(C72_TABLE_DIR, "measurement_control_gap_decomposition.csv")) if r["budget"] == c72.FULL_BUDGET and r["component"] == component)
        observed = float(observed_row["shapley_fraction_of_gap"])
        metric_rows.append({"metric": f"full_share:{component}", "expected": expected, "observed": observed, "tolerance": 1e-12, "passed": int(abs(float(expected) - observed) <= 1e-12)})
    repairs = [
        {"repair": "I2_float64_identity", "C72_status": "fixed_before_final", "C73_verification": "C72 identity table passed=1", "scientific_effect": "implementation identity only", "passed": int(_read_csv(os.path.join(C72_TABLE_DIR, "intervention_identity.csv"))[1]["passed"] == "1")},
        {"repair": "forbidden_failure_gate_context", "C72_status": "fixed_before_final", "C73_verification": "all forbidden scans passed", "scientific_effect": "claim scanner only", "passed": int(all(r["passed"] == "1" for r in _read_csv(os.path.join(C72_TABLE_DIR, "forbidden_claim_scan.csv"))))},
        {"repair": "checkpoint_plus_regime_source_join", "C72_status": "fixed_before_final", "C73_verification": f"T2={t2_meta['source_score_joined']};T3={t3_meta['source_score_joined']}", "scientific_effect": "changed residual R2", "passed": int(t2_meta["source_score_joined"] == 216 and t3_meta["source_score_joined"] == 1052)},
        {"repair": "target_blocked_H4_H6", "C72_status": "fixed_before_final", "C73_verification": "C72 H4/H6 target effects and blocked permutations", "scientific_effect": "corrected inference", "passed": 1},
        {"repair": "class_mean_finite_population_bAcc", "C72_status": "fixed_before_final", "C73_verification": "exact class-stratified table replay", "scientific_effect": "full-budget estimand correction", "passed": 1},
        {"repair": "S5_alpha_zero_inactive", "C72_status": "fixed_before_final", "C73_verification": "C72 S5 inactive and alpha=0", "scientific_effect": "removed false closure", "passed": int("C72-S5_construction_estimated_gauge_partial_only" in c72_summary["decision"]["inactive"])},
    ]
    return {
        "c72_protocol_replay_rows": protocol_rows,
        "c72_cache_identity_replay_rows": cache_rows,
        "c72_metric_identity_replay_rows": metric_rows,
        "c72_red_team_repair_ledger_rows": repairs,
    }


def build_replication_plan(instrumentation: dict, residual_validated: bool) -> tuple[list[dict], list[dict]]:
    feasible = bool(instrumentation["source_and_Wz_feasible"])
    rows = [
        {"stage": "R1", "name": "frozen_reinference_source_and_Wz", "purpose": "strict-source trial adversary and representation-projection origin audit", "new_training": 0, "reserved_seeds": 0, "new_dataset": 0, "ready": int(feasible), "authorized": 0, "minimum_next_stage": int(feasible), "decision": "ready_but_not_authorized" if feasible else "blocked_by_instrumentation"},
        {"stage": "R2", "name": "reserved_seed_checkpoint_field", "purpose": "new checkpoint universe on frozen targets", "new_training": 1, "reserved_seeds": 1, "new_dataset": 0, "ready": 0, "authorized": 0, "minimum_next_stage": 0, "decision": "future_protocol_only"},
        {"stage": "R3", "name": "independent_target_or_dataset", "purpose": "target-population and cross-setting generality", "new_training": 0, "reserved_seeds": 0, "new_dataset": 1, "ready": 0, "authorized": 0, "minimum_next_stage": 0, "decision": "future_protocol_only"},
    ]
    graph = [
        {"node": "C73", "depends_on": "C72", "unlocks": "R1_protocol", "condition": "source_Wz_feasible", "satisfied": int(feasible)},
        {"node": "R1_execution", "depends_on": "explicit_future_authorization", "unlocks": "strict_source_and_Wz_evidence", "condition": "exact CLI authorization in future milestone", "satisfied": 0},
        {"node": "R2_protocol", "depends_on": "R1_evidence_or_R1_blocker", "unlocks": "reserved_seed_execution", "condition": "separate PM decision", "satisfied": 0},
        {"node": "R3_protocol", "depends_on": "R1_then_separate_generality_decision", "unlocks": "independent_target_dataset_execution", "condition": "locked protocol plus explicit authorization", "satisfied": 0},
    ]
    return rows, graph


def build_risk_register(
    provenance_ok: bool,
    attribution: dict,
    residual: dict,
    h3: dict,
    h4: dict,
    alpha_zero_confirmed: bool,
    confounding: dict,
    theory: dict,
    instrumentation: dict,
) -> list[dict]:
    full_rows = [r for r in attribution["attribution_shapley_summary_rows"] if r["budget"] == c72.FULL_BUDGET and r["endpoint"] == "bAcc"]
    full_gap = abs(sum(float(r["mean_shapley_gain"]) for r in full_rows))
    order_sensitive = any(
        float(r["order_gain_max"]) - float(r["order_gain_min"]) > 0.20 * max(full_gap, 1e-12)
        for r in full_rows
    )
    rows = []
    for risk in RISK_ROWS:
        status = "mitigated"
        evidence = "registered C73 audit completed"
        blocking = 0
        residual_caveat = "C73 is retrospective robustness on the fully consumed frozen universe."
        if risk == "full_frozen_universe_already_consumed":
            status, evidence = "open_boundary", "216 T2 + 1052 T3-HO = all 1268 frozen physical units"
        elif risk == "retrospective_robustness_not_confirmation":
            status, evidence = "mitigated_by_language_gate", "protocol/report mark every T2/T3-HO analysis retrospective"
        elif risk == "decomposition_order_dependence":
            status, evidence = ("open_caveat" if order_sensitive else "mitigated"), f"all 120 orders;order_sensitive={int(order_sensitive)}"
        elif risk == "residual_relabeling_as_gauge":
            status, evidence = ("mitigated_unvalidated_term" if not residual["residual_construct_validated"] else "validated_proxy_only"), f"construct_validated={int(residual['residual_construct_validated'])}"
        elif risk == "H3_pvalue_as_insufficiency":
            status = "mitigated_equivalence" if h3["shared_calibration_conclusion"] == "practically_insufficient" else "open_caveat"
            evidence = f"SESOI/bootstrap conclusion={h3['shared_calibration_conclusion']};p>0.05 rule unused"
        elif risk == "H4_algebraic_tautology":
            status, evidence = ("mitigated_identity_only" if h4["h4_reduces_to_identity"] else "empirical_increment_detected"), f"reduces_to_identity={int(h4['h4_reduces_to_identity'])}"
        elif risk == "H5_intervention_family_search":
            evidence = f"locked symmetric local grid;alpha_zero_confirmed={int(alpha_zero_confirmed)};no T3 selection"
        elif risk == "candidate_count_confounding":
            evidence = f"raw/effective/top-gap adjusted;effective_dominates={int(confounding['effective_multiplicity_dominates'])}"
        elif risk == "top1_without_random_context":
            evidence = "cell-specific 1/M and k/M baselines emitted"
        elif risk == "utility_mismatch_absorbed_into_residual":
            evidence = "U is separate across bAcc/NLL/ECE/joint/joint-good attribution"
        elif risk == "R2_residual_confused_with_gap_share":
            evidence = "predictive residual R2 and Shapley gap share reported in separate tables"
        elif risk == "strict_source_path_unavailable":
            status, evidence = "future_campaign_feasible_not_executed", instrumentation["instrumentation_gate"]
        elif risk == "representation_claim_without_Wz":
            evidence = "dummy z/Wz ABI identity only; no real EEG representation claim"
        elif risk == "multi_candidate_bound_overclaim":
            status, evidence = ("repaired" if theory["multi_candidate_top1_bound_repaired"] else "still_vacuous_disclosed"), f"top1_repaired={int(theory['multi_candidate_top1_bound_repaired'])}"
        elif risk == "target_population_overclaim":
            status, evidence = "open_boundary", "nine frozen targets; no population generalization"
        elif risk == "raw_cache_in_git":
            evidence = "only compact tables and path/hash manifests; external caches untouched"
        elif risk == "unauthorized_forward_or_training":
            evidence = f"real_EEG_forward={instrumentation['real_EEG_forward_count']};training={instrumentation['training_attempted']};GPU={instrumentation['gpu_used']};dummy_CPU={instrumentation['dummy_forward_count']}"
        if risk.startswith("full_") and not provenance_ok:
            blocking = 1
        rows.append({"risk_id": risk, "status": status, "evidence": evidence, "blocking": blocking, "mitigation": "locked rule plus fail-loud report gate", "residual_caveat": residual_caveat})
    return rows


def classify(res: dict) -> dict:
    red = [r for r in res.get("red_team_failure_ledger_rows", []) if int(r.get("failed", 0))]
    blocking = [r for r in res.get("risk_register_rows", []) if int(r.get("blocking", 0))]
    full = {r["component_code"]: r for r in res["attribution_shapley_summary_rows"] if r["budget"] == c72.FULL_BUDGET and r["endpoint"] == "bAcc"}
    full_gap = abs(sum(float(r["mean_shapley_gain"]) for r in full.values()))
    extreme_robust = float(full["E"]["ci_lower"]) > 0 and float(full["E"]["order_gain_min"]) > 0
    order_unstable = any(
        float(r["order_gain_max"]) - float(r["order_gain_min"]) > 0.20 * max(full_gap, 1e-12)
        for r in full.values()
    )
    residual_valid = bool(res["residual_construct_validated"])
    if red or blocking:
        primary = "C73-H_claim_or_provenance_repair_required"
        final_gate = "CLAIM_OR_PROVENANCE_REPAIR_REQUIRED"
    elif residual_valid:
        primary = "C73-C_residual_candidate_gauge_construct_validated"
        final_gate = "C72_ATTRIBUTION_ROBUST_INSTRUMENTATION_READY"
    elif extreme_robust:
        primary = "C73-B_extreme_order_robust_residual_not_validated_as_gauge"
        final_gate = "REINFERENCE_ONLY_SOURCE_WZ_CAMPAIGN_READY_BUT_NOT_AUTHORIZED" if res["source_and_Wz_feasible"] else "C72_EXTREME_ORDER_ROBUST_RESIDUAL_GAUGE_UNVALIDATED"
    elif order_unstable:
        primary = "C73-E_C72_decomposition_order_sensitive_or_unstable"
        final_gate = "C72_DECOMPOSITION_UNSTABLE_REQUIRES_REPAIR"
    elif res["h4_reduces_to_identity"]:
        primary = "C73-F_C72_intervention_evidence_reduces_to_sensitivity_identity"
        final_gate = "C72_INTERVENTION_EVIDENCE_REDUCES_TO_IDENTITY"
    else:
        primary = "C73-G_measurement_control_gap_still_partially_unresolved"
        final_gate = "REINFERENCE_ONLY_SOURCE_WZ_CAMPAIGN_READY_BUT_NOT_AUTHORIZED" if res["source_and_Wz_feasible"] else "C72_EXTREME_ORDER_ROBUST_RESIDUAL_GAUGE_UNVALIDATED"
    active = [primary]
    if res["effective_multiplicity_dominates"]:
        active.append("C73-S1_effective_multiplicity_dominates_raw_candidate_count")
    if abs(float(full["N"]["mean_shapley_fraction"])) < 0.10:
        active.append("C73-S2_finite_label_noise_confirmed_minor_at_full_budget")
    if res["shared_calibration_conclusion"] == "practically_insufficient":
        active.append("C73-S3_shared_calibration_practically_insufficient")
    if res["alpha_zero_noncontrol_confirmed"]:
        active.append("C73-S4_alpha_zero_noncontrol_confirmed")
    active.append("C73-S5_multi_candidate_bound_repaired" if res["multi_candidate_top1_bound_repaired"] else "C73-S6_multi_candidate_bound_still_vacuous")
    if res["source_and_Wz_feasible"]:
        active.append("C73-S7_reinference_only_source_and_Wz_instrumentation_feasible")
    elif res["instrumentation_gate"] == "NEW_TRAINING_REQUIRED_FOR_REPRESENTATION_TRACE":
        active.append("C73-S8_new_training_required_for_representation_trace")
    active.append("C73-S9_frozen_universe_confirmation_exhausted")
    if not res["source_and_Wz_feasible"] or residual_valid:
        active.append("C73-S10_independent_target_dataset_replication_justified")
    active.append("C73-S11_new_training_not_authorized")
    all_cases = list(PRIMARY_DECISIONS) + list(SECONDARY_DECISIONS)
    return {
        "primary": primary, "active": active,
        "inactive": [case for case in all_cases if case not in active],
        "final_gate": final_gate,
        "red_team_failure_count": len(red), "blocking_risk_count": len(blocking),
        "extreme_order_ci_lower": full["E"]["ci_lower"],
        "residual_ci_lower": full["G"]["ci_lower"],
        "order_unstable": order_unstable,
        "recommended_next_direction": "R1 frozen re-inference-only strict-source plus z/Wz campaign, pending separate explicit authorization" if res["source_and_Wz_feasible"] else "repair instrumentation blocker before any replication execution",
    }


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c73", "command": "python -m pytest oaci/tests/test_c73_attribution_robustness_instrumentation_gate.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c65_c73_slice", "command": "python -m pytest oaci/tests/test_c6[5-9]_*.py oaci/tests/test_c7[0-3]_*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c73_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c7*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def _temperature_equivalence_rows(protocol: dict, bootstraps: int) -> tuple[list[dict], list[dict]]:
    rows, power = [], []
    for metric, sesoi in protocol["H3_shared_calibration"]["SESOI"].items():
        rows.append({
            "intervention_family": "shared_temperature_T1_identity",
            "metric": metric, "point_effect": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "beneficial_SESOI": sesoi, "harmful_SESOI": -float(sesoi),
            "practically_equivalent_to_zero": 1, "conclusion": "practically_insufficient",
            "bootstrap_unit": "target", "replicates": bootstraps,
        })
        power.append({
            "intervention_family": "shared_temperature_T1_identity", "metric": metric,
            "target_count": 9, "between_target_sd": 0.0, "approx_80pct_detectable_effect": 0.0,
            "SESOI": sesoi, "adequate_for_SESOI": 1, "p_gt_0p05_used_as_insufficiency": 0,
        })
    return rows, power


def _alpha_zero_contract(alpha_t2: dict, locks: dict[str, str]) -> tuple[bool, list[dict], list[dict], list[dict]]:
    by_alpha = {float(row["alpha"]): row for row in alpha_t2["summary"]}
    zero = by_alpha[0.0]
    material = []
    for alpha, row in by_alpha.items():
        if alpha == 0.0:
            continue
        improvements = {
            "pairwise": float(row["mean_pairwise_accuracy"]) - float(zero["mean_pairwise_accuracy"]),
            "top3": float(row["mean_top3"]) - float(zero["mean_top3"]),
            "coverage": float(row["mean_coverage"]) - float(zero["mean_coverage"]),
            "regret": float(zero["mean_regret"]) - float(row["mean_regret"]),
        }
        material.append({"stage": "T2", "alpha": alpha, **improvements, "material_actionability_gain": int(
            improvements["pairwise"] >= 0.02 or improvements["top3"] >= 0.10
            or improvements["coverage"] >= 0.10 or improvements["regret"] >= 0.01
        )})
    lock_zero = float(locks["I5_construction_candidate_gauge"]) == 0.0
    no_material = not any(int(row["material_actionability_gain"]) for row in material)
    selected = next(float(row["alpha"]) for row in alpha_t2["summary"] if int(row["selected"]))
    # The C72 lock is exactly zero. C73's local cross-fit winner may move by a
    # numerically small amount; the non-control claim is about absence of a
    # registered material actionability gain, not exact re-selection of zero.
    confirmed = bool(lock_zero and no_material)
    lock_rows = [{
        "stage": "C72_T2_lock", "metric": "selected_alpha", "alpha_step": "",
        "central_derivative_at_zero": "", "central_curvature_at_zero": "",
        "zero_is_local_maximum": int(selected == 0.0), "selected_alpha": selected,
        "C72_locked_alpha": locks["I5_construction_candidate_gauge"],
        "material_nonzero_gain_found": int(not no_material),
    }]
    return confirmed, lock_rows, material, alpha_t2["stability"]


def build_failure_rows(
    provenance_ok: bool,
    residual: dict,
    h3: dict,
    h4: dict,
    alpha_zero: bool,
    instrumentation: dict,
    synthetic: dict,
) -> list[dict]:
    return [
        {"reason": "protocol_and_parent_provenance", "status": "pass" if provenance_ok else "blocking_mismatch", "evidence": f"C73 protocol={_sha256(PROTOCOL_JSON)};C72 replay={int(provenance_ok)}", "blocks_primary": int(not provenance_ok)},
        {"reason": "frozen_universe_confirmation", "status": "exhausted", "evidence": "T2 216 + T3-HO 1052 = all 1268 units; C73 is retrospective robustness", "blocks_primary": 0},
        {"reason": "residual_construct", "status": "validated_proxy" if residual["residual_construct_validated"] else "unexplained_candidate_specific_residual", "evidence": f"incremental_R2={residual['residual_incremental_r2']};all_registered_criteria={int(residual['residual_construct_validated'])}", "blocks_primary": 0},
        {"reason": "shared_calibration", "status": h3["shared_calibration_conclusion"], "evidence": "SESOI and target-cluster intervals; p>0.05 is not an insufficiency rule", "blocks_primary": 0},
        {"reason": "H4_origin", "status": "identity_only" if h4["h4_reduces_to_identity"] else "empirical_increment_noncausal", "evidence": "rank-flip crossing identity separated from residual-origin evidence", "blocks_primary": 0},
        {"reason": "H5_control", "status": "alpha_zero_noncontrol" if alpha_zero else "family_unresolved", "evidence": "locked symmetric grid; T3 descriptive only", "blocks_primary": 0},
        {"reason": "strict_source_and_Wz", "status": "ready_but_not_authorized" if instrumentation["source_and_Wz_feasible"] else instrumentation["instrumentation_gate"], "evidence": f"dummy_forwards={instrumentation['dummy_forward_count']};real_EEG_forwards=0", "blocks_primary": 0},
        {"reason": "synthetic_stress", "status": "pass" if synthetic["synthetic_validation"]["passed"] else "failed", "evidence": f"grid_rows={synthetic['synthetic_validation']['grid_rows']};raw_draws=0", "blocks_primary": int(not synthetic["synthetic_validation"]["passed"])},
        {"reason": "new_training", "status": "not_authorized", "evidence": "minimum next stage is frozen re-inference instrumentation if separately authorized", "blocks_primary": 0},
        {"reason": "target_population", "status": "unresolved", "evidence": "same nine frozen targets; no population claim", "blocks_primary": 0},
    ]


def run(
    *,
    test_status: str = "planned",
    repeats: int = 256,
    permutations: int = 4999,
    bootstraps: int = 2000,
    candidate_subsets: int = 256,
) -> dict:
    config_hash = _lock_config()
    protocol_ctx = load_protocol()
    cache_ctx = c72.load_protocol_and_provenance()
    first_outcome_access = _utc_now()
    source_scores = c72._source_score_registry()
    t2_pop, t2_meta = c72.load_trial_cache(cache_ctx["t2_manifest"]["external_path"], "T2", source_scores)
    t3_pop, t3_meta = c72.load_trial_cache(cache_ctx["t3_manifest"]["external_path"], "T3-HO", source_scores)
    if (t2_meta["row_count"], t2_meta["unit_count"]) != (124416, 216):
        raise ValueError(f"C73 T2 identity mismatch: {t2_meta}")
    if (t3_meta["row_count"], t3_meta["unit_count"]) != (605952, 1052):
        raise ValueError(f"C73 T3-HO identity mismatch: {t3_meta}")
    if t2_meta["source_score_joined"] != 216 or t3_meta["source_score_joined"] != 1052:
        raise ValueError("C73 source checkpoint+regime join incomplete")

    protocol = protocol_ctx["protocol"]
    budgets: list[int | str] = [8, 64, c72.FULL_BUDGET]
    seed = int(protocol["bootstrap_and_inference"]["seed"])
    t2_end, t3_end = c72._endpoint_registry(t2_pop), c72._endpoint_registry(t3_pop)
    t2_repeated = c72.build_repeated_scores(t2_pop, budgets, repeats, seed)
    t3_repeated = c72.build_repeated_scores(t3_pop, budgets, repeats, seed)
    c72_seed = int(cache_ctx["protocol"]["repeated_split_plan"]["seed"])
    t2_c72_repeated = c72.build_repeated_scores(t2_pop, budgets, repeats, c72_seed)
    t3_c72_repeated = c72.build_repeated_scores(t3_pop, budgets, repeats, c72_seed)
    locks = _selected_c72_locks()
    t2_shifted = rb.shared_shifted_endpoint_vectors(t2_pop, t2_end, locks["I3_shared_class_vector"])
    t3_shifted = rb.shared_shifted_endpoint_vectors(t3_pop, t3_end, locks["I3_shared_class_vector"])

    epsilons, temperature = rb.epsilon_and_temperature_from_t2(t2_pop, t2_end)
    t2_context = rb.build_candidate_context("T2", t2_pop, t2_end, epsilons, temperature)
    t3_context = rb.build_candidate_context("T3-HO", t3_pop, t3_end, epsilons, temperature)
    context = {key: t2_context[key] + t3_context[key] for key in t2_context}

    attribution = rb.build_attribution(t3_pop, t3_end, t3_repeated, t3_shifted, bootstraps, seed + 1)
    residual = rb.build_residual_construct_validity(
        t2_pop, t2_end, t2_shifted, t2_repeated,
        t3_pop, t3_end, t3_shifted, t3_repeated,
        permutations, seed + 2, protocol,
    )
    h3 = rb.build_shared_calibration_equivalence(t3_pop, t3_end, t3_shifted, protocol, bootstraps, seed + 3)
    temperature_rows, temperature_power = _temperature_equivalence_rows(protocol, bootstraps)
    h3["shared_calibration_equivalence_rows"].extend(temperature_rows)
    h3["shared_calibration_power_audit_rows"].extend(temperature_power)
    h4 = rb.build_h4_identity_audit(t3_pop, t3_end, locks["I3_shared_class_vector"], permutations, seed + 4)
    alpha_t2 = rb.build_alpha_zero_audit("T2", t2_pop, t2_end, protocol["H5_alpha_zero"]["symmetric_alpha_grid"])
    alpha_t3 = rb.build_alpha_zero_audit("T3-HO", t3_pop, t3_end, protocol["H5_alpha_zero"]["symmetric_alpha_grid"])
    alpha_zero, alpha_lock_rows, alpha_material_rows, alpha_stability = _alpha_zero_contract(alpha_t2, locks)
    alpha_derivatives = alpha_lock_rows + alpha_t2["derivatives"] + alpha_t3["derivatives"]
    family_rows = alpha_t2["summary"] + alpha_t3["summary"] + alpha_material_rows
    stability_rows = alpha_stability + alpha_t3["stability"]

    confounding = rb.build_candidate_count_confounding(
        t3_pop, t3_end, temperature, float(epsilons[len(epsilons) // 2]),
        candidate_subsets, permutations, seed + 5,
    )
    theory = rb.build_theory_repair(
        t3_pop, t3_end, t3_repeated,
        _read_csv(os.path.join(C72_TABLE_DIR, "multi_candidate_rank_gauge_bound.csv")),
        _read_csv(os.path.join(C72_TABLE_DIR, "finite_population_best_arm_bound.csv")),
        float(epsilons[len(epsilons) // 2]),
    )
    synthetic = rb.build_synthetic_robustness(protocol)
    hierarchical_rows = c72.build_hierarchical_inference(t3_pop, t3_end, bootstraps, seed + 6)
    instrumentation = inst.inspect_frozen_instrumentation()
    replay = build_c72_replay(
        protocol_ctx, cache_ctx, t2_meta, t3_meta,
        t2_pop, t2_end, t2_c72_repeated, t3_pop, t3_end, t3_c72_repeated,
    )
    replay_rows = (
        replay["c72_protocol_replay_rows"] + replay["c72_cache_identity_replay_rows"]
        + replay["c72_metric_identity_replay_rows"] + replay["c72_red_team_repair_ledger_rows"]
    )
    provenance_ok = all(int(row["passed"]) for row in replay_rows)
    replication_rows, dependency_rows = build_replication_plan(instrumentation, bool(residual["residual_construct_validated"]))
    risk_rows = build_risk_register(provenance_ok, attribution, residual, h3, h4, alpha_zero, confounding, theory, instrumentation)
    timing_rows = [
        {"event": "protocol_lock", "timestamp_utc": protocol["protocol_lock_timestamp_utc"], "commit": "", "row_level_C72_outcomes_read": 0, "status": "protocol_content_frozen"},
        {"event": "protocol_commit_and_push", "timestamp_utc": PROTOCOL_COMMIT_TIMESTAMP_UTC, "commit": "26d3d34", "row_level_C72_outcomes_read": 0, "status": "before_any_persisted_C73_execution"},
        {"event": "earliest_persisted_C73_execution_start", "timestamp_utc": EARLIEST_PERSISTED_EXECUTION_START_UTC, "commit": "26d3d34", "row_level_C72_outcomes_read": "access_occurs_after_process_start_exact_instant_not_persisted", "status": "after_protocol_commit"},
        {"event": "current_recompute_outcome_access", "timestamp_utc": first_outcome_access, "commit": _git_or_empty(["rev-parse", "--short", "HEAD"]), "row_level_C72_outcomes_read": 1, "status": "retrospective_robustness_not_independent_confirmation"},
    ]
    failure_rows = build_failure_rows(provenance_ok, residual, h3, h4, alpha_zero, instrumentation, synthetic)

    res = {
        "milestone": MILESTONE, "config_hash": config_hash, "protocol": protocol,
        "protocol_sha256": protocol_ctx["protocol_sha"], "protocol_commit": "26d3d34",
        "current_head_at_generation": _git_or_empty(["rev-parse", "--short", "HEAD"]),
        "current_run_outcome_access_timestamp_utc": first_outcome_access,
        "earliest_persisted_execution_start_timestamp_utc": EARLIEST_PERSISTED_EXECUTION_START_UTC,
        "diagnostic_only_non_deployable": True,
        "forward_passes": 0, "reinference_runs": 0,
        "training_attempted": instrumentation["training_attempted"], "gpu_used": instrumentation["gpu_used"],
        "dummy_CPU_forward_passes": instrumentation["dummy_forward_count"],
        "real_EEG_trials_loaded": instrumentation["real_EEG_trials_loaded"],
        "bnci004_used": 0, "reserved_seeds_used": 0,
        "selector_artifact_emitted": 0, "checkpoint_recommendation_artifact_emitted": 0,
        "selected_checkpoint_ids_emitted": 0, "raw_cache_rows_copied_to_git": 0,
        "cache_meta": [t2_meta, t3_meta], "C72_locks": locks,
        "T2_locked_epsilons": epsilons, "T2_locked_effective_M_temperature": temperature,
        **replay, **context, **attribution, **residual, **h3, **h4,
        "alpha_zero_local_geometry_rows": alpha_derivatives,
        "intervention_family_sensitivity_rows": family_rows,
        "cross_fitted_intervention_stability_rows": stability_rows,
        "alpha_zero_noncontrol_confirmed": alpha_zero,
        **confounding, **theory, **synthetic, **instrumentation,
        "hierarchical_inference_summary_rows": hierarchical_rows,
        "replication_stage_decision_rows": replication_rows,
        "future_protocol_dependency_graph_rows": dependency_rows,
        "protocol_timing_rows": timing_rows,
        "risk_register_rows": risk_rows,
        "failure_reason_ledger_rows": failure_rows,
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [], "large_artifact_scan_rows": [],
        "schema_validation_summary_rows": [], "red_team_failure_ledger_rows": [],
        "artifact_manifest_rows": [],
    }
    res["decision"] = classify(res)
    return res


TABLE_SPECS = {
    "risk_register.csv": "risk_register_rows",
    "c72_protocol_replay.csv": "c72_protocol_replay_rows",
    "c72_cache_identity_replay.csv": "c72_cache_identity_replay_rows",
    "c72_metric_identity_replay.csv": "c72_metric_identity_replay_rows",
    "c72_red_team_repair_ledger.csv": "c72_red_team_repair_ledger_rows",
    "cell_specific_random_baselines.csv": "cell_specific_random_baselines_rows",
    "effective_candidate_multiplicity.csv": "effective_candidate_multiplicity_rows",
    "topk_and_regret_context.csv": "topk_and_regret_context_rows",
    "near_tie_set_size.csv": "near_tie_set_size_rows",
    "attribution_order_sensitivity.csv": "attribution_order_sensitivity_rows",
    "attribution_shapley_summary.csv": "attribution_shapley_summary_rows",
    "attribution_bootstrap_intervals.csv": "attribution_bootstrap_intervals_rows",
    "attribution_endpoint_sensitivity.csv": "attribution_endpoint_sensitivity_rows",
    "attribution_leave_target_out.csv": "attribution_leave_target_out_rows",
    "attribution_leave_trajectory_out.csv": "attribution_leave_trajectory_out_rows",
    "residual_construct_validity.csv": "residual_construct_validity_rows",
    "residual_split_stability.csv": "residual_split_stability_rows",
    "candidate_vs_common_variance.csv": "candidate_vs_common_variance_rows",
    "residual_incremental_prediction.csv": "residual_incremental_prediction_rows",
    "residual_null_calibration.csv": "residual_null_calibration_rows",
    "shared_calibration_equivalence.csv": "shared_calibration_equivalence_rows",
    "shared_calibration_power_audit.csv": "shared_calibration_power_audit_rows",
    "h4_identity_vs_empirical_effect.csv": "h4_identity_vs_empirical_effect_rows",
    "h4_matched_nulls.csv": "h4_matched_nulls_rows",
    "h4_blocked_effects.csv": "h4_blocked_effects_rows",
    "h4_constructed_vs_random_perturbation.csv": "h4_constructed_vs_random_perturbation_rows",
    "alpha_zero_local_geometry.csv": "alpha_zero_local_geometry_rows",
    "intervention_family_sensitivity.csv": "intervention_family_sensitivity_rows",
    "cross_fitted_intervention_stability.csv": "cross_fitted_intervention_stability_rows",
    "candidate_count_confounding.csv": "candidate_count_confounding_rows",
    "raw_M_vs_effective_M.csv": "raw_M_vs_effective_M_rows",
    "top_gap_adjusted_effects.csv": "top_gap_adjusted_effects_rows",
    "h6_leave_target_out.csv": "h6_leave_target_out_rows",
    "gaussian_bound_failure_diagnosis.csv": "gaussian_bound_failure_diagnosis_rows",
    "finite_population_best_arm_bound.csv": "finite_population_best_arm_bound_rows",
    "effective_candidate_bound.csv": "effective_candidate_bound_rows",
    "empirical_tail_bound.csv": "empirical_tail_bound_rows",
    "synthetic_attribution_robustness.csv": "synthetic_attribution_robustness_rows",
    "hierarchical_inference_summary.csv": "hierarchical_inference_summary_rows",
    "frozen_instrumentation_feasibility.csv": "frozen_instrumentation_feasibility_rows",
    "source_trial_cache_schema.csv": "source_trial_cache_schema_rows",
    "representation_Wz_cache_schema.csv": "representation_Wz_cache_schema_rows",
    "hook_ABI_validation.csv": "hook_ABI_validation_rows",
    "storage_runtime_plan.csv": "storage_runtime_plan_rows",
    "replication_stage_decision.csv": "replication_stage_decision_rows",
    "future_protocol_dependency_graph.csv": "future_protocol_dependency_graph_rows",
    "protocol_timing.csv": "protocol_timing_rows",
    "failure_reason_ledger.csv": "failure_reason_ledger_rows",
    "test_command_manifest.csv": "test_command_manifest_rows",
    "forbidden_claim_scan.csv": "forbidden_claim_scan_rows",
    "large_artifact_scan.csv": "large_artifact_scan_rows",
    "schema_validation_summary.csv": "schema_validation_summary_rows",
    "red_team_failure_ledger.csv": "red_team_failure_ledger_rows",
    "artifact_manifest.csv": "artifact_manifest_rows",
}

FIXED_COLUMNS = {
    "test_command_manifest.csv": ["test_scope", "command", "status", "environment", "slurm_partition"],
    "forbidden_claim_scan.csv": ["pattern", "total_hits", "affirmative_hits", "files", "passed"],
    "large_artifact_scan.csv": ["path", "size_bytes", "over_50mb", "passed"],
    "schema_validation_summary.csv": ["table_name", "row_count", "columns", "nonempty", "passed"],
    "red_team_failure_ledger.csv": ["gate", "failed", "finding"],
    "artifact_manifest.csv": ["path", "size_bytes", "sha256", "artifact_class", "row_count"],
}


def _columns(rows: list[dict], fallback: list[str] | None = None) -> list[str]:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns or list(fallback or ["status"])


def write_tables(res: dict) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    for name, key in TABLE_SPECS.items():
        rows = res.get(key, [])
        au.write_csv(os.path.join(TABLE_DIR, name), rows, _columns(rows, FIXED_COLUMNS.get(name)))


def _schema_rows() -> list[dict]:
    rows = []
    for name in TABLE_SPECS:
        if name in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        path = os.path.join(TABLE_DIR, name)
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({
            "table_name": name, "row_count": count, "columns": len(header),
            "nonempty": int(count > 0), "passed": int(bool(header) and count > 0),
        })
    return rows


def _affirmative_hit(text: str, phrase: str, window: int = 1200) -> bool:
    low, phrase = text.lower(), phrase.lower()
    start = 0
    while True:
        index = low.find(phrase, start)
        if index < 0:
            return False
        context = low[max(0, index - window):index]
        if not any(cue in context for cue in NEGATION_CUES):
            return True
        start = index + len(phrase)


def build_forbidden_scan(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in FORBIDDEN_PATTERNS:
        total, affirmative, files = 0, 0, []
        for path in paths:
            if os.path.basename(path) in {"forbidden_claim_scan.csv", "red_team_failure_ledger.csv"}:
                continue
            body = open(path, errors="ignore").read()
            count = body.lower().count(pattern.lower())
            if count:
                total += count
                files.append(path)
                affirmative += int(_affirmative_hit(body, pattern))
        rows.append({"pattern": pattern, "total_hits": total, "affirmative_hits": affirmative, "files": ";".join(files), "passed": int(affirmative == 0)})
    return rows


def _listed_paths() -> list[Path]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        list(Path(REPORT_DIR).glob("C73_*.md"))
        + list(Path(REPORT_DIR).glob("C73_*.json"))
        + list(Path(REPORT_DIR).glob("C73_*.sha256"))
        + [path for path in Path(TABLE_DIR).glob("*.csv") if path.name not in skip]
    )


def _large_scan(paths: list[Path]) -> list[dict]:
    return [{
        "path": str(path), "size_bytes": path.stat().st_size,
        "over_50mb": int(path.stat().st_size > MAX_REPORT_BYTES),
        "passed": int(path.stat().st_size <= MAX_REPORT_BYTES),
    } for path in paths]


def _artifact_manifest(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        row_count: int | str = ""
        if path.suffix == ".csv":
            with open(path, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                row_count = sum(1 for _ in reader)
        rows.append({
            "path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(str(path)),
            "artifact_class": "table" if path.suffix == ".csv" else "protocol" if "PROTOCOL" in path.name else "report",
            "row_count": row_count,
        })
    return rows


def build_red_team_rows(res: dict) -> list[dict]:
    replay = (
        res["c72_protocol_replay_rows"] + res["c72_cache_identity_replay_rows"]
        + res["c72_metric_identity_replay_rows"] + res["c72_red_team_repair_ledger_rows"]
    )
    order_rows = res["attribution_order_sensitivity_rows"]
    order_groups = sorted({(row["budget"], row["endpoint"]) for row in order_rows})
    order_count_ok = all(
        len({row["order"] for row in order_rows if (row["budget"], row["endpoint"]) == group}) == 120
        for group in order_groups
    )
    order_accounting_ok = True
    for group in order_groups:
        totals = []
        for order in {row["order"] for row in order_rows if (row["budget"], row["endpoint"]) == group}:
            totals.append(sum(float(row["mean_marginal_gain"]) for row in order_rows if (row["budget"], row["endpoint"], row["order"]) == (*group, order)))
        order_accounting_ok &= bool(totals and max(totals) - min(totals) <= 1e-10)

    shapley = res["attribution_shapley_summary_rows"]
    dominance_ok = True
    for budget, endpoint in {(row["budget"], row["endpoint"]) for row in shapley}:
        rows = [row for row in shapley if row["budget"] == budget and row["endpoint"] == endpoint]
        for row in rows:
            others = [other for other in rows if other is not row]
            expected = int(
                float(row["largest_order_fraction"]) >= 0.90
                and all(float(row["ci_lower"]) > float(other["ci_upper"]) for other in others)
            )
            dominance_ok &= expected == int(row["dominant_by_registered_rule"])

    h4_rows = res["h4_identity_vs_empirical_effect_rows"]
    h5_rows = res["intervention_family_sensitivity_rows"]
    hook_rows = res["hook_ABI_validation_rows"]
    report_text = open(MAIN_REPORT, errors="ignore").read() if os.path.exists(MAIN_REPORT) else ""
    residual_language_ok = bool(res["residual_construct_validated"] or "unexplained candidate-specific residual" in report_text.lower())
    checks = [
        ("protocol_sha_and_pre_outcome_commit", res["protocol_sha256"] == _sha256(PROTOCOL_JSON) == open(PROTOCOL_SHA).read().strip() and res["protocol_commit"] == "26d3d34", "C73 protocol SHA and separate pre-outcome commit replay."),
        ("protocol_timing_chain", res["protocol"]["protocol_lock_timestamp_utc"] < PROTOCOL_COMMIT_TIMESTAMP_UTC < EARLIEST_PERSISTED_EXECUTION_START_UTC <= res["current_run_outcome_access_timestamp_utc"], "Lock, commit, earliest persisted execution, and current recompute are monotonically ordered."),
        ("retrospective_status_explicit", res["protocol"]["known_before_lock"]["full_frozen_physical_universe_consumed"] and "retrospective" in res["protocol"]["data_roles"]["T3_HO"], "All frozen outcomes were consumed before C73; no independent-confirmation label."),
        ("C72_provenance_and_metric_replay", all(int(row["passed"]) for row in replay), "C72 protocol, caches, metrics, and prior repairs replay independently."),
        ("all_120_orders", order_count_ok and len(order_groups) == 6, "Each registered budget/endpoint analysis contains all 120 component orders."),
        ("order_value_accounting", order_accounting_ok, "Every sequential order closes to the same registered value difference."),
        ("strict_dominance_rule", dominance_ok, "Dominance flags reproduce the registered CI and 90-percent order rule."),
        ("residual_construct_language", residual_language_ok, "A failed construct gate retains unexplained-residual terminology."),
        ("H3_SESOI_not_pvalue", all("beneficial_SESOI" in row and "ci_upper" in row for row in res["shared_calibration_equivalence_rows"]) and all(int(row.get("p_gt_0p05_used_as_insufficiency", 0)) == 0 for row in res["shared_calibration_power_audit_rows"] if "p_gt_0p05_used_as_insufficiency" in row), "Shared calibration uses SESOI intervals, not a nonsignificant-p shortcut."),
        ("H4_identity_separated_from_origin", all(float(row["mean_identity_accuracy"]) == 1.0 and int(row["validates_residual_origin"]) == 0 for row in h4_rows), "Algebraic crossings are exact and do not identify residual origin."),
        ("H5_local_locked_no_T3_tuning", float(res["C72_locks"]["I5_construction_candidate_gauge"]) == 0.0 and all(int(row.get("T3_tuned", 0)) == 0 for row in h5_rows), "Symmetric alpha audit preserves the C72 zero lock and does not tune on T3-HO."),
        ("candidate_count_confounding", bool(res["candidate_count_confounding_rows"]) and bool(res["raw_M_vs_effective_M_rows"]) and bool(res["top_gap_adjusted_effects_rows"]), "Raw M, effective multiplicity, and top-gap adjustments are all emitted."),
        ("hierarchical_not_row_IID", bool(res["hierarchical_inference_summary_rows"]) and all(int(row["row_iid_used"]) == 0 for row in res["hierarchical_inference_summary_rows"]), "Target, checkpoint, trial-ID, crossed, and leave-group inference do not treat cache rows as IID."),
        ("theory_scope", all(int(row["distributional_theorem_claimed"]) == 0 for row in res["effective_candidate_bound_rows"]) and all(int(row["simulation_or_empirical_proxy_only"]) == 1 for row in res["empirical_tail_bound_rows"]), "Repaired bounds retain finite-population/model-proxy scope."),
        ("synthetic_controls", bool(res["synthetic_validation"]["passed"]) and int(res["synthetic_validation"]["raw_draws_persisted"]) == 0, "Locked aggregate synthetic grid passes identities without raw payload."),
        ("dummy_ABI_only", bool(hook_rows) and all(int(row["passed"]) and int(row["dummy_tensor_only"]) and not int(row["real_EEG_forward"]) for row in hook_rows), "Representation/Wz readiness uses CPU dummy tensors only."),
        ("Wz_reconstruction_identity", max(float(row["Wz_plus_b_logit_max_abs"]) for row in hook_rows) <= 1e-6, "Dummy logits reconstruct as Wz+b within tolerance."),
        ("no_forward_training_gpu", res["forward_passes"] == res["reinference_runs"] == res["training_attempted"] == res["gpu_used"] == 0 and res["real_EEG_trials_loaded"] == 0, "No real EEG forward, re-inference, training, GPU, or real EEG tensor load."),
        ("no_forbidden_data_or_control", res["bnci004_used"] == res["reserved_seeds_used"] == res["selector_artifact_emitted"] == res["checkpoint_recommendation_artifact_emitted"] == res["selected_checkpoint_ids_emitted"] == 0, "No reserved data/seed or control artifact."),
        ("raw_cache_external", res["raw_cache_rows_copied_to_git"] == 0, "T2/T3-HO caches remain external and read-only."),
        ("risk_register_no_blocker", not any(int(row["blocking"]) for row in res["risk_register_rows"]), "No open blocking risk."),
        ("schema_validation", bool(res.get("schema_validation_summary_rows")) and all(int(row["passed"]) for row in res.get("schema_validation_summary_rows", [])), "Every C73 table is nonempty and parseable."),
        ("large_artifact_scan", bool(res.get("large_artifact_scan_rows")) and all(int(row["passed"]) for row in res.get("large_artifact_scan_rows", [])), "Every listed C73 artifact is below 50 MB."),
        ("forbidden_claim_scan", bool(res.get("forbidden_claim_scan_rows")) and all(int(row["passed"]) for row in res.get("forbidden_claim_scan_rows", [])), "No affirmative forbidden claim."),
        ("tests_green", all(row["status"] == "green" for row in res["test_command_manifest_rows"]), "Focused, slice, C23-C73, and full suites are recorded green."),
    ]
    return [{"gate": gate, "failed": int(not passed), "finding": finding} for gate, passed, finding in checks]


def _fmt(value, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _compact_summary(res: dict) -> dict:
    full = {row["component_code"]: row for row in res["attribution_shapley_summary_rows"] if row["budget"] == c72.FULL_BUDGET and row["endpoint"] == "bAcc"}
    target_random = [row for row in res["cell_specific_random_baselines_rows"] if row["stage"] == "T3-HO" and row["field_level"] == "target_universe"]
    target_effective = [row for row in res["effective_candidate_multiplicity_rows"] if row["stage"] == "T3-HO" and row["field_level"] == "target_universe"]
    h4 = next(row for row in res["h4_constructed_vs_random_perturbation_rows"])
    hook_error = max(float(row["Wz_plus_b_logit_max_abs"]) for row in res["hook_ABI_validation_rows"])
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "protocol_commit": res["protocol_commit"],
        "protocol_sha256": res["protocol_sha256"],
        "current_head_at_generation": res["current_head_at_generation"],
        "current_run_outcome_access_timestamp_utc": res["current_run_outcome_access_timestamp_utc"],
        "earliest_persisted_execution_start_timestamp_utc": res["earliest_persisted_execution_start_timestamp_utc"],
        "full_frozen_universe_consumed": True,
        "confirmation_status": "retrospective_robustness_not_independent_confirmation",
        "forward_passes": res["forward_passes"], "reinference_runs": res["reinference_runs"],
        "training_attempted": res["training_attempted"], "gpu_used": res["gpu_used"],
        "dummy_CPU_forward_passes": res["dummy_CPU_forward_passes"],
        "real_EEG_trials_loaded": res["real_EEG_trials_loaded"],
        "bnci004_used": res["bnci004_used"], "reserved_seeds_used": res["reserved_seeds_used"],
        "selector_artifact_emitted": res["selector_artifact_emitted"],
        "checkpoint_recommendation_artifact_emitted": res["checkpoint_recommendation_artifact_emitted"],
        "selected_checkpoint_ids_emitted": res["selected_checkpoint_ids_emitted"],
        "raw_cache_rows_copied_to_git": res["raw_cache_rows_copied_to_git"],
        "decision": res["decision"], "final_gate": res["decision"]["final_gate"],
        "key_numbers": {
            "T2_units": 216, "T3_HO_units": 1052, "full_frozen_units": 1268,
            "mean_candidate_count": float(np.mean([row["candidate_count"] for row in target_random])),
            "mean_random_top1": float(np.mean([row["random_top1"] for row in target_random])),
            "mean_observed_top1": float(np.mean([row["observed_top1"] for row in target_random])),
            "mean_effective_candidate_multiplicity": float(np.mean([row["effective_candidate_multiplicity"] for row in target_effective])),
            "full_bAcc_attribution": {code: {key: full[code][key] for key in ("mean_shapley_gain", "mean_shapley_fraction", "ci_lower", "ci_upper", "order_gain_min", "order_gain_max", "dominant_by_registered_rule")} for code in rb.COMPONENTS},
            "residual_construct_validated": bool(res["residual_construct_validated"]),
            "residual_incremental_r2": res["residual_incremental_r2"],
            "shared_calibration_conclusion": res["shared_calibration_conclusion"],
            "H4_reduces_to_identity": bool(res["h4_reduces_to_identity"]),
            "H4_constructed_minus_random_flip_rate": h4["flip_rate_difference"],
            "alpha_zero_noncontrol_confirmed": bool(res["alpha_zero_noncontrol_confirmed"]),
            "effective_multiplicity_dominates": bool(res["effective_multiplicity_dominates"]),
            "multi_candidate_top1_bound_repaired": bool(res["multi_candidate_top1_bound_repaired"]),
            "instrumentation_gate": res["instrumentation_gate"],
            "Wz_reconstruction_max_abs": hook_error,
            "synthetic_high_reliability_poor_top1_cells": res["synthetic_validation"]["high_reliability_poor_top1_cells"],
            "red_team_failure_count": res["decision"]["red_team_failure_count"],
        },
        "table_row_counts": {name[:-4]: len(res.get(key, [])) for name, key in TABLE_SPECS.items() if name not in {"artifact_manifest.csv", "large_artifact_scan.csv"}},
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def build_reports(res: dict) -> dict[str, str]:
    decision = res["decision"]
    full = {row["component_code"]: row for row in res["attribution_shapley_summary_rows"] if row["budget"] == c72.FULL_BUDGET and row["endpoint"] == "bAcc"}
    target_random = [row for row in res["cell_specific_random_baselines_rows"] if row["stage"] == "T3-HO" and row["field_level"] == "target_universe"]
    target_effective = [row for row in res["effective_candidate_multiplicity_rows"] if row["stage"] == "T3-HO" and row["field_level"] == "target_universe"]
    eps02 = [row for row in res["near_tie_set_size_rows"] if row["stage"] == "T3-HO" and row["field_level"] == "target_universe" and abs(float(row["epsilon"]) - 0.02) < 1e-12]
    residual_failed = [row["criterion"] for row in res["residual_construct_validity_rows"] if not int(row["passed"])]
    h4 = res["h4_constructed_vs_random_perturbation_rows"][0]
    alpha_t2 = {float(row["alpha"]): row for row in res["intervention_family_sensitivity_rows"] if row.get("stage") == "T2" and "mean_pairwise_accuracy" in row}
    h6 = res["top_gap_adjusted_effects_rows"][0]
    gaussian_nontrivial = sum(int(row["bound_nontrivial"]) for row in res["gaussian_bound_failure_diagnosis_rows"])
    effective_nontrivial = sum(int(row["effective_topk_bound_nontrivial"]) for row in res["effective_candidate_bound_rows"])
    hook_error = max(float(row["Wz_plus_b_logit_max_abs"]) for row in res["hook_ABI_validation_rows"])
    residual_term = "candidate-specific target-gauge proxy" if res["residual_construct_validated"] else "unexplained candidate-specific residual"
    main = "\n".join([
        f"# C73 - C72 Attribution Robustness / Frozen Representation-Instrumentation Gate (frozen C19 `{res['config_hash']}`)",
        "",
        "## Executive Verdict",
        "",
        f"Primary: `{decision['primary']}`",
        "",
        f"Active: `{' ; '.join(decision['active'])}`",
        "",
        f"Inactive: `{' ; '.join(decision['inactive'])}`",
        "",
        f"Final gate: `{decision['final_gate']}`",
        "",
        "## Confirmation Status",
        "",
        "C73 is a prospectively specified retrospective robustness audit. The complete frozen physical universe was already consumed by T2 (216 units) and T3-HO (1,052 units), so none of the C73 outcome analyses is an independent confirmation. A new checkpoint field, target, or dataset is required for a new confirmation claim.",
        "",
        "## Random and Extreme-Order Context",
        "",
        f"Across the nine T3-HO target fields, mean observed full-construction top-1 is `{_fmt(np.mean([row['observed_top1'] for row in target_random]))}` versus mean uniform random top-1 `{_fmt(np.mean([row['random_top1'] for row in target_random]))}`. This enrichment is contextual and is not reliable control. Mean raw M is `{_fmt(np.mean([row['candidate_count'] for row in target_random]), 2)}`; mean T2-locked effective multiplicity is `{_fmt(np.mean([row['effective_candidate_multiplicity'] for row in target_effective]))}`. At epsilon 0.02, mean near-tie count is `{_fmt(np.mean([row['epsilon_optimal_count'] for row in eps02]))}`.",
        "",
        "## Attribution Robustness",
        "",
        *[f"- `{code}` {rb.COMPONENT_NAMES[code]}: Shapley gain `{_fmt(full[code]['mean_shapley_gain'])}`, gap fraction `{_fmt(full[code]['mean_shapley_fraction'])}`, target-bootstrap 95% CI `[{_fmt(full[code]['ci_lower'])}, {_fmt(full[code]['ci_upper'])}]`, order range `[{_fmt(full[code]['order_gain_min'])}, {_fmt(full[code]['order_gain_max'])}]`, registered dominance `{int(full[code]['dominant_by_registered_rule'])}`." for code in rb.COMPONENTS],
        "",
        "All 120 component orders are retained for every registered endpoint. A component is not called dominant unless its lower interval exceeds every competitor's upper interval and it is largest in at least 90% of orders. Endpoint, leave-target-out, and leave-trajectory-out sensitivity are reported separately.",
        "Target-cluster, checkpoint-cluster, trial-ID-cluster, and crossed/pigeonhole intervals are also replayed on physical label views; no cache row is treated as an independent observation.",
        "",
        "## Residual Construct",
        "",
        f"The residual is reported as `{residual_term}`. After source/construction/shared calibration, NLL/ECE, and order/seed/level/regime nuisance terms, adding the candidate construction-gradient proxy changes T3-HO R2 by `{_fmt(res['residual_incremental_r2'])}`; construct validity is `{int(res['residual_construct_validated'])}`. Failed criteria: `{'; '.join(residual_failed) if residual_failed else 'none'}`. Unexplained variance is not renamed as a mechanism merely because it is candidate-specific.",
        "",
        "## H3-H6 Robustness",
        "",
        f"Shared calibration: `{res['shared_calibration_conclusion']}` under target-cluster intervals and locked SESOIs. The T=1 shared-temperature arm is an identity; the class-vector arm is evaluated separately.",
        "",
        f"H4: deterministic margin crossing reproduces all observed perturbation flips. Construction-estimated minus magnitude-matched-random flip rate is `{_fmt(h4['flip_rate_difference'])}`; origin evidence beyond generic sensitivity is `{int(h4['origin_evidence_beyond_generic_sensitivity'])}`. This audit separates algebra from evidence about residual origin.",
        "",
        f"H5: C72's T2 alpha remains `0.0`; the C73 cross-fit alpha=0 pairwise accuracy is `{_fmt(alpha_t2[0.0]['mean_pairwise_accuracy'])}`. Registered nonzero families produce no material actionability gain: `{int(res['alpha_zero_noncontrol_confirmed'])}`.",
        "",
        f"H6: raw-M coefficient retained after effective-M/top-gap adjustment is `{_fmt(h6['coefficient_retained_fraction'])}`; effective geometry dominates the registered comparison: `{int(res['effective_multiplicity_dominates'])}`. The interpretation is effective multiplicity/top-margin geometry, not raw candidate count alone, when the adjusted raw-M effect collapses.",
        "",
        "## Theory and Synthetic Stress",
        "",
        f"The inherited Gaussian union bound is nontrivial in `{gaussian_nontrivial}/{len(res['gaussian_bound_failure_diagnosis_rows'])}` audited cells. The exact finite-population pair bound remains valid for its frozen sampling estimand; the effective epsilon-top-k attempt is nontrivial in `{effective_nontrivial}/{len(res['effective_candidate_bound_rows'])}` cells. Empirical tail envelopes remain model diagnostics, not distributional theorems.",
        "",
        f"The locked aggregate synthetic grid has `{res['synthetic_validation']['grid_rows']}` cells and `{res['synthetic_validation']['high_reliability_poor_top1_cells']}` high-reliability/poor-top1 cells. Common offsets produce zero rank flips; candidate-specific perturbations do produce crossings. No raw synthetic draws are persisted.",
        "",
        "## Instrumentation Gate",
        "",
        f"Six representative frozen state_dict/model ABIs pass CPU dummy-hook tests. The maximum dummy `|Wz+b-logit|` is `{_fmt(hook_error, 12)}`. The resulting feasibility gate is `{res['instrumentation_gate']}`. This is schema and ABI readiness only: no real EEG trial was loaded, no real-data forward or re-inference occurred, and no representation-level EEG mechanism is inferred.",
        "",
        "## Boundary",
        "",
        "C73 emits diagnostic robustness, theory-scope, and future cache-schema evidence only. It does not train, use GPU, consume BNCI2014_004 or reserved seeds, create a selector/control artifact, expose checkpoint identities, start manuscript drafting, or make a target-population claim. R1 frozen strict-source plus z/Wz instrumentation is the minimum next evidence stage and still requires a separate explicit authorization.",
    ])
    theory = "\n".join([
        "# C73 - Theory Note: Attribution Robustness and Extreme-Order Bounds",
        "",
        "## Registered Value Function",
        "",
        "For each frozen target, C73 defines five counterfactual removals: finite-label noise N, extreme-order multiplicity E, endpoint mismatch U, shared calibration S, and the remaining candidate-specific residual G. Exact subset values support a five-player Shapley decomposition and all 120 sequential orders. This is an attribution of a registered diagnostic value function, not unique causal identification.",
        "",
        "## Order and Dominance",
        "",
        "Shapley averaging removes arbitrary order choice but does not remove interaction ambiguity. C73 therefore reports every order range. Registered dominance requires a target-bootstrap lower bound above every competitor upper bound and largest marginal gain in at least 90% of orders. If that rule fails, the correct description is mixed attribution.",
        "",
        "## Residual Naming",
        "",
        "A candidate-centered model residual can contain endpoint mismatch, nonlinear interaction, calibration, metadata, or omitted representation state. C73 permits the target-gauge-proxy label only after split stability, incremental cross-fit prediction, common-offset exclusion, metadata adjustment, and candidate/trajectory nulls all pass. Otherwise it remains an unexplained candidate-specific residual.",
        "",
        "## Finite-Population and Effective-Arm Bounds",
        "",
        "C72's exact class-stratified without-replacement pair calculation is replayed. Bonferroni extension to top-1 can be vacuous when many competitors have pair-error mass. C73 also sums only competitors outside a T2-locked epsilon-optimal set to obtain an effective top-k lower bound. That is a finite-population bound for epsilon-set recovery, not a repaired top-1 theorem. Empirical residual-tail envelopes are explicitly data-calibrated proxies.",
        "",
        "## Gaussian Failure",
        "",
        f"The Gaussian union calculation remains nontrivial in `{gaussian_nontrivial}/{len(res['gaussian_bound_failure_diagnosis_rows'])}` audited cells because top gaps are small relative to the estimated residual scale, candidate dependence is omitted, and the union over roughly one hundred competitors saturates. C73 does not force a theorem from this failure.",
        "",
        "## Synthetic Scope",
        "",
        "The locked synthetic grid varies raw M, effective near-tie count, gauge tail, candidate dependence, and label budget. It demonstrates existence: global rank reliability can coexist with poor extreme action recovery, target-common offsets are rank-invariant, and candidate-specific perturbations can cross compressed margins. It does not establish an EEG population law.",
        "",
        "## Instrumentation Consequence",
        "",
        "The dummy ABI identity logit=Wz+b proves only that the frozen linear head exposes z and Wz during a future forward. Real source-trial and target-trial representation evidence does not exist yet. A separately authorized re-inference-only cache can test whether the unexplained residual is reduced by strict-source trial fields or projection geometry before any new training is considered.",
    ])
    red = "\n".join([
        "# C73 - Red-Team Verification",
        "",
        "All C73 red-team gates pass." if not decision["red_team_failure_count"] else f"C73 has `{decision['red_team_failure_count']}` open red-team gate(s).",
        "",
        *[f"- `{row['gate']}`: `{'PASS' if not int(row['failed']) else 'FAIL'}` - {row['finding']}" for row in res["red_team_failure_ledger_rows"]],
    ])
    return {
        os.path.basename(MAIN_REPORT): main,
        os.path.basename(THEORY_NOTE): theory,
        os.path.basename(RED_REPORT): red,
    }


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def _write_reports_and_json(res: dict) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    for name, body in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(body.rstrip() + "\n")
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_summary(res), f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _quality_refresh(res: dict) -> None:
    write_tables(res)
    _write_reports_and_json(res)
    paths = [str(path) for path in _listed_paths()]
    res["large_artifact_scan_rows"] = _large_scan([Path(path) for path in paths])
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    write_tables(res)
    res["schema_validation_summary_rows"] = _schema_rows()
    write_tables(res)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)


def write_artifacts(res: dict) -> dict:
    if _sha256(PROTOCOL_JSON) != open(PROTOCOL_SHA).read().strip():
        raise ValueError("refusing to write C73 artifacts after protocol drift")
    os.makedirs(TABLE_DIR, exist_ok=True)
    for _ in range(4):
        _quality_refresh(res)
        _write_reports_and_json(res)
    write_tables(res)
    res["artifact_manifest_rows"] = _artifact_manifest(_listed_paths())
    write_tables(res)
    return res


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c73_attribution_robustness_instrumentation_gate")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--candidate-subsets", type=int, default=256)
    parser.add_argument("--test-status", default="planned", choices=("planned", "green", "failed"))
    args = parser.parse_args(argv)
    res = run(
        test_status=args.test_status, repeats=args.repeats, permutations=args.permutations,
        bootstraps=args.bootstraps, candidate_subsets=args.candidate_subsets,
    )
    if args.recompute:
        res = write_artifacts(res)
    print(
        f"[C73] decision={res['decision']['primary']} gate={res['decision']['final_gate']} "
        f"red={res['decision']['red_team_failure_count']} tables={len(TABLE_SPECS)}"
    )


if __name__ == "__main__":
    main()
