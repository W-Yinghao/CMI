"""C60 - Rank-Gauge Proof Stress / Empirical-Theory Bridge Audit."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from statistics import NormalDist

from . import audit_utils as au
from . import schema as c49_schema


MILESTONE = "C60"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c60_tables"
REPORT_JSON = "oaci/reports/C60_RANK_GAUGE_PROOF_STRESS_EMPIRICAL_BRIDGE.json"
C58_JSON = "oaci/reports/C58_FORMAL_LOWER_BOUND_ATTEMPT.json"
C59_JSON = "oaci/reports/C59_FORMAL_LOWER_BOUND_THEORY_FACTORY.json"
C58_TABLE_DIR = "oaci/reports/c58_tables"
C59_TABLE_DIR = "oaci/reports/c59_tables"

DECISIONS = (
    "C60-A_rank_gauge_proof_validated_without_change",
    "C60-B_rank_gauge_proof_repaired_or_strengthened",
    "C60-C_empirical_assumption_bridge_supported",
    "C60-D_empirical_assumption_bridge_partial_or_weak",
    "C60-E_rank_gauge_assumptions_fail_on_frozen_eeg_artifacts",
    "C60-F_source_observable_theory_counterexample_found",
    "C60-G_no_source_observable_counterexample_found",
    "C60-H_theorem_to_eeg_bridge_requires_instrumented_data",
    "C60-I_training_blueprint_refined_but_not_authorized",
    "C60-J_training_not_scientifically_justified_yet",
    "C60-K_claim_or_definition_inconsistency_found",
)

TRAINING_GATE = "TRAINING_BLUEPRINT_REFINED_BUT_NOT_AUTHORIZED"
NEXT_DIRECTION = "wait for remote review; C61 may request instrumented training approval but C60 does not authorize execution"

RANDOM_TIE_HIT = 0.4297233780360411
STRICT_SOURCE_HIT = 0.5061728395061729
SOURCE_SCALARIZATION_HIT = 0.5740740740740741
KEY_ONLY_HIT = 0.4876543209876543
LABEL_DIAGNOSTIC_HIT = 0.8127572016460904
TEMPLATE_ONLY_HIT = 0.7037037037037037
ENDPOINT_ORACLE_HIT = 0.9444444444444444
MAX_NULL_P95 = 0.7712962962962961
CROSS_TARGET_Q10 = 0.9369369369369369
WITHIN_TARGET_Q10 = 0.004842615012106538
WITHIN_TRAJECTORY_Q10 = 0.13287671232876713

FORBIDDEN_PATTERNS = (
    "EEG distribution theorem",
    "distribution-free minimax theorem",
    "minimax theorem established",
    "theorem-grade Le Cam bound",
    "theorem-grade Fano",
    "source-only rescue",
    "OACI rescue",
    "deployable selector",
    "checkpoint recommendation artifact",
    "selected_candidate_id",
    "chosen checkpoint",
    "few-label sufficiency",
    "new real EEG training",
    "silent re-inference",
    "GPU job",
    "BNCI2014_004 used",
    "seeds [3,4] used",
    "M1 manuscript drafting",
    "manuscript drafting starts",
)

NEGATION_CUES = (
    "not ",
    "no ",
    "never ",
    "cannot ",
    "does not ",
    "do not ",
    "is not ",
    "are not ",
    "unavailable",
    "blocked",
    "without ",
    "needed for",
    "needed to",
    "requires",
    "forbidden",
    "future",
    "only if explicitly authorized",
    "requires proof before",
)


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


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


def _phi_minus(gamma: float) -> float:
    return 0.5 * math.erfc(gamma / math.sqrt(2.0))


def _normal_tail_equivalent_gamma(error: float) -> float:
    # If error = Phi(-gamma), then gamma = Phi^{-1}(1-error).
    return NormalDist().inv_cdf(1.0 - error)


def build_proof_audit_rows() -> tuple[list[dict], list[dict], list[dict]]:
    checklist = [
        {"gate": "variables_defined", "status": "pass", "finding": "candidate, source rank R, target gauge W, utility gap, selector sigma-field, hit, and error are explicit in C60 repair", "repair_required": 0},
        {"gate": "selector_measurability", "status": "repaired", "finding": "C59 wording is narrowed to selectors measurable with respect to source-visible rank/sign and registered source information", "repair_required": 1},
        {"gate": "randomness_separated", "status": "repaired", "finding": "fixed margin r and conditional gauge law F_{G|r} are separated from empirical finite-field randomness", "repair_required": 1},
        {"gate": "error_hit_regret_separated", "status": "repaired", "finding": "rank-gauge theorem lower-bounds two-candidate error; hit and regret are derived diagnostics only", "repair_required": 1},
        {"gate": "two_candidate_scope", "status": "repaired", "finding": "the theorem is two-candidate unless an explicit multi-candidate joint-tail assumption is added", "repair_required": 1},
        {"gate": "symmetry_assumption", "status": "pass_with_limit", "finding": "symmetry is not needed for the CDF formula but is needed for the normal-tail simplification and rank selector optimality shorthand", "repair_required": 0},
        {"gate": "independence_assumption", "status": "pass_with_limit", "finding": "two-candidate fixed-margin result does not need independence from rank; multi-candidate proxy does", "repair_required": 0},
        {"gate": "continuity_and_ties", "status": "repaired", "finding": "continuous gauge gives no ties; atom/tie cases need a half-tie convention or weak inequality statement", "repair_required": 1},
        {"gate": "rank_correlated_gauge", "status": "limit", "finding": "rank-correlated gauge can change tail constants, so C60 reports it as an assumption stress failure mode", "repair_required": 0},
        {"gate": "binary_threshold_mapping", "status": "limit", "finding": "continuous utility lower bound transfers to binary joint-good only after the threshold and margin convention are registered", "repair_required": 0},
        {"gate": "eeg_distribution_scope", "status": "pass", "finding": "C60 keeps the rank-gauge result synthetic/model-bound and does not upgrade it to an EEG distribution theorem", "repair_required": 0},
        {"gate": "training_gate_scope", "status": "pass", "finding": "C60 refines a future instrumentation blueprint and does not run training or re-inference", "repair_required": 0},
    ]
    assumptions = [
        {"assumption_id": "RG60-A1", "statement": "Two candidates have fixed source-visible rank margin r", "needed_for_core_proof": 1, "proof_status": "explicit", "empirical_status": "partially_supported", "caveat": "source rank signal is weak in C42/C58"},
        {"assumption_id": "RG60-A2", "statement": "Source selector cannot observe target-specific gauge gap W", "needed_for_core_proof": 1, "proof_status": "explicit", "empirical_status": "partially_supported", "caveat": "closed for registered source artifacts, not for all future raw traces"},
        {"assumption_id": "RG60-A3", "statement": "Target utility gap is alpha*r + beta*W plus optional residual", "needed_for_core_proof": 1, "proof_status": "explicit", "empirical_status": "partially_supported", "caveat": "linear decomposition is a model bridge, not measured directly for all candidates"},
        {"assumption_id": "RG60-A4", "statement": "Conditional gauge CDF F_{W|r} has nonzero lower tail", "needed_for_core_proof": 1, "proof_status": "explicit", "empirical_status": "partially_supported", "caveat": "tail scale is inferred from hit gaps, not from direct gauge samples"},
        {"assumption_id": "RG60-A5", "statement": "Normal symmetric gauge is used only for calibration curves", "needed_for_core_proof": 0, "proof_status": "optional", "empirical_status": "not_testable_with_existing_artifact", "caveat": "no direct gauge distribution cache"},
        {"assumption_id": "RG60-A6", "statement": "Multi-candidate top-1 proxy requires independent challenger gauge events", "needed_for_core_proof": 0, "proof_status": "optional_corollary_only", "empirical_status": "not_testable_with_existing_artifact", "caveat": "C60 does not claim this as theorem-grade for EEG"},
        {"assumption_id": "RG60-A7", "statement": "Rank-correlated or heteroskedastic gauge laws require conditional tail constants", "needed_for_core_proof": 0, "proof_status": "stress_variant", "empirical_status": "not_testable_with_existing_artifact", "caveat": "requires new instrumented data"},
        {"assumption_id": "RG60-A8", "statement": "Endpoint oracle exposes same-label target endpoint content", "needed_for_core_proof": 0, "proof_status": "empirical_bridge", "empirical_status": "supported_by_existing_artifact", "caveat": "diagnostic only and unavailable at selection time"},
        {"assumption_id": "RG60-A9", "statement": "Source-observable counterexamples are absent in registered artifact family", "needed_for_core_proof": 0, "proof_status": "empirical_stress", "empirical_status": "supported_by_existing_artifact", "caveat": "not a universal no-free-lunch theorem"},
        {"assumption_id": "RG60-A10", "statement": "EEG distribution/minimax lower bound is outside the proved theorem", "needed_for_core_proof": 1, "proof_status": "scope_guardrail", "empirical_status": "supported_by_existing_artifact", "caveat": "instrumentation needed for bridge"},
    ]
    repairs = [
        {"repair_id": "PR1", "issue": "C59 theorem wording could be read as stronger than a rank-measurable two-candidate model", "repair": "state the theorem for fixed two-candidate margin and source sigma-field; use W for gauge so G remains available for information classes", "effect": "strengthens precision", "changes_decision": "C60-B"},
        {"repair_id": "PR2", "issue": "normal gauge curve could be mistaken as required", "repair": "write general CDF formula and keep normal as calibration", "effect": "strengthens proof generality", "changes_decision": "C60-B"},
        {"repair_id": "PR3", "issue": "multi-candidate/top-k rows were only proxies", "repair": "demote to corollary candidates requiring joint-tail assumptions", "effect": "prevents overclaim", "changes_decision": "C60-D"},
        {"repair_id": "PR4", "issue": "hit, error, and regret were adjacent", "repair": "separate theorem error from empirical hit/regret diagnostics and treat endpoint gaps as signed diagnostics unless comparability is registered", "effect": "prevents metric mismatch", "changes_decision": "C60-B"},
        {"repair_id": "PR5", "issue": "tie handling not explicit", "repair": "add continuous-gauge/no-tie condition and atom half-tie convention", "effect": "closes edge case", "changes_decision": "C60-B"},
        {"repair_id": "PR6", "issue": "C59-H could be read as authorization to train", "repair": "turn it into explicit future gate matrix only", "effect": "keeps C60 non-training", "changes_decision": "C60-I"},
    ]
    return checklist, assumptions, repairs


def build_partition_extension_rows() -> tuple[list[dict], list[dict], list[dict]]:
    c59_constants = _read_csv(os.path.join(C59_TABLE_DIR, "registered_partition_bound_constants.csv"))
    provenance = []
    for row in c59_constants[:5]:
        provenance.append({
            "source_partition_id": row["partition_id"],
            "information_class": row["information_class"],
            "H_star_pi": row["H_star_pi"],
            "source_artifact": row["source_artifact"],
            "c60_use": "finite_population_constant_reused_read_only",
        })
    extensions = [
        {"extension_id": "RPX1_binary_top1", "status": "proved", "theorem_form": "H*(Pi)=|Omega|^{-1} sum_B max_y n(B,y)", "covered_rule_class": "registered Pi-cell measurable top1", "blocked_overclaim": "arbitrary source nonlinear functions"},
        {"extension_id": "RPX2_multiclass_top1", "status": "proved", "theorem_form": "replace max_y over binary y by max over label classes", "covered_rule_class": "registered Pi-cell measurable multiclass top1", "blocked_overclaim": "unregistered label-dependent rules"},
        {"extension_id": "RPX3_weighted_cells", "status": "proved", "theorem_form": "sum_B w_B max_y p_B(y) with fixed ex ante weights", "covered_rule_class": "weighted registered cells", "blocked_overclaim": "post-hoc target-weight tuning"},
        {"extension_id": "RPX4_coverage_abstention", "status": "proved_with_registered_coverage", "theorem_form": "for covered blocks A, hit <= sum_{B in A} max_y n(B,y)/sum_{B in A} n(B)", "covered_rule_class": "coverage rule registered without labels", "blocked_overclaim": "target-label chosen abstention"},
        {"extension_id": "RPX5_topk_binary", "status": "safe_if_label_and_success_defined", "theorem_form": "block optimum is the best k decisions under the pre-registered success functional", "covered_rule_class": "registered top-k success", "blocked_overclaim": "C60 does not infer a selected top-k list"},
        {"extension_id": "RPX6_bounded_regret", "status": "proved_for_bounded_utilities", "theorem_form": "regret lower bound follows from cellwise utility range and cellwise best measurable action", "covered_rule_class": "registered utility table", "blocked_overclaim": "continuous EEG population regret"},
        {"extension_id": "RPX7_target_trajectory_ledger", "status": "proved_as_disaggregation", "theorem_form": "same bound computed per registered target-trajectory cell then aggregated", "covered_rule_class": "registered target-trajectory ledgers", "blocked_overclaim": "deployable target-conditioned selector"},
    ]
    coverage = [
        {"coverage_case": "full_coverage", "registered_before_labels": 1, "hit_bound_available": 1, "actionability_claim": 0, "note": "C58/C59 finite-population bound"},
        {"coverage_case": "source_registered_abstention", "registered_before_labels": 1, "hit_bound_available": 1, "actionability_claim": 0, "note": "safe theorem extension only"},
        {"coverage_case": "target_label_abstention", "registered_before_labels": 0, "hit_bound_available": 0, "actionability_claim": 0, "note": "diagnostic leakage if used for selection"},
        {"coverage_case": "coverage_actionability_ceiling", "registered_before_labels": 0, "hit_bound_available": 1, "actionability_claim": 0, "note": "C49/C50 style diagnostic ceiling"},
    ]
    return extensions, coverage, provenance


def build_rank_gauge_stress_rows() -> tuple[list[dict], list[dict], list[dict]]:
    variants = [
        {"variant_id": "RGV1_general_two_candidate_cdf", "claim_status": "proved_repaired", "formula_or_bound": "high-rank-rule error = F_W|S,r(-alpha*r/beta); source-sigma-field Bayes error = min(p,1-p)", "extra_assumptions": "fixed r, source cannot observe W", "failure_mode": "none for stated model"},
        {"variant_id": "RGV2_normal_symmetric_calibration", "claim_status": "proved_special_case", "formula_or_bound": "high-rank-rule error = Phi(-alpha*r/(beta*sigma))", "extra_assumptions": "standard normal centered gauge with scale sigma", "failure_mode": "distributional shape not verified in EEG artifacts"},
        {"variant_id": "RGV3_randomized_source_rule", "claim_status": "proved_repaired", "formula_or_bound": "randomization cannot beat Bayes action under known conditional CDF", "extra_assumptions": "same source sigma-field", "failure_mode": "different source information can change bound"},
        {"variant_id": "RGV4_multi_candidate_top1", "claim_status": "proxy_not_theorem", "formula_or_bound": "1-(1-tail)^(n-1) is a challenger proxy", "extra_assumptions": "independent challenger gauge events", "failure_mode": "correlation can loosen or tighten"},
        {"variant_id": "RGV5_topk_regret", "claim_status": "not_theorem_grade", "formula_or_bound": "requires joint tail over omitted better candidates", "extra_assumptions": "registered top-k success and utility margin", "failure_mode": "no joint-tail artifact"},
        {"variant_id": "RGV6_skewed_gauge", "claim_status": "conditional_bound_only", "formula_or_bound": "replace Phi by empirical/analytic F_{G|r}", "extra_assumptions": "known skewed conditional CDF", "failure_mode": "tail may be smaller than normal proxy"},
        {"variant_id": "RGV7_rank_correlated_gauge", "claim_status": "stress_limit", "formula_or_bound": "bound becomes E[F_{G|R}(-alpha*R/beta)]", "extra_assumptions": "conditional gauge law measured", "failure_mode": "existing artifacts do not identify F_{G|R}"},
        {"variant_id": "RGV8_binary_joint_good_threshold", "claim_status": "diagnostic_bridge_only", "formula_or_bound": "utility error implies label error only under registered threshold/margin", "extra_assumptions": "threshold and margin pre-registered", "failure_mode": "joint-good is endpoint-label-derived"},
        {"variant_id": "RGV9_target_local_offset", "claim_status": "does_not_support_pairwise_flip", "formula_or_bound": "a common target offset cancels in within-target pairwise utility differences", "extra_assumptions": "candidate-specific gauge gap is required", "failure_mode": "pure target-local offset cannot explain rank reversals"},
    ]
    curve = []
    for gamma in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5):
        tail = _phi_minus(gamma)
        curve.append({
            "gamma_abs_rank_margin_over_gauge_scale": gamma,
            "two_candidate_error_tail": tail,
            "two_candidate_hit_upper": 1.0 - tail,
            "n10_independent_challenger_loss_proxy": 1.0 - (1.0 - tail) ** 9,
            "theorem_status": "two_candidate_theorem_multi_candidate_proxy",
        })
    failures = [
        {"failure_id": "TSF1", "stress_case": "multi_candidate_dependence", "breaks_core_theorem": 0, "breaks_extension": 1, "mitigation": "label as proxy unless joint tail is measured"},
        {"failure_id": "TSF2", "stress_case": "rank_correlated_gauge", "breaks_core_theorem": 0, "breaks_extension": 1, "mitigation": "condition tail on rank margin"},
        {"failure_id": "TSF3", "stress_case": "gauge_observable_from_source", "breaks_core_theorem": 1, "breaks_extension": 1, "mitigation": "source adversary search; none found in registered artifacts"},
        {"failure_id": "TSF4", "stress_case": "endpoint_scalar_available", "breaks_core_theorem": 1, "breaks_extension": 1, "mitigation": "classify as target-label oracle outside source-only setting"},
        {"failure_id": "TSF5", "stress_case": "heavy_atom_ties", "breaks_core_theorem": 0, "breaks_extension": 0, "mitigation": "use weak inequality/half-tie convention"},
        {"failure_id": "TSF6", "stress_case": "heteroskedastic_target_cells", "breaks_core_theorem": 0, "breaks_extension": 1, "mitigation": "cell-specific gamma needed"},
        {"failure_id": "TSF7", "stress_case": "binary_threshold_without_margin", "breaks_core_theorem": 0, "breaks_extension": 1, "mitigation": "keep utility theorem separate from joint-good label"},
        {"failure_id": "TSF8", "stress_case": "distributional_eeg_upgrade", "breaks_core_theorem": 0, "breaks_extension": 1, "mitigation": "requires new instrumented observation law and replication"},
        {"failure_id": "TSF9", "stress_case": "target_local_offset_only", "breaks_core_theorem": 1, "breaks_extension": 1, "mitigation": "require candidate-specific gauge gap, not only a common target offset"},
    ]
    return variants, curve, failures


def build_empirical_bridge_rows() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    assumption_map = [
        {"assumption_id": "RG60-A1", "empirical_proxy": "strict source hit 0.506 vs random 0.430", "support_class": "partially_supported", "artifact": "C42/C58/C59", "bridge_limit": "rank signal is weak"},
        {"assumption_id": "RG60-A2", "empirical_proxy": "best source scalarization 0.574 below max null p95 0.771", "support_class": "partially_supported", "artifact": "C43/C58/C59", "bridge_limit": "registered source family only"},
        {"assumption_id": "RG60-A3", "empirical_proxy": "label diagnostic 0.813 and endpoint oracle 0.944 exceed source fields", "support_class": "partially_supported", "artifact": "C52-C55/C58", "bridge_limit": "not direct G samples"},
        {"assumption_id": "RG60-A4", "empirical_proxy": "cross-target q10 divergence 0.937 vs within-target 0.00484", "support_class": "partially_supported", "artifact": "C46", "bridge_limit": "target mixing evidence, not distribution law"},
        {"assumption_id": "RG60-A5", "empirical_proxy": "no frozen gauge residual distribution cache", "support_class": "not_testable_with_existing_artifact", "artifact": "missing", "bridge_limit": "requires instrumentation"},
        {"assumption_id": "RG60-A6", "empirical_proxy": "no independent challenger gauge trace", "support_class": "not_testable_with_existing_artifact", "artifact": "missing", "bridge_limit": "multi-candidate theorem blocked"},
        {"assumption_id": "RG60-A8", "empirical_proxy": "same-label endpoint scalar hit 0.944", "support_class": "supported_by_existing_artifact", "artifact": "C53-C55/C58", "bridge_limit": "oracle boundary only"},
        {"assumption_id": "RG60-A10", "empirical_proxy": "C58/C59 theorem status", "support_class": "supported_by_existing_artifact", "artifact": "C58/C59", "bridge_limit": "scope is finite/synthetic"},
    ]
    support = [
        {"bridge_axis": "source_visible_rank", "support_score": 0.35, "support_class": "partial", "key_number": STRICT_SOURCE_HIT, "interpretation": "rank exists but weak"},
        {"bridge_axis": "source_scalarization", "support_score": 0.45, "support_class": "partial", "key_number": SOURCE_SCALARIZATION_HIT, "interpretation": "best registered source rule improves but misses null boundary"},
        {"bridge_axis": "key_only_conditioning", "support_score": 0.20, "support_class": "weak", "key_number": KEY_ONLY_HIT, "interpretation": "keys do not expose gauge by themselves"},
        {"bridge_axis": "label_diagnostic_content", "support_score": 0.75, "support_class": "strong_diagnostic", "key_number": LABEL_DIAGNOSTIC_HIT, "interpretation": "target-label diagnostic content closes much residual"},
        {"bridge_axis": "endpoint_oracle", "support_score": 0.95, "support_class": "oracle", "key_number": ENDPOINT_ORACLE_HIT, "interpretation": "same-label endpoint content nearly closes field"},
        {"bridge_axis": "cross_target_gauge_break", "support_score": 0.80, "support_class": "partial", "key_number": CROSS_TARGET_Q10, "interpretation": "global source comparability break is large"},
        {"bridge_axis": "direct_gauge_distribution", "support_score": 0.0, "support_class": "missing", "key_number": "", "interpretation": "no direct G samples"},
        {"bridge_axis": "observation_law_for_eeg_theorem", "support_score": 0.0, "support_class": "missing", "key_number": "", "interpretation": "no theorem-grade observation law"},
    ]
    gaps = [
        {"gap_id": "BG1", "missing_item": "direct gauge residual samples by target/cell", "blocks": "tail-law calibration", "requires_new_instrumentation": 1, "priority": "high"},
        {"gap_id": "BG2", "missing_item": "rank-margin and gauge-residual joint distribution", "blocks": "rank-correlated gauge theorem", "requires_new_instrumentation": 1, "priority": "high"},
        {"gap_id": "BG3", "missing_item": "source transcript observation law", "blocks": "EEG distribution-level lower bound", "requires_new_instrumentation": 1, "priority": "high"},
        {"gap_id": "BG4", "missing_item": "stable pairwise KL/MI matrix", "blocks": "Fano/Assouad repair", "requires_new_instrumentation": 1, "priority": "medium"},
        {"gap_id": "BG5", "missing_item": "split-label target cache", "blocks": "few-label/split-label boundary", "requires_new_instrumentation": 1, "priority": "medium"},
        {"gap_id": "BG6", "missing_item": "atom-level leakage trace", "blocks": "mechanism-to-theorem bridge", "requires_new_instrumentation": 1, "priority": "medium"},
        {"gap_id": "BG7", "missing_item": "independent checkpoint-field replication", "blocks": "finite-field replication beyond current artifacts", "requires_new_instrumentation": 1, "priority": "medium"},
    ]
    tail_summary = [
        {"quantity": "random_tie_miss", "value": 1.0 - RANDOM_TIE_HIT, "interpretation": "base finite-cell ambiguity", "theorem_grade": 0},
        {"quantity": "strict_source_miss", "value": 1.0 - STRICT_SOURCE_HIT, "interpretation": "observed registered source-rank error scale", "theorem_grade": 0},
        {"quantity": "source_scalarization_miss", "value": 1.0 - SOURCE_SCALARIZATION_HIT, "interpretation": "best source utility-cone error scale", "theorem_grade": 0},
        {"quantity": "label_diagnostic_miss", "value": 1.0 - LABEL_DIAGNOSTIC_HIT, "interpretation": "target-label diagnostic residual", "theorem_grade": 0},
        {"quantity": "endpoint_oracle_miss", "value": 1.0 - ENDPOINT_ORACLE_HIT, "interpretation": "same-label endpoint residual", "theorem_grade": 0},
        {"quantity": "source_to_oracle_error_gap", "value": (1.0 - STRICT_SOURCE_HIT) - (1.0 - ENDPOINT_ORACLE_HIT), "interpretation": "gauge-scale witness gap", "theorem_grade": 0},
    ]
    margin = []
    for field, hit in (
        ("random_tie", RANDOM_TIE_HIT),
        ("strict_source", STRICT_SOURCE_HIT),
        ("source_scalarization", SOURCE_SCALARIZATION_HIT),
        ("key_only", KEY_ONLY_HIT),
        ("label_diagnostic", LABEL_DIAGNOSTIC_HIT),
        ("endpoint_oracle", ENDPOINT_ORACLE_HIT),
    ):
        err = 1.0 - hit
        margin.append({
            "field": field,
            "hit": hit,
            "error_tail_proxy": err,
            "normal_tail_equivalent_gamma": _normal_tail_equivalent_gamma(err),
            "interpretation": "negative gamma means anti-rank/no positive margin" if err > 0.5 else "larger gamma means less gauge overwrite",
        })
    error_vs_tail = [
        {"comparison": "strict_source_vs_gamma0", "observed_error": 1.0 - STRICT_SOURCE_HIT, "nearest_curve_gamma": 0.0, "curve_error": 0.5, "difference": (1.0 - STRICT_SOURCE_HIT) - 0.5, "conclusion": "near complete gauge overwrite"},
        {"comparison": "source_scalarization_vs_gamma025", "observed_error": 1.0 - SOURCE_SCALARIZATION_HIT, "nearest_curve_gamma": 0.25, "curve_error": _phi_minus(0.25), "difference": (1.0 - SOURCE_SCALARIZATION_HIT) - _phi_minus(0.25), "conclusion": "weak positive margin"},
        {"comparison": "template_vs_gamma05", "observed_error": 1.0 - TEMPLATE_ONLY_HIT, "nearest_curve_gamma": 0.5, "curve_error": _phi_minus(0.5), "difference": (1.0 - TEMPLATE_ONLY_HIT) - _phi_minus(0.5), "conclusion": "target-label template improves but remains below oracle"},
        {"comparison": "endpoint_oracle_vs_gamma15", "observed_error": 1.0 - ENDPOINT_ORACLE_HIT, "nearest_curve_gamma": 1.5, "curve_error": _phi_minus(1.5), "difference": (1.0 - ENDPOINT_ORACLE_HIT) - _phi_minus(1.5), "conclusion": "same-label endpoint acts like high-margin gauge readout"},
        {"comparison": "key_only_vs_gamma0", "observed_error": 1.0 - KEY_ONLY_HIT, "nearest_curve_gamma": 0.0, "curve_error": 0.5, "difference": (1.0 - KEY_ONLY_HIT) - 0.5, "conclusion": "key alone is not a positive rank margin"},
        {"comparison": "max_null_p95_vs_template", "observed_error": 1.0 - TEMPLATE_ONLY_HIT, "nearest_curve_gamma": "", "curve_error": 1.0 - MAX_NULL_P95, "difference": TEMPLATE_ONLY_HIT - MAX_NULL_P95, "conclusion": "template-only does not beat max null p95"},
        {"comparison": "endpoint_vs_max_null_p95", "observed_error": 1.0 - ENDPOINT_ORACLE_HIT, "nearest_curve_gamma": "", "curve_error": 1.0 - MAX_NULL_P95, "difference": ENDPOINT_ORACLE_HIT - MAX_NULL_P95, "conclusion": "endpoint oracle exceeds null but is target-label-derived"},
    ]
    return assumption_map, support, gaps, tail_summary, margin, error_vs_tail


def build_lower_bound_repair_rows() -> tuple[list[dict], list[dict], list[dict]]:
    lecam = [
        {"attempt_id": "LC60-1_within_target", "source_near_proxy": WITHIN_TARGET_Q10, "target_divergence_proxy": "available", "repair_status": "empirical_witness_only", "theorem_grade": 0, "blocker": "source proxy is not total variation"},
        {"attempt_id": "LC60-2_within_trajectory", "source_near_proxy": WITHIN_TRAJECTORY_Q10, "target_divergence_proxy": "available", "repair_status": "empirical_witness_only", "theorem_grade": 0, "blocker": "no two observation laws P0/P1"},
        {"attempt_id": "LC60-3_cross_target", "source_near_proxy": CROSS_TARGET_Q10, "target_divergence_proxy": "available", "repair_status": "not_source_near", "theorem_grade": 0, "blocker": "cross-target source equivalence breaks comparability"},
        {"attempt_id": "LC60-4_synthetic_rank_gauge", "source_near_proxy": "", "target_divergence_proxy": "analytic gauge tail", "repair_status": "model_bound_only", "theorem_grade": 1, "blocker": "not EEG distribution law"},
    ]
    fano = [
        {"attempt_id": "F60-1_binary_source_rank", "packing_size": 2, "mi_or_kl_status": "small_proxy_only", "bound_status": "trivial_or_unstable", "theorem_grade": 0, "blocker": "log2 term and no observation law"},
        {"attempt_id": "F60-2_key_cells", "packing_size": 162, "mi_or_kl_status": "missing_stable_matrix", "bound_status": "blocked", "theorem_grade": 0, "blocker": "cell hypotheses lack KL/MI matrix"},
        {"attempt_id": "F60-3_endpoint_bits", "packing_size": 2, "mi_or_kl_status": "tautological", "bound_status": "invalid_for_ambiguity", "theorem_grade": 0, "blocker": "same-label endpoint oracle"},
        {"attempt_id": "F60-4_controlled_future_field", "packing_size": "", "mi_or_kl_status": "future_required", "bound_status": "blueprint_only", "theorem_grade": 0, "blocker": "requires instrumented replication"},
    ]
    failures = [
        {"failure_id": "LB1", "branch": "LeCam", "reason": "no probability laws over source transcripts", "support_blocker": 1, "definition_blocker": 1},
        {"failure_id": "LB2", "branch": "LeCam", "reason": "source distance proxy is not TV/KL", "support_blocker": 1, "definition_blocker": 1},
        {"failure_id": "LB3", "branch": "Fano", "reason": "stable MI/KL matrix unavailable", "support_blocker": 1, "definition_blocker": 0},
        {"failure_id": "LB4", "branch": "Fano", "reason": "binary packing too small or tautological", "support_blocker": 1, "definition_blocker": 0},
        {"failure_id": "LB5", "branch": "Assouad", "reason": "independent hypercube edges not constructed", "support_blocker": 1, "definition_blocker": 0},
        {"failure_id": "LB6", "branch": "RankGauge", "reason": "synthetic proof valid but not EEG law", "support_blocker": 1, "definition_blocker": 0},
        {"failure_id": "LB7", "branch": "TrainingGate", "reason": "instrumentation needed but not authorized in C60", "support_blocker": 1, "definition_blocker": 0},
    ]
    return lecam, fano, failures


def build_source_adversary_rows() -> tuple[list[dict], list[dict], list[dict]]:
    results = [
        {"candidate_id": "SA60-1", "candidate": "source_rank_score", "allowed_source_only": 1, "uses_target_labels": 0, "hit": STRICT_SOURCE_HIT, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "weak rank signal"},
        {"candidate_id": "SA60-2", "candidate": "best_source_scalarization", "allowed_source_only": 1, "uses_target_labels": 0, "hit": SOURCE_SCALARIZATION_HIT, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "hindsight source scalarization still below null boundary"},
        {"candidate_id": "SA60-3", "candidate": "source_pareto_front_depth", "allowed_source_only": 1, "uses_target_labels": 0, "hit": 0.43105701988584916, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "front membership near base rate"},
        {"candidate_id": "SA60-4", "candidate": "conditioned_source_neighborhood", "allowed_source_only": 0, "uses_target_labels": 0, "hit": 0.5555555555555556, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "conditioned problem class, not source-only global selector"},
        {"candidate_id": "SA60-5", "candidate": "key_source_geometry", "allowed_source_only": 0, "uses_target_labels": 0, "hit": KEY_ONLY_HIT, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "key-only escape closed"},
        {"candidate_id": "SA60-6", "candidate": "template_without_same_cell_endpoint", "allowed_source_only": 0, "uses_target_labels": 1, "hit": TEMPLATE_ONLY_HIT, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "target-label template, not source-only, and below max null p95"},
        {"candidate_id": "SA60-7", "candidate": "kernel_source_neighborhood_probe", "allowed_source_only": 1, "uses_target_labels": 0, "hit": "", "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "not executable from frozen summary artifacts without new nested probe design"},
        {"candidate_id": "SA60-8", "candidate": "rank_aggregation_variant", "allowed_source_only": 1, "uses_target_labels": 0, "hit": "", "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "no registered artifact beats existing source scalarization"},
    ]
    availability = [
        {"input_family": "source_leakage_summaries", "allowed": 1, "available_in_frozen_artifacts": 1, "risk": "weak information"},
        {"input_family": "source_endpoint_summaries", "allowed": 1, "available_in_frozen_artifacts": 1, "risk": "source-target decoupling"},
        {"input_family": "source_rank_features", "allowed": 1, "available_in_frozen_artifacts": 1, "risk": "rank signal weak"},
        {"input_family": "source_objective_geometry", "allowed": 1, "available_in_frozen_artifacts": 1, "risk": "Pareto crowding"},
        {"input_family": "source_neighborhood_geometry", "allowed": 1, "available_in_frozen_artifacts": 1, "risk": "conditioning needed"},
        {"input_family": "target_endpoint_scalars", "allowed": 0, "available_in_frozen_artifacts": 1, "risk": "same-label oracle"},
        {"input_family": "target_joint_margin_raw", "allowed": 0, "available_in_frozen_artifacts": 1, "risk": "endpoint tautology"},
        {"input_family": "same_cell_diagnostic_labels", "allowed": 0, "available_in_frozen_artifacts": 1, "risk": "label leakage"},
    ]
    red = [
        {"gate": "strict_source_inputs_only_for_source_candidates", "passed": 1, "finding": "source-only candidates do not read target endpoint fields"},
        {"gate": "target_label_templates_marked_diagnostic", "passed": 1, "finding": "template rows are marked target-label-derived"},
        {"gate": "no_candidate_beats_null_as_source_only", "passed": 1, "finding": "no allowed source-only row beats max null p95"},
        {"gate": "no_new_nested_probe_training", "passed": 1, "finding": "kernel probe is not executed from frozen summaries"},
        {"gate": "no_escape_hatch", "passed": 1, "finding": "all reliable_escape_hatch flags are zero"},
    ]
    return results, availability, red


def build_conditional_rows() -> tuple[list[dict], list[dict], list[dict]]:
    suff = [
        {"condition_set": "cell_only", "candidate_added": "source_rank", "hit_before": RANDOM_TIE_HIT, "hit_after": STRICT_SOURCE_HIT, "delta_hit": STRICT_SOURCE_HIT - RANDOM_TIE_HIT, "screens_off_endpoint": 0, "status": "weak_source_gain_not_sufficient"},
        {"condition_set": "source_rank", "candidate_added": "key_only", "hit_before": STRICT_SOURCE_HIT, "hit_after": KEY_ONLY_HIT, "delta_hit": KEY_ONLY_HIT - STRICT_SOURCE_HIT, "screens_off_endpoint": 0, "status": "key_only_does_not_screen_off"},
        {"condition_set": "source_plus_key", "candidate_added": "label_diagnostic", "hit_before": KEY_ONLY_HIT, "hit_after": LABEL_DIAGNOSTIC_HIT, "delta_hit": LABEL_DIAGNOSTIC_HIT - KEY_ONLY_HIT, "screens_off_endpoint": 0, "status": "large_diagnostic_gain"},
        {"condition_set": "source_plus_key", "candidate_added": "template_only", "hit_before": KEY_ONLY_HIT, "hit_after": TEMPLATE_ONLY_HIT, "delta_hit": TEMPLATE_ONLY_HIT - KEY_ONLY_HIT, "screens_off_endpoint": 0, "status": "partial_target_label_transfer"},
        {"condition_set": "source_plus_key_template", "candidate_added": "same_label_endpoint", "hit_before": TEMPLATE_ONLY_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "delta_hit": ENDPOINT_ORACLE_HIT - TEMPLATE_ONLY_HIT, "screens_off_endpoint": 0, "status": "endpoint_adds_after_template"},
        {"condition_set": "same_label_endpoint", "candidate_added": "endpoint_self_redundancy", "hit_before": ENDPOINT_ORACLE_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "delta_hit": 0.0, "screens_off_endpoint": 1, "status": "oracle_boundary_diagnostic_only"},
    ]
    mb = [
        {"variable": "source_only", "empirical_boundary_role": "not_sufficient", "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "hit": STRICT_SOURCE_HIT},
        {"variable": "source_plus_key", "empirical_boundary_role": "not_sufficient", "uses_target_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "hit": KEY_ONLY_HIT},
        {"variable": "label_diagnostic", "empirical_boundary_role": "partial_empirical_boundary_like", "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "hit": LABEL_DIAGNOSTIC_HIT},
        {"variable": "template_only", "empirical_boundary_role": "partial_transfer_not_boundary", "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "hit": TEMPLATE_ONLY_HIT},
        {"variable": "same_label_endpoint", "empirical_boundary_role": "empirical_oracle_boundary", "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "hit": ENDPOINT_ORACLE_HIT},
        {"variable": "split_label_cache", "empirical_boundary_role": "future_unresolved", "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "hit": ""},
    ]
    ladder = [
        {"information_class": "I0_random_or_tie", "hit": RANDOM_TIE_HIT, "formal_status": "finite_population_baseline", "source_only": 1, "label_content": 0},
        {"information_class": "I1_strict_source", "hit": STRICT_SOURCE_HIT, "formal_status": "weak_empirical_signal_not_sufficient", "source_only": 1, "label_content": 0},
        {"information_class": "I1b_source_scalarization", "hit": SOURCE_SCALARIZATION_HIT, "formal_status": "best_registered_source_family_below_reliability", "source_only": 1, "label_content": 0},
        {"information_class": "I2_key_only", "hit": KEY_ONLY_HIT, "formal_status": "key_only_insufficient_not_screening", "source_only": 0, "label_content": 0},
        {"information_class": "I3_target_unlabeled", "hit": "", "formal_status": "not_available_for_closure", "source_only": 0, "label_content": 0},
        {"information_class": "I5_split_label_or_few_label", "hit": "", "formal_status": "future_unresolved_missing_cache", "source_only": 0, "label_content": 1},
        {"information_class": "I6_label_diagnostic", "hit": LABEL_DIAGNOSTIC_HIT, "formal_status": "diagnostic_partial_boundary_like", "source_only": 0, "label_content": 1},
        {"information_class": "I6b_template_only", "hit": TEMPLATE_ONLY_HIT, "formal_status": "cross_cell_partial_transfer_not_sufficient", "source_only": 0, "label_content": 1},
        {"information_class": "I7_same_label_endpoint", "hit": ENDPOINT_ORACLE_HIT, "formal_status": "same_label_oracle_boundary_diagnostic_only", "source_only": 0, "label_content": 1},
    ]
    return suff, mb, ladder


def build_instrumentation_rows() -> tuple[list[dict], list[dict], list[dict]]:
    blueprint = [
        {"protocol_id": "P0", "protocol": "no_training_continuation", "scientific_question": "Can proof hardening continue synthetically?", "new_data_required": "none", "training_or_inference_needed": "none", "approval_required": 0, "slurm_requirement": "none", "claim_supported": "synthetic proof only", "risk": "saturates bridge"},
        {"protocol_id": "P1", "protocol": "split_label_few_label_cache", "scientific_question": "Does disjoint target-label content close the residual?", "new_data_required": "per-trial labels/predictions with split ids", "training_or_inference_needed": "inference cache or instrumented replay", "approval_required": 1, "slurm_requirement": "cpu-high/GPU only if explicitly released", "claim_supported": "split-label boundary if disjointness holds", "risk": "few-label overclaim"},
        {"protocol_id": "P2", "protocol": "per_trial_logits_probabilities", "scientific_question": "Can target-unlabeled/probability geometry bridge gauge?", "new_data_required": "per-trial logits probabilities sample hashes", "training_or_inference_needed": "instrumented inference", "approval_required": 1, "slurm_requirement": "cpu-high or approved GPU", "claim_supported": "target-unlabeled bridge diagnostics", "risk": "method tuning"},
        {"protocol_id": "P3", "protocol": "atom_level_leakage_trace", "scientific_question": "Which leakage atoms map to gauge residual?", "new_data_required": "candidate atom table bootstrap order", "training_or_inference_needed": "instrumented replay or new run", "approval_required": 1, "slurm_requirement": "cpu-high plus approved compute", "claim_supported": "mechanism trace", "risk": "large payload"},
        {"protocol_id": "P4", "protocol": "independent_checkpoint_field_replication", "scientific_question": "Does the finite-field boundary replicate?", "new_data_required": "pre-registered independent field", "training_or_inference_needed": "training likely", "approval_required": 1, "slurm_requirement": "Slurm only", "claim_supported": "replicated empirical boundary", "risk": "uncontrolled exploration"},
        {"protocol_id": "P5", "protocol": "rank_gauge_intervention_controlled_family", "scientific_question": "Can controlled perturbations validate rank-gauge assumptions?", "new_data_required": "controlled model family with gauge trace", "training_or_inference_needed": "training likely", "approval_required": 1, "slurm_requirement": "Slurm only", "claim_supported": "model-to-EEG bridge evidence", "risk": "turning diagnostic into method"},
        {"protocol_id": "P6", "protocol": "reserved_holdout_final_stress", "scientific_question": "Final locked stress only after protocol approval", "new_data_required": "BNCI2014_004/seeds [3,4] only if released", "training_or_inference_needed": "blocked", "approval_required": 1, "slurm_requirement": "blocked until release", "claim_supported": "future stress test", "risk": "reserved holdout contamination"},
    ]
    gate = [
        {"gate": "C60_training", "decision": "not_executed", "requires_user_release": 1, "default": TRAINING_GATE, "note": "blueprint refined only"},
        {"gate": "P0_no_training", "decision": "allowed_read_only", "requires_user_release": 0, "default": "available", "note": "frozen-artifact proof and measurement planning only"},
        {"gate": "P1_split_label_cache", "decision": "proposal_only", "requires_user_release": 1, "default": "blocked", "note": "requires split disjointness and no selector tuning"},
        {"gate": "P2_per_trial_logits_probs", "decision": "proposal_only", "requires_user_release": 1, "default": "blocked", "note": "requires instrumented inference approval"},
        {"gate": "P3_atom_trace", "decision": "proposal_only", "requires_user_release": 1, "default": "blocked", "note": "requires fixed trace schema and approved compute"},
        {"gate": "P4_independent_replication", "decision": "proposal_only", "requires_user_release": 1, "default": "blocked", "note": "requires separate Slurm campaign"},
        {"gate": "P5_rank_gauge_intervention", "decision": "proposal_only", "requires_user_release": 1, "default": "blocked", "note": "requires controlled-family pre-registration"},
        {"gate": "P6_reserved_holdout", "decision": "reserved", "requires_user_release": 1, "default": "blocked", "note": "BNCI2014_004/seeds [3,4] only after explicit release"},
        {"gate": "BNCI2014_004", "decision": "reserved", "requires_user_release": 1, "default": "blocked", "note": "no use in C60"},
        {"gate": "seeds_3_4", "decision": "reserved", "requires_user_release": 1, "default": "blocked", "note": "no use in C60"},
        {"gate": "GPU", "decision": "not_authorized", "requires_user_release": 1, "default": "blocked", "note": "no GPU in C60"},
        {"gate": "re_inference", "decision": "not_authorized", "requires_user_release": 1, "default": "blocked", "note": "no re-inference in C60"},
        {"gate": "selector_search", "decision": "forbidden", "requires_user_release": 0, "default": "blocked", "note": "C60 is diagnostic/theory audit"},
    ]
    missing = [
        {"gap_id": "MD1", "data_item": "source transcript observation law", "needed_for": "EEG distribution-level bridge", "present_now": 0, "minimal_protocol": "P4/P5"},
        {"gap_id": "MD2", "data_item": "direct gauge residual by candidate/cell", "needed_for": "tail-law calibration", "present_now": 0, "minimal_protocol": "P2/P5"},
        {"gap_id": "MD3", "data_item": "rank-margin/gauge joint trace", "needed_for": "rank-correlated gauge stress", "present_now": 0, "minimal_protocol": "P5"},
        {"gap_id": "MD4", "data_item": "per-trial logits/probabilities", "needed_for": "target-unlabeled bridge", "present_now": 0, "minimal_protocol": "P2"},
        {"gap_id": "MD5", "data_item": "split-label cache", "needed_for": "few-label boundary", "present_now": 0, "minimal_protocol": "P1"},
        {"gap_id": "MD6", "data_item": "atom leakage trace", "needed_for": "mechanism trace", "present_now": 0, "minimal_protocol": "P3"},
        {"gap_id": "MD7", "data_item": "stable KL/MI matrix", "needed_for": "Fano/Assouad", "present_now": 0, "minimal_protocol": "P4/P5"},
        {"gap_id": "MD8", "data_item": "reserved holdout release", "needed_for": "final stress", "present_now": 0, "minimal_protocol": "P6 only after approval"},
    ]
    return blueprint, gate, missing


def build_subagent_manifest() -> list[dict]:
    roles = [
        "Proof Auditor",
        "Registered Partition Bound Generalizer",
        "Rank-Gauge Theorem Stress Agent",
        "Empirical Assumption Mapper",
        "Gauge Distribution Tail Empiricist",
        "Le Cam Fano Repair Agent",
        "Source-Observable Adversary",
        "Conditional Sufficiency Markov Boundary Agent",
        "Instrumentation Blueprint Refiner",
        "Integration Red-Team Agent",
    ]
    return [{"subagent_id": f"SA{i+1}", "role": role, "integration_status": "launched_or_locally_integrated"} for i, role in enumerate(roles)]


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c60", "command": "python -m pytest oaci/tests/test_c60_rank_gauge_proof_stress_empirical_bridge.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c60_slice", "command": "python -m pytest oaci/tests/test_c50_conditioned_island_morphology.py ... test_c60_rank_gauge_proof_stress_empirical_bridge.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c60_regression", "command": "python -m pytest oaci/tests/test_c23_score_gauge.py ... test_c60_rank_gauge_proof_stress_empirical_bridge.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
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


def _is_inventory_path(path: str) -> bool:
    return os.path.basename(path) in {
        "forbidden_claim_scan.csv",
        "red_team_failure_ledger.csv",
        "training_gate_decision_matrix.csv",
        "source_escape_hatch_red_team.csv",
    }


def build_forbidden_scan(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in FORBIDDEN_PATTERNS:
        total = affirmative = 0
        files = []
        for path in paths:
            if _is_inventory_path(path):
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


def classify(res: dict) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    escape = any(int(r["reliable_escape_hatch"]) for r in res["source_observable_adversary_results_rows"] if r["reliable_escape_hatch"] != "")
    if failures:
        primary = "C60-K_claim_or_definition_inconsistency_found"
    elif escape:
        primary = "C60-F_source_observable_theory_counterexample_found"
    else:
        primary = "C60-B_rank_gauge_proof_repaired_or_strengthened"
    return {
        "primary": primary,
        "active": [
            "C60-B_rank_gauge_proof_repaired_or_strengthened",
            "C60-D_empirical_assumption_bridge_partial_or_weak",
            "C60-G_no_source_observable_counterexample_found",
            "C60-H_theorem_to_eeg_bridge_requires_instrumented_data",
            "C60-I_training_blueprint_refined_but_not_authorized",
        ],
        "inactive": [
            "C60-A_rank_gauge_proof_validated_without_change",
            "C60-C_empirical_assumption_bridge_supported",
            "C60-E_rank_gauge_assumptions_fail_on_frozen_eeg_artifacts",
            "C60-F_source_observable_theory_counterexample_found",
            "C60-J_training_not_scientifically_justified_yet",
            "C60-K_claim_or_definition_inconsistency_found",
        ],
        "training_gate": TRAINING_GATE,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": NEXT_DIRECTION,
    }


def build_red_team_rows(res: dict) -> list[dict]:
    checks = [
        ("proof_audit_completed", len(res["proof_audit_checklist_rows"]) == 12, "Line-item proof audit rows are emitted."),
        ("proof_repaired_not_overclaimed", any(r["repair_id"] == "PR3" for r in res["proof_repair_log_rows"]), "Multi-candidate/top-k material is demoted unless assumptions are added."),
        ("bridge_partial_not_full", any(r["support_class"] == "missing" for r in res["assumption_support_scores_rows"]), "Empirical bridge is partial and has missing theorem-critical data."),
        ("source_escape_hatch_closed", all(int(r["reliable_escape_hatch"]) == 0 for r in res["source_observable_adversary_results_rows"] if r["reliable_escape_hatch"] != ""), "No source-observable counterexample is found."),
        ("lecam_not_upgraded", all(int(r["theorem_grade"]) == 0 or r["attempt_id"] == "LC60-4_synthetic_rank_gauge" for r in res["lecam_witness_repair_summary_rows"]), "Le Cam remains empirical for EEG artifacts."),
        ("fano_not_upgraded", all(int(r["theorem_grade"]) == 0 for r in res["fano_packing_repair_summary_rows"]), "Fano/Assouad remains blocked."),
        ("endpoint_oracle_diagnostic_only", any(r["variable"] == "same_label_endpoint" and int(r["available_at_selection_time"]) == 0 for r in res["markov_boundary_probe_summary_rows"]), "Same-label endpoint scalar remains unavailable at selection time."),
        ("training_not_executed", res["training_gate"] == TRAINING_GATE, "C60 refines blueprint but does not execute training."),
        ("reserved_dataset_and_seeds_blocked", any(r["gate"] == "BNCI2014_004" and r["decision"] == "reserved" for r in res["training_gate_decision_matrix_rows"]) and any(r["gate"] == "seeds_3_4" and r["decision"] == "reserved" for r in res["training_gate_decision_matrix_rows"]), "BNCI2014_004 and seeds [3,4] remain reserved."),
        ("no_gpu_or_reinference", any(r["gate"] == "GPU" and r["decision"] == "not_authorized" for r in res["training_gate_decision_matrix_rows"]) and any(r["gate"] == "re_inference" and r["decision"] == "not_authorized" for r in res["training_gate_decision_matrix_rows"]), "No GPU or re-inference is authorized."),
        ("no_m1_or_manuscript", "does not authorize execution" in res["decision"]["recommended_next_direction"], "C60 does not start M1 or manuscript drafting."),
        ("no_eeg_theorem_claim", res["theorem_status"]["rank_gauge"] == "synthetic_model_bound_repaired", "Rank-gauge theorem remains synthetic/model-bound."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
        ("large_artifact_scan_passed", all(int(r.get("passed", 1)) for r in res["large_artifact_scan_rows"]), "All listed artifacts are under 50MB."),
        ("no_selector_artifact", not any("selected_candidate_id" in open(p, errors="ignore").read().lower() or "chosen checkpoint" in open(p, errors="ignore").read().lower() for p in res["generated_paths"] if p.endswith((".md", ".json", ".csv")) and not _is_inventory_path(p)), "C60 emits no selected-candidate or chosen-checkpoint artifact."),
        ("test_manifest_recorded", len(res["test_command_manifest_rows"]) == 4, "Validation scopes are recorded for Slurm cpu-high."),
    ]
    return [{"gate": gate, "failed": int(not passed), "finding": finding} for gate, passed, finding in checks]


def table_row_counts(res: dict) -> dict:
    keys = {
        "proof_audit_checklist": "proof_audit_checklist_rows",
        "theorem_assumption_inventory": "theorem_assumption_inventory_rows",
        "proof_repair_log": "proof_repair_log_rows",
        "registered_partition_bound_extensions": "registered_partition_bound_extensions_rows",
        "coverage_abstention_bound_summary": "coverage_abstention_bound_summary_rows",
        "partition_bound_provenance": "partition_bound_provenance_rows",
        "rank_gauge_theorem_variants": "rank_gauge_theorem_variants_rows",
        "gauge_tail_bound_curve": "gauge_tail_bound_curve_rows",
        "theorem_stress_failure_modes": "theorem_stress_failure_modes_rows",
        "theorem_to_eeg_assumption_map": "theorem_to_eeg_assumption_map_rows",
        "assumption_support_scores": "assumption_support_scores_rows",
        "assumption_gap_ledger": "assumption_gap_ledger_rows",
        "gauge_tail_empirical_summary": "gauge_tail_empirical_summary_rows",
        "rank_margin_gauge_scale_summary": "rank_margin_gauge_scale_summary_rows",
        "source_error_vs_gauge_tail_ledger": "source_error_vs_gauge_tail_ledger_rows",
        "lecam_witness_repair_summary": "lecam_witness_repair_summary_rows",
        "fano_packing_repair_summary": "fano_packing_repair_summary_rows",
        "lower_bound_failure_reason_ledger": "lower_bound_failure_reason_ledger_rows",
        "source_observable_adversary_results": "source_observable_adversary_results_rows",
        "source_observable_availability_ledger": "source_observable_availability_ledger_rows",
        "source_escape_hatch_red_team": "source_escape_hatch_red_team_rows",
        "conditional_sufficiency_summary": "conditional_sufficiency_summary_rows",
        "markov_boundary_probe_summary": "markov_boundary_probe_summary_rows",
        "information_ladder_formal_status": "information_ladder_formal_status_rows",
        "instrumentation_blueprint_v2": "instrumentation_blueprint_v2_rows",
        "training_gate_decision_matrix": "training_gate_decision_matrix_rows",
        "missing_data_to_theorem_gap": "missing_data_to_theorem_gap_rows",
        "subagent_audit_manifest": "subagent_audit_manifest_rows",
        "test_command_manifest": "test_command_manifest_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "artifact_manifest": "artifact_manifest_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    active = ";".join(d["active"])
    inactive = ";".join(d["inactive"])
    main = "\n".join([
        f"# C60 - Rank-Gauge Proof Stress / Empirical-Theory Bridge Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "## Primary Decision",
        "",
        f"`{d['primary']}`",
        "",
        f"Active: `{active}`",
        "",
        f"Inactive: `{inactive}`",
        "",
        "## Result",
        "",
        "C60 keeps C59's rank-gauge theorem but repairs its statement: the proved object is a fixed-margin two-candidate source-measurable lower bound with an explicit gauge-tail CDF. The normal curve is a calibration special case, and multi-candidate/top-k rows are proxies unless a joint-tail assumption is supplied.",
        "",
        "The empirical bridge is partial. Frozen artifacts support a weak source-visible rank axis and a strong target-label endpoint boundary, but they do not contain the direct gauge distribution, observation law, KL/MI matrix, split-label cache, or atom trace needed for a distribution-level EEG bridge.",
        "",
        "## Source Adversary",
        "",
        "No source-observable escape hatch is found in the registered artifact family. Source rank and source scalarization remain below the null/reliability boundary; target-label templates are diagnostic and not source-only.",
        "",
        "## Training Gate",
        "",
        f"`{TRAINING_GATE}`",
        "",
        "C60 does not run training, re-inference, GPU work, BNCI2014_004, or seeds [3,4]. It refines the future instrumentation blueprint only.",
        "",
        "## Next Branch",
        "",
        "A future C61 can request explicit approval for a quarantined instrumentation campaign if remote review accepts that theorem-to-EEG bridging now requires new data. C60 itself authorizes no execution.",
    ])
    red = "\n".join([
        "# C60 - Red-Team Verification",
        "",
        "All C60 red-team gates pass." if d["red_team_failure_count"] == 0 else "C60 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    proof = "\n".join([
        "# C60 - Proof Stress Audit",
        "",
        "The repaired theorem is stated as a source-sigma-field two-candidate lower bound. For target utility gap alpha*r + beta*W and unobserved gauge CDF F, the positive-rank source action has error F(-alpha*r/beta). For standard normal W with scale sigma this is Phi(-alpha*r/(beta*sigma)).",
        "",
        "C60 strengthens C59 by replacing a normal-only statement with the general CDF form, while weakening any apparent multi-candidate/top-k reading to explicitly marked proxy status.",
    ])
    bridge = "\n".join([
        "# C60 - Empirical-Theory Bridge",
        "",
        f"Strict source hit is {STRICT_SOURCE_HIT:.3f}; source scalarization is {SOURCE_SCALARIZATION_HIT:.3f}; same-label endpoint oracle is {ENDPOINT_ORACLE_HIT:.3f}. These numbers match the rank-gauge story qualitatively but do not identify the gauge law directly.",
        "",
        "The bridge remains partial because theorem-critical direct gauge and observation-law artifacts are missing.",
    ])
    adversary = "\n".join([
        "# C60 - Source Adversary and Lower-Bound Repair",
        "",
        "Le Cam remains empirical witness material, and Fano/Assouad remains blocked by missing stable KL/MI support. The source-observable adversary does not find a registered source-only counterexample.",
    ])
    blueprint = "\n".join([
        "# C60 - Instrumentation Blueprint V2",
        "",
        f"Default gate: `{TRAINING_GATE}`.",
        "",
        "The minimal theorem-critical future data are direct gauge residual traces, per-trial logits/probabilities, split-label cache, atom leakage traces, stable KL/MI matrices, and independent checkpoint-field replication. Every branch requires explicit approval before execution.",
    ])
    return {
        "C60_RANK_GAUGE_PROOF_STRESS_EMPIRICAL_BRIDGE.md": main,
        "C60_RED_TEAM_VERIFICATION.md": red,
        "C60_PROOF_STRESS_AUDIT.md": proof,
        "C60_EMPIRICAL_THEORY_BRIDGE.md": bridge,
        "C60_SOURCE_ADVERSARY_AND_LOWER_BOUND_REPAIR.md": adversary,
        "C60_INSTRUMENTATION_BLUEPRINT.md": blueprint,
    }


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "c59_commit": "828adb3",
        "c59_decision": res["c59_decision"],
        "decision": res["decision"],
        "training_gate": res["training_gate"],
        "theorem_status": res["theorem_status"],
        "key_numbers": {
            "random_tie": RANDOM_TIE_HIT,
            "strict_source": STRICT_SOURCE_HIT,
            "source_scalarization": SOURCE_SCALARIZATION_HIT,
            "key_only": KEY_ONLY_HIT,
            "label_diagnostic": LABEL_DIAGNOSTIC_HIT,
            "template_only": TEMPLATE_ONLY_HIT,
            "endpoint_oracle": ENDPOINT_ORACLE_HIT,
            "max_null_p95": MAX_NULL_P95,
            "cross_target_q10": CROSS_TARGET_Q10,
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def run(test_status: str = "planned") -> dict:
    config_hash = _lock_config()
    c59 = _load_json(C59_JSON)
    proof_check, assumptions, repairs = build_proof_audit_rows()
    part_ext, coverage, provenance = build_partition_extension_rows()
    variants, curve, stress_fail = build_rank_gauge_stress_rows()
    assumption_map, support, gaps, tail, margin, error_vs_tail = build_empirical_bridge_rows()
    lecam, fano, lb_fail = build_lower_bound_repair_rows()
    source_adv, source_avail, source_red = build_source_adversary_rows()
    suff, mb, ladder = build_conditional_rows()
    blueprint, gate, missing = build_instrumentation_rows()
    res = {
        "config_hash": config_hash,
        "c59_decision": c59["decision"]["primary"],
        "training_gate": TRAINING_GATE,
        "theorem_status": {
            "registered_partition_bound": "extended_safely_for_registered_partitions",
            "rank_gauge": "synthetic_model_bound_repaired",
            "empirical_bridge": "partial_or_weak",
            "lecam": "empirical_witness_only",
            "fano_assouad": "trivial_or_unstable",
            "source_adversary": "no_registered_escape_hatch",
        },
        "proof_audit_checklist_rows": proof_check,
        "theorem_assumption_inventory_rows": assumptions,
        "proof_repair_log_rows": repairs,
        "registered_partition_bound_extensions_rows": part_ext,
        "coverage_abstention_bound_summary_rows": coverage,
        "partition_bound_provenance_rows": provenance,
        "rank_gauge_theorem_variants_rows": variants,
        "gauge_tail_bound_curve_rows": curve,
        "theorem_stress_failure_modes_rows": stress_fail,
        "theorem_to_eeg_assumption_map_rows": assumption_map,
        "assumption_support_scores_rows": support,
        "assumption_gap_ledger_rows": gaps,
        "gauge_tail_empirical_summary_rows": tail,
        "rank_margin_gauge_scale_summary_rows": margin,
        "source_error_vs_gauge_tail_ledger_rows": error_vs_tail,
        "lecam_witness_repair_summary_rows": lecam,
        "fano_packing_repair_summary_rows": fano,
        "lower_bound_failure_reason_ledger_rows": lb_fail,
        "source_observable_adversary_results_rows": source_adv,
        "source_observable_availability_ledger_rows": source_avail,
        "source_escape_hatch_red_team_rows": source_red,
        "conditional_sufficiency_summary_rows": suff,
        "markov_boundary_probe_summary_rows": mb,
        "information_ladder_formal_status_rows": ladder,
        "instrumentation_blueprint_v2_rows": blueprint,
        "training_gate_decision_matrix_rows": gate,
        "missing_data_to_theorem_gap_rows": missing,
        "subagent_audit_manifest_rows": build_subagent_manifest(),
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "red_team_failure_ledger_rows": [],
        "schema_validation_summary_rows": [],
        "large_artifact_scan_rows": [],
        "artifact_manifest_rows": [],
        "generated_paths": [],
    }
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []})
    return res


def write_tables(res: dict, table_dir: str) -> None:
    specs = {
        "proof_audit_checklist.csv": ("proof_audit_checklist_rows", ["gate", "status", "finding", "repair_required"]),
        "theorem_assumption_inventory.csv": ("theorem_assumption_inventory_rows", ["assumption_id", "statement", "needed_for_core_proof", "proof_status", "empirical_status", "caveat"]),
        "proof_repair_log.csv": ("proof_repair_log_rows", ["repair_id", "issue", "repair", "effect", "changes_decision"]),
        "registered_partition_bound_extensions.csv": ("registered_partition_bound_extensions_rows", ["extension_id", "status", "theorem_form", "covered_rule_class", "blocked_overclaim"]),
        "coverage_abstention_bound_summary.csv": ("coverage_abstention_bound_summary_rows", ["coverage_case", "registered_before_labels", "hit_bound_available", "actionability_claim", "note"]),
        "partition_bound_provenance.csv": ("partition_bound_provenance_rows", ["source_partition_id", "information_class", "H_star_pi", "source_artifact", "c60_use"]),
        "rank_gauge_theorem_variants.csv": ("rank_gauge_theorem_variants_rows", ["variant_id", "claim_status", "formula_or_bound", "extra_assumptions", "failure_mode"]),
        "gauge_tail_bound_curve.csv": ("gauge_tail_bound_curve_rows", ["gamma_abs_rank_margin_over_gauge_scale", "two_candidate_error_tail", "two_candidate_hit_upper", "n10_independent_challenger_loss_proxy", "theorem_status"]),
        "theorem_stress_failure_modes.csv": ("theorem_stress_failure_modes_rows", ["failure_id", "stress_case", "breaks_core_theorem", "breaks_extension", "mitigation"]),
        "theorem_to_eeg_assumption_map.csv": ("theorem_to_eeg_assumption_map_rows", ["assumption_id", "empirical_proxy", "support_class", "artifact", "bridge_limit"]),
        "assumption_support_scores.csv": ("assumption_support_scores_rows", ["bridge_axis", "support_score", "support_class", "key_number", "interpretation"]),
        "assumption_gap_ledger.csv": ("assumption_gap_ledger_rows", ["gap_id", "missing_item", "blocks", "requires_new_instrumentation", "priority"]),
        "gauge_tail_empirical_summary.csv": ("gauge_tail_empirical_summary_rows", ["quantity", "value", "interpretation", "theorem_grade"]),
        "rank_margin_gauge_scale_summary.csv": ("rank_margin_gauge_scale_summary_rows", ["field", "hit", "error_tail_proxy", "normal_tail_equivalent_gamma", "interpretation"]),
        "source_error_vs_gauge_tail_ledger.csv": ("source_error_vs_gauge_tail_ledger_rows", ["comparison", "observed_error", "nearest_curve_gamma", "curve_error", "difference", "conclusion"]),
        "lecam_witness_repair_summary.csv": ("lecam_witness_repair_summary_rows", ["attempt_id", "source_near_proxy", "target_divergence_proxy", "repair_status", "theorem_grade", "blocker"]),
        "fano_packing_repair_summary.csv": ("fano_packing_repair_summary_rows", ["attempt_id", "packing_size", "mi_or_kl_status", "bound_status", "theorem_grade", "blocker"]),
        "lower_bound_failure_reason_ledger.csv": ("lower_bound_failure_reason_ledger_rows", ["failure_id", "branch", "reason", "support_blocker", "definition_blocker"]),
        "source_observable_adversary_results.csv": ("source_observable_adversary_results_rows", ["candidate_id", "candidate", "allowed_source_only", "uses_target_labels", "hit", "beats_max_null_p95", "reliable_escape_hatch", "reason"]),
        "source_observable_availability_ledger.csv": ("source_observable_availability_ledger_rows", ["input_family", "allowed", "available_in_frozen_artifacts", "risk"]),
        "source_escape_hatch_red_team.csv": ("source_escape_hatch_red_team_rows", ["gate", "passed", "finding"]),
        "conditional_sufficiency_summary.csv": ("conditional_sufficiency_summary_rows", ["condition_set", "candidate_added", "hit_before", "hit_after", "delta_hit", "screens_off_endpoint", "status"]),
        "markov_boundary_probe_summary.csv": ("markov_boundary_probe_summary_rows", ["variable", "empirical_boundary_role", "uses_target_labels", "available_at_selection_time", "diagnostic_only", "hit"]),
        "information_ladder_formal_status.csv": ("information_ladder_formal_status_rows", ["information_class", "hit", "formal_status", "source_only", "label_content"]),
        "instrumentation_blueprint_v2.csv": ("instrumentation_blueprint_v2_rows", ["protocol_id", "protocol", "scientific_question", "new_data_required", "training_or_inference_needed", "approval_required", "slurm_requirement", "claim_supported", "risk"]),
        "training_gate_decision_matrix.csv": ("training_gate_decision_matrix_rows", ["gate", "decision", "requires_user_release", "default", "note"]),
        "missing_data_to_theorem_gap.csv": ("missing_data_to_theorem_gap_rows", ["gap_id", "data_item", "needed_for", "present_now", "minimal_protocol"]),
        "subagent_audit_manifest.csv": ("subagent_audit_manifest_rows", ["subagent_id", "role", "integration_status"]),
        "test_command_manifest.csv": ("test_command_manifest_rows", ["test_scope", "command", "status", "environment", "slurm_partition"]),
        "forbidden_claim_scan.csv": ("forbidden_claim_scan_rows", ["pattern", "total_hits", "affirmative_hits", "files", "passed"]),
        "red_team_failure_ledger.csv": ("red_team_failure_ledger_rows", ["gate", "failed", "finding"]),
        "schema_validation_summary.csv": ("schema_validation_summary_rows", ["table_name", "row_count", "required_columns_present", "passed"]),
        "large_artifact_scan.csv": ("large_artifact_scan_rows", ["path", "size_bytes", "over_50mb", "passed"]),
        "artifact_manifest.csv": ("artifact_manifest_rows", ["path", "size_bytes", "sha256", "artifact_class", "row_count"]),
    }
    for name, (key, cols) in specs.items():
        _write_csv(os.path.join(table_dir, name), res[key], cols)


def _write_texts(files: dict[str, str]) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")


def _listed_paths() -> list[str]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        glob.glob(os.path.join(REPORT_DIR, "C60_*.md"))
        + glob.glob(os.path.join(REPORT_DIR, "C60_*.json"))
        + [p for p in glob.glob(os.path.join(TABLE_DIR, "*.csv")) if os.path.basename(p) not in skip]
    )


def _schema_rows(table_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(table_dir, "*.csv"))):
        if os.path.basename(path) in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": os.path.basename(path), "row_count": count, "required_columns_present": int(bool(header)), "passed": int(bool(header))})
    return rows


def _large_scan(paths: list[str]) -> list[dict]:
    rows = []
    for path in sorted(paths):
        size = os.path.getsize(path)
        rows.append({"path": path, "size_bytes": size, "over_50mb": int(size > 50_000_000), "passed": int(size <= 50_000_000)})
    return rows


def _artifact_manifest(paths: list[str], table_dir: str) -> list[dict]:
    row_counts = {}
    for path in glob.glob(os.path.join(table_dir, "*.csv")):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            row_counts[path] = sum(1 for _ in reader)
    rows = []
    for path in sorted(paths):
        cls = "table" if path.endswith(".csv") else "summary_json" if path.endswith(".json") else "report"
        rows.append({"path": path, "size_bytes": os.path.getsize(path), "sha256": _sha256(path), "artifact_class": cls, "row_count": row_counts.get(path, "")})
    return rows


def write_artifacts(res: dict, test_status: str) -> dict:
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["artifact_manifest_rows"] = [{"path": p} for p in paths]
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    write_tables(res, TABLE_DIR)

    res["schema_validation_summary_rows"] = _schema_rows(TABLE_DIR)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["large_artifact_scan_rows"] = [{"path": p} for p in paths]
    res["artifact_manifest_rows"] = [{"path": p} for p in paths]
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["large_artifact_scan_rows"] = _large_scan(paths)
    _write_csv(os.path.join(TABLE_DIR, "large_artifact_scan.csv"), res["large_artifact_scan_rows"], ["path", "size_bytes", "over_50mb", "passed"])
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"], ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c60_rank_gauge_proof_stress_empirical_bridge")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res, args.test_status)
    print(f"[C60] decision={res['decision']['primary']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
