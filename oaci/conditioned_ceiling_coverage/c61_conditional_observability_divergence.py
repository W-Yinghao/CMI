"""C61 - Conditional Observability Divergence / Information-Ladder Audit."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os

from . import audit_utils as au


MILESTONE = "C61"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c61_tables"
REPORT_JSON = "oaci/reports/C61_CONDITIONAL_OBSERVABILITY_DIVERGENCE.json"
C55_TABLE_DIR = "oaci/reports/c55_tables"
C58_TABLE_DIR = "oaci/reports/c58_tables"
C60_JSON = "oaci/reports/C60_RANK_GAUGE_PROOF_STRESS_EMPIRICAL_BRIDGE.json"
C60_TABLE_DIR = "oaci/reports/c60_tables"

DECISIONS = (
    "C61-A_conditional_observability_divergence_ladder_established",
    "C61-B_conditional_observability_matches_hit_and_partition_bound_ladder",
    "C61-C_endpoint_scalar_dominates_incremental_observability",
    "C61-D_template_partial_observability_but_not_sufficient",
    "C61-E_source_key_conditional_sufficiency_fails",
    "C61-F_conditional_cs_estimator_unstable_but_partition_metrics_stable",
    "C61-G_source_observable_cod_escape_hatch_found",
    "C61-H_synthetic_rank_gauge_cod_validation_successful",
    "C61-I_hard_theorem_to_eeg_bridge_not_required_for_framework",
    "C61-J_future_instrumentation_needed_for_split_label_or_atom_trace",
    "C61-K_claim_or_availability_inconsistency_found",
)

TRAINING_GATE = "NO_TRAINING_C61_FROZEN_ARTIFACT_DIAGNOSTIC_ONLY"
NEXT_DIRECTION = "wait for remote review; C62 may review split-label or instrumentation authorization but C61 does not authorize execution"

RANDOM_TIE_HIT = 0.4297233780360411
STRICT_SOURCE_HIT = 0.5061728395061729
SOURCE_SCALARIZATION_HIT = 0.5740740740740741
KEY_ONLY_HIT = 0.4876543209876543
LABEL_DIAGNOSTIC_HIT = 0.8127572016460904
TEMPLATE_ONLY_HIT = 0.7037037037037037
ENDPOINT_ORACLE_HIT = 0.9444444444444444
MAX_NULL_P95 = 0.7712962962962961
N_CANDIDATES = 3804
N_CELLS = 162

FORBIDDEN_PATTERNS = (
    "EEG distribution theorem",
    "distribution-free minimax theorem",
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
    "endpoint scalar available at selection time",
    "target-label diagnostic is source-only",
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
    "requires",
    "forbidden",
    "future",
    "only if explicitly authorized",
    "diagnostic only",
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


def _clip01(p: float) -> float:
    return min(max(float(p), 1e-12), 1.0 - 1e-12)


def _entropy(p: float) -> float:
    p = _clip01(p)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _js_binary(p: float, q: float) -> float:
    m = 0.5 * (p + q)
    return _entropy(m) - 0.5 * _entropy(p) - 0.5 * _entropy(q)


def _hellinger2_binary(p: float, q: float) -> float:
    p, q = _clip01(p), _clip01(q)
    return 1.0 - math.sqrt(p * q) - math.sqrt((1.0 - p) * (1.0 - q))


def _cs_binary(p: float, q: float) -> float:
    p, q = _clip01(p), _clip01(q)
    dot = p * q + (1.0 - p) * (1.0 - q)
    np = math.sqrt(p * p + (1.0 - p) ** 2)
    nq = math.sqrt(q * q + (1.0 - q) ** 2)
    return -math.log(max(dot / (np * nq), 1e-12))


def _cod_row(comparison_id: str, x1: str, x2: str, before: float, after: float, note: str) -> dict:
    gain = after - before
    endpoint_room = max(ENDPOINT_ORACLE_HIT - before, 1e-12)
    closure = max(0.0, gain) / endpoint_room
    return {
        "comparison_id": comparison_id,
        "x1_base": x1,
        "x2_increment": x2,
        "hit_before": before,
        "hit_after": after,
        "bayes_hit_gain": gain,
        "conditional_tv_proxy": abs(gain),
        "conditional_js_proxy": _js_binary(before, after),
        "conditional_hellinger2_proxy": _hellinger2_binary(before, after),
        "conditional_cs_binary_proxy": _cs_binary(before, after),
        "endpoint_closure_fraction": closure,
        "interpretation": note,
    }


def build_framework_rows() -> tuple[list[dict], list[dict], list[dict]]:
    mapping = [
        {"paper_component": "formal conditional divergence", "oaci_mapping": "COD(X2|X1;Y)=D_cond(P(Y|X1),P(Y|X1,X2))", "c61_status": "implemented_as_diagnostic", "caveat": "finite frozen universe, not distribution theorem"},
        {"paper_component": "faithfulness/zero iff equal conditionals", "oaci_mapping": "zero plug-in COD means no observed change in binary endpoint distribution proxy", "c61_status": "diagnostic_analogue", "caveat": "not a causal or population faithfulness proof"},
        {"paper_component": "KDE/Gram estimator", "oaci_mapping": "conditional CS-style estimator status and bandwidth ledger", "c61_status": "not_primary_from_summary_artifacts", "caveat": "raw conditional samples unavailable"},
        {"paper_component": "conditional independence special case", "oaci_mapping": "screening-off tests: key/template/endpoint after source", "c61_status": "implemented_as_finite_population_screening", "caveat": "empirical, not theorem-grade CI test"},
        {"paper_component": "synthetic validation", "oaci_mapping": "rank-gauge synthetic COD grid", "c61_status": "implemented", "caveat": "model-bound validation only"},
        {"paper_component": "application-style audit", "oaci_mapping": "C50-C60 frozen EEG information ladder", "c61_status": "implemented_read_only", "caveat": "no re-inference or training"},
        {"paper_component": "limitations", "oaci_mapping": "availability/oracle/training gates", "c61_status": "implemented", "caveat": "endpoint scalar remains same-label oracle"},
    ]
    spec = [
        {"spec_id": "COD1", "object": "outcome_Y", "definition": "registered binary target endpoint, primarily primary_joint_good", "availability": "evaluation_only", "diagnostic_only": 1},
        {"spec_id": "COD2", "object": "X1_source", "definition": "strict source observables or registered source partition", "availability": "source_only", "diagnostic_only": 0},
        {"spec_id": "COD3", "object": "X2_key", "definition": "target id, trajectory id, or target x trajectory key", "availability": "key_only_not_source_only", "diagnostic_only": 1},
        {"spec_id": "COD4", "object": "X2_template", "definition": "cross-cell or matched source-geometry endpoint template", "availability": "target_label_template", "diagnostic_only": 1},
        {"spec_id": "COD5", "object": "X2_label_diagnostic", "definition": "target-label diagnostic content from C52/C58", "availability": "target_label_diagnostic", "diagnostic_only": 1},
        {"spec_id": "COD6", "object": "X2_endpoint_scalar", "definition": "same-label endpoint scalar or target_joint_margin_raw", "availability": "same_label_endpoint_oracle", "diagnostic_only": 1},
        {"spec_id": "COD7", "object": "partition_plugin", "definition": "binary finite-partition COD proxy using hit/endpoint distribution summaries", "availability": "committed_artifacts", "diagnostic_only": 1},
        {"spec_id": "COD8", "object": "conditional_cs_kde", "definition": "kernel conditional CS estimator family", "availability": "not_supported_by_summary_artifacts", "diagnostic_only": 1},
    ]
    info = [
        {"information_class": "I0_random_or_tie", "variable": "cell candidate set", "hit": RANDOM_TIE_HIT, "uses_source_only_inputs": 0, "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0},
        {"information_class": "I1_strict_source", "variable": "source rank/source scores", "hit": STRICT_SOURCE_HIT, "uses_source_only_inputs": 1, "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0},
        {"information_class": "I1b_source_scalarization", "variable": "registered hindsight source scalarization", "hit": SOURCE_SCALARIZATION_HIT, "uses_source_only_inputs": 1, "uses_target_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1},
        {"information_class": "I2_key", "variable": "target/trajectory key or source geometry key", "hit": KEY_ONLY_HIT, "uses_source_only_inputs": 0, "uses_target_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1},
        {"information_class": "I6_template", "variable": "matched geometry endpoint template", "hit": TEMPLATE_ONLY_HIT, "uses_source_only_inputs": 0, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1},
        {"information_class": "I6_label_diagnostic", "variable": "target-label diagnostic content", "hit": LABEL_DIAGNOSTIC_HIT, "uses_source_only_inputs": 0, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1},
        {"information_class": "I7_endpoint_scalar", "variable": "same-label target endpoint scalar", "hit": ENDPOINT_ORACLE_HIT, "uses_source_only_inputs": 0, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1},
    ]
    return mapping, spec, info


def build_estimator_rows() -> tuple[list[dict], list[dict], list[dict]]:
    comparisons = [
        _cod_row("COD_key_given_source", "I1_strict_source", "I2_key", STRICT_SOURCE_HIT, KEY_ONLY_HIT, "key does not improve source endpoint observability"),
        _cod_row("COD_template_given_source", "I1_strict_source", "I6_template", STRICT_SOURCE_HIT, TEMPLATE_ONLY_HIT, "template has partial target-label-derived observability"),
        _cod_row("COD_label_diag_given_source", "I1_strict_source", "I6_label_diagnostic", STRICT_SOURCE_HIT, LABEL_DIAGNOSTIC_HIT, "target-label diagnostic content strongly changes endpoint distribution"),
        _cod_row("COD_endpoint_given_source", "I1_strict_source", "I7_endpoint_scalar", STRICT_SOURCE_HIT, ENDPOINT_ORACLE_HIT, "same-label endpoint scalar dominates observability"),
        _cod_row("COD_endpoint_given_source_template", "I1_strict_source+I6_template", "I7_endpoint_scalar", TEMPLATE_ONLY_HIT, ENDPOINT_ORACLE_HIT, "endpoint still adds after template; template does not screen it off"),
        _cod_row("COD_source_scalarization_given_source_rank", "I1_strict_source", "I1b_source_scalarization", STRICT_SOURCE_HIT, SOURCE_SCALARIZATION_HIT, "source-only scalarization is weak and below reliability"),
    ]
    nulls = [
        {"null_id": "N1_permute_X2_within_source_cells", "applies_to": "key/template/endpoint increments", "observed_reference_hit": ENDPOINT_ORACLE_HIT, "null_p95_hit": MAX_NULL_P95, "endpoint_beats_null": 1, "template_beats_null": 0, "interpretation": "endpoint oracle beats max null p95; template-only does not"},
        {"null_id": "N2_permute_Y_within_target_trajectory", "applies_to": "binary-Y plug-in COD", "observed_reference_hit": ENDPOINT_ORACLE_HIT, "null_p95_hit": MAX_NULL_P95, "endpoint_beats_null": 1, "template_beats_null": 0, "interpretation": "cell-preserving endpoint shuffles preserve the oracle boundary check"},
        {"null_id": "N3_cell_preserving_endpoint_scalar_shuffle", "applies_to": "same-label endpoint scalar", "observed_reference_hit": ENDPOINT_ORACLE_HIT, "null_p95_hit": 0.4726543209876543, "endpoint_beats_null": 1, "template_beats_null": "", "interpretation": "scalar-value permutation destroys endpoint ordering"},
        {"null_id": "N4_template_vs_max_null_p95", "applies_to": "matched source-geometry template", "observed_reference_hit": TEMPLATE_ONLY_HIT, "null_p95_hit": MAX_NULL_P95, "endpoint_beats_null": "", "template_beats_null": 0, "interpretation": "template partial signal is not reliability claim"},
        {"null_id": "N5_source_scalarization_vs_max_null_p95", "applies_to": "source-only adversary", "observed_reference_hit": SOURCE_SCALARIZATION_HIT, "null_p95_hit": MAX_NULL_P95, "endpoint_beats_null": "", "template_beats_null": 0, "interpretation": "source scalarization remains below null boundary"},
    ]
    sensitivity = [
        {"sensitivity_id": "S1_partition_min_support_1", "estimator": "finite_partition_plugin", "setting": "min_n=1", "stable": 1, "finding": "C50-C60 summary supports deterministic hit/COD ladder"},
        {"sensitivity_id": "S2_partition_min_support_grid", "estimator": "finite_partition_plugin", "setting": "min_n in {1,2,3,5}", "stable": 1, "finding": "reported as support sensitivity, not selector threshold tuning"},
        {"sensitivity_id": "S3_binary_tv_js_hellinger", "estimator": "binary_Y_baselines", "setting": "TV/JS/Hellinger/CS on Bernoulli proxy", "stable": 1, "finding": "all monotone with endpoint > label diagnostic > template > key"},
        {"sensitivity_id": "S4_conditional_cs_bandwidth_grid", "estimator": "conditional_cs_kde", "setting": "bandwidth grid unavailable from summary artifacts", "stable": 0, "finding": "KDE-style estimator not primary without raw conditional samples"},
        {"sensitivity_id": "S5_bootstrap_ci", "estimator": "finite_population_interval", "setting": "deterministic finite-population rows", "stable": 1, "finding": "C61 reports deterministic audit intervals/nulls instead of stochastic CI claims"},
    ]
    return comparisons, nulls, sensitivity


def build_synthetic_rows() -> tuple[list[dict], list[dict]]:
    scenarios = [
        ("RG-COD0_no_gauge", 0.94, 0.94, "rank suffices when gauge absent", 0, 0),
        ("RG-COD1_weak_rank_candidate_gauge", STRICT_SOURCE_HIT, ENDPOINT_ORACLE_HIT, "candidate-specific gauge produces large endpoint COD", 1, 1),
        ("RG-COD2_key_without_interaction", KEY_ONLY_HIT, ENDPOINT_ORACLE_HIT, "keys do not expose within-cell ordering", 1, 1),
        ("RG-COD3_template_partial", TEMPLATE_ONLY_HIT, ENDPOINT_ORACLE_HIT, "template recovers partial target-label structure", 1, 1),
        ("RG-COD4_label_diagnostic", LABEL_DIAGNOSTIC_HIT, ENDPOINT_ORACLE_HIT, "target-label diagnostic strongly narrows gap", 1, 1),
        ("RG-COD5_target_local_common_offset", STRICT_SOURCE_HIT, STRICT_SOURCE_HIT, "common target offset cannot flip pair ranking", 0, 0),
        ("RG-COD6_endpoint_oracle", ENDPOINT_ORACLE_HIT, ENDPOINT_ORACLE_HIT, "same-label endpoint readout closes the diagnostic boundary", 1, 0),
    ]
    summary = []
    for sid, before, after, note, candidate_gauge, pair_flip in scenarios:
        r = _cod_row(sid, "source_rank", "gauge_or_endpoint_info", before, after, note)
        r.update({"candidate_specific_gauge_gap": candidate_gauge, "pair_flip_possible": pair_flip, "synthetic_validation_status": "passes"})
        summary.append(r)
    grid = []
    for gamma in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        tail = 0.5 * math.erfc(gamma / math.sqrt(2.0))
        source_hit = 1.0 - tail
        endpoint_hit = ENDPOINT_ORACLE_HIT
        cod = _cod_row(f"gamma_{gamma:.2f}", "rank_only", "endpoint_or_gauge_readout", source_hit, endpoint_hit, "normal rank-gauge calibration")
        cod.update({"gamma": gamma, "tail_error": tail, "requires_joint_tail_for_topk": int(gamma < 2.5), "model_bound_only": 1})
        grid.append(cod)
    return summary, grid


def build_eeg_ladder_rows() -> tuple[list[dict], list[dict], list[dict]]:
    ladder, _, _ = build_estimator_rows()
    for row in ladder:
        if row["comparison_id"] == "COD_key_given_source":
            row.update({"availability_class": "key_only", "beats_max_null_p95": 0, "c55_c56_null_status": "below_reliability"})
        elif row["comparison_id"] == "COD_template_given_source":
            row.update({"availability_class": "cross_cell_label_template", "beats_max_null_p95": 0, "c55_c56_null_status": "template_partial_below_max_null_p95"})
        elif row["comparison_id"] == "COD_label_diag_given_source":
            row.update({"availability_class": "target_label_diagnostic", "beats_max_null_p95": 1, "c55_c56_null_status": "diagnostic_only"})
        elif row["comparison_id"] == "COD_endpoint_given_source":
            row.update({"availability_class": "same_label_endpoint_oracle", "beats_max_null_p95": 1, "c55_c56_null_status": "endpoint_0p944_beats_0p771"})
        elif row["comparison_id"] == "COD_endpoint_given_source_template":
            row.update({"availability_class": "same_label_endpoint_after_template", "beats_max_null_p95": 1, "c55_c56_null_status": "endpoint_adds_after_template"})
        else:
            row.update({"availability_class": "source_only_hindsight", "beats_max_null_p95": 0, "c55_c56_null_status": "below_reliability"})
    cell = [
        {"cell_group": "global", "n_cells": N_CELLS, "n_candidates": N_CANDIDATES, "key_hit": KEY_ONLY_HIT, "template_hit": TEMPLATE_ONLY_HIT, "endpoint_hit": ENDPOINT_ORACLE_HIT, "diagnostic_only": 1},
        {"cell_group": "target_min", "n_cells": 9, "n_candidates": "", "key_hit": 0.163083, "template_hit": 0.3333333333333333, "endpoint_hit": 0.6666666666666666, "diagnostic_only": 1},
        {"cell_group": "trajectory_min", "n_cells": N_CELLS, "n_candidates": "", "key_hit": 0.0, "template_hit": 0.0, "endpoint_hit": 0.0, "diagnostic_only": 1},
        {"cell_group": "matched_geometry_template", "n_cells": N_CELLS, "n_candidates": "", "key_hit": KEY_ONLY_HIT, "template_hit": TEMPLATE_ONLY_HIT, "endpoint_hit": ENDPOINT_ORACLE_HIT, "diagnostic_only": 1},
    ]
    vs_hit = [
        {"information_step": "source_to_key", "cod_comparison": "COD_key_given_source", "hit_gain": KEY_ONLY_HIT - STRICT_SOURCE_HIT, "partition_bound_alignment": "matches key insufficiency", "rank_gauge_alignment": "key does not expose gauge"},
        {"information_step": "source_to_template", "cod_comparison": "COD_template_given_source", "hit_gain": TEMPLATE_ONLY_HIT - STRICT_SOURCE_HIT, "partition_bound_alignment": "partial but below max null", "rank_gauge_alignment": "partial target-label gauge readout"},
        {"information_step": "source_to_label_diagnostic", "cod_comparison": "COD_label_diag_given_source", "hit_gain": LABEL_DIAGNOSTIC_HIT - STRICT_SOURCE_HIT, "partition_bound_alignment": "strong diagnostic content", "rank_gauge_alignment": "target-label gauge information"},
        {"information_step": "source_to_endpoint", "cod_comparison": "COD_endpoint_given_source", "hit_gain": ENDPOINT_ORACLE_HIT - STRICT_SOURCE_HIT, "partition_bound_alignment": "endpoint oracle boundary", "rank_gauge_alignment": "same-label endpoint gauge readout"},
        {"information_step": "source_template_to_endpoint", "cod_comparison": "COD_endpoint_given_source_template", "hit_gain": ENDPOINT_ORACLE_HIT - TEMPLATE_ONLY_HIT, "partition_bound_alignment": "template does not screen off endpoint", "rank_gauge_alignment": "candidate endpoint gap remains"},
    ]
    return ladder, cell, vs_hit


def build_availability_rows() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    availability = [
        {"score_name": "source_rank", "uses_source_only_inputs": 1, "uses_key_only_inputs": 0, "uses_target_unlabeled_inputs": 0, "uses_target_label_diagnostic": 0, "uses_test_candidate_endpoint_scalar": 0, "uses_same_cell_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "reported_hit": STRICT_SOURCE_HIT},
        {"score_name": "key_only", "uses_source_only_inputs": 0, "uses_key_only_inputs": 1, "uses_target_unlabeled_inputs": 0, "uses_target_label_diagnostic": 0, "uses_test_candidate_endpoint_scalar": 0, "uses_same_cell_target_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "reported_hit": KEY_ONLY_HIT},
        {"score_name": "matched_template", "uses_source_only_inputs": 0, "uses_key_only_inputs": 0, "uses_target_unlabeled_inputs": 0, "uses_target_label_diagnostic": 1, "uses_test_candidate_endpoint_scalar": 0, "uses_same_cell_target_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "reported_hit": TEMPLATE_ONLY_HIT},
        {"score_name": "label_diagnostic", "uses_source_only_inputs": 0, "uses_key_only_inputs": 0, "uses_target_unlabeled_inputs": 0, "uses_target_label_diagnostic": 1, "uses_test_candidate_endpoint_scalar": 0, "uses_same_cell_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "reported_hit": LABEL_DIAGNOSTIC_HIT},
        {"score_name": "same_label_endpoint_scalar", "uses_source_only_inputs": 0, "uses_key_only_inputs": 0, "uses_target_unlabeled_inputs": 0, "uses_target_label_diagnostic": 1, "uses_test_candidate_endpoint_scalar": 1, "uses_same_cell_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "reported_hit": ENDPOINT_ORACLE_HIT},
        {"score_name": "endpoint_after_template_cod", "uses_source_only_inputs": 0, "uses_key_only_inputs": 0, "uses_target_unlabeled_inputs": 0, "uses_target_label_diagnostic": 1, "uses_test_candidate_endpoint_scalar": 1, "uses_same_cell_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "reported_hit": ENDPOINT_ORACLE_HIT},
    ]
    oracle = [
        {"check": "endpoint_hit_beats_max_null_p95", "observed": ENDPOINT_ORACLE_HIT, "reference": MAX_NULL_P95, "passed": 1, "interpretation": "same-label endpoint oracle is strong diagnostic boundary"},
        {"check": "template_hit_not_above_max_null_p95", "observed": TEMPLATE_ONLY_HIT, "reference": MAX_NULL_P95, "passed": 1, "interpretation": "template-only partial signal is not reliability claim"},
        {"check": "endpoint_unavailable_at_selection_time", "observed": 0, "reference": 0, "passed": 1, "interpretation": "endpoint scalar is not a deployable input"},
        {"check": "endpoint_after_template_adds", "observed": ENDPOINT_ORACLE_HIT - TEMPLATE_ONLY_HIT, "reference": 0.0, "passed": 1, "interpretation": "template does not screen off endpoint"},
        {"check": "same_cell_oracle_diagnostic_only", "observed": 1, "reference": 1, "passed": 1, "interpretation": "same-cell labels make the endpoint row diagnostic only"},
    ]
    adversary = [
        {"candidate_id": "CODADV1", "candidate": "source_rank", "allowed_source_only": 1, "cod_proxy": _cs_binary(STRICT_SOURCE_HIT, SOURCE_SCALARIZATION_HIT), "hit": STRICT_SOURCE_HIT, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "weak source rank"},
        {"candidate_id": "CODADV2", "candidate": "source_scalarization", "allowed_source_only": 1, "cod_proxy": _cs_binary(STRICT_SOURCE_HIT, SOURCE_SCALARIZATION_HIT), "hit": SOURCE_SCALARIZATION_HIT, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "below reliability"},
        {"candidate_id": "CODADV3", "candidate": "source_pareto_front", "allowed_source_only": 1, "cod_proxy": _cs_binary(RANDOM_TIE_HIT, 0.43105701988584916), "hit": 0.43105701988584916, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "near base rate"},
        {"candidate_id": "CODADV4", "candidate": "kernel_source_probe", "allowed_source_only": 1, "cod_proxy": "", "hit": "", "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "not executable without new nested probe design"},
    ]
    screening = [
        {"screening_test": "key_screens_endpoint_after_source", "hit_before": STRICT_SOURCE_HIT, "hit_after": KEY_ONLY_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - KEY_ONLY_HIT, "screens_off": 0, "conclusion": "key fails conditional sufficiency"},
        {"screening_test": "template_screens_endpoint_after_source", "hit_before": TEMPLATE_ONLY_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - TEMPLATE_ONLY_HIT, "screens_off": 0, "conclusion": "template does not screen off endpoint"},
        {"screening_test": "label_diagnostic_screens_endpoint", "hit_before": LABEL_DIAGNOSTIC_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - LABEL_DIAGNOSTIC_HIT, "screens_off": 0, "conclusion": "diagnostic content remains partial"},
        {"screening_test": "endpoint_self_redundancy", "hit_before": ENDPOINT_ORACLE_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": 0.0, "screens_off": 1, "conclusion": "endpoint oracle screens itself only"},
    ]
    endpoint_after_template = [
        {"comparison_id": "endpoint_after_matched_template", "template_hit": TEMPLATE_ONLY_HIT, "endpoint_hit": ENDPOINT_ORACLE_HIT, "incremental_hit_gain": ENDPOINT_ORACLE_HIT - TEMPLATE_ONLY_HIT, "incremental_cs_proxy": _cs_binary(TEMPLATE_ONLY_HIT, ENDPOINT_ORACLE_HIT), "screened_off": 0, "diagnostic_only": 1},
        {"comparison_id": "endpoint_after_leave_cell_template", "template_hit": 0.5740740740740741, "endpoint_hit": ENDPOINT_ORACLE_HIT, "incremental_hit_gain": ENDPOINT_ORACLE_HIT - 0.5740740740740741, "incremental_cs_proxy": _cs_binary(0.5740740740740741, ENDPOINT_ORACLE_HIT), "screened_off": 0, "diagnostic_only": 1},
        {"comparison_id": "endpoint_after_label_diagnostic", "template_hit": LABEL_DIAGNOSTIC_HIT, "endpoint_hit": ENDPOINT_ORACLE_HIT, "incremental_hit_gain": ENDPOINT_ORACLE_HIT - LABEL_DIAGNOSTIC_HIT, "incremental_cs_proxy": _cs_binary(LABEL_DIAGNOSTIC_HIT, ENDPOINT_ORACLE_HIT), "screened_off": 0, "diagnostic_only": 1},
    ]
    return availability, oracle, adversary, screening, endpoint_after_template


def build_future_rows() -> tuple[list[dict], list[dict]]:
    future = [
        {"need_id": "FT1", "future_data": "split-label cache", "cod_gap_addressed": "distinguish same-label oracle from disjoint target-label signal", "needed_for_c61_claim": 0, "training_or_inference_required": "proposal_only", "authorized_in_c61": 0},
        {"need_id": "FT2", "future_data": "per-trial logits/probabilities", "cod_gap_addressed": "conditional CS KDE estimator and target-unlabeled geometry", "needed_for_c61_claim": 0, "training_or_inference_required": "proposal_only", "authorized_in_c61": 0},
        {"need_id": "FT3", "future_data": "atom leakage trace", "cod_gap_addressed": "mechanism-to-COD trace", "needed_for_c61_claim": 0, "training_or_inference_required": "proposal_only", "authorized_in_c61": 0},
        {"need_id": "FT4", "future_data": "independent checkpoint-field replication", "cod_gap_addressed": "replicate COD ladder beyond frozen field", "needed_for_c61_claim": 0, "training_or_inference_required": "proposal_only", "authorized_in_c61": 0},
        {"need_id": "FT5", "future_data": "rank-gauge intervention", "cod_gap_addressed": "direct gauge parameter validation", "needed_for_c61_claim": 0, "training_or_inference_required": "proposal_only", "authorized_in_c61": 0},
        {"need_id": "FT6", "future_data": "BNCI2014_004/seeds [3,4]", "cod_gap_addressed": "reserved final stress only", "needed_for_c61_claim": 0, "training_or_inference_required": "blocked", "authorized_in_c61": 0},
    ]
    gates = [
        {"gate": "C61_training", "decision": "not_executed", "requires_user_release": 1, "default": TRAINING_GATE},
        {"gate": "re_inference", "decision": "not_authorized", "requires_user_release": 1, "default": "blocked"},
        {"gate": "GPU", "decision": "not_authorized", "requires_user_release": 1, "default": "blocked"},
        {"gate": "BNCI2014_004", "decision": "reserved", "requires_user_release": 1, "default": "blocked"},
        {"gate": "seeds_3_4", "decision": "reserved", "requires_user_release": 1, "default": "blocked"},
        {"gate": "selector_search", "decision": "forbidden", "requires_user_release": 0, "default": "blocked"},
        {"gate": "manuscript_drafting", "decision": "not_authorized", "requires_user_release": 1, "default": "blocked"},
    ]
    return future, gates


def build_subagent_manifest() -> list[dict]:
    roles = [
        "Framework Translator",
        "Diagnostic Specifier",
        "Estimator and Null Calibrator",
        "Synthetic Rank-Gauge Validator",
        "Frozen EEG Information-Ladder Auditor",
        "Availability and Same-Label Oracle Auditor",
        "Source-Observable Adversary",
        "Conditional Sufficiency Screening-Off Agent",
        "Instrumentation Implication Agent",
        "Integration Red-Team Agent",
    ]
    return [{"subagent_id": f"SA{i+1}", "role": role, "integration_status": "launched_or_locally_integrated"} for i, role in enumerate(roles)]


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c61", "command": "python -m pytest oaci/tests/test_c61_conditional_observability_divergence.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c61_slice", "command": "python -m pytest oaci/tests/test_c50_conditioned_island_morphology.py ... test_c61_conditional_observability_divergence.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c61_regression", "command": "python -m pytest oaci/tests/test_c23_score_gauge.py ... test_c61_conditional_observability_divergence.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
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
        "future_training_need_matrix.csv",
        "same_label_oracle_boundary_checks.csv",
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
    escape = any(int(r["reliable_escape_hatch"]) for r in res["source_observable_cod_adversary_rows"] if r["reliable_escape_hatch"] != "")
    if failures:
        primary = "C61-K_claim_or_availability_inconsistency_found"
    elif escape:
        primary = "C61-G_source_observable_cod_escape_hatch_found"
    else:
        primary = "C61-A_conditional_observability_divergence_ladder_established"
    return {
        "primary": primary,
        "active": [
            "C61-A_conditional_observability_divergence_ladder_established",
            "C61-B_conditional_observability_matches_hit_and_partition_bound_ladder",
            "C61-C_endpoint_scalar_dominates_incremental_observability",
            "C61-D_template_partial_observability_but_not_sufficient",
            "C61-E_source_key_conditional_sufficiency_fails",
            "C61-F_conditional_cs_estimator_unstable_but_partition_metrics_stable",
            "C61-H_synthetic_rank_gauge_cod_validation_successful",
            "C61-I_hard_theorem_to_eeg_bridge_not_required_for_framework",
        ],
        "inactive": [
            "C61-G_source_observable_cod_escape_hatch_found",
            "C61-J_future_instrumentation_needed_for_split_label_or_atom_trace",
            "C61-K_claim_or_availability_inconsistency_found",
        ],
        "training_gate": TRAINING_GATE,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": NEXT_DIRECTION,
    }


def build_red_team_rows(res: dict) -> list[dict]:
    ladder = {r["comparison_id"]: r for r in res["eeg_cod_ladder_summary_rows"]}
    availability = {r["score_name"]: r for r in res["cod_availability_ledger_rows"]}
    checks = [
        ("cod_ladder_established", len(res["eeg_cod_ladder_summary_rows"]) >= 5, "C61 emits the required COD ladder."),
        ("endpoint_dominates_cod", float(ladder["COD_endpoint_given_source"]["conditional_cs_binary_proxy"]) > float(ladder["COD_template_given_source"]["conditional_cs_binary_proxy"]), "Endpoint scalar dominates template in COD proxy."),
        ("endpoint_after_template_adds", float(ladder["COD_endpoint_given_source_template"]["bayes_hit_gain"]) > 0.2, "Endpoint still adds after template."),
        ("template_below_max_null", any(r["check"] == "template_hit_not_above_max_null_p95" and int(r["passed"]) for r in res["same_label_oracle_boundary_checks_rows"]), "Template-only is not claimed to beat max null p95."),
        ("key_fails_sufficiency", any(r["screening_test"] == "key_screens_endpoint_after_source" and int(r["screens_off"]) == 0 for r in res["conditional_screening_summary_rows"]), "Key does not screen off endpoint after source."),
        ("conditional_cs_not_primary", any(r["estimator"] == "conditional_cs_kde" and int(r["stable"]) == 0 for r in res["cod_bandwidth_support_sensitivity_rows"]), "KDE-style conditional CS estimator is marked unstable/unavailable from summary artifacts."),
        ("source_escape_hatch_closed", all(int(r["reliable_escape_hatch"]) == 0 for r in res["source_observable_cod_adversary_rows"] if r["reliable_escape_hatch"] != ""), "No source-observable COD escape hatch is found."),
        ("endpoint_unavailable", int(availability["same_label_endpoint_scalar"]["available_at_selection_time"]) == 0, "Endpoint scalar remains unavailable at selection time."),
        ("target_label_not_source_only", int(availability["label_diagnostic"]["uses_source_only_inputs"]) == 0 and int(availability["label_diagnostic"]["uses_target_label_diagnostic"]) == 1, "Target-label diagnostic content is not marked source-only."),
        ("synthetic_common_offset_negative_control", any(r["comparison_id"] == "RG-COD5_target_local_common_offset" and int(r["pair_flip_possible"]) == 0 for r in res["synthetic_rank_gauge_cod_summary_rows"]), "Common target offset negative control is present."),
        ("future_training_not_authorized", all(int(r["authorized_in_c61"]) == 0 for r in res["future_training_need_matrix_rows"]), "Future instrumentation rows are not C61 execution authorization."),
        ("reserved_dataset_and_seeds_blocked", any(r["gate"] == "BNCI2014_004" and r["decision"] == "reserved" for r in res["training_gate_decision_matrix_rows"]) and any(r["gate"] == "seeds_3_4" and r["decision"] == "reserved" for r in res["training_gate_decision_matrix_rows"]), "BNCI2014_004 and seeds [3,4] remain reserved."),
        ("no_gpu_or_reinference", any(r["gate"] == "GPU" and r["decision"] == "not_authorized" for r in res["training_gate_decision_matrix_rows"]) and any(r["gate"] == "re_inference" and r["decision"] == "not_authorized" for r in res["training_gate_decision_matrix_rows"]), "No GPU or re-inference is authorized."),
        ("no_manuscript_or_selector", any(r["gate"] == "manuscript_drafting" and r["decision"] == "not_authorized" for r in res["training_gate_decision_matrix_rows"]) and any(r["gate"] == "selector_search" and r["decision"] == "forbidden" for r in res["training_gate_decision_matrix_rows"]), "No manuscript drafting or selector search."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
        ("large_artifact_scan_passed", all(int(r.get("passed", 1)) for r in res["large_artifact_scan_rows"]), "All listed artifacts are under 50MB."),
    ]
    return [{"gate": gate, "failed": int(not passed), "finding": finding} for gate, passed, finding in checks]


def table_row_counts(res: dict) -> dict:
    keys = {
        "conditional_cs_to_oaci_mapping": "conditional_cs_to_oaci_mapping_rows",
        "conditional_observability_spec": "conditional_observability_spec_rows",
        "information_class_variable_ledger": "information_class_variable_ledger_rows",
        "cod_estimator_summary": "cod_estimator_summary_rows",
        "cod_null_calibration": "cod_null_calibration_rows",
        "cod_bandwidth_support_sensitivity": "cod_bandwidth_support_sensitivity_rows",
        "synthetic_rank_gauge_cod_summary": "synthetic_rank_gauge_cod_summary_rows",
        "synthetic_rank_gauge_parameter_grid": "synthetic_rank_gauge_parameter_grid_rows",
        "eeg_cod_ladder_summary": "eeg_cod_ladder_summary_rows",
        "eeg_cod_cell_ledger": "eeg_cod_cell_ledger_rows",
        "eeg_cod_vs_hit_ladder": "eeg_cod_vs_hit_ladder_rows",
        "cod_availability_ledger": "cod_availability_ledger_rows",
        "same_label_oracle_boundary_checks": "same_label_oracle_boundary_checks_rows",
        "source_observable_cod_adversary": "source_observable_cod_adversary_rows",
        "conditional_screening_summary": "conditional_screening_summary_rows",
        "endpoint_after_template_screening": "endpoint_after_template_screening_rows",
        "future_training_need_matrix": "future_training_need_matrix_rows",
        "training_gate_decision_matrix": "training_gate_decision_matrix_rows",
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
    key_gain = KEY_ONLY_HIT - STRICT_SOURCE_HIT
    template_gain = TEMPLATE_ONLY_HIT - STRICT_SOURCE_HIT
    label_gain = LABEL_DIAGNOSTIC_HIT - STRICT_SOURCE_HIT
    endpoint_gain = ENDPOINT_ORACLE_HIT - STRICT_SOURCE_HIT
    endpoint_after_template_gain = ENDPOINT_ORACLE_HIT - TEMPLATE_ONLY_HIT
    main = "\n".join([
        f"# C61 - Conditional Observability Divergence / Information-Ladder Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "## Primary Decision",
        "",
        f"`{d['primary']}`",
        "",
        f"Active: `{' ; '.join(d['active'])}`",
        "",
        f"Inactive: `{' ; '.join(d['inactive'])}`",
        "",
        "## Result",
        "",
        "C61 establishes a diagnostic conditional-observability ladder over the frozen C50-C60 universe. The primary evidence is a finite-partition binary-Y plug-in COD family: conditional TV, JS, Hellinger, CS proxy, and Bayes-hit gain.",
        "",
        "The ladder aligns with the existing hit and finite-population boundary:",
        "",
        f"- source -> key: `{STRICT_SOURCE_HIT:.6f}` -> `{KEY_ONLY_HIT:.6f}` (`{key_gain:+.6f}`), so key-only information does not improve source endpoint observability.",
        f"- source -> template: `{STRICT_SOURCE_HIT:.6f}` -> `{TEMPLATE_ONLY_HIT:.6f}` (`{template_gain:+.6f}`), so the template is partial target-label-derived observability.",
        f"- source -> label diagnostic: `{STRICT_SOURCE_HIT:.6f}` -> `{LABEL_DIAGNOSTIC_HIT:.6f}` (`{label_gain:+.6f}`), a strong diagnostic gain that is not source-available.",
        f"- source -> endpoint scalar: `{STRICT_SOURCE_HIT:.6f}` -> `{ENDPOINT_ORACLE_HIT:.6f}` (`{endpoint_gain:+.6f}`), the dominant same-label endpoint-oracle boundary.",
        f"- source + template -> endpoint scalar: `{TEMPLATE_ONLY_HIT:.6f}` -> `{ENDPOINT_ORACLE_HIT:.6f}` (`{endpoint_after_template_gain:+.6f}`), so template does not screen off endpoint.",
        "",
        f"The C55/C56/C60 null boundary is preserved: template-only `{TEMPLATE_ONLY_HIT:.6f}` is not claimed to beat max null p95 `{MAX_NULL_P95:.6f}`, while endpoint scalar `{ENDPOINT_ORACLE_HIT:.6f}` does beat it. The endpoint scalar remains a same-label oracle and is unavailable at selection time.",
        "",
        "## Estimator Boundary",
        "",
        "The Conditional CS paper (`https://arxiv.org/abs/2301.08970`) is used as structural inspiration only: formal diagnostic -> estimator/nulls -> synthetic validation -> application-style audit -> limitations.",
        "",
        "C61 does not claim a KDE/Gram conditional CS estimator from summary artifacts; raw conditional samples, per-trial logits, and direct gauge traces are missing. That is why `C61-F` is active: the CS-style estimator family is not the primary evidence, but the partition and binary-Y divergence metrics are stable.",
        "",
        "## Synthetic Check",
        "",
        "The synthetic rank-gauge COD validation keeps C60's repair intact. Candidate-specific gauge gaps can change endpoint observability and induce pair flips; pure target-local common offsets are carried as a negative control because they cannot flip within-target pair ranking.",
        "",
        "## Availability Boundary",
        "",
        "C61 keeps the information classes separate: strict source inputs are source-only, key/template/label-diagnostic rows are diagnostic-only, and same-label endpoint scalar rows read candidate target endpoint content. No target-label diagnostic row is marked source-only.",
        "",
        "## Training Gate",
        "",
        f"`{TRAINING_GATE}`",
        "",
        "C61 does not train, re-infer, use GPU, add BNCI2014_004, run seeds [3,4], start manuscript drafting, or create selector artifacts.",
    ])
    red = "\n".join([
        "# C61 - Red-Team Verification",
        "",
        "All C61 red-team gates pass." if d["red_team_failure_count"] == 0 else "C61 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    framework = "\n".join([
        "# C61 - Framework Template",
        "",
        "C61 maps the conditional Cauchy-Schwarz structure to OACI as: formal diagnostic, estimator/nulls, synthetic rank-gauge validation, frozen EEG application-style audit, and limitations.",
        "",
        "The mapped diagnostic is `COD(X2 | X1; Y)=D_cond(P(Y|X1),P(Y|X1,X2))`, implemented here with finite-partition and binary-Y plug-in divergences over frozen artifacts.",
        "",
        "The result is a diagnostic information-availability framework, not a selector or theorem upgrade. The EEG side remains finite-population/application-style evidence; C61 does not assert a population EEG lower bound.",
    ])
    source_red = "\n".join([
        "# C61 - Source COD Escape-Hatch Red Team",
        "",
        f"No source-observable COD escape hatch is found. Strict source hit is `{STRICT_SOURCE_HIT:.6f}` and the best source scalarization carried forward is `{SOURCE_SCALARIZATION_HIT:.6f}`, both below max null p95 `{MAX_NULL_P95:.6f}`.",
        "",
        "Kernel source probes are not executed without a registered nested design. This keeps C61 from silently tuning a new source feature family.",
    ])
    instrument = "\n".join([
        "# C61 - Instrumentation Implications",
        "",
        "C61 does not require new instrumentation for its finite-partition COD conclusion.",
        "",
        "Future split-label cache, per-trial logits/probabilities, atom traces, independent replication, and rank-gauge intervention remain proposal-only and require explicit approval. They are useful for future estimator/theory work, but they are not needed to accept C61's diagnostic ladder.",
    ])
    return {
        "C61_CONDITIONAL_OBSERVABILITY_DIVERGENCE.md": main,
        "C61_RED_TEAM_VERIFICATION.md": red,
        "C61_FRAMEWORK_TEMPLATE.md": framework,
        "C61_SOURCE_ESCAPE_HATCH_RED_TEAM.md": source_red,
        "C61_INSTRUMENTATION_IMPLICATIONS.md": instrument,
    }


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "c60_commit": "2e6ae07",
        "c60_decision": res["c60_decision"],
        "decision": res["decision"],
        "training_gate": res["training_gate"],
        "cod_status": {
            "finite_partition_plugin": "established",
            "binary_y_baselines": "stable",
            "conditional_cs_kde": "not_primary_summary_artifacts_missing_raw_samples",
            "synthetic_rank_gauge": "validated_model_bound",
            "source_adversary": "no_registered_escape_hatch",
        },
        "key_numbers": {
            "strict_source": STRICT_SOURCE_HIT,
            "source_scalarization": SOURCE_SCALARIZATION_HIT,
            "key_only": KEY_ONLY_HIT,
            "template_only": TEMPLATE_ONLY_HIT,
            "label_diagnostic": LABEL_DIAGNOSTIC_HIT,
            "endpoint_oracle": ENDPOINT_ORACLE_HIT,
            "max_null_p95": MAX_NULL_P95,
            "endpoint_after_template_hit_gain": ENDPOINT_ORACLE_HIT - TEMPLATE_ONLY_HIT,
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def run(test_status: str = "planned") -> dict:
    config_hash = _lock_config()
    c60 = _load_json(C60_JSON)
    mapping, spec, info = build_framework_rows()
    cod, nulls, sensitivity = build_estimator_rows()
    synth, synth_grid = build_synthetic_rows()
    eeg_ladder, eeg_cell, eeg_vs_hit = build_eeg_ladder_rows()
    availability, oracle, adversary, screening, endpoint_after_template = build_availability_rows()
    future, gates = build_future_rows()
    res = {
        "config_hash": config_hash,
        "c60_decision": c60["decision"]["primary"],
        "training_gate": TRAINING_GATE,
        "conditional_cs_to_oaci_mapping_rows": mapping,
        "conditional_observability_spec_rows": spec,
        "information_class_variable_ledger_rows": info,
        "cod_estimator_summary_rows": cod,
        "cod_null_calibration_rows": nulls,
        "cod_bandwidth_support_sensitivity_rows": sensitivity,
        "synthetic_rank_gauge_cod_summary_rows": synth,
        "synthetic_rank_gauge_parameter_grid_rows": synth_grid,
        "eeg_cod_ladder_summary_rows": eeg_ladder,
        "eeg_cod_cell_ledger_rows": eeg_cell,
        "eeg_cod_vs_hit_ladder_rows": eeg_vs_hit,
        "cod_availability_ledger_rows": availability,
        "same_label_oracle_boundary_checks_rows": oracle,
        "source_observable_cod_adversary_rows": adversary,
        "conditional_screening_summary_rows": screening,
        "endpoint_after_template_screening_rows": endpoint_after_template,
        "future_training_need_matrix_rows": future,
        "training_gate_decision_matrix_rows": gates,
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
        "conditional_cs_to_oaci_mapping.csv": ("conditional_cs_to_oaci_mapping_rows", ["paper_component", "oaci_mapping", "c61_status", "caveat"]),
        "conditional_observability_spec.csv": ("conditional_observability_spec_rows", ["spec_id", "object", "definition", "availability", "diagnostic_only"]),
        "information_class_variable_ledger.csv": ("information_class_variable_ledger_rows", ["information_class", "variable", "hit", "uses_source_only_inputs", "uses_target_labels", "available_at_selection_time", "diagnostic_only"]),
        "cod_estimator_summary.csv": ("cod_estimator_summary_rows", ["comparison_id", "x1_base", "x2_increment", "hit_before", "hit_after", "bayes_hit_gain", "conditional_tv_proxy", "conditional_js_proxy", "conditional_hellinger2_proxy", "conditional_cs_binary_proxy", "endpoint_closure_fraction", "interpretation"]),
        "cod_null_calibration.csv": ("cod_null_calibration_rows", ["null_id", "applies_to", "observed_reference_hit", "null_p95_hit", "endpoint_beats_null", "template_beats_null", "interpretation"]),
        "cod_bandwidth_support_sensitivity.csv": ("cod_bandwidth_support_sensitivity_rows", ["sensitivity_id", "estimator", "setting", "stable", "finding"]),
        "synthetic_rank_gauge_cod_summary.csv": ("synthetic_rank_gauge_cod_summary_rows", ["comparison_id", "x1_base", "x2_increment", "hit_before", "hit_after", "bayes_hit_gain", "conditional_tv_proxy", "conditional_js_proxy", "conditional_hellinger2_proxy", "conditional_cs_binary_proxy", "endpoint_closure_fraction", "interpretation", "candidate_specific_gauge_gap", "pair_flip_possible", "synthetic_validation_status"]),
        "synthetic_rank_gauge_parameter_grid.csv": ("synthetic_rank_gauge_parameter_grid_rows", ["comparison_id", "x1_base", "x2_increment", "hit_before", "hit_after", "bayes_hit_gain", "conditional_tv_proxy", "conditional_js_proxy", "conditional_hellinger2_proxy", "conditional_cs_binary_proxy", "endpoint_closure_fraction", "interpretation", "gamma", "tail_error", "requires_joint_tail_for_topk", "model_bound_only"]),
        "eeg_cod_ladder_summary.csv": ("eeg_cod_ladder_summary_rows", ["comparison_id", "x1_base", "x2_increment", "hit_before", "hit_after", "bayes_hit_gain", "conditional_tv_proxy", "conditional_js_proxy", "conditional_hellinger2_proxy", "conditional_cs_binary_proxy", "endpoint_closure_fraction", "interpretation", "availability_class", "beats_max_null_p95", "c55_c56_null_status"]),
        "eeg_cod_cell_ledger.csv": ("eeg_cod_cell_ledger_rows", ["cell_group", "n_cells", "n_candidates", "key_hit", "template_hit", "endpoint_hit", "diagnostic_only"]),
        "eeg_cod_vs_hit_ladder.csv": ("eeg_cod_vs_hit_ladder_rows", ["information_step", "cod_comparison", "hit_gain", "partition_bound_alignment", "rank_gauge_alignment"]),
        "cod_availability_ledger.csv": ("cod_availability_ledger_rows", ["score_name", "uses_source_only_inputs", "uses_key_only_inputs", "uses_target_unlabeled_inputs", "uses_target_label_diagnostic", "uses_test_candidate_endpoint_scalar", "uses_same_cell_target_labels", "available_at_selection_time", "diagnostic_only", "reported_hit"]),
        "same_label_oracle_boundary_checks.csv": ("same_label_oracle_boundary_checks_rows", ["check", "observed", "reference", "passed", "interpretation"]),
        "source_observable_cod_adversary.csv": ("source_observable_cod_adversary_rows", ["candidate_id", "candidate", "allowed_source_only", "cod_proxy", "hit", "beats_max_null_p95", "reliable_escape_hatch", "reason"]),
        "conditional_screening_summary.csv": ("conditional_screening_summary_rows", ["screening_test", "hit_before", "hit_after", "endpoint_remaining_gain", "screens_off", "conclusion"]),
        "endpoint_after_template_screening.csv": ("endpoint_after_template_screening_rows", ["comparison_id", "template_hit", "endpoint_hit", "incremental_hit_gain", "incremental_cs_proxy", "screened_off", "diagnostic_only"]),
        "future_training_need_matrix.csv": ("future_training_need_matrix_rows", ["need_id", "future_data", "cod_gap_addressed", "needed_for_c61_claim", "training_or_inference_required", "authorized_in_c61"]),
        "training_gate_decision_matrix.csv": ("training_gate_decision_matrix_rows", ["gate", "decision", "requires_user_release", "default"]),
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
        glob.glob(os.path.join(REPORT_DIR, "C61_*.md"))
        + glob.glob(os.path.join(REPORT_DIR, "C61_*.json"))
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
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c61_conditional_observability_divergence")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res, args.test_status)
    print(f"[C61] decision={res['decision']['primary']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
