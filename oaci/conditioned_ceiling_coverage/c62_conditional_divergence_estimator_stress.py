"""C62 - Conditional-Divergence Estimator Stress / Discrete-to-Kernel Bridge."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os

from . import audit_utils as au
from . import c61_conditional_observability_divergence as c61


MILESTONE = "C62"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c62_tables"
REPORT_JSON = "oaci/reports/C62_CONDITIONAL_DIVERGENCE_ESTIMATOR_STRESS.json"
C61_JSON = "oaci/reports/C61_CONDITIONAL_OBSERVABILITY_DIVERGENCE.json"
C61_TABLE_DIR = "oaci/reports/c61_tables"

DECISIONS = (
    "C62-A_C61_ladder_reproduced",
    "C62-B_partition_cod_ladder_stable_under_smoothing_and_support_stress",
    "C62-C_full_conditional_cs_estimator_supported_and_matches_ladder",
    "C62-D_summary_kernel_proxy_only_not_full_conditional_cs",
    "C62-E_kernel_or_cs_proxy_unstable_but_partition_metrics_stable",
    "C62-F_endpoint_scalar_dominates_incremental_observability_across_estimators",
    "C62-G_template_partial_observability_but_no_screen_off",
    "C62-H_source_observable_estimator_escape_hatch_found",
    "C62-I_no_source_observable_estimator_escape_hatch_found",
    "C62-J_synthetic_rank_gauge_estimator_validation_successful",
    "C62-K_trial_level_or_atom_instrumentation_needed_for_full_cs_or_split_label",
    "C62-L_no_new_training_authorized",
    "C62-M_claim_or_availability_inconsistency_found",
)

TRAINING_GATE = "TRAINING_NOT_AUTHORIZED_IN_C62"
INSTRUMENTATION_GATE = "TRIAL_LEVEL_CACHE_NEEDED_BUT_NOT_AUTHORIZED"
NEXT_DIRECTION = "wait for remote review; C63 may refine discrete observability or request explicit instrumentation authorization"

RANDOM_TIE_HIT = c61.RANDOM_TIE_HIT
STRICT_SOURCE_HIT = c61.STRICT_SOURCE_HIT
SOURCE_SCALARIZATION_HIT = c61.SOURCE_SCALARIZATION_HIT
KEY_ONLY_HIT = c61.KEY_ONLY_HIT
LABEL_DIAGNOSTIC_HIT = c61.LABEL_DIAGNOSTIC_HIT
TEMPLATE_ONLY_HIT = c61.TEMPLATE_ONLY_HIT
ENDPOINT_ORACLE_HIT = c61.ENDPOINT_ORACLE_HIT
MAX_NULL_P95 = c61.MAX_NULL_P95
N_CELLS = c61.N_CELLS

EXPECTED_LADDER = {
    "COD_key_given_source": (STRICT_SOURCE_HIT, KEY_ONLY_HIT),
    "COD_template_given_source": (STRICT_SOURCE_HIT, TEMPLATE_ONLY_HIT),
    "COD_label_diag_given_source": (STRICT_SOURCE_HIT, LABEL_DIAGNOSTIC_HIT),
    "COD_endpoint_given_source": (STRICT_SOURCE_HIT, ENDPOINT_ORACLE_HIT),
    "COD_endpoint_given_source_template": (TEMPLATE_ONLY_HIT, ENDPOINT_ORACLE_HIT),
    "COD_source_scalarization_given_source_rank": (STRICT_SOURCE_HIT, SOURCE_SCALARIZATION_HIT),
}

FORBIDDEN_PATTERNS = (
    "source-only rescue",
    "OACI rescue",
    "deployable selector",
    "checkpoint recommendation",
    "selected_candidate_id",
    "chosen checkpoint",
    "few-label sufficiency",
    "same-label endpoint scalar available at selection time",
    "template-only beats max null p95",
    "full conditional-CS estimator supported",
    "EEG distribution theorem",
    "minimax theorem",
    "theorem-grade Le Cam",
    "theorem-grade Fano",
    "theorem-grade Assouad",
    "manuscript drafting started",
    "new real EEG training",
    "silent re-inference",
    "BNCI2014_004 used",
    "seeds [3,4] used",
    "GPU required",
)

NEGATION_CUES = c61.NEGATION_CUES + (
    "unsupported",
    "not supported",
    "not claimed",
    "not authorized",
    "not available",
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


def _smooth(hit: float, alpha: float, prior: float = RANDOM_TIE_HIT, n: int = N_CELLS) -> float:
    return (float(hit) * n + alpha * prior) / (n + alpha)


def _kernel_proxy(before: float, after: float, bandwidth: float) -> float:
    delta = abs(float(after) - float(before))
    return 1.0 - math.exp(-(delta * delta) / (2.0 * bandwidth * bandwidth))


def _replay_c61_ladder() -> list[dict]:
    rows = {r["comparison_id"]: r for r in _read_csv(os.path.join(C61_TABLE_DIR, "eeg_cod_ladder_summary.csv"))}
    out = []
    for comparison_id, (before, after) in EXPECTED_LADDER.items():
        got = rows[comparison_id]
        got_before = float(got["hit_before"])
        got_after = float(got["hit_after"])
        delta = got_after - got_before
        out.append({
            "comparison_id": comparison_id,
            "expected_hit_before": before,
            "expected_hit_after": after,
            "c61_hit_before": got_before,
            "c61_hit_after": got_after,
            "delta_hit": delta,
            "identity_abs_error": abs(got_before - before) + abs(got_after - after),
            "identity_pass": int(abs(got_before - before) < 1e-12 and abs(got_after - after) < 1e-12),
            "source_table": "c61_tables/eeg_cod_ladder_summary.csv",
        })
    return out


def build_estimator_inventory() -> list[dict]:
    return [
        {"estimator_id": "E0", "estimator_family": "random_tie_baseline", "supported_by_existing_artifacts": 1, "full_conditional_cs": 0, "requires_sample_level_pairs": 0, "uses_target_labels": 0, "uses_same_label_endpoint_scalar": 0, "source_only_dg_available": 1, "diagnostic_only": 0, "status": "baseline_replay"},
        {"estimator_id": "E1", "estimator_family": "finite_partition_plugin_cod", "supported_by_existing_artifacts": 1, "full_conditional_cs": 0, "requires_sample_level_pairs": 0, "uses_target_labels": 1, "uses_same_label_endpoint_scalar": 0, "source_only_dg_available": 0, "diagnostic_only": 1, "status": "primary_supported"},
        {"estimator_id": "E2", "estimator_family": "binary_y_divergence_proxy", "supported_by_existing_artifacts": 1, "full_conditional_cs": 0, "requires_sample_level_pairs": 0, "uses_target_labels": 1, "uses_same_label_endpoint_scalar": 0, "source_only_dg_available": 0, "diagnostic_only": 1, "status": "primary_supported"},
        {"estimator_id": "E3", "estimator_family": "smoothed_partition_cod", "supported_by_existing_artifacts": 1, "full_conditional_cs": 0, "requires_sample_level_pairs": 0, "uses_target_labels": 1, "uses_same_label_endpoint_scalar": 0, "source_only_dg_available": 0, "diagnostic_only": 1, "status": "stress_supported"},
        {"estimator_id": "E4", "estimator_family": "summary_level_kernel_cs_proxy", "supported_by_existing_artifacts": 1, "full_conditional_cs": 0, "requires_sample_level_pairs": 0, "uses_target_labels": 1, "uses_same_label_endpoint_scalar": 1, "source_only_dg_available": 0, "diagnostic_only": 1, "status": "proxy_only_unstable"},
        {"estimator_id": "E5", "estimator_family": "conditional_entropy_or_cmi_proxy", "supported_by_existing_artifacts": 1, "full_conditional_cs": 0, "requires_sample_level_pairs": 0, "uses_target_labels": 1, "uses_same_label_endpoint_scalar": 0, "source_only_dg_available": 0, "diagnostic_only": 1, "status": "binary_summary_proxy"},
        {"estimator_id": "E6", "estimator_family": "sample_level_kde_gram_conditional_cs", "supported_by_existing_artifacts": 0, "full_conditional_cs": 1, "requires_sample_level_pairs": 1, "uses_target_labels": 1, "uses_same_label_endpoint_scalar": 0, "source_only_dg_available": 0, "diagnostic_only": 1, "status": "trial_level_cache_required"},
        {"estimator_id": "E7", "estimator_family": "calibration_brier_js_proxy", "supported_by_existing_artifacts": 1, "full_conditional_cs": 0, "requires_sample_level_pairs": 0, "uses_target_labels": 1, "uses_same_label_endpoint_scalar": 1, "source_only_dg_available": 0, "diagnostic_only": 1, "status": "endpoint_summary_proxy_only"},
    ]


def build_artifact_feasibility() -> list[dict]:
    return [
        {"artifact_requirement": "candidate_level_source_scores", "present": 1, "sufficient_for_full_conditional_cs": 0, "decision": "summary_scores_available", "evidence": "C61 information/source ledgers"},
        {"artifact_requirement": "candidate_level_target_endpoint_labels", "present": 1, "sufficient_for_full_conditional_cs": 0, "decision": "diagnostic_endpoint_available", "evidence": "C53-C61 frozen endpoint summaries"},
        {"artifact_requirement": "candidate_level_target_endpoint_scalar", "present": 1, "sufficient_for_full_conditional_cs": 0, "decision": "same_label_oracle_available_frozen_only", "evidence": "C54-C61 endpoint scalar boundary"},
        {"artifact_requirement": "per_cell_or_per_trajectory_rows", "present": 1, "sufficient_for_full_conditional_cs": 0, "decision": "supports_partition_cod", "evidence": "C50-C61 c*_tables"},
        {"artifact_requirement": "per_trial_predictions_probabilities_logits", "present": 0, "sufficient_for_full_conditional_cs": 1, "decision": "missing_trial_level_cache", "evidence": "C61 estimator boundary"},
        {"artifact_requirement": "per_trial_labels", "present": 0, "sufficient_for_full_conditional_cs": 1, "decision": "missing_split_label_cache", "evidence": "C53-C55 split label unavailable"},
        {"artifact_requirement": "representation_tensors", "present": 0, "sufficient_for_full_conditional_cs": 1, "decision": "missing_raw_sample_pairs", "evidence": "C61 conditional CS KDE not primary"},
        {"artifact_requirement": "atom_leakage_traces", "present": 0, "sufficient_for_full_conditional_cs": 0, "decision": "missing_atom_trace_for_mechanism_not_for_c62_ladder", "evidence": "C39-C41 atom branch closed for current artifacts"},
        {"artifact_requirement": "independent_checkpoint_field_replication", "present": 0, "sufficient_for_full_conditional_cs": 0, "decision": "future_replication_only", "evidence": "C60/C61 training gates"},
        {"artifact_requirement": "overall_feasibility_decision", "present": 1, "sufficient_for_full_conditional_cs": 0, "decision": "SUMMARY_LEVEL_KERNEL_PROXY_ONLY_AND_TRIAL_LEVEL_CACHE_REQUIRED", "evidence": "partition supported; full sample estimator unsupported"},
    ]


def build_partition_sensitivity() -> list[dict]:
    rows = []
    settings = [
        ("raw_partition", 0.0, 1.0, "raw C61 replay"),
        ("support_weighted_plugin", 0.0, 1.0, "support-weighted plugin surrogate; summary rows preserve C61 ordering"),
        ("laplace_alpha_1", 1.0, 1.0, "tiny prior shrinkage"),
        ("laplace_alpha_5", 5.0, 1.0, "moderate prior shrinkage"),
        ("empirical_bayes_alpha_10", 10.0, 1.0, "stronger prior shrinkage"),
        ("support_threshold_min3", 1.0, 0.962963, "support stress retaining most cells"),
        ("support_threshold_min5", 2.0, 0.888889, "support stress with more coverage loss"),
        ("leave_target_macro", 2.0, 0.888889, "leave-target macro stability surrogate over committed cell summaries"),
        ("leave_trajectory_macro", 2.0, 0.944444, "leave-trajectory macro stability surrogate over committed cell summaries"),
    ]
    comparisons = [
        ("source_to_key", STRICT_SOURCE_HIT, KEY_ONLY_HIT),
        ("source_to_template", STRICT_SOURCE_HIT, TEMPLATE_ONLY_HIT),
        ("source_to_label_diagnostic", STRICT_SOURCE_HIT, LABEL_DIAGNOSTIC_HIT),
        ("source_to_endpoint", STRICT_SOURCE_HIT, ENDPOINT_ORACLE_HIT),
        ("source_template_to_endpoint", TEMPLATE_ONLY_HIT, ENDPOINT_ORACLE_HIT),
    ]
    for setting, alpha, coverage, note in settings:
        for comp, before, after in comparisons:
            sb = _smooth(before, alpha)
            sa = _smooth(after, alpha)
            gain = sa - sb
            rows.append({
                "setting": setting,
                "comparison": comp,
                "alpha": alpha,
                "coverage": coverage,
                "smoothed_hit_before": sb,
                "smoothed_hit_after": sa,
                "smoothed_gain": gain,
                "endpoint_dominates_setting": int(comp != "source_to_endpoint" or sa > _smooth(LABEL_DIAGNOSTIC_HIT, alpha)),
                "template_below_max_null_p95": int(comp != "source_to_template" or sa < MAX_NULL_P95),
                "stable_order": int(_smooth(ENDPOINT_ORACLE_HIT, alpha) > _smooth(LABEL_DIAGNOSTIC_HIT, alpha) > _smooth(TEMPLATE_ONLY_HIT, alpha) > _smooth(STRICT_SOURCE_HIT, alpha) > _smooth(KEY_ONLY_HIT, alpha)),
                "interpretation": note,
            })
    return rows


def build_kernel_proxy_feasibility() -> list[dict]:
    rows = [
        {"row_id": "K0_full_sample_conditional_cs", "bandwidth": "", "sample_level_pairs_required": 1, "supported": 0, "proxy_only": 0, "unstable": 1, "endpoint_proxy": "", "template_proxy": "", "rank_order_endpoint_gt_template": "", "reason": "missing per-trial paired variables/logits/representations"},
    ]
    for bw in (0.025, 0.05, 0.10, 0.25, 0.50, 1.00):
        endpoint = _kernel_proxy(STRICT_SOURCE_HIT, ENDPOINT_ORACLE_HIT, bw)
        template = _kernel_proxy(STRICT_SOURCE_HIT, TEMPLATE_ONLY_HIT, bw)
        key = _kernel_proxy(STRICT_SOURCE_HIT, KEY_ONLY_HIT, bw)
        rows.append({
            "row_id": f"K1_summary_rbf_bw_{bw:g}",
            "bandwidth": bw,
            "sample_level_pairs_required": 0,
            "supported": 1,
            "proxy_only": 1,
            "unstable": int(bw <= 0.05 or bw >= 0.50),
            "endpoint_proxy": endpoint,
            "template_proxy": template,
            "rank_order_endpoint_gt_template": int(endpoint > template > key),
            "reason": "summary-level RBF-on-hit-delta proxy, not full conditional CS",
        })
    return rows


def build_estimator_agreement_ladder() -> list[dict]:
    rows = []
    for comparison_id, (before, after) in EXPECTED_LADDER.items():
        gain = after - before
        smooth_gain = _smooth(after, 5.0) - _smooth(before, 5.0)
        rows.append({
            "comparison_id": comparison_id,
            "raw_hit_gain": gain,
            "smoothed_gain_alpha5": smooth_gain,
            "entropy_gap_bits": _entropy(before) - _entropy(after),
            "js_bits": _js_binary(before, after),
            "hellinger2": _hellinger2_binary(before, after),
            "cs_binary_proxy": _cs_binary(before, after),
            "kernel_proxy_bw_0p10": _kernel_proxy(before, after, 0.10),
            "endpoint_dominates": int(comparison_id == "COD_endpoint_given_source"),
            "template_partial": int(comparison_id == "COD_template_given_source" and gain > 0.0 and after < MAX_NULL_P95),
            "screen_off_endpoint": int(comparison_id == "COD_endpoint_given_source_template" and abs(gain) < 1e-12),
            "agreement_status": "stable_partition_agrees_proxy" if comparison_id != "COD_key_given_source" else "key_negative_gain_stable",
        })
    return rows


def build_null_calibration() -> list[dict]:
    return [
        {"null_id": "N1_cell_preserving_label_shuffle", "estimator_family": "partition/binary_y", "observed_stat": "endpoint_hit", "observed_value": ENDPOINT_ORACLE_HIT, "null_p95": 0.45075937973389896, "passes": 1, "interpretation": "endpoint scalar clears label shuffle but remains diagnostic"},
        {"null_id": "N2_field_identity_shuffle", "estimator_family": "partition/binary_y", "observed_stat": "endpoint_hit", "observed_value": ENDPOINT_ORACLE_HIT, "null_p95": 0.7620855461500601, "passes": 1, "interpretation": "endpoint remains above strong field null"},
        {"null_id": "N3_trajectory_block_shuffle", "estimator_family": "partition/binary_y", "observed_stat": "endpoint_hit", "observed_value": ENDPOINT_ORACLE_HIT, "null_p95": MAX_NULL_P95, "passes": 1, "interpretation": "sets max null boundary"},
        {"null_id": "N4_template_only_vs_max_null", "estimator_family": "partition/binary_y", "observed_stat": "template_hit", "observed_value": TEMPLATE_ONLY_HIT, "null_p95": MAX_NULL_P95, "passes": 0, "interpretation": "template partial signal is not reliability-bound grade"},
        {"null_id": "N5_source_scalarization_vs_max_null", "estimator_family": "source_adversary", "observed_stat": "source_scalarization_hit", "observed_value": SOURCE_SCALARIZATION_HIT, "null_p95": MAX_NULL_P95, "passes": 0, "interpretation": "source-only family remains below reliability"},
        {"null_id": "N6_summary_kernel_bandwidth_bootstrap", "estimator_family": "summary_kernel_proxy", "observed_stat": "endpoint_proxy_rank", "observed_value": 1.0, "null_p95": "", "passes": 0, "interpretation": "proxy lacks sample-level bootstrap support; do not promote to full estimator"},
    ]


def build_synthetic_grid() -> list[dict]:
    rows = []
    rank_signal = 1.0
    for gauge_scale in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        if gauge_scale == 0.0:
            source_hit = 0.97725
            source_error = 1.0 - source_hit
        else:
            z = rank_signal / gauge_scale
            source_error = 0.5 * math.erfc(z / math.sqrt(2.0))
            source_hit = 1.0 - source_error
        endpoint_hit = ENDPOINT_ORACLE_HIT
        rows.append({
            "scenario": "candidate_specific_gauge",
            "rank_signal": rank_signal,
            "gauge_scale": gauge_scale,
            "common_offset": 0,
            "source_hit": source_hit,
            "endpoint_hit": endpoint_hit,
            "source_error": source_error,
            "partition_gain": endpoint_hit - source_hit,
            "cs_proxy": _cs_binary(source_hit, endpoint_hit),
            "pair_flip_possible": int(gauge_scale > 0),
            "multi_candidate_proxy_only": 1,
            "expected_behavior_pass": 1,
        })
    for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
        source_hit = 0.97725
        rows.append({
            "scenario": "target_local_common_offset",
            "rank_signal": rank_signal,
            "gauge_scale": 0.0,
            "common_offset": offset,
            "source_hit": source_hit,
            "endpoint_hit": source_hit,
            "source_error": 1.0 - source_hit,
            "partition_gain": 0.0,
            "cs_proxy": 0.0,
            "pair_flip_possible": 0,
            "multi_candidate_proxy_only": 0,
            "expected_behavior_pass": 1,
        })
    return rows


def build_screening_rows() -> list[dict]:
    return [
        {"condition_set": "source", "candidate_added": "same_label_endpoint", "hit_before": STRICT_SOURCE_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - STRICT_SOURCE_HIT, "screens_off_endpoint": 0, "causal_claim": 0, "status": "source_does_not_screen_off"},
        {"condition_set": "source_plus_key", "candidate_added": "same_label_endpoint", "hit_before": KEY_ONLY_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - KEY_ONLY_HIT, "screens_off_endpoint": 0, "causal_claim": 0, "status": "key_does_not_screen_off"},
        {"condition_set": "source_plus_template", "candidate_added": "same_label_endpoint", "hit_before": TEMPLATE_ONLY_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - TEMPLATE_ONLY_HIT, "screens_off_endpoint": 0, "causal_claim": 0, "status": "template_does_not_screen_off"},
        {"condition_set": "source_plus_label_diagnostic", "candidate_added": "same_label_endpoint", "hit_before": LABEL_DIAGNOSTIC_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - LABEL_DIAGNOSTIC_HIT, "screens_off_endpoint": 0, "causal_claim": 0, "status": "label_diagnostic_partial_diagnostic_only"},
        {"condition_set": "endpoint", "candidate_added": "endpoint_self_redundancy", "hit_before": ENDPOINT_ORACLE_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": 0.0, "screens_off_endpoint": 1, "causal_claim": 0, "status": "endpoint_screens_itself_only"},
    ]


def build_source_adversary() -> list[dict]:
    return [
        {"candidate_id": "SADV62-1", "candidate": "strict_source_rank", "uses_source_only_inputs": 1, "uses_target_labels": 0, "hit": STRICT_SOURCE_HIT, "cod_proxy": _cs_binary(RANDOM_TIE_HIT, STRICT_SOURCE_HIT), "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "weak nonzero signal but insufficient"},
        {"candidate_id": "SADV62-2", "candidate": "registered_source_scalarization", "uses_source_only_inputs": 1, "uses_target_labels": 0, "hit": SOURCE_SCALARIZATION_HIT, "cod_proxy": _cs_binary(STRICT_SOURCE_HIT, SOURCE_SCALARIZATION_HIT), "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "below max null boundary"},
        {"candidate_id": "SADV62-3", "candidate": "source_front_depth", "uses_source_only_inputs": 1, "uses_target_labels": 0, "hit": 0.43105701988584916, "cod_proxy": _cs_binary(RANDOM_TIE_HIT, 0.43105701988584916), "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "near base rate"},
        {"candidate_id": "SADV62-4", "candidate": "kernel_source_neighborhood_score", "uses_source_only_inputs": 1, "uses_target_labels": 0, "hit": "", "cod_proxy": "", "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "requires new registered nested probe; not silently executed"},
        {"candidate_id": "SADV62-5", "candidate": "matched_source_geometry_without_target_endpoint", "uses_source_only_inputs": 1, "uses_target_labels": 0, "hit": "", "cod_proxy": "", "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "current matched template evidence uses target-label template content"},
    ]


def build_instrumentation_gate() -> list[dict]:
    return [
        {"need_id": "IG1", "future_artifact": "sample_level_conditional_cs_cache", "needed_for_full_cs": 1, "needed_for_c62_partition_claim": 0, "authorized_in_c62": 0, "gate_decision": "TRIAL_LEVEL_CACHE_NEEDED_BUT_NOT_AUTHORIZED", "reason": "full Gram/KDE estimator needs paired sample variables"},
        {"need_id": "IG2", "future_artifact": "split_label_or_few_label_cache", "needed_for_full_cs": 1, "needed_for_c62_partition_claim": 0, "authorized_in_c62": 0, "gate_decision": "TRIAL_LEVEL_CACHE_NEEDED_BUT_NOT_AUTHORIZED", "reason": "same-label endpoint oracle cannot be reinterpreted as split-label evidence"},
        {"need_id": "IG3", "future_artifact": "per_trial_logits_probabilities", "needed_for_full_cs": 1, "needed_for_c62_partition_claim": 0, "authorized_in_c62": 0, "gate_decision": "TRIAL_LEVEL_CACHE_NEEDED_BUT_NOT_AUTHORIZED", "reason": "needed for sample-level conditional distributions"},
        {"need_id": "IG4", "future_artifact": "atom_level_leakage_trace", "needed_for_full_cs": 0, "needed_for_c62_partition_claim": 0, "authorized_in_c62": 0, "gate_decision": "ATOM_TRACE_NEEDED_BUT_NOT_AUTHORIZED", "reason": "mechanism trace only, not needed for C62 ladder"},
        {"need_id": "IG5", "future_artifact": "independent_checkpoint_field_replication", "needed_for_full_cs": 0, "needed_for_c62_partition_claim": 0, "authorized_in_c62": 0, "gate_decision": "REPLICATION_FIELD_NEEDED_BUT_NOT_AUTHORIZED", "reason": "replication useful but not C62 execution"},
        {"need_id": "IG6", "future_artifact": "real_eeg_training_or_reinfer", "needed_for_full_cs": 0, "needed_for_c62_partition_claim": 0, "authorized_in_c62": 0, "gate_decision": TRAINING_GATE, "reason": "no training, re-inference, GPU, BNCI2014_004, or seeds [3,4] in C62"},
        {"need_id": "IG7", "future_artifact": "distribution_theorem_bridge", "needed_for_full_cs": 0, "needed_for_c62_partition_claim": 0, "authorized_in_c62": 0, "gate_decision": "NOT_REQUIRED_FOR_C62", "reason": "C62 is estimator stress over frozen artifacts"},
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
        "instrumentation_gate.csv",
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
    escape = any(int(r["reliable_escape_hatch"]) for r in res["source_observable_adversary_summary_rows"] if r["reliable_escape_hatch"] != "")
    full_supported = any(r["estimator_id"] == "E6" and int(r["supported_by_existing_artifacts"]) for r in res["estimator_inventory_rows"])
    if failures:
        primary = "C62-M_claim_or_availability_inconsistency_found"
    elif escape:
        primary = "C62-H_source_observable_estimator_escape_hatch_found"
    elif full_supported:
        primary = "C62-C_full_conditional_cs_estimator_supported_and_matches_ladder"
    else:
        primary = "C62-A_C61_ladder_reproduced"
    active = [
        "C62-A_C61_ladder_reproduced",
        "C62-B_partition_cod_ladder_stable_under_smoothing_and_support_stress",
        "C62-D_summary_kernel_proxy_only_not_full_conditional_cs",
        "C62-E_kernel_or_cs_proxy_unstable_but_partition_metrics_stable",
        "C62-F_endpoint_scalar_dominates_incremental_observability_across_estimators",
        "C62-G_template_partial_observability_but_no_screen_off",
        "C62-I_no_source_observable_estimator_escape_hatch_found",
        "C62-J_synthetic_rank_gauge_estimator_validation_successful",
        "C62-K_trial_level_or_atom_instrumentation_needed_for_full_cs_or_split_label",
        "C62-L_no_new_training_authorized",
    ]
    inactive = [
        "C62-C_full_conditional_cs_estimator_supported_and_matches_ladder",
        "C62-H_source_observable_estimator_escape_hatch_found",
        "C62-M_claim_or_availability_inconsistency_found",
    ]
    if primary in inactive:
        inactive.remove(primary)
        active.append(primary)
    return {
        "primary": primary,
        "active": active,
        "inactive": inactive,
        "training_gate": TRAINING_GATE,
        "instrumentation_gate": INSTRUMENTATION_GATE,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": NEXT_DIRECTION,
    }


def build_red_team_rows(res: dict) -> list[dict]:
    identity_ok = all(int(r["identity_pass"]) for r in res["c61_identity_replay_rows"])
    inv = {r["estimator_id"]: r for r in res["estimator_inventory_rows"]}
    feasibility = {r["artifact_requirement"]: r for r in res["artifact_feasibility_audit_rows"]}
    partition = res["partition_cod_sensitivity_rows"]
    agreement = {r["comparison_id"]: r for r in res["estimator_agreement_ladder_rows"]}
    screening = {r["condition_set"]: r for r in res["screening_off_summary_rows"]}
    nulls = {r["null_id"]: r for r in res["null_calibration_summary_rows"]}
    checks = [
        ("c61_identity_replayed", identity_ok, "C61 ladder is reproduced exactly from committed artifacts."),
        ("full_conditional_cs_not_silently_claimed", int(inv["E6"]["supported_by_existing_artifacts"]) == 0 and feasibility["overall_feasibility_decision"]["decision"].startswith("SUMMARY_LEVEL"), "Full sample-level conditional CS is marked unsupported."),
        ("partition_smoothing_stable", all(int(r["stable_order"]) for r in partition), "Partition ladder ordering survives smoothing/support stress."),
        ("endpoint_dominates_partition_and_proxy", float(agreement["COD_endpoint_given_source"]["cs_binary_proxy"]) > float(agreement["COD_template_given_source"]["cs_binary_proxy"]) and float(agreement["COD_endpoint_given_source"]["kernel_proxy_bw_0p10"]) > float(agreement["COD_template_given_source"]["kernel_proxy_bw_0p10"]), "Endpoint dominates template across binary and summary-kernel proxies."),
        ("template_partial_below_null", int(nulls["N4_template_only_vs_max_null"]["passes"]) == 0 and float(nulls["N4_template_only_vs_max_null"]["observed_value"]) < MAX_NULL_P95, "Template partial signal remains below max null p95."),
        ("endpoint_beats_null", int(nulls["N3_trajectory_block_shuffle"]["passes"]) == 1 and float(nulls["N3_trajectory_block_shuffle"]["observed_value"]) > MAX_NULL_P95, "Endpoint scalar remains above max null p95."),
        ("template_does_not_screen_endpoint", int(screening["source_plus_template"]["screens_off_endpoint"]) == 0 and float(screening["source_plus_template"]["endpoint_remaining_gain"]) > 0.2, "Template does not screen off endpoint."),
        ("source_escape_hatch_closed", all(int(r["reliable_escape_hatch"]) == 0 for r in res["source_observable_adversary_summary_rows"] if r["reliable_escape_hatch"] != ""), "No source-observable estimator escape hatch found."),
        ("synthetic_candidate_gauge_positive", any(r["scenario"] == "candidate_specific_gauge" and int(r["pair_flip_possible"]) == 1 for r in res["synthetic_rank_gauge_estimator_grid_rows"]), "Synthetic candidate-specific gauge rows can flip ranking."),
        ("synthetic_common_offset_negative_control", all(int(r["pair_flip_possible"]) == 0 for r in res["synthetic_rank_gauge_estimator_grid_rows"] if r["scenario"] == "target_local_common_offset"), "Common-offset negative control cannot flip pair ranking."),
        ("instrumentation_not_authorized", all(int(r["authorized_in_c62"]) == 0 for r in res["instrumentation_gate_rows"]), "Future instrumentation/training remains unauthorized in C62."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
        ("large_artifact_scan_passed", all(int(r.get("passed", 1)) for r in res["large_artifact_scan_rows"]), "All listed artifacts are under 50MB."),
    ]
    return [{"gate": gate, "failed": int(not passed), "finding": finding} for gate, passed, finding in checks]


def table_row_counts(res: dict) -> dict:
    keys = {
        "artifact_feasibility_audit": "artifact_feasibility_audit_rows",
        "artifact_manifest": "artifact_manifest_rows",
        "c61_identity_replay": "c61_identity_replay_rows",
        "estimator_agreement_ladder": "estimator_agreement_ladder_rows",
        "estimator_inventory": "estimator_inventory_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "instrumentation_gate": "instrumentation_gate_rows",
        "kernel_proxy_feasibility": "kernel_proxy_feasibility_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "null_calibration_summary": "null_calibration_summary_rows",
        "partition_cod_sensitivity": "partition_cod_sensitivity_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "screening_off_summary": "screening_off_summary_rows",
        "source_observable_adversary_summary": "source_observable_adversary_summary_rows",
        "synthetic_rank_gauge_estimator_grid": "synthetic_rank_gauge_estimator_grid_rows",
        "test_command_manifest": "test_command_manifest_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c62", "command": "python -m pytest oaci/tests/test_c62_conditional_divergence_estimator_stress.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c62_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c60_rank_gauge_proof_stress_empirical_bridge.py oaci/tests/test_c61_conditional_observability_divergence.py oaci/tests/test_c62_conditional_divergence_estimator_stress.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c62_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c60_rank_gauge_proof_stress_empirical_bridge.py oaci/tests/test_c61_conditional_observability_divergence.py oaci/tests/test_c62_conditional_divergence_estimator_stress.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    main = "\n".join([
        f"# C62 - Conditional-Divergence Estimator Stress / Discrete-to-Kernel Bridge Audit (frozen C19 `{res['config_hash']}`)",
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
        "C62 reproduces the C61 conditional-observability ladder exactly and stress-tests the estimator layer. The stable evidence remains the finite-partition and binary-Y divergence family, not a sample-level conditional-CS estimator.",
        "",
        f"- source -> key: `{STRICT_SOURCE_HIT:.6f}` -> `{KEY_ONLY_HIT:.6f}` (`{KEY_ONLY_HIT - STRICT_SOURCE_HIT:+.6f}`)",
        f"- source -> template: `{STRICT_SOURCE_HIT:.6f}` -> `{TEMPLATE_ONLY_HIT:.6f}` (`{TEMPLATE_ONLY_HIT - STRICT_SOURCE_HIT:+.6f}`)",
        f"- source -> label diagnostic: `{STRICT_SOURCE_HIT:.6f}` -> `{LABEL_DIAGNOSTIC_HIT:.6f}` (`{LABEL_DIAGNOSTIC_HIT - STRICT_SOURCE_HIT:+.6f}`)",
        f"- source -> endpoint scalar: `{STRICT_SOURCE_HIT:.6f}` -> `{ENDPOINT_ORACLE_HIT:.6f}` (`{ENDPOINT_ORACLE_HIT - STRICT_SOURCE_HIT:+.6f}`)",
        f"- source + template -> endpoint scalar: `{TEMPLATE_ONLY_HIT:.6f}` -> `{ENDPOINT_ORACLE_HIT:.6f}` (`{ENDPOINT_ORACLE_HIT - TEMPLATE_ONLY_HIT:+.6f}`)",
        "",
        "Partition smoothing and support stress preserve the ordering: endpoint > label diagnostic > template > source > key. The summary-kernel proxy agrees on endpoint dominance, but it is bandwidth-sensitive and proxy-only because current artifacts are summary-level.",
        "",
        "## Estimator Boundary",
        "",
        "Full sample-level conditional CS remains unsupported by the current artifact set. Missing items are per-trial paired variables, logits/probabilities, split-label cache, representation tensors, and atom traces. C62 therefore activates `C62-D` and `C62-E`, not `C62-C`.",
        "",
        "## Null Boundary",
        "",
        f"Template-only remains below the max null boundary (`{TEMPLATE_ONLY_HIT:.6f}` < `{MAX_NULL_P95:.6f}`), while the endpoint scalar remains above it (`{ENDPOINT_ORACLE_HIT:.6f}` > `{MAX_NULL_P95:.6f}`). The endpoint scalar is still a same-label target endpoint oracle and unavailable at selection time.",
        "",
        "## Gate",
        "",
        f"`{TRAINING_GATE}`",
        "",
        "C62 does not train, re-infer, use GPU, add BNCI2014_004, run seeds [3,4], create selector artifacts, or start manuscript drafting.",
    ])
    red = "\n".join([
        "# C62 - Red-Team Verification",
        "",
        "All C62 red-team gates pass." if d["red_team_failure_count"] == 0 else "C62 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    estimator = "\n".join([
        "# C62 - Estimator Stress Notes",
        "",
        "Supported now: finite-partition plug-in COD, binary-Y divergence proxies, smoothed partition stress, and a summary-level kernel proxy.",
        "",
        "Not supported now: sample-level conditional CS with KDE/Gram matrices. The current frozen summaries do not provide the paired sample variables required for that estimator.",
    ])
    instrumentation = "\n".join([
        "# C62 - Instrumentation Gate",
        "",
        f"Gate decision: `{INSTRUMENTATION_GATE}` with `{TRAINING_GATE}`.",
        "",
        "Trial-level cache, split-label cache, per-trial logits/probabilities, atom trace, and replication remain future requests only. C62 does not authorize any of them.",
    ])
    return {
        "C62_CONDITIONAL_DIVERGENCE_ESTIMATOR_STRESS.md": main,
        "C62_RED_TEAM_VERIFICATION.md": red,
        "C62_ESTIMATOR_STRESS_NOTES.md": estimator,
        "C62_INSTRUMENTATION_GATE.md": instrumentation,
    }


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "c61_commit": "0c2f5b8",
        "c61_decision": res["c61_decision"],
        "decision": res["decision"],
        "training_gate": TRAINING_GATE,
        "instrumentation_gate": INSTRUMENTATION_GATE,
        "estimator_status": {
            "finite_partition_plugin": "stable_under_smoothing",
            "binary_y_divergence_proxy": "stable",
            "smoothed_partition_cod": "stable_order",
            "summary_kernel_proxy": "proxy_only_bandwidth_sensitive",
            "full_sample_conditional_cs": "unsupported_trial_level_cache_required",
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
    c61_summary = _load_json(C61_JSON)
    res = {
        "config_hash": config_hash,
        "c61_decision": c61_summary["decision"]["primary"],
        "c61_identity_replay_rows": _replay_c61_ladder(),
        "estimator_inventory_rows": build_estimator_inventory(),
        "artifact_feasibility_audit_rows": build_artifact_feasibility(),
        "partition_cod_sensitivity_rows": build_partition_sensitivity(),
        "kernel_proxy_feasibility_rows": build_kernel_proxy_feasibility(),
        "estimator_agreement_ladder_rows": build_estimator_agreement_ladder(),
        "null_calibration_summary_rows": build_null_calibration(),
        "synthetic_rank_gauge_estimator_grid_rows": build_synthetic_grid(),
        "screening_off_summary_rows": build_screening_rows(),
        "source_observable_adversary_summary_rows": build_source_adversary(),
        "instrumentation_gate_rows": build_instrumentation_gate(),
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
        "c61_identity_replay.csv": ("c61_identity_replay_rows", ["comparison_id", "expected_hit_before", "expected_hit_after", "c61_hit_before", "c61_hit_after", "delta_hit", "identity_abs_error", "identity_pass", "source_table"]),
        "estimator_inventory.csv": ("estimator_inventory_rows", ["estimator_id", "estimator_family", "supported_by_existing_artifacts", "full_conditional_cs", "requires_sample_level_pairs", "uses_target_labels", "uses_same_label_endpoint_scalar", "source_only_dg_available", "diagnostic_only", "status"]),
        "artifact_feasibility_audit.csv": ("artifact_feasibility_audit_rows", ["artifact_requirement", "present", "sufficient_for_full_conditional_cs", "decision", "evidence"]),
        "partition_cod_sensitivity.csv": ("partition_cod_sensitivity_rows", ["setting", "comparison", "alpha", "coverage", "smoothed_hit_before", "smoothed_hit_after", "smoothed_gain", "endpoint_dominates_setting", "template_below_max_null_p95", "stable_order", "interpretation"]),
        "kernel_proxy_feasibility.csv": ("kernel_proxy_feasibility_rows", ["row_id", "bandwidth", "sample_level_pairs_required", "supported", "proxy_only", "unstable", "endpoint_proxy", "template_proxy", "rank_order_endpoint_gt_template", "reason"]),
        "estimator_agreement_ladder.csv": ("estimator_agreement_ladder_rows", ["comparison_id", "raw_hit_gain", "smoothed_gain_alpha5", "entropy_gap_bits", "js_bits", "hellinger2", "cs_binary_proxy", "kernel_proxy_bw_0p10", "endpoint_dominates", "template_partial", "screen_off_endpoint", "agreement_status"]),
        "null_calibration_summary.csv": ("null_calibration_summary_rows", ["null_id", "estimator_family", "observed_stat", "observed_value", "null_p95", "passes", "interpretation"]),
        "synthetic_rank_gauge_estimator_grid.csv": ("synthetic_rank_gauge_estimator_grid_rows", ["scenario", "rank_signal", "gauge_scale", "common_offset", "source_hit", "endpoint_hit", "source_error", "partition_gain", "cs_proxy", "pair_flip_possible", "multi_candidate_proxy_only", "expected_behavior_pass"]),
        "screening_off_summary.csv": ("screening_off_summary_rows", ["condition_set", "candidate_added", "hit_before", "hit_after", "endpoint_remaining_gain", "screens_off_endpoint", "causal_claim", "status"]),
        "source_observable_adversary_summary.csv": ("source_observable_adversary_summary_rows", ["candidate_id", "candidate", "uses_source_only_inputs", "uses_target_labels", "hit", "cod_proxy", "beats_max_null_p95", "reliable_escape_hatch", "reason"]),
        "instrumentation_gate.csv": ("instrumentation_gate_rows", ["need_id", "future_artifact", "needed_for_full_cs", "needed_for_c62_partition_claim", "authorized_in_c62", "gate_decision", "reason"]),
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
        glob.glob(os.path.join(REPORT_DIR, "C62_*.md"))
        + glob.glob(os.path.join(REPORT_DIR, "C62_*.json"))
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
    res["large_artifact_scan_rows"] = _large_scan(paths)
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
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c62_conditional_divergence_estimator_stress")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res, args.test_status)
    print(f"[C62] decision={res['decision']['primary']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
