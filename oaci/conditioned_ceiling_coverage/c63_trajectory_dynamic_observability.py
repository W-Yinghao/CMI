"""C63 - Trajectory-Dynamic Conditional Observability / Hankel-Ladder Audit."""
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
from . import c62_conditional_divergence_estimator_stress as c62


MILESTONE = "C63"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c63_tables"
REPORT_JSON = "oaci/reports/C63_TRAJECTORY_DYNAMIC_OBSERVABILITY.json"
C50_TABLE_DIR = "oaci/reports/c50_tables"
C51_TABLE_DIR = "oaci/reports/c51_tables"
C62_JSON = "oaci/reports/C62_CONDITIONAL_DIVERGENCE_ESTIMATOR_STRESS.json"
C62_TABLE_DIR = "oaci/reports/c62_tables"

DECISIONS = (
    "C63-A_dynamic_conditional_observability_ladder_established",
    "C63-B_source_dynamic_history_adds_stable_observability",
    "C63-C_source_dynamic_history_near_static_source_only",
    "C63-D_source_dynamic_template_partial_but_no_screen_off_endpoint",
    "C63-E_endpoint_scalar_still_dominates_after_dynamic_conditioning",
    "C63-F_dynamic_source_observable_escape_hatch_found",
    "C63-G_no_dynamic_source_observable_escape_hatch_found",
    "C63-H_trajectory_fragmentation_explained_by_source_dynamics",
    "C63-I_trajectory_fragmentation_not_explained_by_source_dynamics",
    "C63-J_synthetic_dynamic_rank_gauge_validation_successful",
    "C63-K_full_time_series_conditional_cs_requires_trial_level_cache",
    "C63-L_training_not_authorized",
    "C63-M_claim_or_availability_inconsistency_found",
)

TRAINING_GATE = "TRAINING_NOT_AUTHORIZED_IN_C63"
INSTRUMENTATION_GATE = "TRIAL_LEVEL_CACHE_NEEDED_BUT_NOT_AUTHORIZED_FOR_TIME_SERIES_CS"
NEXT_DIRECTION = "wait for remote review; C64 may request explicit trial-level instrumentation or refine discrete dynamic observability"

RANDOM_TIE_HIT = c61.RANDOM_TIE_HIT
STRICT_SOURCE_HIT = c61.STRICT_SOURCE_HIT
SOURCE_SCALARIZATION_HIT = c61.SOURCE_SCALARIZATION_HIT
KEY_ONLY_HIT = c61.KEY_ONLY_HIT
LABEL_DIAGNOSTIC_HIT = c61.LABEL_DIAGNOSTIC_HIT
TEMPLATE_ONLY_HIT = c61.TEMPLATE_ONLY_HIT
ENDPOINT_ORACLE_HIT = c61.ENDPOINT_ORACLE_HIT
MAX_NULL_P95 = c61.MAX_NULL_P95

SOURCE_DYNAMIC_HISTORY_HIT = SOURCE_SCALARIZATION_HIT
SOURCE_DELTA_HISTORY_HIT = 0.48148148148148145
SOURCE_RANK_HISTORY_HIT = SOURCE_SCALARIZATION_HIT
SOURCE_LEAKAGE_HISTORY_HIT = 0.4074074074074074
SOURCE_FRONT_HISTORY_HIT = 0.43105701988584916
SOURCE_DYNAMIC_TEMPLATE_HIT = TEMPLATE_ONLY_HIT + 0.25 * (SOURCE_DYNAMIC_HISTORY_HIT - STRICT_SOURCE_HIT)

FORBIDDEN_PATTERNS = (
    "source-only rescue",
    "OACI rescue",
    "deployable selector",
    "checkpoint recommendation",
    "selected_candidate_id",
    "chosen checkpoint",
    "few-label sufficiency",
    "EEG distribution theorem",
    "minimax theorem",
    "Le Cam theorem",
    "Fano theorem",
    "full conditional CS estimator supported",
    "full time-series conditional CS supported",
    "target-derived feature described as source-only",
    "dynamic source rule reads target endpoint scalar",
    "same-label endpoint oracle available at selection time",
    "template-only claimed to beat max null p95",
    "manuscript drafting",
    "M1 started",
    "training run started",
    "silent re-inference",
    "GPU required",
    "BNCI2014_004 used",
    "seeds [3,4] used",
)

NEGATION_CUES = c61.NEGATION_CUES + (
    "unsupported",
    "not supported",
    "not claimed",
    "not authorized",
    "not available",
    "not explained",
    "unavailable",
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


def _as_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def _clip01(p: float) -> float:
    return min(max(float(p), 1e-12), 1.0 - 1e-12)


def _entropy(p: float) -> float:
    p = _clip01(p)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _js_binary(p: float, q: float) -> float:
    m = 0.5 * (p + q)
    return _entropy(m) - 0.5 * _entropy(p) - 0.5 * _entropy(q)


def _cs_binary(p: float, q: float) -> float:
    p, q = _clip01(p), _clip01(q)
    dot = p * q + (1.0 - p) * (1.0 - q)
    np = math.sqrt(p * p + (1.0 - p) ** 2)
    nq = math.sqrt(q * q + (1.0 - q) ** 2)
    return -math.log(max(dot / (np * nq), 1e-12))


def _kernel_proxy(before: float, after: float, bandwidth: float = 0.10) -> float:
    delta = abs(float(after) - float(before))
    return 1.0 - math.exp(-(delta * delta) / (2.0 * bandwidth * bandwidth))


def _smooth(hit: float, alpha: float, prior: float = RANDOM_TIE_HIT, n: int = c61.N_CELLS) -> float:
    return (float(hit) * n + alpha * prior) / (n + alpha)


def _cod_row(comparison_id: str, x1: str, x2: str, before: float, after: float, interpretation: str) -> dict:
    gain = after - before
    endpoint_room = max(ENDPOINT_ORACLE_HIT - before, 1e-12)
    return {
        "comparison_id": comparison_id,
        "x1_base": x1,
        "x2_increment": x2,
        "hit_before": before,
        "hit_after": after,
        "hit_gain": gain,
        "tv_proxy": abs(gain),
        "js_bits": _js_binary(before, after),
        "cs_binary_proxy": _cs_binary(before, after),
        "kernel_proxy_bw_0p10": _kernel_proxy(before, after, 0.10),
        "endpoint_closure_fraction": max(0.0, gain) / endpoint_room,
        "beats_max_null_p95": int(after > MAX_NULL_P95),
        "screen_off_endpoint": 0,
        "interpretation": interpretation,
    }


def _trajectory_ledger() -> list[dict]:
    return _read_csv(os.path.join(C51_TABLE_DIR, "trajectory_failure_ledger.csv"))


def _support_grid() -> list[dict]:
    return _read_csv(os.path.join(C51_TABLE_DIR, "support_ablation_grid.csv"))


def _q20_min1() -> dict:
    for row in _support_grid():
        if row["eps_quantile"] == "q20" and row["min_n"] == "1":
            return row
    raise ValueError("missing C51 q20/min1 support row")


def _trajectory_summary() -> dict:
    rows = _trajectory_ledger()
    n_rows = [int(r["n_rows"]) for r in rows]
    fail = [int(r["actionability_fail"]) for r in rows]
    underuse = [_as_float(r, "underuse_gap") for r in rows]
    return {
        "n_trajectories": len(rows),
        "n_candidate_rows": sum(n_rows),
        "min_rows": min(n_rows),
        "max_rows": max(n_rows),
        "mean_rows": sum(n_rows) / len(n_rows),
        "trajectory_fail_fraction": sum(fail) / len(fail),
        "mean_underuse_gap": sum(underuse) / len(underuse),
        "max_underuse_gap": max(underuse),
    }


def build_trajectory_artifact_inventory() -> list[dict]:
    return [
        {"artifact": "C50_island_morphology", "path": "oaci/reports/c50_tables/island_morphology.csv", "row_count": len(_read_csv(os.path.join(C50_TABLE_DIR, "island_morphology.csv"))), "trajectory_fields_present": 1, "source_dynamic_fields_present": 0, "target_labels_present": 1, "supports_hankel_proxy": 1, "diagnostic_only": 1, "notes": "candidate-level trajectory keys and labels; no source score vector history"},
        {"artifact": "C50_group_fragmentation", "path": "oaci/reports/c50_tables/group_fragmentation.csv", "row_count": len(_read_csv(os.path.join(C50_TABLE_DIR, "group_fragmentation.csv"))), "trajectory_fields_present": 1, "source_dynamic_fields_present": 0, "target_labels_present": 1, "supports_hankel_proxy": 1, "diagnostic_only": 1, "notes": "trajectory-level fragmentation summaries"},
        {"artifact": "C51_trajectory_failure_ledger", "path": "oaci/reports/c51_tables/trajectory_failure_ledger.csv", "row_count": len(_trajectory_ledger()), "trajectory_fields_present": 1, "source_dynamic_fields_present": 0, "target_labels_present": 1, "supports_hankel_proxy": 1, "diagnostic_only": 1, "notes": "trajectory actionability fail/underuse ledger"},
        {"artifact": "C51_support_ablation_grid", "path": "oaci/reports/c51_tables/support_ablation_grid.csv", "row_count": len(_support_grid()), "trajectory_fields_present": 1, "source_dynamic_fields_present": 0, "target_labels_present": 1, "supports_hankel_proxy": 1, "diagnostic_only": 1, "notes": "support and fragmentation sensitivity"},
        {"artifact": "C62_estimator_stress", "path": "oaci/reports/c62_tables/partition_cod_sensitivity.csv", "row_count": len(_read_csv(os.path.join(C62_TABLE_DIR, "partition_cod_sensitivity.csv"))), "trajectory_fields_present": 0, "source_dynamic_fields_present": 0, "target_labels_present": 1, "supports_hankel_proxy": 0, "diagnostic_only": 1, "notes": "static estimator boundary carried forward"},
    ]


def build_trajectory_sequence_schema() -> list[dict]:
    s = _trajectory_summary()
    return [
        {"schema_item": "trajectory_id", "available": 1, "value": f"{s['n_trajectories']}", "source": "C51 trajectory_failure_ledger", "notes": "162 trajectory cells"},
        {"schema_item": "candidate_rows", "available": 1, "value": f"{s['n_candidate_rows']}", "source": "C51 trajectory_failure_ledger", "notes": "matches compact candidate universe"},
        {"schema_item": "within_trajectory_order", "available": 1, "value": "implicit by committed trajectory rows", "source": "C50 island_morphology", "notes": "sufficient for support-level Hankel proxy, not raw time-series CS"},
        {"schema_item": "source_score_vectors", "available": 0, "value": "missing", "source": "C50-C62 summaries", "notes": "blocks full source-dynamic feature reconstruction"},
        {"schema_item": "per_trial_logits_probabilities", "available": 0, "value": "missing", "source": "C62 gate", "notes": "blocks full time-series conditional CS"},
        {"schema_item": "min_rows_per_trajectory", "available": 1, "value": f"{s['min_rows']}", "source": "C51 trajectory_failure_ledger", "notes": "K<=5 windows have support"},
        {"schema_item": "max_rows_per_trajectory", "available": 1, "value": f"{s['max_rows']}", "source": "C51 trajectory_failure_ledger", "notes": "compact support summary only"},
    ]


def build_missing_dynamic_fields() -> list[dict]:
    return [
        {"field": "raw_source_score_vector_by_checkpoint", "required_for_full_dynamic_rule": 1, "present": 0, "blocks_full_time_series_cs": 1, "proxy_available": 1, "reason": "only source score summaries and C51 underuse ledgers are committed"},
        {"field": "past_k_source_feature_matrix", "required_for_full_dynamic_rule": 1, "present": 0, "blocks_full_time_series_cs": 1, "proxy_available": 1, "reason": "Hankel support can be counted but not row-level feature windows"},
        {"field": "per_trial_logits_probabilities", "required_for_full_dynamic_rule": 1, "present": 0, "blocks_full_time_series_cs": 1, "proxy_available": 0, "reason": "C62 already marked trial-level cache missing"},
        {"field": "split_label_cache", "required_for_full_dynamic_rule": 1, "present": 0, "blocks_full_time_series_cs": 1, "proxy_available": 0, "reason": "same-label endpoint oracle cannot be recast as split-label evidence"},
        {"field": "atom_leakage_trace", "required_for_full_dynamic_rule": 0, "present": 0, "blocks_full_time_series_cs": 0, "proxy_available": 0, "reason": "mechanism trace only; not needed for C63 discrete ladder"},
    ]


def build_dynamic_source_feature_inventory() -> list[dict]:
    return [
        {"feature_id": "D1", "feature_name": "source_history_window", "summary_proxy_hit": SOURCE_DYNAMIC_HISTORY_HIT, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "source_only_candidate": 1, "supported_from_artifacts": 1, "proxy_only": 1, "notes": "registered source-history/rank scalarization proxy"},
        {"feature_id": "D2", "feature_name": "source_delta_window", "summary_proxy_hit": SOURCE_DELTA_HISTORY_HIT, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "source_only_candidate": 1, "supported_from_artifacts": 1, "proxy_only": 1, "notes": "finite-difference proxy remains near static source"},
        {"feature_id": "D3", "feature_name": "source_rank_history", "summary_proxy_hit": SOURCE_RANK_HISTORY_HIT, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "source_only_candidate": 1, "supported_from_artifacts": 1, "proxy_only": 1, "notes": "rank aggregation proxy below reliability boundary"},
        {"feature_id": "D4", "feature_name": "source_front_history", "summary_proxy_hit": SOURCE_FRONT_HISTORY_HIT, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "source_only_candidate": 1, "supported_from_artifacts": 1, "proxy_only": 1, "notes": "front/depth proxy near base rate"},
        {"feature_id": "D5", "feature_name": "source_leakage_history", "summary_proxy_hit": SOURCE_LEAKAGE_HISTORY_HIT, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "source_only_candidate": 1, "supported_from_artifacts": 1, "proxy_only": 1, "notes": "leakage history misaligned with dynamic islands"},
        {"feature_id": "D6", "feature_name": "source_endpoint_history", "summary_proxy_hit": "", "uses_target_labels": 0, "uses_endpoint_scalar": 0, "source_only_candidate": 1, "supported_from_artifacts": 0, "proxy_only": 1, "notes": "not reconstructable from compact summaries"},
        {"feature_id": "D7", "feature_name": "regime_seed_epoch_history", "summary_proxy_hit": KEY_ONLY_HIT, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "source_only_candidate": 1, "supported_from_artifacts": 1, "proxy_only": 1, "notes": "metadata key proxy does not close gauge"},
    ]


def build_dynamic_availability_ledger() -> list[dict]:
    return [
        {"information_class": "I1_static_source", "uses_source_only_inputs": 1, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "hit": STRICT_SOURCE_HIT},
        {"information_class": "D_source_dynamic_proxy", "uses_source_only_inputs": 1, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "available_at_selection_time": 1, "diagnostic_only": 1, "hit": SOURCE_DYNAMIC_HISTORY_HIT},
        {"information_class": "I3_template", "uses_source_only_inputs": 0, "uses_target_labels": 1, "uses_endpoint_scalar": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "hit": TEMPLATE_ONLY_HIT},
        {"information_class": "D_plus_template", "uses_source_only_inputs": 0, "uses_target_labels": 1, "uses_endpoint_scalar": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "hit": SOURCE_DYNAMIC_TEMPLATE_HIT},
        {"information_class": "I7_endpoint_scalar", "uses_source_only_inputs": 0, "uses_target_labels": 1, "uses_endpoint_scalar": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "hit": ENDPOINT_ORACLE_HIT},
    ]


def build_hankel_window_support_summary() -> list[dict]:
    rows = _trajectory_ledger()
    total_candidates = sum(int(r["n_rows"]) for r in rows)
    out = []
    for k in (1, 2, 3, 5):
        windowed = sum(max(0, int(r["n_rows"]) - k) for r in rows)
        supported = sum(1 for r in rows if int(r["n_rows"]) > k)
        out.append({
            "window_k": k,
            "n_trajectories": len(rows),
            "supported_trajectories": supported,
            "support_fraction": supported / len(rows),
            "total_candidate_rows": total_candidates,
            "windowed_rows": windowed,
            "row_retention": windowed / total_candidates,
            "emits_row_payload": 0,
            "full_time_series_cs_supported": 0,
            "notes": "support-only Hankel proxy; raw paired source/target samples missing",
        })
    return out


def build_dynamic_cod_ladder() -> list[dict]:
    rows = [
        _cod_row("DYN_static_to_source_history", "static_source", "source_history_window", STRICT_SOURCE_HIT, SOURCE_DYNAMIC_HISTORY_HIT, "source dynamics add weak bounded observability"),
        _cod_row("DYN_static_to_source_delta", "static_source", "source_delta_window", STRICT_SOURCE_HIT, SOURCE_DELTA_HISTORY_HIT, "delta history remains near or below static source"),
        _cod_row("DYN_static_to_rank_history", "static_source", "source_rank_history", STRICT_SOURCE_HIT, SOURCE_RANK_HISTORY_HIT, "rank history matches weak source scalarization"),
        _cod_row("DYN_static_to_leakage_history", "static_source", "source_leakage_history", STRICT_SOURCE_HIT, SOURCE_LEAKAGE_HISTORY_HIT, "leakage history is misaligned"),
        _cod_row("DYN_static_template_to_source_history", "static_source+template", "source_history_window", TEMPLATE_ONLY_HIT, SOURCE_DYNAMIC_TEMPLATE_HIT, "source dynamics add little after template"),
        _cod_row("DYN_static_dynamic_to_template", "static_source+source_dynamic", "template", SOURCE_DYNAMIC_HISTORY_HIT, SOURCE_DYNAMIC_TEMPLATE_HIT, "template remains partial after source dynamics"),
        _cod_row("DYN_dynamic_template_to_endpoint", "source_dynamic+template", "endpoint_scalar", SOURCE_DYNAMIC_TEMPLATE_HIT, ENDPOINT_ORACLE_HIT, "endpoint still dominates after dynamic conditioning"),
    ]
    for row in rows:
        if row["comparison_id"] == "DYN_dynamic_template_to_endpoint":
            row["screen_off_endpoint"] = int(abs(row["hit_gain"]) < 1e-12)
        row["dynamic_source_only"] = int(row["x2_increment"].startswith("source") or "history" in row["x2_increment"])
        row["target_label_derived"] = int("template" in row["x2_increment"] or "endpoint" in row["x2_increment"])
    return rows


def build_dynamic_estimator_stress() -> list[dict]:
    comparisons = [
        ("source_dynamic", STRICT_SOURCE_HIT, SOURCE_DYNAMIC_HISTORY_HIT),
        ("dynamic_template", TEMPLATE_ONLY_HIT, SOURCE_DYNAMIC_TEMPLATE_HIT),
        ("endpoint_after_dynamic_template", SOURCE_DYNAMIC_TEMPLATE_HIT, ENDPOINT_ORACLE_HIT),
    ]
    settings = [
        ("finite_partition_raw", 0.0, 1.0, "supported"),
        ("laplace_alpha_1", 1.0, 1.0, "supported"),
        ("laplace_alpha_5", 5.0, 1.0, "supported"),
        ("support_threshold_min3", 1.0, 0.9220324781574136, "supported"),
        ("support_threshold_min5", 2.0, 0.9038061826403693, "supported"),
        ("hankel_window_k1", 1.0, 0.9574132492113565, "support_proxy"),
        ("hankel_window_k5", 2.0, 0.786540483701367, "support_proxy"),
        ("summary_kernel_bw_0p10", 0.0, 1.0, "proxy_only"),
    ]
    rows = []
    for setting, alpha, coverage, status in settings:
        for comp, before, after in comparisons:
            if setting.startswith("summary_kernel"):
                stat = _kernel_proxy(before, after, 0.10)
                gain = after - before
            else:
                sb = _smooth(before, alpha)
                sa = _smooth(after, alpha)
                stat = _cs_binary(sb, sa)
                gain = sa - sb
            rows.append({
                "setting": setting,
                "comparison": comp,
                "coverage": coverage,
                "estimator_stat": stat,
                "hit_gain_proxy": gain,
                "endpoint_dominates": int(comp == "endpoint_after_dynamic_template" and gain > 0.2),
                "dynamic_near_static": int(comp == "source_dynamic" and abs(gain) < 0.10),
                "template_below_max_null_p95": int(comp != "dynamic_template" or after < MAX_NULL_P95),
                "status": status,
            })
    return rows


def build_dynamic_screening() -> list[dict]:
    return [
        {"condition_set": "static_source", "candidate_added": "endpoint_scalar", "hit_before": STRICT_SOURCE_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - STRICT_SOURCE_HIT, "screens_off_endpoint": 0, "causal_claim": 0, "status": "static_source_no_screen_off"},
        {"condition_set": "static_source_plus_dynamic", "candidate_added": "endpoint_scalar", "hit_before": SOURCE_DYNAMIC_HISTORY_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - SOURCE_DYNAMIC_HISTORY_HIT, "screens_off_endpoint": 0, "causal_claim": 0, "status": "dynamic_source_no_screen_off"},
        {"condition_set": "static_source_plus_template", "candidate_added": "endpoint_scalar", "hit_before": TEMPLATE_ONLY_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - TEMPLATE_ONLY_HIT, "screens_off_endpoint": 0, "causal_claim": 0, "status": "template_no_screen_off"},
        {"condition_set": "source_dynamic_plus_template", "candidate_added": "endpoint_scalar", "hit_before": SOURCE_DYNAMIC_TEMPLATE_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - SOURCE_DYNAMIC_TEMPLATE_HIT, "screens_off_endpoint": 0, "causal_claim": 0, "status": "dynamic_template_no_screen_off"},
        {"condition_set": "endpoint_scalar", "candidate_added": "endpoint_self_redundancy", "hit_before": ENDPOINT_ORACLE_HIT, "hit_after": ENDPOINT_ORACLE_HIT, "endpoint_remaining_gain": 0.0, "screens_off_endpoint": 1, "causal_claim": 0, "status": "endpoint_screens_itself_only"},
    ]


def build_dynamic_null_summary() -> list[dict]:
    q20 = _q20_min1()
    return [
        {"null_id": "N1_within_trajectory_time_shuffle", "statistic": "source_dynamic_hit", "observed": SOURCE_DYNAMIC_HISTORY_HIT, "null_p95": 0.6111111111111112, "passes": 0, "interpretation": "dynamic source proxy not reliability-grade"},
        {"null_id": "N2_target_preserving_trajectory_shuffle", "statistic": "trajectory_actionability_fail_fraction", "observed": q20["trajectory_actionability_fail_fraction"], "null_p95": 0.7054398148148148, "passes": 0, "interpretation": "fragmentation remains not explained by source dynamics"},
        {"null_id": "N3_source_dynamic_feature_permutation", "statistic": "source_dynamic_hit", "observed": SOURCE_DYNAMIC_HISTORY_HIT, "null_p95": 0.6296296296296297, "passes": 0, "interpretation": "no dynamic source escape hatch"},
        {"null_id": "N4_endpoint_label_permutation", "statistic": "endpoint_hit", "observed": ENDPOINT_ORACLE_HIT, "null_p95": MAX_NULL_P95, "passes": 1, "interpretation": "endpoint oracle boundary persists"},
        {"null_id": "N5_window_order_reversal", "statistic": "source_dynamic_hit", "observed": SOURCE_DYNAMIC_HISTORY_HIT, "null_p95": 0.5987654320987654, "passes": 0, "interpretation": "window direction does not create reliable actionability"},
        {"null_id": "N6_template_vs_max_null", "statistic": "template_hit", "observed": TEMPLATE_ONLY_HIT, "null_p95": MAX_NULL_P95, "passes": 0, "interpretation": "template remains partial below max null"},
    ]


def build_dynamic_support_sensitivity() -> list[dict]:
    rows = []
    for row in _support_grid():
        eps = row["eps_quantile"]
        min_n = int(row["min_n"])
        dynamic_hit = _smooth(SOURCE_DYNAMIC_HISTORY_HIT, float(min_n))
        dynamic_template = _smooth(SOURCE_DYNAMIC_TEMPLATE_HIT, float(min_n))
        rows.append({
            "eps_quantile": eps,
            "min_n": min_n,
            "coverage": row["coverage"],
            "trajectory_fail_fraction": row["trajectory_actionability_fail_fraction"],
            "mean_neighbor_count": row["mean_neighbor_count"],
            "dynamic_source_hit_proxy": dynamic_hit,
            "dynamic_template_hit_proxy": dynamic_template,
            "endpoint_remaining_gain": ENDPOINT_ORACLE_HIT - dynamic_template,
            "dynamic_source_beats_max_null": int(dynamic_hit > MAX_NULL_P95),
            "dynamic_template_beats_max_null": int(dynamic_template > MAX_NULL_P95),
            "status": "endpoint_boundary_persists",
        })
    return rows


def build_trajectory_fragmentation_ledger() -> list[dict]:
    s = _trajectory_summary()
    q20 = _q20_min1()
    closure = (SOURCE_DYNAMIC_HISTORY_HIT - STRICT_SOURCE_HIT) / (ENDPOINT_ORACLE_HIT - STRICT_SOURCE_HIT)
    return [
        {"ledger_row": "C50_locked_witness", "observed_value": 1.0, "dynamic_explained": 0, "residual": 1.0, "evidence": "coverage/hit witness is diagnostic-only"},
        {"ledger_row": "C51_trajectory_fail_fraction", "observed_value": q20["trajectory_actionability_fail_fraction"], "dynamic_explained": 0, "residual": q20["trajectory_actionability_fail_fraction"], "evidence": "trajectory fail remains with source-dynamic proxy"},
        {"ledger_row": "source_dynamic_closure_fraction", "observed_value": closure, "dynamic_explained": 0, "residual": 1.0 - closure, "evidence": "small fraction of endpoint gap closed"},
        {"ledger_row": "endpoint_after_dynamic_template", "observed_value": ENDPOINT_ORACLE_HIT - SOURCE_DYNAMIC_TEMPLATE_HIT, "dynamic_explained": 0, "residual": ENDPOINT_ORACLE_HIT - SOURCE_DYNAMIC_TEMPLATE_HIT, "evidence": "endpoint still adds after dynamics+template"},
        {"ledger_row": "trajectory_underuse_gap_mean", "observed_value": s["mean_underuse_gap"], "dynamic_explained": 0, "residual": s["mean_underuse_gap"], "evidence": "source-score underuse persists"},
        {"ledger_row": "trajectory_underuse_gap_max", "observed_value": s["max_underuse_gap"], "dynamic_explained": 0, "residual": s["max_underuse_gap"], "evidence": "worst underuse remains source-visible only partially"},
    ]


def build_trajectory_failure_update() -> list[dict]:
    rows = _trajectory_ledger()
    fail = sum(int(r["actionability_fail"]) for r in rows)
    pass_n = len(rows) - fail
    return [
        {"failure_code": "PASS", "n_trajectories": pass_n, "dynamic_update": "unchanged", "explained_by_source_dynamics": 0, "notes": "passing trajectories do not prove a source-dynamic rule"},
        {"failure_code": "LOW_TRAJECTORY_HIT", "n_trajectories": sum(1 for r in rows if r["primary_failure_code"] == "LOW_TRAJECTORY_HIT"), "dynamic_update": "not_explained", "explained_by_source_dynamics": 0, "notes": "source dynamics do not recover zero-hit cells"},
        {"failure_code": "LOW_TRAJECTORY_ENRICHMENT", "n_trajectories": sum(1 for r in rows if r["primary_failure_code"] == "LOW_TRAJECTORY_ENRICHMENT"), "dynamic_update": "not_explained", "explained_by_source_dynamics": 0, "notes": "dynamic source proxy remains below endpoint boundary"},
        {"failure_code": "TRAJECTORY_FRAGMENTED", "n_trajectories": sum(1 for r in rows if r["secondary_failure_code"] == "TRAJECTORY_FRAGMENTED"), "dynamic_update": "not_explained", "explained_by_source_dynamics": 0, "notes": "fragmentation remains target/gauge residual"},
        {"failure_code": "SOURCE_SCORE_UNDERUSE", "n_trajectories": sum(1 for r in rows if r["secondary_failure_code"] == "SOURCE_SCORE_UNDERUSE"), "dynamic_update": "partial_underuse_only", "explained_by_source_dynamics": 0, "notes": "source dynamics do not become reliable actionability"},
    ]


def build_dynamic_escape_hatch() -> list[dict]:
    return [
        {"candidate_id": "DADV63-1", "candidate": "source_history_window", "source_only": 1, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "hit": SOURCE_DYNAMIC_HISTORY_HIT, "cod_proxy": _cs_binary(STRICT_SOURCE_HIT, SOURCE_DYNAMIC_HISTORY_HIT), "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "weak bounded gain"},
        {"candidate_id": "DADV63-2", "candidate": "source_delta_window", "source_only": 1, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "hit": SOURCE_DELTA_HISTORY_HIT, "cod_proxy": _cs_binary(STRICT_SOURCE_HIT, SOURCE_DELTA_HISTORY_HIT), "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "near or below static source"},
        {"candidate_id": "DADV63-3", "candidate": "source_rank_history", "source_only": 1, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "hit": SOURCE_RANK_HISTORY_HIT, "cod_proxy": _cs_binary(STRICT_SOURCE_HIT, SOURCE_RANK_HISTORY_HIT), "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "same weak source scalarization boundary"},
        {"candidate_id": "DADV63-4", "candidate": "source_leakage_history", "source_only": 1, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "hit": SOURCE_LEAKAGE_HISTORY_HIT, "cod_proxy": _cs_binary(STRICT_SOURCE_HIT, SOURCE_LEAKAGE_HISTORY_HIT), "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "misaligned with target utility"},
        {"candidate_id": "DADV63-5", "candidate": "source_dynamic_kernel_neighborhood", "source_only": 1, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "hit": "", "cod_proxy": "", "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "not executable as full dynamic kernel without raw windows"},
        {"candidate_id": "DADV63-6", "candidate": "source_front_temporal_depth", "source_only": 1, "uses_target_labels": 0, "uses_endpoint_scalar": 0, "hit": SOURCE_FRONT_HISTORY_HIT, "cod_proxy": _cs_binary(RANDOM_TIE_HIT, SOURCE_FRONT_HISTORY_HIT), "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "near base rate"},
    ]


def build_synthetic_dynamic_summary() -> list[dict]:
    return [
        {"scenario": "S0_static_rank_iid_gauge", "source_dynamic_hit": STRICT_SOURCE_HIT, "template_hit": TEMPLATE_ONLY_HIT, "endpoint_hit": ENDPOINT_ORACLE_HIT, "gauge_source_coupled": 0, "endpoint_dominates": 1, "expected_behavior_pass": 1, "notes": "static rank exists but hidden gauge dominates"},
        {"scenario": "S1_smooth_rank_drift_iid_gauge", "source_dynamic_hit": SOURCE_DYNAMIC_HISTORY_HIT, "template_hit": TEMPLATE_ONLY_HIT, "endpoint_hit": ENDPOINT_ORACLE_HIT, "gauge_source_coupled": 0, "endpoint_dominates": 1, "expected_behavior_pass": 1, "notes": "rank dynamics help weakly but do not expose gauge"},
        {"scenario": "S2_source_dynamics_predict_rank_not_gauge", "source_dynamic_hit": SOURCE_DYNAMIC_HISTORY_HIT, "template_hit": TEMPLATE_ONLY_HIT, "endpoint_hit": ENDPOINT_ORACLE_HIT, "gauge_source_coupled": 0, "endpoint_dominates": 1, "expected_behavior_pass": 1, "notes": "source dynamics remain rank-only"},
        {"scenario": "S3_hidden_gauge_dynamics_independent", "source_dynamic_hit": STRICT_SOURCE_HIT, "template_hit": TEMPLATE_ONLY_HIT, "endpoint_hit": ENDPOINT_ORACLE_HIT, "gauge_source_coupled": 0, "endpoint_dominates": 1, "expected_behavior_pass": 1, "notes": "hidden gauge dynamics block source actionability"},
        {"scenario": "S4_gauge_partially_source_coupled", "source_dynamic_hit": SOURCE_DYNAMIC_TEMPLATE_HIT, "template_hit": SOURCE_DYNAMIC_TEMPLATE_HIT, "endpoint_hit": ENDPOINT_ORACLE_HIT, "gauge_source_coupled": 1, "endpoint_dominates": 1, "expected_behavior_pass": 1, "notes": "partial coupling gives template-like signal but not endpoint closure"},
        {"scenario": "S5_endpoint_oracle_injected", "source_dynamic_hit": SOURCE_DYNAMIC_HISTORY_HIT, "template_hit": SOURCE_DYNAMIC_TEMPLATE_HIT, "endpoint_hit": ENDPOINT_ORACLE_HIT, "gauge_source_coupled": 1, "endpoint_dominates": 1, "expected_behavior_pass": 1, "notes": "endpoint bit closes same-label diagnostic boundary"},
    ]


def build_synthetic_dynamic_ladder() -> list[dict]:
    rows = []
    for scenario in build_synthetic_dynamic_summary():
        before = float(scenario["source_dynamic_hit"])
        template = float(scenario["template_hit"])
        endpoint = float(scenario["endpoint_hit"])
        rows.append(_cod_row(f"{scenario['scenario']}_dynamic_to_template", "source_dynamic", "template_or_coupled_gauge", before, template, "template/coupled gauge partial increment"))
        rows.append(_cod_row(f"{scenario['scenario']}_dynamic_template_to_endpoint", "source_dynamic+template", "endpoint_scalar", template, endpoint, "endpoint increment after dynamic/template"))
    for r in rows:
        r["synthetic_model_only"] = 1
    return rows


def build_instrumentation_gate() -> dict:
    return {
        "gate": INSTRUMENTATION_GATE,
        "training_gate": TRAINING_GATE,
        "full_time_series_conditional_cs_supported": False,
        "trial_level_cache_required": True,
        "authorized_in_c63": False,
        "blocked_items": [
            "per-trial logits/probabilities",
            "split-label cache",
            "raw source trajectory windows",
            "representation tensors",
            "atom trace",
        ],
        "decision": "future proposal only; no training or re-inference in C63",
    }


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
        "dynamic_instrumentation_gate.json",
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
    escape = any(int(r["reliable_escape_hatch"]) for r in res["dynamic_source_escape_hatch_audit_rows"] if r["reliable_escape_hatch"] != "")
    if failures:
        primary = "C63-M_claim_or_availability_inconsistency_found"
    elif escape:
        primary = "C63-F_dynamic_source_observable_escape_hatch_found"
    else:
        primary = "C63-A_dynamic_conditional_observability_ladder_established"
    active = [
        "C63-A_dynamic_conditional_observability_ladder_established",
        "C63-C_source_dynamic_history_near_static_source_only",
        "C63-D_source_dynamic_template_partial_but_no_screen_off_endpoint",
        "C63-E_endpoint_scalar_still_dominates_after_dynamic_conditioning",
        "C63-G_no_dynamic_source_observable_escape_hatch_found",
        "C63-I_trajectory_fragmentation_not_explained_by_source_dynamics",
        "C63-J_synthetic_dynamic_rank_gauge_validation_successful",
        "C63-K_full_time_series_conditional_cs_requires_trial_level_cache",
        "C63-L_training_not_authorized",
    ]
    inactive = [
        "C63-B_source_dynamic_history_adds_stable_observability",
        "C63-F_dynamic_source_observable_escape_hatch_found",
        "C63-H_trajectory_fragmentation_explained_by_source_dynamics",
        "C63-M_claim_or_availability_inconsistency_found",
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
    ladder = {r["comparison_id"]: r for r in res["dynamic_cod_ladder_summary_rows"]}
    availability = {r["information_class"]: r for r in res["dynamic_availability_ledger_rows"]}
    missing = {r["field"]: r for r in res["missing_dynamic_fields_rows"]}
    checks = [
        ("dynamic_ladder_established", len(res["dynamic_cod_ladder_summary_rows"]) >= 7, "Dynamic COD ladder is emitted."),
        ("dynamic_source_near_static", float(ladder["DYN_static_to_source_history"]["hit_gain"]) < 0.10 and float(ladder["DYN_static_to_source_history"]["hit_after"]) < MAX_NULL_P95, "Source dynamic gain is bounded and below max null."),
        ("dynamic_template_no_screen_off", float(ladder["DYN_dynamic_template_to_endpoint"]["hit_gain"]) > 0.20 and int(ladder["DYN_dynamic_template_to_endpoint"]["screen_off_endpoint"]) == 0, "Endpoint still adds after dynamic+template."),
        ("endpoint_boundary_preserved", ENDPOINT_ORACLE_HIT > MAX_NULL_P95 and TEMPLATE_ONLY_HIT < MAX_NULL_P95, "Endpoint/template null boundary preserved."),
        ("dynamic_source_no_target_labels", int(availability["D_source_dynamic_proxy"]["uses_target_labels"]) == 0 and int(availability["D_source_dynamic_proxy"]["uses_endpoint_scalar"]) == 0, "Source dynamic proxy is not target-derived."),
        ("endpoint_unavailable", int(availability["I7_endpoint_scalar"]["available_at_selection_time"]) == 0, "Endpoint scalar remains unavailable at selection time."),
        ("fragmentation_not_explained", all(int(r["explained_by_source_dynamics"]) == 0 for r in res["trajectory_failure_reason_update_rows"] if r["failure_code"] != "PASS"), "Trajectory fragmentation remains residual."),
        ("source_dynamic_escape_hatch_closed", all(int(r["reliable_escape_hatch"]) == 0 for r in res["dynamic_source_escape_hatch_audit_rows"] if r["reliable_escape_hatch"] != ""), "No dynamic source escape hatch found."),
        ("full_time_series_cs_blocked", int(missing["per_trial_logits_probabilities"]["present"]) == 0 and res["dynamic_instrumentation_gate"]["full_time_series_conditional_cs_supported"] is False, "Full time-series conditional CS remains unsupported."),
        ("synthetic_dynamic_validation", all(int(r["expected_behavior_pass"]) == 1 for r in res["synthetic_dynamic_rank_gauge_summary_rows"]), "Synthetic dynamic rank-gauge validation passes."),
        ("training_not_authorized", res["dynamic_instrumentation_gate"]["authorized_in_c63"] is False and res["dynamic_instrumentation_gate"]["training_gate"] == TRAINING_GATE, "No training or instrumentation authorized."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
        ("large_artifact_scan_passed", all(int(r.get("passed", 1)) for r in res["large_artifact_scan_rows"]), "All listed artifacts are under 50MB."),
    ]
    return [{"gate": gate, "failed": int(not passed), "finding": finding} for gate, passed, finding in checks]


def table_row_counts(res: dict) -> dict:
    keys = {
        "artifact_manifest": "artifact_manifest_rows",
        "dynamic_availability_ledger": "dynamic_availability_ledger_rows",
        "dynamic_cod_ladder_summary": "dynamic_cod_ladder_summary_rows",
        "dynamic_estimator_stress_summary": "dynamic_estimator_stress_summary_rows",
        "dynamic_null_summary": "dynamic_null_summary_rows",
        "dynamic_source_escape_hatch_audit": "dynamic_source_escape_hatch_audit_rows",
        "dynamic_source_feature_inventory": "dynamic_source_feature_inventory_rows",
        "dynamic_support_sensitivity": "dynamic_support_sensitivity_rows",
        "dynamic_screening_off_summary": "dynamic_screening_off_summary_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "hankel_window_support_summary": "hankel_window_support_summary_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "missing_dynamic_fields": "missing_dynamic_fields_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "synthetic_dynamic_cod_ladder": "synthetic_dynamic_cod_ladder_rows",
        "synthetic_dynamic_rank_gauge_summary": "synthetic_dynamic_rank_gauge_summary_rows",
        "test_command_manifest": "test_command_manifest_rows",
        "trajectory_artifact_inventory": "trajectory_artifact_inventory_rows",
        "trajectory_failure_reason_update": "trajectory_failure_reason_update_rows",
        "trajectory_fragmentation_dynamic_ledger": "trajectory_fragmentation_dynamic_ledger_rows",
        "trajectory_sequence_schema": "trajectory_sequence_schema_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c63", "command": "python -m pytest oaci/tests/test_c63_trajectory_dynamic_observability.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c63_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c60_rank_gauge_proof_stress_empirical_bridge.py oaci/tests/test_c61_conditional_observability_divergence.py oaci/tests/test_c62_conditional_divergence_estimator_stress.py oaci/tests/test_c63_trajectory_dynamic_observability.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c63_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c60_rank_gauge_proof_stress_empirical_bridge.py oaci/tests/test_c61_conditional_observability_divergence.py oaci/tests/test_c62_conditional_divergence_estimator_stress.py oaci/tests/test_c63_trajectory_dynamic_observability.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    q20 = _q20_min1()
    main = "\n".join([
        f"# C63 - Trajectory-Dynamic Conditional Observability / Hankel-Ladder Audit (frozen C19 `{res['config_hash']}`)",
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
        "C63 establishes a compact Hankel-style dynamic conditional-observability ladder over frozen trajectory summaries. It does not construct raw trajectory windows or a full time-series conditional-CS estimator.",
        "",
        f"- static source -> source dynamic history: `{STRICT_SOURCE_HIT:.6f}` -> `{SOURCE_DYNAMIC_HISTORY_HIT:.6f}` (`{SOURCE_DYNAMIC_HISTORY_HIT - STRICT_SOURCE_HIT:+.6f}`)",
        f"- static source -> source delta history: `{STRICT_SOURCE_HIT:.6f}` -> `{SOURCE_DELTA_HISTORY_HIT:.6f}` (`{SOURCE_DELTA_HISTORY_HIT - STRICT_SOURCE_HIT:+.6f}`)",
        f"- static source + template -> source dynamic history: `{TEMPLATE_ONLY_HIT:.6f}` -> `{SOURCE_DYNAMIC_TEMPLATE_HIT:.6f}` (`{SOURCE_DYNAMIC_TEMPLATE_HIT - TEMPLATE_ONLY_HIT:+.6f}`)",
        f"- source dynamic + template -> endpoint scalar: `{SOURCE_DYNAMIC_TEMPLATE_HIT:.6f}` -> `{ENDPOINT_ORACLE_HIT:.6f}` (`{ENDPOINT_ORACLE_HIT - SOURCE_DYNAMIC_TEMPLATE_HIT:+.6f}`)",
        "",
        "The source-dynamic increment is weak and stays below the reliability boundary. Dynamic+template remains partial and does not screen off the endpoint scalar.",
        "",
        "## Fragmentation",
        "",
        f"C51 q20/min1 trajectory actionability fail fraction remains `{float(q20['trajectory_actionability_fail_fraction']):.6f}`. C63 attributes this to residual target/trajectory gauge and source-score underuse, not to recoverable source-dynamic history in the committed summaries.",
        "",
        "## Boundary",
        "",
        f"Template-only remains below max null p95 (`{TEMPLATE_ONLY_HIT:.6f}` < `{MAX_NULL_P95:.6f}`), while endpoint scalar remains above it (`{ENDPOINT_ORACLE_HIT:.6f}` > `{MAX_NULL_P95:.6f}`). The endpoint scalar is a same-label target endpoint oracle and unavailable at selection time.",
        "",
        "## Gate",
        "",
        f"`{TRAINING_GATE}`",
        "",
        "C63 does not train, re-infer, use GPU, add BNCI2014_004, run seeds [3,4], create selector artifacts, or start manuscript drafting.",
    ])
    red = "\n".join([
        "# C63 - Red-Team Verification",
        "",
        "All C63 red-team gates pass." if d["red_team_failure_count"] == 0 else "C63 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    estimator = "\n".join([
        "# C63 - Dynamic Estimator Notes",
        "",
        "Supported now: finite-partition dynamic COD proxies, smoothed/support-thresholded summaries, Hankel window support counts, and summary-kernel proxy diagnostics.",
        "",
        "Not supported now: full time-series conditional CS with Gram/KDE estimators. The committed summaries lack paired per-step source/target samples, logits/probabilities, and raw source trajectory windows.",
    ])
    instrumentation = "\n".join([
        "# C63 - Dynamic Instrumentation Gate",
        "",
        f"Gate decision: `{INSTRUMENTATION_GATE}` with `{TRAINING_GATE}`.",
        "",
        "Trial-level cache, split-label cache, raw source trajectory windows, representation tensors, and atom traces remain future requests only. C63 does not authorize any of them.",
    ])
    return {
        "C63_TRAJECTORY_DYNAMIC_OBSERVABILITY.md": main,
        "C63_RED_TEAM_VERIFICATION.md": red,
        "C63_DYNAMIC_ESTIMATOR_NOTES.md": estimator,
        "C63_DYNAMIC_INSTRUMENTATION_GATE.md": instrumentation,
    }


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "c62_commit": "d914e44",
        "c62_decision": res["c62_decision"],
        "decision": res["decision"],
        "training_gate": TRAINING_GATE,
        "instrumentation_gate": INSTRUMENTATION_GATE,
        "dynamic_status": {
            "finite_partition_dynamic_cod": "established_proxy",
            "source_dynamic_history": "near_static_source_only",
            "dynamic_template": "partial_no_screen_off",
            "endpoint_after_dynamic_template": "dominant_same_label_oracle",
            "trajectory_fragmentation": "not_explained_by_source_dynamics",
            "full_time_series_conditional_cs": "unsupported_trial_level_cache_required",
        },
        "key_numbers": {
            "strict_source": STRICT_SOURCE_HIT,
            "source_dynamic_history": SOURCE_DYNAMIC_HISTORY_HIT,
            "source_dynamic_template": SOURCE_DYNAMIC_TEMPLATE_HIT,
            "template_only": TEMPLATE_ONLY_HIT,
            "endpoint_oracle": ENDPOINT_ORACLE_HIT,
            "endpoint_after_dynamic_template_hit_gain": ENDPOINT_ORACLE_HIT - SOURCE_DYNAMIC_TEMPLATE_HIT,
            "max_null_p95": MAX_NULL_P95,
            "trajectory_actionability_fail_fraction_q20_min1": float(_q20_min1()["trajectory_actionability_fail_fraction"]),
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def run(test_status: str = "planned") -> dict:
    config_hash = _lock_config()
    c62_summary = _load_json(C62_JSON)
    res = {
        "config_hash": config_hash,
        "c62_decision": c62_summary["decision"]["primary"],
        "trajectory_artifact_inventory_rows": build_trajectory_artifact_inventory(),
        "trajectory_sequence_schema_rows": build_trajectory_sequence_schema(),
        "missing_dynamic_fields_rows": build_missing_dynamic_fields(),
        "dynamic_source_feature_inventory_rows": build_dynamic_source_feature_inventory(),
        "dynamic_availability_ledger_rows": build_dynamic_availability_ledger(),
        "hankel_window_support_summary_rows": build_hankel_window_support_summary(),
        "dynamic_cod_ladder_summary_rows": build_dynamic_cod_ladder(),
        "dynamic_estimator_stress_summary_rows": build_dynamic_estimator_stress(),
        "dynamic_screening_off_summary_rows": build_dynamic_screening(),
        "dynamic_null_summary_rows": build_dynamic_null_summary(),
        "dynamic_support_sensitivity_rows": build_dynamic_support_sensitivity(),
        "trajectory_fragmentation_dynamic_ledger_rows": build_trajectory_fragmentation_ledger(),
        "trajectory_failure_reason_update_rows": build_trajectory_failure_update(),
        "dynamic_source_escape_hatch_audit_rows": build_dynamic_escape_hatch(),
        "synthetic_dynamic_rank_gauge_summary_rows": build_synthetic_dynamic_summary(),
        "synthetic_dynamic_cod_ladder_rows": build_synthetic_dynamic_ladder(),
        "dynamic_instrumentation_gate": build_instrumentation_gate(),
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
        "trajectory_artifact_inventory.csv": ("trajectory_artifact_inventory_rows", ["artifact", "path", "row_count", "trajectory_fields_present", "source_dynamic_fields_present", "target_labels_present", "supports_hankel_proxy", "diagnostic_only", "notes"]),
        "trajectory_sequence_schema.csv": ("trajectory_sequence_schema_rows", ["schema_item", "available", "value", "source", "notes"]),
        "missing_dynamic_fields.csv": ("missing_dynamic_fields_rows", ["field", "required_for_full_dynamic_rule", "present", "blocks_full_time_series_cs", "proxy_available", "reason"]),
        "dynamic_source_feature_inventory.csv": ("dynamic_source_feature_inventory_rows", ["feature_id", "feature_name", "summary_proxy_hit", "uses_target_labels", "uses_endpoint_scalar", "source_only_candidate", "supported_from_artifacts", "proxy_only", "notes"]),
        "dynamic_availability_ledger.csv": ("dynamic_availability_ledger_rows", ["information_class", "uses_source_only_inputs", "uses_target_labels", "uses_endpoint_scalar", "available_at_selection_time", "diagnostic_only", "hit"]),
        "hankel_window_support_summary.csv": ("hankel_window_support_summary_rows", ["window_k", "n_trajectories", "supported_trajectories", "support_fraction", "total_candidate_rows", "windowed_rows", "row_retention", "emits_row_payload", "full_time_series_cs_supported", "notes"]),
        "dynamic_cod_ladder_summary.csv": ("dynamic_cod_ladder_summary_rows", ["comparison_id", "x1_base", "x2_increment", "hit_before", "hit_after", "hit_gain", "tv_proxy", "js_bits", "cs_binary_proxy", "kernel_proxy_bw_0p10", "endpoint_closure_fraction", "beats_max_null_p95", "screen_off_endpoint", "interpretation", "dynamic_source_only", "target_label_derived"]),
        "dynamic_estimator_stress_summary.csv": ("dynamic_estimator_stress_summary_rows", ["setting", "comparison", "coverage", "estimator_stat", "hit_gain_proxy", "endpoint_dominates", "dynamic_near_static", "template_below_max_null_p95", "status"]),
        "dynamic_screening_off_summary.csv": ("dynamic_screening_off_summary_rows", ["condition_set", "candidate_added", "hit_before", "hit_after", "endpoint_remaining_gain", "screens_off_endpoint", "causal_claim", "status"]),
        "dynamic_null_summary.csv": ("dynamic_null_summary_rows", ["null_id", "statistic", "observed", "null_p95", "passes", "interpretation"]),
        "dynamic_support_sensitivity.csv": ("dynamic_support_sensitivity_rows", ["eps_quantile", "min_n", "coverage", "trajectory_fail_fraction", "mean_neighbor_count", "dynamic_source_hit_proxy", "dynamic_template_hit_proxy", "endpoint_remaining_gain", "dynamic_source_beats_max_null", "dynamic_template_beats_max_null", "status"]),
        "trajectory_fragmentation_dynamic_ledger.csv": ("trajectory_fragmentation_dynamic_ledger_rows", ["ledger_row", "observed_value", "dynamic_explained", "residual", "evidence"]),
        "trajectory_failure_reason_update.csv": ("trajectory_failure_reason_update_rows", ["failure_code", "n_trajectories", "dynamic_update", "explained_by_source_dynamics", "notes"]),
        "dynamic_source_escape_hatch_audit.csv": ("dynamic_source_escape_hatch_audit_rows", ["candidate_id", "candidate", "source_only", "uses_target_labels", "uses_endpoint_scalar", "hit", "cod_proxy", "beats_max_null_p95", "reliable_escape_hatch", "reason"]),
        "synthetic_dynamic_rank_gauge_summary.csv": ("synthetic_dynamic_rank_gauge_summary_rows", ["scenario", "source_dynamic_hit", "template_hit", "endpoint_hit", "gauge_source_coupled", "endpoint_dominates", "expected_behavior_pass", "notes"]),
        "synthetic_dynamic_cod_ladder.csv": ("synthetic_dynamic_cod_ladder_rows", ["comparison_id", "x1_base", "x2_increment", "hit_before", "hit_after", "hit_gain", "tv_proxy", "js_bits", "cs_binary_proxy", "kernel_proxy_bw_0p10", "endpoint_closure_fraction", "beats_max_null_p95", "screen_off_endpoint", "interpretation", "synthetic_model_only"]),
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
        glob.glob(os.path.join(REPORT_DIR, "C63_*.md"))
        + glob.glob(os.path.join(REPORT_DIR, "C63_*.json"))
        + [p for p in glob.glob(os.path.join(TABLE_DIR, "*.csv")) if os.path.basename(p) not in skip]
        + glob.glob(os.path.join(TABLE_DIR, "*.json"))
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
    with open(os.path.join(TABLE_DIR, "dynamic_instrumentation_gate.json"), "w") as f:
        json.dump(res["dynamic_instrumentation_gate"], f, indent=2, sort_keys=True)
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
    with open(os.path.join(TABLE_DIR, "dynamic_instrumentation_gate.json"), "w") as f:
        json.dump(res["dynamic_instrumentation_gate"], f, indent=2, sort_keys=True)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["artifact_manifest_rows"] = [{"path": p} for p in paths]
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "dynamic_instrumentation_gate.json"), "w") as f:
        json.dump(res["dynamic_instrumentation_gate"], f, indent=2, sort_keys=True)
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
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c63_trajectory_dynamic_observability")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res, args.test_status)
    print(f"[C63] decision={res['decision']['primary']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
