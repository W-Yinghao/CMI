"""Post-C81 frozen-selection baseline recovery with atomic result publication.

C82P commands operate only on committed metadata and injected synthetic fixtures.
The real adapter is fail-closed behind the C82 execution lock and a fresh direct
PI authorization record. Selection is replayed by identity and is never rebuilt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c82p_tables"
PROTOCOL_PATH = REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY_PROTOCOL.sha256"
LOCK_PATH = REPORT_DIR / "C82_ANALYSIS_EXECUTION_LOCK.json"
LOCK_SHA_PATH = REPORT_DIR / "C82_ANALYSIS_EXECUTION_LOCK.sha256"
AUTHORIZATION_PATH = REPORT_DIR / "C82E_PI_AUTHORIZATION_RECORD.json"
METHOD_REGISTRY_PATH = REPORT_DIR / "C81_BASELINE_METHOD_REGISTRY.json"
METRIC_APPLICABILITY_PATH = TABLE_DIR / "method_metric_applicability.csv"
OUTPUT_REGISTRY_PATH = TABLE_DIR / "output_table_registry.csv"

PRIMARY_TARGETS = (1, 2, 3, 5, 6, 7, 8, 9)
SEEDS = (3, 4)
LEVELS = (0, 1)
CANDIDATES = 81
MATERIAL_MARGIN = 0.05
NONINFERIORITY_MARGIN = 0.05
LOTO_MINIMUM = 12
SELECTION_METHODS = (
    "B1", "B2", "B3", "B4O", "B4S", "S1", "S2",
    "U1", "U2", "U3", "U4", "U5", "U6", "U7", "U11", "U12",
    "U13", "U14", "U15",
)
CONTEXT_METHODS = SELECTION_METHODS + ("B0", "B5")
PRIMARY_ZERO_METHODS = ("U7", "U5", "U11", "U13", "U14", "U15")
RANK_METHODS = (
    "S1", "S2", "U1", "U2", "U3", "U4", "U5", "U6", "U7",
    "U11", "U12", "U13", "U14", "U15",
)
PERFORMANCE_ESTIMATE_METHODS = ("S1", "U6", "U7", "U13", "U15")
INFORMATION_CLASS = {
    "B0": "I0", "B1": "I0", "B2": "I0", "B3": "I0", "B4O": "I0", "B4S": "I0",
    "S1": "IS", "S2": "IS",
    "U1": "IU", "U2": "IU", "U3": "IU", "U4": "IU", "U5": "IU",
    "U6": "ISU", "U7": "ISU", "U11": "IU", "U12": "ISU",
    "U13": "ISU", "U14": "IU", "U15": "ISU",
    "L1": "ILc", "L7": "ILc", "B5": "IOr",
}

METHOD_CONTEXT_FIELDS = (
    "seed",
    "target",
    "level",
    "method_id",
    "standardized_regret",
    "selected_utility",
    "top1",
    "top5",
    "top10",
    "coverage_top1",
    "coverage_top5",
    "coverage_top10",
    "selected_regime",
    "evaluation_label_access_after_selection_freeze",
    "same_label_oracle_accessed",
    "target4_primary",
)
VALID_REGIMES = {"ERM", "OACI", "SRC", "RANDOM", "ORACLE"}

GATE_E = "C82-E_post_C81_recovery_protocol_implementation_or_provenance_blocker"
GATE_D = "C82-D_zero_label_comparison_training_seed_method_identity_or_target_heterogeneous"
GATE_A = "C82-A_same_zero_label_selector_matches_one_label_frontier_across_seeds"
GATE_B = "C82-B_same_zero_label_selector_improves_source_but_not_one_label_frontier"
GATE_C = "C82-C_no_registered_zero_label_selector_materially_improves_source"


class C82ValidationError(RuntimeError):
    """Raised before publication when a locked C82 contract does not replay."""


class C82PostEvaluationFailure(RuntimeError):
    """A terminal failure after evaluation access; the authorization is consumed."""

    authorization_consumed = True
    final_gate = GATE_E


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def last_commit(path: str | Path) -> str:
    relative = Path(path).resolve().relative_to(REPO_ROOT.resolve())
    value = _git("log", "-1", "--format=%H", "--", str(relative))
    if len(value) != 40:
        raise C82ValidationError(f"cannot resolve committed identity for {relative}")
    return value


def load_protocol() -> tuple[dict[str, Any], str]:
    expected = PROTOCOL_SHA_PATH.read_text().strip()
    observed = sha256_file(PROTOCOL_PATH)
    if observed != expected:
        raise C82ValidationError("C82 protocol hash mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["frozen_selection"]["selection_recomputation_allowed"]:
        raise C82ValidationError("C82 selection recomputation scope drift")
    if protocol["field_and_view_binding"]["target4_primary"]:
        raise C82ValidationError("C82 target4 scope drift")
    if protocol["field_and_view_binding"]["same_label_oracle_reachable"]:
        raise C82ValidationError("C82 oracle scope drift")
    if tuple(protocol["canonical_method_context_schema"]["field_order"]) != METHOD_CONTEXT_FIELDS:
        raise C82ValidationError("C82 canonical field order drift")
    if protocol["LORO"]["decision"] != "REMOVED_FROM_OPERATIVE_C82_INFERENCE":
        raise C82ValidationError("C82 LORO decision drift")
    return protocol, observed


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise C82ValidationError(f"boolean is not numeric for {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise C82ValidationError(f"invalid numeric value for {field}") from exc
    if not math.isfinite(result):
        raise C82ValidationError(f"non-finite value for {field}")
    return result


def canonicalize_method_context_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact field set and return protocol order, ignoring insertion order."""
    keys = set(row)
    required = set(METHOD_CONTEXT_FIELDS)
    if keys != required:
        missing = sorted(required - keys)
        unknown = sorted(keys - required)
        raise C82ValidationError(f"method-context schema mismatch missing={missing} unknown={unknown}")
    if not _is_int(row["seed"]) or int(row["seed"]) not in SEEDS:
        raise C82ValidationError("invalid seed")
    if not _is_int(row["target"]) or int(row["target"]) not in PRIMARY_TARGETS:
        raise C82ValidationError("invalid primary target")
    if not _is_int(row["level"]) or int(row["level"]) not in LEVELS:
        raise C82ValidationError("invalid level")
    method = str(row["method_id"])
    if method not in CONTEXT_METHODS:
        raise C82ValidationError(f"unregistered C82 context method: {method}")
    numeric: dict[str, float] = {}
    for field in (
        "standardized_regret", "selected_utility", "top1", "top5", "top10",
        "coverage_top1", "coverage_top5", "coverage_top10",
    ):
        numeric[field] = _finite_float(row[field], field)
    if not 0.0 <= numeric["standardized_regret"] <= 1.0:
        raise C82ValidationError("standardized regret outside [0,1]")
    for field in ("top1", "top5", "top10", "coverage_top1", "coverage_top5", "coverage_top10"):
        if not 0.0 <= numeric[field] <= 1.0:
            raise C82ValidationError(f"{field} outside [0,1]")
    regime = str(row["selected_regime"])
    if regime not in VALID_REGIMES:
        raise C82ValidationError(f"invalid selected regime: {regime}")
    for field, expected in (
        ("evaluation_label_access_after_selection_freeze", True),
        ("same_label_oracle_accessed", False),
        ("target4_primary", False),
    ):
        if not isinstance(row[field], (bool, np.bool_)) or bool(row[field]) is not expected:
            raise C82ValidationError(f"protected boolean drift: {field}")
    normalized = {
        "seed": int(row["seed"]),
        "target": int(row["target"]),
        "level": int(row["level"]),
        "method_id": method,
        **numeric,
        "selected_regime": regime,
        "evaluation_label_access_after_selection_freeze": True,
        "same_label_oracle_accessed": False,
        "target4_primary": False,
    }
    return {field: normalized[field] for field in METHOD_CONTEXT_FIELDS}


def validate_method_context_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [canonicalize_method_context_row(row) for row in rows]
    if len(normalized) != 672:
        raise C82ValidationError(f"expected 672 method-context rows, observed {len(normalized)}")
    observed = {(row["seed"], row["target"], row["level"], row["method_id"]) for row in normalized}
    expected = {
        (seed, target, level, method)
        for seed in SEEDS for target in PRIMARY_TARGETS for level in LEVELS for method in CONTEXT_METHODS
    }
    if observed != expected or len(observed) != len(normalized):
        raise C82ValidationError("C82 method-context coverage or uniqueness drift")
    return normalized


def _studentized_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    result = np.divide(mean, scale, out=np.zeros(values.shape[1]), where=scale > 1e-15)
    degenerate = scale <= 1e-15
    result[degenerate & (mean > 0.0)] = np.inf
    result[degenerate & (mean < 0.0)] = -np.inf
    return result


def exact_signflip_maxT(effects: np.ndarray, margin: float = 0.0) -> dict[str, np.ndarray]:
    effects = np.asarray(effects, dtype=float)
    if effects.ndim != 2 or not 2 <= effects.shape[0] <= len(PRIMARY_TARGETS):
        raise C82ValidationError("C82 maxT effect matrix shape drift")
    shifted = effects - margin
    observed = _studentized_mean(shifted)
    centered = effects - np.mean(effects, axis=0, keepdims=True)
    null_statistics: list[float] = []
    null_mean_max: list[float] = []
    for signs in itertools.product((-1.0, 1.0), repeat=effects.shape[0]):
        sign = np.asarray(signs)[:, None]
        null_statistics.append(float(np.max(_studentized_mean(shifted * sign))))
        null_mean_max.append(float(np.max(np.mean(centered * sign, axis=0))))
    null_statistics_array = np.asarray(null_statistics)
    pvalue = np.asarray([
        (1.0 + np.sum(null_statistics_array >= value - 1e-15)) / (len(null_statistics_array) + 1.0)
        for value in observed
    ])
    critical = float(np.quantile(np.asarray(null_mean_max), 0.95, method="higher"))
    mean = np.mean(effects, axis=0)
    return {"pvalue": pvalue, "mean": mean, "lower": mean - critical, "upper": mean + critical}


def _passes_q1(effects: np.ndarray, pvalue: float) -> bool:
    return bool(
        float(np.mean(effects)) >= MATERIAL_MARGIN
        and pvalue <= 0.05
        and int(np.sum(effects > 0.0)) >= math.ceil(0.75 * len(effects))
        and float(np.min(effects)) >= -0.10
    )


def _passes_q2(difference: np.ndarray, simultaneous_upper: float, pvalue: float) -> bool:
    return bool(
        float(np.mean(difference)) <= NONINFERIORITY_MARGIN
        and simultaneous_upper <= NONINFERIORITY_MARGIN
        and pvalue <= 0.05
        and int(np.sum(difference <= NONINFERIORITY_MARGIN)) >= math.ceil(0.75 * len(difference))
        and float(np.max(difference)) <= 0.20
    )


Q0_BUDGETS = ("1", "2", "4", "8", "16", "32", "FULL")
Q0_FIELDS = (
    "seed", "target", "level", "budget", "standardized_regret",
    "top1", "top5", "top10", "coverage_top1", "coverage_top5", "coverage_top10",
)


def validate_q0_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if set(row) != set(Q0_FIELDS):
            raise C82ValidationError("frozen Q0 schema drift")
        seed, target, level = int(row["seed"]), int(row["target"]), int(row["level"])
        budget = str(row["budget"])
        if seed not in SEEDS or target not in PRIMARY_TARGETS or level not in LEVELS or budget not in Q0_BUDGETS:
            raise C82ValidationError("frozen Q0 key drift")
        values = {field: _finite_float(row[field], field) for field in Q0_FIELDS[4:]}
        if not 0.0 <= values["standardized_regret"] <= 1.0:
            raise C82ValidationError("frozen Q0 regret outside [0,1]")
        if any(not 0.0 <= values[field] <= 1.0 for field in Q0_FIELDS[5:]):
            raise C82ValidationError("frozen Q0 probability outside [0,1]")
        normalized.append({
            "seed": seed, "target": target, "level": level, "budget": budget, **values,
        })
    expected = {
        (seed, target, level, budget)
        for seed in SEEDS for target in PRIMARY_TARGETS for level in LEVELS for budget in Q0_BUDGETS
    }
    observed = {(row["seed"], row["target"], row["level"], row["budget"]) for row in normalized}
    if len(normalized) != 224 or observed != expected or len(observed) != len(normalized):
        raise C82ValidationError("frozen Q0 context coverage drift")
    return normalized


def _target_method_value(
    rows: Sequence[Mapping[str, Any]], seed: int, target: int, method: str, field: str,
) -> float:
    values = [
        float(row[field]) for row in rows
        if row["seed"] == seed and row["target"] == target and row["method_id"] == method
    ]
    if len(values) != len(LEVELS):
        raise C82ValidationError(f"target-method coverage drift: {seed}/{target}/{method}/{field}")
    return float(np.mean(values))


def _target_q0_value(
    rows: Sequence[Mapping[str, Any]], seed: int, target: int, budget: str, field: str,
) -> float:
    values = [
        float(row[field]) for row in rows
        if row["seed"] == seed and row["target"] == target and row["budget"] == budget
    ]
    if len(values) != len(LEVELS):
        raise C82ValidationError(f"Q0 target coverage drift: {seed}/{target}/{budget}/{field}")
    return float(np.mean(values))


def _seed_category(q1: Mapping[str, bool], q2: Mapping[str, bool]) -> str:
    if any(q1.get(method, False) and q2.get(method, False) for method in PRIMARY_ZERO_METHODS):
        return "A"
    if any(q1.get(method, False) for method in PRIMARY_ZERO_METHODS):
        return "B"
    return "C"


def _comparison_evidence(
    rows: Sequence[Mapping[str, Any]],
    q0_rows: Sequence[Mapping[str, Any]],
    targets: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, bool]], dict[int, dict[str, bool]]]:
    tests = [(seed, method) for seed in SEEDS for method in PRIMARY_ZERO_METHODS]
    q1 = np.column_stack([
        [
            _target_method_value(rows, seed, target, "S1", "standardized_regret")
            - _target_method_value(rows, seed, target, method, "standardized_regret")
            for target in targets
        ]
        for seed, method in tests
    ])
    q2_difference = np.column_stack([
        [
            _target_method_value(rows, seed, target, method, "standardized_regret")
            - _target_q0_value(q0_rows, seed, target, "1", "standardized_regret")
            for target in targets
        ]
        for seed, method in tests
    ])
    q1_evidence = exact_signflip_maxT(q1, margin=MATERIAL_MARGIN)
    q2_evidence = exact_signflip_maxT(NONINFERIORITY_MARGIN - q2_difference)
    q2_band = exact_signflip_maxT(q2_difference)
    per_seed_q1 = {seed: {} for seed in SEEDS}
    per_seed_q2 = {seed: {} for seed in SEEDS}
    evidence_rows: list[dict[str, Any]] = []
    for index, (seed, method) in enumerate(tests):
        q1_pass = _passes_q1(q1[:, index], float(q1_evidence["pvalue"][index]))
        q2_pass = _passes_q2(
            q2_difference[:, index],
            float(q2_band["upper"][index]),
            float(q2_evidence["pvalue"][index]),
        )
        per_seed_q1[seed][method] = q1_pass
        per_seed_q2[seed][method] = q2_pass
        evidence_rows.append({
            "seed": seed,
            "method_id": method,
            "target_count": len(targets),
            "mean_regret_improvement_vs_source": float(np.mean(q1[:, index])),
            "Q1_simultaneous_lower": float(q1_evidence["lower"][index]),
            "Q1_maxT_p": float(q1_evidence["pvalue"][index]),
            "Q1_favorable_targets": int(np.sum(q1[:, index] > 0.0)),
            "Q1_worst_target": float(np.min(q1[:, index])),
            "Q1_pass": int(q1_pass),
            "mean_regret_difference_vs_Q0_B1": float(np.mean(q2_difference[:, index])),
            "Q2_simultaneous_upper": float(q2_band["upper"][index]),
            "Q2_maxT_p": float(q2_evidence["pvalue"][index]),
            "Q2_favorable_targets": int(np.sum(q2_difference[:, index] <= NONINFERIORITY_MARGIN)),
            "Q2_worst_target": float(np.max(q2_difference[:, index])),
            "Q2_pass": int(q2_pass),
        })
    return evidence_rows, per_seed_q1, per_seed_q2


def _method_sets(
    q1: Mapping[int, Mapping[str, bool]], q2: Mapping[int, Mapping[str, bool]],
) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    a_sets = {
        seed: {method for method in PRIMARY_ZERO_METHODS if q1[seed][method] and q2[seed][method]}
        for seed in SEEDS
    }
    b_sets = {
        seed: {method for method in PRIMARY_ZERO_METHODS if q1[seed][method]}
        for seed in SEEDS
    }
    return a_sets, b_sets


def classify_same_method_taxonomy(
    *,
    q1: Mapping[int, Mapping[str, bool]],
    q2: Mapping[int, Mapping[str, bool]],
    loto_preserved: int,
    blocker: bool = False,
) -> dict[str, Any]:
    a_sets, b_sets = _method_sets(q1, q2)
    a_common = a_sets[3] & a_sets[4]
    b_common = b_sets[3] & b_sets[4]
    categories = {seed: _seed_category(q1[seed], q2[seed]) for seed in SEEDS}
    identity_failure = (
        (categories[3] == categories[4] == "A" and not a_common)
        or (categories[3] == categories[4] == "B" and not b_common)
    )
    if blocker:
        gate = GATE_E
    elif categories[3] != categories[4] or identity_failure or loto_preserved < LOTO_MINIMUM:
        gate = GATE_D
    elif categories[3] == "A" and a_common:
        gate = GATE_A
    elif categories[3] == "B" and not a_common and b_common:
        gate = GATE_B
    elif categories[3] == "C" and not b_common:
        gate = GATE_C
    else:
        gate = GATE_D
    return {
        "seed3_category": categories[3],
        "seed4_category": categories[4],
        "A_seed3": sorted(a_sets[3]),
        "A_seed4": sorted(a_sets[4]),
        "A_intersection": sorted(a_common),
        "B_seed3": sorted(b_sets[3]),
        "B_seed4": sorted(b_sets[4]),
        "B_intersection": sorted(b_common),
        "same_method_identity_failure": identity_failure,
        "LOTO_preserved": int(loto_preserved),
        "LOTO_total": 16,
        "primary_taxonomy": gate,
    }


def _loto_analysis(
    rows: Sequence[Mapping[str, Any]],
    q0_rows: Sequence[Mapping[str, Any]],
    full_q1: Mapping[int, Mapping[str, bool]],
    full_q2: Mapping[int, Mapping[str, bool]],
) -> tuple[list[dict[str, Any]], int]:
    full_a, full_b = _method_sets(full_q1, full_q2)
    full_category = {seed: _seed_category(full_q1[seed], full_q2[seed]) for seed in SEEDS}
    common_a = full_a[3] & full_a[4]
    common_b = full_b[3] & full_b[4]
    output: list[dict[str, Any]] = []
    preserved = 0
    for left_target in PRIMARY_TARGETS:
        keep = [target for target in PRIMARY_TARGETS if target != left_target]
        _, q1_sub, q2_sub = _comparison_evidence(rows, q0_rows, keep)
        for seed in SEEDS:
            category = _seed_category(q1_sub[seed], q2_sub[seed])
            if full_category[seed] == "A":
                supporting = sorted(
                    method for method in common_a if q1_sub[seed][method] and q2_sub[seed][method]
                )
                method_preserved = bool(supporting)
            elif full_category[seed] == "B":
                supporting = sorted(method for method in common_b if q1_sub[seed][method])
                method_preserved = bool(supporting)
            else:
                supporting = []
                method_preserved = not any(q1_sub[seed].values())
            panel_preserved = category == full_category[seed] and method_preserved
            preserved += int(panel_preserved)
            output.append({
                "seed": seed,
                "left_out_target": left_target,
                "full_category": full_category[seed],
                "LOTO_category": category,
                "supporting_same_methods": "|".join(supporting) if supporting else "NONE",
                "same_method_preserved": int(method_preserved),
                "category_preserved": int(category == full_category[seed]),
                "panel_preserved": int(panel_preserved),
            })
    return output, preserved


def _pairwise_order_accuracy(score: np.ndarray, utility: np.ndarray) -> float:
    left, right = np.triu_indices(len(score), 1)
    score_delta = score[left] - score[right]
    utility_delta = utility[left] - utility[right]
    informative = (np.abs(score_delta) > 1e-15) & (np.abs(utility_delta) > 1e-15)
    if not np.any(informative):
        return 0.5
    return float(np.mean(np.sign(score_delta[informative]) == np.sign(utility_delta[informative])))


def _measurement_outputs(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {
        (seed, target, level, method)
        for seed in SEEDS for target in PRIMARY_TARGETS for level in LEVELS for method in RANK_METHODS
    }
    observed: set[tuple[int, int, int, str]] = set()
    rows: list[dict[str, Any]] = []
    u16: list[dict[str, Any]] = []
    for record in records:
        if set(record) != {"seed", "target", "level", "method_id", "scores", "utility"}:
            raise C82ValidationError("candidate measurement record schema drift")
        key = (int(record["seed"]), int(record["target"]), int(record["level"]), str(record["method_id"]))
        if key not in expected or key in observed:
            raise C82ValidationError(f"candidate measurement record identity drift: {key}")
        observed.add(key)
        score = np.asarray(record["scores"], dtype=float)
        utility = np.asarray(record["utility"], dtype=float)
        if score.shape != (CANDIDATES,) or utility.shape != (CANDIDATES,):
            raise C82ValidationError("candidate measurement vector shape drift")
        if not np.all(np.isfinite(score)) or not np.all(np.isfinite(utility)):
            raise C82ValidationError("candidate measurement contains non-finite values")
        spearman = float(stats.spearmanr(score, utility).statistic)
        kendall = float(stats.kendalltau(score, utility).statistic)
        if not math.isfinite(spearman):
            spearman = 0.0
        if not math.isfinite(kendall):
            kendall = 0.0
        estimate_applicable = key[3] in PERFORMANCE_ESTIMATE_METHODS
        if estimate_applicable:
            residual = utility - score
            mae: float | str = float(np.mean(np.abs(residual)))
            denominator = float(np.sum((utility - np.mean(utility)) ** 2))
            incremental_r2: float | str = (
                float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 1e-15 else 0.0
            )
            design = np.column_stack((np.ones(CANDIDATES), score))
            coefficients = np.linalg.lstsq(design, utility, rcond=None)[0]
            intercept: float | str = float(coefficients[0])
            slope: float | str = float(coefficients[1])
        else:
            mae = incremental_r2 = intercept = slope = "NA"
        rows.append({
            "seed": key[0], "target": key[1], "level": key[2], "method_id": key[3],
            "spearman": spearman,
            "kendall": kendall,
            "pairwise_order_accuracy": _pairwise_order_accuracy(score, utility),
            "performance_estimate_applicable": int(estimate_applicable),
            "utility_estimation_MAE": mae,
            "incremental_R2": incremental_r2,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        })
        if key[3] == "U15":
            u16.append({
                "seed": key[0], "target": key[1], "level": key[2],
                "source_method_id": "U15",
                "accuracy_on_the_line_MAE": float(mae),
                "accuracy_on_the_line_incremental_R2": float(incremental_r2),
                "calibration_intercept": float(intercept),
                "calibration_slope": float(slope),
                "diagnostic_only": 1,
                "enters_primary_taxonomy": 0,
            })
    if observed != expected:
        raise C82ValidationError(f"candidate measurement coverage drift: missing={len(expected - observed)}")
    return rows, u16


def _method_seed_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        for method in CONTEXT_METHODS:
            selected = [row for row in rows if row["seed"] == seed and row["method_id"] == method]
            if len(selected) != 16:
                raise C82ValidationError("seed-method context coverage drift")
            source_regret = np.mean([
                row["standardized_regret"] for row in rows if row["seed"] == seed and row["method_id"] == "S1"
            ])
            output.append({
                "seed": seed,
                "method_id": method,
                "information_class": INFORMATION_CLASS[method],
                "mean_standardized_regret": float(np.mean([row["standardized_regret"] for row in selected])),
                "mean_selected_utility": float(np.mean([row["selected_utility"] for row in selected])),
                "source_relative_regret_gain": float(
                    source_regret - np.mean([row["standardized_regret"] for row in selected])
                ),
                "top1": float(np.mean([row["top1"] for row in selected])),
                "top5": float(np.mean([row["top5"] for row in selected])),
                "top10": float(np.mean([row["top10"] for row in selected])),
                "coverage_top1": float(np.mean([row["coverage_top1"] for row in selected])),
                "coverage_top5": float(np.mean([row["coverage_top5"] for row in selected])),
                "coverage_top10": float(np.mean([row["coverage_top10"] for row in selected])),
                "target_count": 8,
                "context_count": 16,
            })
    return output


def _derive_tables(
    rows: list[dict[str, Any]],
    q0_rows: list[dict[str, Any]],
    measurement_records: Sequence[Mapping[str, Any]],
    *,
    blocker: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    evidence, full_q1, full_q2 = _comparison_evidence(rows, q0_rows, PRIMARY_TARGETS)
    loto_rows, loto_preserved = _loto_analysis(rows, q0_rows, full_q1, full_q2)
    taxonomy = classify_same_method_taxonomy(
        q1=full_q1, q2=full_q2, loto_preserved=loto_preserved, blocker=blocker,
    )
    a_sets, b_sets = _method_sets(full_q1, full_q2)
    summary = _method_seed_summary(rows)
    summary_lookup = {(row["seed"], row["method_id"]): row for row in summary}
    measurements, u16 = _measurement_outputs(measurement_records)

    intersections = [{
        "set_id": "A_seed3", "methods": "|".join(sorted(a_sets[3])) or "NONE",
        "count": len(a_sets[3]),
    }, {
        "set_id": "A_seed4", "methods": "|".join(sorted(a_sets[4])) or "NONE",
        "count": len(a_sets[4]),
    }, {
        "set_id": "A_intersection", "methods": "|".join(taxonomy["A_intersection"]) or "NONE",
        "count": len(taxonomy["A_intersection"]),
    }, {
        "set_id": "B_seed3", "methods": "|".join(sorted(b_sets[3])) or "NONE",
        "count": len(b_sets[3]),
    }, {
        "set_id": "B_seed4", "methods": "|".join(sorted(b_sets[4])) or "NONE",
        "count": len(b_sets[4]),
    }, {
        "set_id": "B_intersection", "methods": "|".join(taxonomy["B_intersection"]) or "NONE",
        "count": len(taxonomy["B_intersection"]),
    }]

    identity = [{
        "seed3_category": taxonomy["seed3_category"],
        "seed4_category": taxonomy["seed4_category"],
        "A_intersection": "|".join(taxonomy["A_intersection"]) or "NONE",
        "B_intersection": "|".join(taxonomy["B_intersection"]) or "NONE",
        "same_method_identity_failure": int(taxonomy["same_method_identity_failure"]),
        "LOTO_preserved": taxonomy["LOTO_preserved"],
        "LOTO_total": taxonomy["LOTO_total"],
        "stability_pass": int(taxonomy["primary_taxonomy"] not in {GATE_D, GATE_E}),
        "primary_taxonomy": taxonomy["primary_taxonomy"],
    }]

    regret_table = [{
        "seed": row["seed"], "method_id": row["method_id"],
        "information_class": row["information_class"],
        "mean_standardized_regret": row["mean_standardized_regret"],
    } for row in summary]
    utility_table = [{
        "seed": row["seed"], "method_id": row["method_id"],
        "information_class": row["information_class"],
        "mean_selected_utility": row["mean_selected_utility"],
    } for row in summary]
    gain_table = [{
        "seed": row["seed"], "method_id": row["method_id"],
        "information_class": row["information_class"],
        "source_relative_regret_gain": row["source_relative_regret_gain"],
    } for row in summary]
    topk_table = [{
        "seed": row["seed"], "method_id": row["method_id"],
        "top1": row["top1"], "top5": row["top5"], "top10": row["top10"],
    } for row in summary]

    q1_table = [{
        "seed": row["seed"], "method_id": row["method_id"],
        "mean_regret_improvement_vs_source": row["mean_regret_improvement_vs_source"],
        "simultaneous_lower": row["Q1_simultaneous_lower"],
        "maxT_p": row["Q1_maxT_p"],
        "favorable_targets": row["Q1_favorable_targets"],
        "worst_target": row["Q1_worst_target"],
        "Q1_pass": row["Q1_pass"],
    } for row in evidence]
    q2_table = [{
        "seed": row["seed"], "method_id": row["method_id"],
        "mean_regret_difference_vs_Q0_B1": row["mean_regret_difference_vs_Q0_B1"],
        "simultaneous_upper": row["Q2_simultaneous_upper"],
        "maxT_p": row["Q2_maxT_p"],
        "favorable_targets": row["Q2_favorable_targets"],
        "worst_target": row["Q2_worst_target"],
        "Q2_pass": row["Q2_pass"],
    } for row in evidence]

    cross_seed = []
    for method in CONTEXT_METHODS:
        seed3 = summary_lookup[(3, method)]
        seed4 = summary_lookup[(4, method)]
        cross_seed.append({
            "method_id": method,
            "seed3_regret": seed3["mean_standardized_regret"],
            "seed4_regret": seed4["mean_standardized_regret"],
            "paired_difference_seed4_minus_seed3": (
                seed4["mean_standardized_regret"] - seed3["mean_standardized_regret"]
            ),
            "same_direction_vs_source": int(
                np.sign(seed3["source_relative_regret_gain"]) == np.sign(seed4["source_relative_regret_gain"])
            ),
            "seed_is_paired_factor": 1,
        })

    q3 = []
    evidence_lookup = {(row["seed"], row["method_id"]): row for row in evidence}
    for seed in SEEDS:
        source = summary_lookup[(seed, "S1")]
        for method in PRIMARY_ZERO_METHODS:
            item = summary_lookup[(seed, method)]
            for endpoint in ("standardized_regret", "top1", "top5", "top10"):
                if endpoint == "standardized_regret":
                    difference = source["mean_standardized_regret"] - item["mean_standardized_regret"]
                    favorable = evidence_lookup[(seed, method)]["Q1_pass"]
                else:
                    difference = item[endpoint] - source[endpoint]
                    favorable = int(difference > 0.0)
                q3.append({
                    "seed": seed, "method_id": method, "endpoint": endpoint,
                    "difference_vs_source": difference,
                    "favorable_direction": int(favorable),
                    "regret_Q1_pass": evidence_lookup[(seed, method)]["Q1_pass"],
                    "endpoint_substitutes_for_regret": 0,
                })

    q5 = []
    class_methods = {
        "I0": ("B0", "B1", "B2", "B3", "B4O", "B4S"),
        "IS": ("S1", "S2"),
        "IU_ISU": ("U1", "U2", "U3", "U4", "U5", "U6", "U7", "U11", "U12", "U13", "U14", "U15"),
    }
    for seed in SEEDS:
        for information_class, methods in class_methods.items():
            candidates = [summary_lookup[(seed, method)] for method in methods]
            best = min(candidates, key=lambda row: (row["mean_standardized_regret"], row["method_id"]))
            q5.append({
                "seed": seed, "information_class": information_class,
                "registered_methods": "|".join(methods),
                "best_registered_method": best["method_id"],
                "best_mean_standardized_regret": best["mean_standardized_regret"],
                "outcome_dependent_family_redefinition": 0,
                "field_specific_only": 1,
            })
        labeled = {
            label: float(np.mean([
                row["standardized_regret"]
                for row in q0_rows if row["seed"] == seed and row["budget"] == budget
            ]))
            for budget, label in (("1", "L1"), ("FULL", "L7"))
        }
        best_label = min(labeled, key=lambda label: (labeled[label], label))
        q5.append({
            "seed": seed, "information_class": "ILc",
            "registered_methods": "L1|L7",
            "best_registered_method": best_label,
            "best_mean_standardized_regret": labeled[best_label],
            "outcome_dependent_family_redefinition": 0,
            "field_specific_only": 1,
        })

    catastrophic = []
    for seed in SEEDS:
        for target in PRIMARY_TARGETS:
            source = _target_method_value(rows, seed, target, "S1", "standardized_regret")
            for method in CONTEXT_METHODS:
                regret = _target_method_value(rows, seed, target, method, "standardized_regret")
                catastrophic.append({
                    "seed": seed, "target": target, "method_id": method,
                    "target_regret": regret,
                    "source_relative_gain": source - regret,
                    "Q1_catastrophic_floor_breached": int(source - regret < -0.10),
                    "Q2_catastrophic_excess_breached": int(
                        regret - _target_q0_value(q0_rows, seed, target, "1", "standardized_regret") > 0.20
                    ),
                })

    coverage = []
    for row in summary:
        for k in (1, 5, 10):
            coverage.append({
                "seed": row["seed"], "method_id": row["method_id"], "k": k,
                "coverage": row[f"coverage_top{k}"],
            })

    regimes = []
    for seed in SEEDS:
        for method in CONTEXT_METHODS:
            selected = [row["selected_regime"] for row in rows if row["seed"] == seed and row["method_id"] == method]
            for regime in sorted(VALID_REGIMES):
                regimes.append({
                    "seed": seed, "method_id": method, "selected_regime": regime,
                    "count": selected.count(regime), "fraction": selected.count(regime) / len(selected),
                })

    measurement_by_seed_method: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in measurements:
        measurement_by_seed_method.setdefault((row["seed"], row["method_id"]), []).append(row)
    separation = []
    for key, selected in sorted(measurement_by_seed_method.items()):
        seed, method = key
        summary_row = summary_lookup[(seed, method)]
        separation.append({
            "seed": seed, "method_id": method,
            "mean_spearman": float(np.mean([row["spearman"] for row in selected])),
            "mean_pairwise_order_accuracy": float(np.mean([row["pairwise_order_accuracy"] for row in selected])),
            "mean_standardized_regret": summary_row["mean_standardized_regret"],
            "source_relative_regret_gain": summary_row["source_relative_regret_gain"],
            "measurement_substitutes_for_decision": 0,
        })

    tables = {
        "method_context_results.csv": rows,
        "seed_method_Q1_Q2.csv": evidence,
        "cross_seed_qualifying_method_intersection.csv": intersections,
        "cross_seed_method_identity_stability.csv": identity,
        "primary_method_regret_table.csv": regret_table,
        "primary_method_selected_utility_table.csv": utility_table,
        "primary_method_source_relative_gain.csv": gain_table,
        "primary_method_topk_table.csv": topk_table,
        "zero_label_vs_strict_source_maxT.csv": q1_table,
        "zero_label_vs_Q0_B1_noninferiority.csv": q2_table,
        "seed_specific_method_results.csv": summary,
        "cross_seed_stability.csv": cross_seed,
        "leave_one_target_method_stability.csv": loto_rows,
        "objective_dependence_Q3.csv": q3,
        "information_class_summary_Q5.csv": q5,
        "measurement_vs_decision_separation.csv": separation,
        "method_measurement_metrics.csv": measurements,
        "accuracy_on_the_line_diagnostic.csv": u16,
        "target_level_catastrophic_failures.csv": catastrophic,
        "coverage_summary.csv": coverage,
        "selected_regime_distribution.csv": regimes,
        "q0_budget_context.csv": q0_rows,
        "loro_status.csv": [{
            "analysis_id": "LORO", "operative": 0,
            "status": "REMOVED_FROM_OPERATIVE_C82_INFERENCE",
            "reason": "mixed_81_candidate_context_estimand_not_executable",
            "cross_regime_transport_claim": 0,
        }],
    }
    return tables, taxonomy


TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "method_context_results.csv": METHOD_CONTEXT_FIELDS,
    "seed_method_Q1_Q2.csv": (
        "seed", "method_id", "target_count", "mean_regret_improvement_vs_source",
        "Q1_simultaneous_lower", "Q1_maxT_p", "Q1_favorable_targets", "Q1_worst_target",
        "Q1_pass", "mean_regret_difference_vs_Q0_B1", "Q2_simultaneous_upper",
        "Q2_maxT_p", "Q2_favorable_targets", "Q2_worst_target", "Q2_pass",
    ),
    "cross_seed_qualifying_method_intersection.csv": ("set_id", "methods", "count"),
    "cross_seed_method_identity_stability.csv": (
        "seed3_category", "seed4_category", "A_intersection", "B_intersection",
        "same_method_identity_failure", "LOTO_preserved", "LOTO_total", "stability_pass",
        "primary_taxonomy",
    ),
    "primary_method_regret_table.csv": (
        "seed", "method_id", "information_class", "mean_standardized_regret",
    ),
    "primary_method_selected_utility_table.csv": (
        "seed", "method_id", "information_class", "mean_selected_utility",
    ),
    "primary_method_source_relative_gain.csv": (
        "seed", "method_id", "information_class", "source_relative_regret_gain",
    ),
    "primary_method_topk_table.csv": ("seed", "method_id", "top1", "top5", "top10"),
    "zero_label_vs_strict_source_maxT.csv": (
        "seed", "method_id", "mean_regret_improvement_vs_source", "simultaneous_lower",
        "maxT_p", "favorable_targets", "worst_target", "Q1_pass",
    ),
    "zero_label_vs_Q0_B1_noninferiority.csv": (
        "seed", "method_id", "mean_regret_difference_vs_Q0_B1", "simultaneous_upper",
        "maxT_p", "favorable_targets", "worst_target", "Q2_pass",
    ),
    "seed_specific_method_results.csv": (
        "seed", "method_id", "information_class", "mean_standardized_regret",
        "mean_selected_utility", "source_relative_regret_gain", "top1", "top5", "top10",
        "coverage_top1", "coverage_top5", "coverage_top10", "target_count", "context_count",
    ),
    "cross_seed_stability.csv": (
        "method_id", "seed3_regret", "seed4_regret", "paired_difference_seed4_minus_seed3",
        "same_direction_vs_source", "seed_is_paired_factor",
    ),
    "leave_one_target_method_stability.csv": (
        "seed", "left_out_target", "full_category", "LOTO_category", "supporting_same_methods",
        "same_method_preserved", "category_preserved", "panel_preserved",
    ),
    "objective_dependence_Q3.csv": (
        "seed", "method_id", "endpoint", "difference_vs_source", "favorable_direction",
        "regret_Q1_pass", "endpoint_substitutes_for_regret",
    ),
    "information_class_summary_Q5.csv": (
        "seed", "information_class", "registered_methods", "best_registered_method",
        "best_mean_standardized_regret", "outcome_dependent_family_redefinition",
        "field_specific_only",
    ),
    "measurement_vs_decision_separation.csv": (
        "seed", "method_id", "mean_spearman", "mean_pairwise_order_accuracy",
        "mean_standardized_regret", "source_relative_regret_gain",
        "measurement_substitutes_for_decision",
    ),
    "method_measurement_metrics.csv": (
        "seed", "target", "level", "method_id", "spearman", "kendall",
        "pairwise_order_accuracy", "performance_estimate_applicable", "utility_estimation_MAE",
        "incremental_R2", "calibration_intercept", "calibration_slope",
    ),
    "accuracy_on_the_line_diagnostic.csv": (
        "seed", "target", "level", "source_method_id", "accuracy_on_the_line_MAE",
        "accuracy_on_the_line_incremental_R2", "calibration_intercept", "calibration_slope",
        "diagnostic_only", "enters_primary_taxonomy",
    ),
    "target_level_catastrophic_failures.csv": (
        "seed", "target", "method_id", "target_regret", "source_relative_gain",
        "Q1_catastrophic_floor_breached", "Q2_catastrophic_excess_breached",
    ),
    "coverage_summary.csv": ("seed", "method_id", "k", "coverage"),
    "selected_regime_distribution.csv": (
        "seed", "method_id", "selected_regime", "count", "fraction",
    ),
    "q0_budget_context.csv": Q0_FIELDS,
    "loro_status.csv": (
        "analysis_id", "operative", "status", "reason", "cross_regime_transport_claim",
    ),
}

EXPECTED_TABLE_ROWS = {
    "method_context_results.csv": 672,
    "seed_method_Q1_Q2.csv": 12,
    "cross_seed_qualifying_method_intersection.csv": 6,
    "cross_seed_method_identity_stability.csv": 1,
    "primary_method_regret_table.csv": 42,
    "primary_method_selected_utility_table.csv": 42,
    "primary_method_source_relative_gain.csv": 42,
    "primary_method_topk_table.csv": 42,
    "zero_label_vs_strict_source_maxT.csv": 12,
    "zero_label_vs_Q0_B1_noninferiority.csv": 12,
    "seed_specific_method_results.csv": 42,
    "cross_seed_stability.csv": 21,
    "leave_one_target_method_stability.csv": 16,
    "objective_dependence_Q3.csv": 48,
    "information_class_summary_Q5.csv": 8,
    "measurement_vs_decision_separation.csv": 28,
    "method_measurement_metrics.csv": 448,
    "accuracy_on_the_line_diagnostic.csv": 32,
    "target_level_catastrophic_failures.csv": 336,
    "coverage_summary.csv": 126,
    "selected_regime_distribution.csv": 210,
    "q0_budget_context.csv": 224,
    "loro_status.csv": 1,
}


def _validate_table_rows(name: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if name not in TABLE_FIELDS:
        raise C82ValidationError(f"unregistered output table: {name}")
    if not rows:
        raise C82ValidationError(f"refusing empty required C82 table: {name}")
    if len(rows) != EXPECTED_TABLE_ROWS[name]:
        raise C82ValidationError(
            f"table row-count mismatch {name}: expected={EXPECTED_TABLE_ROWS[name]} observed={len(rows)}"
        )
    fields = TABLE_FIELDS[name]
    required = set(fields)
    normalized = []
    for row in rows:
        if set(row) != required:
            raise C82ValidationError(
                f"table schema mismatch {name}: missing={sorted(required - set(row))} "
                f"unknown={sorted(set(row) - required)}"
            )
        normalized.append({field: row[field] for field in fields})
    return normalized


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="raise", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _registered_csv_outputs() -> set[str]:
    rows = read_csv(OUTPUT_REGISTRY_PATH)
    names = {row["artifact"] for row in rows if row["artifact"].endswith(".csv")}
    if names != set(TABLE_FIELDS):
        raise C82ValidationError(
            f"output registry/code mismatch missing={sorted(names - set(TABLE_FIELDS))} "
            f"unknown={sorted(set(TABLE_FIELDS) - names)}"
        )
    return names


def _validate_manifest(stage: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(TABLE_FIELDS):
        raise C82ValidationError("artifact manifest cardinality drift")
    for item in artifacts:
        path = stage / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise C82ValidationError(f"artifact size replay failed: {item['path']}")
        if sha256_file(path) != item["sha256"]:
            raise C82ValidationError(f"artifact hash replay failed: {item['path']}")


def atomic_freeze(
    final_directory: str | Path,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    result: Mapping[str, Any],
    *,
    inject_partial_write_failure: bool = False,
) -> dict[str, Any]:
    """Publish a complete result directory or leave no final directory."""
    final_directory = Path(final_directory)
    if final_directory.exists():
        raise C82ValidationError(f"C82 final result directory already exists: {final_directory}")
    if set(tables) != _registered_csv_outputs():
        raise C82ValidationError("C82 complete output table set drift")
    validated = {name: _validate_table_rows(name, tables[name]) for name in sorted(tables)}
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{final_directory.name}.staging-", dir=final_directory.parent))
    try:
        artifact_rows = []
        for index, name in enumerate(sorted(validated)):
            _write_csv(stage / name, TABLE_FIELDS[name], validated[name])
            if inject_partial_write_failure and index == 0:
                raise C82ValidationError("injected partial table-write failure")
            path = stage / name
            artifact_rows.append({
                "path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                "rows": len(validated[name]),
            })
        manifest = {
            "schema_version": "c82_atomic_result_manifest_v1",
            "artifacts": artifact_rows,
            "table_count": len(artifact_rows),
            "method_context_rows": len(validated["method_context_results.csv"]),
            "all_tables_validated_before_result": True,
        }
        manifest_path = stage / "artifact_manifest.json"
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2)
            handle.write("\n")
        _validate_manifest(stage, manifest)
        final_result = dict(result)
        final_result["artifact_manifest_sha256"] = sha256_file(manifest_path)
        final_result["artifact_manifest_table_count"] = len(artifact_rows)
        result_path = stage / "C82_POST_C81_BASELINE_RECOVERY.json"
        with result_path.open("x") as handle:
            json.dump(final_result, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.rename(stage, final_directory)
        return {
            "result_directory": str(final_directory),
            "artifact_manifest_sha256": sha256_file(final_directory / "artifact_manifest.json"),
            "result_sha256": sha256_file(final_directory / "C82_POST_C81_BASELINE_RECOVERY.json"),
            "table_count": len(artifact_rows),
            "method_context_rows": len(validated["method_context_results.csv"]),
        }
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _validate_selection_identity(identity: Mapping[str, Any], *, synthetic: bool) -> None:
    required = {"mode", "manifest_self_sha256", "payload_sha256", "contexts", "methods", "recomputed"}
    if set(identity) != required:
        raise C82ValidationError("selection identity schema drift")
    if int(identity["contexts"]) != 32 or int(identity["methods"]) != 19 or bool(identity["recomputed"]):
        raise C82ValidationError("selection identity cardinality or recomputation drift")
    if synthetic:
        if identity["mode"] != "synthetic_fixture":
            raise C82ValidationError("synthetic selection mode drift")
        return
    protocol, _ = load_protocol()
    frozen = protocol["frozen_selection"]
    if identity["mode"] != "real_frozen_C81_selection":
        raise C82ValidationError("real selection mode drift")
    if identity["manifest_self_sha256"] != frozen["manifest_self_sha256"]:
        raise C82ValidationError("selection manifest self-hash drift")
    if identity["payload_sha256"] != frozen["payload_sha256"]:
        raise C82ValidationError("selection payload hash drift")


def run_recovery(
    *,
    method_context_rows: Sequence[Mapping[str, Any]],
    q0_rows: Sequence[Mapping[str, Any]],
    measurement_records: Sequence[Mapping[str, Any]],
    selection_identity: Mapping[str, Any],
    final_directory: str | Path,
    synthetic: bool,
    blocker: bool = False,
    inject_partial_write_failure: bool = False,
    inject_post_evaluation_failure: bool = False,
) -> dict[str, Any]:
    """Shared synthetic/real entrypoint; callers inject only already-frozen selection."""
    _validate_selection_identity(selection_identity, synthetic=synthetic)
    rows = validate_method_context_rows(method_context_rows)
    frozen_q0 = validate_q0_rows(q0_rows)
    if inject_post_evaluation_failure:
        raise C82PostEvaluationFailure("injected post-evaluation failure; authorization consumed")
    tables, taxonomy = _derive_tables(rows, frozen_q0, measurement_records, blocker=blocker)
    protocol, protocol_sha = load_protocol()
    result = {
        "schema_version": "c82_post_c81_baseline_recovery_result_v1",
        "milestone": "C82",
        "protocol_sha256": protocol_sha,
        "base_C81_protocol_sha256": protocol["base_scientific_registry"]["protocol_sha256"],
        "selection_manifest_self_sha256": selection_identity["manifest_self_sha256"],
        "selection_payload_sha256": selection_identity["payload_sha256"],
        "selection_recomputed": False,
        "method_context_rows": 672,
        "contexts": 32,
        "selection_methods": 19,
        "target4_primary": False,
        "same_label_oracle_accessed": False,
        "post_C81_outcome_access_recovery": True,
        "C81_gate_unchanged": "C81-E_protocol_input_implementation_or_provenance_blocker",
        "independent_confirmation": False,
        "external_validity_claim": False,
        **taxonomy,
    }
    freeze = atomic_freeze(
        final_directory, tables, result,
        inject_partial_write_failure=inject_partial_write_failure,
    )
    return {**result, **freeze}


def load_execution_lock() -> tuple[dict[str, Any], str]:
    if not LOCK_PATH.is_file() or not LOCK_SHA_PATH.is_file():
        raise C82ValidationError("C82 scope-specific analysis execution lock is absent")
    expected = LOCK_SHA_PATH.read_text().strip()
    observed = sha256_file(LOCK_PATH)
    if observed != expected:
        raise C82ValidationError("C82 analysis execution lock hash mismatch")
    lock = json.loads(LOCK_PATH.read_text())
    protocol, protocol_sha = load_protocol()
    if lock["protocol"]["sha256"] != protocol_sha:
        raise C82ValidationError("C82 lock protocol hash binding mismatch")
    if lock["protocol"]["commit"] != last_commit(PROTOCOL_PATH):
        raise C82ValidationError("C82 lock protocol commit binding mismatch")
    if lock["method_registry"]["sha256"] != sha256_file(METHOD_REGISTRY_PATH):
        raise C82ValidationError("C82 method registry lock drift")
    for item in lock["implementation"] + lock["registry_artifacts"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise C82ValidationError(f"C82 locked object drift: {item['path']}")
    frozen = lock["frozen_selection"]
    if frozen["selection_recomputation_allowed"] or frozen["payload_sha256"] != protocol["frozen_selection"]["payload_sha256"]:
        raise C82ValidationError("C82 frozen selection lock drift")
    scope = lock["scope"]
    if any(scope[key] for key in ("training", "forward", "reinference", "GPU", "target4_primary", "same_label_oracle")):
        raise C82ValidationError("C82 protected execution scope drift")
    return lock, observed


def require_c82e_authorization() -> dict[str, Any]:
    """Fail before any external selection payload or evaluation descriptor is opened."""
    lock, lock_sha = load_execution_lock()
    if not AUTHORIZATION_PATH.is_file():
        raise C82ValidationError("direct C82E PI authorization record is absent")
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    if not authorization.get("authorization_received") or not authorization.get("authorization_active"):
        raise C82ValidationError("C82E direct authorization is not active")
    if authorization.get("protocol_commit") != lock["protocol"]["commit"]:
        raise C82ValidationError("C82E authorization protocol commit mismatch")
    if authorization.get("protocol_sha256") != lock["protocol"]["sha256"]:
        raise C82ValidationError("C82E authorization protocol hash mismatch")
    if authorization.get("analysis_lock_commit") != last_commit(LOCK_PATH):
        raise C82ValidationError("C82E authorization lock commit mismatch")
    if authorization.get("analysis_lock_sha256") != lock_sha:
        raise C82ValidationError("C82E authorization lock hash mismatch")
    if authorization.get("selection_manifest_sha256") != lock["frozen_selection"]["manifest_self_sha256"]:
        raise C82ValidationError("C82E authorization selection mismatch")
    if authorization.get("field_view_manifest_digest") != lock["field_view_manifest_digest"]:
        raise C82ValidationError("C82E authorization field/view mismatch")
    return {"lock": lock, "lock_sha256": lock_sha, "authorization": authorization}


def _verify_file(path: str | Path, expected_sha256: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise C82ValidationError(f"C82 bound artifact mismatch: {path}")
    return path


def _consume_authorization(context: Mapping[str, Any]) -> Path:
    marker = Path(context["lock"]["runtime"]["authorization_consumption_marker"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "c82e_authorization_consumption_v1",
        "analysis_lock_sha256": context["lock_sha256"],
        "protocol_sha256": context["lock"]["protocol"]["sha256"],
        "authorization_record_sha256": sha256_file(AUTHORIZATION_PATH),
        "authorization_consumed_before_selection_payload_or_evaluation_open": True,
        "same_identity_rerun_allowed": False,
    }
    try:
        with marker.open("x") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise C82ValidationError("C82E authorization already consumed; same-identity rerun blocked") from exc
    return marker


def _load_real_q0_rows(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    binding = lock["C80_Q0_comparators"]["target_level_regret"]
    path = _verify_file(binding["path"], binding["sha256"])
    raw = read_csv(path)
    output = []
    for row in raw:
        target = int(row["target"])
        if target not in PRIMARY_TARGETS:
            continue
        output.append({
            "seed": int(row["seed"]), "target": target, "level": int(row["level"]),
            "budget": row["budget"],
            "standardized_regret": float(row["expected_standardized_regret"]),
            "top1": float(row["top1"]), "top5": float(row["top5"]), "top10": float(row["top10"]),
            "coverage_top1": float(row["coverage_top1"]),
            "coverage_top5": float(row["coverage_top5"]),
            "coverage_top10": float(row["coverage_top10"]),
        })
    return validate_q0_rows(output)


def _random_control(utility: np.ndarray, joint_good: np.ndarray) -> dict[str, float]:
    spread = float(np.max(utility) - np.min(utility))
    regret = np.zeros(CANDIDATES) if spread <= 1e-15 else (np.max(utility) - utility) / spread
    good = int(np.sum(joint_good))
    result = {"standardized_regret": float(np.mean(regret)), "selected_utility": float(np.mean(utility))}
    for k in (1, 5, 10):
        result[f"top{k}"] = k / CANDIDATES
        result[f"coverage_top{k}"] = (
            1.0 if CANDIDATES - good < k
            else float(1.0 - math.comb(CANDIDATES - good, k) / math.comb(CANDIDATES, k))
        )
    return result


def _evaluate_order(order: np.ndarray, utility: np.ndarray, joint_good: np.ndarray) -> dict[str, float]:
    order = np.asarray(order, dtype=int)
    spread = float(np.max(utility) - np.min(utility))
    regret = 0.0 if spread <= 1e-15 else float((np.max(utility) - utility[order[0]]) / spread)
    best = int(np.lexsort((np.arange(CANDIDATES), -utility))[0])
    return {
        "standardized_regret": regret,
        "selected_utility": float(utility[order[0]]),
        "top1": float(best in set(order[:1])),
        "top5": float(best in set(order[:5])),
        "top10": float(best in set(order[:10])),
        "coverage_top1": float(np.any(joint_good[order[:1]])),
        "coverage_top5": float(np.any(joint_good[order[:5]])),
        "coverage_top10": float(np.any(joint_good[order[:10]])),
    }


def _verify_split_audit_before_evaluation() -> None:
    rows = read_csv(REPORT_DIR / "c78s_tables" / "label_split_isolation.csv")
    if {int(row["target_id"]) for row in rows} != set(PRIMARY_TARGETS):
        raise C82ValidationError("construction/evaluation split target coverage drift")
    if any(
        row["passed"] != "1" or row["overlap_rows"] != "0"
        or row["same_label_oracle_accessed"] != "0"
        or row["trial_id_used_as_predictor"] != "0"
        or row["row_order_used_as_predictor"] != "0"
        for row in rows
    ):
        raise C82ValidationError("construction/evaluation split isolation drift")


def _load_preserved_selection(lock: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from . import c81_baseline_comparison as c81

    binding = lock["frozen_selection"]
    path = _verify_file(binding["manifest_path"], binding["manifest_file_sha256"])
    manifest = c81._self_hashed_manifest(path)
    if manifest["manifest_sha256"] != binding["manifest_self_sha256"]:
        raise C82ValidationError("C82 preserved selection manifest self-hash drift")
    if manifest["descriptor"]["sha256"] != binding["payload_sha256"]:
        raise C82ValidationError("C82 preserved selection payload binding drift")
    if manifest["evaluation_labels_accessed"] or manifest["same_label_oracle_accessed"] or manifest["target4_accessed"]:
        raise C82ValidationError("C82 preserved selection isolation drift")
    selection = c81._load_selection(manifest)
    return manifest, selection


def _real_evaluation_bundle(
    lock: Mapping[str, Any], selection: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from . import c75_data
    from . import c80_label_budget_frontier as frontier
    from . import c80r_existing_field_adapter as c80

    _verify_split_audit_before_evaluation()
    method_position = {method: index for index, method in enumerate(selection["method_ids"].astype(str))}
    rows: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    for seed_key in ("seed3", "seed4"):
        binding = lock["runtime_inputs"][seed_key]
        unlabeled_path = _verify_file(
            binding["unlabeled_cache_manifest_path"], binding["unlabeled_cache_manifest_sha256"],
        )
        arrays = c80._load_unlabeled(unlabeled_path)
        route = c80._route(Path(binding["primary_route_path"]), binding["primary_route_sha256"])
        seed = int(seed_key.removeprefix("seed"))
        trial_lookup = {
            int(target): arrays["target_trial_id"][index]
            for index, target in enumerate(arrays["target_trial_id_target"])
        }
        for target in PRIMARY_TARGETS:
            evaluation = c80._load_label_view(route, target, "target_evaluation_view")
            evaluation_indices, evaluation_labels = c80._align_label_view(
                trial_lookup[target], evaluation, "target_eval",
            )
            full_labels = np.full(576, -1, dtype=int)
            full_labels[evaluation_indices] = evaluation_labels
            for level in LEVELS:
                candidate_indices = c80._cell_indices(arrays, target, level)
                cell = np.where(
                    (selection["cell_seed"] == seed)
                    & (selection["cell_target"] == target)
                    & (selection["cell_level"] == level)
                )[0]
                if len(cell) != 1 or not np.array_equal(selection["candidate_global_indices"][cell[0]], candidate_indices):
                    raise C82ValidationError("C82 selection/evaluation candidate alignment drift")
                endpoint = c80.endpoint_metrics_all_candidates(
                    arrays["target_logits"][candidate_indices], full_labels, evaluation_indices,
                )
                utility = frontier.score_from_endpoint_metrics(endpoint)
                oriented = np.column_stack((
                    c75_data.midrank_percentile(endpoint[:, 0]),
                    c75_data.midrank_percentile(-endpoint[:, 1]),
                    c75_data.midrank_percentile(-endpoint[:, 2]),
                ))
                joint_good = np.all(oriented >= 0.75, axis=1)
                regimes = arrays["regime"][candidate_indices].astype(str)
                for method in SELECTION_METHODS:
                    order = selection["selected_top10"][cell[0], method_position[method]].astype(int)
                    result = _evaluate_order(order, utility, joint_good)
                    rows.append({
                        "seed": seed, "target": target, "level": level, "method_id": method,
                        **result, "selected_regime": str(regimes[order[0]]),
                        "evaluation_label_access_after_selection_freeze": True,
                        "same_label_oracle_accessed": False, "target4_primary": False,
                    })
                    if method in RANK_METHODS:
                        measurements.append({
                            "seed": seed, "target": target, "level": level, "method_id": method,
                            "scores": selection["scores"][cell[0], method_position[method]].astype(float),
                            "utility": utility,
                        })
                random = _random_control(utility, joint_good)
                rows.append({
                    "seed": seed, "target": target, "level": level, "method_id": "B0",
                    **random, "selected_regime": "RANDOM",
                    "evaluation_label_access_after_selection_freeze": True,
                    "same_label_oracle_accessed": False, "target4_primary": False,
                })
                oracle_order = np.lexsort((np.arange(CANDIDATES), -utility))
                oracle = _evaluate_order(oracle_order, utility, joint_good)
                rows.append({
                    "seed": seed, "target": target, "level": level, "method_id": "B5",
                    **oracle, "selected_regime": "ORACLE",
                    "evaluation_label_access_after_selection_freeze": True,
                    "same_label_oracle_accessed": False, "target4_primary": False,
                })
    return rows, measurements


def run_real() -> dict[str, Any]:
    """Execute C82E once after a fresh lock-bound direct authorization."""
    context = require_c82e_authorization()
    lock = context["lock"]
    for item in lock["field_and_view_manifests"]:
        _verify_file(item["path"], item["sha256"])
    _consume_authorization(context)
    manifest, selection = _load_preserved_selection(lock)
    rows, measurements = _real_evaluation_bundle(lock, selection)
    q0_rows = _load_real_q0_rows(lock)
    identity = {
        "mode": "real_frozen_C81_selection",
        "manifest_self_sha256": manifest["manifest_sha256"],
        "payload_sha256": manifest["descriptor"]["sha256"],
        "contexts": 32,
        "methods": 19,
        "recomputed": False,
    }
    return run_recovery(
        method_context_rows=rows,
        q0_rows=q0_rows,
        measurement_records=measurements,
        selection_identity=identity,
        final_directory=lock["runtime"]["external_result_directory"],
        synthetic=False,
    )


def schema_dry_run() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    return {
        "protocol_sha256": protocol_sha,
        "protocol_commit": last_commit(PROTOCOL_PATH),
        "canonical_fields": len(METHOD_CONTEXT_FIELDS),
        "expected_method_context_rows": 672,
        "registered_output_csv_tables": len(_registered_csv_outputs()),
        "method_registry_rows": len(json.loads(METHOD_REGISTRY_PATH.read_text())["methods"]),
        "selection_payload_opened": False,
        "evaluation_view_opened": False,
        "selection_recomputed": False,
        "C82E_authorized": AUTHORIZATION_PATH.exists(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("schema-dry-run", "run-real"))
    args = parser.parse_args(argv)
    result = schema_dry_run() if args.command == "schema-dry-run" else run_real()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
