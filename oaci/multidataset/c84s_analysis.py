"""Complete C84S Stage-C analysis and atomic result publication.

The public entrypoint consumes immutable method-context decisions after the
selection freeze. It never imports label provisioning, selector, training,
forward, model, or checkpoint code.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import c84s_inference as inference
from . import c84s_taxonomy as taxonomy
from .c84s_common import (
    C84SContractError, atomic_publish_directory, canonical_sha256, read_json,
    require, sha256_file, write_csv, write_json,
)


DATASET_TARGET_COUNTS = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
PANELS = ("A", "B")
SEEDS = (5, 6)
LEVELS = (0, 1)
PRIMARY_METHODS = inference.PRIMARY_METHODS
PRIMARY_Q0 = ("Q0_B1", "Q0_B2", "Q0_B4", "Q0_B8", "Q0_FULL")
SECONDARY_Q0 = ("Q0_B16", "Q0_B32")
COMMON_METHODS = (
    "B0", "B1", "B2", "B3", "B4O", "B4S", "B5", "S1",
    *PRIMARY_METHODS, *PRIMARY_Q0,
)

METHOD_CONTEXT_FIELDS = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
    "method_id", "standardized_regret", "selected_utility",
    "source_relative_regret_gain", "top1", "top5", "top10", "coverage",
    "selected_regime", "catastrophic_failure", "Spearman", "Kendall",
    "pairwise_ordering_accuracy", "accuracy_estimation_MAE",
)

RESULT_TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "method_context_decisions.csv": METHOD_CONTEXT_FIELDS,
    "target_level_method_effects.csv": (
        "dataset", "target_subject_id", "method_id", "level",
        "method_regret", "source_regret", "Q0_B1_regret", "Q1_effect",
        "Q2_excess",
    ),
    "target_level_catastrophic_failures.csv": (
        "dataset", "target_subject_id", "method_id", "level",
        "Q1_catastrophic_floor_breached", "Q2_catastrophic_excess_breached",
    ),
    "dataset_Q1_Q2.csv": (
        "dataset", "method_id", "Q1_pass", "Q1_mean", "Q1_pvalue",
        "Q1_favorable_targets", "Q1_worst_target", "Q2_pass",
        "Q2_mean_excess", "Q2_simultaneous_upper", "Q2_pvalue",
        "Q2_within_margin_targets", "Q2_worst_excess",
        "panel_seed_Q1_all_directional", "panel_seed_Q2_all_within_margin",
    ),
    "level_specific_Q1_Q2.csv": (
        "dataset", "level", "method_id", "Q1_pass", "Q1_mean",
        "Q1_pvalue", "Q2_pass", "Q2_mean_excess", "Q2_pvalue",
        "level_heterogeneity",
    ),
    "panel_seed_stability.csv": (
        "dataset", "method_id", "Q1_A5", "Q1_A6", "Q1_B5", "Q1_B6",
        "Q2_A5", "Q2_A6", "Q2_B5", "Q2_B6",
        "Q1_all_directional", "Q2_all_within_margin",
    ),
    "leave_one_target_out.csv": (
        "dataset", "left_out_target", "full_category", "LOTO_category",
        "full_supporting_methods", "LOTO_supporting_methods",
        "same_method_preserved", "label_full_Bstar", "label_LOTO_Bstar",
        "label_Bstar_preserved",
    ),
    "label_budget_frontier.csv": (
        "dataset", "budget", "mean_effect", "maxT_pvalue",
        "direct_qualification", "closure_qualification", "Bstar",
        "level0_Bstar", "level1_Bstar", "LOTO_preserved",
        "LOTO_minimum", "registered_heterogeneity",
    ),
    "label_budget_context.csv": (
        "dataset", "target_subject_id", "budget", "primary_effect",
        "level0_effect", "level1_effect",
    ),
    "topk_decision_summary.csv": (
        "dataset", "method_id", "mean_top1", "mean_top5", "mean_top10",
        "Q1_pass", "endpoint_substitutes_for_regret",
    ),
    "selected_utility_summary.csv": (
        "dataset", "method_id", "mean_selected_utility", "mean_regret",
    ),
    "coverage_summary.csv": (
        "dataset", "method_id", "mean_coverage",
    ),
    "selected_regime_distribution.csv": (
        "dataset", "method_id", "selected_regime", "contexts", "fraction",
    ),
    "source_relative_regret_gain.csv": (
        "dataset", "method_id", "mean_source_relative_regret_gain",
    ),
    "measurement_vs_decision.csv": (
        "dataset", "method_id", "mean_Spearman", "mean_Kendall",
        "mean_pairwise_ordering_accuracy", "accuracy_estimation_MAE",
        "Q1_pass", "measurement_substitutes_for_regret",
    ),
    "cross_dataset_method_intersection.csv": (
        "set_id", "dataset", "methods", "count", "primary_gate",
        "label_frontier_tag",
    ),
}


def expected_methods(dataset: str) -> tuple[str, ...]:
    require(dataset in DATASET_TARGET_COUNTS, f"unknown C84S dataset: {dataset}")
    if dataset in {"Lee2019_MI", "Cho2017"}:
        return COMMON_METHODS + SECONDARY_Q0
    return COMMON_METHODS


def _finite(value: Any, field: str) -> float:
    result = float(value)
    require(np.isfinite(result), f"nonfinite method-context field: {field}")
    return result


def validate_method_context_rows(
    rows: Sequence[Mapping[str, Any]],
    *, expected_method_provider: Callable[[str], tuple[str, ...]] = expected_methods,
    expected_row_count: int = 18608,
) -> list[dict[str, Any]]:
    """Validate the complete mixed-method table before opening a result root."""
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    groups: dict[tuple[str, str, str, int, int], set[str]] = defaultdict(set)
    targets: dict[str, set[str]] = defaultdict(set)
    expected_fields = set(METHOD_CONTEXT_FIELDS)
    for raw in rows:
        require(set(raw) == expected_fields, "method-context field-set drift")
        row = {field: raw[field] for field in METHOD_CONTEXT_FIELDS}
        row["dataset"] = str(row["dataset"])
        row["target_subject_id"] = str(row["target_subject_id"])
        row["panel"] = str(row["panel"])
        row["training_seed"] = int(row["training_seed"])
        row["level"] = int(row["level"])
        row["method_id"] = str(row["method_id"])
        row["catastrophic_failure"] = int(row["catastrophic_failure"])
        require(row["panel"] in PANELS, "panel identity drift")
        require(row["training_seed"] in SEEDS, "training-seed identity drift")
        require(row["level"] in LEVELS, "level identity drift")
        require(row["method_id"] in expected_method_provider(row["dataset"]), "method identity drift")
        for field in (
            "standardized_regret", "selected_utility",
            "source_relative_regret_gain", "top1", "top5", "top10",
            "coverage", "Spearman", "Kendall", "pairwise_ordering_accuracy",
        ):
            row[field] = _finite(row[field], field)
        mae = row["accuracy_estimation_MAE"]
        row["accuracy_estimation_MAE"] = None if mae in (None, "") else _finite(mae, "accuracy_estimation_MAE")
        require(0.0 <= row["standardized_regret"] <= 1.0 + 1e-12, "standardized regret outside [0,1]")
        require(0.0 <= row["selected_utility"] <= 1.0 + 1e-12, "selected utility outside [0,1]")
        require(0.0 <= row["top1"] <= row["top5"] <= row["top10"] <= 1.0,
                "top-k ordering/value drift")
        require(0.0 <= row["coverage"] <= 1.0, "coverage outside [0,1]")
        require(row["catastrophic_failure"] in (0, 1), "catastrophic-failure flag drift")
        require(-1.0 <= row["Spearman"] <= 1.0 and -1.0 <= row["Kendall"] <= 1.0,
                "rank association outside [-1,1]")
        require(0.0 <= row["pairwise_ordering_accuracy"] <= 1.0,
                "pairwise ordering outside [0,1]")
        identity = tuple(row[field] for field in METHOD_CONTEXT_FIELDS[:6])
        require(identity not in identities, "duplicate method-context row")
        identities.add(identity)
        context = tuple(row[field] for field in METHOD_CONTEXT_FIELDS[:5])
        groups[context].add(row["method_id"])
        targets[row["dataset"]].add(row["target_subject_id"])
        normalized.append(row)

    require(set(targets) == set(DATASET_TARGET_COUNTS), "dataset coverage drift")
    for dataset, count in DATASET_TARGET_COUNTS.items():
        require(len(targets[dataset]) == count, f"target count drift: {dataset}")
        for target in targets[dataset]:
            contexts = {
                (panel, seed, level)
                for ds, current_target, panel, seed, level in groups
                if ds == dataset and current_target == target
            }
            require(
                contexts == {(panel, seed, level) for panel in PANELS for seed in SEEDS for level in LEVELS},
                f"eight-context coverage drift: {dataset}/{target}",
            )
    for context, methods in groups.items():
        require(methods == set(expected_method_provider(context[0])), f"method set drift: {context}")
    expected_rows = sum(
        DATASET_TARGET_COUNTS[dataset] * 8 * len(expected_method_provider(dataset))
        for dataset in DATASET_TARGET_COUNTS
    )
    require(len(normalized) == expected_rows == expected_row_count, "method-context row-count drift")

    lookup = {
        tuple(row[field] for field in METHOD_CONTEXT_FIELDS[:5]) + (row["method_id"],): row
        for row in normalized
    }
    for context in groups:
        source = lookup[context + ("S1",)]["standardized_regret"]
        for method in groups[context]:
            row = lookup[context + (method,)]
            expected_gain = 0.0 if source <= 1e-15 else (source - row["standardized_regret"]) / source
            require(
                abs(row["source_relative_regret_gain"] - expected_gain) <= 1e-10,
                f"source-relative regret-gain drift: {context}/{method}",
            )
        require(abs(lookup[context + ("B5",)]["standardized_regret"]) <= 1e-12,
                "oracle ceiling regret is nonzero")
    return sorted(
        normalized,
        key=lambda row: (
            row["dataset"], row["target_subject_id"], row["panel"],
            row["training_seed"], row["level"], row["method_id"],
        ),
    )


def _category_and_support(decisions: Mapping[str, Mapping[str, Any]]) -> tuple[str, set[str]]:
    a_set, b_set = taxonomy.dataset_method_sets(decisions)
    if a_set:
        return "A", a_set
    if b_set:
        return "B", b_set
    return "C", set()


def _derive_tables(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    blocker: bool,
    expected_method_provider: Callable[[str], tuple[str, ...]] = expected_methods,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    lookup = {
        (
            row["dataset"], row["target_subject_id"], row["panel"],
            row["training_seed"], row["level"], row["method_id"],
        ): row
        for row in rows
    }
    targets = {
        dataset: sorted({row["target_subject_id"] for row in rows if row["dataset"] == dataset})
        for dataset in DATASET_TARGET_COUNTS
    }

    def values(dataset: str, target: str, method: str, level: int | None = None) -> list[dict[str, Any]]:
        levels = LEVELS if level is None else (level,)
        return [
            lookup[(dataset, target, panel, seed, current_level, method)]
            for panel in PANELS for seed in SEEDS for current_level in levels
        ]

    def mean_regret(dataset: str, target: str, method: str, level: int | None = None) -> float:
        return float(np.mean([row["standardized_regret"] for row in values(dataset, target, method, level)]))

    def effects(
        dataset: str,
        method: str,
        target_subset: Sequence[str],
        level: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
        q1 = np.asarray([
            mean_regret(dataset, target, "S1", level) - mean_regret(dataset, target, method, level)
            for target in target_subset
        ])
        q2 = np.asarray([
            mean_regret(dataset, target, method, level) - mean_regret(dataset, target, "Q0_B1", level)
            for target in target_subset
        ])
        selected_levels = LEVELS if level is None else (level,)
        q1_cells, q2_cells = [], []
        for panel, seed in inference.PANEL_SEED_CELLS:
            q1_cells.append(float(np.mean([
                lookup[(dataset, target, panel, seed, current_level, "S1")]["standardized_regret"]
                - lookup[(dataset, target, panel, seed, current_level, method)]["standardized_regret"]
                for target in target_subset for current_level in selected_levels
            ])))
            q2_cells.append(float(np.mean([
                lookup[(dataset, target, panel, seed, current_level, method)]["standardized_regret"]
                - lookup[(dataset, target, panel, seed, current_level, "Q0_B1")]["standardized_regret"]
                for target in target_subset for current_level in selected_levels
            ])))
        return q1, q2, q1_cells, q2_cells

    def decisions_for(
        dataset: str,
        target_subset: Sequence[str],
        level: int | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray, list[float], list[float]]]]:
        payload = {
            method: effects(dataset, method, target_subset, level)
            for method in PRIMARY_METHODS
        }
        decision = inference.dataset_q1_q2(
            dataset=dataset,
            q1_effects={method: payload[method][0] for method in PRIMARY_METHODS},
            q2_excess={method: payload[method][1] for method in PRIMARY_METHODS},
            q1_panel_seed={method: payload[method][2] for method in PRIMARY_METHODS},
            q2_panel_seed={method: payload[method][3] for method in PRIMARY_METHODS},
            draws=draws,
        )
        return decision, payload

    target_level_rows: list[dict[str, Any]] = []
    catastrophic_rows: list[dict[str, Any]] = []
    for dataset in DATASET_TARGET_COUNTS:
        for target in targets[dataset]:
            for method in PRIMARY_METHODS:
                for level in LEVELS:
                    method_regret = mean_regret(dataset, target, method, level)
                    source_regret = mean_regret(dataset, target, "S1", level)
                    q0_regret = mean_regret(dataset, target, "Q0_B1", level)
                    target_level_rows.append({
                        "dataset": dataset, "target_subject_id": target,
                        "method_id": method, "level": level,
                        "method_regret": method_regret,
                        "source_regret": source_regret,
                        "Q0_B1_regret": q0_regret,
                        "Q1_effect": source_regret - method_regret,
                        "Q2_excess": method_regret - q0_regret,
                    })
                    catastrophic_rows.append({
                        "dataset": dataset, "target_subject_id": target,
                        "method_id": method, "level": level,
                        "Q1_catastrophic_floor_breached": int(source_regret - method_regret < -0.10),
                        "Q2_catastrophic_excess_breached": int(method_regret - q0_regret > 0.20),
                    })

    full_decisions: dict[str, dict[str, dict[str, Any]]] = {}
    full_payload: dict[str, dict[str, tuple[np.ndarray, np.ndarray, list[float], list[float]]]] = {}
    level_decisions: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    dataset_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    level_flags_by_dataset: dict[str, dict[str, bool]] = {}
    for dataset in DATASET_TARGET_COUNTS:
        full_decisions[dataset], full_payload[dataset] = decisions_for(dataset, targets[dataset])
        level_decisions[dataset] = {}
        for level in LEVELS:
            level_decisions[dataset][level], _ = decisions_for(dataset, targets[dataset], level)
        level_flags_by_dataset[dataset] = inference.level_heterogeneity(level_decisions[dataset])
        for method in PRIMARY_METHODS:
            row = full_decisions[dataset][method]
            dataset_rows.append({"dataset": dataset, "method_id": method, **row})
            q1_cells, q2_cells = full_payload[dataset][method][2:]
            panel_rows.append({
                "dataset": dataset, "method_id": method,
                **{f"Q1_{panel}{seed}": q1_cells[index] for index, (panel, seed) in enumerate(inference.PANEL_SEED_CELLS)},
                **{f"Q2_{panel}{seed}": q2_cells[index] for index, (panel, seed) in enumerate(inference.PANEL_SEED_CELLS)},
                "Q1_all_directional": int(row["panel_seed_Q1_all_directional"]),
                "Q2_all_within_margin": int(row["panel_seed_Q2_all_within_margin"]),
            })
            for level in LEVELS:
                current = level_decisions[dataset][level][method]
                level_rows.append({
                    "dataset": dataset, "level": level, "method_id": method,
                    "Q1_pass": int(current["Q1_pass"]),
                    "Q1_mean": current["Q1_mean"], "Q1_pvalue": current["Q1_pvalue"],
                    "Q2_pass": int(current["Q2_pass"]),
                    "Q2_mean_excess": current["Q2_mean_excess"],
                    "Q2_pvalue": current["Q2_pvalue"],
                    "level_heterogeneity": int(level_flags_by_dataset[dataset][method]),
                })

    budget_methods: dict[int | str, str] = {
        1: "Q0_B1", 2: "Q0_B2", 4: "Q0_B4", 8: "Q0_B8", "FULL": "Q0_FULL",
    }

    def budget_curve(
        dataset: str,
        target_subset: Sequence[str],
        level: int | None = None,
    ) -> tuple[dict[str, Any], dict[int | str, np.ndarray]]:
        selected_levels = LEVELS if level is None else (level,)
        by_budget: dict[int | str, np.ndarray] = {}
        cells: dict[int | str, list[float]] = {}
        for budget, method in budget_methods.items():
            by_budget[budget] = np.asarray([
                mean_regret(dataset, target, "S1", level) - mean_regret(dataset, target, method, level)
                for target in target_subset
            ])
            cells[budget] = [float(np.mean([
                lookup[(dataset, target, panel, seed, current_level, "S1")]["standardized_regret"]
                - lookup[(dataset, target, panel, seed, current_level, method)]["standardized_regret"]
                for target in target_subset for current_level in selected_levels
            ])) for panel, seed in inference.PANEL_SEED_CELLS]
        return inference.qualify_budget_curve(
            by_budget, cells, dataset=dataset, draws=draws,
        ), by_budget

    full_budget: dict[str, dict[str, Any]] = {}
    level_budget: dict[str, dict[int, dict[str, Any]]] = {}
    loto_budget: dict[str, dict[str, int | str | None]] = {}
    budget_context_rows: list[dict[str, Any]] = []
    for dataset in DATASET_TARGET_COUNTS:
        full_budget[dataset], target_budget_effects = budget_curve(dataset, targets[dataset])
        level_budget[dataset] = {
            level: budget_curve(dataset, targets[dataset], level)[0]
            for level in LEVELS
        }
        loto_budget[dataset] = {}
        for target in targets[dataset]:
            remaining = [value for value in targets[dataset] if value != target]
            loto_budget[dataset][target] = budget_curve(dataset, remaining)[0]["Bstar"]
        for target_index, target in enumerate(targets[dataset]):
            for budget in budget_methods:
                level0 = (
                    mean_regret(dataset, target, "S1", 0)
                    - mean_regret(dataset, target, budget_methods[budget], 0)
                )
                level1 = (
                    mean_regret(dataset, target, "S1", 1)
                    - mean_regret(dataset, target, budget_methods[budget], 1)
                )
                budget_context_rows.append({
                    "dataset": dataset, "target_subject_id": target,
                    "budget": str(budget),
                    "primary_effect": float(target_budget_effects[budget][target_index]),
                    "level0_effect": level0, "level1_effect": level1,
                })

    loto_rows: list[dict[str, Any]] = []
    loto_preserved: dict[str, list[str]] = {}
    label_heterogeneity = False
    label_stability: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_TARGET_COUNTS:
        full_category, full_support = _category_and_support(full_decisions[dataset])
        omitted_sets: list[list[str]] = []
        current_rows: list[dict[str, Any]] = []
        full_bstar = full_budget[dataset]["Bstar"]
        for target in targets[dataset]:
            remaining = [value for value in targets[dataset] if value != target]
            omitted_decisions, _ = decisions_for(dataset, remaining)
            omitted_category, omitted_support = _category_and_support(omitted_decisions)
            omitted_sets.append(sorted(omitted_support))
            same_method = (
                not omitted_support if full_category == "C"
                else bool(full_support & omitted_support)
            )
            omitted_bstar = loto_budget[dataset][target]
            current_rows.append({
                "dataset": dataset, "left_out_target": target,
                "full_category": full_category, "LOTO_category": omitted_category,
                "full_supporting_methods": "|".join(sorted(full_support)) or "NONE",
                "LOTO_supporting_methods": "|".join(sorted(omitted_support)) or "NONE",
                "same_method_preserved": int(same_method),
                "label_full_Bstar": "NONE" if full_bstar is None else str(full_bstar),
                "label_LOTO_Bstar": "NONE" if omitted_bstar is None else str(omitted_bstar),
                "label_Bstar_preserved": int(omitted_bstar == full_bstar),
            })
        preservation = inference.loto_preservation(
            dataset=dataset, full_supporting_methods=sorted(full_support),
            omitted_panel_method_sets=omitted_sets, full_category=full_category,
        )
        loto_preserved[dataset] = list(preservation["preserved_methods"])
        minimum = preservation["minimum"]
        label_count = sum(row["label_Bstar_preserved"] for row in current_rows)
        level_mismatch = level_budget[dataset][0]["Bstar"] != level_budget[dataset][1]["Bstar"]
        label_stability[dataset] = {
            "preserved": label_count, "minimum": minimum,
            "level_mismatch": level_mismatch,
        }
        label_heterogeneity = label_heterogeneity or level_mismatch or label_count < minimum
        loto_rows.extend(current_rows)

    global_level_flags = {
        method: any(level_flags_by_dataset[dataset][method] for dataset in DATASET_TARGET_COUNTS)
        for method in PRIMARY_METHODS
    }
    hidden_level_q1 = any(
        level_decisions[dataset][level][method]["Q1_pass"]
        and not full_decisions[dataset][method]["Q1_pass"]
        for dataset in DATASET_TARGET_COUNTS for level in LEVELS for method in PRIMARY_METHODS
    )
    final_taxonomy = taxonomy.classify_c84(
        dataset_decisions=full_decisions,
        level_heterogeneity=global_level_flags,
        loto_preserved_methods=loto_preserved,
        blocker=blocker,
        hidden_level_q1_pass=hidden_level_q1,
    )
    bstars = {dataset: full_budget[dataset]["Bstar"] for dataset in DATASET_TARGET_COUNTS}
    label_tag = taxonomy.classify_label_frontier(
        bstars, registered_heterogeneity=label_heterogeneity,
    )

    budget_frontier_rows: list[dict[str, Any]] = []
    for dataset in DATASET_TARGET_COUNTS:
        curve = full_budget[dataset]
        for index, budget in enumerate(curve["budgets"]):
            budget_frontier_rows.append({
                "dataset": dataset, "budget": str(budget),
                "mean_effect": curve["means"][index],
                "maxT_pvalue": curve["pvalues"][index],
                "direct_qualification": int(curve["direct"][index]),
                "closure_qualification": int(curve["closure"][index]),
                "Bstar": "NONE" if curve["Bstar"] is None else str(curve["Bstar"]),
                "level0_Bstar": "NONE" if level_budget[dataset][0]["Bstar"] is None else str(level_budget[dataset][0]["Bstar"]),
                "level1_Bstar": "NONE" if level_budget[dataset][1]["Bstar"] is None else str(level_budget[dataset][1]["Bstar"]),
                "LOTO_preserved": label_stability[dataset]["preserved"],
                "LOTO_minimum": label_stability[dataset]["minimum"],
                "registered_heterogeneity": int(
                    label_stability[dataset]["level_mismatch"]
                    or label_stability[dataset]["preserved"] < label_stability[dataset]["minimum"]
                ),
            })

    q1_lookup = {
        (row["dataset"], row["method_id"]): int(row["Q1_pass"])
        for row in dataset_rows
    }

    def summary_rows(table: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for dataset in DATASET_TARGET_COUNTS:
            for method in expected_method_provider(dataset):
                method_rows = [
                    row for row in rows
                    if row["dataset"] == dataset and row["method_id"] == method
                ]
                if table == "topk":
                    output.append({
                        "dataset": dataset, "method_id": method,
                        "mean_top1": float(np.mean([row["top1"] for row in method_rows])),
                        "mean_top5": float(np.mean([row["top5"] for row in method_rows])),
                        "mean_top10": float(np.mean([row["top10"] for row in method_rows])),
                        "Q1_pass": q1_lookup.get((dataset, method), 0),
                        "endpoint_substitutes_for_regret": 0,
                    })
                elif table == "utility":
                    output.append({
                        "dataset": dataset, "method_id": method,
                        "mean_selected_utility": float(np.mean([row["selected_utility"] for row in method_rows])),
                        "mean_regret": float(np.mean([row["standardized_regret"] for row in method_rows])),
                    })
                elif table == "gain":
                    output.append({
                        "dataset": dataset, "method_id": method,
                        "mean_source_relative_regret_gain": float(np.mean([
                            row["source_relative_regret_gain"] for row in method_rows
                        ])),
                    })
                else:
                    maes = [row["accuracy_estimation_MAE"] for row in method_rows if row["accuracy_estimation_MAE"] is not None]
                    output.append({
                        "dataset": dataset, "method_id": method,
                        "mean_Spearman": float(np.mean([row["Spearman"] for row in method_rows])),
                        "mean_Kendall": float(np.mean([row["Kendall"] for row in method_rows])),
                        "mean_pairwise_ordering_accuracy": float(np.mean([
                            row["pairwise_ordering_accuracy"] for row in method_rows
                        ])),
                        "accuracy_estimation_MAE": None if not maes else float(np.mean(maes)),
                        "Q1_pass": q1_lookup.get((dataset, method), 0),
                        "measurement_substitutes_for_regret": 0,
                    })
        return output

    intersection_rows: list[dict[str, Any]] = []
    for set_id, mapping in (("A", final_taxonomy["A_sets"]), ("B", final_taxonomy["B_sets"])):
        for dataset in taxonomy.DATASETS:
            methods = mapping[dataset]
            intersection_rows.append({
                "set_id": f"{set_id}_{dataset}", "dataset": dataset,
                "methods": "|".join(methods) or "NONE", "count": len(methods),
                "primary_gate": final_taxonomy["gate"], "label_frontier_tag": label_tag,
            })
    for set_id, methods in (
        ("A_intersection", final_taxonomy["A_intersection"]),
        ("B_intersection", final_taxonomy["B_intersection"]),
    ):
        intersection_rows.append({
            "set_id": set_id, "dataset": "ALL",
            "methods": "|".join(methods) or "NONE", "count": len(methods),
            "primary_gate": final_taxonomy["gate"], "label_frontier_tag": label_tag,
        })

    coverage_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for dataset in DATASET_TARGET_COUNTS:
        for method in expected_method_provider(dataset):
            method_rows = [row for row in rows if row["dataset"] == dataset and row["method_id"] == method]
            coverage_rows.append({
                "dataset": dataset, "method_id": method,
                "mean_coverage": float(np.mean([row["coverage"] for row in method_rows])),
            })
            counts: dict[str, int] = defaultdict(int)
            for row in method_rows:
                counts[str(row["selected_regime"])] += 1
            for regime in sorted(counts):
                regime_rows.append({
                    "dataset": dataset, "method_id": method,
                    "selected_regime": regime, "contexts": counts[regime],
                    "fraction": counts[regime] / len(method_rows),
                })

    tables = {
        "method_context_decisions.csv": [dict(row) for row in rows],
        "target_level_method_effects.csv": target_level_rows,
        "target_level_catastrophic_failures.csv": catastrophic_rows,
        "dataset_Q1_Q2.csv": dataset_rows,
        "level_specific_Q1_Q2.csv": level_rows,
        "panel_seed_stability.csv": panel_rows,
        "leave_one_target_out.csv": loto_rows,
        "label_budget_frontier.csv": budget_frontier_rows,
        "label_budget_context.csv": budget_context_rows,
        "topk_decision_summary.csv": summary_rows("topk"),
        "selected_utility_summary.csv": summary_rows("utility"),
        "coverage_summary.csv": coverage_rows,
        "selected_regime_distribution.csv": regime_rows,
        "source_relative_regret_gain.csv": summary_rows("gain"),
        "measurement_vs_decision.csv": summary_rows("measurement"),
        "cross_dataset_method_intersection.csv": intersection_rows,
    }
    result = {
        "schema_version": "c84s_multidataset_scientific_result_v1",
        "primary_gate": final_taxonomy["gate"],
        "label_frontier_tag": label_tag,
        "dataset_categories": final_taxonomy["dataset_categories"],
        "A_intersection": final_taxonomy["A_intersection"],
        "B_intersection": final_taxonomy["B_intersection"],
        "Bstar": {dataset: "NONE" if value is None else str(value) for dataset, value in bstars.items()},
        "LEVEL_HETEROGENEITY": final_taxonomy["level_heterogeneity"],
        "hidden_level_Q1_pass": final_taxonomy["hidden_level_Q1_pass"],
        "target_counts": DATASET_TARGET_COUNTS,
        "method_context_rows": len(rows),
        "maxT_draws": draws,
        "target_construction_labels_outside_selection": 0,
        "same_label_oracle": 0,
        "training": 0,
        "forward": 0,
        "GPU": 0,
        "C85_authorized": False,
    }
    return tables, result


def _validate_table_schemas(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    require(set(tables) == set(RESULT_TABLE_FIELDS), "Stage-C result table registry drift")
    for name, fields in RESULT_TABLE_FIELDS.items():
        rows = tables[name]
        require(rows, f"empty Stage-C result table: {name}")
        require(all(tuple(row) == fields for row in rows), f"canonical table field order drift: {name}")


def run_analysis_and_freeze(
    method_context_rows: Sequence[Mapping[str, Any]],
    *,
    selection_freeze_identity: Mapping[str, Any],
    evaluation_view_identity: Mapping[str, Any],
    final_root: str | Path,
    draws: int = inference.MAXT_DRAWS,
    blocker: bool = False,
    synthetic: bool = False,
    failure_injection_after: str | None = None,
) -> dict[str, Any]:
    """Validate all rows in memory and atomically publish every Stage-C output."""
    require(draws >= 1, "max-T draw count is invalid")
    require(
        selection_freeze_identity.get("status")
        == "SELECTION_FROZEN_EVALUATION_DESCRIPTOR_NOT_YET_AVAILABLE",
        "held evaluation lacks an immutable selection freeze",
    )
    require(len(str(selection_freeze_identity.get("sha256", ""))) == 64,
            "selection-freeze identity drift")
    require(evaluation_view_identity.get("kind") == "evaluation",
            "held-evaluation process received a non-evaluation view")
    require(len(str(evaluation_view_identity.get("manifest_sha256", ""))) == 64,
            "evaluation-view identity drift")
    normalized = validate_method_context_rows(method_context_rows)
    tables, result = _derive_tables(normalized, draws=draws, blocker=blocker)
    _validate_table_schemas(tables)
    result.update({
        "selection_freeze_sha256": str(selection_freeze_identity["sha256"]),
        "evaluation_view_manifest_sha256": str(evaluation_view_identity["manifest_sha256"]),
        "synthetic": bool(synthetic),
    })

    def writer(staging: Path) -> None:
        artifacts: list[dict[str, Any]] = []
        for name in RESULT_TABLE_FIELDS:
            digest = write_csv(staging / name, tables[name])
            artifacts.append({"path": name, "rows": len(tables[name]), "sha256": digest})
            if failure_injection_after == name:
                raise C84SContractError("injected Stage-C result-freeze failure")
        manifest = {
            "schema_version": "c84s_result_artifact_manifest_v1",
            "selection_freeze_sha256": str(selection_freeze_identity["sha256"]),
            "evaluation_view_manifest_sha256": str(evaluation_view_identity["manifest_sha256"]),
            "table_count": len(artifacts), "artifacts": artifacts,
            "all_tables_validated_before_publication": True,
        }
        manifest_sha = write_json(staging / "C84S_RESULT_ARTIFACT_MANIFEST.json", manifest)
        replayed = read_json(staging / "C84S_RESULT_ARTIFACT_MANIFEST.json")
        require(replayed == manifest, "result artifact manifest replay drift")
        for identity in artifacts:
            require(sha256_file(staging / identity["path"]) == identity["sha256"],
                    f"result artifact hash drift: {identity['path']}")
        if failure_injection_after == "C84S_RESULT_ARTIFACT_MANIFEST.json":
            raise C84SContractError("injected post-manifest result-freeze failure")
        final_result = {
            **result,
            "artifact_manifest_sha256": manifest_sha,
            "artifact_manifest_table_count": len(artifacts),
            "result_identity_sha256": canonical_sha256({
                "result": result, "artifact_manifest_sha256": manifest_sha,
            }),
        }
        write_json(staging / "C84S_RESULT.json", final_result)

    published = atomic_publish_directory(final_root, writer)
    final_result = read_json(published / "C84S_RESULT.json")
    require(
        sha256_file(published / "C84S_RESULT_ARTIFACT_MANIFEST.json")
        == final_result["artifact_manifest_sha256"],
        "published result/manifest identity drift",
    )
    return final_result
