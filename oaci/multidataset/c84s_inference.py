"""Target-cluster aggregation, max-T, Q1/Q2, LOTO, and label frontier."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .c84s_common import digest_low64, require


PRIMARY_METHODS = ("U5", "U7", "U11", "U13", "U14", "U15")
MATERIAL_MARGIN = 0.05
NONINFERIORITY_MARGIN = 0.05
MAXT_DRAWS = 65536
PANEL_SEED_CELLS = (("A", 5), ("A", 6), ("B", 5), ("B", 6))


def _studentized_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    result = np.divide(mean, scale, out=np.zeros_like(mean), where=scale > 1e-15)
    degenerate = scale <= 1e-15
    result[degenerate & (mean > 0)] = np.inf
    result[degenerate & (mean < 0)] = -np.inf
    return result


def rademacher_maxT(
    effects: np.ndarray,
    *,
    dataset: str,
    family: str,
    null_margin: float = 0.0,
    draws: int = MAXT_DRAWS,
    batch_size: int = 2048,
) -> dict[str, np.ndarray | float | int]:
    """Locked one-sided shared-target Rademacher max-T and mean band."""
    values = np.asarray(effects, dtype=float)
    require(values.ndim == 2 and values.shape[0] >= 3 and values.shape[1] >= 1,
            "max-T effect matrix shape drift")
    require(np.all(np.isfinite(values)) and draws >= 1, "max-T input/draw drift")
    shifted = values - float(null_margin)
    observed = _studentized_mean(shifted)
    centered = values - np.mean(values, axis=0, keepdims=True)
    rng = np.random.Generator(np.random.PCG64(digest_low64(f"C84_MAXT_V1|{dataset}|{family}")))
    exceedances = np.zeros(values.shape[1], dtype=np.int64)
    null_mean_max = np.empty(draws, dtype=float)
    cursor = 0
    while cursor < draws:
        count = min(batch_size, draws - cursor)
        signs = rng.integers(0, 2, size=(count, values.shape[0]), dtype=np.int8) * 2 - 1
        signed_shifted = signs[:, :, None] * shifted[None, :, :]
        mean = np.mean(signed_shifted, axis=1)
        scale = np.std(signed_shifted, axis=1, ddof=1) / math.sqrt(values.shape[0])
        statistic = np.divide(mean, scale, out=np.zeros_like(mean), where=scale > 1e-15)
        degenerate = scale <= 1e-15
        statistic[degenerate & (mean > 0)] = np.inf
        statistic[degenerate & (mean < 0)] = -np.inf
        max_stat = np.max(statistic, axis=1)
        exceedances += np.sum(max_stat[:, None] >= observed[None, :] - 1e-15, axis=0)
        centered_mean = np.mean(signs[:, :, None] * centered[None, :, :], axis=1)
        null_mean_max[cursor:cursor + count] = np.max(centered_mean, axis=1)
        cursor += count
    critical = float(np.quantile(null_mean_max, 0.95, method="higher"))
    mean_effect = np.mean(values, axis=0)
    return {
        "mean": mean_effect,
        "pvalue": (1.0 + exceedances) / (1.0 + draws),
        "lower": mean_effect - critical,
        "upper": mean_effect + critical,
        "critical": critical,
        "draws": int(draws),
        "seed": int(digest_low64(f"C84_MAXT_V1|{dataset}|{family}")),
    }


def aggregate_context_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
) -> dict[str, Any]:
    """Validate eight equal-weight contexts and return registered summaries."""
    require(len(rows) == 8, "target aggregate requires exactly eight contexts")
    keys = {
        (str(row["panel"]), int(row["training_seed"]), int(row["level"]))
        for row in rows
    }
    expected = {(panel, seed, level) for panel, seed in PANEL_SEED_CELLS for level in (0, 1)}
    require(keys == expected, "target context coverage drift")
    values = {(str(row["panel"]), int(row["training_seed"]), int(row["level"])): float(row[value_field]) for row in rows}
    require(all(np.isfinite(value) for value in values.values()), "context value is nonfinite")
    level = {
        current: float(np.mean([values[(panel, seed, current)] for panel, seed in PANEL_SEED_CELLS]))
        for current in (0, 1)
    }
    panel_seed = {
        f"{panel}{seed}": float(np.mean([values[(panel, seed, current)] for current in (0, 1)]))
        for panel, seed in PANEL_SEED_CELLS
    }
    return {
        "primary": float(np.mean(list(values.values()))),
        "level": level,
        "panel_seed": panel_seed,
    }


def q1_pass(
    effects: np.ndarray, pvalue: float, panel_seed_effects: Sequence[float],
) -> bool:
    values = np.asarray(effects, dtype=float)
    cells = np.asarray(panel_seed_effects, dtype=float)
    return bool(
        np.mean(values) >= MATERIAL_MARGIN
        and pvalue <= 0.05
        and np.mean(values > 0.0) >= 0.75
        and np.min(values) >= -0.10
        and np.sum(cells > 0.0) >= 3
    )


def q2_pass(
    excess: np.ndarray, pvalue: float, simultaneous_upper: float,
    panel_seed_excess: Sequence[float],
) -> bool:
    values = np.asarray(excess, dtype=float)
    cells = np.asarray(panel_seed_excess, dtype=float)
    return bool(
        np.mean(values) <= NONINFERIORITY_MARGIN
        and simultaneous_upper <= NONINFERIORITY_MARGIN
        and pvalue <= 0.05
        and np.mean(values <= NONINFERIORITY_MARGIN) >= 0.75
        and np.max(values) <= 0.20
        and np.sum(cells <= NONINFERIORITY_MARGIN) >= 3
    )


def dataset_q1_q2(
    *,
    dataset: str,
    q1_effects: Mapping[str, Sequence[float]],
    q2_excess: Mapping[str, Sequence[float]],
    q1_panel_seed: Mapping[str, Sequence[float]],
    q2_panel_seed: Mapping[str, Sequence[float]],
    draws: int = MAXT_DRAWS,
) -> dict[str, dict[str, Any]]:
    require(set(q1_effects) == set(q2_excess) == set(PRIMARY_METHODS), "Q1/Q2 method family drift")
    q1_matrix = np.column_stack([np.asarray(q1_effects[method], dtype=float) for method in PRIMARY_METHODS])
    q2_matrix = np.column_stack([np.asarray(q2_excess[method], dtype=float) for method in PRIMARY_METHODS])
    require(q1_matrix.shape == q2_matrix.shape, "Q1/Q2 target coverage mismatch")
    q1_test = rademacher_maxT(q1_matrix, dataset=dataset, family="Q1_ZERO_LABEL", null_margin=MATERIAL_MARGIN, draws=draws)
    q2_test = rademacher_maxT(NONINFERIORITY_MARGIN - q2_matrix, dataset=dataset, family="Q2_ZERO_LABEL", draws=draws)
    q2_band = rademacher_maxT(q2_matrix, dataset=dataset, family="Q2_EXCESS_BAND", draws=draws)
    output: dict[str, dict[str, Any]] = {}
    for index, method in enumerate(PRIMARY_METHODS):
        q1_values = q1_matrix[:, index]
        q2_values = q2_matrix[:, index]
        q1_cells = np.asarray(q1_panel_seed[method], dtype=float)
        q2_cells = np.asarray(q2_panel_seed[method], dtype=float)
        require(q1_cells.shape == q2_cells.shape == (4,), "panel/seed cell effect shape drift")
        pass_q1 = q1_pass(q1_values, float(q1_test["pvalue"][index]), q1_cells)
        pass_q2 = q2_pass(q2_values, float(q2_test["pvalue"][index]), float(q2_band["upper"][index]), q2_cells)
        output[method] = {
            "Q1_pass": pass_q1,
            "Q1_mean": float(np.mean(q1_values)),
            "Q1_pvalue": float(q1_test["pvalue"][index]),
            "Q1_favorable_targets": int(np.sum(q1_values > 0)),
            "Q1_worst_target": float(np.min(q1_values)),
            "Q2_pass": pass_q2,
            "Q2_mean_excess": float(np.mean(q2_values)),
            "Q2_simultaneous_upper": float(q2_band["upper"][index]),
            "Q2_pvalue": float(q2_test["pvalue"][index]),
            "Q2_within_margin_targets": int(np.sum(q2_values <= NONINFERIORITY_MARGIN)),
            "Q2_worst_excess": float(np.max(q2_values)),
            "panel_seed_Q1_all_directional": bool(np.all(q1_cells > 0)),
            "panel_seed_Q2_all_within_margin": bool(np.all(q2_cells <= NONINFERIORITY_MARGIN)),
        }
    return output


def level_heterogeneity(level_decisions: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> dict[str, bool]:
    require(set(level_decisions) == {0, 1}, "level-decision registry drift")
    output: dict[str, bool] = {}
    for method in PRIMARY_METHODS:
        left, right = level_decisions[0][method], level_decisions[1][method]
        output[method] = bool(
            bool(left["Q1_pass"]) != bool(right["Q1_pass"])
            or bool(left["Q2_pass"]) != bool(right["Q2_pass"])
            or np.sign(float(left["Q1_mean"])) != np.sign(float(right["Q1_mean"]))
            or np.sign(float(left["Q2_mean_excess"])) != np.sign(float(right["Q2_mean_excess"]))
        )
    return output


def loto_preservation(
    *,
    dataset: str,
    full_supporting_methods: Sequence[str],
    omitted_panel_method_sets: Sequence[Sequence[str]],
    full_category: str = "B",
) -> dict[str, Any]:
    totals = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
    minima = {"Lee2019_MI": 17, "Cho2017": 15, "PhysionetMI": 57}
    require(dataset in totals and len(omitted_panel_method_sets) == totals[dataset], "LOTO panel count drift")
    require(full_category in {"A", "B", "C"}, "LOTO full category drift")
    fixed = set(full_supporting_methods)
    if full_category == "C":
        require(not fixed, "stable-C LOTO must not register a supporting Q1 method")
        per_method = {"NO_Q1": sum(len(set(panel)) == 0 for panel in omitted_panel_method_sets)}
    else:
        require(fixed, "LOTO supporting-method set is empty")
        per_method = {
            method: sum(method in set(panel) for panel in omitted_panel_method_sets)
            for method in sorted(fixed)
        }
    preserved_methods = [method for method, count in per_method.items() if count >= minima[dataset]]
    return {
        "dataset": dataset, "total": totals[dataset], "minimum": minima[dataset],
        "per_method": per_method, "preserved_methods": preserved_methods,
        "pass": bool(preserved_methods),
    }


def qualify_budget_curve(
    effects_by_budget: Mapping[int | str, Sequence[float]],
    panel_seed_by_budget: Mapping[int | str, Sequence[float]],
    *,
    dataset: str,
    draws: int = MAXT_DRAWS,
) -> dict[str, Any]:
    budgets: tuple[int | str, ...] = (1, 2, 4, 8, "FULL")
    require(tuple(effects_by_budget) == budgets and tuple(panel_seed_by_budget) == budgets,
            "primary budget order/grid drift")
    matrix = np.column_stack([np.asarray(effects_by_budget[budget], dtype=float) for budget in budgets])
    test = rademacher_maxT(matrix, dataset=dataset, family="Q0_PRIMARY_BUDGET", null_margin=MATERIAL_MARGIN, draws=draws)
    direct: list[bool] = []
    for index, budget in enumerate(budgets):
        values = matrix[:, index]
        cells = np.asarray(panel_seed_by_budget[budget], dtype=float)
        require(cells.shape == (4,), "budget panel/seed cell shape drift")
        direct.append(bool(
            np.mean(values) >= MATERIAL_MARGIN
            and float(test["pvalue"][index]) <= 0.05
            and np.mean(values > 0) >= 0.75
            and np.min(values) >= -0.10
            and np.sum(cells > 0) >= 3
        ))
    closure = [all(direct[index:]) for index in range(len(budgets))]
    bstar = next((budget for budget, passed in zip(budgets, closure) if passed), None)
    return {
        "budgets": budgets, "direct": direct, "closure": closure, "Bstar": bstar,
        "means": np.mean(matrix, axis=0).tolist(),
        "pvalues": np.asarray(test["pvalue"]).tolist(),
    }
