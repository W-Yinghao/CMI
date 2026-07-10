"""Statistical robustness helpers for C73.

All functions are retrospective diagnostics over already-consumed C69/C71
caches.  They emit no checkpoint identifiers and define no action rule.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import math

import numpy as np

from . import c70_split_label_information_budget as c70
from . import c72_measurement_control_gap as c72


COMPONENTS = ("N", "E", "U", "S", "G")
COMPONENT_NAMES = {
    "N": "finite_label_noise",
    "E": "extreme_order_geometry",
    "U": "utility_mismatch",
    "S": "shared_target_calibration",
    "G": "candidate_specific_unexplained_residual",
}
ENDPOINTS = ("bAcc", "NLL", "ECE", "continuous_joint_utility", "primary_joint_good")


def endpoint_vectors(endpoint_data: dict, endpoint: str) -> tuple[np.ndarray, np.ndarray]:
    construct = endpoint_data["construct_metrics"]
    evaluation = endpoint_data["eval_metrics"]
    if endpoint == "bAcc":
        return endpoint_data["construct_bacc"], endpoint_data["eval_bacc"]
    if endpoint == "NLL":
        return -np.asarray([r["NLL"] for r in construct]), -np.asarray([r["NLL"] for r in evaluation])
    if endpoint == "ECE":
        return -np.asarray([r["ECE"] for r in construct]), -np.asarray([r["ECE"] for r in evaluation])
    if endpoint == "continuous_joint_utility":
        return endpoint_data["construct_joint"], endpoint_data["eval_joint"]
    if endpoint == "primary_joint_good":
        return endpoint_data["construct_joint"], endpoint_data["primary_joint_good"].astype(float)
    raise ValueError(endpoint)


def shared_shifted_endpoint_vectors(
    populations: dict[str, c72.TargetData],
    endpoints: dict[str, dict],
    shared_lock: str,
) -> dict[str, dict[str, np.ndarray]]:
    template, magnitude = shared_lock.split("|")
    out: dict[str, dict[str, np.ndarray]] = {}
    for target, pop in populations.items():
        assert pop.labels is not None
        idx = pop.indices("target_construct")
        vector = c72._shared_template(pop, template, float(magnitude))
        metrics = [
            c72._endpoint_metrics(unit.logits.astype(float) + vector[None, :], pop.labels, idx, pop.classes)
            for unit in pop.units
        ]
        oriented = np.column_stack([
            c72._midrank_percentile(np.asarray([r["bAcc"] for r in metrics])),
            c72._midrank_percentile(-np.asarray([r["NLL"] for r in metrics])),
            c72._midrank_percentile(-np.asarray([r["ECE"] for r in metrics])),
        ])
        out[target] = {
            "bAcc": np.asarray([r["bAcc"] for r in metrics]),
            "NLL": -np.asarray([r["NLL"] for r in metrics]),
            "ECE": -np.asarray([r["ECE"] for r in metrics]),
            "continuous_joint_utility": np.mean(oriented, axis=1),
            "primary_joint_good": np.mean(oriented, axis=1),
        }
    return out


def epsilon_and_temperature_from_t2(t2_pop: dict[str, c72.TargetData], t2_end: dict[str, dict]) -> tuple[list[float], float]:
    gaps = []
    for target in sorted(t2_pop, key=int):
        utility = np.sort(np.asarray(t2_end[target]["eval_bacc"], dtype=float))
        if len(utility) >= 2:
            gaps.append(float(utility[-1] - utility[-2]))
    positive = [g for g in gaps if g > 1e-12]
    base = positive if positive else gaps
    quantiles = [float(np.quantile(base, q)) for q in (0.25, 0.50, 0.75)]
    eps = sorted({0.0, 0.02, *[round(x, 12) for x in quantiles]})
    temperature = float(np.median(base)) if base else 0.01
    return eps, max(temperature, 1e-6)


def effective_multiplicity(utility: np.ndarray, temperature: float) -> float:
    u = np.asarray(utility, dtype=float)
    weights = np.exp(np.clip((u - float(np.max(u))) / max(temperature, 1e-12), -80.0, 0.0))
    weights /= float(np.sum(weights))
    return float(1.0 / np.sum(weights ** 2))


def build_candidate_context(
    stage: str,
    populations: dict[str, c72.TargetData],
    endpoints: dict[str, dict],
    epsilons: list[float],
    temperature: float,
) -> dict[str, list[dict]]:
    random_rows, effective_rows, topk_rows, near_rows = [], [], [], []
    for target, pop in populations.items():
        utility_all = np.asarray(endpoints[target]["eval_bacc"], dtype=float)
        score_all = np.asarray(endpoints[target]["construct_bacc"], dtype=float)
        good_all = np.asarray(endpoints[target]["primary_joint_good"], dtype=int)
        for level in ("target_universe", "trajectory_cell"):
            for field_index, (trajectory, units) in enumerate(c72._field_groups(pop, level)):
                idx = c72._unit_indices(pop, units)
                if len(idx) < 2:
                    continue
                utility, score, good = utility_all[idx], score_all[idx], good_all[idx]
                m = len(idx)
                sorted_u = np.sort(utility)[::-1]
                top1, top3, regret = c72._top_metrics(score, utility, min(3, m))
                eff = effective_multiplicity(utility, temperature)
                trajectory_label = trajectory if level == "trajectory_cell" else "ALL_TARGET_CANDIDATES"
                random_rows.append({
                    "stage": stage, "field_level": level, "target_id": target,
                    "trajectory": trajectory_label, "field_index": field_index,
                    "candidate_count": m, "random_top1": 1.0 / m,
                    "random_top3": min(3, m) / m,
                    "random_top5": min(5, m) / m,
                    "random_top10pct": max(1, math.ceil(0.1 * m)) / m,
                    "joint_good_prevalence": float(np.mean(good)),
                    "observed_top1": top1, "observed_top3": top3,
                    "observed_top1_enrichment": top1 * m,
                    "reliable_control_inferred": 0,
                })
                effective_rows.append({
                    "stage": stage, "field_level": level, "target_id": target,
                    "trajectory": trajectory_label, "field_index": field_index,
                    "raw_candidate_count": m, "effective_candidate_multiplicity": eff,
                    "effective_to_raw_ratio": eff / m,
                    "temperature_from_T2": temperature,
                    "best_minus_second": float(sorted_u[0] - sorted_u[1]),
                    "best_minus_third": float(sorted_u[0] - sorted_u[min(2, m - 1)]),
                    "utility_iqr": float(np.quantile(utility, 0.75) - np.quantile(utility, 0.25)),
                    "utility_range": float(np.max(utility) - np.min(utility)),
                })
                for k_name, k in (("top1", 1), ("top3", 3), ("top5", 5), ("top10pct", max(1, math.ceil(0.1 * m)))):
                    k = min(k, m)
                    if k == 1:
                        hit = top1
                    else:
                        _, hit, _ = c72._top_metrics(score, utility, k)
                    topk_rows.append({
                        "stage": stage, "field_level": level, "target_id": target,
                        "trajectory": trajectory_label, "candidate_count": m,
                        "k_definition": k_name, "k": k, "observed_hit": hit,
                        "random_hit": k / m, "enrichment": hit / max(k / m, 1e-12),
                        "continuous_regret": regret,
                        "regret_to_utility_range": regret / max(float(np.ptp(utility)), 1e-12),
                    })
                for epsilon in epsilons:
                    count = int(np.sum(utility >= float(np.max(utility)) - epsilon - 1e-12))
                    near_rows.append({
                        "stage": stage, "field_level": level, "target_id": target,
                        "trajectory": trajectory_label, "candidate_count": m,
                        "epsilon": epsilon, "epsilon_source": "T2_locked_rule",
                        "epsilon_optimal_count": count,
                        "epsilon_optimal_fraction": count / m,
                        "effective_candidate_multiplicity": eff,
                    })
    return {
        "cell_specific_random_baselines_rows": random_rows,
        "effective_candidate_multiplicity_rows": effective_rows,
        "topk_and_regret_context_rows": topk_rows,
        "near_tie_set_size_rows": near_rows,
    }


def _pair_reduced_top1(score: np.ndarray, utility: np.ndarray) -> float:
    return c72._pair_reduced_top1(np.asarray(score, dtype=float), np.asarray(utility, dtype=float))


def _value(
    score_rows: np.ndarray,
    matched_score: np.ndarray,
    shared_base_score: np.ndarray,
    shared_matched_score: np.ndarray,
    utility: np.ndarray,
    removed: frozenset[str],
) -> float:
    if "G" in removed:
        candidates = np.asarray(utility, dtype=float)[None, :]
    else:
        if "U" in removed:
            selected = shared_matched_score if "S" in removed else matched_score
        else:
            if "S" in removed:
                selected = shared_base_score
            elif "N" in removed:
                selected = np.mean(score_rows, axis=0)
            else:
                selected = None
        candidates = np.asarray(selected, dtype=float)[None, :] if selected is not None else score_rows
    vals = []
    for score in candidates:
        vals.append(_pair_reduced_top1(score, utility) if "E" in removed else c72._top_metrics(score, utility, 1)[0])
    return float(np.mean(vals))


def _target_attribution(
    score_rows: np.ndarray,
    matched_score: np.ndarray,
    shared_base_score: np.ndarray,
    shared_matched_score: np.ndarray,
    utility: np.ndarray,
) -> tuple[dict[frozenset[str], float], dict[str, float], list[dict]]:
    values: dict[frozenset[str], float] = {}
    for n in range(len(COMPONENTS) + 1):
        for subset in itertools.combinations(COMPONENTS, n):
            values[frozenset(subset)] = _value(score_rows, matched_score, shared_base_score, shared_matched_score, utility, frozenset(subset))
    m = len(COMPONENTS)
    shapley = {c: 0.0 for c in COMPONENTS}
    for c in COMPONENTS:
        others = [x for x in COMPONENTS if x != c]
        for n in range(len(others) + 1):
            for subset in itertools.combinations(others, n):
                s = frozenset(subset)
                weight = math.factorial(n) * math.factorial(m - n - 1) / math.factorial(m)
                shapley[c] += weight * (values[s | {c}] - values[s])
    order_rows = []
    for order in itertools.permutations(COMPONENTS):
        removed = frozenset()
        before = values[removed]
        for step, component in enumerate(order, 1):
            after_set = removed | {component}
            after = values[after_set]
            order_rows.append({
                "order": "->".join(order), "step": step, "component": component,
                "top1_before": before, "top1_after": after,
                "marginal_top1_gain": after - before,
            })
            removed, before = after_set, after
    return values, shapley, order_rows


def build_attribution(
    populations: dict[str, c72.TargetData],
    endpoints: dict[str, dict],
    repeated: dict[tuple[str, str], np.ndarray],
    shifted: dict[str, dict[str, np.ndarray]],
    bootstraps: int,
    seed: int,
) -> dict[str, list[dict]]:
    target_records: list[dict] = []
    order_records: list[dict] = []
    trajectory_records: list[dict] = []
    analyses = [("8", "bAcc")] + [(c72.FULL_BUDGET, endpoint) for endpoint in ENDPOINTS]
    for budget, endpoint in analyses:
        for target, pop in populations.items():
            construct_bacc, utility = endpoint_vectors(endpoints[target], endpoint)
            matched = construct_bacc
            score_rows = repeated[(target, budget)] if budget == "8" else endpoints[target]["construct_bacc"][None, :]
            shared_base = shifted[target]["bAcc"]
            shared_matched = shifted[target][endpoint]
            values, shapley, orders = _target_attribution(score_rows, matched, shared_base, shared_matched, utility)
            gap = 1.0 - values[frozenset()]
            for component in COMPONENTS:
                target_records.append({
                    "budget": budget, "endpoint": endpoint, "target_id": target,
                    "component": COMPONENT_NAMES[component], "component_code": component,
                    "observed_top1": values[frozenset()], "control_gap": gap,
                    "shapley_gain": shapley[component],
                    "shapley_fraction": shapley[component] / gap if gap > 1e-12 else math.nan,
                })
            for row in orders:
                order_records.append({"budget": budget, "endpoint": endpoint, "target_id": target, **row})

            if budget == c72.FULL_BUDGET:
                for trajectory, units in c72._field_groups(pop, "trajectory_cell"):
                    idx = c72._unit_indices(pop, units)
                    if len(idx) < 2:
                        continue
                    _, local_shapley, _ = _target_attribution(
                        score_rows[:, idx], matched[idx], shared_base[idx], shared_matched[idx], utility[idx]
                    )
                    for component in COMPONENTS:
                        trajectory_records.append({
                            "endpoint": endpoint, "target_id": target,
                            "trajectory_hash": hashlib.sha256(trajectory.encode()).hexdigest()[:16],
                            "candidate_count": len(idx), "component": COMPONENT_NAMES[component],
                            "component_code": component, "shapley_gain": local_shapley[component],
                        })

    shapley_rows, order_rows, bootstrap_rows, endpoint_rows = [], [], [], []
    grouped_target: dict[tuple, list[dict]] = defaultdict(list)
    for row in target_records:
        grouped_target[(row["budget"], row["endpoint"], row["component_code"])].append(row)
    rng = np.random.default_rng(seed)
    for (budget, endpoint, component), rows in sorted(grouped_target.items()):
        vals = np.asarray([r["shapley_gain"] for r in rows], dtype=float)
        samples = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(bootstraps)]
        order_component = [r for r in order_records if r["budget"] == budget and r["endpoint"] == endpoint and r["component"] == component]
        by_order: dict[str, list[float]] = defaultdict(list)
        for row in order_component:
            by_order[row["order"]].append(float(row["marginal_top1_gain"]))
        order_means = {order: float(np.mean(v)) for order, v in by_order.items()}
        largest_count = 0
        all_orders = sorted({r["order"] for r in order_records if r["budget"] == budget and r["endpoint"] == endpoint})
        for order in all_orders:
            means = {}
            for code in COMPONENTS:
                vv = [float(r["marginal_top1_gain"]) for r in order_records if r["budget"] == budget and r["endpoint"] == endpoint and r["order"] == order and r["component"] == code]
                means[code] = float(np.mean(vv))
            largest_count += int(component == max(means, key=means.get))
        mean_gain = float(np.mean(vals))
        shapley_rows.append({
            "budget": budget, "endpoint": endpoint,
            "component": COMPONENT_NAMES[component], "component_code": component,
            "mean_shapley_gain": mean_gain,
            "mean_shapley_fraction": float(np.nanmean([r["shapley_fraction"] for r in rows])),
            "ci_lower": float(np.quantile(samples, 0.025)),
            "ci_upper": float(np.quantile(samples, 0.975)),
            "order_gain_min": min(order_means.values()), "order_gain_max": max(order_means.values()),
            "largest_order_fraction": largest_count / max(len(all_orders), 1),
            "dominant_by_registered_rule": 0,
        })
        bootstrap_rows.append({
            "budget": budget, "endpoint": endpoint,
            "component": COMPONENT_NAMES[component], "bootstrap_unit": "target",
            "replicates": bootstraps, "point": mean_gain,
            "ci_lower": float(np.quantile(samples, 0.025)), "ci_upper": float(np.quantile(samples, 0.975)),
            "row_IID_used": 0,
        })
        for order, gain in order_means.items():
            order_rows.append({
                "budget": budget, "endpoint": endpoint, "order": order,
                "component": COMPONENT_NAMES[component], "component_code": component,
                "mean_marginal_gain": gain,
            })

    # Apply the strict dominance rule after all component intervals exist.
    for budget, endpoint in sorted({(r["budget"], r["endpoint"]) for r in shapley_rows}):
        rows = [r for r in shapley_rows if r["budget"] == budget and r["endpoint"] == endpoint]
        for row in rows:
            others = [r for r in rows if r is not row]
            row["dominant_by_registered_rule"] = int(
                row["largest_order_fraction"] >= 0.90
                and all(float(row["ci_lower"]) > float(other["ci_upper"]) for other in others)
            )
            endpoint_rows.append({
                "budget": budget, "endpoint": endpoint, "component": row["component"],
                "mean_shapley_fraction": row["mean_shapley_fraction"],
                "ci_lower": row["ci_lower"], "ci_upper": row["ci_upper"],
                "order_range": float(row["order_gain_max"] - row["order_gain_min"]),
                "dominant": row["dominant_by_registered_rule"],
            })

    loto_rows = []
    for left in sorted(populations, key=int):
        kept = [r for r in target_records if r["target_id"] != left and r["budget"] == c72.FULL_BUDGET and r["endpoint"] == "bAcc"]
        for component in COMPONENTS:
            vals = [r["shapley_gain"] for r in kept if r["component_code"] == component]
            loto_rows.append({
                "left_out_target": left, "component": COMPONENT_NAMES[component],
                "mean_shapley_gain": float(np.mean(vals)),
                "largest_component": "",
            })
        subset = loto_rows[-len(COMPONENTS):]
        largest = max(subset, key=lambda r: r["mean_shapley_gain"])["component"]
        for row in subset:
            row["largest_component"] = largest

    lotraj_rows = []
    trajectories = sorted({r["trajectory_hash"] for r in trajectory_records})
    for left in trajectories:
        kept = [r for r in trajectory_records if r["trajectory_hash"] != left and r["endpoint"] == "bAcc"]
        for component in COMPONENTS:
            vals = [r["shapley_gain"] for r in kept if r["component_code"] == component]
            lotraj_rows.append({
                "left_out_trajectory_hash": left, "component": COMPONENT_NAMES[component],
                "mean_shapley_gain": float(np.mean(vals)) if vals else math.nan,
                "trajectory_count_remaining": len(trajectories) - 1,
            })
    return {
        "attribution_order_sensitivity_rows": order_rows,
        "attribution_shapley_summary_rows": shapley_rows,
        "attribution_bootstrap_intervals_rows": bootstrap_rows,
        "attribution_endpoint_sensitivity_rows": endpoint_rows,
        "attribution_leave_target_out_rows": loto_rows,
        "attribution_leave_trajectory_out_rows": lotraj_rows,
        "_attribution_target_rows": target_records,
    }


def _design_rows(
    populations: dict[str, c72.TargetData],
    endpoints: dict[str, dict],
    shifted: dict[str, dict[str, np.ndarray]],
    construct_override: dict[str, np.ndarray] | None = None,
    gradient_override: dict[str, list[np.ndarray]] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], list[str]]:
    regimes = sorted({u.regime for pop in populations.values() for u in pop.units})
    matrices: dict[str, list[np.ndarray]] = defaultdict(list)
    ys, target_ids, trajectory_ids = [], [], []
    for target, pop in populations.items():
        source = np.asarray([u.source_score for u in pop.units], dtype=float)
        construct = construct_override[target] if construct_override is not None else endpoints[target]["construct_bacc"]
        shared_response = shifted[target]["bAcc"] - endpoints[target]["construct_bacc"]
        gradients = gradient_override[target] if gradient_override is not None else endpoints[target]["gradients"]
        gradient_matrix = np.vstack(gradients)
        neg_nll = -np.asarray([r["NLL"] for r in endpoints[target]["construct_metrics"]], dtype=float)
        neg_ece = -np.asarray([r["ECE"] for r in endpoints[target]["construct_metrics"]], dtype=float)
        order = np.asarray([u.candidate_order for u in pop.units], dtype=float)
        seed = np.asarray([u.seed for u in pop.units], dtype=float)
        level = np.asarray([u.level for u in pop.units], dtype=float)
        regime_cols = np.column_stack([[float(u.regime == regime) for u in pop.units] for regime in regimes])
        center = lambda v: np.asarray(v, dtype=float) - float(np.mean(v))
        base = np.column_stack([center(source), center(construct), center(shared_response)])
        utility_adjusted = np.column_stack([base, center(neg_nll), center(neg_ece)])
        metadata = np.column_stack([
            center(order), center(seed), center(level),
            regime_cols - np.mean(regime_cols, axis=0, keepdims=True),
        ])
        nuisance = np.column_stack([utility_adjusted, metadata])
        candidate_signal = gradient_matrix - np.mean(gradient_matrix, axis=0, keepdims=True)
        full = np.column_stack([nuisance, candidate_signal])
        y = center(endpoints[target]["eval_bacc"])
        valid = np.all(np.isfinite(full), axis=1) & np.isfinite(y)
        for name, matrix in (
            ("base", base), ("utility_adjusted", utility_adjusted),
            ("nuisance", nuisance), ("full", full),
        ):
            matrices[name].append(matrix[valid])
        ys.append(y[valid])
        target_ids.extend([target] * int(np.sum(valid)))
        trajectory_ids.extend([pop.units[i].trajectory_id for i in np.flatnonzero(valid)])
    return {name: np.vstack(parts) for name, parts in matrices.items()}, np.concatenate(ys), target_ids, trajectory_ids


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return beta, x_test @ beta


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return 1.0 - float(np.mean((y - pred) ** 2)) / max(float(np.mean(y ** 2)), 1e-12)


def build_residual_construct_validity(
    t2_pop: dict[str, c72.TargetData],
    t2_end: dict[str, dict],
    t2_shifted: dict[str, dict[str, np.ndarray]],
    t2_repeated: dict[tuple[str, str], np.ndarray],
    t3_pop: dict[str, c72.TargetData],
    t3_end: dict[str, dict],
    t3_shifted: dict[str, dict[str, np.ndarray]],
    t3_repeated: dict[tuple[str, str], np.ndarray],
    permutations: int,
    seed: int,
    protocol: dict,
) -> dict[str, list[dict] | bool | float]:
    x2, y2, t2_targets, _ = _design_rows(t2_pop, t2_end, t2_shifted)
    x3, y3, t3_targets, t3_trajectories = _design_rows(t3_pop, t3_end, t3_shifted)
    predictions = {}
    r2s = {}
    for model in ("base", "utility_adjusted", "nuisance", "full"):
        _, predictions[model] = _fit_predict(x2[model], y2, x3[model])
        r2s[model] = _r2(y3, predictions[model])
    increment = r2s["full"] - r2s["nuisance"]
    total_construct_increment = r2s["full"] - r2s["base"]
    residual = y3 - predictions["nuisance"]
    increment_proxy = predictions["full"] - predictions["nuisance"]

    # Two disjoint, candidate-shared construction halves provide independent
    # candidate-gradient proxies. Evaluation labels fit neither proxy on T3-HO.
    split2 = {target: _split_construct_indices(pop) for target, pop in t2_pop.items()}
    split3 = {target: _split_construct_indices(pop) for target, pop in t3_pop.items()}
    override2a = {target: c72._bacc_vector(t2_pop[target], split2[target][0]) for target in t2_pop}
    override2b = {target: c72._bacc_vector(t2_pop[target], split2[target][1]) for target in t2_pop}
    override3a = {target: c72._bacc_vector(t3_pop[target], split3[target][0]) for target in t3_pop}
    override3b = {target: c72._bacc_vector(t3_pop[target], split3[target][1]) for target in t3_pop}
    grad2a = {target: _gradients_for_indices(t2_pop[target], split2[target][0]) for target in t2_pop}
    grad2b = {target: _gradients_for_indices(t2_pop[target], split2[target][1]) for target in t2_pop}
    grad3a = {target: _gradients_for_indices(t3_pop[target], split3[target][0]) for target in t3_pop}
    grad3b = {target: _gradients_for_indices(t3_pop[target], split3[target][1]) for target in t3_pop}
    x2a, _, _, _ = _design_rows(t2_pop, t2_end, t2_shifted, override2a, grad2a)
    x2b, _, _, _ = _design_rows(t2_pop, t2_end, t2_shifted, override2b, grad2b)
    x3a, _, _, _ = _design_rows(t3_pop, t3_end, t3_shifted, override3a, grad3a)
    x3b, _, _, _ = _design_rows(t3_pop, t3_end, t3_shifted, override3b, grad3b)
    _, pna = _fit_predict(x2a["nuisance"], y2, x3a["nuisance"])
    _, pfa = _fit_predict(x2a["full"], y2, x3a["full"])
    _, pnb = _fit_predict(x2b["nuisance"], y2, x3b["nuisance"])
    _, pfb = _fit_predict(x2b["full"], y2, x3b["full"])
    proxy_a, proxy_b = pfa - pna, pfb - pnb
    split_corr = c72._spearman(proxy_a, proxy_b)

    target_arr2 = np.asarray(t2_targets)
    loto = []
    for left in sorted(set(t2_targets), key=int):
        train = target_arr2 != left
        test = target_arr2 == left
        _, pn = _fit_predict(x2["nuisance"][train], y2[train], x2["nuisance"][test])
        _, pf = _fit_predict(x2["full"][train], y2[train], x2["full"][test])
        loto.append({
            "left_out_target": left, "nuisance_r2": _r2(y2[test], pn),
            "full_r2": _r2(y2[test], pf),
            "incremental_r2": _r2(y2[test], pf) - _r2(y2[test], pn),
        })

    # Null calibration is performed on T3 prediction increments, preserving
    # target or trajectory blocks.  No refit uses T3 outcomes.
    target_arr3 = np.asarray(t3_targets)
    trajectory_arr3 = np.asarray(t3_trajectories)
    rng = np.random.default_rng(seed)
    null_candidate, null_trajectory = [], []
    for _ in range(permutations):
        perm = increment_proxy.copy()
        for target in sorted(set(t3_targets), key=int):
            idx = np.flatnonzero(target_arr3 == target)
            perm[idx] = rng.permutation(perm[idx])
        null_candidate.append(_r2(y3, predictions["nuisance"] + perm) - r2s["nuisance"])
        perm_t = increment_proxy.copy()
        for trajectory in sorted(set(t3_trajectories)):
            idx = np.flatnonzero(trajectory_arr3 == trajectory)
            perm_t[idx] = rng.permutation(perm_t[idx])
        null_trajectory.append(_r2(y3, predictions["nuisance"] + perm_t) - r2s["nuisance"])
    p_candidate = (1 + sum(v >= increment for v in null_candidate)) / (permutations + 1)
    p_trajectory = (1 + sum(v >= increment for v in null_trajectory)) / (permutations + 1)

    common_pred = np.zeros_like(increment_proxy)
    for target in sorted(set(t3_targets), key=int):
        idx = np.flatnonzero(target_arr3 == target)
        common_pred[idx] = float(np.mean(increment_proxy[idx]))
    common_increment = _r2(y3, predictions["nuisance"] + common_pred) - r2s["nuisance"]

    target_means = []
    within_vars = []
    for target in sorted(set(t3_targets), key=int):
        idx = np.flatnonzero(target_arr3 == target)
        target_means.append(float(np.mean(residual[idx])))
        within_vars.append(float(np.var(residual[idx])))
    common_var = float(np.var(target_means))
    candidate_var = float(np.mean(within_vars))
    common_fraction = common_var / max(common_var + candidate_var, 1e-12)

    validity = (
        split_corr >= float(protocol["residual_construct_validation"]["minimum_split_to_split_spearman"])
        and increment >= float(protocol["residual_construct_validation"]["minimum_incremental_heldout_R2"])
        and common_fraction <= float(protocol["residual_construct_validation"]["maximum_target_common_variance_fraction"])
        and p_candidate < 0.05 and p_trajectory < 0.05
        and float(np.median([r["incremental_r2"] for r in loto])) > 0
        and increment > common_increment
    )
    validity_rows = [
        {"criterion": "within_target_candidate_specificity", "threshold": "candidate variance > common variance", "observed": candidate_var / max(common_var, 1e-12), "passed": int(candidate_var > common_var), "notes": "target means removed by within-target design"},
        {"criterion": "target_common_variance_fraction", "threshold": protocol["residual_construct_validation"]["maximum_target_common_variance_fraction"], "observed": common_fraction, "passed": int(common_fraction <= float(protocol["residual_construct_validation"]["maximum_target_common_variance_fraction"])), "notes": "common-versus-within-target variance decomposition"},
        {"criterion": "split_to_split_proxy_stability", "threshold": protocol["residual_construct_validation"]["minimum_split_to_split_spearman"], "observed": split_corr, "passed": int(split_corr >= float(protocol["residual_construct_validation"]["minimum_split_to_split_spearman"])), "notes": "construction-only proxy predictions"},
        {"criterion": "nonreducible_to_utility_mismatch_and_metadata", "threshold": protocol["residual_construct_validation"]["minimum_incremental_heldout_R2"], "observed": increment, "passed": int(increment >= float(protocol["residual_construct_validation"]["minimum_incremental_heldout_R2"])), "notes": "candidate gradients added only after NLL/ECE/order/seed/level/regime nuisance model"},
        {"criterion": "leave_target_out_increment", "threshold": ">0 median", "observed": float(np.median([r["incremental_r2"] for r in loto])), "passed": int(float(np.median([r["incremental_r2"] for r in loto])) > 0), "notes": "T2 leave-one-target-out"},
        {"criterion": "target_common_replacement", "threshold": "candidate increment > common replacement", "observed": increment - common_increment, "passed": int(increment > common_increment), "notes": "target-common offset negative control"},
        {"criterion": "candidate_permutation_null", "threshold": "p<0.05", "observed": p_candidate, "passed": int(p_candidate < 0.05), "notes": "target-blocked"},
        {"criterion": "trajectory_preserving_null", "threshold": "p<0.05", "observed": p_trajectory, "passed": int(p_trajectory < 0.05), "notes": "trajectory-blocked"},
        {"criterion": "overall_construct_validity", "threshold": "all registered criteria", "observed": int(validity), "passed": int(validity), "notes": "failure requires unexplained-residual terminology"},
    ]
    split_rows = [
        {"comparison": "construction_proxy_A_vs_B", "spearman": split_corr, "shared_evaluation_outcome_in_proxy": 0, "split_A_trials_candidate_shared": 1, "split_B_trials_candidate_shared": 1, "passed": int(split_corr >= float(protocol["residual_construct_validation"]["minimum_split_to_split_spearman"]))},
    ]
    variance_rows = [
        {"component": "target_common_residual_mean", "variance": common_var, "variance_fraction": common_fraction, "removed_by_within_target_centering": 1, "validates_gauge": 0},
        {"component": "within_target_candidate_residual", "variance": candidate_var, "variance_fraction": 1.0 - common_fraction, "removed_by_within_target_centering": 0, "validates_gauge": 0},
    ]
    prediction_rows = [
        {"model": "source_construction_shared", "fit_stage": "T2", "evaluation_stage": "T3-HO", "r2": r2s["base"], "incremental_r2": 0.0, "evaluation_labels_fit": 0},
        {"model": "plus_utility_mismatch", "fit_stage": "T2", "evaluation_stage": "T3-HO", "r2": r2s["utility_adjusted"], "incremental_r2": r2s["utility_adjusted"] - r2s["base"], "evaluation_labels_fit": 0},
        {"model": "plus_checkpoint_regime_metadata", "fit_stage": "T2", "evaluation_stage": "T3-HO", "r2": r2s["nuisance"], "incremental_r2": r2s["nuisance"] - r2s["utility_adjusted"], "evaluation_labels_fit": 0},
        {"model": "plus_candidate_construction_gradient", "fit_stage": "T2", "evaluation_stage": "T3-HO", "r2": r2s["full"], "incremental_r2": increment, "total_increment_over_base": total_construct_increment, "evaluation_labels_fit": 0},
        *[{"model": f"T2_LOTO_{r['left_out_target']}", "fit_stage": "T2_minus_target", "evaluation_stage": "T2_left_target", "r2": r["full_r2"], "incremental_r2": r["incremental_r2"], "evaluation_labels_fit": 0} for r in loto],
    ]
    null_rows = [
        {"null": "candidate_permutation_within_target", "observed_incremental_r2": increment, "null_mean": float(np.mean(null_candidate)), "null_q95": float(np.quantile(null_candidate, 0.95)), "p_value": p_candidate, "permutations": permutations, "passed_against_null": int(p_candidate < 0.05)},
        {"null": "trajectory_preserving_shuffle", "observed_incremental_r2": increment, "null_mean": float(np.mean(null_trajectory)), "null_q95": float(np.quantile(null_trajectory, 0.95)), "p_value": p_trajectory, "permutations": permutations, "passed_against_null": int(p_trajectory < 0.05)},
        {"null": "target_common_residual_replacement", "observed_incremental_r2": increment, "null_mean": common_increment, "null_q95": common_increment, "p_value": "", "permutations": 0, "passed_against_null": int(increment > common_increment)},
    ]
    return {
        "residual_construct_validity_rows": validity_rows,
        "residual_split_stability_rows": split_rows,
        "candidate_vs_common_variance_rows": variance_rows,
        "residual_incremental_prediction_rows": prediction_rows,
        "residual_null_calibration_rows": null_rows,
        "residual_construct_validated": validity,
        "residual_incremental_r2": increment,
    }


def _bootstrap_mean_interval(values: list[float], replicates: int, rng: np.random.Generator) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    draws = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(replicates)]
    return float(np.mean(vals)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_shared_calibration_equivalence(
    populations: dict[str, c72.TargetData],
    endpoints: dict[str, dict],
    shifted: dict[str, dict[str, np.ndarray]],
    protocol: dict,
    bootstraps: int,
    seed: int,
) -> dict[str, list[dict] | str]:
    effects: dict[str, list[float]] = defaultdict(list)
    per_target = []
    for target, pop in populations.items():
        utility = endpoints[target]["eval_bacc"]
        baseline = endpoints[target]["construct_bacc"]
        shared = shifted[target]["bAcc"]
        base_gauge = c70._gauge_recovery(baseline, utility)[0]
        shared_gauge = c70._gauge_recovery(shared, utility)[0]
        b1, b3, br = c72._top_metrics(baseline, utility, 3)
        s1, s3, sr = c72._top_metrics(shared, utility, 3)
        vals = {
            "gauge_recovery_increment": shared_gauge - base_gauge,
            "coverage_increment": float(sr <= 0.02) - float(br <= 0.02),
            "topk_increment": s3 - b3,
            "regret_reduction_bAcc": br - sr,
        }
        for metric, value in vals.items():
            effects[metric].append(value)
            per_target.append({"target_id": target, "metric": metric, "effect": value})
    rng = np.random.default_rng(seed)
    rows = []
    all_equivalent = True
    for metric, values in effects.items():
        point, lo, hi = _bootstrap_mean_interval(values, bootstraps, rng)
        sesoi = float(protocol["H3_shared_calibration"]["SESOI"][metric])
        equivalent = hi < sesoi and lo > -sesoi
        all_equivalent &= equivalent
        rows.append({
            "intervention_family": "shared_class_vector",
            "metric": metric, "point_effect": point, "ci_lower": lo, "ci_upper": hi,
            "beneficial_SESOI": sesoi, "harmful_SESOI": -sesoi,
            "practically_equivalent_to_zero": int(equivalent),
            "conclusion": "practically_insufficient" if equivalent else "potentially_meaningful_but_unresolved",
            "bootstrap_unit": "target", "replicates": bootstraps,
        })
    conclusion = "practically_insufficient" if all_equivalent else "potentially_meaningful_but_unresolved"
    power_rows = []
    for metric, values in effects.items():
        sd = float(np.std(values, ddof=1))
        detectable = 2.8 * sd / math.sqrt(max(len(values), 1))
        sesoi = float(protocol["H3_shared_calibration"]["SESOI"][metric])
        power_rows.append({
            "intervention_family": "shared_class_vector",
            "metric": metric, "target_count": len(values), "between_target_sd": sd,
            "approx_80pct_detectable_effect": detectable, "SESOI": sesoi,
            "adequate_for_SESOI": int(detectable <= sesoi),
            "p_gt_0p05_used_as_insufficiency": 0,
        })
    return {
        "shared_calibration_equivalence_rows": rows,
        "shared_calibration_power_audit_rows": power_rows + [
            {"intervention_family": "shared_class_vector", **row} for row in per_target
        ],
        "shared_calibration_conclusion": conclusion,
    }


def _trajectory_shared_shifts(pop: c72.TargetData, gradients: list[np.ndarray], scale: float) -> list[np.ndarray]:
    by_trajectory: dict[str, list[int]] = defaultdict(list)
    for i, unit in enumerate(pop.units):
        by_trajectory[unit.trajectory_id].append(i)
    shifts = [np.zeros_like(np.asarray(gradients[0])) for _ in pop.units]
    for trajectory, idx in by_trajectory.items():
        seed = int(hashlib.sha256((trajectory + "|C73").encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=len(gradients[0]))
        raw -= float(np.mean(raw))
        norm = float(np.linalg.norm(raw))
        target_norm = scale * float(np.mean([np.linalg.norm(gradients[i]) for i in idx]))
        vector = raw * target_norm / max(norm, 1e-12)
        for i in idx:
            shifts[i] = vector
    return shifts


def _permuted_gradient_shifts(pop: c72.TargetData, gradients: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    order = np.argsort([hashlib.sha256((u.checkpoint_id + "|C73perm").encode()).hexdigest() for u in pop.units])
    perm = np.roll(order, 1)
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = perm
    return [alpha * np.asarray(gradients[int(inverse[i])]) for i in range(len(pop.units))]


def build_h4_identity_audit(
    populations: dict[str, c72.TargetData],
    endpoints: dict[str, dict],
    shared_lock: str,
    permutations: int,
    seed: int,
) -> dict[str, list[dict] | bool]:
    template, magnitude = shared_lock.split("|")
    family_target: dict[str, list[dict]] = defaultdict(list)
    pair_summaries: dict[str, list[dict]] = defaultdict(list)
    for target, pop in populations.items():
        assert pop.labels is not None
        eval_idx = pop.indices("target_eval")
        original = np.asarray(endpoints[target]["eval_bacc"], dtype=float)
        gradients = endpoints[target]["gradients"]
        shared = c72._shared_template(pop, template, float(magnitude))
        families = {
            "target_shared": [shared] * len(pop.units),
            "magnitude_matched_random_candidate": [c72._random_matched_shift(u, np.asarray(g), 0.25) for u, g in zip(pop.units, gradients)],
            "random_sign_construction": [(-1.0 if int(hashlib.sha256(u.checkpoint_id.encode()).hexdigest()[-1], 16) % 2 else 1.0) * 0.10 * np.asarray(g) for u, g in zip(pop.units, gradients)],
            "trajectory_preserving": _trajectory_shared_shifts(pop, gradients, 0.25),
            "construction_estimated": [0.10 * np.asarray(g) for g in gradients],
            "candidate_permuted_construction": _permuted_gradient_shifts(pop, gradients, 0.10),
        }
        for family, shifts in families.items():
            perturbed = np.asarray([
                c72._endpoint_metrics(u.logits.astype(float) + shift[None, :], pop.labels, eval_idx, pop.classes)["bAcc"]
                for u, shift in zip(pop.units, shifts)
            ])
            delta = perturbed - original
            pair_rows = []
            for i in range(len(pop.units)):
                for j in range(i + 1, len(pop.units)):
                    d = float(original[i] - original[j])
                    q = float(delta[i] - delta[j])
                    if abs(d) <= 1e-12:
                        continue
                    actual = int(d * float(perturbed[i] - perturbed[j]) < 0)
                    identity = int(d * (d + q) < 0)
                    pair_rows.append({
                        "target_id": target, "family": family,
                        "margin_abs": abs(d), "perturbation_gap_abs": abs(q),
                        "ratio": abs(q) / max(abs(d), 1e-12),
                        "rank_flip": actual, "identity_prediction": identity,
                    })
            pair_summaries[family].extend(pair_rows)
            ratios = np.asarray([r["ratio"] for r in pair_rows], dtype=float)
            flips = np.asarray([r["rank_flip"] for r in pair_rows], dtype=float)
            slope = float(np.polyfit(np.log1p(ratios), flips, 1)[0]) if len(np.unique(ratios)) > 1 else 0.0
            family_target[family].append({
                "target_id": target, "family": family, "pair_count": len(pair_rows),
                "rank_flip_rate": float(np.mean(flips)) if len(flips) else 0.0,
                "identity_accuracy": float(np.mean([r["rank_flip"] == r["identity_prediction"] for r in pair_rows])) if pair_rows else 1.0,
                "calibration_slope_log_ratio": slope,
                "increment_beyond_identity": float(np.mean([r["rank_flip"] - r["identity_prediction"] for r in pair_rows])) if pair_rows else 0.0,
            })

    identity_rows, null_rows, blocked_rows = [], [], []
    rng = np.random.default_rng(seed)
    for family, rows in family_target.items():
        flip_vals = [r["rank_flip_rate"] for r in rows]
        slope_vals = [r["calibration_slope_log_ratio"] for r in rows]
        identity_vals = [r["identity_accuracy"] for r in rows]
        increments = [r["increment_beyond_identity"] for r in rows]
        p_slope, exceed = c72._sign_flip_p(slope_vals, permutations, seed + len(identity_rows) * 17)
        identity_rows.append({
            "family": family, "target_count": len(rows),
            "pair_count": sum(r["pair_count"] for r in rows),
            "mean_rank_flip_rate": float(np.mean(flip_vals)),
            "mean_identity_accuracy": float(np.mean(identity_vals)),
            "mean_calibration_slope_log_ratio": float(np.mean(slope_vals)),
            "mean_increment_beyond_identity": float(np.mean(increments)),
            "blocked_slope_p": p_slope, "permutations": permutations,
            "algebraic_identity_fully_explains_observed_flips": int(min(identity_vals) == 1.0 and max(abs(v) for v in increments) <= 1e-12),
            "validates_residual_origin": 0,
        })
        blocked_rows.extend(rows)
        null_rows.append({
            "family": family, "observed_flip_rate": float(np.mean(flip_vals)),
            "observed_slope": float(np.mean(slope_vals)),
            "target_cluster_p": p_slope, "target_count": len(rows),
            "null_role": "generic_sensitivity" if "random" in family or "permuted" in family else "structured_intervention",
        })
    by_family = {r["family"]: r for r in identity_rows}
    constructed = by_family["construction_estimated"]
    random = by_family["magnitude_matched_random_candidate"]
    comparison_rows = [{
        "comparison": "construction_estimated_minus_magnitude_matched_random",
        "constructed_flip_rate": constructed["mean_rank_flip_rate"],
        "random_flip_rate": random["mean_rank_flip_rate"],
        "flip_rate_difference": constructed["mean_rank_flip_rate"] - random["mean_rank_flip_rate"],
        "constructed_slope": constructed["mean_calibration_slope_log_ratio"],
        "random_slope": random["mean_calibration_slope_log_ratio"],
        "origin_evidence_beyond_generic_sensitivity": int(constructed["mean_rank_flip_rate"] > random["mean_rank_flip_rate"] and constructed["mean_increment_beyond_identity"] > 0),
    }]
    reduces_to_identity = all(int(r["algebraic_identity_fully_explains_observed_flips"]) for r in identity_rows) and not comparison_rows[0]["origin_evidence_beyond_generic_sensitivity"]
    return {
        "h4_identity_vs_empirical_effect_rows": identity_rows,
        "h4_matched_nulls_rows": null_rows,
        "h4_blocked_effects_rows": blocked_rows,
        "h4_constructed_vs_random_perturbation_rows": comparison_rows,
        "h4_reduces_to_identity": bool(reduces_to_identity),
    }


def _split_construct_indices(pop: c72.TargetData) -> tuple[np.ndarray, np.ndarray]:
    pool = pop.indices("target_construct")
    a, b = [], []
    assert pop.labels is not None
    for cls in pop.classes:
        idx = [i for i in pool if pop.labels[i] == cls]
        idx = sorted(idx, key=lambda i: hashlib.sha256((pop.trial_ids[i] + "|C73split").encode()).hexdigest())
        cut = len(idx) // 2
        a.extend(idx[:cut])
        b.extend(idx[cut:])
    return np.asarray(sorted(a), dtype=int), np.asarray(sorted(b), dtype=int)


def _gradients_for_indices(pop: c72.TargetData, indices: np.ndarray) -> list[np.ndarray]:
    assert pop.labels is not None
    gradients = []
    for unit in pop.units:
        probs = c72._softmax(unit.logits[indices])
        onehot = np.eye(unit.logits.shape[1], dtype=float)[pop.labels[indices]]
        grad = np.mean(onehot - probs, axis=0)
        gradients.append(grad - float(np.mean(grad)))
    return gradients


def build_alpha_zero_audit(
    stage: str,
    populations: dict[str, c72.TargetData],
    endpoints: dict[str, dict],
    alpha_grid: list[float],
) -> dict[str, list[dict]]:
    target_alpha_fold: list[dict] = []
    for target, pop in populations.items():
        assert pop.labels is not None
        a, b = _split_construct_indices(pop)
        utility = endpoints[target]["eval_bacc"]
        for fold, fit_idx, score_idx in (("A_to_B", a, b), ("B_to_A", b, a)):
            gradients = _gradients_for_indices(pop, fit_idx)
            for alpha in alpha_grid:
                shifts = [float(alpha) * g for g in gradients]
                score = c72._bacc_vector(pop, score_idx, shifts=shifts)
                top1, top3, regret = c72._top_metrics(score, utility, 3)
                target_alpha_fold.append({
                    "stage": stage, "target_id": target, "fold": fold,
                    "alpha": alpha, "spearman": c72._spearman(score, utility),
                    "pairwise_accuracy": c72._pairwise_accuracy(score, utility),
                    "top1": top1, "top3": top3, "regret": regret,
                    "coverage": int(regret <= 0.02), "T3_option_selected": 0,
                })
    summary_rows = []
    for alpha in alpha_grid:
        rows = [r for r in target_alpha_fold if float(r["alpha"]) == float(alpha)]
        summary_rows.append({
            "stage": stage, "alpha": alpha, "target_fold_rows": len(rows),
            "mean_spearman": float(np.mean([r["spearman"] for r in rows])),
            "mean_pairwise_accuracy": float(np.mean([r["pairwise_accuracy"] for r in rows])),
            "mean_top1": float(np.mean([r["top1"] for r in rows])),
            "mean_top3": float(np.mean([r["top3"] for r in rows])),
            "mean_regret": float(np.mean([r["regret"] for r in rows])),
            "mean_coverage": float(np.mean([r["coverage"] for r in rows])),
            "selected": int(stage == "T2" and alpha == max(alpha_grid, key=lambda x: (float(np.mean([r["pairwise_accuracy"] for r in target_alpha_fold if float(r["alpha"]) == float(x)])), -abs(x)))),
            "T3_tuned": 0,
        })
    by_alpha = {float(r["alpha"]): r for r in summary_rows}
    h = 0.05
    derivative_rows = []
    for metric in ("mean_pairwise_accuracy", "mean_top3", "mean_coverage", "mean_regret"):
        derivative = (float(by_alpha[h][metric]) - float(by_alpha[-h][metric])) / (2 * h)
        curvature = (float(by_alpha[h][metric]) - 2 * float(by_alpha[0.0][metric]) + float(by_alpha[-h][metric])) / (h ** 2)
        derivative_rows.append({
            "stage": stage, "metric": metric, "alpha_step": h,
            "central_derivative_at_zero": derivative,
            "central_curvature_at_zero": curvature,
            "zero_is_local_maximum": int(curvature < 0 and abs(derivative) <= 0.05),
        })
    stability_rows = []
    for target in sorted(populations, key=int):
        best = {}
        for fold in ("A_to_B", "B_to_A"):
            rows = [r for r in target_alpha_fold if r["target_id"] == target and r["fold"] == fold]
            chosen = max(rows, key=lambda r: (r["pairwise_accuracy"], -abs(float(r["alpha"]))))
            best[fold] = float(chosen["alpha"])
        stability_rows.append({
            "stage": stage, "target_id": target,
            "best_alpha_A_to_B": best["A_to_B"], "best_alpha_B_to_A": best["B_to_A"],
            "same_sign": int(np.sign(best["A_to_B"]) == np.sign(best["B_to_A"])),
            "same_exact_alpha": int(best["A_to_B"] == best["B_to_A"]),
        })
    return {
        "summary": summary_rows,
        "derivatives": derivative_rows,
        "stability": stability_rows,
        "target_fold": target_alpha_fold,
    }


def _ols_coefficients(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    return beta, _r2(y - float(np.mean(y)), pred - float(np.mean(pred)))


def build_candidate_count_confounding(
    populations: dict[str, c72.TargetData],
    endpoints: dict[str, dict],
    temperature: float,
    epsilon: float,
    subsets: int,
    permutations: int,
    seed: int,
) -> dict[str, list[dict] | bool]:
    rng = np.random.default_rng(seed)
    raw_rows = []
    for target, pop in populations.items():
        utility = np.asarray(endpoints[target]["eval_bacc"], dtype=float)
        score = np.asarray(endpoints[target]["construct_bacc"], dtype=float)
        for m in (2, 4, 8, 16, 32, 64):
            if m > len(pop.units):
                continue
            for repeat in range(subsets):
                idx = np.sort(rng.choice(np.arange(len(pop.units)), size=m, replace=False))
                u, s = utility[idx], score[idx]
                top1, _, regret = c72._top_metrics(s, u, 1)
                sorted_u = np.sort(u)[::-1]
                raw_rows.append({
                    "target_id": target, "field_level": "target_subsample",
                    "repeat": repeat, "raw_M": m,
                    "effective_M": effective_multiplicity(u, temperature),
                    "near_tie_M": int(np.sum(u >= float(np.max(u)) - epsilon - 1e-12)),
                    "top_gap": float(sorted_u[0] - sorted_u[1]),
                    "top1": top1, "regret": regret,
                })
        for trajectory, units in c72._field_groups(pop, "trajectory_cell"):
            idx = c72._unit_indices(pop, units)
            if len(idx) < 2:
                continue
            u, s = utility[idx], score[idx]
            top1, _, regret = c72._top_metrics(s, u, 1)
            sorted_u = np.sort(u)[::-1]
            raw_rows.append({
                "target_id": target, "field_level": "trajectory_cell",
                "repeat": hashlib.sha256(trajectory.encode()).hexdigest()[:12],
                "raw_M": len(idx), "effective_M": effective_multiplicity(u, temperature),
                "near_tie_M": int(np.sum(u >= float(np.max(u)) - epsilon - 1e-12)),
                "top_gap": float(sorted_u[0] - sorted_u[1]),
                "top1": top1, "regret": regret,
            })

    model_specs = {
        "raw_M_only": ("log_raw_M",),
        "effective_M_only": ("log_effective_M",),
        "raw_M_plus_top_gap": ("log_raw_M", "top_gap"),
        "raw_M_effective_M_top_gap": ("log_raw_M", "log_effective_M", "top_gap"),
    }
    target_effects: dict[str, list[dict]] = defaultdict(list)
    for target in sorted(populations, key=int):
        rows = [r for r in raw_rows if r["target_id"] == target and r["field_level"] == "target_subsample"]
        y = np.asarray([r["top1"] for r in rows], dtype=float)
        data = {
            "log_raw_M": np.log(np.asarray([r["raw_M"] for r in rows], dtype=float)),
            "log_effective_M": np.log(np.asarray([r["effective_M"] for r in rows], dtype=float)),
            "top_gap": np.asarray([r["top_gap"] for r in rows], dtype=float),
        }
        for model, cols in model_specs.items():
            x = np.column_stack([data[c] for c in cols])
            x = (x - np.mean(x, axis=0)) / np.maximum(np.std(x, axis=0), 1e-12)
            beta, r2 = _ols_coefficients(x, y)
            target_effects[model].append({
                "target_id": target, "model": model, "predictors": ";".join(cols),
                "raw_M_coefficient": beta[1 + cols.index("log_raw_M")] if "log_raw_M" in cols else math.nan,
                "effective_M_coefficient": beta[1 + cols.index("log_effective_M")] if "log_effective_M" in cols else math.nan,
                "top_gap_coefficient": beta[1 + cols.index("top_gap")] if "top_gap" in cols else math.nan,
                "r2": r2,
            })
    model_rows = []
    for model, rows in target_effects.items():
        raw_vals = [r["raw_M_coefficient"] for r in rows if math.isfinite(r["raw_M_coefficient"])]
        eff_vals = [r["effective_M_coefficient"] for r in rows if math.isfinite(r["effective_M_coefficient"])]
        gap_vals = [r["top_gap_coefficient"] for r in rows if math.isfinite(r["top_gap_coefficient"])]
        raw_p = c72._sign_flip_p([-v for v in raw_vals], permutations, seed + len(model_rows) * 31)[0] if raw_vals else math.nan
        eff_p = c72._sign_flip_p([-v for v in eff_vals], permutations, seed + len(model_rows) * 31 + 1)[0] if eff_vals else math.nan
        model_rows.append({
            "model": model, "predictors": rows[0]["predictors"], "target_count": len(rows),
            "mean_raw_M_coefficient": float(np.mean(raw_vals)) if raw_vals else math.nan,
            "raw_M_blocked_p": raw_p,
            "mean_effective_M_coefficient": float(np.mean(eff_vals)) if eff_vals else math.nan,
            "effective_M_blocked_p": eff_p,
            "mean_top_gap_coefficient": float(np.mean(gap_vals)) if gap_vals else math.nan,
            "mean_r2": float(np.mean([r["r2"] for r in rows])),
            "row_IID_used": 0,
        })
    by_model = {r["model"]: r for r in model_rows}
    raw_only = float(by_model["raw_M_only"]["mean_raw_M_coefficient"])
    adjusted = float(by_model["raw_M_effective_M_top_gap"]["mean_raw_M_coefficient"])
    effective_dominates = abs(adjusted) < 0.5 * abs(raw_only) or float(by_model["raw_M_effective_M_top_gap"]["raw_M_blocked_p"]) >= 0.05

    raw_eff_rows = []
    for level in ("target_subsample", "trajectory_cell"):
        rows = [r for r in raw_rows if r["field_level"] == level]
        raw_eff_rows.append({
            "field_level": level, "rows": len(rows),
            "raw_M_mean": float(np.mean([r["raw_M"] for r in rows])),
            "effective_M_mean": float(np.mean([r["effective_M"] for r in rows])),
            "near_tie_M_mean": float(np.mean([r["near_tie_M"] for r in rows])),
            "raw_effective_spearman": c72._spearman(np.asarray([r["raw_M"] for r in rows]), np.asarray([r["effective_M"] for r in rows])),
            "effective_top1_spearman": c72._spearman(np.asarray([r["effective_M"] for r in rows]), np.asarray([r["top1"] for r in rows])),
            "raw_top1_spearman": c72._spearman(np.asarray([r["raw_M"] for r in rows]), np.asarray([r["top1"] for r in rows])),
        })
    adjusted_rows = [
        {"comparison": "raw_M_before_after_effective_gap_adjustment", "raw_only_coefficient": raw_only, "adjusted_raw_M_coefficient": adjusted, "coefficient_retained_fraction": adjusted / raw_only if abs(raw_only) > 1e-12 else math.nan, "effective_geometry_dominates": int(effective_dominates)},
    ]
    loto_rows = []
    for left in sorted(populations, key=int):
        kept = [r for model_rows_target in target_effects.values() for r in model_rows_target if r["target_id"] != left and r["model"] == "raw_M_effective_M_top_gap"]
        loto_rows.append({
            "left_out_target": left,
            "mean_raw_M_coefficient": float(np.mean([r["raw_M_coefficient"] for r in kept])),
            "mean_effective_M_coefficient": float(np.mean([r["effective_M_coefficient"] for r in kept])),
            "mean_top_gap_coefficient": float(np.mean([r["top_gap_coefficient"] for r in kept])),
            "raw_M_sign_stable": int(np.sign(np.mean([r["raw_M_coefficient"] for r in kept])) == np.sign(adjusted)),
        })
    return {
        "candidate_count_confounding_rows": model_rows,
        "raw_M_vs_effective_M_rows": raw_eff_rows,
        "top_gap_adjusted_effects_rows": adjusted_rows,
        "h6_leave_target_out_rows": loto_rows,
        "effective_multiplicity_dominates": bool(effective_dominates),
    }


def build_theory_repair(
    populations: dict[str, c72.TargetData],
    endpoints: dict[str, dict],
    repeated: dict[tuple[str, str], np.ndarray],
    c72_model_bounds: list[dict],
    c72_finite_bounds: list[dict],
    epsilon: float,
) -> dict[str, list[dict] | bool]:
    diagnosis_rows = []
    by_target_bound = {(r["target_id"], r["budget"]): r for r in c72_model_bounds}
    for target, pop in populations.items():
        utility = endpoints[target]["eval_bacc"]
        sorted_u = np.sort(utility)[::-1]
        gap = float(sorted_u[0] - sorted_u[1])
        for budget in ("8", c72.FULL_BUDGET):
            row = by_target_bound.get((target, budget), {})
            gauge_sd = float(row.get("gauge_sd", math.nan))
            diagnosis_rows.append({
                "target_id": target, "budget": budget, "candidate_count": len(pop.units),
                "top_gap": gap, "gauge_sd": gauge_sd,
                "gauge_to_gap_ratio": gauge_sd / max(gap, 1e-12),
                "union_bound": float(row.get("top1_error_union_bound", 1.0)),
                "bound_nontrivial": int(row.get("bound_nontrivial", 0)),
                "failure_union_bound_looseness": int(float(row.get("top1_error_union_bound", 1.0)) >= 1.0),
                "failure_gap_smaller_than_gauge": int(gauge_sd > gap) if math.isfinite(gauge_sd) else 1,
                "candidate_dependence_unmodeled": 1,
                "heteroskedasticity_unmodeled": 1,
            })

    effective_rows, tail_rows = [], []
    for target, pop in populations.items():
        assert pop.labels is not None
        utility = np.asarray(endpoints[target]["eval_bacc"], dtype=float)
        best = int(np.argmax(utility))
        pool = pop.indices("target_construct")
        correct = pop.correctness()
        top_set = set(np.flatnonzero(utility >= float(np.max(utility)) - epsilon - 1e-12))
        for budget in (8, 64, c72.FULL_BUDGET):
            outside_errors = []
            all_errors = []
            residual_gaps = []
            margins = []
            mean_score = np.mean(repeated[(target, str(budget))], axis=0)
            residual = (utility - float(np.mean(utility))) - (mean_score - float(np.mean(mean_score)))
            for j in range(len(pop.units)):
                if j == best:
                    continue
                err, _, _ = c72.exact_stratified_pair_order_error(correct[best] - correct[j], pop.labels, pool, pop.classes, budget)
                all_errors.append(err)
                if j not in top_set:
                    outside_errors.append(err)
                residual_gaps.append(abs(float(residual[best] - residual[j])))
                margins.append(abs(float(utility[best] - utility[j])))
            top1_union = min(1.0, float(np.sum(all_errors)))
            topk_union = min(1.0, float(np.sum(outside_errors)))
            effective_rows.append({
                "target_id": target, "budget": str(budget),
                "candidate_count": len(pop.units), "epsilon": epsilon,
                "epsilon_optimal_count": len(top_set),
                "top1_union_bound": top1_union,
                "epsilon_topk_union_bound": topk_union,
                "top1_hit_lower_bound": max(0.0, 1.0 - top1_union),
                "epsilon_topk_hit_lower_bound": max(0.0, 1.0 - topk_union),
                "top1_bound_nontrivial": int(top1_union < 1.0),
                "effective_topk_bound_nontrivial": int(topk_union < 1.0),
                "distributional_theorem_claimed": 0,
            })
            exceed = np.asarray(residual_gaps) >= np.asarray(margins)
            n = len(exceed)
            rate = float(np.mean(exceed)) if n else math.nan
            # Hoeffding upper confidence envelope, explicitly empirical.
            upper = min(1.0, rate + math.sqrt(math.log(40.0) / (2 * max(n, 1)))) if n else math.nan
            tail_rows.append({
                "target_id": target, "budget": str(budget), "pair_count": n,
                "empirical_residual_exceeds_margin_rate": rate,
                "hoeffding_upper_95": upper,
                "target_blocked_interpretation": 1,
                "simulation_or_empirical_proxy_only": 1,
            })
    finite_rows = [{**r, "source_milestone": "C72_exact_replay", "simulation_proxy": 0} for r in c72_finite_bounds]
    full_rows = [r for r in effective_rows if r["budget"] == c72.FULL_BUDGET]
    nontrivial_rate = float(np.mean([r["top1_bound_nontrivial"] for r in effective_rows])) if effective_rows else 0.0
    repaired = bool(
        full_rows
        and all(int(r["top1_bound_nontrivial"]) for r in full_rows)
        and nontrivial_rate >= 0.80
    )
    return {
        "gaussian_bound_failure_diagnosis_rows": diagnosis_rows,
        "finite_population_best_arm_bound_rows": finite_rows,
        "effective_candidate_bound_rows": effective_rows,
        "empirical_tail_bound_rows": tail_rows,
        "multi_candidate_top1_bound_repaired": repaired,
    }


def _synthetic_gauge(
    candidate_count: int,
    shape: str,
    dependence: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a centered candidate-specific field with a registered tail/dependence form."""
    independent = rng.normal(size=candidate_count)
    loading = np.linspace(-1.0, 1.0, candidate_count)
    rng.shuffle(loading)
    structured = float(rng.normal()) * loading
    raw = math.sqrt(dependence) * structured + math.sqrt(1.0 - dependence) * independent
    if shape == "skewed":
        raw = np.exp(np.clip(raw, -3.0, 3.0))
    elif shape == "heteroskedastic":
        raw = raw * (0.35 + 1.30 * np.abs(loading))
    elif shape != "gaussian":
        raise ValueError(f"unsupported synthetic gauge tail: {shape}")
    raw = raw - float(np.mean(raw))
    return raw / max(float(np.std(raw)), 1e-12)


def _pairwise_accuracy_vectorized(
    score: np.ndarray,
    utility: np.ndarray,
    pair_indices: tuple[np.ndarray, np.ndarray],
) -> float:
    left, right = pair_indices
    dx = np.asarray(score, dtype=float)[left] - np.asarray(score, dtype=float)[right]
    dy = np.asarray(utility, dtype=float)[left] - np.asarray(utility, dtype=float)[right]
    valid = np.abs(dy) >= 1e-12
    if not np.any(valid):
        return math.nan
    return float(np.mean(np.where(np.abs(dx[valid]) < 1e-12, 0.5, (dx[valid] * dy[valid] > 0).astype(float))))


def build_synthetic_robustness(protocol: dict) -> dict[str, list[dict] | dict]:
    """Run the locked aggregate synthetic stress grid without persisting raw draws."""
    grid = protocol["synthetic_grid"]
    replicates = int(grid["replicates_per_cell"])
    rng = np.random.default_rng(int(grid["seed"]))
    rows = []
    pair_indices = {int(m): np.triu_indices(int(m), 1) for m in grid["candidate_counts"]}
    for m in grid["candidate_counts"]:
        for requested_near in grid["effective_near_tie_counts"]:
            if int(requested_near) > int(m):
                continue
            for tail in grid["gauge_tail"]:
                for dependence in grid["dependence"]:
                    for budget in grid["label_budgets"]:
                        metrics: dict[str, list[float]] = defaultdict(list)
                        for _ in range(replicates):
                            source = rng.normal(size=int(m))
                            source = (source - float(np.mean(source))) / max(float(np.std(source)), 1e-12)
                            gauge = _synthetic_gauge(int(m), str(tail), float(dependence), rng)
                            latent = source + 0.22 * gauge

                            # Lock the requested extreme-order geometry after drawing
                            # the rank/gauge field. The rest of the ordering remains real.
                            order = np.argsort(latent)
                            top = order[-int(requested_near):]
                            outside = order[:-int(requested_near)]
                            peak = float(np.max(latent))
                            latent[top] = peak - rng.uniform(0.0, 0.012, size=len(top))
                            if len(outside):
                                latent[outside] = np.minimum(
                                    latent[outside], peak - 0.035 - rng.exponential(0.02, size=len(outside))
                                )

                            finite_sd = 0.12 / math.sqrt(max(float(budget) / 8.0, 1.0))
                            construction = source + 0.10 * gauge + rng.normal(scale=finite_sd, size=int(m))
                            common = float(rng.normal())
                            shifted = latent + common
                            top1, top3, regret = c72._top_metrics(construction, latent, min(3, int(m)))
                            common_flip = int(not np.array_equal(np.argsort(latent), np.argsort(shifted)))
                            candidate_shift = 0.12 * _synthetic_gauge(int(m), str(tail), float(dependence), rng)
                            candidate_flip = int(not np.array_equal(np.argsort(latent), np.argsort(latent + candidate_shift)))
                            metrics["spearman"].append(c72._spearman(construction, latent))
                            metrics["pairwise_accuracy"].append(
                                _pairwise_accuracy_vectorized(construction, latent, pair_indices[int(m)])
                            )
                            metrics["top1"].append(top1)
                            metrics["top3"].append(top3)
                            metrics["regret"].append(regret)
                            metrics["effective_M"].append(effective_multiplicity(latent, 0.012))
                            metrics["realized_near_tie_M"].append(float(np.sum(latent >= float(np.max(latent)) - 0.02)))
                            metrics["common_flip"].append(common_flip)
                            metrics["candidate_flip"].append(candidate_flip)
                        mean = {key: float(np.mean(value)) for key, value in metrics.items()}
                        rows.append({
                            "candidate_count": int(m),
                            "requested_near_tie_count": int(requested_near),
                            "gauge_tail": str(tail),
                            "candidate_dependence": float(dependence),
                            "label_budget": int(budget),
                            "replicates": replicates,
                            "mean_spearman": mean["spearman"],
                            "mean_pairwise_accuracy": mean["pairwise_accuracy"],
                            "mean_top1": mean["top1"],
                            "mean_top3": mean["top3"],
                            "mean_regret": mean["regret"],
                            "mean_effective_M": mean["effective_M"],
                            "mean_realized_near_tie_M": mean["realized_near_tie_M"],
                            "common_offset_rank_flip_rate": mean["common_flip"],
                            "candidate_specific_rank_flip_rate": mean["candidate_flip"],
                            "high_reliability_poor_top1": int(mean["spearman"] >= 0.50 and mean["top1"] <= 0.25),
                            "raw_draws_persisted": 0,
                        })
    validation = {
        "grid_rows": len(rows),
        "raw_draws_persisted": 0,
        "common_offset_identity_passed": bool(rows and all(float(r["common_offset_rank_flip_rate"]) == 0.0 for r in rows)),
        "candidate_specific_crossings_present": bool(rows and any(float(r["candidate_specific_rank_flip_rate"]) > 0 for r in rows)),
        "high_reliability_poor_top1_cells": sum(int(r["high_reliability_poor_top1"]) for r in rows),
    }
    validation["passed"] = bool(
        validation["common_offset_identity_passed"]
        and validation["candidate_specific_crossings_present"]
        and validation["high_reliability_poor_top1_cells"] > 0
    )
    return {"synthetic_attribution_robustness_rows": rows, "synthetic_validation": validation}
