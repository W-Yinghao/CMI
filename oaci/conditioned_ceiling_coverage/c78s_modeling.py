"""Hierarchical prediction, association, geometry, and null helpers for C78S."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import math
from typing import Any

import numpy as np
from scipy import linalg, stats

from . import c75_modeling
from . import c76_statistics
from . import c78s_protocol as protocol


def center_within_groups(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return c75_modeling.center_within_groups(np.asarray(values, dtype=float), np.asarray(groups))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    return c75_modeling.safe_spearman(np.asarray(left, dtype=float), np.asarray(right, dtype=float))


def pairwise_accuracy(y: np.ndarray, score: np.ndarray) -> float:
    return c75_modeling.pairwise_accuracy(np.asarray(y, dtype=float), np.asarray(score, dtype=float))


def centered_r2(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> float:
    yc = center_within_groups(np.asarray(y, dtype=float)[:, None], groups)[:, 0]
    denominator = float(np.sum(yc ** 2))
    if denominator <= 1e-15:
        return math.nan
    return 1.0 - float(np.sum((yc - prediction) ** 2)) / denominator


def _select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    inner_groups: np.ndarray,
) -> float:
    losses = {alpha: [] for alpha in protocol.RIDGE_ALPHAS}
    groups = sorted(set(inner_groups[train_indices].tolist()))
    if len(groups) < 2:
        return 1.0
    for held_group in groups:
        inner_test = train_indices[inner_groups[train_indices] == held_group]
        inner_train = train_indices[inner_groups[train_indices] != held_group]
        if len(inner_test) == 0 or len(inner_train) == 0:
            continue
        for alpha in protocol.RIDGE_ALPHAS:
            prediction, _ = c75_modeling.ridge_fold_predict(
                X[inner_train], y[inner_train], X[inner_test], alpha, column_space=True,
            )
            losses[alpha].append(float(np.mean((y[inner_test] - prediction) ** 2)))
    return min(
        protocol.RIDGE_ALPHAS,
        key=lambda alpha: (float(np.mean(losses[alpha])) if losses[alpha] else math.inf, alpha),
    )


@dataclass
class CrossfitResult:
    prediction: np.ndarray
    fold_rows: list[dict[str, Any]]
    alphas: dict[str, float]


def crossfit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    outer_groups: np.ndarray,
    inner_groups: np.ndarray,
    center_groups: np.ndarray,
    fixed_alphas: dict[str, float] | None = None,
) -> CrossfitResult:
    Xc = center_within_groups(X, center_groups)
    yc = center_within_groups(np.asarray(y, dtype=float)[:, None], center_groups)[:, 0]
    prediction = np.empty(len(yc), dtype=float)
    fold_rows = []
    alphas = {}
    for held_group in sorted(set(outer_groups.tolist())):
        test = np.where(outer_groups == held_group)[0]
        train = np.where(outer_groups != held_group)[0]
        alpha = (
            float(fixed_alphas[str(held_group)])
            if fixed_alphas is not None
            else _select_alpha(Xc, yc, train, inner_groups)
        )
        prediction[test], audit = c75_modeling.ridge_fold_predict(
            Xc[train], yc[train], Xc[test], alpha, column_space=True,
        )
        alphas[str(held_group)] = alpha
        fold_rows.append({
            "held_group": str(held_group),
            "train_rows": len(train),
            "test_rows": len(test),
            "alpha": alpha,
            **audit,
        })
    return CrossfitResult(prediction=prediction, fold_rows=fold_rows, alphas=alphas)


def per_target_increment(
    y: np.ndarray,
    prior: np.ndarray,
    full: np.ndarray,
    targets: np.ndarray,
    center_groups: np.ndarray,
) -> list[dict[str, Any]]:
    yc = center_within_groups(np.asarray(y, dtype=float)[:, None], center_groups)[:, 0]
    rows = []
    for target in sorted(set(targets.tolist())):
        mask = targets == target
        residual = yc[mask] - prior[mask]
        increment = full[mask] - prior[mask]
        prior_rho = safe_spearman(yc[mask], prior[mask])
        full_rho = safe_spearman(yc[mask], full[mask])
        increment_rho = safe_spearman(residual, increment)
        denominator = float(np.sum(residual ** 2))
        incremental_r2 = (
            1.0 - float(np.sum((residual - increment) ** 2)) / denominator
            if denominator > 1e-15 else math.nan
        )
        rows.append({
            "target_id": int(target),
            "prior_rho": prior_rho,
            "full_rho": full_rho,
            "delta_rho": full_rho - prior_rho if math.isfinite(prior_rho) and math.isfinite(full_rho) else math.nan,
            "increment_residual_rho": increment_rho,
            "incremental_R2": incremental_r2,
            "positive_increment": int(math.isfinite(increment_rho) and increment_rho > 0),
        })
    return rows


def cell_actionability(
    utility: np.ndarray,
    joint_good: np.ndarray,
    scores: dict[str, np.ndarray],
    cell_ids: np.ndarray,
    targets: np.ndarray,
    levels: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for cell in sorted(set(cell_ids.tolist())):
        indices = np.where(cell_ids == cell)[0]
        true_order = np.argsort(utility[indices])[::-1]
        best = float(np.max(utility[indices]))
        worst = float(np.min(utility[indices]))
        row: dict[str, Any] = {
            "cell_id": str(cell),
            "target_id": int(targets[indices[0]]),
            "level": int(levels[indices[0]]),
            "candidate_count": len(indices),
            "joint_good_prevalence": float(np.mean(joint_good[indices] > 0.5)),
            "utility_range": best - worst,
            "random_top1": 1.0 / len(indices),
            "random_top5": min(5, len(indices)) / len(indices),
            "random_top10": min(10, len(indices)) / len(indices),
            "random_expected_regret": best - float(np.mean(utility[indices])),
        }
        good_count = int(np.sum(joint_good[indices] > 0.5))
        for k in (1, 5, 10):
            if good_count == 0:
                random_coverage = 0.0
            else:
                misses = math.comb(len(indices) - good_count, min(k, len(indices)))
                total = math.comb(len(indices), min(k, len(indices)))
                random_coverage = 1.0 - misses / total if total else 0.0
            row[f"random_joint_good_coverage_top{k}"] = random_coverage
        for name, score in scores.items():
            local_score = score[indices]
            predicted_order = np.argsort(local_score)[::-1]
            selected = int(predicted_order[0])
            selected_true_rank = int(np.where(true_order == selected)[0][0]) + 1
            row[f"{name}_spearman"] = safe_spearman(utility[indices], local_score)
            row[f"{name}_pairwise"] = pairwise_accuracy(utility[indices], local_score)
            row[f"{name}_selected_true_rank"] = selected_true_rank
            row[f"{name}_regret"] = best - float(utility[indices][selected])
            row[f"{name}_standardized_regret"] = (
                row[f"{name}_regret"] / (best - worst) if best - worst > 1e-15 else 0.0
            )
            for k in (1, 5, 10):
                top_predicted = set(map(int, predicted_order[:k]))
                row[f"{name}_oracle_best_in_predicted_top{k}"] = int(int(true_order[0]) in top_predicted)
                row[f"{name}_selected_within_true_top{k}"] = int(selected_true_rank <= k)
                row[f"{name}_joint_good_coverage_top{k}"] = int(
                    np.any(joint_good[indices][list(top_predicted)] > 0.5)
                )
        rows.append(row)
    return rows


def summarize_actionability(
    rows: list[dict[str, Any]],
    prior_name: str,
    full_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cells": len(rows),
        "targets": len({int(row["target_id"]) for row in rows}),
    }
    for k in (1, 5, 10):
        for metric in ("oracle_best_in_predicted_top", "selected_within_true_top", "joint_good_coverage_top"):
            prior = float(np.mean([row[f"{prior_name}_{metric}{k}"] for row in rows]))
            full = float(np.mean([row[f"{full_name}_{metric}{k}"] for row in rows]))
            result[f"prior_{metric}{k}"] = prior
            result[f"full_{metric}{k}"] = full
            result[f"delta_{metric}{k}"] = full - prior
    prior_regret = float(np.mean([row[f"{prior_name}_standardized_regret"] for row in rows]))
    full_regret = float(np.mean([row[f"{full_name}_standardized_regret"] for row in rows]))
    result.update({
        "prior_standardized_regret": prior_regret,
        "full_standardized_regret": full_regret,
        "standardized_regret_reduction": prior_regret - full_regret,
    })
    per_target = defaultdict(list)
    for row in rows:
        per_target[int(row["target_id"])].append(row)
    target_regret = []
    for target_rows in per_target.values():
        target_regret.append(float(np.mean([
            row[f"{prior_name}_standardized_regret"] - row[f"{full_name}_standardized_regret"]
            for row in target_rows
        ])))
    result["positive_regret_reduction_targets"] = int(np.sum(np.asarray(target_regret) > 0))
    result["material_topk"] = int(max(
        result["delta_oracle_best_in_predicted_top5"],
        result["delta_oracle_best_in_predicted_top10"],
    ) >= 0.05)
    result["material_regret"] = int(result["standardized_regret_reduction"] >= 0.05)
    result["material_actionability"] = int(result["material_topk"] or result["material_regret"])
    return result


def exact_sign_flip_p(values: np.ndarray, *, two_sided: bool = True) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    observed = abs(float(np.mean(values))) if two_sided else float(np.mean(values))
    distribution = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        candidate = float(np.mean(values * np.asarray(signs)))
        distribution.append(abs(candidate) if two_sided else candidate)
    return (1 + int(np.sum(np.asarray(distribution) >= observed - 1e-15))) / (1 + len(distribution))


def holm_adjust(rows: list[dict[str, Any]], p_key: str = "raw_p") -> list[dict[str, Any]]:
    finite = [(index, float(row[p_key])) for index, row in enumerate(rows) if math.isfinite(float(row[p_key]))]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted = [math.nan] * len(rows)
    running = 0.0
    m = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * value))
        adjusted[index] = running
    output = []
    for index, row in enumerate(rows):
        output.append({
            **row,
            "Holm_p": adjusted[index],
            "Holm_reject_0.05": int(math.isfinite(adjusted[index]) and adjusted[index] < 0.05),
        })
    return output


def blocked_permutation(
    scheme: str,
    arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    targets = arrays["target_id"].astype(int)
    levels = arrays["level"].astype(int)
    regimes = arrays["regime"].astype(str)
    orders = arrays["candidate_order"].astype(int)
    trajectories = arrays["trajectory_id"].astype(str)
    permutation = np.arange(len(targets))
    if scheme == "target_block_permutation":
        target_values = sorted(set(targets.tolist()))
        mapped = dict(zip(target_values, rng.permutation(target_values)))
        lookup = {
            (int(targets[index]), int(levels[index]), str(regimes[index]), int(orders[index])): index
            for index in range(len(targets))
        }
        for index in range(len(targets)):
            permutation[index] = lookup[(mapped[int(targets[index])], int(levels[index]), str(regimes[index]), int(orders[index]))]
        return permutation
    if scheme == "checkpoint_block_permutation":
        for level in protocol.LEVELS:
            for regime in protocol.REGIMES:
                order_values = sorted(set(orders[(levels == level) & (regimes == regime)].tolist()))
                mapping = dict(zip(order_values, rng.permutation(order_values)))
                lookup = {
                    (int(targets[index]), int(orders[index])): index
                    for index in np.where((levels == level) & (regimes == regime))[0]
                }
                for index in np.where((levels == level) & (regimes == regime))[0]:
                    permutation[index] = lookup[(int(targets[index]), mapping[int(orders[index])])]
        return permutation
    if scheme == "trajectory_preserving_permutation":
        groups = trajectories
    elif scheme == "candidate_within_target_regime_permutation":
        groups = np.asarray([
            f"target-{target}|{regime}" for target, regime in zip(targets, regimes)
        ])
    elif scheme == "nested_bandwidth_null":
        groups = trajectories
    else:
        raise ValueError(scheme)
    for group in sorted(set(groups.tolist())):
        indices = np.where(groups == group)[0]
        permutation[indices] = rng.permutation(indices)
    return permutation


def matched_gaussian_features(
    features: np.ndarray,
    metadata: np.ndarray,
    center_groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return c76_statistics.matched_gaussian_features(features, metadata, center_groups, rng)


def association_statistic(kernel: np.ndarray, outcome: np.ndarray, statistic: str) -> float:
    y = np.asarray(outcome, dtype=float)
    y = y - float(np.mean(y))
    yy = y[:, None] * y[None, :]
    if statistic == "normalized_alignment":
        mask = ~np.eye(len(y), dtype=bool)
        left, right = kernel[mask], yy[mask]
    elif statistic == "centered_hsic":
        centered_kernel = (
            kernel
            - np.mean(kernel, axis=0, keepdims=True)
            - np.mean(kernel, axis=1, keepdims=True)
            + float(np.mean(kernel))
        )
        left, right = centered_kernel.ravel(), yy.ravel()
    else:
        raise ValueError(statistic)
    denominator = math.sqrt(
        max(float(np.sum(left ** 2)), 1e-15)
        * max(float(np.sum(right ** 2)), 1e-15)
    )
    return float(np.sum(left * right)) / denominator


def _scale_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std[std < 1e-12] = 1.0
    return (train - mean) / std, (test - mean) / std


def _pairwise_distances(features: np.ndarray) -> np.ndarray:
    squared = np.maximum(
        np.sum(features ** 2, axis=1)[:, None]
        + np.sum(features ** 2, axis=1)[None, :]
        - 2.0 * features @ features.T,
        0.0,
    )
    return np.sqrt(squared)


def _median_positive_distance(features: np.ndarray) -> float:
    distances = _pairwise_distances(features)
    values = distances[np.triu_indices(len(features), 1)]
    positive = values[values > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def _kernel(distances: np.ndarray, bandwidth: float, family: str) -> np.ndarray:
    bandwidth = max(float(bandwidth), 1e-12)
    if family == "rbf":
        return np.exp(-(distances ** 2) / (2.0 * bandwidth ** 2))
    if family == "laplacian":
        return np.exp(-distances / bandwidth)
    raise ValueError(family)


def crossfit_association(
    features: np.ndarray,
    outcome: np.ndarray,
    targets: np.ndarray,
    center_groups: np.ndarray,
    *,
    kernel_family: str,
    bandwidth_factor: float,
    statistic: str,
) -> tuple[float, list[dict[str, Any]]]:
    X = center_within_groups(features, center_groups)
    y = center_within_groups(np.asarray(outcome, dtype=float)[:, None], center_groups)[:, 0]
    folds = []
    for target in sorted(set(targets.tolist())):
        train = targets != target
        test = targets == target
        train_scaled, test_scaled = _scale_train_test(X[train], X[test])
        bandwidth = bandwidth_factor * _median_positive_distance(train_scaled)
        kernel = _kernel(_pairwise_distances(test_scaled), bandwidth, kernel_family)
        folds.append({
            "target_id": int(target),
            "association": association_statistic(kernel, y[test], statistic),
            "bandwidth": bandwidth,
            "rows": int(np.sum(test)),
        })
    return float(np.mean([row["association"] for row in folds])), folds


def association_family(
    feature_paths: dict[str, np.ndarray],
    residual_paths: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    targets = arrays["target_id"].astype(int)
    cells = arrays["cell_id"].astype(str)
    rows = []
    for path, features in feature_paths.items():
        X = center_within_groups(features, cells)
        y = center_within_groups(np.asarray(residual_paths[path], dtype=float)[:, None], cells)[:, 0]
        family_folds: dict[tuple[str, float, str], list[dict[str, Any]]] = {
            (kernel, factor, statistic): []
            for kernel in protocol.KERNEL_FAMILIES
            for factor in protocol.BANDWIDTH_FACTORS
            for statistic in protocol.ASSOCIATION_STATISTICS
        }
        for target in sorted(set(targets.tolist())):
            train = targets != target
            test = targets == target
            train_scaled, test_scaled = _scale_train_test(X[train], X[test])
            base_bandwidth = _median_positive_distance(train_scaled)
            distances = _pairwise_distances(test_scaled)
            for kernel_family in protocol.KERNEL_FAMILIES:
                for factor in protocol.BANDWIDTH_FACTORS:
                    bandwidth = factor * base_bandwidth
                    kernel = _kernel(distances, bandwidth, kernel_family)
                    for statistic in protocol.ASSOCIATION_STATISTICS:
                        family_folds[(kernel_family, factor, statistic)].append({
                            "target_id": int(target),
                            "association": association_statistic(kernel, y[test], statistic),
                            "bandwidth": bandwidth,
                            "rows": int(np.sum(test)),
                        })
        for (kernel_family, factor, statistic), folds in family_folds.items():
            values = np.asarray([row["association"] for row in folds], dtype=float)
            rows.append({
                "path": path,
                "kernel": kernel_family,
                "bandwidth_factor": factor,
                "statistic": statistic,
                "association": float(np.mean(values)),
                "median_target_association": float(np.median(values)),
                "positive_targets": int(np.sum(values > 0)),
                "folds": folds,
            })
    return rows


def topology_association(
    features: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    *,
    kernel_family: str,
    bandwidth_factor: float,
    statistic: str,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if int(np.sum(mask)) < 4:
            continue
        X, _ = _scale_train_test(features[mask], features[mask])
        bandwidth = bandwidth_factor * _median_positive_distance(X)
        kernel = _kernel(_pairwise_distances(X), bandwidth, kernel_family)
        rows.append({
            "group": str(group),
            "rows": int(np.sum(mask)),
            "association": association_statistic(kernel, outcome[mask], statistic),
        })
    return (float(np.mean([row["association"] for row in rows])) if rows else math.nan), rows


def _cross_distances(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    squared = np.maximum(
        np.sum(test ** 2, axis=1)[:, None]
        + np.sum(train ** 2, axis=1)[None, :]
        - 2.0 * test @ train.T,
        0.0,
    )
    return np.sqrt(squared)


def crossfit_krr_fixed(
    features: np.ndarray,
    residual: np.ndarray,
    targets: np.ndarray,
    center_groups: np.ndarray,
    *,
    kernel_family: str,
    bandwidth_factor: float,
    alpha: float,
    outer_groups: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    X = center_within_groups(features, center_groups)
    y = center_within_groups(np.asarray(residual, dtype=float)[:, None], center_groups)[:, 0]
    prediction = np.empty(len(y), dtype=float)
    folds = []
    outer = targets if outer_groups is None else np.asarray(outer_groups)
    for held_group in sorted(set(outer.tolist())):
        train = np.where(outer != held_group)[0]
        test = np.where(outer == held_group)[0]
        train_X, test_X = _scale_train_test(X[train], X[test])
        bandwidth = bandwidth_factor * _median_positive_distance(train_X)
        train_kernel = _kernel(_pairwise_distances(train_X), bandwidth, kernel_family)
        cross_kernel = _kernel(_cross_distances(train_X, test_X), bandwidth, kernel_family)
        mean = float(np.mean(y[train]))
        coefficient = linalg.solve(
            train_kernel + (alpha + 1e-10) * np.eye(len(train)),
            y[train] - mean,
            assume_a="pos",
        )
        prediction[test] = mean + cross_kernel @ coefficient
        folds.append({
            "held_group": str(held_group),
            "kernel": kernel_family,
            "bandwidth_factor": bandwidth_factor,
            "bandwidth": bandwidth,
            "alpha": alpha,
            "train_rows": len(train),
            "test_rows": len(test),
        })
    return prediction, folds


def krr_trajectory_nulls(
    features: np.ndarray,
    residual: np.ndarray,
    prior_prediction: np.ndarray,
    outcome: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    kernel_family: str,
    bandwidth_factor: float,
    alpha: float,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Vectorized fixed-family KRR nulls with one factorization per target."""

    targets = arrays["target_id"].astype(int)
    cells = arrays["cell_id"].astype(str)
    trajectories = arrays["trajectory_id"].astype(str)
    X = center_within_groups(features, cells)
    y = center_within_groups(np.asarray(residual, dtype=float)[:, None], cells)[:, 0]
    rng = np.random.default_rng(seed)
    permutations = np.empty((replicates, len(y)), dtype=np.int32)
    for replicate in range(replicates):
        mapping = np.arange(len(y), dtype=np.int32)
        for trajectory in sorted(set(trajectories.tolist())):
            indices = np.where(trajectories == trajectory)[0]
            mapping[indices] = rng.permutation(indices)
        permutations[replicate] = mapping
    null_predictions = np.empty((len(y), replicates), dtype=float)
    for target in sorted(set(targets.tolist())):
        train = np.where(targets != target)[0]
        test = np.where(targets == target)[0]
        train_lookup = {int(index): position for position, index in enumerate(train)}
        test_lookup = {int(index): position for position, index in enumerate(test)}
        train_X, test_X = _scale_train_test(X[train], X[test])
        bandwidth = bandwidth_factor * _median_positive_distance(train_X)
        train_kernel = _kernel(_pairwise_distances(train_X), bandwidth, kernel_family)
        cross_kernel = _kernel(_cross_distances(train_X, test_X), bandwidth, kernel_family)
        factor = linalg.cho_factor(
            train_kernel + (alpha + 1e-10) * np.eye(len(train)),
            lower=True,
            check_finite=False,
        )
        train_mean = float(np.mean(y[train]))
        rhs = np.empty((len(train), replicates), dtype=float)
        test_maps = np.empty((len(test), replicates), dtype=np.int32)
        for replicate in range(replicates):
            train_map = np.asarray([train_lookup[int(index)] for index in permutations[replicate, train]], dtype=np.int32)
            test_map = np.asarray([test_lookup[int(index)] for index in permutations[replicate, test]], dtype=np.int32)
            inverse_train_map = np.argsort(train_map)
            rhs[:, replicate] = y[train][inverse_train_map] - train_mean
            test_maps[:, replicate] = test_map
        coefficients = linalg.cho_solve(factor, rhs, check_finite=False)
        base_predictions = train_mean + cross_kernel @ coefficients
        for replicate in range(replicates):
            null_predictions[test, replicate] = base_predictions[test_maps[:, replicate], replicate]
    prior_r2 = centered_r2(outcome, prior_prediction, cells)
    return np.asarray([
        centered_r2(outcome, prior_prediction + null_predictions[:, replicate], cells) - prior_r2
        for replicate in range(replicates)
    ])


def target_cluster_bootstrap(
    values_by_target: dict[int, float],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    targets = np.asarray(sorted(values_by_target), dtype=int)
    values = np.asarray([values_by_target[int(target)] for target in targets], dtype=float)
    return np.asarray([
        float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
        for _ in range(replicates)
    ])


def _logistic_fit(X: np.ndarray, y: np.ndarray, penalty: float = 1e-4) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-12] = 1.0
    design = np.column_stack((np.ones(len(X)), (X - mean) / std))
    beta = np.zeros(design.shape[1], dtype=float)
    penalty_matrix = np.eye(len(beta)) * penalty
    penalty_matrix[0, 0] = 0.0
    for _ in range(100):
        eta = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(probability * (1.0 - probability), 1e-8, None)
        gradient = design.T @ (y - probability) - penalty_matrix @ beta
        hessian = design.T @ (weight[:, None] * design) + penalty_matrix
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return beta, float(mean[0]) if len(mean) else 0.0, mean, std


def logistic_predict(model: tuple[np.ndarray, float, np.ndarray, np.ndarray], X: np.ndarray) -> np.ndarray:
    beta, _, mean, std = model
    design = np.column_stack((np.ones(len(X)), (X - mean) / std))
    return 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30.0, 30.0)))


def crossfit_logistic_deviance(
    raw_X: np.ndarray,
    full_X: np.ndarray,
    y: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    raw_prediction = np.empty(len(y), dtype=float)
    full_prediction = np.empty(len(y), dtype=float)
    coefficient_rows = []
    for target in sorted(set(targets.tolist())):
        train = targets != target
        test = targets == target
        raw_model = _logistic_fit(raw_X[train], y[train])
        full_model = _logistic_fit(full_X[train], y[train])
        raw_prediction[test] = logistic_predict(raw_model, raw_X[test])
        full_prediction[test] = logistic_predict(full_model, full_X[test])
        coefficient_rows.append({
            "held_target": int(target),
            "effective_M_coefficient": float(full_model[0][2]) if full_X.shape[1] >= 2 else math.nan,
            "inverse_gap_coefficient": float(full_model[0][3]) if full_X.shape[1] >= 3 else math.nan,
        })
    def deviance(prediction: np.ndarray) -> float:
        clipped = np.clip(prediction, 1e-10, 1.0 - 1e-10)
        return float(-2.0 * np.sum(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))
    raw_deviance = deviance(raw_prediction)
    full_deviance = deviance(full_prediction)
    return {
        "raw_deviance": raw_deviance,
        "full_deviance": full_deviance,
        "incremental_deviance_reduction": raw_deviance - full_deviance,
        "raw_prediction": raw_prediction,
        "full_prediction": full_prediction,
        "coefficient_rows": coefficient_rows,
    }
