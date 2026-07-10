"""Registered C76 nonlinear association, kernel prediction, and control helpers."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np

from . import c75_modeling
from . import c76_protocol


def center_within_groups(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return c75_modeling.center_within_groups(values, groups)


def scale_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std[std < 1e-12] = 1.0
    return (train - mean) / std, (test - mean) / std


def pairwise_squared_distances(features: np.ndarray) -> np.ndarray:
    # Preserve C75's registered arithmetic order for exact RBF replay.
    return np.sum(
        (features[:, None, :] - features[None, :, :]) ** 2,
        axis=2,
    )


def pairwise_distances(features: np.ndarray) -> np.ndarray:
    return np.sqrt(pairwise_squared_distances(features))


def median_positive_distance(features: np.ndarray) -> float:
    distances = pairwise_distances(features)
    values = distances[np.triu_indices(len(features), 1)]
    positive = values[values > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def kernel_from_distances(distances: np.ndarray, bandwidth: float, family: str) -> np.ndarray:
    bandwidth = max(float(bandwidth), 1e-12)
    if family == "rbf":
        return np.exp(-(distances ** 2) / (2.0 * bandwidth ** 2))
    if family == "laplacian":
        return np.exp(-distances / bandwidth)
    raise ValueError(family)


def association_statistic(kernel: np.ndarray, outcome: np.ndarray, statistic: str) -> float:
    outcome = np.asarray(outcome, dtype=float)
    outcome = outcome - float(np.mean(outcome))
    outcome_kernel = outcome[:, None] * outcome[None, :]
    if statistic == "normalized_alignment":
        mask = ~np.eye(len(outcome), dtype=bool)
        left, right = kernel[mask], outcome_kernel[mask]
    elif statistic == "centered_hsic":
        H = np.eye(len(outcome)) - np.ones((len(outcome), len(outcome))) / len(outcome)
        left = (H @ kernel @ H).ravel()
        right = (H @ outcome_kernel @ H).ravel()
    else:
        raise ValueError(statistic)
    denominator = math.sqrt(
        max(float(np.sum(left ** 2)), 1e-15)
        * max(float(np.sum(right ** 2)), 1e-15)
    )
    return float(np.sum(left * right)) / denominator


def crossfit_association(
    features: np.ndarray, outcome: np.ndarray, targets: np.ndarray,
    *, kernel_family: str, bandwidth_factor: float, statistic: str,
) -> tuple[float, list[dict]]:
    X = center_within_groups(np.asarray(features, dtype=float), targets)
    y = center_within_groups(np.asarray(outcome, dtype=float)[:, None], targets)[:, 0]
    fold_rows = []
    for held_target in sorted(set(targets.tolist())):
        train = targets != held_target
        test = targets == held_target
        train_scaled, test_scaled = scale_train_test(X[train], X[test])
        train_squared = pairwise_squared_distances(train_scaled)
        train_upper = train_squared[np.triu_indices(len(train_scaled), 1)]
        train_positive = train_upper[train_upper > 1e-15]
        median_distance = math.sqrt(float(np.median(train_positive))) if len(train_positive) else 1.0
        bandwidth = bandwidth_factor * median_distance
        test_squared = pairwise_squared_distances(test_scaled)
        if kernel_family == "rbf":
            kernel = np.exp(-test_squared / (2.0 * max(bandwidth, 1e-12) ** 2))
        else:
            kernel = kernel_from_distances(np.sqrt(test_squared), bandwidth, kernel_family)
        fold_rows.append({
            "target_id": int(held_target),
            "statistic_value": association_statistic(kernel, y[test], statistic),
            "training_bandwidth": bandwidth,
            "candidate_count": int(np.sum(test)),
        })
    return float(np.mean([row["statistic_value"] for row in fold_rows])), fold_rows


def association_family(
    feature_paths: dict[str, np.ndarray], residual_paths: dict[str, np.ndarray], targets: np.ndarray,
) -> list[dict]:
    rows = []
    for path in ("strict_source", "target_unlabeled"):
        for kernel in c76_protocol.KERNEL_FAMILIES:
            for factor in c76_protocol.BANDWIDTH_FACTORS:
                for statistic in c76_protocol.ASSOCIATION_STATISTICS:
                    value, folds = crossfit_association(
                        feature_paths[path], residual_paths[path], targets,
                        kernel_family=kernel, bandwidth_factor=factor, statistic=statistic,
                    )
                    rows.append({
                        "path": path, "kernel": kernel, "bandwidth_factor": factor,
                        "statistic": statistic, "association": value,
                        "median_target_association": float(np.median([row["statistic_value"] for row in folds])),
                        "positive_targets": sum(row["statistic_value"] > 0 for row in folds),
                        "fold_rows": folds,
                    })
    return rows


def blocked_permutation(
    scheme: str, targets: np.ndarray, trajectory: np.ndarray,
    seed: np.ndarray, level: np.ndarray, candidate_order: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(targets)
    permutation = np.arange(n)
    if scheme == "N1_target_block":
        target_values = sorted(set(targets.tolist()))
        mapping = dict(zip(target_values, rng.permutation(target_values)))
        keys = {}
        for index in range(n):
            keys[(int(targets[index]), int(seed[index]), int(level[index]), int(candidate_order[index]))] = index
        for index in range(n):
            permutation[index] = keys[(mapping[int(targets[index])], int(seed[index]), int(level[index]), int(candidate_order[index]))]
        return permutation
    if scheme == "N2_checkpoint_block":
        groups = [(int(seed[i]), int(level[i]), int(candidate_order[i])) for i in range(n)]
    elif scheme == "N3_trajectory_preserving":
        groups = [(int(targets[i]), str(trajectory[i])) for i in range(n)]
    elif scheme == "N4_candidate_within_target":
        groups = [(int(targets[i]),) for i in range(n)]
    else:
        raise ValueError(scheme)
    for group in sorted(set(groups)):
        indices = np.asarray([index for index, value in enumerate(groups) if value == group], dtype=int)
        permutation[indices] = rng.permutation(indices)
    return permutation


def matched_gaussian_features(
    features: np.ndarray, metadata: np.ndarray, targets: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    X = center_within_groups(np.asarray(features, dtype=float), targets)
    M = center_within_groups(np.asarray(metadata, dtype=float), targets)
    beta = np.linalg.pinv(M) @ X
    smooth = M @ beta
    residual = X - smooth
    covariance = np.cov(residual, rowvar=False)
    covariance = np.atleast_2d(covariance) + 1e-8 * np.eye(residual.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    square_root = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T
    noise = rng.normal(size=residual.shape) @ square_root.T
    return smooth + noise


def topology_association(
    features: np.ndarray, outcome: np.ndarray, groups: np.ndarray,
    *, kernel_family: str = "rbf", bandwidth_factor: float = 1.0,
    statistic: str = "normalized_alignment",
) -> tuple[float, list[float]]:
    X = np.asarray(features, dtype=float)
    y = np.asarray(outcome, dtype=float)
    values = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if int(np.sum(mask)) < 4:
            continue
        X_group = X[mask]
        y_group = y[mask]
        X_scaled, _ = scale_train_test(X_group, X_group)
        bandwidth = bandwidth_factor * median_positive_distance(X_scaled)
        kernel = kernel_from_distances(pairwise_distances(X_scaled), bandwidth, kernel_family)
        values.append(association_statistic(kernel, y_group, statistic))
    return (float(np.mean(values)) if values else math.nan), values


def candidate_density_order(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    X = center_within_groups(np.asarray(features, dtype=float), targets)
    scores = np.empty(len(X), dtype=float)
    for target in sorted(set(targets.tolist())):
        mask = targets == target
        scaled, _ = scale_train_test(X[mask], X[mask])
        bandwidth = median_positive_distance(scaled)
        kernel = kernel_from_distances(pairwise_distances(scaled), bandwidth, "rbf")
        np.fill_diagonal(kernel, 0.0)
        scores[mask] = np.mean(kernel, axis=1)
    return scores


def orbit_order_spearman(reference: np.ndarray, alternative: np.ndarray, targets: np.ndarray) -> float:
    values = []
    for target in sorted(set(targets.tolist())):
        mask = targets == target
        values.append(c75_modeling.safe_spearman(reference[mask], alternative[mask]))
    return float(np.nanmedian(values))


def _kernel_train_test(
    train_X: np.ndarray, test_X: np.ndarray, family: str, bandwidth_factor: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_scaled, test_scaled = scale_train_test(train_X, test_X)
    bandwidth = bandwidth_factor * median_positive_distance(train_scaled)
    train_distances = pairwise_distances(train_scaled)
    cross_squared = np.maximum(
        np.sum(test_scaled ** 2, axis=1)[:, None]
        + np.sum(train_scaled ** 2, axis=1)[None, :]
        - 2.0 * test_scaled @ train_scaled.T,
        0.0,
    )
    cross_distances = np.sqrt(cross_squared)
    return (
        kernel_from_distances(train_distances, bandwidth, family),
        kernel_from_distances(cross_distances, bandwidth, family),
        bandwidth,
    )


def kernel_ridge_predict(
    train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray,
    family: str, bandwidth_factor: float, alpha: float,
) -> np.ndarray:
    train_kernel, test_kernel, _ = _kernel_train_test(train_X, test_X, family, bandwidth_factor)
    centered_y = train_y - float(np.mean(train_y))
    coefficient = np.linalg.solve(
        train_kernel + (float(alpha) + 1e-10) * np.eye(len(train_kernel)), centered_y,
    )
    return float(np.mean(train_y)) + test_kernel @ coefficient


def select_krr_hyperparameters(
    X: np.ndarray, y: np.ndarray, targets: np.ndarray, train_indices: np.ndarray,
) -> tuple[str, float, float]:
    candidates = list(itertools.product(
        c76_protocol.KERNEL_FAMILIES, c76_protocol.BANDWIDTH_FACTORS, c76_protocol.KRR_ALPHAS,
    ))
    losses = {candidate: [] for candidate in candidates}
    for held_target in sorted(set(targets[train_indices].tolist())):
        inner_test = train_indices[targets[train_indices] == held_target]
        inner_train = train_indices[targets[train_indices] != held_target]
        for family, factor, alpha in candidates:
            prediction = kernel_ridge_predict(
                X[inner_train], y[inner_train], X[inner_test], family, factor, alpha,
            )
            losses[(family, factor, alpha)].append(float(np.mean((y[inner_test] - prediction) ** 2)))
    return min(candidates, key=lambda candidate: (float(np.mean(losses[candidate])), candidate))


@dataclass
class KrrCrossfitResult:
    prediction: np.ndarray
    fold_rows: list[dict]


def crossfit_krr(
    features: np.ndarray, outcome: np.ndarray, targets: np.ndarray,
    *, fixed_hyperparameters: dict[int, tuple[str, float, float]] | None = None,
) -> KrrCrossfitResult:
    X = center_within_groups(np.asarray(features, dtype=float), targets)
    y = center_within_groups(np.asarray(outcome, dtype=float)[:, None], targets)[:, 0]
    prediction = np.empty(len(y), dtype=float)
    fold_rows = []
    for held_target in sorted(set(targets.tolist())):
        test = np.where(targets == held_target)[0]
        train = np.where(targets != held_target)[0]
        hyperparameters = (
            fixed_hyperparameters[int(held_target)]
            if fixed_hyperparameters is not None
            else select_krr_hyperparameters(X, y, targets, train)
        )
        family, factor, alpha = hyperparameters
        prediction[test] = kernel_ridge_predict(X[train], y[train], X[test], family, factor, alpha)
        fold_rows.append({
            "held_target": int(held_target), "kernel": family,
            "bandwidth_factor": factor, "alpha": alpha,
        })
    return KrrCrossfitResult(prediction=prediction, fold_rows=fold_rows)


def hyperparameter_map(result: KrrCrossfitResult) -> dict[int, tuple[str, float, float]]:
    return {
        int(row["held_target"]): (
            str(row["kernel"]), float(row["bandwidth_factor"]), float(row["alpha"]),
        ) for row in result.fold_rows
    }


def exact_sign_permutation_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))
    distribution = [
        float(np.mean(values * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return (1 + sum(value >= observed for value in distribution)) / (1 + len(distribution))


def bootstrap_target_mean(values: np.ndarray, repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    return np.asarray([
        float(np.mean(rng.choice(values, size=len(values), replace=True)))
        for _ in range(repeats)
    ])
