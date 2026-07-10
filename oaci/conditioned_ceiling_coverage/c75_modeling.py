"""Registered C75 cross-fitting, null, actionability, and kernel-proxy helpers."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import stats

from . import c75_protocol


def center_within_groups(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values.copy()
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        centered[mask] -= np.mean(centered[mask], axis=0, keepdims=True)
    return centered


def _scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std[std < 1e-12] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def _column_basis(train: np.ndarray, tolerance: float = c75_protocol.SVD_RANK_TOLERANCE) -> tuple[np.ndarray, int, np.ndarray]:
    if train.shape[1] == 0:
        return np.empty((0, 0)), 0, np.empty(0)
    _, singular, vt = np.linalg.svd(train, full_matrices=False)
    threshold = tolerance * max(float(singular[0]) if len(singular) else 0.0, 1.0)
    rank = int(np.sum(singular > threshold))
    return vt[:rank].T, rank, singular


def ridge_fold_predict(
    train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray,
    alpha: float, *, column_space: bool = True,
) -> tuple[np.ndarray, dict]:
    train_scaled, test_scaled, mean, std = _scale(train_X, test_X)
    if column_space:
        basis, rank, singular = _column_basis(train_scaled)
        if rank:
            inverse_singular = 1.0 / singular[:rank]
            # Canonical orthonormal sample-space coordinates make ridge invariant
            # to duplicated columns and invertible feature reparameterizations.
            train_design = (train_scaled @ basis) * inverse_singular
            test_design = (test_scaled @ basis) * inverse_singular
        else:
            train_design = np.empty((len(train_X), 0))
            test_design = np.empty((len(test_X), 0))
    else:
        rank = int(np.linalg.matrix_rank(train_scaled))
        singular = np.linalg.svd(train_scaled, full_matrices=False, compute_uv=False)
        train_design, test_design = train_scaled, test_scaled
    if train_design.shape[1] == 0:
        prediction = np.full(len(test_X), float(np.mean(train_y)))
    else:
        centered_y = train_y - float(np.mean(train_y))
        beta = np.linalg.solve(
            train_design.T @ train_design + float(alpha) * np.eye(train_design.shape[1]),
            train_design.T @ centered_y,
        )
        prediction = float(np.mean(train_y)) + test_design @ beta
    nonzero = singular[singular > 1e-12]
    condition = float(nonzero[0] / nonzero[-1]) if len(nonzero) else math.inf
    return prediction, {
        "rank": rank, "raw_columns": train_X.shape[1],
        "zero_scale_columns": int(np.sum(np.std(train_X, axis=0) < 1e-12)),
        "condition_number": condition,
    }


def select_alpha(
    X: np.ndarray, y: np.ndarray, targets: np.ndarray, train_indices: np.ndarray,
    *, column_space: bool = True,
) -> float:
    train_targets = sorted(set(targets[train_indices].tolist()))
    losses = {alpha: [] for alpha in c75_protocol.RIDGE_ALPHAS}
    for held_target in train_targets:
        inner_test = train_indices[targets[train_indices] == held_target]
        inner_train = train_indices[targets[train_indices] != held_target]
        for alpha in c75_protocol.RIDGE_ALPHAS:
            prediction, _ = ridge_fold_predict(
                X[inner_train], y[inner_train], X[inner_test], alpha,
                column_space=column_space,
            )
            losses[alpha].append(float(np.mean((y[inner_test] - prediction) ** 2)))
    return min(c75_protocol.RIDGE_ALPHAS, key=lambda alpha: (float(np.mean(losses[alpha])), alpha))


@dataclass
class CrossfitResult:
    prediction: np.ndarray
    alphas: dict[int, float]
    fold_rows: list[dict]


def crossfit_loto(
    X: np.ndarray, y: np.ndarray, targets: np.ndarray,
    *, column_space: bool = True, fixed_alphas: dict[int, float] | None = None,
) -> CrossfitResult:
    Xc = center_within_groups(X, targets)
    yc = center_within_groups(np.asarray(y, dtype=float)[:, None], targets)[:, 0]
    prediction = np.empty(len(yc), dtype=float)
    alphas: dict[int, float] = {}
    fold_rows = []
    for held_target in sorted(set(targets.tolist())):
        test = np.where(targets == held_target)[0]
        train = np.where(targets != held_target)[0]
        alpha = fixed_alphas[held_target] if fixed_alphas is not None else select_alpha(
            Xc, yc, targets, train, column_space=column_space,
        )
        prediction[test], audit = ridge_fold_predict(
            Xc[train], yc[train], Xc[test], alpha, column_space=column_space,
        )
        alphas[int(held_target)] = float(alpha)
        fold_rows.append({"held_target": int(held_target), "alpha": float(alpha), **audit})
    return CrossfitResult(prediction=prediction, alphas=alphas, fold_rows=fold_rows)


def crossfit_fixed_holdout(
    X: np.ndarray, y: np.ndarray, targets: np.ndarray, holdout: np.ndarray,
    *, alpha: float, column_space: bool = True,
) -> np.ndarray:
    Xc = center_within_groups(X, targets)
    yc = center_within_groups(np.asarray(y, dtype=float)[:, None], targets)[:, 0]
    prediction = np.empty(len(yc), dtype=float)
    for group in sorted(set(holdout.tolist())):
        test = np.where(holdout == group)[0]
        train = np.where(holdout != group)[0]
        prediction[test], _ = ridge_fold_predict(
            Xc[train], yc[train], Xc[test], alpha, column_space=column_space,
        )
    return prediction


def r2(y: np.ndarray, prediction: np.ndarray, targets: np.ndarray) -> float:
    yc = center_within_groups(np.asarray(y, dtype=float)[:, None], targets)[:, 0]
    denominator = float(np.sum(yc ** 2))
    return 1.0 - float(np.sum((yc - prediction) ** 2)) / denominator if denominator else math.nan


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or float(np.std(x)) <= 1e-15 or float(np.std(y)) <= 1e-15:
        return math.nan
    result = stats.spearmanr(x, y)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def per_target_increment_rows(
    y: np.ndarray, prior: np.ndarray, full: np.ndarray, targets: np.ndarray,
) -> list[dict]:
    yc = center_within_groups(np.asarray(y, dtype=float)[:, None], targets)[:, 0]
    rows = []
    for target in sorted(set(targets.tolist())):
        mask = targets == target
        residual = yc[mask] - prior[mask]
        increment = full[mask] - prior[mask]
        prior_rho = safe_spearman(yc[mask], prior[mask])
        full_rho = safe_spearman(yc[mask], full[mask])
        increment_rho = safe_spearman(residual, increment)
        rows.append({
            "target_id": int(target), "prior_rho": prior_rho, "full_rho": full_rho,
            "delta_rho": full_rho - prior_rho if math.isfinite(prior_rho) and math.isfinite(full_rho) else math.nan,
            "increment_residual_rho": increment_rho,
            "positive_increment": int(math.isfinite(increment_rho) and increment_rho > 0),
        })
    return rows


def pairwise_accuracy(y: np.ndarray, score: np.ndarray) -> float:
    correct = 0
    comparable = 0
    for left in range(len(y)):
        for right in range(left + 1, len(y)):
            y_sign = np.sign(y[left] - y[right])
            score_sign = np.sign(score[left] - score[right])
            if y_sign and score_sign:
                comparable += 1
                correct += int(y_sign == score_sign)
    return correct / comparable if comparable else math.nan


def actionability_rows(
    y: np.ndarray, joint_good: np.ndarray, prior: np.ndarray, full: np.ndarray,
    targets: np.ndarray,
) -> list[dict]:
    rows = []
    for target in sorted(set(targets.tolist())):
        indices = np.where(targets == target)[0]
        utility = y[indices]
        true_order = np.argsort(utility)[::-1]
        target_row = {"target_id": int(target), "candidate_count": len(indices)}
        for label, score in (("prior", prior[indices]), ("full", full[indices])):
            selected = int(np.argmax(score))
            target_row.update({
                f"{label}_spearman": safe_spearman(utility, score),
                f"{label}_pairwise": pairwise_accuracy(utility, score),
                f"{label}_top1": int(selected == int(true_order[0])),
                f"{label}_top3": int(selected in set(map(int, true_order[:3]))),
                f"{label}_regret": float(np.max(utility) - utility[selected]),
                f"{label}_joint_good_coverage": int(joint_good[indices][selected] > 0.5),
            })
        for metric in ("spearman", "pairwise", "top1", "top3", "joint_good_coverage"):
            target_row[f"delta_{metric}"] = target_row[f"full_{metric}"] - target_row[f"prior_{metric}"]
        target_row["regret_reduction"] = target_row["prior_regret"] - target_row["full_regret"]
        rows.append(target_row)
    return rows


def blocked_permutation_indices(
    targets: np.ndarray, trajectory_ids: np.ndarray, rng: np.random.Generator,
    *, within_trajectory: bool,
) -> np.ndarray:
    permutation = np.arange(len(targets))
    keys = [(int(targets[index]), str(trajectory_ids[index]) if within_trajectory else "ALL") for index in range(len(targets))]
    for key in sorted(set(keys)):
        indices = np.asarray([index for index, value in enumerate(keys) if value == key], dtype=int)
        permutation[indices] = rng.permutation(indices)
    return permutation


def hierarchical_bootstrap_increment(
    y: np.ndarray, prior: np.ndarray, full: np.ndarray, targets: np.ndarray,
    *, repeats: int, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_targets = sorted(set(targets.tolist()))
    values = []
    for _ in range(repeats):
        sampled_y, sampled_prior, sampled_full, sampled_group = [], [], [], []
        for bootstrap_group, target in enumerate(rng.choice(unique_targets, size=len(unique_targets), replace=True)):
            indices = np.where(targets == target)[0]
            selected = rng.choice(indices, size=len(indices), replace=True)
            sampled_y.extend(y[selected])
            sampled_prior.extend(prior[selected])
            sampled_full.extend(full[selected])
            sampled_group.extend([bootstrap_group] * len(selected))
        sampled_y = np.asarray(sampled_y)
        sampled_prior = np.asarray(sampled_prior)
        sampled_full = np.asarray(sampled_full)
        sampled_group = np.asarray(sampled_group)
        values.append(r2(sampled_y, sampled_full, sampled_group) - r2(sampled_y, sampled_prior, sampled_group))
    return np.asarray(values)


def _median_pairwise_distance(features: np.ndarray) -> float:
    squared = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2)
    upper = squared[np.triu_indices(len(features), 1)]
    positive = upper[upper > 1e-15]
    return math.sqrt(float(np.median(positive))) if len(positive) else 1.0


def _kernel_alignment_at_bandwidth(features: np.ndarray, residual: np.ndarray, bandwidth: float) -> float:
    squared = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-squared / (2.0 * bandwidth ** 2))
    off_diagonal = ~np.eye(len(features), dtype=bool)
    yy = residual[:, None] * residual[None, :]
    numerator = float(np.sum(kernel[off_diagonal] * yy[off_diagonal]))
    denominator = math.sqrt(
        max(float(np.sum(kernel[off_diagonal] ** 2)), 1e-15)
        * max(float(np.sum(yy[off_diagonal] ** 2)), 1e-15)
    )
    return numerator / denominator


def crossfit_kernel_alignment_statistic(
    features: np.ndarray, residual: np.ndarray, targets: np.ndarray, bandwidth_factor: float,
) -> tuple[float, list[float]]:
    X = center_within_groups(features, targets)
    y = center_within_groups(residual[:, None], targets)[:, 0]
    statistics = []
    bandwidths = []
    for held_target in sorted(set(targets.tolist())):
        train = targets != held_target
        test = targets == held_target
        train_scaled, test_scaled, _, _ = _scale(X[train], X[test])
        bandwidth = max(
            bandwidth_factor * _median_pairwise_distance(train_scaled), 1e-12,
        )
        statistics.append(_kernel_alignment_at_bandwidth(test_scaled, y[test], bandwidth))
        bandwidths.append(bandwidth)
    return float(np.mean(statistics)), bandwidths
