"""C76 synthetic association-prediction separation benchmark."""
from __future__ import annotations

import math
import os

from joblib import Parallel, delayed
import numpy as np

from . import c75_modeling
from . import c76_protocol
from . import c76_statistics


CASES = (
    "S0_no_association", "S1_coordinate_artifact", "S2_pooled_identity",
    "S3_local_nonlinear_nontransport", "S4_factorization_invariant_endpoint",
    "S5_association_no_extreme_action", "S6_predictive_actionable",
)


def _fixed_kernel_crossfit(features: np.ndarray, outcome: np.ndarray, targets: np.ndarray) -> np.ndarray:
    X = c76_statistics.center_within_groups(features, targets)
    y = c76_statistics.center_within_groups(outcome[:, None], targets)[:, 0]
    prediction = np.empty(len(y), dtype=float)
    for target in sorted(set(targets.tolist())):
        train = targets != target
        test = targets == target
        prediction[test] = c76_statistics.kernel_ridge_predict(
            X[train], y[train], X[test], "rbf", 1.0, 0.1,
        )
    return prediction


def _top1(prediction: np.ndarray, outcome: np.ndarray, targets: np.ndarray) -> float:
    values = []
    for target in sorted(set(targets.tolist())):
        indices = np.where(targets == target)[0]
        values.append(int(int(np.argmax(prediction[indices])) == int(np.argmax(outcome[indices]))))
    return float(np.mean(values))


def _case_data(case: str, rng: np.random.Generator, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(targets)
    functional = rng.normal(size=n)
    x = rng.normal(size=(n, 2))
    noise = rng.normal(scale=0.35, size=n)
    target_effect = rng.normal(size=9)[targets]
    invariant = x[:, 0] ** 2 - 1.0
    if case == "S0_no_association":
        architecture, outcome = x, functional + noise
    elif case == "S1_coordinate_artifact":
        architecture = np.column_stack((target_effect + 0.05 * x[:, 0], x[:, 1]))
        outcome = functional + noise
    elif case == "S2_pooled_identity":
        architecture = np.column_stack((target_effect + 0.05 * x[:, 0], x[:, 1]))
        outcome = target_effect + noise
    elif case == "S3_local_nonlinear_nontransport":
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=9)
        architecture, outcome = x, signs[targets] * invariant + noise
    elif case == "S4_factorization_invariant_endpoint":
        architecture = np.column_stack((invariant, x[:, 1]))
        outcome = invariant + noise
    elif case == "S5_association_no_extreme_action":
        architecture = x
        outcome = 0.8 * invariant + noise
        for target in range(9):
            indices = np.where(targets == target)[0]
            # Preserve the nonlinear bulk relation while assigning the best arm
            # to a uniformly random candidate. This makes association real but
            # top-1 control unavailable from the architecture block.
            winner = int(rng.choice(indices))
            outcome[winner] = float(np.max(outcome[indices])) + 1.0
    elif case == "S6_predictive_actionable":
        architecture, outcome = x, invariant + 0.15 * noise
    else:  # pragma: no cover
        raise ValueError(case)
    return architecture, outcome, functional


def _checkpoint_orbit(features: np.ndarray, rng: np.random.Generator, invariant: bool) -> np.ndarray:
    if invariant:
        return features.copy()
    output = np.empty_like(features)
    for index, row in enumerate(features):
        angle = rng.uniform(-math.pi, math.pi)
        scale = np.exp(rng.uniform(math.log(0.7), math.log(1.4), size=2))
        rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        output[index] = (rotation @ np.diag(scale) @ row[:, None])[:, 0]
    return output


def _one_replicate(replicate: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed + replicate)
    targets = np.repeat(np.arange(9), 24)
    rows = []
    for case in CASES:
        architecture, outcome, functional = _case_data(case, rng, targets)
        functional_prediction = c75_modeling.crossfit_loto(
            functional[:, None], outcome, targets, column_space=True,
        ).prediction
        centered_outcome = c76_statistics.center_within_groups(outcome[:, None], targets)[:, 0]
        residual = centered_outcome - functional_prediction
        association, _ = c76_statistics.crossfit_association(
            architecture, residual, targets, kernel_family="rbf",
            bandwidth_factor=1.0, statistic="centered_hsic",
        )
        pooled, _ = c76_statistics.topology_association(
            architecture, outcome, np.zeros(len(targets), dtype=int),
        )
        nonlinear_prediction = _fixed_kernel_crossfit(architecture, residual, targets)
        full_prediction = functional_prediction + nonlinear_prediction
        prior_r2 = c75_modeling.r2(outcome, functional_prediction, targets)
        full_r2 = c75_modeling.r2(outcome, full_prediction, targets)
        orbit_features = _checkpoint_orbit(
            architecture, rng, invariant=case in {"S4_factorization_invariant_endpoint"},
        )
        orbit_association, _ = c76_statistics.crossfit_association(
            orbit_features, residual, targets, kernel_family="rbf",
            bandwidth_factor=1.0, statistic="centered_hsic",
        )
        prior_top1 = _top1(functional_prediction, outcome, targets)
        full_top1 = _top1(full_prediction, outcome, targets)
        rows.append({
            "replicate": replicate, "case": case,
            "within_target_association": association,
            "pooled_association": pooled,
            "orbit_association": orbit_association,
            "orbit_effect_retention": abs(orbit_association) / max(abs(association), 1e-12),
            "incremental_R2": full_r2 - prior_r2,
            "prior_top1": prior_top1, "top1": full_top1,
            "top1_increment": full_top1 - prior_top1,
            "random_top1": 1.0 / 24.0,
        })
    return rows


def run_benchmark(
    replicates: int = c76_protocol.SYNTHETIC_REPLICATES,
    seed: int = c76_protocol.RNG_SEED + 300,
) -> tuple[list[dict], list[dict]]:
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    chunks = Parallel(n_jobs=workers, backend="loky")(
        delayed(_one_replicate)(replicate, seed) for replicate in range(replicates)
    )
    rows = [row for chunk in chunks for row in chunk]
    null_threshold = float(np.quantile([
        row["within_target_association"] for row in rows if row["case"] == "S0_no_association"
    ], 0.95))
    summary = []
    for case in CASES:
        selected = [row for row in rows if row["case"] == case]
        summary.append({
            "case": case, "replicates": len(selected),
            "association_detection_rate": float(np.mean([
                row["within_target_association"] > null_threshold for row in selected
            ])),
            "median_within_target_association": float(np.median([row["within_target_association"] for row in selected])),
            "median_pooled_association": float(np.median([row["pooled_association"] for row in selected])),
            "median_orbit_effect_retention": float(np.median([row["orbit_effect_retention"] for row in selected])),
            "median_incremental_R2": float(np.median([row["incremental_R2"] for row in selected])),
            "mean_top1": float(np.mean([row["top1"] for row in selected])),
            "mean_prior_top1": float(np.mean([row["prior_top1"] for row in selected])),
            "mean_top1_increment": float(np.mean([row["top1_increment"] for row in selected])),
            "null_association_p95": null_threshold,
        })
    return rows, summary
