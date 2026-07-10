"""Registered synthetic C75 factorization and construct-validity benchmark."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from . import c75_protocol


def _transforms(rng: np.random.Generator, dimension: int) -> list[tuple[str, np.ndarray]]:
    q1, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    q2, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    scales = np.linspace(0.5, 2.0, dimension)
    return [
        ("identity", np.eye(dimension)),
        ("orthogonal_QR", q1),
        ("diagonal_scale_0.5_to_2", np.diag(scales)),
        ("nonorthogonal_condition_le_4", q1 @ np.diag(scales) @ q2.T),
    ]


def factorization_reparameterization_audit(seed: int = c75_protocol.RNG_SEED + 100) -> list[dict]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(512, 16))
    W = rng.normal(size=(4, 16))
    bias = rng.normal(size=4)
    reference_wz = z @ W.T
    reference_logits = reference_wz + bias
    rows = []
    for name, A in _transforms(rng, 16):
        transformed_z = z @ A.T
        transformed_W = W @ np.linalg.inv(A)
        transformed_wz = transformed_z @ transformed_W.T
        transformed_logits = transformed_wz + bias
        rows.append({
            "transform": name,
            "condition_number_A": float(np.linalg.cond(A)),
            "Wz_max_abs_error": float(np.max(np.abs(transformed_wz - reference_wz))),
            "logit_max_abs_error": float(np.max(np.abs(transformed_logits - reference_logits))),
            "probability_max_abs_error": _probability_error(reference_logits, transformed_logits),
            "mean_z_norm_delta": float(np.mean(np.linalg.norm(transformed_z, axis=1) - np.linalg.norm(z, axis=1))),
            "W_frobenius_delta": float(np.linalg.norm(transformed_W) - np.linalg.norm(W)),
            "z_coordinate_mean_l2_delta": float(np.linalg.norm(np.mean(transformed_z, axis=0) - np.mean(z, axis=0))),
            "function_invariant": int(np.max(np.abs(transformed_logits - reference_logits)) < 1e-10),
            "coordinate_geometry_invariant": int(np.allclose(np.linalg.norm(transformed_z, axis=1), np.linalg.norm(z, axis=1), atol=1e-10)),
        })
    return rows


def _probability_error(left_logits: np.ndarray, right_logits: np.ndarray) -> float:
    def softmax(values):
        shifted = values - np.max(values, axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / np.sum(exponent, axis=1, keepdims=True)
    return float(np.max(np.abs(softmax(left_logits) - softmax(right_logits))))


def construct_validity_benchmark(
    replicates: int = 500,
    seed: int = c75_protocol.RNG_SEED + 100,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        n = 512
        target = np.tile(np.arange(9), math.ceil(n / 9))[:n]
        functional = rng.normal(size=n)
        architecture = rng.normal(size=n)
        stable_identity = rng.normal(size=9)[target] + 0.02 * rng.normal(size=n)
        noise = rng.normal(scale=0.7, size=n)
        cases = {
            "stable_endpoint_irrelevant": (functional + noise, stable_identity),
            "incremental_representation": (functional + 0.6 * architecture + noise, architecture),
            "functionally_redundant": (functional + noise, functional.copy()),
        }
        for case, (outcome, block) in cases.items():
            residual = _target_center(outcome, target) - _target_center(functional, target)
            block_centered = _target_center(block, target)
            statistic = _safe_correlation(residual, block_centered)
            permutation = []
            for _ in range(49):
                permuted = block_centered.copy()
                for group in range(9):
                    indices = np.where(target == group)[0]
                    permuted[indices] = rng.permutation(permuted[indices])
                permutation.append(_safe_correlation(residual, permuted))
            p_value = (1 + sum(value >= statistic for value in permutation)) / 50.0
            rows.append({
                "replicate": replicate, "case": case,
                "residual_block_correlation": statistic,
                "permutation_p": p_value,
                "detected_at_0.05": int(p_value < 0.05),
                "stable_descriptor": int(case == "stable_endpoint_irrelevant"),
                "truly_incremental": int(case == "incremental_representation"),
                "redundant_with_functional": int(case == "functionally_redundant"),
            })
    summary = []
    for case in sorted({row["case"] for row in rows}):
        selected = [row for row in rows if row["case"] == case]
        summary.append({
            "case": case, "replicates": len(selected),
            "detection_rate": float(np.mean([row["detected_at_0.05"] for row in selected])),
            "median_correlation": float(np.median([row["residual_block_correlation"] for row in selected])),
            "expected_behavior": "low_false_positive" if case == "stable_endpoint_irrelevant" else "high_power" if case == "incremental_representation" else "redundancy_not_architecture_gain",
        })
    return rows, summary


def _target_center(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    for group in sorted(set(target.tolist())):
        mask = target == group
        output[mask] -= float(np.mean(output[mask]))
    return output


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return 0.0
    result = stats.pearsonr(left, right)
    return float(result.statistic if hasattr(result, "statistic") else result[0])
