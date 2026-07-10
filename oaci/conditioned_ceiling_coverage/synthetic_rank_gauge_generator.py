"""Reusable synthetic multi-candidate rank-gauge benchmark for C72.

The generator keeps target-common offsets, candidate-specific gauge, and
finite-label measurement noise separate.  It can emit paired trial outcomes
for focused checks and a faster aggregate draw for the registered phase grid.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SyntheticInstance:
    source_rank: np.ndarray
    candidate_gauge: np.ndarray
    common_offset: float
    latent_utility: np.ndarray
    construction_score: np.ndarray
    evaluation_score: np.ndarray
    labels: np.ndarray
    paired_correctness: np.ndarray


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    rx, ry = _rankdata(x), _rankdata(y)
    sx, sy = float(np.std(rx)), float(np.std(ry))
    if sx <= 0 or sy <= 0:
        return math.nan
    return float(np.mean((rx - np.mean(rx)) * (ry - np.mean(ry))) / (sx * sy))


def top_metrics(score: np.ndarray, utility: np.ndarray, k: int = 3) -> tuple[float, float, float]:
    score = np.asarray(score, dtype=float)
    utility = np.asarray(utility, dtype=float)
    selected = np.flatnonzero(np.isclose(score, np.max(score), atol=1e-12, rtol=0.0))
    oracle = set(np.flatnonzero(np.isclose(utility, np.max(utility), atol=1e-12, rtol=0.0)))
    top1 = sum(int(i in oracle) for i in selected) / max(len(selected), 1)
    order = np.argsort(-score, kind="mergesort")[: min(k, len(score))]
    topk = float(any(int(i) in oracle for i in order))
    regret = float(np.max(utility) - np.mean(utility[selected]))
    return float(top1), topk, regret


def _gauge_draw(source_rank: np.ndarray, gauge_sd: float, shape: str, rng: np.random.Generator) -> np.ndarray:
    m = len(source_rank)
    if gauge_sd <= 0:
        return np.zeros(m, dtype=float)
    if shape == "gaussian":
        raw = rng.normal(size=m)
    elif shape == "skewed":
        raw = rng.lognormal(mean=0.0, sigma=0.8, size=m)
        raw = raw - float(np.mean(raw))
    elif shape == "heteroskedastic":
        scale = 0.35 + np.abs(source_rank) / max(float(np.mean(np.abs(source_rank))), 1e-8)
        raw = rng.normal(size=m) * scale
    else:
        raise ValueError(f"unknown gauge shape: {shape}")
    sd = float(np.std(raw))
    return raw * (gauge_sd / sd) if sd > 0 else np.zeros(m, dtype=float)


def generate_trial_instance(
    *,
    candidate_count: int,
    rank_snr: float,
    gauge_sd: float,
    gauge_shape: str,
    label_budget: int,
    outcome_type: str,
    rng: np.random.Generator,
    eval_trials_per_class: int = 256,
    gauge_capture: float = 0.45,
) -> SyntheticInstance:
    """Generate shared-trial paired outcomes with known rank/gauge pieces."""
    classes = 2 if outcome_type == "binary" else 4
    source = rng.normal(scale=max(rank_snr, 1e-8), size=candidate_count)
    source -= float(np.mean(source))
    gauge = _gauge_draw(source, gauge_sd, gauge_shape, rng)
    common = float(rng.normal(scale=0.5))
    latent = source + gauge + common
    competence = 0.15 + 0.75 / (1.0 + np.exp(-np.clip(latent, -40.0, 40.0)))

    n_eval = classes * eval_trials_per_class
    labels = np.repeat(np.arange(classes), eval_trials_per_class)
    shared_u = rng.uniform(size=n_eval)
    idiosyncratic = rng.uniform(size=(candidate_count, n_eval))
    # A common draw makes candidate contrasts paired while retaining crossings.
    mix = 0.65 * shared_u[None, :] + 0.35 * idiosyncratic
    correct = (mix < competence[:, None]).astype(np.int8)
    evaluation = np.mean(
        np.stack([np.mean(correct[:, labels == cls], axis=1) for cls in range(classes)]),
        axis=0,
    )

    construction_parts = []
    for cls in range(classes):
        cls_idx = np.flatnonzero(labels == cls)
        chosen = rng.choice(cls_idx, size=min(label_budget, len(cls_idx)), replace=False)
        construction_parts.append(np.mean(correct[:, chosen], axis=1))
    empirical = np.mean(np.stack(construction_parts), axis=0)
    captured_latent = source + gauge_capture * gauge + common
    construction = 0.5 * empirical + 0.5 * (0.15 + 0.75 / (1.0 + np.exp(-np.clip(captured_latent, -40.0, 40.0))))
    return SyntheticInstance(source, gauge, common, latent, construction, evaluation, labels, correct)


def aggregate_instance_metrics(
    *,
    candidate_count: int,
    rank_snr: float,
    gauge_sd: float,
    gauge_shape: str,
    label_budget: int,
    outcome_type: str,
    rng: np.random.Generator,
    gauge_capture: float = 0.45,
) -> dict[str, float]:
    """Fast registered-grid draw preserving the same structural decomposition."""
    source = rng.normal(scale=max(rank_snr, 1e-8), size=candidate_count)
    source -= float(np.mean(source))
    gauge = _gauge_draw(source, gauge_sd, gauge_shape, rng)
    common = float(rng.normal(scale=0.5))
    latent = source + gauge + common
    p = 0.15 + 0.75 / (1.0 + np.exp(-np.clip(latent, -40.0, 40.0)))
    classes = 2 if outcome_type == "binary" else 4
    n = max(1, classes * int(label_budget))
    measured_latent = source + gauge_capture * gauge + common
    measured_p = 0.15 + 0.75 / (1.0 + np.exp(-np.clip(measured_latent, -40.0, 40.0)))
    construction = rng.binomial(n, np.clip(measured_p, 1e-6, 1 - 1e-6)) / n
    evaluation = rng.binomial(classes * 256, np.clip(p, 1e-6, 1 - 1e-6)) / (classes * 256)
    rho = spearman(construction, evaluation)
    top1, top3, regret = top_metrics(construction, evaluation, k=3)
    pair_total = pair_ok = 0.0
    for i in range(candidate_count):
        for j in range(i + 1, candidate_count):
            dy = float(evaluation[i] - evaluation[j])
            if abs(dy) <= 1e-12:
                continue
            dx = float(construction[i] - construction[j])
            pair_total += 1.0
            pair_ok += 0.5 if abs(dx) <= 1e-12 else float(dx * dy > 0)
    common_shift = float(rng.normal())
    common_flip = int(not np.array_equal(np.argsort(latent), np.argsort(latent + common_shift)))
    candidate_shift = rng.normal(scale=max(gauge_sd, 0.25), size=candidate_count)
    candidate_flip = int(not np.array_equal(np.argsort(latent), np.argsort(latent + candidate_shift)))
    return {
        "spearman": rho,
        "pairwise_accuracy": pair_ok / pair_total if pair_total else math.nan,
        "top1": top1,
        "top3": top3,
        "regret": regret,
        "common_offset_rank_flip": float(common_flip),
        "candidate_specific_rank_flip": float(candidate_flip),
        "gauge_recovery": max(0.0, min(1.0, spearman(construction - source, gauge) ** 2)) if gauge_sd > 0 else 0.0,
    }
