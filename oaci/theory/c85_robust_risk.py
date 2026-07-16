"""Robust target-risk and near-optimal action-geometry contracts for C85P."""
from __future__ import annotations

import math
from typing import Hashable, Sequence

from .c85_decision_experiments import DecisionContractError, TOLERANCE


def _losses_and_weights(
    losses: Sequence[float], weights: Sequence[float] | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = tuple(float(value) for value in losses)
    if not values or any(not math.isfinite(value) for value in values):
        raise DecisionContractError("losses must be finite and nonempty")
    if weights is None:
        mass = tuple(1.0 / len(values) for _ in values)
    else:
        if len(weights) != len(values):
            raise DecisionContractError("weights must match losses")
        raw = tuple(float(weight) for weight in weights)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in raw):
            raise DecisionContractError("weights must be finite and nonnegative")
        total = sum(raw)
        if total <= 0.0:
            raise DecisionContractError("weights require positive total mass")
        mass = tuple(weight / total for weight in raw)
    return values, mass


def mean_risk(losses: Sequence[float], weights: Sequence[float] | None = None) -> float:
    values, mass = _losses_and_weights(losses, weights)
    return sum(value * weight for value, weight in zip(values, mass))


def worst_group_risk(
    group_losses: Sequence[float] | dict[Hashable, float],
) -> float:
    values = (
        tuple(group_losses.values())
        if isinstance(group_losses, dict)
        else tuple(group_losses)
    )
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise DecisionContractError("group losses must be finite and nonempty")
    return max(float(value) for value in values)


def upper_tail_cvar(
    losses: Sequence[float],
    alpha: float,
    weights: Sequence[float] | None = None,
) -> float:
    """Upper-loss-tail CVaR under the Rockafellar-Uryasev convention.

    Computes inf_eta eta + E[(L-eta)_+]/(1-alpha) exactly for a finite weighted
    distribution by checking every loss breakpoint.  The symbolic protocol
    permits any alpha in (0,1); no empirical C84 alpha is selected here.
    """

    confidence = float(alpha)
    if not 0.0 < confidence < 1.0:
        raise DecisionContractError("CVaR alpha must lie strictly in (0,1)")
    values, mass = _losses_and_weights(losses, weights)
    objectives = []
    for eta in sorted(set(values)):
        objectives.append(
            eta
            + sum(weight * max(value - eta, 0.0) for value, weight in zip(values, mass))
            / (1.0 - confidence)
        )
    return min(objectives)


def near_optimal_set(utilities: Sequence[float], epsilon: float) -> tuple[int, ...]:
    values = tuple(float(value) for value in utilities)
    tolerance = float(epsilon)
    if not values or any(not math.isfinite(value) for value in values):
        raise DecisionContractError("utilities must be finite and nonempty")
    if tolerance < 0.0:
        raise DecisionContractError("epsilon must be nonnegative")
    best = max(values)
    return tuple(index for index, value in enumerate(values) if best - value <= tolerance + TOLERANCE)


def soft_gap_weights(utilities: Sequence[float], tau: float) -> tuple[float, ...]:
    values = tuple(float(value) for value in utilities)
    temperature = float(tau)
    if not values or any(not math.isfinite(value) for value in values):
        raise DecisionContractError("utilities must be finite and nonempty")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise DecisionContractError("tau must be finite and positive")
    best = max(values)
    log_weights = tuple(-(best - value) / temperature for value in values)
    shift = max(log_weights)
    raw = tuple(math.exp(value - shift) for value in log_weights)
    total = sum(raw)
    return tuple(value / total for value in raw)


def hill2_effective_size(weights: Sequence[float]) -> float:
    _, mass = _losses_and_weights(tuple(0.0 for _ in weights), weights)
    return 1.0 / sum(weight * weight for weight in mass)


def entropy_effective_size(weights: Sequence[float]) -> float:
    _, mass = _losses_and_weights(tuple(0.0 for _ in weights), weights)
    entropy = -sum(weight * math.log(weight) for weight in mass if weight > 0.0)
    return math.exp(entropy)


def candidate_subgaussian_outside_near_optimal_bound(
    gaps: Sequence[float], epsilon: float, pairwise_scales: Sequence[float],
) -> float:
    """Protocol candidate union bound; proof status remains OPEN until C85T."""

    if len(gaps) != len(pairwise_scales) or not gaps:
        raise DecisionContractError("gaps and scales must have equal positive length")
    eps = float(epsilon)
    if eps < 0.0:
        raise DecisionContractError("epsilon must be nonnegative")
    terms = []
    for gap, scale in zip(gaps, pairwise_scales):
        delta = float(gap)
        sigma = float(scale)
        if not math.isfinite(delta) or delta < 0.0:
            raise DecisionContractError("gaps must be finite and nonnegative")
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise DecisionContractError("pairwise scales must be finite and positive")
        if delta > eps:
            terms.append(math.exp(-((delta - eps) ** 2) / (2.0 * sigma * sigma)))
    return min(1.0, sum(terms))


def compare_mean_and_tail(
    reference_losses: Sequence[float],
    policy_losses: Sequence[float],
    alpha: float,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    if len(reference_losses) != len(policy_losses):
        raise DecisionContractError("policy and reference require common groups")
    return {
        "mean_improvement": mean_risk(reference_losses, weights) - mean_risk(policy_losses, weights),
        "worst_improvement": worst_group_risk(reference_losses) - worst_group_risk(policy_losses),
        "cvar_improvement": upper_tail_cvar(reference_losses, alpha, weights)
        - upper_tail_cvar(policy_losses, alpha, weights),
    }
