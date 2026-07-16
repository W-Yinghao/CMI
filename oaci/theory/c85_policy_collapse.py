"""Realized action dependence and fixed-policy collapse contracts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Hashable, Sequence

from .c85_decision_experiments import DecisionContractError


@dataclass(frozen=True)
class RealizedPolicyUse:
    observations: int
    action_divergence: float
    action_entropy: float
    policy_risk: float
    reference_risk: float
    incremental_fixed_policy_risk_value: float
    exact_collapse: bool


def action_divergence(
    policy_actions: Sequence[Hashable], reference_actions: Sequence[Hashable],
    weights: Sequence[float] | None = None,
) -> float:
    if len(policy_actions) != len(reference_actions) or not policy_actions:
        raise DecisionContractError("action sequences must have equal positive length")
    if weights is None:
        mass = [1.0 / len(policy_actions)] * len(policy_actions)
    else:
        if len(weights) != len(policy_actions):
            raise DecisionContractError("action weights have wrong length")
        raw = [float(weight) for weight in weights]
        if any(not math.isfinite(weight) or weight < 0.0 for weight in raw) or sum(raw) <= 0.0:
            raise DecisionContractError("action weights must be finite nonnegative mass")
        mass = [weight / sum(raw) for weight in raw]
    return sum(
        weight for action, reference, weight in zip(policy_actions, reference_actions, mass)
        if action != reference
    )


def action_entropy(actions: Sequence[Hashable], weights: Sequence[float] | None = None) -> float:
    if not actions:
        raise DecisionContractError("actions cannot be empty")
    if weights is None:
        mass = [1.0 / len(actions)] * len(actions)
    else:
        if len(weights) != len(actions):
            raise DecisionContractError("entropy weights have wrong length")
        raw = [float(weight) for weight in weights]
        if any(not math.isfinite(weight) or weight < 0.0 for weight in raw) or sum(raw) <= 0.0:
            raise DecisionContractError("entropy weights must be finite nonnegative mass")
        mass = [weight / sum(raw) for weight in raw]
    distribution: dict[Hashable, float] = Counter()
    for action, weight in zip(actions, mass):
        distribution[action] += weight
    return -sum(probability * math.log(probability) for probability in distribution.values() if probability > 0.0)


def summarize_realized_policy_use(
    policy_actions: Sequence[Hashable],
    reference_actions: Sequence[Hashable],
    policy_losses: Sequence[float],
    reference_losses: Sequence[float],
    weights: Sequence[float] | None = None,
) -> RealizedPolicyUse:
    size = len(policy_actions)
    if any(len(values) != size for values in (reference_actions, policy_losses, reference_losses)) or size == 0:
        raise DecisionContractError("policy-use arrays must share a positive length")
    if weights is None:
        mass = [1.0 / size] * size
    else:
        if len(weights) != size:
            raise DecisionContractError("policy-use weights have wrong length")
        raw = [float(weight) for weight in weights]
        if any(not math.isfinite(weight) or weight < 0.0 for weight in raw) or sum(raw) <= 0.0:
            raise DecisionContractError("policy-use weights must be finite nonnegative mass")
        mass = [weight / sum(raw) for weight in raw]
    p_losses = [float(value) for value in policy_losses]
    r_losses = [float(value) for value in reference_losses]
    if any(not math.isfinite(value) for value in p_losses + r_losses):
        raise DecisionContractError("policy-use losses must be finite")
    policy_risk = sum(weight * loss for weight, loss in zip(mass, p_losses))
    reference_risk = sum(weight * loss for weight, loss in zip(mass, r_losses))
    divergence = action_divergence(policy_actions, reference_actions, mass)
    return RealizedPolicyUse(
        observations=size,
        action_divergence=divergence,
        action_entropy=action_entropy(policy_actions, mass),
        policy_risk=policy_risk,
        reference_risk=reference_risk,
        incremental_fixed_policy_risk_value=reference_risk - policy_risk,
        exact_collapse=divergence == 0.0,
    )


def assert_policy_collapse_loss_identity(
    policy_actions: Sequence[Hashable],
    reference_actions: Sequence[Hashable],
    policy_losses: Sequence[float],
    reference_losses: Sequence[float],
) -> None:
    if tuple(policy_actions) != tuple(reference_actions):
        raise DecisionContractError("realized action maps do not collapse exactly")
    if tuple(float(value) for value in policy_losses) != tuple(float(value) for value in reference_losses):
        raise DecisionContractError("equal realized actions require identical registered losses")
