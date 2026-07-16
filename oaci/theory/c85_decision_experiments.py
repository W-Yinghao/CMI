"""Finite statistical-experiment and decision-risk contracts for C85P."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Hashable, Iterable, Sequence


TOLERANCE = 1e-12


class DecisionContractError(ValueError):
    """Raised when a finite decision object violates the locked contract."""


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise DecisionContractError(f"{name} must be finite")
    return number


def _probability_vector(
    values: Sequence[float], name: str, *, tolerance: float = TOLERANCE,
) -> tuple[float, ...]:
    vector = tuple(_finite(value, name) for value in values)
    if not vector:
        raise DecisionContractError(f"{name} cannot be empty")
    if any(value < -tolerance for value in vector):
        raise DecisionContractError(f"{name} contains a negative probability")
    if abs(sum(vector) - 1.0) > tolerance:
        raise DecisionContractError(f"{name} must sum to one")
    return tuple(0.0 if abs(value) <= tolerance else value for value in vector)


@dataclass(frozen=True)
class FiniteExperiment:
    """A finite Markov kernel from states to observations."""

    states: tuple[Hashable, ...]
    observations: tuple[Hashable, ...]
    channel: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.states or len(set(self.states)) != len(self.states):
            raise DecisionContractError("experiment states must be nonempty and unique")
        if not self.observations or len(set(self.observations)) != len(self.observations):
            raise DecisionContractError(
                "experiment observations must be nonempty and unique"
            )
        if len(self.channel) != len(self.states):
            raise DecisionContractError("channel requires one row per state")
        normalized = []
        for index, row in enumerate(self.channel):
            if len(row) != len(self.observations):
                raise DecisionContractError("channel width does not match observations")
            normalized.append(_probability_vector(row, f"channel[{index}]"))
        object.__setattr__(self, "channel", tuple(normalized))

    @classmethod
    def from_rows(
        cls,
        states: Sequence[Hashable],
        observations: Sequence[Hashable],
        channel: Sequence[Sequence[float]],
    ) -> "FiniteExperiment":
        return cls(
            tuple(states), tuple(observations),
            tuple(tuple(float(value) for value in row) for row in channel),
        )


@dataclass(frozen=True)
class FiniteDecisionProblem:
    """Finite prior and bounded utility table sharing an experiment state space."""

    states: tuple[Hashable, ...]
    actions: tuple[Hashable, ...]
    prior: tuple[float, ...]
    utilities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.states or len(set(self.states)) != len(self.states):
            raise DecisionContractError("decision states must be nonempty and unique")
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise DecisionContractError("actions must be nonempty and unique")
        if len(self.prior) != len(self.states):
            raise DecisionContractError("prior length does not match states")
        object.__setattr__(self, "prior", _probability_vector(self.prior, "prior"))
        if len(self.utilities) != len(self.states):
            raise DecisionContractError("utility table requires one row per state")
        rows: list[tuple[float, ...]] = []
        for row in self.utilities:
            if len(row) != len(self.actions):
                raise DecisionContractError("utility width does not match actions")
            normalized = tuple(_finite(value, "utility") for value in row)
            if any(value < 0.0 or value > 1.0 for value in normalized):
                raise DecisionContractError("utilities must lie in [0,1]")
            rows.append(normalized)
        object.__setattr__(self, "utilities", tuple(rows))

    @property
    def losses(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(max(row) - utility for utility in row)
            for row in self.utilities
        )


def validate_common_state_space(
    experiment: FiniteExperiment, problem: FiniteDecisionProblem,
) -> None:
    if experiment.states != problem.states:
        raise DecisionContractError("experiment and decision state spaces differ")


def garble_experiment(
    rich: FiniteExperiment,
    coarse_observations: Sequence[Hashable],
    kernel: Sequence[Sequence[float]],
) -> FiniteExperiment:
    """Apply an observation-only Markov kernel to a finite experiment."""

    if len(kernel) != len(rich.observations):
        raise DecisionContractError("garbling requires one row per rich observation")
    normalized_kernel = tuple(
        _probability_vector(row, f"garbling[{index}]")
        for index, row in enumerate(kernel)
    )
    width = len(tuple(coarse_observations))
    if width == 0 or any(len(row) != width for row in normalized_kernel):
        raise DecisionContractError("garbling width does not match coarse observations")
    channel = []
    for state_row in rich.channel:
        channel.append(tuple(
            sum(state_row[z] * normalized_kernel[z][y] for z in range(len(state_row)))
            for y in range(width)
        ))
    return FiniteExperiment.from_rows(rich.states, coarse_observations, channel)


def channels_equal(
    left: FiniteExperiment,
    right: FiniteExperiment,
    *, tolerance: float = TOLERANCE,
) -> bool:
    if left.states != right.states or left.observations != right.observations:
        return False
    return all(
        abs(a - b) <= tolerance
        for left_row, right_row in zip(left.channel, right.channel)
        for a, b in zip(left_row, right_row)
    )


def deterministic_rule_risk(
    experiment: FiniteExperiment,
    problem: FiniteDecisionProblem,
    rule: Sequence[int],
) -> float:
    """Bayes risk of an observation-indexed deterministic action rule."""

    validate_common_state_space(experiment, problem)
    if len(rule) != len(experiment.observations):
        raise DecisionContractError("rule requires one action index per observation")
    if any(action < 0 or action >= len(problem.actions) for action in rule):
        raise DecisionContractError("rule contains an invalid action index")
    losses = problem.losses
    return sum(
        problem.prior[s] * experiment.channel[s][z] * losses[s][rule[z]]
        for s in range(len(problem.states))
        for z in range(len(experiment.observations))
    )


def randomized_rule_risk(
    experiment: FiniteExperiment,
    problem: FiniteDecisionProblem,
    rule: Sequence[Sequence[float]],
) -> float:
    """Bayes risk of an observation-indexed randomized action kernel."""

    validate_common_state_space(experiment, problem)
    if len(rule) != len(experiment.observations):
        raise DecisionContractError("randomized rule requires one row per observation")
    rows = tuple(
        _probability_vector(row, f"rule[{index}]")
        for index, row in enumerate(rule)
    )
    if any(len(row) != len(problem.actions) for row in rows):
        raise DecisionContractError("randomized rule width does not match actions")
    losses = problem.losses
    return sum(
        problem.prior[s]
        * experiment.channel[s][z]
        * rows[z][a]
        * losses[s][a]
        for s in range(len(problem.states))
        for z in range(len(experiment.observations))
        for a in range(len(problem.actions))
    )


def deterministic_rules(
    observation_count: int, action_count: int,
) -> Iterable[tuple[int, ...]]:
    if observation_count < 1 or action_count < 1:
        raise DecisionContractError("rule dimensions must be positive")
    return product(range(action_count), repeat=observation_count)


def unrestricted_optimal_risk(
    experiment: FiniteExperiment, problem: FiniteDecisionProblem,
) -> float:
    """Exact finite Bayes risk over all measurable randomized rules.

    The objective is linear in each action distribution, so a deterministic
    extreme point attains the finite optimum.
    """

    validate_common_state_space(experiment, problem)
    losses = problem.losses
    total = 0.0
    for z in range(len(experiment.observations)):
        action_risks = [
            sum(
                problem.prior[s] * experiment.channel[s][z] * losses[s][a]
                for s in range(len(problem.states))
            )
            for a in range(len(problem.actions))
        ]
        total += min(action_risks)
    return total


def registered_policy_risk(
    experiment: FiniteExperiment,
    problem: FiniteDecisionProblem,
    rules: Sequence[Sequence[int]],
) -> float:
    if not rules:
        raise DecisionContractError("registered policy class cannot be empty")
    return min(deterministic_rule_risk(experiment, problem, rule) for rule in rules)


def policy_approximation_gap(
    experiment: FiniteExperiment,
    problem: FiniteDecisionProblem,
    rules: Sequence[Sequence[int]],
) -> float:
    gap = registered_policy_risk(experiment, problem, rules) - unrestricted_optimal_risk(
        experiment, problem
    )
    if gap < -TOLERANCE:
        raise DecisionContractError("registered risk cannot beat unrestricted risk")
    return max(0.0, gap)


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise DecisionContractError("TV inputs must have equal positive length")
    p = _probability_vector(left, "left law")
    q = _probability_vector(right, "right law")
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def candidate_two_state_lecam_regret_bound(delta: float, tv: float) -> float:
    """Return the protocol candidate expression, not a C85P theorem.

    C85T must prove the exact constants and action-separation assumptions before
    this expression can receive a PROVED status.
    """

    gap = _finite(delta, "delta")
    distance = _finite(tv, "tv")
    if gap < 0.0 or not 0.0 <= distance <= 1.0:
        raise DecisionContractError("delta must be nonnegative and TV in [0,1]")
    return 0.5 * gap * (1.0 - distance)
