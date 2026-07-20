"""Exact scenario mechanics for the future authorized C85T run.

The functions are parameterized and are exercised by C85TL only on shadow
fixtures.  The registered dispatcher requires the authorization token emitted
by the runtime guard.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import math
from typing import Any, Mapping, Sequence

from .c85_decision_experiments import DecisionContractError
from .c85t_execution_guard import require_registered_capability


def as_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    raise DecisionContractError(f"cannot convert {type(value).__name__} to Fraction")


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def regret_rows(utilities: Sequence[Sequence[Any]]) -> tuple[tuple[Fraction, ...], ...]:
    rows: list[tuple[Fraction, ...]] = []
    for raw in utilities:
        values = tuple(as_fraction(value) for value in raw)
        if not values:
            raise DecisionContractError("utility row cannot be empty")
        optimum = max(values)
        rows.append(tuple(optimum - value for value in values))
    if not rows or len({len(row) for row in rows}) != 1:
        raise DecisionContractError("utility table must be nonempty and rectangular")
    return tuple(rows)


def fixed_action_risk(
    prior: Sequence[Any], regrets: Sequence[Sequence[Any]], action: int
) -> Fraction:
    masses = tuple(as_fraction(value) for value in prior)
    rows = tuple(tuple(as_fraction(value) for value in row) for row in regrets)
    if sum(masses, Fraction()) != 1 or any(value < 0 for value in masses):
        raise DecisionContractError("state probabilities must be nonnegative and sum to one")
    if not rows or not 0 <= action < len(rows[0]):
        raise DecisionContractError("fixed action is outside the action set")
    return sum((mass * row[action] for mass, row in zip(masses, rows)), Fraction())


def bayes_policy_and_risk(
    prior: Sequence[Any],
    channel: Sequence[Sequence[Any]],
    regrets: Sequence[Sequence[Any]],
) -> tuple[tuple[int, ...], Fraction]:
    masses = tuple(as_fraction(value) for value in prior)
    laws = tuple(tuple(as_fraction(value) for value in row) for row in channel)
    rows = tuple(tuple(as_fraction(value) for value in row) for row in regrets)
    if len(masses) != len(laws) or len(masses) != len(rows):
        raise DecisionContractError("state dimensions differ")
    observation_count = len(laws[0])
    if any(len(row) != observation_count for row in laws):
        raise DecisionContractError("observation channel is not rectangular")
    if any(sum(row, Fraction()) != 1 or any(value < 0 for value in row) for row in laws):
        raise DecisionContractError("observation laws must be probabilities")
    action_count = len(rows[0])
    selected: list[int] = []
    risk = Fraction()
    for observation in range(observation_count):
        values = tuple(
            sum(
                (
                    masses[state]
                    * laws[state][observation]
                    * rows[state][action]
                    for state in range(len(masses))
                ),
                Fraction(),
            )
            for action in range(action_count)
        )
        action = min(range(action_count), key=lambda index: (values[index], index))
        selected.append(action)
        risk += values[action]
    return tuple(selected), risk


def stable_argmax(values: Sequence[float]) -> int:
    if not values:
        raise DecisionContractError("argmax requires values")
    return max(range(len(values)), key=lambda index: (values[index], -index))


def stable_argmin(values: Sequence[float]) -> int:
    if not values:
        raise DecisionContractError("argmin requires values")
    return min(range(len(values)), key=lambda index: (values[index], index))


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return tuple(ranks)


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise DecisionContractError("Spearman inputs must have equal length >=2")
    ranks_a = _average_ranks(values_a)
    ranks_b = _average_ranks(values_b)
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    norm_a = sum((a - mean_a) ** 2 for a in ranks_a)
    norm_b = sum((b - mean_b) ** 2 for b in ranks_b)
    if norm_a == 0 or norm_b == 0:
        raise DecisionContractError("Spearman correlation is undefined for constant ranks")
    return numerator / math.sqrt(norm_a * norm_b)


def weighted_upper_cvar(
    losses: Sequence[Any], probabilities: Sequence[Any], alpha: Any
) -> Fraction:
    values = tuple(as_fraction(value) for value in losses)
    masses = tuple(as_fraction(value) for value in probabilities)
    level = as_fraction(alpha)
    if not Fraction() < level < Fraction(1):
        raise DecisionContractError("CVaR alpha must be strictly inside (0,1)")
    if len(values) != len(masses) or sum(masses, Fraction()) != 1:
        raise DecisionContractError("CVaR distribution is malformed")
    candidates: list[Fraction] = []
    for eta in sorted(set(values)):
        excess = sum(
            (mass * max(loss - eta, Fraction()) for loss, mass in zip(values, masses)),
            Fraction(),
        )
        candidates.append(eta + excess / (1 - level))
    return min(candidates)


def s5_candidate_cvar_region(scenario: Mapping[str, Any]) -> dict[str, str]:
    probabilities = tuple(as_fraction(value) for value in scenario["group_probabilities"])
    reference = tuple(as_fraction(value) for value in scenario["reference_losses"])
    policy = tuple(as_fraction(value) for value in scenario["policy_losses"])
    policy_mean = sum((p * loss for p, loss in zip(probabilities, policy)), Fraction())
    reference_mean = sum((p * loss for p, loss in zip(probabilities, reference)), Fraction())
    if policy_mean >= reference_mean or max(policy) <= max(reference):
        raise DecisionContractError("S5 does not separate mean and worst risk")

    # With masses 9/10 at 3/10 and 1/10 at 1, the lower branch crosses
    # the reference CVaR 1/2 at alpha=13/20 and remains worse thereafter.
    crossing = Fraction(13, 20)
    left = weighted_upper_cvar(policy, probabilities, Fraction(1, 2))
    right = weighted_upper_cvar(policy, probabilities, Fraction(9, 10))
    if not left < Fraction(1, 2) or right != 1:
        raise DecisionContractError("S5 candidate CVaR region derivation drifted")
    return {
        "candidate_open_lower": fraction_text(crossing),
        "candidate_open_upper": "1",
        "endpoint_policy": "both endpoints excluded",
        "status": "CANDIDATE_PROOF_TARGET_NOT_PROVED_BY_EXECUTION_MODE",
    }


def near_optimal_geometry(
    utilities: Sequence[float], epsilon: float, tau: float, pairwise_sigma: float
) -> dict[str, Any]:
    if not utilities or epsilon < 0 or tau <= 0 or pairwise_sigma <= 0:
        raise DecisionContractError("invalid near-optimal geometry input")
    optimum = max(utilities)
    gaps = tuple(optimum - value for value in utilities)
    near = tuple(index for index, gap in enumerate(gaps) if gap <= epsilon)
    raw = tuple(math.exp(-gap / tau) for gap in gaps)
    total = sum(raw)
    weights = tuple(value / total for value in raw)
    hill2 = 1.0 / sum(value * value for value in weights)
    entropy = math.exp(-sum(value * math.log(value) for value in weights if value > 0))
    primary = min(
        1.0,
        sum(
            math.exp(-(gap * gap) / (2.0 * pairwise_sigma * pairwise_sigma))
            for gap in gaps
            if gap > epsilon
        ),
    )
    looser = min(
        1.0,
        sum(
            math.exp(-((gap - epsilon) ** 2) / (2.0 * pairwise_sigma * pairwise_sigma))
            for gap in gaps
            if gap > epsilon
        ),
    )
    return {
        "gaps": gaps,
        "epsilon_near_optimal_set": near,
        "near_optimal_count": len(near),
        "hill_2_effective_size": hill2,
        "entropy_effective_size": entropy,
        "t7_primary_union_bound": primary,
        "historical_looser_diagnostic": looser,
    }


def _solve_square(matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]) -> tuple[Fraction, ...] | None:
    n = len(matrix)
    augmented = [list(row) + [rhs_value] for row, rhs_value in zip(matrix, rhs)]
    if any(len(row) != n + 1 for row in augmented):
        raise DecisionContractError("linear system is not square")
    for column in range(n):
        pivot = next((row for row in range(column, n) if augmented[row][column]), None)
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    current - factor * pivot_value
                    for current, pivot_value in zip(augmented[row], augmented[column])
                ]
    return tuple(augmented[row][-1] for row in range(n))


def exact_minimax_regret_lp(extreme_points: Sequence[Sequence[Any]]) -> dict[str, Any]:
    utilities = tuple(tuple(as_fraction(value) for value in row) for row in extreme_points)
    if not utilities or len({len(row) for row in utilities}) != 1:
        raise DecisionContractError("identified-set extreme points must be rectangular")
    action_count = len(utilities[0])
    regret_vectors = tuple(
        tuple(max(row) - value for value in row) for row in utilities
    )
    # Variables are q_0,...,q_{M-1},t. The simplex equality is always active.
    constraints: list[tuple[tuple[Fraction, ...], Fraction, str]] = []
    for action in range(action_count):
        coefficients = [Fraction()] * (action_count + 1)
        coefficients[action] = Fraction(1)
        constraints.append((tuple(coefficients), Fraction(), f"q{action}=0"))
    for point, regrets in enumerate(regret_vectors):
        coefficients = list(regrets) + [Fraction(-1)]
        constraints.append((tuple(coefficients), Fraction(), f"point{point}=t"))

    simplex = tuple([Fraction(1)] * action_count + [Fraction()])
    feasible: list[tuple[Fraction, tuple[Fraction, ...], tuple[str, ...]]] = []
    for chosen in combinations(range(len(constraints)), action_count):
        matrix = [simplex] + [constraints[index][0] for index in chosen]
        rhs = [Fraction(1)] + [constraints[index][1] for index in chosen]
        solution = _solve_square(matrix, rhs)
        if solution is None:
            continue
        q = solution[:-1]
        t = solution[-1]
        if any(value < 0 for value in q) or t < 0:
            continue
        scenario_values = tuple(
            sum((weight * regret for weight, regret in zip(q, regrets)), Fraction())
            for regrets in regret_vectors
        )
        if any(value > t for value in scenario_values):
            continue
        feasible.append((t, q, tuple(constraints[index][2] for index in chosen)))
    if not feasible:
        raise DecisionContractError("minimax-regret LP has no enumerated feasible vertex")
    optimum_t, optimum_q, active = min(feasible, key=lambda row: (row[0], row[1], row[2]))
    slacks = tuple(
        optimum_t
        - sum((weight * regret for weight, regret in zip(optimum_q, regrets)), Fraction())
        for regrets in regret_vectors
    )
    pure_values = tuple(max(regrets[action] for regrets in regret_vectors) for action in range(action_count))
    pure_optimum = min(pure_values)
    diameter = max(
        abs(left[action] - right[action])
        for left in utilities
        for right in utilities
        for action in range(action_count)
    )
    return {
        "identified_set_infinity_diameter": diameter,
        "optimal_randomized_action_distribution": optimum_q,
        "minimax_regret": optimum_t,
        "extreme_point_constraint_slacks": slacks,
        "active_constraints": active,
        "pure_action_minimax_regret": pure_optimum,
        "randomization_gain": pure_optimum - optimum_t,
    }


def _serialize(value: Any) -> Any:
    if isinstance(value, Fraction):
        return fraction_text(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def execute_registered_exact_scenarios(
    contract: Mapping[str, Any], *, capability: object
) -> dict[str, Any]:
    """Execute authoritative exact portions after runtime authorization replay."""

    require_registered_capability(capability)
    scenarios = {row["id"]: row for row in contract["scenarios"]}
    if tuple(scenarios) != tuple(f"S{i}" for i in range(11)):
        raise DecisionContractError("registered scenario order drifted")
    output: dict[str, Any] = {}

    s0 = scenarios["S0"]
    s0_regrets = regret_rows(s0["utilities"])
    output["S0"] = {"optimal_constant_risk": fixed_action_risk(s0["state_probabilities"], s0_regrets, 0)}

    s1 = scenarios["S1"]
    s1_regrets = regret_rows(s1["utilities"])
    _, e1_risk = bayes_policy_and_risk(s1["state_probabilities"], s1["experiments"]["E1"], s1_regrets)
    _, e2_unrestricted = bayes_policy_and_risk(s1["state_probabilities"], s1["experiments"]["E2"], s1_regrets)
    # The registered rich rule chooses the action opposite the revealed state.
    e2_registered = sum(
        (as_fraction(mass) * s1_regrets[state][1 - state] for state, mass in enumerate(s1["state_probabilities"])),
        Fraction(),
    )
    output["S1"] = {"coarse_registered_risk": e1_risk, "rich_unrestricted_risk": e2_unrestricted, "rich_registered_risk": e2_registered}

    s2 = scenarios["S2"]
    s2_regrets = regret_rows(s2["utilities"])
    collapsed = fixed_action_risk(s2["state_probabilities"], s2_regrets, 0)
    output["S2"] = {"action_divergence": Fraction(), "registered_risk": collapsed, "reference_risk": collapsed}

    s3 = scenarios["S3"]
    utilities3 = tuple(float(value) for value in s3["utilities"][0])
    scores3 = tuple(float(value) for value in s3["selector_scores"][0])
    selected3 = stable_argmax(scores3)
    output["S3"] = {"selected_action": selected3, "regret": max(utilities3) - utilities3[selected3], "spearman": spearman(scores3, utilities3)}

    s4 = scenarios["S4"]
    utilities4 = tuple(as_fraction(value) for value in s4["utilities"][0])
    selected4 = int(s4["selected_action"])
    output["S4"] = {"top4_localization": int(utilities4.index(max(utilities4)) in s4["reported_topk_candidate_set"]), "selected_regret": max(utilities4) - utilities4[selected4]}

    s5 = scenarios["S5"]
    output["S5"] = s5_candidate_cvar_region(s5)

    for scenario_id in ("S6", "S7"):
        row = scenarios[scenario_id]
        output[scenario_id] = near_optimal_geometry(
            tuple(float(value) for value in row["utilities"][0]),
            float(row["epsilon"]),
            float(row["tau"]),
            float(row["pairwise_sigma"]),
        )

    output["S8"] = exact_minimax_regret_lp(scenarios["S8"]["utility_extreme_points"])

    from .c85r_synthetic_semantic_repair import exact_s9_audit, exact_s10_audit

    s9 = exact_s9_audit()
    output["S9"] = {
        "population_means": s9["population_means"],
        "passive_allocation": s9["passive_allocation"],
        "neyman_allocation": s9["neyman_allocation"],
        "passive_analytic_variance": s9["passive_variance"],
        "neyman_analytic_variance": s9["neyman_variance"],
    }
    output["S10"] = exact_s10_audit()
    return _serialize(output)
