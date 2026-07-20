"""C85R additive synthetic-contract semantic satisfiability validation.

This module performs exact contract preflight only. It does not run the
registered 4,096-replicate synthetic experiments, complete a proof, or change
any theorem status.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .c85_decision_experiments import DecisionContractError
from . import c85_synthetic_contract as c85p


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
C85P_TABLE_DIR = REPORT_DIR / "c85p_tables"
C85R_TABLE_DIR = REPORT_DIR / "c85r_tables"
HISTORICAL_CONTRACT_PATH = C85P_TABLE_DIR / "synthetic_generator_contract.json"
REPAIR_PROTOCOL_PATH = REPORT_DIR / "C85R_SYNTHETIC_CONTRACT_SEMANTIC_REPAIR_PROTOCOL.json"
REPAIR_PROTOCOL_SIDECAR = REPORT_DIR / "C85R_SYNTHETIC_CONTRACT_SEMANTIC_REPAIR_PROTOCOL.sha256"
V2_CONTRACT_PATH = C85R_TABLE_DIR / "synthetic_generator_contract_v2.json"
V2_CONTRACT_SIDECAR = C85R_TABLE_DIR / "synthetic_generator_contract_v2.sha256"

EXPECTED_HISTORICAL_SHA256 = "c87fec6a6572291fad8849f6c08bea2cb3f49467e243ded1d44c1f38e3d0b297"
EXPECTED_REPAIR_PROTOCOL_SHA256 = "e37bb444fdd174ba4ca1f95e91d9193378f11dd0ef2aeac3e03cbf6249a34b68"
EXPECTED_V2_CONTRACT_SHA256 = "e055c2a785374a3067ce90746a5941b39847b88a4f33e4ff8da5ca8adfde355a"
EXPECTED_REPAIR_PROTOCOL_COMMIT = "03bb684e59e3432ae6f484c8c8a537213f52a6cd"
EXPECTED_SCENARIOS = tuple(f"S{index}" for index in range(11))
SEMANTIC_STATUS = "SEMANTICALLY_VALIDATED_NOT_SCIENTIFICALLY_EXECUTED"
SUCCESS_GATE = "C85_SYNTHETIC_CONTRACT_V2_SEMANTICALLY_REPAIRED_READY_FOR_C85T_PM_REVIEW"
FAILURE_GATE = "C85_SYNTHETIC_SCENARIO_GENERATIVE_SEMANTICS_OR_PROOF_OBLIGATION_RECONCILIATION_REQUIRED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sidecar_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.strip()] = digest
    return entries


def _fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise DecisionContractError("boolean is not a rational contract value")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value)
    raise DecisionContractError(f"unsupported rational value {value!r}")


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _decimal_text(value: Fraction) -> str:
    return format(float(value), ".17g")


def _scenario_map(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    scenarios = contract.get("scenarios")
    if not isinstance(scenarios, list):
        raise DecisionContractError("synthetic scenarios must be a list")
    result = {str(row["id"]): dict(row) for row in scenarios}
    if tuple(row["id"] for row in scenarios) != EXPECTED_SCENARIOS:
        raise DecisionContractError("S0-S10 coverage or canonical order drifted")
    if len(result) != len(EXPECTED_SCENARIOS):
        raise DecisionContractError("duplicate synthetic scenario ID")
    return result


def _compare_unaffected_scenarios(
    historical: Mapping[str, Any], v2: Mapping[str, Any]
) -> None:
    old = _scenario_map(historical)
    new = _scenario_map(v2)
    for scenario_id in ("S0", "S1", "S2", "S3", "S4", "S5", "S8"):
        if old[scenario_id] != new[scenario_id]:
            raise DecisionContractError(f"unaffected {scenario_id} object drifted in V2")

    for scenario_id in ("S6", "S7"):
        reduced = dict(new[scenario_id])
        reduced.pop("estimation_error_law", None)
        reduced.pop("required_C85T_outputs", None)
        if reduced != old[scenario_id]:
            raise DecisionContractError(f"{scenario_id} changed beyond additive error-law completion")

    reduced_s10 = dict(new["S10"])
    reduced_s10.pop("historical_registered_policy", None)
    reduced_s10.pop("expected_exact_risks", None)
    reduced_s10.pop("registered_reversal_attribution", None)
    reduced_s10["registered_policies"] = dict(reduced_s10["registered_policies"])
    reduced_s10["registered_policies"]["rich"] = "always_action_1"
    if reduced_s10 != old["S10"]:
        raise DecisionContractError("S10 changed beyond the rich registered-policy repair")

    old_s9 = old["S9"]
    new_s9 = new["S9"]
    if new_s9["actions"] != old_s9["actions"]:
        raise DecisionContractError("S9 action set drifted")
    if tuple(_fraction(x) for x in new_s9["stratum_probabilities"]) != tuple(
        _fraction(x) for x in old_s9["stratum_probabilities"]
    ):
        raise DecisionContractError("S9 stratum masses drifted")
    if tuple(_fraction(new_s9["loss_vector_law"]["sigma"][h]) for h in ("L", "H")) != tuple(
        _fraction(x) for x in old_s9["pairwise_loss_sd"]
    ):
        raise DecisionContractError("S9 registered pairwise scales drifted")
    for field in (
        "query_observation",
        "passive_policy",
        "active_policy",
        "query_budget",
        "risk_functionals",
        "sample_size",
        "success_criterion",
        "theorem_targets",
    ):
        if new_s9[field] != old_s9[field]:
            raise DecisionContractError(f"S9 historical field drifted: {field}")


def validate_locked_contracts() -> dict[str, Any]:
    c85p_locked = c85p.validate_locked_inputs()
    c85p.validate_materialized_tables()
    historical_sha = sha256_file(HISTORICAL_CONTRACT_PATH)
    repair_sha = sha256_file(REPAIR_PROTOCOL_PATH)
    v2_sha = sha256_file(V2_CONTRACT_PATH)
    if historical_sha != EXPECTED_HISTORICAL_SHA256:
        raise DecisionContractError("historical C85P synthetic contract drifted")
    if repair_sha != EXPECTED_REPAIR_PROTOCOL_SHA256:
        raise DecisionContractError("C85R repair protocol bytes drifted")
    if v2_sha != EXPECTED_V2_CONTRACT_SHA256:
        raise DecisionContractError("C85R V2 synthetic contract bytes drifted")
    if _sidecar_entries(REPAIR_PROTOCOL_SIDECAR).get(REPAIR_PROTOCOL_PATH.name) != repair_sha:
        raise DecisionContractError("C85R repair protocol sidecar mismatch")
    if _sidecar_entries(V2_CONTRACT_SIDECAR).get(V2_CONTRACT_PATH.name) != v2_sha:
        raise DecisionContractError("C85R V2 contract sidecar mismatch")

    historical = read_json(HISTORICAL_CONTRACT_PATH)
    protocol = read_json(REPAIR_PROTOCOL_PATH)
    v2 = read_json(V2_CONTRACT_PATH)
    if protocol["status"] != "PROTOCOL_LOCKED_BEFORE_V2_IMPLEMENTATION_OR_SEMANTIC_VALIDATION":
        raise DecisionContractError("C85R chronology status drifted")
    if v2["status"] != "SEMANTICALLY_REPAIRED_LOCKED_NOT_SCIENTIFICALLY_EXECUTED":
        raise DecisionContractError("C85R V2 execution boundary drifted")
    if v2["semantic_validation"]["status"] != SEMANTIC_STATUS:
        raise DecisionContractError("C85R semantic validation status drifted")
    if v2["semantic_validation"]["scientific_simulation_executed"] is not False:
        raise DecisionContractError("C85R cannot execute synthetic science")
    if v2["semantic_validation"]["proof_executed"] is not False:
        raise DecisionContractError("C85R cannot complete project proofs")
    if v2.get("outcome_informed_design") is not False:
        raise DecisionContractError("C85R V2 parameters must remain outcome independent")
    if v2.get("theorem_statuses") != {f"T{i}": "OPEN" for i in range(1, 8)}:
        raise DecisionContractError("T1-T7 must remain OPEN")
    if protocol["proof_obligation_precision"]["theorem_status_change"] is not False:
        raise DecisionContractError("proof precision addendum cannot change theorem status")
    _compare_unaffected_scenarios(historical, v2)
    return {
        "historical": historical,
        "protocol": protocol,
        "v2": v2,
        "c85p": c85p_locked,
        "historical_sha256": historical_sha,
        "repair_protocol_sha256": repair_sha,
        "v2_sha256": v2_sha,
    }


def _regret_table(utilities: Sequence[Sequence[Fraction]]) -> tuple[tuple[Fraction, ...], ...]:
    result: list[tuple[Fraction, ...]] = []
    for row in utilities:
        optimum = max(row)
        result.append(tuple(optimum - value for value in row))
    return tuple(result)


def _fixed_action_risk(
    prior: Sequence[Fraction], regrets: Sequence[Sequence[Fraction]], action: int
) -> Fraction:
    return sum((prior[state] * regrets[state][action] for state in range(len(prior))), Fraction(0))


def _bayes_policy_and_risk(
    prior: Sequence[Fraction],
    channel: Sequence[Sequence[Fraction]],
    regrets: Sequence[Sequence[Fraction]],
) -> tuple[tuple[int, ...], Fraction]:
    observations = len(channel[0])
    actions = len(regrets[0])
    policy: list[int] = []
    risk = Fraction(0)
    for observation in range(observations):
        action_risks = []
        for action in range(actions):
            action_risks.append(sum(
                (
                    prior[state]
                    * channel[state][observation]
                    * regrets[state][action]
                    for state in range(len(prior))
                ),
                Fraction(0),
            ))
        selected = min(range(actions), key=lambda action: (action_risks[action], action))
        policy.append(selected)
        risk += action_risks[selected]
    return tuple(policy), risk


def exact_s10_audit(locked: Mapping[str, Any] | None = None) -> dict[str, Any]:
    values = dict(locked or validate_locked_contracts())
    historical = _scenario_map(values["historical"])["S10"]
    v2 = _scenario_map(values["v2"])["S10"]
    prior = tuple(_fraction(value) for value in historical["state_probabilities"])
    utilities = tuple(tuple(_fraction(value) for value in row) for row in historical["utilities"])
    regrets = _regret_table(utilities)
    coarse = tuple(tuple(_fraction(value) for value in row) for row in historical["experiments"]["coarse"])
    rich = tuple(tuple(_fraction(value) for value in row) for row in historical["experiments"]["rich"])
    garbling = tuple(tuple(_fraction(value) for value in row) for row in historical["garbling_rich_to_coarse"])
    if rich != ((Fraction(1), Fraction(0), Fraction(0)), (Fraction(0), Fraction(1), Fraction(0)), (Fraction(0), Fraction(0), Fraction(1))):
        raise DecisionContractError("S10 rich experiment is not exact state revelation")
    if garbling != coarse:
        raise DecisionContractError("S10 rich-to-coarse garbling witness drifted")

    coarse_policy, coarse_risk = _bayes_policy_and_risk(prior, coarse, regrets)
    historical_rich_risk = _fixed_action_risk(prior, regrets, 1)
    v2_rich_risk = _fixed_action_risk(prior, regrets, 0)
    rich_unrestricted_risk = Fraction(0)
    rich_gap = v2_rich_risk - rich_unrestricted_risk
    reversal = v2_rich_risk - coarse_risk
    expected = {
        "coarse_risk": Fraction(11, 40),
        "historical_rich_risk": Fraction(11, 40),
        "rich_unrestricted_risk": Fraction(0),
        "v2_rich_risk": Fraction(3, 5),
        "rich_gap": Fraction(3, 5),
        "reversal": Fraction(13, 40),
    }
    observed = {
        "coarse_risk": coarse_risk,
        "historical_rich_risk": historical_rich_risk,
        "rich_unrestricted_risk": rich_unrestricted_risk,
        "v2_rich_risk": v2_rich_risk,
        "rich_gap": rich_gap,
        "reversal": reversal,
    }
    if coarse_policy != (1, 1):
        raise DecisionContractError("S10 coarse Bayes action map is not action 1 at y0/y1")
    if observed != expected:
        raise DecisionContractError("S10 exact risks do not satisfy the V2 repair")
    if historical_rich_risk != coarse_risk:
        raise DecisionContractError("historical S10 contradiction was not detected")
    if not (rich_unrestricted_risk < coarse_risk < v2_rich_risk):
        raise DecisionContractError("S10 V2 semantic inequalities fail")
    if v2["registered_policies"]["rich"] != "always_action_0":
        raise DecisionContractError("S10 V2 rich policy is not the registered repair")
    return {"coarse_policy": coarse_policy, **observed}


def largest_remainder_allocation(weights: Sequence[Fraction], budget: int) -> tuple[int, ...]:
    if budget <= 0 or not weights or any(weight < 0 for weight in weights):
        raise DecisionContractError("invalid largest-remainder allocation input")
    total = sum(weights, Fraction(0))
    if total <= 0:
        raise DecisionContractError("allocation weights must have positive mass")
    ideals = [Fraction(budget) * weight / total for weight in weights]
    allocation = [value.numerator // value.denominator for value in ideals]
    remaining = budget - sum(allocation)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(ideals[index] - allocation[index]), index),
    )
    for index in order[:remaining]:
        allocation[index] += 1
    return tuple(allocation)


def _stratified_variance(
    masses: Sequence[Fraction], sigmas: Sequence[Fraction], allocation: Sequence[int]
) -> Fraction:
    if len(masses) != len(sigmas) or len(masses) != len(allocation):
        raise DecisionContractError("stratified variance dimensions differ")
    if any(count <= 0 for count in allocation):
        raise DecisionContractError("stratum allocation must be positive")
    return sum(
        (mass * mass * sigma * sigma / count for mass, sigma, count in zip(masses, sigmas, allocation)),
        Fraction(0),
    )


def exact_s9_audit(locked: Mapping[str, Any] | None = None) -> dict[str, Any]:
    values = dict(locked or validate_locked_contracts())
    scenario = _scenario_map(values["v2"])["S9"]
    masses = tuple(_fraction(value) for value in scenario["stratum_probabilities"])
    sigmas = tuple(_fraction(scenario["loss_vector_law"]["sigma"][name]) for name in ("L", "H"))
    if masses != (Fraction(4, 5), Fraction(1, 5)):
        raise DecisionContractError("S9 stratum masses drifted")
    if sigmas != (Fraction(1, 50), Fraction(1, 5)):
        raise DecisionContractError("S9 pairwise scales drifted")

    support_rows: list[dict[str, Any]] = []
    population_means = [Fraction(0) for _ in range(4)]
    for stratum_index, stratum in enumerate(("L", "H")):
        sigma = sigmas[stratum_index]
        for rademacher in (-1, 1):
            losses = (
                Fraction(3, 10),
                Fraction(3, 10) + Fraction(1, 20) + sigma * rademacher,
                Fraction(13, 20),
                Fraction(17, 20),
            )
            if any(loss < 0 or loss > 1 for loss in losses):
                raise DecisionContractError("S9 loss-vector support leaves [0,1]")
            joint_probability = masses[stratum_index] * Fraction(1, 2)
            for action, loss in enumerate(losses):
                population_means[action] += joint_probability * loss
            support_rows.append({
                "stratum": stratum,
                "rademacher": rademacher,
                "joint_probability": joint_probability,
                "losses": losses,
                "difference_1_minus_0": losses[1] - losses[0],
            })

    expected_means = (Fraction(3, 10), Fraction(7, 20), Fraction(13, 20), Fraction(17, 20))
    if tuple(population_means) != expected_means:
        raise DecisionContractError("S9 population loss means drifted")
    if min(range(4), key=lambda action: (population_means[action], action)) != 0:
        raise DecisionContractError("S9 action 0 is not uniquely population optimal")

    for stratum_index, stratum in enumerate(("L", "H")):
        differences = [row["difference_1_minus_0"] for row in support_rows if row["stratum"] == stratum]
        mean = sum(differences, Fraction(0)) / len(differences)
        variance = sum(((value - mean) ** 2 for value in differences), Fraction(0)) / len(differences)
        if mean != Fraction(1, 20) or variance != sigmas[stratum_index] ** 2:
            raise DecisionContractError(f"S9 {stratum} pairwise moments drifted")

    budget = int(scenario["query_budget"])
    passive = largest_remainder_allocation(masses, budget)
    neyman_weights = tuple(mass * sigma for mass, sigma in zip(masses, sigmas))
    neyman = largest_remainder_allocation(neyman_weights, budget)
    if passive != (51, 13) or neyman != (18, 46):
        raise DecisionContractError("S9 fixed allocations drifted")
    if scenario["passive_allocation"] != {"L": 51, "H": 13}:
        raise DecisionContractError("S9 passive allocation registry mismatch")
    if scenario["neyman_allocation"] != {"L": 18, "H": 46}:
        raise DecisionContractError("S9 Neyman allocation registry mismatch")
    passive_variance = _stratified_variance(masses, sigmas, passive)
    neyman_variance = _stratified_variance(masses, sigmas, neyman)
    if not neyman_variance < passive_variance:
        raise DecisionContractError("S9 fixed Neyman variance is not below passive variance")
    return {
        "support_rows": support_rows,
        "population_means": tuple(population_means),
        "masses": masses,
        "sigmas": sigmas,
        "passive_allocation": passive,
        "neyman_allocation": neyman,
        "passive_variance": passive_variance,
        "neyman_variance": neyman_variance,
    }


def exact_s6_s7_noise_audit(locked: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    values = dict(locked or validate_locked_contracts())
    scenarios = _scenario_map(values["v2"])
    result: dict[str, dict[str, Any]] = {}
    for scenario_id in ("S6", "S7"):
        scenario = scenarios[scenario_id]
        sigma = _fraction(scenario["pairwise_sigma"])
        action_error_variance = sigma * sigma / 2
        pairwise_variance = action_error_variance * 2
        shared_star_covariance = action_error_variance
        pairwise_correlation = shared_star_covariance / pairwise_variance
        law = scenario["estimation_error_law"]
        if law["action_errors"] != "iid Normal(0,pairwise_sigma^2/2)":
            raise DecisionContractError(f"{scenario_id} action-error law drifted")
        if "not independent" not in law["pairwise_dependence"]:
            raise DecisionContractError(f"{scenario_id} dependence disclosure missing")
        if pairwise_variance != sigma * sigma or pairwise_correlation != Fraction(1, 2):
            raise DecisionContractError(f"{scenario_id} pairwise scale/coupling identity fails")
        if int(scenario["sample_size"]) != 4096:
            raise DecisionContractError(f"{scenario_id} replicate contract drifted")
        result[scenario_id] = {
            "pairwise_sigma": sigma,
            "action_error_variance": action_error_variance,
            "pairwise_variance": pairwise_variance,
            "shared_star_covariance": shared_star_covariance,
            "pairwise_correlation": pairwise_correlation,
            "replicates": int(scenario["sample_size"]),
            "outputs": tuple(scenario["required_C85T_outputs"]),
        }
    return result


def _validate_t7_and_proof_precision(locked: Mapping[str, Any]) -> None:
    protocol = locked["protocol"]
    t7 = protocol["T7_bound_repair"]
    if "Delta_i^2" not in t7["primary_target"] or "Delta_i-epsilon" in t7["primary_target"]:
        raise DecisionContractError("T7 primary bound must use Delta_i, not Delta_i-epsilon")
    if "Delta_i-epsilon" not in t7["historical_looser_diagnostic"]:
        raise DecisionContractError("T7 historical looser diagnostic was not preserved")
    if t7["independence_assumed"] is not False or t7["proof_status"] != "OPEN":
        raise DecisionContractError("T7 dependence or proof status drifted")
    precision = protocol["proof_obligation_precision"]
    if "action kernels" not in precision["T3"] or "coupled draw is insufficient" not in precision["T3"]:
        raise DecisionContractError("T3 randomized-kernel precision is incomplete")
    if "disjoint optimal-action sets" not in precision["T4"] or "equal prior" not in precision["T4"]:
        raise DecisionContractError("T4 optimum/decoder precision is incomplete")
    if "open interval (0,1)" not in precision["T6"]:
        raise DecisionContractError("T6 CVaR alpha endpoint exclusion is incomplete")


def semantic_preflight() -> dict[str, Any]:
    locked = validate_locked_contracts()
    s10 = exact_s10_audit(locked)
    s9 = exact_s9_audit(locked)
    noise = exact_s6_s7_noise_audit(locked)
    _validate_t7_and_proof_precision(locked)
    scenarios = _scenario_map(locked["v2"])
    for scenario_id, scenario in scenarios.items():
        if not scenario.get("risk_functionals"):
            raise DecisionContractError(f"{scenario_id} has no risk functional")
        if not scenario.get("success_criterion"):
            raise DecisionContractError(f"{scenario_id} has no success criterion")
        if not isinstance(scenario.get("sample_size"), int) or scenario["sample_size"] < 0:
            raise DecisionContractError(f"{scenario_id} sample-size schema invalid")
        seeds = [c85p.deterministic_seed(scenario_id, replicate) for replicate in (0, 1, 4095)]
        if len(set(seeds)) != len(seeds):
            raise DecisionContractError(f"{scenario_id} deterministic seed collision in preflight keys")
    if any(status != "OPEN" for status in locked["v2"]["theorem_statuses"].values()):
        raise DecisionContractError("C85R cannot transition a theorem status")
    return {
        "status": SEMANTIC_STATUS,
        "scenario_count": len(scenarios),
        "theorem_statuses": dict(locked["v2"]["theorem_statuses"]),
        "S10": {
            "historical_equal_risk": _fraction_text(s10["historical_rich_risk"]),
            "v2_rich_risk": _fraction_text(s10["v2_rich_risk"]),
            "reversal": _fraction_text(s10["reversal"]),
        },
        "S9": {
            "passive_allocation": list(s9["passive_allocation"]),
            "neyman_allocation": list(s9["neyman_allocation"]),
            "passive_variance": _fraction_text(s9["passive_variance"]),
            "neyman_variance": _fraction_text(s9["neyman_variance"]),
        },
        "S6_S7": {
            scenario_id: {
                "pairwise_variance": _fraction_text(row["pairwise_variance"]),
                "shared_star_correlation": _fraction_text(row["pairwise_correlation"]),
            }
            for scenario_id, row in noise.items()
        },
        "scientific_simulations": 0,
        "proofs_completed": 0,
    }


def _rows(*rows: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    if not result:
        raise DecisionContractError("registry cannot be empty")
    fields = tuple(result[0])
    if any(tuple(row) != fields for row in result):
        raise DecisionContractError("registry row schemas differ")
    return result


def build_tables() -> dict[str, list[dict[str, Any]]]:
    locked = validate_locked_contracts()
    preflight = semantic_preflight()
    s10 = exact_s10_audit(locked)
    s9 = exact_s9_audit(locked)
    noise = exact_s6_s7_noise_audit(locked)
    protocol = locked["protocol"]

    supersession = _rows(
        {"object": "historical_generator_contract_v1", "historical_sha256": EXPECTED_HISTORICAL_SHA256, "issue": "schema-valid but semantic blockers in S10/S9/S6/S7", "v2_action": "preserve immutable and add V2", "in_place_modified": 0, "operative_status": "SUPERSEDED_FOR_C85T_BY_V2"},
        {"object": "S10", "historical_sha256": EXPECTED_HISTORICAL_SHA256, "issue": "registered risks equal 11/40", "v2_action": "change only rich registered policy to always action 0", "in_place_modified": 0, "operative_status": "V2_REPAIRED"},
        {"object": "S9", "historical_sha256": EXPECTED_HISTORICAL_SHA256, "issue": "joint loss-vector law absent", "v2_action": "add exact two-stratum Rademacher full-information law", "in_place_modified": 0, "operative_status": "V2_COMPLETED"},
        {"object": "S6_S7", "historical_sha256": EXPECTED_HISTORICAL_SHA256, "issue": "stochastic action-error law absent", "v2_action": "add iid Gaussian action errors and shared-star dependence", "in_place_modified": 0, "operative_status": "V2_COMPLETED"},
    )

    s10_rows = _rows(
        {"object": "coarse_Bayes_policy", "policy": f"y0->{s10['coarse_policy'][0]}|y1->{s10['coarse_policy'][1]}", "exact_risk": _fraction_text(s10["coarse_risk"]), "decimal_risk": _decimal_text(s10["coarse_risk"]), "comparison": "historical_coarse_registered", "status": "EXACT_PREFLIGHT_PASS"},
        {"object": "historical_rich_registered", "policy": "always_action_1", "exact_risk": _fraction_text(s10["historical_rich_risk"]), "decimal_risk": _decimal_text(s10["historical_rich_risk"]), "comparison": "equals_coarse_not_strictly_greater", "status": "HISTORICAL_CONTRADICTION_DETECTED"},
        {"object": "V2_rich_unrestricted", "policy": "statewise_optimal", "exact_risk": _fraction_text(s10["rich_unrestricted_risk"]), "decimal_risk": _decimal_text(s10["rich_unrestricted_risk"]), "comparison": "strictly_below_coarse", "status": "EXACT_PREFLIGHT_PASS"},
        {"object": "V2_rich_registered", "policy": "always_action_0", "exact_risk": _fraction_text(s10["v2_rich_risk"]), "decimal_risk": _decimal_text(s10["v2_rich_risk"]), "comparison": "strictly_above_coarse", "status": "EXACT_PREFLIGHT_PASS"},
        {"object": "V2_rich_policy_gap", "policy": "registered_minus_unrestricted", "exact_risk": _fraction_text(s10["rich_gap"]), "decimal_risk": _decimal_text(s10["rich_gap"]), "comparison": "positive", "status": "EXACT_PREFLIGHT_PASS"},
        {"object": "V2_registered_reversal", "policy": "rich_registered_minus_coarse_registered", "exact_risk": _fraction_text(s10["reversal"]), "decimal_risk": _decimal_text(s10["reversal"]), "comparison": "13/40", "status": "EXACT_PREFLIGHT_PASS"},
    )

    loss_rows = _rows(*(
        {
            "stratum": row["stratum"],
            "R": row["rademacher"],
            "joint_probability": _fraction_text(row["joint_probability"]),
            "loss_action_0": _fraction_text(row["losses"][0]),
            "loss_action_1": _fraction_text(row["losses"][1]),
            "loss_action_2": _fraction_text(row["losses"][2]),
            "loss_action_3": _fraction_text(row["losses"][3]),
            "loss1_minus_loss0": _fraction_text(row["difference_1_minus_0"]),
            "population_mean_loss_action_0": _fraction_text(s9["population_means"][0]),
            "population_mean_loss_action_1": _fraction_text(s9["population_means"][1]),
            "population_mean_loss_action_2": _fraction_text(s9["population_means"][2]),
            "population_mean_loss_action_3": _fraction_text(s9["population_means"][3]),
            "stratum_pairwise_mean": "1/20",
            "stratum_pairwise_sd": _fraction_text(s9["sigmas"][("L", "H").index(row["stratum"])]),
            "support_in_0_1": 1,
            "status": SEMANTIC_STATUS,
        }
        for row in s9["support_rows"]
    ))

    allocation_rows = _rows(
        {"design": "passive_proportional", "budget": 64, "ideal_weight_L": "4/5", "ideal_weight_H": "1/5", "rounding": "largest_remainder_ties_L_then_H", "n_L": s9["passive_allocation"][0], "n_H": s9["passive_allocation"][1], "status": SEMANTIC_STATUS},
        {"design": "Neyman_p_times_sigma", "budget": 64, "ideal_weight_L": "2/125", "ideal_weight_H": "1/25", "rounding": "largest_remainder_ties_L_then_H", "n_L": s9["neyman_allocation"][0], "n_H": s9["neyman_allocation"][1], "status": SEMANTIC_STATUS},
    )
    variance_rows = _rows(
        {"design": "passive_proportional", "formula": "sum_h p_h^2*sigma_h^2/n_h", "exact_variance": _fraction_text(s9["passive_variance"]), "decimal_variance": _decimal_text(s9["passive_variance"]), "below_passive": "NA_REFERENCE", "universal_superiority_claim": 0, "status": SEMANTIC_STATUS},
        {"design": "Neyman_p_times_sigma", "formula": "sum_h p_h^2*sigma_h^2/n_h", "exact_variance": _fraction_text(s9["neyman_variance"]), "decimal_variance": _decimal_text(s9["neyman_variance"]), "below_passive": 1, "universal_superiority_claim": 0, "status": SEMANTIC_STATUS},
    )

    coupling_rows = _rows(*(
        {"scenario": scenario_id, "pairwise_sigma": _fraction_text(row["pairwise_sigma"]), "action_error_variance": _fraction_text(row["action_error_variance"]), "pairwise_difference_variance": _fraction_text(row["pairwise_variance"]), "shared_star_covariance": _fraction_text(row["shared_star_covariance"]), "pairwise_difference_correlation": _fraction_text(row["pairwise_correlation"]), "pairwise_errors_independent": 0, "replicates_future_C85T": row["replicates"], "scientific_replicates_executed_C85R": 0, "status": SEMANTIC_STATUS}
        for scenario_id, row in noise.items()
    ))
    output_rows = _rows(*(
        {"scenario": scenario_id, "output": output, "required_in_C85T": 1, "computed_in_C85R": 0, "status": "LOCKED_NOT_SCIENTIFICALLY_EXECUTED"}
        for scenario_id, row in noise.items()
        for output in row["outputs"]
    ))

    t7_rows = _rows(
        {"version": "historical_C85P", "role": "looser_candidate_diagnostic", "expression": protocol["T7_bound_repair"]["historical_looser_diagnostic"], "uses_Delta_minus_epsilon": 1, "primary_target": 0, "independence_required": 0, "theorem_status": "OPEN"},
        {"version": "C85R_V2", "role": "primary_union_bound_target", "expression": protocol["T7_bound_repair"]["primary_target"], "uses_Delta_minus_epsilon": 0, "primary_target": 1, "independence_required": 0, "theorem_status": "OPEN"},
    )
    precision_rows = _rows(
        {"theorem_id": "T3", "clarification": protocol["proof_obligation_precision"]["T3"], "required_condition": "statewise almost-sure action-kernel equality", "excluded_shortcut": "one coupled action draw", "status": "OPEN"},
        {"theorem_id": "T4", "clarification": protocol["proof_obligation_precision"]["T4"], "required_condition": "unique different optima or disjoint optimum sets plus decoder", "excluded_shortcut": "distinct named actions without separation", "status": "OPEN"},
        {"theorem_id": "T6", "clarification": protocol["proof_obligation_precision"]["T6"], "required_condition": "derive exact alpha region inside (0,1)", "excluded_shortcut": "alpha endpoints 0 or 1", "status": "OPEN"},
    )

    semantic_checks = [
        ("SV01", "historical contract SHA replay", 1),
        ("SV02", "repair protocol SHA replay", 1),
        ("SV03", "V2 contract SHA replay", 1),
        ("SV04", "S0-S5 and S8 unchanged", 1),
        ("SV05", "S6/S7 additive-only completion", 1),
        ("SV06", "S10 minimal rich-policy replacement", 1),
        ("SV07", "historical S10 equality detected", 1),
        ("SV08", "V2 S10 exact reversal satisfied", 1),
        ("SV09", "S9 support and moments exact", 1),
        ("SV10", "S9 allocations and variances exact", 1),
        ("SV11", "S6/S7 pairwise scale and coupling exact", 1),
        ("SV12", "T7 primary Delta bound registered", 1),
        ("SV13", "T3/T4/T6 precision present", 1),
        ("SV14", "all deterministic seed keys replay", 1),
        ("SV15", "T1-T7 remain OPEN", 1),
        ("SV16", "scientific simulations and proofs executed", 0),
    ]
    semantic_rows = _rows(*(
        {"check_id": check_id, "check": check, "observed": observed, "required": observed, "result": "PASS", "status": SEMANTIC_STATUS}
        for check_id, check, observed in semantic_checks
    ))
    theorem_rows = _rows(*(
        {"theorem_id": theorem_id, "historical_status": "OPEN", "V2_status": status, "proof_executed_C85R": 0, "status_transition": 0, "result": "PASS"}
        for theorem_id, status in locked["v2"]["theorem_statuses"].items()
    ))
    seed_rows = _rows(*(
        {"scenario_id": scenario_id, "replicate_0_seed": c85p.deterministic_seed(scenario_id, 0), "replicate_1_seed": c85p.deterministic_seed(scenario_id, 1), "replicate_4095_seed": c85p.deterministic_seed(scenario_id, 4095), "seed_rule_changed": 0, "scientific_draw_generated": 0, "status": SEMANTIC_STATUS}
        for scenario_id in EXPECTED_SCENARIOS
    ))
    identity_rows = _rows(
        {"object": "C85P_protocol", "path": "oaci/reports/C85_TPAMI_DECISION_THEORY_PROTOCOL.json", "sha256": c85p.EXPECTED_PROTOCOL_SHA256, "replay": "PASS", "modified": 0},
        {"object": "historical_generator_v1", "path": "oaci/reports/c85p_tables/synthetic_generator_contract.json", "sha256": locked["historical_sha256"], "replay": "PASS", "modified": 0},
        {"object": "C85R_repair_protocol", "path": "oaci/reports/C85R_SYNTHETIC_CONTRACT_SEMANTIC_REPAIR_PROTOCOL.json", "sha256": locked["repair_protocol_sha256"], "replay": "PASS", "modified": 0},
        {"object": "generator_v2", "path": "oaci/reports/c85r_tables/synthetic_generator_contract_v2.json", "sha256": locked["v2_sha256"], "replay": "PASS", "modified": 0},
    )

    risks = [
        ("R01", "historical contract rewritten", "immutable SHA and supersession ledger"),
        ("R02", "S10 utility table tuned", "minimal policy-only comparison"),
        ("R03", "S9 active superiority generalized", "fixed-scenario variance only"),
        ("R04", "S9 loss exits [0,1]", "exact support enumeration"),
        ("R05", "S6/S7 pairwise errors called independent", "shared-star covariance registry"),
        ("R06", "T7 looser diagnostic remains primary", "explicit V2 supersession"),
        ("R07", "T3 coupled draw substituted for kernel equality", "proof precision addendum"),
        ("R08", "T4 optimal-action sets overlap silently", "unique/disjoint condition"),
        ("R09", "T6 uses alpha endpoint", "open interval contract"),
        ("R10", "semantic preflight called scientific simulation", "zero-replicate execution counters"),
        ("R11", "theorem status changes in repair", "all OPEN replay"),
        ("R12", "real project arrays or active acquisition accessed", "static import and boundary tests"),
    ]
    risk_rows = _rows(*(
        {"risk_id": risk_id, "risk": risk, "mitigation": mitigation, "blocking_if_open": 1, "status": "CLOSED_AT_READINESS"}
        for risk_id, risk, mitigation in risks
    ))
    failure_rows = _rows(
        {"reason_id": "NONE_OPEN", "category": "READINESS", "status": "CLOSED", "blocking": 0, "detail": "No open C85R generative-semantics or proof-obligation blocker after exact semantic preflight."}
    )

    if preflight["status"] != SEMANTIC_STATUS:
        raise DecisionContractError("semantic preflight did not reach the locked non-scientific status")
    return {
        "historical_contract_supersession.csv": supersession,
        "S10_historical_exact_risk_audit.csv": s10_rows,
        "S9_loss_vector_law.csv": loss_rows,
        "S9_allocation_contract.csv": allocation_rows,
        "S9_analytic_variance_contract.csv": variance_rows,
        "S6_S7_noise_coupling_contract.csv": coupling_rows,
        "S6_S7_output_contract.csv": output_rows,
        "T7_bound_supersession.csv": t7_rows,
        "proof_obligation_precision_addendum.csv": precision_rows,
        "semantic_satisfiability_validation.csv": semantic_rows,
        "theorem_status_replay.csv": theorem_rows,
        "deterministic_seed_replay.csv": seed_rows,
        "artifact_identity_replay.csv": identity_rows,
        "risk_register.csv": risk_rows,
        "failure_reason_ledger.csv": failure_rows,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise DecisionContractError(f"refusing to write empty table {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise DecisionContractError(f"schema drift in {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def materialize_tables() -> dict[str, int]:
    tables = build_tables()
    for name, rows in tables.items():
        write_csv(C85R_TABLE_DIR / name, rows)
    return {name: len(rows) for name, rows in tables.items()}


def validate_materialized_tables() -> dict[str, int]:
    expected = build_tables()
    observed: dict[str, int] = {}
    for name, rows in expected.items():
        path = C85R_TABLE_DIR / name
        if not path.is_file():
            raise DecisionContractError(f"missing materialized C85R table {name}")
        with path.open(newline="") as handle:
            actual = list(csv.DictReader(handle))
        canonical_expected = [{key: str(value) for key, value in row.items()} for row in rows]
        if actual != canonical_expected:
            raise DecisionContractError(f"materialized C85R table drifted: {name}")
        observed[name] = len(actual)
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-contract")
    subparsers.add_parser("semantic-preflight")
    subparsers.add_parser("materialize-tables")
    subparsers.add_parser("validate-tables")
    args = parser.parse_args(argv)
    if args.command == "validate-contract":
        locked = validate_locked_contracts()
        result = {
            "historical_sha256": locked["historical_sha256"],
            "repair_protocol_sha256": locked["repair_protocol_sha256"],
            "v2_sha256": locked["v2_sha256"],
            "status": "LOCKED_V2_NOT_SCIENTIFICALLY_EXECUTED",
        }
    elif args.command == "semantic-preflight":
        result = semantic_preflight()
    elif args.command == "materialize-tables":
        result = {"tables": materialize_tables(), "status": SEMANTIC_STATUS}
    else:
        result = {"tables": validate_materialized_tables(), "status": SEMANTIC_STATUS}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
