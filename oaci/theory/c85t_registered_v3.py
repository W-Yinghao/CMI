"""Registered C85T V3 exact, Monte Carlo, and proof-candidate dispatchers."""
from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .c85_decision_experiments import DecisionContractError
from .c85r_synthetic_semantic_repair import exact_s10_audit, exact_s9_audit
from .c85t_exact_scenarios import (
    _serialize,
    as_fraction,
    bayes_policy_and_risk,
    exact_minimax_regret_lp,
    fixed_action_risk,
    near_optimal_geometry,
    regret_rows,
    s5_candidate_cvar_region,
    spearman,
    stable_argmax,
)
from .c85t_execution_context_v3 import (
    ValidatedC85TExecutionContext,
    validate_registered_execution_context,
)
from .c85t_monte_carlo import (
    _design_estimate,
    _summarize_s9_arrays_v2,
    summarize_near_replicates_v2,
)
from .c85t_proofs import (
    PROOF_FILENAMES,
    THEOREM_IDS,
    _CANDIDATE_DISPOSITIONS,
    _INTERNAL_CHECK_LABEL,
    _future_candidates,
    _render_proof_candidate_v2,
    validate_proof_candidate_markdown_v2,
)


REGISTERED_SCENARIOS = tuple(f"S{index}" for index in range(11))
SEED_NAMESPACE = "C85_SYNTHETIC_V1"


def _scenario_map(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    scenarios = {row["id"]: row for row in contract["scenarios"]}
    if tuple(scenarios) != REGISTERED_SCENARIOS:
        raise DecisionContractError("registered C85T V3 scenario order drifted")
    return scenarios


def _deterministic_seed(
    scenario_id: str, replicate_id: int, *, context: object
) -> int:
    validate_registered_execution_context(context)
    if scenario_id not in REGISTERED_SCENARIOS:
        raise DecisionContractError("unknown registered C85T V3 scenario")
    if not isinstance(replicate_id, int) or not 0 <= replicate_id <= 4095:
        raise DecisionContractError("C85T V3 replicate ID must be in 0..4095")
    payload = f"{SEED_NAMESPACE}|{scenario_id}|{replicate_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _generator(
    scenario_id: str, replicate_id: int, *, context: object
) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64DXSM(
            _deterministic_seed(scenario_id, replicate_id, context=context)
        )
    )


def _standard_normal(
    scenario_id: str, replicate_id: int, count: int, *, context: object
) -> np.ndarray:
    values = _generator(scenario_id, replicate_id, context=context).standard_normal(
        count, dtype=np.float64
    )
    return np.asarray(values, dtype="<f8")


def _s9_int64_draws(
    replicate_id: int, *, context: object
) -> tuple[np.ndarray, np.ndarray]:
    rng = _generator("S9", replicate_id, context=context)
    low = np.asarray(2 * rng.integers(0, 2, size=51, dtype=np.int64) - 1, dtype="<i8")
    high = np.asarray(2 * rng.integers(0, 2, size=46, dtype=np.int64) - 1, dtype="<i8")
    return low, high


def _raw_digest_row(replicate_id: int, low: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    low_bytes = np.ascontiguousarray(low, dtype="<i8").tobytes()
    high_bytes = np.ascontiguousarray(high, dtype="<i8").tobytes()
    combined = hashlib.sha256(low_bytes + high_bytes).hexdigest()
    return {
        "replicate_id": replicate_id,
        "L_sha256": hashlib.sha256(low_bytes).hexdigest(),
        "H_sha256": hashlib.sha256(high_bytes).hexdigest(),
        "combined_sha256": combined,
        "dtype": "<i8",
        "L_count": 51,
        "H_count": 46,
    }


def execute_registered_exact_scenarios_v3(
    contract: Mapping[str, Any], *, context: object
) -> dict[str, Any]:
    """Execute the authoritative exact scenarios after receipt revalidation."""

    validate_registered_execution_context(context)
    scenarios = _scenario_map(contract)
    output: dict[str, Any] = {}

    s0 = scenarios["S0"]
    s0_regrets = regret_rows(s0["utilities"])
    output["S0"] = {
        "optimal_constant_risk": fixed_action_risk(
            s0["state_probabilities"], s0_regrets, 0
        )
    }

    s1 = scenarios["S1"]
    s1_regrets = regret_rows(s1["utilities"])
    _, coarse = bayes_policy_and_risk(
        s1["state_probabilities"], s1["experiments"]["E1"], s1_regrets
    )
    _, rich_unrestricted = bayes_policy_and_risk(
        s1["state_probabilities"], s1["experiments"]["E2"], s1_regrets
    )
    rich_registered = sum(
        (
            as_fraction(mass) * s1_regrets[state][1 - state]
            for state, mass in enumerate(s1["state_probabilities"])
        ),
        Fraction(),
    )
    output["S1"] = {
        "coarse_registered_risk": coarse,
        "rich_unrestricted_risk": rich_unrestricted,
        "rich_registered_risk": rich_registered,
    }

    s2 = scenarios["S2"]
    s2_regrets = regret_rows(s2["utilities"])
    collapsed = fixed_action_risk(s2["state_probabilities"], s2_regrets, 0)
    output["S2"] = {
        "action_divergence": Fraction(),
        "registered_risk": collapsed,
        "reference_risk": collapsed,
    }

    s3 = scenarios["S3"]
    utilities3 = tuple(float(value) for value in s3["utilities"][0])
    scores3 = tuple(float(value) for value in s3["selector_scores"][0])
    selected3 = stable_argmax(scores3)
    output["S3"] = {
        "selected_action": selected3,
        "regret": max(utilities3) - utilities3[selected3],
        "spearman": spearman(scores3, utilities3),
    }

    s4 = scenarios["S4"]
    utilities4 = tuple(as_fraction(value) for value in s4["utilities"][0])
    selected4 = int(s4["selected_action"])
    output["S4"] = {
        "top4_localization": int(
            utilities4.index(max(utilities4)) in s4["reported_topk_candidate_set"]
        ),
        "selected_regret": max(utilities4) - utilities4[selected4],
    }

    output["S5"] = s5_candidate_cvar_region(scenarios["S5"])
    for scenario_id in ("S6", "S7"):
        row = scenarios[scenario_id]
        output[scenario_id] = near_optimal_geometry(
            tuple(float(value) for value in row["utilities"][0]),
            float(row["epsilon"]),
            float(row["tau"]),
            float(row["pairwise_sigma"]),
        )
    output["S8"] = exact_minimax_regret_lp(
        scenarios["S8"]["utility_extreme_points"]
    )
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


def _simulate_near(
    scenario_id: str, row: Mapping[str, Any], *, context: object
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    utilities = np.asarray(row["utilities"][0], dtype="<f8")
    epsilon = float(row["epsilon"])
    pairwise_sigma = float(row["pairwise_sigma"])
    geometry = near_optimal_geometry(
        tuple(float(value) for value in utilities),
        epsilon,
        float(row["tau"]),
        pairwise_sigma,
    )
    optimum = int(np.argmax(utilities))
    near = set(geometry["epsilon_near_optimal_set"])
    arrays = {
        "replicate_id": np.arange(4096, dtype="<u2"),
        "selected_action": np.empty(4096, dtype="<u2"),
        "top1": np.empty(4096, dtype=np.uint8),
        "outside_A_epsilon": np.empty(4096, dtype=np.uint8),
        "selection_regret": np.empty(4096, dtype="<f8"),
    }
    scale = pairwise_sigma / math.sqrt(2.0)
    for replicate in range(4096):
        selected = int(
            np.argmax(
                utilities
                + _standard_normal(
                    scenario_id, replicate, utilities.size, context=context
                )
                * scale
            )
        )
        arrays["selected_action"][replicate] = selected
        arrays["top1"][replicate] = int(selected == optimum)
        arrays["outside_A_epsilon"][replicate] = int(selected not in near)
        arrays["selection_regret"][replicate] = utilities[optimum] - utilities[selected]
    summary = summarize_near_replicates_v2(scenario_id, arrays, geometry)
    return summary, arrays


def _simulate_s9(
    row: Mapping[str, Any], *, context: object
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    population = np.asarray(
        [float(as_fraction(value)) for value in row["population_mean_losses"]],
        dtype="<f8",
    )
    true_best = int(np.argmin(population))
    base = population.copy()
    base[1] = base[0]
    masses = tuple(float(as_fraction(value)) for value in row["stratum_probabilities"])
    sigmas = (1 / 50, 1 / 5)
    allocations = {
        "passive": (row["passive_allocation"]["L"], row["passive_allocation"]["H"]),
        "neyman": (row["neyman_allocation"]["L"], row["neyman_allocation"]["H"]),
    }
    arrays: dict[str, np.ndarray] = {}
    for design in ("passive", "neyman"):
        arrays[f"{design}_replicate_id"] = np.arange(4096, dtype="<u2")
        arrays[f"{design}_selected_action"] = np.empty(4096, dtype=np.uint8)
        arrays[f"{design}_correct_best"] = np.empty(4096, dtype=np.uint8)
        arrays[f"{design}_top2_coverage"] = np.empty(4096, dtype=np.uint8)
        arrays[f"{design}_selection_regret"] = np.empty(4096, dtype="<f8")
        arrays[f"{design}_D_hat"] = np.empty(4096, dtype="<f8")
    digest_rows: list[dict[str, Any]] = []
    for replicate in range(4096):
        low_raw, high_raw = _s9_int64_draws(replicate, context=context)
        digest_rows.append(_raw_digest_row(replicate, low_raw, high_raw))
        low = low_raw.astype(np.int8)
        high = high_raw.astype(np.int8)
        for design, (low_count, high_count) in allocations.items():
            estimated = _design_estimate(
                low,
                high,
                low_count=low_count,
                high_count=high_count,
                stratum_masses=masses,
                base_losses=base,
                action1_offset=1 / 20,
                sigmas=sigmas,
            )
            order = np.argsort(estimated, kind="stable")
            selected = int(order[0])
            arrays[f"{design}_selected_action"][replicate] = selected
            arrays[f"{design}_correct_best"][replicate] = int(selected == true_best)
            arrays[f"{design}_top2_coverage"][replicate] = int(true_best in order[:2])
            arrays[f"{design}_selection_regret"][replicate] = (
                population[selected] - population[true_best]
            )
            arrays[f"{design}_D_hat"][replicate] = estimated[1] - estimated[0]
    for endpoint in ("selection_regret", "correct_best", "top2_coverage", "D_hat"):
        arrays[f"paired_passive_minus_neyman_{endpoint}"] = np.asarray(
            arrays[f"passive_{endpoint}"].astype("<f8")
            - arrays[f"neyman_{endpoint}"].astype("<f8"),
            dtype="<f8",
        )
    summary = _summarize_s9_arrays_v2(arrays, population)
    summary["analytic_variance"] = {
        "passive_d_hat_variance": sum(
            mass * mass * sigma * sigma / count
            for mass, sigma, count in zip(masses, sigmas, allocations["passive"])
        ),
        "neyman_d_hat_variance": sum(
            mass * mass * sigma * sigma / count
            for mass, sigma, count in zip(masses, sigmas, allocations["neyman"])
        ),
    }
    summary["universal_active_superiority_claim"] = False
    return summary, arrays, digest_rows


def execute_registered_monte_carlo_v3(
    contract: Mapping[str, Any], *, context: object
) -> dict[str, Any]:
    """Run the three registered deterministic Monte Carlo streams."""

    validate_registered_execution_context(context)
    scenarios = _scenario_map(contract)
    result: dict[str, Any] = {}
    for scenario_id in ("S6", "S7"):
        summary, arrays = _simulate_near(
            scenario_id, scenarios[scenario_id], context=context
        )
        result[scenario_id] = {"summary": summary, "arrays": arrays}
    summary, arrays, digest_rows = _simulate_s9(scenarios["S9"], context=context)
    result["S9"] = {
        "summary": summary,
        "arrays": arrays,
        "raw_draw_digest_rows": digest_rows,
    }
    return result


def replay_registered_s9_digest_rows_v3(
    rows: Sequence[Mapping[str, Any]], *, context: object
) -> int:
    """Rerun the consumed S9 stream only to replay its frozen raw digests."""

    validate_registered_execution_context(context)
    if len(rows) != 4096:
        raise DecisionContractError("C85T V3 S9 digest registry count drifted")
    for replicate, observed in enumerate(rows):
        low, high = _s9_int64_draws(replicate, context=context)
        expected = _raw_digest_row(replicate, low, high)
        if dict(observed) != {key: str(value) for key, value in expected.items()}:
            raise DecisionContractError(f"C85T V3 S9 raw digest drifted: {replicate}")
    return len(rows)


def replay_registered_s9_artifacts_v3(
    contract: Mapping[str, Any], *, context: object
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Recompute S9 under the consumed attempt for semantic artifact replay."""

    validate_registered_execution_context(context)
    return _simulate_s9(_scenario_map(contract)["S9"], context=context)


def execute_proof_candidate_pipeline_v3(
    *,
    statements: Mapping[str, str],
    exact_results: Mapping[str, Any],
    output_dir: Path,
    dispositions_path: Path,
    context: object,
) -> dict[str, dict[str, str]]:
    """Freeze non-dispositive proof candidates while all statuses remain OPEN."""

    validate_registered_execution_context(context)
    if output_dir.exists() or dispositions_path.exists():
        raise DecisionContractError("C85T V3 proof-candidate outputs must be fresh")
    candidates = _future_candidates(statements, exact_results)
    output_dir.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    result: dict[str, dict[str, str]] = {}
    for theorem_id in THEOREM_IDS:
        candidate = candidates[theorem_id]
        text = _render_proof_candidate_v2(candidate)
        validate_proof_candidate_markdown_v2(text, theorem_id)
        path = output_dir / PROOF_FILENAMES[theorem_id]
        path.write_text(text)
        row = {
            "theorem_id": theorem_id,
            "historical_status": "OPEN",
            "candidate_disposition": _CANDIDATE_DISPOSITIONS[theorem_id],
            "formal_status": "OPEN",
            "check_class": _INTERNAL_CHECK_LABEL,
            "proof_candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        rows.append(row)
        result[theorem_id] = dict(row)
    dispositions_path.parent.mkdir(parents=True, exist_ok=True)
    with dispositions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(row["formal_status"] != "OPEN" for row in rows):
        raise DecisionContractError("C85T V3 attempted a theorem transition")
    return result


__all__ = (
    "execute_proof_candidate_pipeline_v3",
    "execute_registered_exact_scenarios_v3",
    "execute_registered_monte_carlo_v3",
    "replay_registered_s9_artifacts_v3",
    "replay_registered_s9_digest_rows_v3",
)
