"""Synthetic calibration for the C80P label-budget frontier protocol."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import c80_label_budget_frontier as frontier


REPORT_DIR = frontier.REPORT_DIR
TABLE_DIR = frontier.TABLE_DIR
SCENARIO_REPLICATES = 512
SEED = 8080

SCENARIOS: dict[str, dict[str, Any]] = {
    "S0_no_label_information_flat_random_frontier": {
        "seed3": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        "seed4": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        "expected": (None, None), "stable": False, "reliability": 0.00,
    },
    "S1_reliable_measurement_no_regret_improvement": {
        "seed3": [0.00, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
        "seed4": [0.00, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
        "expected": (None, None), "stable": False, "reliability": 0.85,
    },
    "S2_low_budget_actionability_stable_monotone_frontier": {
        "seed3": [0.00, 0.02, 0.04, 0.14, 0.15, 0.16, 0.17],
        "seed4": [0.00, 0.02, 0.04, 0.14, 0.15, 0.16, 0.17],
        "expected": (8, 8), "stable": True, "reliability": 0.75,
    },
    "S3_actionability_only_near_FULL": {
        "seed3": [-0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.11],
        "seed4": [-0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.11],
        "expected": ("FULL", "FULL"), "stable": True, "reliability": 0.65,
    },
    "S4_seed_dependent_frontier": {
        "seed3": [0.00, 0.02, 0.04, 0.14, 0.15, 0.16, 0.17],
        "seed4": [0.00, 0.00, 0.01, 0.02, 0.03, 0.14, 0.15],
        "expected": (8, 32), "stable": False, "reliability": 0.70,
    },
    "S5_target_heterogeneous_catastrophic_subgroups": {
        "seed3": [0.00, 0.03, 0.06, 0.09, 0.11, 0.12, 0.13],
        "seed4": [0.00, 0.03, 0.06, 0.09, 0.11, 0.12, 0.13],
        "expected": (None, None), "stable": False, "reliability": 0.70,
        "catastrophic": True,
    },
    "S6_near_tie_requires_more_labels": {
        "seed3": [-0.01, 0.00, 0.01, 0.02, 0.04, 0.10, 0.12],
        "seed4": [-0.01, 0.00, 0.01, 0.02, 0.04, 0.10, 0.12],
        "expected": (32, 32), "stable": True, "reliability": 0.60,
    },
    "S7_nonmonotone_finite_sample_curve": {
        "seed3": [0.09, 0.02, 0.09, 0.03, 0.10, 0.11, 0.12],
        "seed4": [0.09, 0.02, 0.09, 0.03, 0.10, 0.11, 0.12],
        "expected": (16, 16), "stable": True, "reliability": 0.55,
    },
    "S8_pseudoreplication_trap": {
        "seed3": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        "seed4": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        "expected": (None, None), "stable": False, "reliability": 0.00,
    },
}


def _scenario_effects(name: str, replicate: int) -> tuple[np.ndarray, np.ndarray]:
    spec = SCENARIOS[name]
    rng = np.random.default_rng(SEED + 10000 * list(SCENARIOS).index(name) + replicate)
    shared_target = rng.normal(0.0, 0.003, size=(frontier.TARGETS, 1))
    seed3 = np.asarray(spec["seed3"], dtype=float)[None, :] + shared_target
    seed4 = np.asarray(spec["seed4"], dtype=float)[None, :] + shared_target
    seed3 = seed3 + rng.normal(0.0, 0.002, size=seed3.shape)
    seed4 = seed4 + rng.normal(0.0, 0.002, size=seed4.shape)
    if spec.get("catastrophic"):
        seed3[:2] -= 0.25
        seed4[:2] -= 0.25
    return seed3, seed4


def _stable(left: int | str | None, right: int | str | None) -> bool:
    distance = frontier.bstar_grid_distance(left, right)
    return bool(distance is not None and distance <= 1)


def calibrate_frontiers() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    bstar_rows = []
    for name, spec in SCENARIOS.items():
        recovered = {3: 0, 4: 0}
        stable_correct = 0
        observed: dict[int, list[str]] = {3: [], 4: []}
        for replicate in range(SCENARIO_REPLICATES):
            seed3, seed4 = _scenario_effects(name, replicate)
            result3 = frontier.budget_qualification(seed3)
            result4 = frontier.budget_qualification(seed4)
            for seed, result, expected in ((3, result3, spec["expected"][0]), (4, result4, spec["expected"][1])):
                recovered[seed] += int(result["Bstar"] == expected)
                observed[seed].append(str(result["Bstar"]))
            stable_correct += int(_stable(result3["Bstar"], result4["Bstar"]) == bool(spec["stable"]))
        for seed in (3, 4):
            rate = recovered[seed] / SCENARIO_REPLICATES
            expected = spec["expected"][seed - 3]
            bstar_rows.append({
                "scenario": name, "seed": seed, "expected_Bstar": expected,
                "recovery_rate": rate, "replicates": SCENARIO_REPLICATES,
                "criterion": 0.90, "passed": int(rate >= 0.90),
            })
        stable_rate = stable_correct / SCENARIO_REPLICATES
        summary_rows.append({
            "scenario": name,
            "expected_seed3_Bstar": spec["expected"][0],
            "expected_seed4_Bstar": spec["expected"][1],
            "expected_stable": int(spec["stable"]),
            "seed3_recovery": recovered[3] / SCENARIO_REPLICATES,
            "seed4_recovery": recovered[4] / SCENARIO_REPLICATES,
            "stability_classification": stable_rate,
            "reliability_parameter": spec["reliability"],
            "passed": int(
                recovered[3] / SCENARIO_REPLICATES >= 0.90
                and recovered[4] / SCENARIO_REPLICATES >= 0.90
                and stable_rate >= 0.90
            ),
        })
    return summary_rows, bstar_rows


def calibrate_familywise_error() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 500000)
    maxT_rejections = 0
    unadjusted_rejections = 0
    for _ in range(SCENARIO_REPLICATES):
        # Boundary-null effects have mean exactly the 0.05 materiality margin.
        target = 0.05 + rng.normal(0.0, 0.025, size=(frontier.TARGETS, len(frontier.BUDGETS)))
        pvalues = frontier.exact_maxT_pvalues(target)
        maxT_rejections += int(np.any(pvalues <= 0.05))
        # Registered negative control: seven unadjusted one-sided normal approximations.
        means = np.mean(target - 0.05, axis=0)
        se = np.std(target - 0.05, axis=0, ddof=1) / math.sqrt(frontier.TARGETS)
        z = np.divide(means, se, out=np.zeros_like(means), where=se > 0)
        unadjusted_rejections += int(np.any(z >= 1.6448536269514722))
    maxT_fwer = maxT_rejections / SCENARIO_REPLICATES
    naive_fwer = unadjusted_rejections / SCENARIO_REPLICATES
    return [
        {
            "method": "exact_target_signflip_maxT", "null": "materiality_boundary",
            "replicates": SCENARIO_REPLICATES, "familywise_error": maxT_fwer,
            "criterion_max": 0.06, "passed": int(maxT_fwer <= 0.06),
        },
        {
            "method": "unadjusted_per_budget_negative_control", "null": "materiality_boundary",
            "replicates": SCENARIO_REPLICATES, "familywise_error": naive_fwer,
            "criterion_max": "not_applicable_negative_control", "passed": 1,
        },
    ]


def calibrate_monte_carlo_precision() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 600000)
    rows = []
    candidates = (256, 512, 1024, 2048)
    for count in candidates:
        maximum_errors = []
        for _ in range(SCENARIO_REPLICATES):
            # Eight simultaneous bounded integration cells at the worst variance p=0.5.
            estimates = rng.binomial(count, 0.5, size=8) / count
            maximum_errors.append(float(np.max(np.abs(estimates - 0.5))))
        p95 = float(np.quantile(maximum_errors, 0.95))
        hoeffding = math.sqrt(math.log(2 / 0.05) / (2 * count))
        pass_empirical = p95 <= 0.035
        pass_bound = hoeffding <= 0.035
        rows.append({
            "candidate_chains": count,
            "Hoeffding_95_halfwidth": hoeffding,
            "simultaneous_8cell_max_error_p95": p95,
            "bound_criterion": 0.035,
            "empirical_criterion": 0.035,
            "bound_pass": int(pass_bound),
            "empirical_pass": int(pass_empirical),
            "selected": int(count == 2048),
            "selection_valid": int((count == 2048) == (pass_bound and pass_empirical)),
        })
    passing = [row["candidate_chains"] for row in rows if row["bound_pass"] and row["empirical_pass"]]
    if not passing or min(passing) != 2048:
        raise RuntimeError(f"C80 Monte Carlo precision did not select 2048: {passing}")
    return rows


def _target_bootstrap_coverage() -> float:
    rng = np.random.default_rng(SEED + 700000)
    covered = 0
    replicates = 256
    bootstrap_replicates = 2048
    for _ in range(replicates):
        target_values = rng.normal(0.08, 0.04, size=frontier.TARGETS)
        sampled = target_values[
            rng.integers(0, frontier.TARGETS, size=(bootstrap_replicates, frontier.TARGETS))
        ]
        draw_mean = np.mean(sampled, axis=1)
        draw_se = np.std(sampled, axis=1, ddof=1) / math.sqrt(frontier.TARGETS)
        observed_mean = float(np.mean(target_values))
        observed_se = float(np.std(target_values, ddof=1) / math.sqrt(frontier.TARGETS))
        studentized = np.divide(
            draw_mean - observed_mean,
            draw_se,
            out=np.zeros_like(draw_mean),
            where=draw_se > 1e-15,
        )
        q_low, q_high = np.quantile(studentized, [0.025, 0.975])
        low = observed_mean - q_high * observed_se
        high = observed_mean - q_low * observed_se
        covered += int(low <= 0.08 <= high)
    return covered / replicates


def _pseudoreplication_rates() -> tuple[float, float]:
    rng = np.random.default_rng(SEED + 800000)
    correct = 0
    naive = 0
    for _ in range(SCENARIO_REPLICATES):
        target_effect = rng.normal(0.0, 0.04, size=frontier.TARGETS)
        # Correct exact one-budget sign test, embedded in the seven-budget max-T family.
        matrix = np.repeat(target_effect[:, None], len(frontier.BUDGETS), axis=1) + frontier.MATERIAL_REGRET
        correct += int(np.any(frontier.exact_maxT_pvalues(matrix) <= 0.05))
        rows = np.repeat(target_effect, 256) + rng.normal(0.0, 0.005, size=frontier.TARGETS * 256)
        z = float(np.mean(rows) / (np.std(rows, ddof=1) / math.sqrt(len(rows))))
        naive += int(abs(z) >= 1.959963984540054)
    return correct / SCENARIO_REPLICATES, naive / SCENARIO_REPLICATES


def calibrate_dependence() -> list[dict[str, Any]]:
    coverage = _target_bootstrap_coverage()
    correct_fpr, naive_fpr = _pseudoreplication_rates()
    trap_detected = int(correct_fpr <= 0.06 and naive_fpr >= 0.50)
    return [
        {
            "audit": "target_cluster_bootstrap_coverage", "observed": coverage,
            "lower": 0.90, "upper": 0.99, "passed": int(0.90 <= coverage <= 0.99),
        },
        {
            "audit": "correct_target_level_false_positive", "observed": correct_fpr,
            "lower": 0.0, "upper": 0.06, "passed": int(correct_fpr <= 0.06),
        },
        {
            "audit": "naive_row_iid_false_positive_negative_control", "observed": naive_fpr,
            "lower": 0.50, "upper": 1.0, "passed": int(naive_fpr >= 0.50),
        },
        {
            "audit": "pseudoreplication_trap_detected", "observed": trap_detected,
            "lower": 0.95, "upper": 1.0, "passed": trap_detected,
        },
    ]


def run_calibration() -> dict[str, Any]:
    frontier.protocol_audit()
    summary, bstar = calibrate_frontiers()
    fwer = calibrate_familywise_error()
    precision = calibrate_monte_carlo_precision()
    dependence = calibrate_dependence()
    frontier.write_csv(TABLE_DIR / "synthetic_frontier_calibration.csv", summary)
    frontier.write_csv(TABLE_DIR / "synthetic_bstar_recovery.csv", bstar)
    frontier.write_csv(TABLE_DIR / "synthetic_familywise_error.csv", fwer)
    frontier.write_csv(TABLE_DIR / "synthetic_dependence_calibration.csv", dependence)
    frontier.write_csv(TABLE_DIR / "monte_carlo_precision_selection.csv", precision)
    all_pass = (
        all(row["passed"] for row in summary)
        and all(row["passed"] for row in bstar)
        and all(row["passed"] for row in fwer)
        and all(row["passed"] for row in dependence)
        and all(row["selection_valid"] for row in precision)
    )
    result = {
        "schema_version": "c80p_synthetic_calibration_v1",
        "scenario_count": len(SCENARIOS),
        "scenario_replicates": SCENARIO_REPLICATES,
        "selected_MC_chains": 2048,
        "real_data_budget_statistics": 0,
        "same_label_oracle_accessed": False,
        "passed": bool(all_pass),
    }
    if not all_pass:
        raise RuntimeError("C80P synthetic calibration failed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("calibrate",))
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        print(json.dumps(run_calibration(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
