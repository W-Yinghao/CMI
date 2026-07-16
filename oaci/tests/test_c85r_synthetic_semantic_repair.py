"""Exact semantic-satisfiability tests for the additive C85R V2 contract."""
from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oaci.theory import c85r_synthetic_semantic_repair as repair


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci" / "reports"
TABLES = REPORTS / "c85r_tables"


def _csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_historical_s10_equality_is_detected_exactly() -> None:
    audit = repair.exact_s10_audit()
    assert audit["coarse_policy"] == (1, 1)
    assert audit["coarse_risk"] == Fraction(11, 40)
    assert audit["historical_rich_risk"] == Fraction(11, 40)
    assert audit["historical_rich_risk"] == audit["coarse_risk"]


def test_v2_s10_exact_risks_and_reversal() -> None:
    audit = repair.exact_s10_audit()
    assert audit["rich_unrestricted_risk"] == 0
    assert audit["v2_rich_risk"] == Fraction(3, 5)
    assert audit["rich_gap"] == Fraction(3, 5)
    assert audit["reversal"] == Fraction(13, 40)
    assert audit["rich_unrestricted_risk"] < audit["coarse_risk"] < audit["v2_rich_risk"]


def test_s10_v2_changes_only_rich_registered_policy_plus_audit_metadata() -> None:
    locked = repair.validate_locked_contracts()
    old = {row["id"]: row for row in locked["historical"]["scenarios"]}["S10"]
    new = {row["id"]: row for row in locked["v2"]["scenarios"]}["S10"]
    reduced = dict(new)
    reduced.pop("historical_registered_policy")
    reduced.pop("expected_exact_risks")
    reduced.pop("registered_reversal_attribution")
    reduced["registered_policies"] = dict(reduced["registered_policies"])
    reduced["registered_policies"]["rich"] = "always_action_1"
    assert reduced == old
    assert new["registered_policies"]["rich"] == "always_action_0"


def test_s9_all_loss_vectors_stay_in_unit_interval() -> None:
    audit = repair.exact_s9_audit()
    assert len(audit["support_rows"]) == 4
    assert all(
        Fraction(0) <= loss <= Fraction(1)
        for row in audit["support_rows"]
        for loss in row["losses"]
    )


def test_s9_population_means_and_unique_best_action_are_exact() -> None:
    audit = repair.exact_s9_audit()
    assert audit["population_means"] == (
        Fraction(3, 10), Fraction(7, 20), Fraction(13, 20), Fraction(17, 20)
    )
    assert min(range(4), key=audit["population_means"].__getitem__) == 0


def test_s9_pairwise_means_and_standard_deviations_are_exact() -> None:
    audit = repair.exact_s9_audit()
    for stratum, sigma in zip(("L", "H"), audit["sigmas"]):
        differences = [
            row["difference_1_minus_0"]
            for row in audit["support_rows"]
            if row["stratum"] == stratum
        ]
        mean = sum(differences, Fraction(0)) / 2
        variance = sum(((value - mean) ** 2 for value in differences), Fraction(0)) / 2
        assert mean == Fraction(1, 20)
        assert variance == sigma**2


def test_s9_largest_remainder_allocations_are_exact() -> None:
    audit = repair.exact_s9_audit()
    assert audit["passive_allocation"] == (51, 13)
    assert audit["neyman_allocation"] == (18, 46)
    assert repair.largest_remainder_allocation((Fraction(1, 2), Fraction(1, 2)), 3) == (2, 1)


def test_s9_fixed_neyman_variance_is_below_passive_without_universal_claim() -> None:
    audit = repair.exact_s9_audit()
    assert audit["passive_variance"] == Fraction(1327, 10359375)
    assert audit["neyman_variance"] == Fraction(317, 6468750)
    assert audit["neyman_variance"] < audit["passive_variance"]
    rows = _csv("S9_analytic_variance_contract.csv")
    assert all(row["universal_superiority_claim"] == "0" for row in rows)


@pytest.mark.parametrize("scenario_id", ["S6", "S7"])
def test_action_error_variance_gives_registered_pairwise_sigma(scenario_id: str) -> None:
    row = repair.exact_s6_s7_noise_audit()[scenario_id]
    assert row["action_error_variance"] == Fraction(1, 5000)
    assert row["pairwise_variance"] == Fraction(1, 2500)
    assert row["pairwise_variance"] == row["pairwise_sigma"] ** 2
    assert row["replicates"] == 4096


@pytest.mark.parametrize("scenario_id", ["S6", "S7"])
def test_pairwise_errors_are_correlated_through_shared_optimal_error(scenario_id: str) -> None:
    row = repair.exact_s6_s7_noise_audit()[scenario_id]
    assert row["shared_star_covariance"] == Fraction(1, 5000)
    assert row["pairwise_correlation"] == Fraction(1, 2)
    contract_row = next(
        value for value in _csv("S6_S7_noise_coupling_contract.csv")
        if value["scenario"] == scenario_id
    )
    assert contract_row["pairwise_errors_independent"] == "0"


def test_s6_s7_output_contract_is_locked_but_not_computed() -> None:
    rows = _csv("S6_S7_output_contract.csv")
    assert len(rows) == 16
    assert {row["scenario"] for row in rows} == {"S6", "S7"}
    assert all(row["required_in_C85T"] == "1" for row in rows)
    assert all(row["computed_in_C85R"] == "0" for row in rows)


def test_t7_primary_bound_uses_delta_not_delta_minus_epsilon() -> None:
    rows = _csv("T7_bound_supersession.csv")
    primary = next(row for row in rows if row["primary_target"] == "1")
    diagnostic = next(row for row in rows if row["primary_target"] == "0")
    assert "Delta_i^2" in primary["expression"]
    assert "Delta_i-epsilon" not in primary["expression"]
    assert "Delta_i-epsilon" in diagnostic["expression"]
    assert primary["independence_required"] == "0"
    assert primary["theorem_status"] == "OPEN"


def test_t3_kernel_equality_precision_is_present() -> None:
    row = next(row for row in _csv("proof_obligation_precision_addendum.csv") if row["theorem_id"] == "T3")
    assert "action-kernel equality" in row["required_condition"]
    assert row["excluded_shortcut"] == "one coupled action draw"
    assert row["status"] == "OPEN"


def test_t4_unique_or_disjoint_optimum_condition_is_present() -> None:
    row = next(row for row in _csv("proof_obligation_precision_addendum.csv") if row["theorem_id"] == "T4")
    assert "unique different optima or disjoint optimum sets" in row["required_condition"]
    assert "decoder" in row["required_condition"]
    assert row["status"] == "OPEN"


def test_t6_excludes_cvar_alpha_endpoints() -> None:
    row = next(row for row in _csv("proof_obligation_precision_addendum.csv") if row["theorem_id"] == "T6")
    assert "inside (0,1)" in row["required_condition"]
    assert row["excluded_shortcut"] == "alpha endpoints 0 or 1"


def test_semantic_preflight_reports_no_scientific_execution() -> None:
    result = repair.semantic_preflight()
    assert result["status"] == repair.SEMANTIC_STATUS
    assert result["scenario_count"] == 11
    assert result["scientific_simulations"] == 0
    assert result["proofs_completed"] == 0
    assert set(result["theorem_statuses"].values()) == {"OPEN"}


def test_semantic_preflight_cli_is_exact_and_non_scientific() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "oaci.theory.c85r_synthetic_semantic_repair", "semantic-preflight"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    value = json.loads(result.stdout)
    assert value["status"] == repair.SEMANTIC_STATUS
    assert value["S10"] == {
        "historical_equal_risk": "11/40",
        "reversal": "13/40",
        "v2_rich_risk": "3/5",
    }
    assert value["scientific_simulations"] == 0


def test_v2_sidecar_and_materialized_tables_replay() -> None:
    assert hashlib.sha256(repair.V2_CONTRACT_PATH.read_bytes()).hexdigest() == repair.EXPECTED_V2_CONTRACT_SHA256
    observed = repair.validate_materialized_tables()
    assert len(observed) == 15
    assert observed["historical_contract_supersession.csv"] == 4
    assert observed["semantic_satisfiability_validation.csv"] == 16
    assert observed["theorem_status_replay.csv"] == 7

