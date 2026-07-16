"""Shadow-only execution tests for the future C85T production path."""
from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
import pytest

from oaci.theory.c85_decision_experiments import DecisionContractError
from oaci.theory import c85t_exact_scenarios as exact
from oaci.theory import c85t_monte_carlo as monte
from oaci.theory import c85t_proofs as proofs
from oaci.theory import c85t_result_manifest as manifests
from oaci.theory import c85t_rng as rng


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "oaci" / "reports" / "c85tl_tables"


def _rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_registered_rng_streams_require_future_authorization_token() -> None:
    with pytest.raises(DecisionContractError, match="consumed C85T authorization"):
        rng.deterministic_seed("S6", 0)
    with pytest.raises(DecisionContractError, match="unknown"):
        rng.deterministic_seed("NOT_A_FIXTURE", 0)


def test_shadow_seed_low64_replay_is_exact() -> None:
    expected = {
        "SHADOW_NORMAL_A": (1648136560762879262, 14534546671537538876, 5222896606790536404),
        "SHADOW_RADEMACHER_A": (18149165777991548432, 4118874583934873419, 11391186582201792462),
        "SHADOW_RADEMACHER_B": (16669812317105199333, 860564168640180387, 16622526019181634429),
    }
    for fixture_id, values in expected.items():
        assert tuple(rng.deterministic_seed(fixture_id, replicate) for replicate in (0, 1, 4095)) == values


def test_shadow_standard_normal_raw_draw_replays() -> None:
    observed = rng.draw_standard_normal("SHADOW_NORMAL_A", 0, 4)
    expected = np.array(
        [1.0791002582110019, -0.40597079825611726, -0.1495004663262244, 0.9705309751203361],
        dtype="<f8",
    )
    assert observed.dtype == expected.dtype
    assert observed.tobytes() == expected.tobytes()
    assert rng.canonical_array_sha256(observed) == "3518a403117697889b370805d8419fc31b64c47cea50ee32218c91b838885464"


def test_shadow_rademacher_draw_order_and_prefixes_replay() -> None:
    low, high = rng.draw_s9_rademacher_prefixes("SHADOW_RADEMACHER_A", 0)
    assert low.shape == (51,) and high.shape == (46,)
    assert low[:8].tolist() == [-1, -1, 1, 1, -1, 1, 1, -1]
    assert high[:8].tolist() == [1, -1, -1, -1, -1, -1, -1, -1]
    assert hashlib.sha256(low.tobytes()).hexdigest() == "021429ea5d88444c76b1bc2649e1a54edbb3fe235694d0b5a22c981231181de9"
    assert hashlib.sha256(high.tobytes()).hexdigest() == "2b18311ab183498742d4c15ea5e63f51a8d8d47a1bbed2d8c3789f5bfb523694"


def test_shadow_draw_registry_contains_no_registered_scenario() -> None:
    rows = _rows("deterministic_seed_and_raw_draw_replay.csv")
    assert {row["fixture_id"] for row in rows} == set(rng.SHADOW_SCENARIOS)
    assert all(row["registered_scenario"] == row["scientific_result"] == "0" for row in rows)


def test_shadow_exact_cvar_and_rational_lp() -> None:
    assert exact.weighted_upper_cvar(
        [Fraction(1, 5), Fraction(4, 5)],
        [Fraction(3, 4), Fraction(1, 4)],
        Fraction(1, 2),
    ) == Fraction(1, 2)
    result = exact.exact_minimax_regret_lp(
        [[1, Fraction(2, 5), Fraction(1, 5)],
         [Fraction(1, 5), 1, Fraction(1, 2)],
         [Fraction(1, 2), Fraction(1, 10), 1]]
    )
    assert result["optimal_randomized_action_distribution"] == (
        Fraction(57, 160), Fraction(49, 160), Fraction(27, 80)
    )
    assert result["minimax_regret"] == Fraction(363, 800)
    assert result["randomization_gain"] == Fraction(277, 800)
    assert all(slack >= 0 for slack in result["extreme_point_constraint_slacks"])


@pytest.fixture(scope="module")
def shadow_near_result() -> dict[str, object]:
    return monte.simulate_near_optimal_selection(
        scenario_id="SHADOW_NORMAL_A",
        utilities=[1.0, 0.98, 0.7],
        epsilon=0.03,
        tau=0.1,
        pairwise_sigma=0.05,
        replicates=4096,
    )


def test_shadow_near_optimal_mc_replays_exactly(shadow_near_result: dict[str, object]) -> None:
    assert shadow_near_result["selected_action_counts"] == {"0": 2704, "1": 1392, "2": 0}
    assert shadow_near_result["top_1_probability"] == 0.66015625
    assert shadow_near_result["outside_A_epsilon_probability"] == 0.0
    assert shadow_near_result["mean_regret"] == pytest.approx(0.006796875000000006, abs=0.0)
    assert shadow_near_result["raw_output_sha256"] == "abb6791e8a5f4c8b101a42f459ac1f51a318e6f183bdcf235b321576ff401756"


def test_shadow_near_geometry_keeps_primary_and_looser_bounds_distinct(
    shadow_near_result: dict[str, object],
) -> None:
    geometry = shadow_near_result["geometry"]
    assert geometry["epsilon_near_optimal_set"] == (0, 1)
    assert geometry["t7_primary_union_bound"] < geometry["historical_looser_diagnostic"]
    assert geometry["near_optimal_count"] == 2


@pytest.fixture(scope="module")
def shadow_s9_result() -> dict[str, object]:
    return monte.simulate_full_information_designs(
        scenario_id="SHADOW_RADEMACHER_A",
        replicates=4096,
        stratum_masses=(0.7, 0.3),
        sigmas=(0.03, 0.15),
        passive_allocation=(51, 13),
        neyman_allocation=(18, 46),
        population_mean_losses=(0.2, 0.24, 0.6, 0.8),
        action1_offset=0.04,
    )


def test_shadow_s9_estimators_select_over_all_four_actions(shadow_s9_result: dict[str, object]) -> None:
    passive = shadow_s9_result["designs"]["passive"]
    neyman = shadow_s9_result["designs"]["neyman"]
    assert passive["selected_action_counts"] == {"0": 4094, "1": 2, "2": 0, "3": 0}
    assert neyman["selected_action_counts"] == {"0": 4096, "1": 0, "2": 0, "3": 0}
    assert passive["top_2_coverage"] == neyman["top_2_coverage"] == 1.0
    assert passive["raw_output_sha256"] == "519d90d9a412cfc92270e8657689be5294d54dab86568c2ae11a7a1832d9c0fd"
    assert neyman["raw_output_sha256"] == "ee238a554e38ea346ba46a7caaec797ae643c5fb317ebaebf7f27e256e30fafa"


def test_shadow_s9_analytic_variance_is_authoritative(shadow_s9_result: dict[str, object]) -> None:
    analytic = shadow_s9_result["analytic_variance"]
    assert analytic["neyman_d_hat_variance"] < analytic["passive_d_hat_variance"]
    for design in ("passive", "neyman"):
        observed = shadow_s9_result["designs"][design]["d_hat_sample_variance"]
        assert observed == pytest.approx(analytic[f"{design}_d_hat_variance"], rel=0.04)
    assert shadow_s9_result["universal_active_superiority_claim"] is False


def test_proved_transition_fails_without_independent_pass() -> None:
    candidate = proofs.ProofCandidate(
        theorem_id="T1",
        exact_statement="SHADOW statement",
        assumptions=("shadow assumption",),
        proof_or_counterexample="shadow argument",
        boundary_cases=("shadow boundary",),
        proposed_status=proofs.TheoremStatus.PROVED,
    )
    audit = proofs.ProofAudit("T1", "FAIL", ("shadow failure",))
    with pytest.raises(DecisionContractError, match="independent PASS"):
        proofs.apply_status_transition(candidate, audit)


def test_t5_shadow_schema_can_remain_open_without_transition() -> None:
    candidate = proofs.ProofCandidate(
        theorem_id="T5",
        exact_statement="SHADOW open statement",
        assumptions=("shadow finite model",),
        proof_or_counterexample="incomplete shadow attempt retained",
        boundary_cases=("shadow boundary",),
        proposed_status=proofs.TheoremStatus.OPEN,
    )
    audit = proofs.ProofAudit("T5", "FAIL", ("proof intentionally incomplete",))
    text = proofs.render_proof_markdown(candidate, audit)
    proofs.validate_rendered_proof(text, "T5")
    assert text.endswith("`OPEN`\n")


def _shadow_result() -> dict[str, object]:
    return {
        "schema_version": manifests.RESULT_SCHEMA,
        "final_gate": manifests.SUCCESS_GATE,
        "fixture": "SHADOW_ATOMIC_PUBLICATION",
        "registered_scenarios_executed": 0,
    }


def test_shadow_atomic_publication_success_replays(tmp_path: Path) -> None:
    root = tmp_path / "result"
    with manifests.AtomicResultWriter(root) as writer:
        writer.write_json("shadow.json", {"fixture": True})
        writer.publish(_shadow_result())
    manifest = manifests.replay_manifest(root)
    assert root.is_dir()
    assert manifest["artifact_count"] == 2


@pytest.mark.parametrize("point", ["before_result", "before_manifest", "before_publish"])
def test_shadow_atomic_failure_never_publishes_final_root(tmp_path: Path, point: str) -> None:
    root = tmp_path / point
    writer: manifests.AtomicResultWriter
    with pytest.raises(RuntimeError, match="C85T_SHADOW_FAILURE"):
        with manifests.AtomicResultWriter(root, failure_injection=point) as writer:
            writer.write_json("shadow_partial.json", {"fixture": True})
            writer.publish(_shadow_result())
    assert not root.exists()
    assert writer.failed_root is not None and writer.failed_root.is_dir()

