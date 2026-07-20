"""Exact theorem-boundary and prospective verdict tests for C85VP."""
from __future__ import annotations

from fractions import Fraction

import pytest

from oaci.theory.c85_decision_experiments import DecisionContractError
from oaci.theory.c85v_adjudication import adjudicate_theorem
from oaci.theory.c85v_stage_a_derivation import (
    near_optimal_union_bound,
    replay_s10_exact_risks,
    s5_policy_cvar_relation,
    s5_upper_loss_cvar,
    t7_selection_event_inclusion,
    two_state_regret_lower_bound,
)


def test_t2_s10_rational_values_replay_exactly() -> None:
    values = replay_s10_exact_risks()
    assert values == {
        "coarse_registered_risk": Fraction(11, 40),
        "rich_unrestricted_risk": Fraction(0),
        "rich_registered_risk": Fraction(3, 5),
        "registered_reversal": Fraction(13, 40),
    }


def test_t4_factor_of_two_and_tv_boundaries() -> None:
    assert two_state_regret_lower_bound(Fraction(2), Fraction(0)) == 1
    assert two_state_regret_lower_bound(Fraction(2), Fraction(1, 2)) == Fraction(1, 2)
    assert two_state_regret_lower_bound(Fraction(2), Fraction(1)) == 0
    assert two_state_regret_lower_bound(Fraction(0), Fraction(1, 2)) == 0
    with pytest.raises(DecisionContractError):
        two_state_regret_lower_bound(Fraction(1), Fraction(3, 2))


def test_t6_piecewise_cvar_and_open_endpoints() -> None:
    boundary = Fraction(13, 20)
    assert s5_policy_cvar_relation(boundary) == 0
    assert s5_policy_cvar_relation(Fraction(3, 5)) < 0
    assert s5_policy_cvar_relation(Fraction(7, 10)) > 0
    assert s5_upper_loss_cvar(Fraction(9, 10), policy=True) == 1
    with pytest.raises(DecisionContractError):
        s5_upper_loss_cvar(Fraction(0), policy=True)
    with pytest.raises(DecisionContractError):
        s5_upper_loss_cvar(Fraction(1), policy=True)


def test_t7_sigma_zero_empty_set_tie_and_multiple_optimum() -> None:
    assert near_optimal_union_bound([0.0, 0.1], [0.0, 1.0], 0.1) == 0.0
    assert near_optimal_union_bound([0.2], [0.0], 0.1) == 0.0
    assert t7_selection_event_inclusion(
        [1.0, 1.0, 0.5], [1.0, 1.0, 1.0], optimal_index=1, epsilon=0.0
    )
    assert t7_selection_event_inclusion(
        [1.0, 0.5], [0.5, 1.0], optimal_index=0, epsilon=0.1
    )


def _review_inputs(theorem_id: str, *, scope: str, gap: str, sufficient: bool = True):
    stage_a = {
        "theorem_id": theorem_id,
        "statement_sha256": "a" * 64,
        "formal_status_after_stage_A": "OPEN",
        "derivation_scope": scope,
        "unresolved_gaps": [],
    }
    comparison = {
        "theorem_id": theorem_id,
        "statement_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "formal_status_after_stage_B": "OPEN",
        "candidate_gap_label": gap,
    }
    adversarial = {
        "theorem_id": theorem_id,
        "statement_false": False,
        "exact_counterexample_satisfied": theorem_id in {"T2", "T6"},
        "adversarial_checks_pass": True,
        "frozen_statement_sufficient_for_transition": sufficient,
    }
    return stage_a, comparison, adversarial


def test_t5_missing_decoder_conditions_remain_open() -> None:
    inputs = _review_inputs("T5", scope="OPEN_ATTEMPT", gap="INCOMPLETE_OPEN", sufficient=False)
    verdict = adjudicate_theorem(
        stage_a=inputs[0],
        comparison=inputs[1],
        adversarial=inputs[2],
        review_mode="SHADOW_C85VP",
    )
    assert verdict["formal_status"] == "OPEN"


def test_finite_enumeration_cannot_produce_general_proved_status() -> None:
    inputs = _review_inputs("T1", scope="EXACT_FINITE", gap="NONE")
    verdict = adjudicate_theorem(
        stage_a=inputs[0],
        comparison=inputs[1],
        adversarial=inputs[2],
        review_mode="SHADOW_C85VP",
    )
    assert verdict["formal_status"] == "PROVED_FINITE_MODEL_ONLY"
    assert verdict["formal_status"] != "PROVED"


def test_majority_vote_is_never_part_of_a_verdict() -> None:
    inputs = _review_inputs("T2", scope="EXACT_FINITE", gap="NONE")
    verdict = adjudicate_theorem(
        stage_a=inputs[0],
        comparison=inputs[1],
        adversarial=inputs[2],
        review_mode="SHADOW_C85VP",
    )
    assert verdict["formal_status"] == "COUNTEREXAMPLE"
    assert verdict["majority_vote_used"] is False
