"""C86LP shadow validation — Semantics-B isolation, claim-boundary, query server.

Everything runs on synthetic shadow objects.  No real C84 label, prediction, Q0
shard, or C85U utility is opened.
"""
from __future__ import annotations

import numpy as np
import pytest

from oaci import active_testing as at
from oaci.active_testing import constants as K
from oaci.active_testing.contribution import (
    C86LPClaimError,
    assert_linear_claim,
    compute_contribution,
    pairwise_nll_differences,
    unbiasedness_claim,
)
from oaci.active_testing.field import (
    C86LPFieldError,
    DevelopmentField,
    build_manifest,
)
from oaci.active_testing.query_server import (
    C86LPInputUnavailable,
    C86LPQueryError,
    QueryServer,
)


def _t0_trial(field):
    return next(t for t in field.construction_trial_ids if field._target_of[t] == "T0")


# --- 1. field arithmetic is internally consistent -----------------------------
def test_field_arithmetic_consistent():
    assert K.CONSTRUCTION_ROWS + K.HELD_EVAL_ROWS == K.TOTAL_TARGET_TRIALS
    assert sum(K.DATASET_CONSTRUCTION_ROWS.values()) == K.CONSTRUCTION_ROWS
    assert K.CONSTRUCTION_ROWS * K.CONTEXTS_PER_CONSTRUCTION_TRIAL == K.CONTEXT_TRIAL_ROWS
    assert K.CONTEXT_TRIAL_ROWS * K.CANDIDATES_PER_CONTEXT == K.CANDIDATE_TRIAL_CONTEXT_ROWS
    assert K.CANDIDATE_TRIAL_CONTEXT_ROWS * 2 == K.BINARY_PROBABILITY_SCALARS
    assert K.TARGET_SUBJECTS * len(K.BUDGET_GRID) == K.TARGET_BUDGET_CELLS
    assert K.AVAILABLE_CELLS + K.UNAVAILABLE_CELLS == K.TARGET_BUDGET_CELLS
    assert K.Q0_FINITE_ACTION_RECORDS + K.Q0_DETERMINISTIC_FULL == K.Q0_TOTAL


# --- 2. construction / evaluation nonoverlap ----------------------------------
def test_construction_evaluation_nonoverlap():
    field, _ = at.build_shadow_field()
    assert field.construction_trial_ids.isdisjoint(field.evaluation_trial_ids)


def test_overlap_rejected_at_construction():
    with pytest.raises(C86LPFieldError):
        DevelopmentField(
            declared_contexts=1,
            construction_trial_ids=frozenset({"x", "y"}),
            evaluation_trial_ids=frozenset({"y", "z"}),
        )


# --- 3. unlabeled pool carries no forbidden fields ----------------------------
def test_unlabeled_pool_has_no_label_fields():
    field, _ = at.build_shadow_field()
    forbidden = {"true_label", "label", "queried_response", "construction_metric",
                 "selected_action", "c85u_utility", "held_utility", "outcome"}
    for row in field.pool:
        assert set(row.__dataclass_fields__).isdisjoint(forbidden)


# --- 4. linear contribution exactness -----------------------------------------
def test_linear_nll_sum_matches_recompute():
    probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]])
    row = compute_contribution("t", true_label=1, probs=probs)
    expected = -np.log(np.clip(probs[:, 1], K.PROB_FLOOR, 1.0))
    assert np.allclose(row.nll, expected)


def test_pairwise_differences_derived_not_stored():
    probs = np.random.default_rng(1).uniform(0.1, 0.9, size=(81, 2))
    probs = probs / probs.sum(axis=1, keepdims=True)
    row = compute_contribution("t", 0, probs)
    diffs = pairwise_nll_differences(row.nll)
    assert diffs.shape[0] == 81 * 80 // 2 == 3240
    assert not hasattr(row, "pairwise_nll_difference")


# --- 5. nonlinear plugin claim guard ------------------------------------------
@pytest.mark.parametrize("q", sorted(K.NONLINEAR_PLUGINS))
def test_nonlinear_plugins_have_no_unbiasedness_claim(q):
    assert unbiasedness_claim(q) is False
    with pytest.raises(C86LPClaimError):
        assert_linear_claim(q)


@pytest.mark.parametrize("q", sorted(K.LINEAR_MOMENTS))
def test_linear_moments_have_unbiasedness_claim(q):
    assert unbiasedness_claim(q) is True
    assert_linear_claim(q)


# --- 6. Semantics B: one physical label informs all its contexts --------------
def test_one_physical_label_informs_all_contexts():
    field, avail = at.build_shadow_field(n_contexts_per_target=8)
    server = QueryServer(field, avail)
    server.open_attempt("a0", "T0", 4)
    trial = _t0_trial(field)
    resp = server.query("a0", trial)
    assert resp.trial_id == trial
    assert resp.true_label == field._labels[trial]
    # one query returns exactly this trial's contexts (all 8), each with its own row
    assert set(resp.contributions) == set(field.contexts_of(trial))
    assert len(resp.contributions) == 8
    for ctx, row in resp.contributions.items():
        assert np.allclose(row.nll, field._contrib[trial][ctx].nll)


def test_budget_counts_physical_labels_not_context_rows():
    field, avail = at.build_shadow_field(n_physical_trials=6)
    server = QueryServer(field, avail)
    server.open_attempt("a0", "T0", 4)
    t0_trials = [t for t in field.construction_trial_ids if field._target_of[t] == "T0"]
    for t in t0_trials[:4]:
        server.query("a0", t)          # each query = 8 context rows but 1 physical label
    assert server.remaining("a0") == 0  # budget spent = 4 physical labels, not 32 rows


# --- 7. duplicate / unknown / exhausted / cross-target rejection --------------
def test_duplicate_query_rejected():
    field, avail = at.build_shadow_field()
    server = QueryServer(field, avail)
    server.open_attempt("a0", "T0", 4)
    trial = _t0_trial(field)
    server.query("a0", trial)
    with pytest.raises(C86LPQueryError):
        server.query("a0", trial)


def test_unknown_trial_rejected():
    field, avail = at.build_shadow_field()
    server = QueryServer(field, avail)
    server.open_attempt("a0", "T0", 4)
    with pytest.raises(C86LPQueryError):
        server.query("a0", "nope-not-a-trial")


def test_cross_target_trial_rejected():
    field, avail = at.build_shadow_field()
    server = QueryServer(field, avail)
    server.open_attempt("a0", "T0", 4)
    other = next(t for t in field.construction_trial_ids if field._target_of[t] == "T1")
    with pytest.raises(C86LPQueryError):
        server.query("a0", other)


def test_budget_exhaustion_rejected():
    field, avail = at.build_shadow_field(n_physical_trials=6)
    server = QueryServer(field, avail)
    server.open_attempt("a0", "T0", 4)
    trials = [t for t in field.construction_trial_ids if field._target_of[t] == "T0"][:5]
    for t in trials[:4]:
        server.query("a0", t)
    with pytest.raises(C86LPQueryError):
        server.query("a0", trials[4])


# --- 8. no bulk oracle escapes ------------------------------------------------
def test_no_bulk_oracle_on_server():
    field, avail = at.build_shadow_field()
    server = QueryServer(field, avail)
    for banned in ("all_labels", "dump", "labels", "oracle", "contributions"):
        assert not hasattr(server, banned)
    public = [a for a in dir(server) if not a.startswith("_")]
    assert set(public) <= {"open_attempt", "query", "remaining", "receipts"}


# --- 9. held C85U outcome isolation -------------------------------------------
def test_held_outcome_identity_bound_never_opened():
    field, _ = at.build_shadow_field()
    manifest = build_manifest(field)
    assert manifest.c85u_identity == K.FROZEN_INPUT_SHA["c85u_acceptance_manifest"]
    assert manifest.c85u_utility_values_accessed == 0
    for name in dir(at):
        assert "utility" not in name.lower()


# --- 10. isolation is logical/API only, not physical --------------------------
def test_isolation_is_logical_not_physical():
    field, _ = at.build_shadow_field()
    assert build_manifest(field).isolation_level == "logical_api_mock"
    assert K.ISOLATION_LEVEL == "logical_api_mock"


# --- 11. first-index argmax tie rule ------------------------------------------
def test_first_index_tie_rule():
    probs = np.array([[0.5, 0.5]])
    row = compute_contribution("t", 0, probs)
    assert int(row.hard_pred[0]) == 0
    assert int(row.correct[0]) == 1


# --- 12. partial field cannot publish -----------------------------------------
def test_partial_field_cannot_publish():
    field, _ = at.build_shadow_field()
    field.declared_contexts = field.observed_contexts + 1
    manifest = build_manifest(field)
    assert manifest.coverage_complete is False
    with pytest.raises(C86LPFieldError):
        manifest.publish()


# --- 13. corrected gate (not the contradictory locked-ready one) --------------
def test_complete_field_publishes_instrument_gate():
    field, _ = at.build_shadow_field()
    gate = build_manifest(field).publish()
    assert gate == K.GATE_INSTRUMENT
    assert gate == "C86LP_SHADOW_QUERY_INSTRUMENT_IMPLEMENTED_PROBE_CRITERIA_REVIEW_REQUIRED"
    assert "LOCKED_READY_FOR_PI_AUTHORIZATION" not in gate


# --- 14. unsupported budget -> INPUT_UNAVAILABLE, no substitution --------------
def test_unsupported_budget_input_unavailable():
    field, avail = at.build_shadow_field()
    avail[("T0", 32)] = False
    server = QueryServer(field, avail)
    with pytest.raises(C86LPInputUnavailable):
        server.open_attempt("a0", "T0", 32)


# --- 15. frozen scientific gates immutable ------------------------------------
def test_frozen_gates_unchanged():
    assert K.FROZEN_GATES["C84_primary"].startswith("C84-D_")
    assert K.FROZEN_GATES["C84_label_frontier"] == "C84-L4"
    assert (K.FROZEN_GATES["T1"], K.FROZEN_GATES["T5"]) == ("PROVED", "OPEN")
    field, _ = at.build_shadow_field()
    m = build_manifest(field)
    tampered = type(m)(**{**m.__dict__, "frozen_gates": {**m.frozen_gates, "T5": "PROVED"}})
    with pytest.raises(C86LPFieldError):
        tampered.publish()


# --- self-check ---------------------------------------------------------------
def test_validate_smoke():
    out = at.validate()
    assert out["gate"] == K.GATE_INSTRUMENT
    assert out["contexts_per_trial"] == 8
    assert out["isolation_level"] == "logical_api_mock"
    assert out["coverage_complete"] is True
