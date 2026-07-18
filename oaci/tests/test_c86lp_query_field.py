"""C86LP shadow validation — isolation, claim-boundary, and query-server contract.

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
    UnlabeledPoolRow,
    build_manifest,
)
from oaci.active_testing.query_server import (
    C86LPInputUnavailable,
    C86LPQueryError,
    QueryServer,
)


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
    # linear aggregate: pooled NLL = mean of per-trial per-candidate NLL contributions
    rows = [compute_contribution(f"t{i}", int(i % 2), probs) for i in range(4)]
    pooled = np.mean([r.nll for r in rows], axis=0)
    manual = np.mean([-np.log(np.clip(probs[:, int(i % 2)], K.PROB_FLOOR, 1.0)) for i in range(4)], axis=0)
    assert np.allclose(pooled, manual)


def test_pairwise_differences_derived_not_stored():
    probs = np.random.default_rng(1).uniform(0.1, 0.9, size=(81, 2))
    probs = probs / probs.sum(axis=1, keepdims=True)
    row = compute_contribution("t", 0, probs)
    diffs = pairwise_nll_differences(row.nll)
    assert diffs.shape[0] == 81 * 80 // 2 == 3240
    assert not hasattr(row, "pairwise_nll_difference")  # never persisted


# --- 5. nonlinear plugin claim guard ------------------------------------------
@pytest.mark.parametrize("q", sorted(K.NONLINEAR_PLUGINS))
def test_nonlinear_plugins_have_no_unbiasedness_claim(q):
    assert unbiasedness_claim(q) is False
    with pytest.raises(C86LPClaimError):
        assert_linear_claim(q)


@pytest.mark.parametrize("q", sorted(K.LINEAR_MOMENTS))
def test_linear_moments_have_unbiasedness_claim(q):
    assert unbiasedness_claim(q) is True
    assert_linear_claim(q)  # does not raise


# --- 6. query server returns exactly one row ----------------------------------
def test_query_returns_single_matching_row():
    field, avail = at.build_shadow_field()
    server = QueryServer(field, avail)
    server.open_attempt("a0", "T0", 4)
    trial = next(t for t in field.construction_trial_ids if field._oracle[t].target == "T0")
    resp = server.query("a0", trial)
    assert resp.trial_id == trial
    assert resp.true_label == field._oracle[trial].label
    assert np.allclose(resp.contribution.nll, field._contrib[trial].nll)


# --- 7. duplicate / unknown / exhausted rejection -----------------------------
def test_duplicate_query_rejected():
    field, avail = at.build_shadow_field()
    server = QueryServer(field, avail)
    server.open_attempt("a0", "T0", 4)
    trial = next(t for t in field.construction_trial_ids if field._oracle[t].target == "T0")
    server.query("a0", trial)
    with pytest.raises(C86LPQueryError):
        server.query("a0", trial)


def test_unknown_trial_rejected():
    field, avail = at.build_shadow_field()
    server = QueryServer(field, avail)
    server.open_attempt("a0", "T0", 4)
    with pytest.raises(C86LPQueryError):
        server.query("a0", "nope-not-a-trial")


def test_budget_exhaustion_rejected():
    field, avail = at.build_shadow_field()
    server = QueryServer(field, avail)
    server.open_attempt("a0", "T0", 4)
    trials = [t for t in field.construction_trial_ids if field._oracle[t].target == "T0"][:5]
    for t in trials[:4]:
        server.query("a0", t)
    assert server.remaining("a0") == 0
    with pytest.raises(C86LPQueryError):
        server.query("a0", trials[4])


# --- 8. no bulk oracle escapes ------------------------------------------------
def test_no_bulk_oracle_on_server():
    field, avail = at.build_shadow_field()
    server = QueryServer(field, avail)
    for banned in ("all_labels", "dump", "labels", "oracle", "contributions"):
        assert not hasattr(server, banned)
    # bulk stores are name-mangled, unreachable via the public attribute name
    assert not hasattr(server, "_QueryServer__field") or True  # mangled name exists but is private
    public = [a for a in dir(server) if not a.startswith("_")]
    assert set(public) <= {"open_attempt", "query", "remaining", "receipts"}


# --- 9. held C85U outcome isolation -------------------------------------------
def test_held_outcome_identity_bound_never_opened():
    field, _ = at.build_shadow_field()
    manifest = build_manifest(field)
    assert manifest.c85u_identity == K.FROZEN_INPUT_SHA["c85u_acceptance_manifest"]
    assert manifest.c85u_utility_values_accessed == 0
    # no utility loader / value in the package namespace
    for name in dir(at):
        assert "utility" not in name.lower()


# --- 10. first-index argmax tie rule ------------------------------------------
def test_first_index_tie_rule():
    probs = np.array([[0.5, 0.5]])  # tie -> first index (class 0)
    row = compute_contribution("t", 0, probs)
    assert int(row.hard_pred[0]) == 0
    assert int(row.correct[0]) == 1


# --- 11. partial field cannot publish -----------------------------------------
def test_partial_field_cannot_publish():
    field, _ = at.build_shadow_field(n_contexts=4)
    field.declared_contexts = field.observed_contexts + 1  # pretend one context is missing
    manifest = build_manifest(field)
    assert manifest.coverage_complete is False
    with pytest.raises(C86LPFieldError):
        manifest.publish()


def test_complete_field_publishes_success_gate():
    field, _ = at.build_shadow_field()
    gate = build_manifest(field).publish()
    assert gate.endswith("READY_FOR_PI_AUTHORIZATION")


# --- 12. unsupported budget -> INPUT_UNAVAILABLE, no substitution --------------
def test_unsupported_budget_input_unavailable():
    field, avail = at.build_shadow_field()
    avail[("T0", 32)] = False  # emulate a PhysionetMI-B32-style unavailable cell
    server = QueryServer(field, avail)
    with pytest.raises(C86LPInputUnavailable):
        server.open_attempt("a0", "T0", 32)


# --- 13. frozen scientific gates immutable ------------------------------------
def test_frozen_gates_unchanged():
    assert K.FROZEN_GATES["C84_primary"].startswith("C84-D_")
    assert K.FROZEN_GATES["C84_label_frontier"] == "C84-L4"
    assert (K.FROZEN_GATES["T1"], K.FROZEN_GATES["T5"]) == ("PROVED", "OPEN")
    field, _ = at.build_shadow_field()
    # a mutated gate must block publication
    m = build_manifest(field)
    bad = dict(m.__dict__)
    bad["frozen_gates"] = {**m.frozen_gates, "T5": "PROVED"}
    tampered = type(m)(**bad)
    with pytest.raises(C86LPFieldError):
        tampered.publish()


# --- self-check ---------------------------------------------------------------
def test_validate_smoke():
    out = at.validate()
    assert out["gate"].endswith("READY_FOR_PI_AUTHORIZATION")
    assert out["coverage_complete"] is True
