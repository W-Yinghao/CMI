"""C86LP bounded shadow pilot — pre-registered taxonomy + discriminative demo.

Shows the instrument + frozen decision rules classify five known-truth scenarios
correctly across seeds, with target-first aggregation and per-cohort gates.
No real data.
"""
from __future__ import annotations

import pytest

from oaci.active_testing import constants as K
from oaci.active_testing.pilot import (
    COHORTS,
    PolicyMetrics,
    classify,
    run_pilot,
    run_scenario,
)


def _pm(mean, tail, nearopt, top1=0.0, cohorts=None):
    """Build PolicyMetrics; ``cohorts`` maps name->(mean,tail,nearopt), default uniform."""
    if cohorts is None:
        cohorts = {c: (mean, tail, nearopt) for c in COHORTS}
    return PolicyMetrics(
        mean_regret=mean, tail_regret=tail, target_near_opt_prob=nearopt, top1_rate=top1,
        mean_regret_by_cohort={c: v[0] for c, v in cohorts.items()},
        tail_regret_by_cohort={c: v[1] for c, v in cohorts.items()},
        target_near_opt_prob_by_cohort={c: v[2] for c, v in cohorts.items()},
    )


# --- taxonomy is exactly the five pre-registered labels -----------------------
def test_taxonomy_labels():
    assert set(K.BOUNDARY_TAXONOMY) == {
        "BOUNDARY_OPERATIONALLY_CROSSED", "BOUNDARY_WEAKENED_NOT_ROBUST",
        "POLICY_LIMITED", "ACQUISITION_VIEW_NONTRANSPORTABLE", "NO_REGISTERED_ACTIVE_GAIN",
    }


def test_no_gain_is_not_impossibility():
    assert "IMPOSSIBILITY" not in " ".join(K.BOUNDARY_TAXONOMY)
    assert "NO_REGISTERED_ACTIVE_GAIN" in K.BOUNDARY_TAXONOMY


# --- classifier covers every taxonomy outcome (unit-level) --------------------
def test_classifier_acquisition_view_nontransportable():
    # FULL leaves high regret in cohortB only -> still nontransportable
    ceiling = _pm(0.15, 0.15, 0.5, cohorts={"cohortA": (0.0, 0.0, 1.0), "cohortB": (0.3, 0.3, 0.0)})
    out = classify(_pm(0.30, 0.30, 0.0), _pm(0.02, 0.03, 0.9), _pm(0.0, 0.0, 1.0), ceiling)
    assert out == "ACQUISITION_VIEW_NONTRANSPORTABLE"


def test_classifier_no_registered_active_gain():
    ceiling = _pm(0.0, 0.0, 1.0)
    out = classify(_pm(0.20, 0.30, 0.1), _pm(0.20, 0.30, 0.1), _pm(0.20, 0.30, 0.1), ceiling)
    assert out == "NO_REGISTERED_ACTIVE_GAIN"


def test_classifier_policy_limited():
    ceiling = _pm(0.0, 0.0, 1.0)
    out = classify(_pm(0.20, 0.30, 0.1), _pm(0.20, 0.30, 0.1), _pm(0.0, 0.0, 1.0), ceiling)
    assert out == "POLICY_LIMITED"


def test_classifier_crossed():
    ceiling = _pm(0.0, 0.0, 1.0)
    out = classify(_pm(0.20, 0.30, 0.3), _pm(0.02, 0.03, 0.9), _pm(0.0, 0.0, 1.0), ceiling)
    assert out == "BOUNDARY_OPERATIONALLY_CROSSED"


def test_classifier_weakened_mean_only():
    ceiling = _pm(0.0, 0.0, 1.0)
    # mean improves, tail does not
    out = classify(_pm(0.20, 0.30, 0.3), _pm(0.05, 0.30, 0.9), _pm(0.0, 0.0, 1.0), ceiling)
    assert out == "BOUNDARY_WEAKENED_NOT_ROBUST"


def test_classifier_weakened_one_cohort_fails():
    # PM's Simpson case: cohortA fully improves, cohortB's tail does not -> must be WEAKENED,
    # even though pooled tail would pass.
    ceiling = _pm(0.0, 0.0, 1.0)
    passive = _pm(0.20, 0.30, 0.3)
    registered = _pm(0.035, 0.165, 0.9,
                     cohorts={"cohortA": (0.02, 0.03, 0.9), "cohortB": (0.05, 0.30, 0.9)})
    assert classify(passive, registered, _pm(0.0, 0.0, 1.0), ceiling) == "BOUNDARY_WEAKENED_NOT_ROBUST"


# --- end-to-end: five known-truth scenarios classify as intended --------------
@pytest.mark.parametrize("seed", list(range(20)))
def test_pilot_scenarios_classify_as_expected(seed):
    for r in run_pilot(seed=seed):
        assert r.classification == r.expected, (
            f"seed {seed} {r.scenario}: got {r.classification}, expected {r.expected}"
        )


# --- S4 is a genuine TARGET-tail failure: mean improves, a cohort tail does not
def test_s4_is_target_tail_failure():
    r = run_scenario("S4", seed=0)
    assert r.classification == "BOUNDARY_WEAKENED_NOT_ROBUST"
    assert r.registered.mean_regret < r.passive.mean_regret          # mean improves
    tail_fails = any(
        (r.passive.tail_regret_by_cohort[c] - r.registered.tail_regret_by_cohort[c]) < K.TAU_REGRET_MARGIN
        for c in COHORTS
    )
    assert tail_fails                                                 # but target-tail does not


# --- S5 validates the criteria do NOT rely on top-1 ---------------------------
def test_s5_low_top1_still_crossed():
    r = run_scenario("S5", seed=0)
    assert r.classification == "BOUNDARY_OPERATIONALLY_CROSSED"
    assert r.registered.top1_rate < 0.5          # top-1 is low ...
    assert r.registered.target_near_opt_prob >= 0.9     # ... but near-optimal probability is high


# --- metrics are target-based (16 targets, 2 cohorts) -------------------------
def test_metrics_are_target_based():
    r = run_scenario("S1", seed=0)
    assert set(r.passive.mean_regret_by_cohort) == set(COHORTS)
    assert all(len(v) == 8 for v in COHORTS.values())   # 8 target subjects per cohort
