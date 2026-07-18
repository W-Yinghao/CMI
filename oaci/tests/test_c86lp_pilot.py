"""C86LP bounded shadow pilot — pre-registered taxonomy + discriminative demo.

Shows the instrument + frozen decision rules classify five known-truth scenarios
correctly, across seeds.  No real data.
"""
from __future__ import annotations

import pytest

from oaci.active_testing import constants as K
from oaci.active_testing.pilot import (
    PolicyMetrics,
    classify,
    run_pilot,
    run_scenario,
)


def _pm(mean, tail, nearopt, top1=0.0, cohort=None):
    return PolicyMetrics(
        mean_regret=mean, tail_regret=tail, near_opt_prob=nearopt, top1_rate=top1,
        mean_regret_by_cohort=cohort or {"cohortA": mean, "cohortB": mean},
    )


# --- taxonomy is exactly the five pre-registered labels -----------------------
def test_taxonomy_labels():
    assert set(K.BOUNDARY_TAXONOMY) == {
        "BOUNDARY_OPERATIONALLY_CROSSED", "BOUNDARY_WEAKENED_NOT_ROBUST",
        "POLICY_LIMITED", "ACQUISITION_VIEW_NONTRANSPORTABLE", "NO_REGISTERED_ACTIVE_GAIN",
    }


# --- classifier covers every taxonomy outcome (unit-level) --------------------
def test_classifier_acquisition_view_nontransportable():
    # even FULL leaves high regret
    ceiling = _pm(0.30, 0.30, 0.0)
    out = classify(_pm(0.30, 0.30, 0.0), _pm(0.05, 0.05, 0.9), _pm(0.05, 0.05, 0.9), ceiling)
    assert out == "ACQUISITION_VIEW_NONTRANSPORTABLE"


def test_classifier_no_registered_active_gain():
    ceiling = _pm(0.0, 0.0, 1.0)
    out = classify(_pm(0.20, 0.30, 0.1), _pm(0.20, 0.30, 0.1), _pm(0.20, 0.30, 0.1), ceiling)
    assert out == "NO_REGISTERED_ACTIVE_GAIN"


def test_classifier_policy_limited():
    ceiling = _pm(0.0, 0.0, 1.0)
    # registered ~ passive, but oracle beats passive => info cheaply exploitable
    out = classify(_pm(0.20, 0.30, 0.1), _pm(0.20, 0.30, 0.1), _pm(0.0, 0.0, 1.0), ceiling)
    assert out == "POLICY_LIMITED"


def test_classifier_crossed_and_weakened():
    ceiling = _pm(0.0, 0.0, 1.0)
    crossed = classify(_pm(0.20, 0.30, 0.3), _pm(0.02, 0.03, 0.9), _pm(0.0, 0.0, 1.0), ceiling)
    assert crossed == "BOUNDARY_OPERATIONALLY_CROSSED"
    # mean improves but tail does not
    weak = classify(_pm(0.20, 0.30, 0.3), _pm(0.05, 0.30, 0.9), _pm(0.0, 0.0, 1.0), ceiling)
    assert weak == "BOUNDARY_WEAKENED_NOT_ROBUST"


# --- NO_REGISTERED_ACTIVE_GAIN is not an impossibility claim -------------------
def test_no_gain_is_not_impossibility():
    # documented invariant: the label is about registered policies vs P0, not about
    # information-theoretic impossibility.
    assert "IMPOSSIBILITY" not in " ".join(K.BOUNDARY_TAXONOMY)
    assert "NO_REGISTERED_ACTIVE_GAIN" in K.BOUNDARY_TAXONOMY


# --- end-to-end: five known-truth scenarios classify as intended --------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 11])
def test_pilot_scenarios_classify_as_expected(seed):
    for r in run_pilot(seed=seed):
        assert r.classification == r.expected, (
            f"seed {seed} {r.scenario}: got {r.classification}, expected {r.expected}"
        )


# --- S5 validates the criteria do NOT rely on top-1 ---------------------------
def test_s5_low_top1_still_crossed():
    r = run_scenario("S5", seed=0)
    assert r.classification == "BOUNDARY_OPERATIONALLY_CROSSED"
    assert r.registered.top1_rate < 0.5          # top-1 is low ...
    assert r.registered.near_opt_prob >= 0.9     # ... but near-optimal probability is high
