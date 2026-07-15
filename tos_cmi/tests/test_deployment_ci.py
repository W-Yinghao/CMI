"""CMI-Trace P0.4 tests — deployment CI three-state semantics (delta bAcc @ +0.01 threshold)."""
import numpy as np
import pytest

from tos_cmi.eeg.deployment_ci import (deployment_ci_state, deployable_benefit,
                                       practical_gain_ruled_out_everywhere,
                                       CONFIRMED, RULED_OUT, INCONCLUSIVE, PRACTICAL_THRESHOLD)


def test_confirmed_when_lower_bound_above_threshold():
    # lower CI 0.02 > +0.01 -> confirmed practical benefit
    assert deployment_ci_state(0.02, 0.06) == CONFIRMED


def test_ruled_out_uses_upper_bound_not_lower():
    # upper CI 0.005 < +0.01 -> gain ruled out. Critically this is an UPPER-bound decision.
    assert deployment_ci_state(-0.05, 0.005) == RULED_OUT
    # a LOW lower bound with a HIGH upper bound must NOT be 'ruled out' (the old bug)
    assert deployment_ci_state(-0.02, 0.08) == INCONCLUSIVE


def test_inconclusive_when_ci_straddles_threshold():
    assert deployment_ci_state(0.0, 0.05) == INCONCLUSIVE      # lower below, upper above +0.01
    assert deployment_ci_state(0.005, 0.02) == INCONCLUSIVE


def test_threshold_boundary_and_nonfinite():
    # exactly at threshold is not strictly above/below -> inconclusive
    assert deployment_ci_state(0.01, 0.01) == INCONCLUSIVE
    assert deployment_ci_state(float("nan"), 0.02) == INCONCLUSIVE


def test_negative_and_harmful_intervals_ruled_out():
    # clearly harmful erasure: whole CI below +0.01 -> practical gain ruled out
    assert deployment_ci_state(-0.14, -0.06) == RULED_OUT


def test_deployable_benefit_requires_all_conditions():
    # confirmed CI but not source-safe -> not deployable
    assert deployable_benefit(CONFIRMED, source_task_safe=False, beats_random_control=True) is False
    # confirmed CI but does not beat random -> not deployable
    assert deployable_benefit(CONFIRMED, source_task_safe=True, beats_random_control=False) is False
    # all conditions -> deployable
    assert deployable_benefit(CONFIRMED, source_task_safe=True, beats_random_control=True) is True
    # inconclusive/ruled-out -> never deployable
    assert deployable_benefit(INCONCLUSIVE, True, True) is False
    assert deployable_benefit(RULED_OUT, True, True) is False


def test_ruled_out_everywhere_requires_all_cells():
    assert practical_gain_ruled_out_everywhere([RULED_OUT, RULED_OUT]) is True
    assert practical_gain_ruled_out_everywhere([RULED_OUT, INCONCLUSIVE]) is False
    assert practical_gain_ruled_out_everywhere([]) is False


def test_threshold_is_one_percent():
    assert PRACTICAL_THRESHOLD == pytest.approx(0.01)
