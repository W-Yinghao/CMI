"""Guards for acar/v4/risk_control.py (finite-grid risk-control calibration). SYNTHETIC FIXTURES ONLY; NO real DEV
cohort, NO v3 loader, NO V4 candidate, NO freeze. Proves: most-aggressive PASSING λ is selected (incl. the key
NON-MONOTONE risk curve — selection does NOT assume monotonicity and does NOT stop at the first failure); grid order
independence; Holm/Bonferroni adjusted-p properties + passer containment; one-sided p-value direction; ttest +
Hoeffding methods; subject-level losses (mean/positive/harm_indicator) with fallback identity in the denominator;
calibration is a function of CAL losses only; full fail-closed contract + NOT_EVALUABLE for < 2 subjects.
Run: python -m acar.v4.tests.test_risk_control
"""
import math
import numpy as np

from acar.v4 import policies as PO
from acar.v4 import risk_control as RC

ID = PO.IDENTITY


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:                       # noqa
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no exception raised")


def _losses(seed, n, col_means, sd=0.05):
    """[n, L] subject-loss matrix with the given per-column means + small noise (clear pass/fail separation)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(m, sd, size=n) for m in col_means], axis=1)


# ----------------------------------------------------------------------------- selection semantics

def test_selection_basic_and_all_pass_and_no_pass():
    grid = np.array([0.0, 1.0, 2.0])
    # all below budget 0 ⇒ all pass ⇒ most aggressive (largest λ) selected
    r = RC.select_ltt_grid(grid, _losses(1, 40, [-0.5, -0.5, -0.5]), alpha=0.1, budget=0.0,
                           aggressiveness="increasing_lambda")
    assert r.status == "PASS" and _close(r.selected_lambda, 2.0) and bool(r.passer_mask.all())
    # all above budget ⇒ none pass
    r2 = RC.select_ltt_grid(grid, _losses(2, 40, [0.5, 0.5, 0.5]), alpha=0.1, budget=0.0,
                            aggressiveness="increasing_lambda")
    assert r2.status == "NO_PASS" and r2.selected_index is None and not r2.passer_mask.any()
    # decreasing_lambda aggressiveness picks the smallest passing λ
    r3 = RC.select_ltt_grid(grid, _losses(3, 40, [-0.5, -0.5, -0.5]), alpha=0.1, budget=0.0,
                            aggressiveness="decreasing_lambda")
    assert _close(r3.selected_lambda, 0.0)


def test_non_monotone_risk_curve_selects_most_aggressive_passer():
    # λ0 low risk (pass), λ1 HIGH risk (fail), λ2 low risk (pass). A monotone/fixed-sequence selector would stop at λ0;
    # the LTT grid selector must pick the most aggressive PASSING λ = λ2.
    grid = np.array([0.0, 1.0, 2.0])
    sl = _losses(10, 50, [-0.5, +0.5, -0.5])
    r = RC.select_ltt_grid(grid, sl, alpha=0.1, budget=0.0, aggressiveness="increasing_lambda")
    assert list(r.passer_mask) == [True, False, True]
    assert r.status == "PASS" and _close(r.selected_lambda, 2.0)


def test_failing_most_aggressive_does_not_block_less_aggressive():
    # largest λ (most aggressive) fails; a less aggressive λ passes and must be selected.
    grid = np.array([0.0, 1.0, 2.0])
    sl = _losses(11, 50, [-0.5, -0.5, +0.5])
    r = RC.select_ltt_grid(grid, sl, alpha=0.1, budget=0.0, aggressiveness="increasing_lambda")
    assert list(r.passer_mask) == [True, True, False] and _close(r.selected_lambda, 1.0)


def test_grid_order_independence():
    grid = np.array([0.0, 1.0, 2.0])
    sl = _losses(10, 50, [-0.5, +0.5, -0.5])
    base = RC.select_ltt_grid(grid, sl, alpha=0.1, budget=0.0, aggressiveness="increasing_lambda")
    perm = np.array([2, 0, 1])
    shuffled = RC.select_ltt_grid(grid[perm], sl[:, perm], alpha=0.1, budget=0.0,
                                  aggressiveness="increasing_lambda")
    assert _close(shuffled.selected_lambda, base.selected_lambda)   # selection by VALUE, not position


def test_calibration_depends_on_cal_losses_only():
    grid = np.array([0.0, 1.0, 2.0])
    a = RC.select_ltt_grid(grid, _losses(4, 40, [-0.5, +0.5, +0.5]), alpha=0.1, budget=0.0,
                           aggressiveness="increasing_lambda")
    b = RC.select_ltt_grid(grid, _losses(5, 40, [+0.5, +0.5, -0.5]), alpha=0.1, budget=0.0,
                           aggressiveness="increasing_lambda")
    assert _close(a.selected_lambda, 0.0) and _close(b.selected_lambda, 2.0)   # only the CAL losses moved λ*


# ----------------------------------------------------------------------------- corrections

def test_holm_bonferroni_properties_and_containment():
    p = np.array([0.02, 0.03, 0.04, 0.5])
    holm = RC.holm_adjust(p); bonf = RC.bonferroni_adjust(p)
    assert np.all(holm >= p - 1e-12) and np.all(bonf >= p - 1e-12)           # adjusted ≥ raw
    assert np.all(holm <= bonf + 1e-12)                                       # Holm ≥ power ⇒ ≤ Bonferroni
    alpha = 0.1
    passers_none = p <= alpha; passers_holm = holm <= alpha; passers_bonf = bonf <= alpha
    assert set(np.where(passers_bonf)[0]) <= set(np.where(passers_holm)[0]) <= set(np.where(passers_none)[0])
    assert list(passers_bonf) == [True, False, False, False]
    assert list(passers_holm) == [True, True, True, False]


def test_pvalue_direction():
    sl_lo = _losses(20, 40, [-0.5]); sl_hi = _losses(21, 40, [+0.5])
    assert RC.one_sided_mean_risk_pvalue(sl_lo, 0.0)[0] < 0.01
    assert RC.one_sided_mean_risk_pvalue(sl_hi, 0.0)[0] > 0.99


def test_hoeffding_method():
    grid = np.array([0.0, 1.0])
    sl = _losses(30, 80, [0.05, 0.5], sd=0.02)                                # bounded harm-rate-like losses in [0,1]
    sl = np.clip(sl, 0.0, 1.0)
    r = RC.select_ltt_grid(grid, sl, alpha=0.1, budget=0.2, aggressiveness="increasing_lambda",
                           method="hoeffding", loss_bounds=(0.0, 1.0))
    assert list(r.passer_mask) == [True, False] and _close(r.selected_lambda, 0.0)
    assert np.all(np.isfinite(r.upper_confidence))
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, sl, alpha=0.1, budget=0.2,
                                                   aggressiveness="increasing_lambda", method="hoeffding"))  # no bounds
    bad = sl.copy(); bad[0, 0] = 2.0
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, bad, alpha=0.1, budget=0.2,
                                                   aggressiveness="increasing_lambda", method="hoeffding",
                                                   loss_bounds=(0.0, 1.0)))                                   # oob loss


# ----------------------------------------------------------------------------- subject-level losses from a policy

def test_subject_losses_from_policy_and_fallback_denominator():
    subj = np.array(["A", "A", "B"])
    dr = np.array([[-1.0], [2.0], [-0.5]])
    C = np.array([[0, 0, ID], [0, ID, 0]])                                    # [L=2, n=3]
    mean = RC.subject_losses_from_policy(C, dr, subj, loss="mean")
    pos = RC.subject_losses_from_policy(C, dr, subj, loss="positive")
    harm = RC.subject_losses_from_policy(C, dr, subj, loss="harm_indicator")
    assert mean.shape == (2, 2)
    assert np.allclose(mean, [[0.5, -0.5], [0.0, -0.5]])      # A l1 = (-1+0)/2 ⇒ fallback row in subject denominator
    assert np.allclose(pos, [[1.0, 0.0], [0.0, 0.0]])         # negative ΔR clipped to 0
    assert np.allclose(harm, [[0.5, 0.0], [0.0, 0.0]])        # only adapted harmful batch counts


# ----------------------------------------------------------------------------- status + fail-closed

def test_not_evaluable_and_zero_subjects():
    grid = np.array([0.0, 1.0])
    one = RC.select_ltt_grid(grid, np.array([[-0.5, -0.5]]), alpha=0.1, budget=0.0,
                             aggressiveness="increasing_lambda")
    assert one.status == "NOT_EVALUABLE" and one.selected_index is None and not one.passer_mask.any()
    assert np.all(np.isnan(one.p_values))
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, np.zeros((0, 2)), alpha=0.1, budget=0.0,
                                                   aggressiveness="increasing_lambda"))


def test_fail_closed_validation():
    grid = np.array([0.0, 1.0, 2.0])
    good = _losses(6, 20, [-0.5, -0.5, -0.5])
    _expect(ValueError, lambda: RC.select_ltt_grid(np.zeros(0), good, alpha=0.1, budget=0.0,
                                                   aggressiveness="increasing_lambda"))                 # empty grid
    _expect(ValueError, lambda: RC.select_ltt_grid(np.array([0.0, 0.0, 1.0]), good, alpha=0.1, budget=0.0,
                                                   aggressiveness="increasing_lambda"))                 # dup λ
    _expect(ValueError, lambda: RC.select_ltt_grid(np.array([0.0, np.nan, 1.0]), good, alpha=0.1, budget=0.0,
                                                   aggressiveness="increasing_lambda"))                 # nan λ
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, good[:, :2], alpha=0.1, budget=0.0,
                                                   aggressiveness="increasing_lambda"))                 # shape mismatch
    bad = good.copy(); bad[0, 0] = np.inf
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, bad, alpha=0.1, budget=0.0,
                                                   aggressiveness="increasing_lambda"))                 # inf loss
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, good, alpha=1.5, budget=0.0,
                                                   aggressiveness="increasing_lambda"))                 # alpha
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, good, alpha=0.1, budget=np.inf,
                                                   aggressiveness="increasing_lambda"))                 # budget
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, good, alpha=0.1, budget=0.0,
                                                   aggressiveness="increasing_lambda", correction="zzz"))
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, good, alpha=0.1, budget=0.0, aggressiveness="zzz"))
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, good, alpha=0.1, budget=0.0,
                                                   aggressiveness="increasing_lambda", method="zzz"))
    _expect(ValueError, lambda: RC.select_ltt_grid(grid, np.zeros((5, 3, 2)), alpha=0.1, budget=0.0,
                                                   aggressiveness="increasing_lambda"))                 # not 2-D
    # subject_losses_from_policy contract
    dr = np.array([[-1.0], [2.0], [-0.5]]); subj = np.array(["A", "A", "B"])
    _expect(ValueError, lambda: RC.subject_losses_from_policy(np.array([[0.0, 0.0, -1.0]]), dr, subj, loss="mean"))
    _expect(ValueError, lambda: RC.subject_losses_from_policy(np.array([[0, 0, ID]]), dr, subj, loss="zzz"))
    _expect(ValueError, lambda: RC.subject_losses_from_policy(np.array([[0, 0]]), dr, subj, loss="mean"))  # n mismatch
    _expect(ValueError, lambda: RC.subject_losses_from_policy(np.array([[0, 0, ID]]), dr, np.array(["A", "B"]),
                                                              loss="mean"))                              # ids mismatch


def main():
    print("ACAR v4 risk-control calibration guards (synthetic fixtures only):")
    for t in (test_selection_basic_and_all_pass_and_no_pass,
              test_non_monotone_risk_curve_selects_most_aggressive_passer,
              test_failing_most_aggressive_does_not_block_less_aggressive, test_grid_order_independence,
              test_calibration_depends_on_cal_losses_only, test_holm_bonferroni_properties_and_containment,
              test_pvalue_direction, test_hoeffding_method, test_subject_losses_from_policy_and_fallback_denominator,
              test_not_evaluable_and_zero_subjects, test_fail_closed_validation):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 RISK-CONTROL GUARDS PASS")


if __name__ == "__main__":
    main()
