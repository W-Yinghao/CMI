"""Guards for acar/v4/policies.py + acar/v4/frontiers.py. SYNTHETIC FIXTURES ONLY; NO real DEV cohort, NO binding
go/no-go; reads no data, fits nothing, freezes nothing, selects nothing. Covers the frontier-contract hardening:
nested safe-set monotone coverage + action/abstention; SUBJECT-MACRO weighting (differs from batch-weighting; fallback
in the weighted denominator); fail-closed validation (NaN/Inf, zero-action/zero-batch, choice/action_idx out of range,
bad weights); true-oracle ceiling = global max red; perfect-info ⇒ zero info gap; useless ranking ⇒ positive info gap;
info_gap ≥ 0 invariant; single vs UNION score-oracle (union ≥ each single, ≤ true); Pareto upper envelope drops
dominated points; ceiling-gap telescoping identity (exact) vs AUC-gap (descriptive, separate).
Run: python -m acar.v4.tests.test_frontiers_policies
"""
import math
import numpy as np

from acar.v4 import policies as PO
from acar.v4 import frontiers as F

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


# ----------------------------------------------------------------------------- policies

def test_safe_set_monotone_and_action_and_abstain():
    harm = np.array([[0.1, 0.9], [0.5, 0.2], [2.0, 3.0]])
    benefit = np.array([[-1.0, -2.0], [-0.3, -0.5], [-5.0, -9.0]])
    lams = [-1.0, 0.15, 0.55, 1.0, 5.0]
    covs = [PO.coverage(PO.safe_set_policy(harm, benefit, l)) for l in lams]
    assert covs == sorted(covs), f"coverage must be non-decreasing in λ: {covs}"
    assert covs[0] == 0.0 and covs[-1] == 1.0
    assert list(PO.safe_set_policy(harm, benefit, 0.15)) == [0, ID, ID]
    assert list(PO.safe_set_policy(harm, benefit, 0.55)) == [0, 1, ID]
    assert list(PO.safe_set_policy(harm, benefit, 1.0)) == [1, 1, ID]   # b0 admits both → argmin benefit = a1


def test_safe_set_require_benefit():
    harm = np.array([[0.1], [0.1]]); benefit = np.array([[-0.5], [0.3]])
    assert list(PO.safe_set_policy(harm, benefit, 1.0, require_benefit=True)) == [0, ID]
    assert list(PO.safe_set_policy(harm, benefit, 1.0, require_benefit=False)) == [0, 0]


def test_benefit_ranked_and_direct_selective():
    benefit = np.array([[-0.5, -0.1], [-0.05, 0.2], [0.4, 0.5]])
    assert list(PO.benefit_ranked_policy(benefit, tau=-0.2)) == [0, ID, ID]
    gate = np.array([0.9, 0.4, 0.6]); act = np.array([1, 0, 2])
    assert list(PO.direct_selective_policy(gate, act, tau=0.5)) == [1, ID, 2]


def test_accounting_sign_fallback_harm():
    dr = np.array([[-2.0, 1.0], [0.5, 0.3], [-1.0, -0.5], [-3.0, 9.0]])
    choice = np.array([0, 1, 0, ID])                                    # b3 retained as fallback identity
    assert _close(PO.coverage(choice), 3.0 / 4.0)
    assert _close(PO.reduction(choice, dr), 0.675)                      # -(-2+0.3-1+0)/4
    assert _close(PO.harm_rate(choice, dr), 1.0 / 3.0)                  # only b1 harmful among 3 adapted
    none = np.full(4, ID)
    assert math.isnan(PO.harm_rate(none, dr)) and _close(PO.reduction(none, dr), 0.0) and PO.coverage(none) == 0.0


def test_subject_macro_weighting_and_fallback_in_denominator():
    subj = np.array(["A", "B", "B", "B"])                              # 2 subjects; B has 3 batches
    w = PO.subject_macro_weights(subj)
    assert _close(w[0], 0.5) and _close(w[1], 1 / 6) and _close(w.sum(), 1.0)
    dr = np.array([[-1.0], [-0.3], [-0.3], [-0.3]])
    only_a = np.array([0, ID, ID, ID])
    # batch-weighted vs subject-weighted differ
    assert _close(PO.coverage(only_a), 0.25) and _close(PO.coverage(only_a, weights=w), 0.5)
    assert _close(PO.reduction(only_a, dr), 0.25) and _close(PO.reduction(only_a, dr, weights=w), 0.5)
    # fallback row stays in the (subject) denominator: B's mass is split over all 3 of its batches incl. the fallback
    b_two_one_fallback = np.array([0, 0, 0, ID])
    assert _close(PO.coverage(b_two_one_fallback, weights=w), 0.5 + 1 / 6 + 1 / 6)   # = 0.8333… (not 1.0)


def test_fail_closed_validation():
    dr = np.array([[-1.0, 0.5], [0.2, -0.3]])
    _expect(ValueError, lambda: F.frontier_true_oracle(np.array([[np.nan, 0.0], [0.0, 0.0]])))      # NaN dr
    _expect(ValueError, lambda: PO.safe_set_policy(np.array([[np.inf]]), np.array([[0.0]]), 0.0))   # Inf harm
    _expect(ValueError, lambda: F.frontier_single_score_oracle(dr, np.array([np.nan, 0.0]), np.array([0, 0])))
    _expect(ValueError, lambda: PO.coverage(np.array([-2, 0])))                                     # choice=-2
    _expect(ValueError, lambda: PO.realized_dr(np.array([0, 5]), dr))                               # choice>=A
    _expect(ValueError, lambda: F.operating_point(dr, np.array([0, 5])))                            # choice>=A via op
    _expect(ValueError, lambda: PO.reduction(np.array([0, 0]), np.zeros((2, 0))))                   # zero actions
    _expect(ValueError, lambda: F.frontier_true_oracle(np.zeros((0, 2))))                           # zero batches
    _expect(ValueError, lambda: F.frontier_single_score_oracle(dr, np.array([1.0, 0.0]), np.array([0, 9])))  # act oob
    _expect(ValueError, lambda: PO.direct_selective_policy(np.array([0.1, 0.2]), np.array([-1, 0]), 0.0))    # act<0
    _expect(ValueError, lambda: PO.reduction(np.array([0, 0]), dr, weights=np.array([np.nan, 1.0])))         # bad w
    _expect(ValueError, lambda: PO.reduction(np.array([0, 0]), dr, weights=np.array([0.0, 0.0])))            # sum 0
    _expect(ValueError, lambda: PO.reduction(np.array([0, 0]), dr, weights=np.array([-1.0, 2.0])))           # neg w
    _expect(ValueError, lambda: PO.reduction(np.array([0, 0]), dr, weights=np.array([1.0])))                 # wrong len


# ----------------------------------------------------------------------------- frontiers

def test_true_oracle_is_global_max():
    rng = np.random.default_rng(0)
    dr = rng.normal(size=(50, 3))
    f = F.frontier_true_oracle(dr)
    assert _close(f.ceiling(), -float(np.mean(np.minimum(0.0, dr.min(axis=1)))))
    assert _close(f.coverage_at_ceiling(), float(np.mean(dr.min(axis=1) < 0.0)))
    assert f.coverage[0] == 0.0 and _close(f.red[0], 0.0) and math.isnan(f.harm[0]) and _close(f.coverage[-1], 1.0)


def test_perfect_info_zero_gap():
    rng = np.random.default_rng(1)
    dr = rng.normal(size=(40, 3))
    act, val = F.best_action(dr)
    assert _close(F.frontier_true_oracle(dr).ceiling(),
                  F.frontier_single_score_oracle(dr, -val, act).ceiling())


def test_useless_ranking_positive_info_gap():
    dr = np.array([[-3.0], [2.0], [-1.0], [0.5]])
    act = np.zeros(4, dtype=int)
    f_true = F.frontier_true_oracle(dr)
    f_single = F.frontier_single_score_oracle(dr, np.zeros(4), act)
    assert _close(f_true.ceiling(), 1.0) and _close(f_single.ceiling(), 0.75)
    assert (f_true.ceiling() - f_single.ceiling()) > 0.2


def test_info_gap_nonneg_invariant_random():
    rng = np.random.default_rng(2)
    for _ in range(25):
        n, A = int(rng.integers(5, 60)), int(rng.integers(1, 5))
        dr = rng.normal(size=(n, A))
        act = rng.integers(0, A, size=n); rank = rng.normal(size=n)
        assert F.frontier_true_oracle(dr).ceiling() - F.frontier_single_score_oracle(dr, rank, act).ceiling() >= -1e-9


def test_frontier_shapes_and_harm_bounds():
    rng = np.random.default_rng(3)
    dr = rng.normal(size=(30, 3)); act, val = F.best_action(dr)
    f = F.frontier_single_score_oracle(dr, -val, act)
    assert len(f.coverage) == 31 and len(f.red) == 31 and len(f.harm) == 31
    assert f.coverage.min() == 0.0 and _close(f.coverage.max(), 1.0) and math.isnan(f.harm[0])
    h = f.harm[1:]
    assert np.all((h >= -1e-12) & (h <= 1.0 + 1e-12))


def test_score_oracle_union_ge_singles_le_true():
    rng = np.random.default_rng(7)
    dr = rng.normal(size=(45, 3))
    cand1 = (rng.normal(size=45), rng.integers(0, 3, size=45))
    cand2 = (rng.normal(size=45), rng.integers(0, 3, size=45))
    s1 = F.frontier_single_score_oracle(dr, *cand1).ceiling()
    s2 = F.frontier_single_score_oracle(dr, *cand2).ceiling()
    union = F.frontier_score_oracle_union(dr, [cand1, cand2]).ceiling()
    true_c = F.frontier_true_oracle(dr).ceiling()
    assert union >= s1 - 1e-9 and union >= s2 - 1e-9
    assert _close(union, max(s1, s2)) and union <= true_c + 1e-9


def test_pareto_envelope_drops_dominated():
    fr = F.Frontier(coverage=np.array([0.0, 0.2, 0.4, 0.5, 0.6]),
                    red=np.array([0.0, 0.5, 0.3, 0.1, 0.8]),
                    harm=np.array([np.nan, 0.0, 0.0, 0.0, 0.0]))
    env = F.pareto_upper_envelope(fr)
    assert list(np.round(env.coverage, 6)) == [0.0, 0.2, 0.6]
    assert list(np.round(env.red, 6)) == [0.0, 0.5, 0.8]
    assert np.all(np.diff(env.coverage) > 0) and np.all(np.diff(env.red) > 0)


def test_ceiling_telescope_vs_auc_modes():
    rng = np.random.default_rng(4)
    n, A = 80, 3
    dr = rng.normal(size=(n, A))
    benefit = dr + 0.10 * rng.normal(size=(n, A))
    harm = dr + np.abs(0.10 * rng.normal(size=(n, A))) + 0.20
    rank = PO.adapt_rank_from_harm(harm); act, _ = PO.best_benefit_action(benefit)
    lam_grid = np.linspace(harm.min() - 0.5, harm.max() + 0.5, 12)
    fam = [PO.safe_set_policy(harm, benefit, l) for l in lam_grid]
    cands = [(rank, act)]
    cal_tiny = PO.safe_set_policy(harm, benefit, lam_grid[0])
    # ceiling mode: exact telescoping identity + nonneg info gap + large positive calibration gap
    gc = F.gap_decomposition(dr, cands, fam, cal_tiny, mode="ceiling")
    assert gc["info_gap"] >= -1e-9 and gc["calibration_gap"] > 0.01
    tele = gc["info_gap"] + gc["policy_gap"] + gc["calibration_gap"]
    assert _close(tele, gc["true_ceiling"] - gc["calibrated_red"])
    # policy-family raw operating points sorted; Pareto envelope is the plotted frontier
    fr = F.frontier_policy_family(dr, fam)
    assert np.all(np.diff(fr.coverage) >= -1e-12)
    # auc mode: descriptive area gaps, separate keys; never replaces ceiling calibration gap
    ga = F.gap_decomposition(dr, cands, fam, cal_tiny, mode="auc")
    assert {"true_auc", "score_auc", "policy_auc", "calibration_gap_ceiling"} <= set(ga)
    assert "calibration_gap" not in ga                       # auc mode does NOT emit a pass/fail calibration gap
    assert ga["true_auc"] >= ga["score_auc"] - 1e-9 and ga["info_gap"] >= -1e-9
    for k in ("true_auc", "score_auc", "policy_auc", "info_gap", "policy_gap"):
        assert math.isfinite(ga[k])
    _expect(ValueError, lambda: F.gap_decomposition(dr, cands, fam, cal_tiny, mode="bogus"))
    # a generous calibrated point sits much closer to the policy ceiling than the tiny one
    cal_gen = PO.safe_set_policy(harm, benefit, lam_grid[-1])
    gg = F.gap_decomposition(dr, cands, fam, cal_gen, mode="ceiling")
    assert gg["calibration_gap"] <= gc["calibration_gap"] + 1e-9


def main():
    print("ACAR v4 policy + frontier contract guards (synthetic fixtures only):")
    for t in (test_safe_set_monotone_and_action_and_abstain, test_safe_set_require_benefit,
              test_benefit_ranked_and_direct_selective, test_accounting_sign_fallback_harm,
              test_subject_macro_weighting_and_fallback_in_denominator, test_fail_closed_validation,
              test_true_oracle_is_global_max, test_perfect_info_zero_gap, test_useless_ranking_positive_info_gap,
              test_info_gap_nonneg_invariant_random, test_frontier_shapes_and_harm_bounds,
              test_score_oracle_union_ge_singles_le_true, test_pareto_envelope_drops_dominated,
              test_ceiling_telescope_vs_auc_modes):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 FRONTIER/POLICY CONTRACT GUARDS PASS")


if __name__ == "__main__":
    main()
