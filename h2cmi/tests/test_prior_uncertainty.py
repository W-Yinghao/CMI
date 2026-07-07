"""Project A Step 19 — tests for the prior-uncertainty robustness frontier.

Validates the exact L1-ball robust-gain optimizer against a binary closed form and a brute-force
simplex grid.  Run:  python -m h2cmi.tests.test_prior_uncertainty
"""
from __future__ import annotations

from itertools import product

from h2cmi.observability.prior_uncertainty import (robust_gain_bounds_l1, minimal_l1_radius_to_flip,
                                                   build_summary)


def _binary_closed_form(d, rho):
    # gain(π)=π0 d0+(1-π0) d1, ||π-u||_1 = 2|π0-0.5| ≤ ρ -> |π0-0.5| ≤ ρ/2 (clamped to [0,1])
    base = 0.5 * (d[0] + d[1])
    spread = abs(d[0] - d[1])
    lo = max(min(d), base - (rho / 2.0) * spread)
    hi = min(max(d), base + (rho / 2.0) * spread)
    return lo, hi


def _brute_bounds(d, rho, n=120):
    # coarse grid over the simplex; ||π - u||_1 ≤ ρ
    K = len(d)
    u = [1.0 / K] * K
    best_lo, best_hi = float("inf"), float("-inf")
    for combo in product(range(n + 1), repeat=K - 1):
        if sum(combo) > n:
            continue
        pi = [c / n for c in combo] + [1.0 - sum(combo) / n]
        if sum(abs(pi[i] - u[i]) for i in range(K)) > rho + 1e-9:
            continue
        g = sum(pi[i] * d[i] for i in range(K))
        best_lo = min(best_lo, g)
        best_hi = max(best_hi, g)
    return best_lo, best_hi


def test_robust_bounds_l1_match_binary_closed_form():
    for d in ([-0.2, 0.3], [0.1, 0.1], [-0.4, -0.1], [0.5, -0.5]):
        for rho in (0.0, 0.1, 0.3, 0.5, 1.0, 2.0):
            lo, hi = robust_gain_bounds_l1(d, rho)
            clo, chi = _binary_closed_form(d, rho)
            assert abs(lo - clo) < 1e-9 and abs(hi - chi) < 1e-9, (d, rho, lo, clo, hi, chi)


def test_robust_bounds_l1_match_bruteforce_grid_for_three_classes():
    for d in ([-0.2, 0.05, 0.3], [0.1, -0.3, 0.2], [-0.1, -0.1, 0.4], [0.2, 0.2, 0.2]):
        for rho in (0.1, 0.2, 0.5, 1.0):
            lo, hi = robust_gain_bounds_l1(d, rho)
            blo, bhi = _brute_bounds(d, rho, n=120)
            # exact optimum is at least as extreme as any grid point; grid discretisation ~ 1/n
            assert lo <= blo + 1e-6 and hi >= bhi - 1e-6
            assert abs(lo - blo) < 0.02 and abs(hi - bhi) < 0.02, (d, rho, lo, blo, hi, bhi)


def test_bounds_monotone_with_rho():
    d = [-0.2, 0.05, 0.3]
    prev_lo, prev_hi = robust_gain_bounds_l1(d, 0.0)
    for rho in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        lo, hi = robust_gain_bounds_l1(d, rho)
        assert lo <= prev_lo + 1e-12 and hi >= prev_hi - 1e-12    # lower ↓, upper ↑
        prev_lo, prev_hi = lo, hi


def test_minimal_flip_radius_zero_if_uniform_gain_zero():
    assert minimal_l1_radius_to_flip([0.3, -0.3]) == 0.0          # uniform gain 0
    assert minimal_l1_radius_to_flip([0.2, -0.1, -0.1]) == 0.0    # mean 0


def test_minimal_flip_radius_none_if_sign_cannot_flip_over_simplex():
    assert minimal_l1_radius_to_flip([0.1, 0.3, 0.2]) is None     # all >0 -> gain always >0
    assert minimal_l1_radius_to_flip([-0.1, -0.3, -0.2]) is None  # all <0 -> gain always <0


def test_minimal_flip_radius_matches_bound_crossing():
    # the flip radius is where robust_lower (or upper) crosses 0
    d = [-0.05, 0.4]                                              # uniform gain 0.175 > 0 -> push down
    r = minimal_l1_radius_to_flip(d)
    lo, _ = robust_gain_bounds_l1(d, r)
    assert abs(lo) < 1e-6                                         # at the flip radius, lower bound = 0


def test_prior_uncertainty_output_does_not_claim_actual_target_prior_identified():
    ps = {"runs": [{"dataset": "D", "target_subject": 1, "seed": 0, "class_delta_vector": [-0.2, 0.3]}]}
    s = build_summary(ps, [0.0, 0.1, 0.2, 1.0], [0.0])
    assert s["actual_target_prior_identified"] is False
    assert s["deployment_prior_identified_under_R1"] is False
    assert s["prior_uncertainty_contract_required"] == "C15"
    assert "not" in s["claim_boundary"].lower() and "identif" in s["claim_boundary"].lower()


def test_prior_uncertainty_aggregate_flip_fractions():
    # d=[-0.3,0.4] flips at L1≈0.143 (uniform gain 0.05); d=[0.1,0.2] never flips (both >0)
    ps = {"runs": [{"dataset": "D", "target_subject": 1, "seed": 0, "class_delta_vector": [-0.3, 0.4]},
                   {"dataset": "D", "target_subject": 2, "seed": 0, "class_delta_vector": [0.1, 0.2]}]}
    s = build_summary(ps, [0.0, 0.1, 0.2, 0.5, 1.0, 2.0], [0.0])
    assert s["n_unflippable_over_simplex"] == 1                   # the all-positive run
    assert s["fraction_flip_within_l1_0_10"] == 0.0              # 0.143 > 0.10
    assert s["fraction_flip_within_l1_0_20"] == 0.5              # run1 within, run2 unflippable
    assert s["fraction_flip_within_l1_0_50"] == 0.5


ALL_TESTS = [
    test_robust_bounds_l1_match_binary_closed_form,
    test_robust_bounds_l1_match_bruteforce_grid_for_three_classes,
    test_bounds_monotone_with_rho,
    test_minimal_flip_radius_zero_if_uniform_gain_zero,
    test_minimal_flip_radius_none_if_sign_cannot_flip_over_simplex,
    test_minimal_flip_radius_matches_bound_crossing,
    test_prior_uncertainty_output_does_not_claim_actual_target_prior_identified,
    test_prior_uncertainty_aggregate_flip_fractions,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} PRIOR-UNCERTAINTY TESTS PASSED")


if __name__ == "__main__":
    run()
