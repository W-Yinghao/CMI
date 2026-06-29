"""Guards for acar/v4/policies.py (label-free selective policy families) and acar/v4/frontiers.py (Direction C
information-limit risk-coverage frontiers). SYNTHETIC FIXTURES ONLY; NO real DEV cohort, NO binding go/no-go; reads no
data, fits nothing, freezes nothing, selects nothing. Proves: nested safe-set monotone coverage + correct action /
abstention; deployed accounting (coverage/red/harm, fallback in denominator, sign convention); true-oracle ceiling =
global max red; perfect-info ⇒ zero info gap; useless ranking ⇒ strictly positive info gap; info_gap ≥ 0 invariant on
random fixtures; policy-family frontier sorted; calibration gap large for a v3-like tiny-coverage point; telescoping
gap identity (info+policy+calibration = true_ceiling − calibrated_red).
Run: python -m acar.v4.tests.test_frontiers_policies
"""
import math
import numpy as np

from acar.v4 import policies as PO
from acar.v4 import frontiers as F

ID = PO.IDENTITY


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


# ----------------------------------------------------------------------------- policies

def test_safe_set_monotone_and_action_and_abstain():
    # 3 batches x 2 actions. harm/benefit lower = safer/better.
    harm = np.array([[0.1, 0.9],     # b0: a0 safe early, a1 only at high λ
                     [0.5, 0.2],     # b1: a1 safer
                     [2.0, 3.0]])    # b2: nothing safe until λ huge
    benefit = np.array([[-1.0, -2.0],  # b0: a1 more reduction (but harmful unless λ≥0.9)
                        [-0.3, -0.5],  # b1: a1 more reduction
                        [-5.0, -9.0]]) # b2
    lams = [-1.0, 0.15, 0.55, 1.0, 5.0]
    covs = [PO.coverage(PO.safe_set_policy(harm, benefit, l)) for l in lams]
    assert covs == sorted(covs), f"coverage must be non-decreasing in λ: {covs}"
    assert covs[0] == 0.0, "λ below all harm ⇒ no admissible action ⇒ identity everywhere"
    assert covs[-1] == 1.0, "λ above all harm ⇒ full coverage"
    # at λ=0.15: only b0/a0 admitted (harm .1); b1 none (.5,.2>.15); b2 none.
    c = PO.safe_set_policy(harm, benefit, 0.15)
    assert c[0] == 0 and c[1] == ID and c[2] == ID, f"{c}"
    # at λ=0.55: b0 admits {a0}; b1 admits {a1} (.2); pick argmin benefit within admitted.
    c = PO.safe_set_policy(harm, benefit, 0.55)
    assert c[0] == 0 and c[1] == 1 and c[2] == ID, f"{c}"
    # at λ=1.0: b0 admits {a0,a1}->argmin benefit = a1 (-2 < -1); b1 admits {a1}; b2 none.
    c = PO.safe_set_policy(harm, benefit, 1.0)
    assert c[0] == 1 and c[1] == 1 and c[2] == ID, f"{c}"


def test_safe_set_require_benefit():
    harm = np.array([[0.1], [0.1]])
    benefit = np.array([[-0.5], [0.3]])   # b1's only admitted action is predicted NOT to help
    c = PO.safe_set_policy(harm, benefit, 1.0, require_benefit=True)
    assert c[0] == 0 and c[1] == ID, f"require_benefit must abstain when best admitted benefit≥0: {c}"
    c2 = PO.safe_set_policy(harm, benefit, 1.0, require_benefit=False)
    assert c2[0] == 0 and c2[1] == 0, "default family adapts the admitted action regardless of predicted sign"


def test_benefit_ranked_and_direct_selective():
    benefit = np.array([[-0.5, -0.1], [-0.05, 0.2], [0.4, 0.5]])
    c = PO.benefit_ranked_policy(benefit, tau=-0.2)   # adapt iff min benefit ≤ -0.2
    assert c[0] == 0 and c[1] == ID and c[2] == ID, f"{c}"
    gate = np.array([0.9, 0.4, 0.6]); act = np.array([1, 0, 2])
    d = PO.direct_selective_policy(gate, act, tau=0.5)  # adapt iff gate≥0.5
    assert d[0] == 1 and d[1] == ID and d[2] == 2, f"{d}"


def test_accounting_sign_fallback_harm():
    # 4 batches incl. a forced-identity fallback row (choice -1) kept in the denominator.
    dr = np.array([[-2.0, 1.0],   # b0 best a0 (-2) beneficial
                   [ 0.5, 0.3],   # b1 both harmful
                   [-1.0, -0.5],  # b2 beneficial
                   [-3.0, 9.0]])  # b3 fallback (forced identity)
    choice = np.array([0, 1, 0, ID])      # b3 retained as identity
    assert _close(PO.coverage(choice), 3.0 / 4.0)
    # realized ΔR: b0 -2, b1 +0.3, b2 -1, b3 0 ; red = -mean = -(-2+0.3-1+0)/4 = 0.675
    assert _close(PO.reduction(choice, dr), 0.675)
    # harm among adapted (b0,b1,b2): only b1 (+0.3) harmful ⇒ 1/3
    assert _close(PO.harm_rate(choice, dr), 1.0 / 3.0)
    # nothing adapted ⇒ harm NaN, red 0, coverage 0
    none = np.full(4, ID)
    assert math.isnan(PO.harm_rate(none, dr)) and _close(PO.reduction(none, dr), 0.0) and PO.coverage(none) == 0.0


# ----------------------------------------------------------------------------- frontiers

def test_true_oracle_is_global_max():
    rng = np.random.default_rng(0)
    dr = rng.normal(size=(50, 3))
    f = F.frontier_true_oracle(dr)
    global_max = -float(np.mean(np.minimum(0.0, dr.min(axis=1))))
    assert _close(f.ceiling(), global_max), (f.ceiling(), global_max)
    beneficial_frac = float(np.mean(dr.min(axis=1) < 0.0))
    assert _close(f.coverage_at_ceiling(), beneficial_frac), (f.coverage_at_ceiling(), beneficial_frac)
    assert f.coverage[0] == 0.0 and _close(f.red[0], 0.0) and math.isnan(f.harm[0])
    assert _close(f.coverage[-1], 1.0)


def test_perfect_info_zero_gap():
    rng = np.random.default_rng(1)
    dr = rng.normal(size=(40, 3))
    act, val = F.best_action(dr)               # true best action + value
    f_true = F.frontier_true_oracle(dr)
    f_score = F.frontier_score_oracle(dr, rank_score=-val, action_idx=act)  # score == truth
    assert _close(f_true.ceiling(), f_score.ceiling()), (f_true.ceiling(), f_score.ceiling())


def test_useless_ranking_positive_info_gap():
    # deterministic single-action example; constant rank ⇒ original order ⇒ a harmful batch adapted early.
    dr = np.array([[-3.0], [2.0], [-1.0], [0.5]])
    act = np.zeros(4, dtype=int)
    f_true = F.frontier_true_oracle(dr)
    f_score = F.frontier_score_oracle(dr, rank_score=np.zeros(4), action_idx=act)
    assert _close(f_true.ceiling(), 1.0), f_true.ceiling()
    assert _close(f_score.ceiling(), 0.75), f_score.ceiling()
    assert (f_true.ceiling() - f_score.ceiling()) > 0.2


def test_info_gap_nonneg_invariant_random():
    rng = np.random.default_rng(2)
    for _ in range(25):
        n, A = int(rng.integers(5, 60)), int(rng.integers(1, 5))
        dr = rng.normal(size=(n, A))
        act = rng.integers(0, A, size=n)        # arbitrary (not necessarily best) action
        rank = rng.normal(size=n)               # arbitrary ranking
        f_true = F.frontier_true_oracle(dr)
        f_score = F.frontier_score_oracle(dr, rank_score=rank, action_idx=act)
        assert f_true.ceiling() - f_score.ceiling() >= -1e-9, "info_gap must be ≥ 0 for any score-driven frontier"


def test_frontier_shapes_and_harm_bounds():
    rng = np.random.default_rng(3)
    dr = rng.normal(size=(30, 3))
    act, val = F.best_action(dr)
    f = F.frontier_score_oracle(dr, rank_score=-val, action_idx=act)
    assert len(f.coverage) == 31 and len(f.red) == 31 and len(f.harm) == 31
    assert f.coverage.min() == 0.0 and _close(f.coverage.max(), 1.0)
    assert math.isnan(f.harm[0])
    h = f.harm[1:]
    assert np.all((h >= -1e-12) & (h <= 1.0 + 1e-12))


def test_policy_family_frontier_and_calibration_gap_and_telescope():
    rng = np.random.default_rng(4)
    n, A = 80, 3
    dr = rng.normal(size=(n, A))
    benefit = dr + 0.10 * rng.normal(size=(n, A))           # imperfect center estimate
    harm = dr + np.abs(0.10 * rng.normal(size=(n, A))) + 0.20  # conservative upper estimate
    rank = PO.adapt_rank_from_harm(harm)
    act, _ = PO.best_benefit_action(benefit)
    lam_grid = np.linspace(harm.min() - 0.5, harm.max() + 0.5, 12)
    fam = [PO.safe_set_policy(harm, benefit, l) for l in lam_grid]
    fr = F.frontier_policy_family(dr, fam)
    assert np.all(np.diff(fr.coverage) >= -1e-12), "policy-family frontier must be sorted by coverage"
    assert fr.coverage.min() >= 0.0 and fr.coverage.max() <= 1.0
    # v3-like calibrated point: very small budget ⇒ near-zero coverage ⇒ large positive calibration gap
    cal_tiny = PO.safe_set_policy(harm, benefit, lam_grid[0])
    g = F.gap_decomposition(dr, rank_score=rank, action_idx=act, policy_choices=fam, calibrated_choice=cal_tiny)
    assert g["info_gap"] >= -1e-9
    assert g["calibration_gap"] >= -1e-9 and g["calibration_gap"] > 0.01, g["calibration_gap"]
    # telescoping identity: info + policy + calibration == true_ceiling − calibrated_red
    tele = g["info_gap"] + g["policy_gap"] + g["calibration_gap"]
    assert _close(tele, g["true_ceiling"] - g["calibrated_red"]), (tele, g["true_ceiling"], g["calibrated_red"])
    # a generous calibrated point sits much closer to the policy ceiling than the tiny one
    cal_gen = PO.safe_set_policy(harm, benefit, lam_grid[-1])
    g2 = F.gap_decomposition(dr, rank_score=rank, action_idx=act, policy_choices=fam, calibrated_choice=cal_gen)
    assert g2["calibration_gap"] <= g["calibration_gap"] + 1e-9


def main():
    print("ACAR v4 policy + frontier guards (synthetic fixtures only):")
    for t in (test_safe_set_monotone_and_action_and_abstain, test_safe_set_require_benefit,
              test_benefit_ranked_and_direct_selective, test_accounting_sign_fallback_harm,
              test_true_oracle_is_global_max, test_perfect_info_zero_gap,
              test_useless_ranking_positive_info_gap, test_info_gap_nonneg_invariant_random,
              test_frontier_shapes_and_harm_bounds, test_policy_family_frontier_and_calibration_gap_and_telescope):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 FRONTIER/POLICY GUARDS PASS")


if __name__ == "__main__":
    main()
