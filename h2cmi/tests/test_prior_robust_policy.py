"""Project A Step 19 — tests for the prior-robust adaptation policy.

Run:  python -m h2cmi.tests.test_prior_robust_policy
"""
from __future__ import annotations

from h2cmi.observability.prior_robust_policy import build_summary, _cell


def _run(lower, upper, uniform_gain, rho_key="0.1"):
    return {"dataset": "D", "target_subject": 1, "seed": 0, "uniform_gain": uniform_gain,
            "robust_bounds_by_rho": {rho_key: {"lower": lower, "upper": upper}}}


def _frontier(runs, rhos=(0.1,)):
    return {"runs": runs, "rhos": [str(r) for r in rhos]}


def test_prior_robust_policy_adapts_only_when_lower_bound_positive():
    # lower>tau -> adapt; lower<=tau -> not adapt
    runs = [_run(0.05, 0.2, 0.12), _run(-0.01, 0.2, 0.09)]
    s = build_summary(_frontier(runs), harm_thresholds=[0.0], rhos=[0.1])
    c = s["cells"][0]
    assert c["adaptation_coverage"] == 0.5                     # only the lower>0 run adapts
    assert c["robust_prior_safe_adaptation_exists"] is True


def test_prior_robust_policy_blocks_when_upper_bound_negative():
    runs = [_run(-0.3, -0.05, -0.15), _run(-0.3, -0.05, -0.2)]
    c = build_summary(_frontier(runs), harm_thresholds=[0.0], rhos=[0.1])["cells"][0]
    assert c["robust_harm_block_rate"] == 1.0 and c["adaptation_coverage"] == 0.0


def test_prior_robust_policy_abstains_when_ambiguous():
    runs = [_run(-0.1, 0.2, 0.05), _run(-0.2, 0.1, -0.05)]     # bounds straddle 0 -> ambiguous
    c = build_summary(_frontier(runs), harm_thresholds=[0.0], rhos=[0.1])["cells"][0]
    assert c["abstention_rate"] == 1.0


def test_robust_adapt_is_never_uniform_harmful():
    # a robust adapt (lower>tau>=0) implies uniform_gain>=lower>0 -> never uniform-harmful
    runs = [_run(0.05, 0.3, 0.12), _run(0.02, 0.25, 0.08)]
    s = build_summary(_frontier(runs), harm_thresholds=[0.0], rhos=[0.1])
    c = s["cells"][0]
    assert c["harm_rate_among_adapt_decisions_under_uniform"] == 0.0
    assert s["robust_adapt_never_uniform_harmful"] is True and s["claim_boundary_ok"] is True


def test_prior_robust_policy_best_selects_a_safe_cell_or_none():
    # no safe adaptation anywhere -> best is None
    runs = [_run(-0.1, 0.2, 0.05)]
    s = build_summary(_frontier(runs), harm_thresholds=[0.0], rhos=[0.1])
    assert s["best_prior_robust_policy"] is None and s["robust_prior_safe_adaptation_exists_any"] is False


def test_prior_robust_policy_excludes_actual_target_prior_claim():
    s = build_summary(_frontier([_run(0.05, 0.3, 0.12)]), harm_thresholds=[0.0], rhos=[0.1])
    assert s["actual_target_prior_identified"] is False
    assert s["not_deployable_without_r2_class_evidence"] is True
    assert s["prior_uncertainty_contract_required"] == "C15"
    assert "not deployable" in s["claim_boundary"].lower() or "not a deployable" in s["claim_boundary"].lower()


ALL_TESTS = [
    test_prior_robust_policy_adapts_only_when_lower_bound_positive,
    test_prior_robust_policy_blocks_when_upper_bound_negative,
    test_prior_robust_policy_abstains_when_ambiguous,
    test_robust_adapt_is_never_uniform_harmful,
    test_prior_robust_policy_best_selects_a_safe_cell_or_none,
    test_prior_robust_policy_excludes_actual_target_prior_claim,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} PRIOR-ROBUST-POLICY TESTS PASSED")


if __name__ == "__main__":
    run()
