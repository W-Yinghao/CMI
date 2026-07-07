"""Project A Step 16 — tests for the harm-control policy frontier.

Run:  python -m h2cmi.tests.test_policy_frontier
"""
from __future__ import annotations

from h2cmi.observability.policy_frontier import build_summary

# static: a ci policy that is unsafe at 0.05 but safe at 0.20; oracle present but must be excluded
_STATIC = {"cells": [
    {"policy": "ci_three_way", "k": 32, "tau": 0.0, "adaptation_coverage": 0.30,
     "harm_rate_among_adapt_decisions": 0.15, "missed_benefit_rate": 0.4},
    {"policy": "plugin_sign", "k": 16, "tau": 0.0, "adaptation_coverage": 0.10,
     "harm_rate_among_adapt_decisions": 0.60, "missed_benefit_rate": 0.2},
    {"policy": "oracle_full_target", "k": 0, "tau": 0.0, "adaptation_coverage": 0.15,
     "harm_rate_among_adapt_decisions": 0.0, "missed_benefit_rate": 0.0},   # must be EXCLUDED
    {"policy": "always_adapt", "k": 0, "tau": 0.0, "adaptation_coverage": 1.0,
     "harm_rate_among_adapt_decisions": 0.85, "missed_benefit_rate": 0.0}]}   # excluded (not label-based)
_SEQ = {"cells": [
    {"policy": "seq_ci_three_way", "budget": "128", "tau": 0.0, "adaptation_coverage": 0.08,
     "mean_labels_used": 40.0, "harm_rate_among_adapt_decisions": 0.03, "missed_benefit_rate": 0.5}]}


def test_frontier_excludes_oracle_from_deployable():
    s = build_summary(_STATIC, _SEQ)
    assert s["oracle_excluded"] is True
    for f in s["frontier_table"]:
        assert f["best_policy"] != "oracle_full_target"
    for h in ("best_under_harm_0_05", "best_under_harm_0_1", "best_under_harm_0_2", "best_under_harm_0_5"):
        b = s[h]
        assert b is None or b["policy"] != "oracle_full_target"


def test_frontier_reports_when_no_policy_meets_constraint():
    # all cells harm > 0.05 except the seq one (0.03, coverage 0.08 >= 0.05) -> 0.05 has the seq policy;
    # make a version where nothing meets 0.05
    hard = {"cells": [{"policy": "ci_three_way", "k": 8, "tau": 0.0, "adaptation_coverage": 0.5,
                       "harm_rate_among_adapt_decisions": 0.6, "missed_benefit_rate": 0.1}]}
    s = build_summary(hard, {"cells": []})
    assert s["best_under_harm_0_05"] is None and s["any_policy_meets_harm_0_05"] is False


def test_frontier_relaxes_harm_threshold_monotonically():
    s = build_summary(_STATIC, _SEQ)
    cov = {}
    for h, key in ((0.05, "best_under_harm_0_05"), (0.10, "best_under_harm_0_1"),
                   (0.20, "best_under_harm_0_2"), (0.50, "best_under_harm_0_5")):
        b = s[key]
        cov[h] = b["adaptation_coverage"] if b else 0.0
    # relaxing the harm threshold can only add eligible policies -> best coverage non-decreasing
    assert cov[0.05] <= cov[0.10] <= cov[0.20] <= cov[0.50]


ALL_TESTS = [
    test_frontier_excludes_oracle_from_deployable,
    test_frontier_reports_when_no_policy_meets_constraint,
    test_frontier_relaxes_harm_threshold_monotonically,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} POLICY-FRONTIER TESTS PASSED")


if __name__ == "__main__":
    run()
