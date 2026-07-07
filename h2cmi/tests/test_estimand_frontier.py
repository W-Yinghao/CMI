"""Project A Step 17 — tests for the per-estimand harm-control frontier.

Run:  python -m h2cmi.tests.test_estimand_frontier
"""
from __future__ import annotations

from h2cmi.observability.estimand_frontier import build_summary


def _fake_consistency():
    # minimal cells covering all 4 (estimand, sampling) groups; one deployable cell per group
    def cell(e, s, k, policy, cov, harm, req):
        return {"estimand": e, "sampling": s, "k": k, "tau": 0.0, "policy": policy,
                "requires_contract": req, "adaptation_coverage": cov,
                "harm_rate_among_adapt_decisions": harm, "missed_benefit_rate": 0.2,
                "missing_class_rate": 0.0}
    cells = [
        cell("accuracy_gain", "iid", "32", "ci_three_way", 0.30, 0.02, None),
        cell("accuracy_gain", "class_balanced", "32", "ci_three_way", 0.28, 0.03, None),
        cell("balanced_accuracy_gain", "iid", "32", "ci_three_way", 0.10, 0.08, None),
        cell("balanced_accuracy_gain", "class_balanced", "32", "ci_three_way", 0.12, 0.04, "C13"),
    ]
    return {"cells": cells}


def test_frontier_keeps_accuracy_and_bacc_separate():
    s = build_summary(_fake_consistency())
    assert set(s["group_keys"]) == {"accuracy_gain:iid", "accuracy_gain:class_balanced",
                                    "balanced_accuracy_gain:iid", "balanced_accuracy_gain:class_balanced"}
    # each group is its own object with its own estimand; no shared/merged frontier
    for gk, g in s["groups"].items():
        assert g["estimand"] in gk and g["sampling"] in gk


def test_no_overall_best_across_estimands():
    s = build_summary(_fake_consistency())
    assert s["no_overall_best_across_estimands"] is True
    assert s["overall_best_policy"] is None
    assert "overall_best_policy" not in s.get("groups", {})       # never a cross-estimand winner


def test_balanced_accuracy_class_balanced_requires_c13():
    s = build_summary(_fake_consistency())
    assert s["groups"]["balanced_accuracy_gain:class_balanced"]["requires_contract"] == "C13"
    assert s["groups"]["accuracy_gain:class_balanced"]["requires_contract"] is None
    assert s["groups"]["balanced_accuracy_gain:iid"]["requires_contract"] is None
    assert s["class_balanced_bacc_requires_contract"] == "C13"


def test_accuracy_policy_never_controls_bacc():
    s = build_summary(_fake_consistency())
    assert s["accuracy_policy_controls_bacc"] is False


ALL_TESTS = [
    test_frontier_keeps_accuracy_and_bacc_separate,
    test_no_overall_best_across_estimands,
    test_balanced_accuracy_class_balanced_requires_c13,
    test_accuracy_policy_never_controls_bacc,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} ESTIMAND-FRONTIER TESTS PASSED")


if __name__ == "__main__":
    run()
