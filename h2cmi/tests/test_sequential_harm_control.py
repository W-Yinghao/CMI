"""Project A Step 16 — tests for sequential label-acquisition harm-control policies.

Run:  python -m h2cmi.tests.test_sequential_harm_control
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.observability.sequential_harm_control import (_SEQ_POLICIES, _seq_decide, _select_best,
                                                        build_summary)


def _write_run(root, dataset, target, seed, kind):
    y = [0] * 60 + [1] * 60
    if kind == "benefit":
        idp, adp = [1] * 120, list(y)                          # adapt clearly better -> d mostly +
    else:
        idp, adp = list(y), [1] * 120                          # identity better -> d mostly -
    d = Path(root) / f"dataset={dataset}_target={target}_seed={seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw_results.json").write_text(json.dumps({"per_trial_oracle_predictions": {
        "y_true": y, "identity_pred": idp, "adapt_pred": adp}}))
    (d / "run_manifest.json").write_text(json.dumps(
        {"status": "ok", "dataset": dataset, "target_subject": target, "seed": seed, "n_classes": 2}))
    return d


def _summary(root, **kw):
    _write_run(root, "BNCI2014_001", 1, 0, "benefit")
    _write_run(root, "BNCI2014_004", 2, 0, "harm")
    return build_summary([root], kw.get("budgets", [8, 16, "full"]), kw.get("taus", [0.0]),
                         kw.get("repeats", 30), batch=kw.get("batch", 8), seed=0)


def _cell(s, policy, budget, tau=0.0):
    return next(c for c in s["cells"] if c["policy"] == policy and c["budget"] == str(budget) and c["tau"] == tau)


def test_sequential_policy_uses_no_labels_at_k0_or_budget0():
    checks = [(8, 0.5, 0.3, 0.7), (16, 0.5, 0.4, 0.6)]
    for p in _SEQ_POLICIES:
        assert _seq_decide(p, checks, 0.0, cap=0) == ("abstain", 0)     # no labels -> abstain


def test_sequential_policy_stops_when_ci_decisive():
    checks = [(8, 0.5, 0.3, 0.7), (16, 0.5, 0.4, 0.6), (24, 0.5, 0.45, 0.55)]
    action, labels = _seq_decide("seq_ci_adapt_only", checks, tau=0.0, cap=24)
    assert action == "adapt" and labels == 8                            # stops at first decisive batch


def test_sequential_budget_full_uses_all_target_labels_but_marked_oracle_like_evaluation_boundary():
    with tempfile.TemporaryDirectory() as root:
        c = _cell(_summary(root), "seq_ci_three_way", "full")
        assert c["calibration_burden"] == "full" and c["deployable"] is True   # deployable, not oracle


def test_oracle_full_target_not_deployable():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert "oracle_full_target" not in _SEQ_POLICIES
        assert s["oracle_policy_selected_as_deployable"] is False
        assert "oracle_reference" in s and s["oracle_reference"]["note"].endswith("NOT deployable")


def test_best_policy_rule_excludes_oracle():
    with tempfile.TemporaryDirectory() as root:
        b = _summary(root)["best_sequential_policy"]
        assert b.get("policy") in ([None] + _SEQ_POLICIES)             # never oracle_full_target


def test_best_policy_requires_harm_constraint_and_min_coverage():
    # a cell with harm>0.05 or coverage<0.05 must not be selected
    cells = [{"policy": "seq_ci_three_way", "budget": "64", "tau": 0.0, "deployable": True,
              "adaptation_coverage": 0.5, "meets_harm_constraint_0_05": False, "mean_labels_used": 20,
              "harm_rate_among_adapt_decisions": 0.2, "missed_benefit_rate": 0.1},
             {"policy": "seq_ci_adapt_only", "budget": "64", "tau": 0.0, "deployable": True,
              "adaptation_coverage": 0.02, "meets_harm_constraint_0_05": True, "mean_labels_used": 20,
              "harm_rate_among_adapt_decisions": 0.0, "missed_benefit_rate": 0.9}]
    assert _select_best(cells)["policy"] is None                       # neither qualifies


def test_summary_reports_mean_labels_used_and_coverage():
    with tempfile.TemporaryDirectory() as root:
        c = _cell(_summary(root), "seq_ci_adapt_only", 16)
        for key in ("mean_labels_used", "median_labels_used", "adaptation_coverage",
                    "abstention_rate", "harm_rate_among_adapt_decisions"):
            assert key in c


def test_no_claim_r1_gain_identifiability():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert "not r1 target-gain identifiability" in s["claim_boundary"].lower()
        assert s["r2_iid_sampling_contract_required"] is True


ALL_TESTS = [
    test_sequential_policy_uses_no_labels_at_k0_or_budget0,
    test_sequential_policy_stops_when_ci_decisive,
    test_sequential_budget_full_uses_all_target_labels_but_marked_oracle_like_evaluation_boundary,
    test_oracle_full_target_not_deployable,
    test_best_policy_rule_excludes_oracle,
    test_best_policy_requires_harm_constraint_and_min_coverage,
    test_summary_reports_mean_labels_used_and_coverage,
    test_no_claim_r1_gain_identifiability,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} SEQUENTIAL-HARM-CONTROL TESTS PASSED")


if __name__ == "__main__":
    run()
