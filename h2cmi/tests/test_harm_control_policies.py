"""Project A Step 15 — tests for coverage-aware harm-control policies.

Checks the policy decision boundaries (k=0 abstains, CI gates, identity-default), that the oracle
policy is evaluation-only and never deployable, and that the summary separates coverage from harm and
never claims R1 gain identifiability. Run:

    python -m h2cmi.tests.test_harm_control_policies
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.observability.harm_control import _DEPLOYABLE, _decide, build_summary


def _write_run(root, dataset, target, seed, kind):
    # harmful: identity correct, adapt wrong 70/100 -> d mean -0.7 ; beneficial: the reverse
    y = [0] * 100
    if kind == "harm":
        idp, adp = [0] * 100, [1] * 70 + [0] * 30
    else:
        idp, adp = [1] * 70 + [0] * 30, [0] * 100
    d = Path(root) / f"dataset={dataset}_target={target}_seed={seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw_results.json").write_text(json.dumps({"per_trial_oracle_predictions": {
        "y_true": y, "identity_pred": idp, "adapt_pred": adp}}))
    (d / "run_manifest.json").write_text(json.dumps(
        {"status": "ok", "dataset": dataset, "target_subject": target, "seed": seed, "n_classes": 2}))
    return d


def _summary(root, **kw):
    _write_run(root, "BNCI2014_001", 1, 0, "harm")
    _write_run(root, "BNCI2014_004", 2, 0, "benefit")
    return build_summary([root], kw.get("ks", [0, 8, 32]), kw.get("taus", [0.0]),
                         kw.get("repeats", 40), seed=0)


def _cell(s, policy, k, tau=0.0):
    return next(c for c in s["cells"] if c["policy"] == policy and c["k"] == k and c["tau"] == tau)


def test_k0_policy_reports_r1_nonidentifiable():
    # a label-based policy cannot adapt at k=0 (R1 non-identifiable) -> it abstains
    assert _decide("plugin_sign", 0, 0.5, 0.4, 0.6, 0.0, full_gain=0.5) == "abstain"
    assert _decide("ci_three_way", 0, 0.5, 0.4, 0.6, 0.0, full_gain=0.5) == "abstain"
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        c = _cell(s, "ci_three_way", 0)
        assert c["adaptation_coverage"] == 0.0 and c["abstention_rate"] == 1.0


def test_oracle_full_target_policy_marked_evaluation_only():
    assert "oracle_full_target" not in _DEPLOYABLE
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert s["oracle_policy_selected_as_deployable"] is False
        assert s["best_deployable_policy"].get("policy") != "oracle_full_target"


def test_ci_policy_abstains_when_ci_crosses_zero():
    assert _decide("ci_three_way", 8, 0.01, -0.1, 0.2, 0.0, full_gain=0.0) == "abstain"


def test_ci_policy_adapts_only_when_lower_bound_exceeds_tau():
    assert _decide("ci_adapt_only_abstain", 8, 0.3, 0.05, 0.5, tau=0.02, full_gain=0.3) == "adapt"
    assert _decide("ci_adapt_only_abstain", 8, 0.01, -0.01, 0.1, tau=0.02, full_gain=0.01) == "abstain"


def test_identity_default_policy_prevents_adaptation_harm_but_misses_benefit():
    # not decisive -> identity (safety-first): no adaptation harm, but misses benefit
    assert _decide("ci_adapt_only_identity", 8, 0.0, -0.2, 0.2, tau=0.0, full_gain=-0.5) == "identity"
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        c = _cell(s, "ci_adapt_only_identity", 32)
        assert (c["harm_rate_among_adapt_decisions"] in (0.0, None))  # never adapts a harmful cell
        assert c["missed_benefit_rate"] is not None                   # some benefit missed / measured


def test_policy_summary_reports_coverage_and_harm_separately():
    with tempfile.TemporaryDirectory() as root:
        c = _cell(_summary(root), "always_adapt", 32)
        for key in ("adaptation_coverage", "decision_coverage", "abstention_rate",
                    "harm_rate_among_adapt_decisions", "prevented_harm_rate_vs_always_adapt",
                    "missed_benefit_rate"):
            assert key in c
        assert c["adaptation_coverage"] == 1.0 and c["abstention_rate"] == 0.0


def test_harm_control_never_claims_r1_gain_identifiability():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        cb = s["claim_boundary"].lower()
        assert "not r1 target-gain identifiability" in cb and s["r2_iid_sampling_contract_required"] is True


def test_oracle_labels_used_only_in_r2_slice_or_evaluation():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert "evaluation" in s["claim_boundary"].lower()
        assert s["oracle_policy_selected_as_deployable"] is False
        # ci_three_way should recover the safe policy: adapt the beneficial run, identity the harmful
        c = _cell(s, "ci_three_way", 32)
        assert c["harm_rate_among_adapt_decisions"] in (0.0, None) and c["adaptation_coverage"] > 0.0


ALL_TESTS = [
    test_k0_policy_reports_r1_nonidentifiable,
    test_oracle_full_target_policy_marked_evaluation_only,
    test_ci_policy_abstains_when_ci_crosses_zero,
    test_ci_policy_adapts_only_when_lower_bound_exceeds_tau,
    test_identity_default_policy_prevents_adaptation_harm_but_misses_benefit,
    test_policy_summary_reports_coverage_and_harm_separately,
    test_harm_control_never_claims_r1_gain_identifiability,
    test_oracle_labels_used_only_in_r2_slice_or_evaluation,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} HARM-CONTROL TESTS PASSED")


if __name__ == "__main__":
    run()
