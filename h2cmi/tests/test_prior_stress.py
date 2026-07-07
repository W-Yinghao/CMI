"""Project A Step 18 — tests for deployment-prior stress.

Run:  python -m h2cmi.tests.test_prior_stress
"""
from __future__ import annotations

from h2cmi.observability.prior_stress import build_summary, prior_weighted_gain
from h2cmi.observability.audit import check_claim_allowed
from h2cmi.observability.schema import Claim, ContractID as C, Estimand, Regime


def _run(dataset, deltas, target=1, seed=0):
    # a fake harm_mechanisms run: per_class recall_delta only (what prior_stress consumes)
    per_class = {str(i): {"recall_delta": d} for i, d in enumerate(deltas)}
    return {"dataset": dataset, "target_subject": target, "seed": seed,
            "per_class": per_class, "transition_matrix_class_order": list(range(len(deltas)))}


def _hm(*runs):
    return {"runs": list(runs)}


def test_prior_weighted_gain_is_dot_product_of_prior_and_class_deltas():
    prior, deltas = [0.2, 0.3, 0.5], [-0.1, 0.0, 0.4]
    assert abs(prior_weighted_gain(prior, deltas) - (0.2 * -0.1 + 0.3 * 0.0 + 0.5 * 0.4)) < 1e-12


def test_uniform_prior_equals_bacc_gain():
    deltas = [0.1, -0.3, 0.2, 0.0]
    r = build_summary(_hm(_run("D", deltas)))["runs"][0]
    assert abs(r["uniform_gain"] - sum(deltas) / len(deltas)) < 1e-9      # uniform == mean == bAcc gain


def test_prior_dependent_sign_detected_when_min_negative_max_positive():
    r = build_summary(_hm(_run("D", [-0.2, 0.3])))["runs"][0]
    assert r["prior_dependent_sign"] is True
    assert r["min_prior_gain"] < 0 < r["max_prior_gain"]
    assert r["harmful_under_all_priors"] is False and r["beneficial_under_all_priors"] is False


def test_harmful_under_all_priors_when_all_class_deltas_nonpositive():
    r = build_summary(_hm(_run("D", [-0.1, -0.3, 0.0])))["runs"][0]     # all <= 0, some < 0
    assert r["harmful_under_all_priors"] is True
    assert r["prior_dependent_sign"] is False and r["beneficial_under_all_priors"] is False


def test_beneficial_under_all_priors_when_all_class_deltas_nonnegative():
    r = build_summary(_hm(_run("D", [0.1, 0.3, 0.0])))["runs"][0]
    assert r["beneficial_under_all_priors"] is True and r["harmful_under_all_priors"] is False


def test_uniform_harm_but_some_prior_benefit_flagged():
    # mean negative but one class strongly positive -> a skewed prior could benefit
    r = build_summary(_hm(_run("D", [-0.4, -0.4, 0.5])))["runs"][0]
    assert r["uniform_gain"] < 0 and r["uniform_harm_but_some_prior_benefit"] is True


def test_prior_stress_requires_c14_for_deployment_prior_claims():
    # the estimand is gated by the audit: without C14 (and without TU-1) a prior_weighted_gain is rejected
    bare = check_claim_allowed(Claim("pwg", Regime.R2, Estimand.PRIOR_WEIGHTED_GAIN, oracle=True))
    assert bare.rejected and C.C14 in bare.missing_contracts
    ok = check_claim_allowed(Claim("pwg-c14", Regime.R2, Estimand.PRIOR_WEIGHTED_GAIN,
                                   contracts={C.C14}, oracle=True))
    assert ok.allowed
    s = build_summary(_hm(_run("D", [-0.2, 0.3])))
    assert s["prior_contract_required"] == "C14"


def test_prior_stress_does_not_claim_actual_target_prior_identified():
    s = build_summary(_hm(_run("D", [-0.2, 0.3])))
    assert s["deployment_prior_identified"] is False
    assert s["deployment_prior_identified_under_R1"] is False
    assert "not identif" in s["claim_boundary"].lower() or "not identified" in s["claim_boundary"].lower()


ALL_TESTS = [
    test_prior_weighted_gain_is_dot_product_of_prior_and_class_deltas,
    test_uniform_prior_equals_bacc_gain,
    test_prior_dependent_sign_detected_when_min_negative_max_positive,
    test_harmful_under_all_priors_when_all_class_deltas_nonpositive,
    test_beneficial_under_all_priors_when_all_class_deltas_nonnegative,
    test_uniform_harm_but_some_prior_benefit_flagged,
    test_prior_stress_requires_c14_for_deployment_prior_claims,
    test_prior_stress_does_not_claim_actual_target_prior_identified,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} PRIOR-STRESS TESTS PASSED")


if __name__ == "__main__":
    run()
