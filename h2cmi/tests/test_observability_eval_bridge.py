"""Project A — tests for the audited evaluation bridge (Step 6 acceptance gate).

Asserts that harness-style eval outputs become an ObservabilityReport in which every TARGET
metric is oracle/evaluation-only (reportable, identifiable=False), the regime<->observed
integrity check rejects mis-declared coordinates, an offline-TTA prior needs C1∧C2∧C3, and
leakage stays a diagnostic. Run:

    python -m h2cmi.tests.test_observability_eval_bridge
"""
from __future__ import annotations

from h2cmi.observability import (Claim, ContractID as C, Estimand, ForbiddenClaimViolation, Regime,
                                 assert_forbidden_claims_not_made, build_audited_eval_report,
                                 check_claim_allowed, claims_for_leakage, claims_for_offline_tta,
                                 claims_for_online_tta, claims_for_strict_dg, report_to_dict,
                                 validate_observed_coordinates)

# fake harness-style dicts (no training)
STRICT_DG = {"setting": "strict_dg", "balanced_acc": 0.72, "worst_domain_bacc": 0.61}
OFFLINE_TTA = {"identity": {"balanced_acc": 0.70}, "adapt": {"balanced_acc": 0.76},
               "delta_adapt": {"d_balanced_acc": 0.06},
               "selective_risk": {"coverage": 0.8, "avoided_harm": 0.03},
               "per_domain_pi_T": {"1": [0.48, 0.52]}}          # prior estimate present
OFFLINE_TTA_NO_PRIOR = {"delta_adapt": {"d_balanced_acc": 0.06}}  # gain but NO prior estimate
ONLINE_TTA = {"setting": "online_tta", "balanced_acc": 0.73}
LEAKAGE = {"site": {"I_hat": 0.12, "excess": 0.04}, "subject": {"I_hat": 0.30, "excess": 0.21}}


def test_r0_observed_integrity_rejects_target_coordinates():
    # a source-side estimand is normally allowed, but declaring R0 while observing target data fails
    v = check_claim_allowed(Claim("bad-r0", Regime.R0, Estimand.SOURCE_LOSO, observed=("X_T",)))
    assert v.rejected and "regime-observed mismatch" in v.reason
    assert validate_observed_coordinates(
        Claim("bad-r0", Regime.R0, Estimand.SOURCE_LOSO, observed=("X_T",)))


def test_r1_observed_integrity_rejects_target_labels_as_adaptation_data():
    v = check_claim_allowed(Claim("bad-r1", Regime.R1, Estimand.TARGET_PRIOR,
                                  contracts={C.C1, C.C2, C.C3}, observed=("X_T", "Y_T")))
    assert v.rejected and "regime-observed mismatch" in v.reason
    # anchors under R1 also rejected
    v2 = check_claim_allowed(Claim("bad-r1b", Regime.R1, Estimand.TARGET_PRIOR,
                                   contracts={C.C1, C.C2, C.C3}, observed=("X_T", "anchors")))
    assert v2.rejected and "anchors" in v2.reason


def test_r1_oracle_target_metric_reportable_but_not_identifiable():
    # oracle target labels for EVALUATION are permitted under R1 and reportable, but not identified
    v = check_claim_allowed(Claim("r1-oracle-bacc", Regime.R1, Estimand.BALANCED_ACCURACY,
                                  observed=("X_T", "heldout_target_labels"), oracle=True))
    assert v.allowed and v.reportable and not v.identifiable
    assert not v.licenses_target_risk


def test_strict_dg_target_bacc_is_oracle_evaluation_only():
    claims = claims_for_strict_dg(STRICT_DG, has_oracle_target_labels=True)
    assert claims, "strict-DG should emit target bAcc claims"
    for cl in claims:
        v = check_claim_allowed(cl)
        assert v.allowed and v.reportable and not v.identifiable, \
            "strict-DG target bAcc must be reportable oracle-only, not identifiable"
    # without eval labels there is no target metric at all
    assert claims_for_strict_dg(STRICT_DG, has_oracle_target_labels=False) == []


def test_offline_tta_gain_is_oracle_evaluation_only_not_identified():
    claims = claims_for_offline_tta(OFFLINE_TTA, has_oracle_target_labels=True,
                                    prior_contracts={C.C1, C.C2, C.C3})
    gain = [c for c in claims if c.estimand == Estimand.TARGET_GAIN]
    assert gain, "offline TTA should emit an adaptation-gain claim"
    v = check_claim_allowed(gain[0])
    assert v.allowed and v.reportable and not v.identifiable and not v.licenses_target_risk, \
        "measured TTA gain is oracle/eval-only, not an R1-identified gain"


def test_offline_tta_prior_claim_requires_c1_c2_c3_when_payload_present():
    # with C1∧C2∧C3 declared -> allowed under TU-1 (identifiable)
    with_c = claims_for_offline_tta(OFFLINE_TTA, prior_contracts={C.C1, C.C2, C.C3})
    prior = [c for c in with_c if c.estimand == Estimand.TARGET_PRIOR][0]
    v_ok = check_claim_allowed(prior)
    assert v_ok.allowed and v_ok.identifiable and v_ok.theorem == "TU-1"
    # without the contracts -> rejected (CE-R1-2)
    without_c = claims_for_offline_tta(OFFLINE_TTA, prior_contracts=set())
    prior2 = [c for c in without_c if c.estimand == Estimand.TARGET_PRIOR][0]
    v_bad = check_claim_allowed(prior2)
    assert v_bad.rejected and v_bad.failure_certificate == "CE-R1-2"


def test_offline_tta_prior_claim_not_emitted_without_prior_payload():
    # no prior estimate in the harness output -> NO target-prior claim is asserted at all
    claims = claims_for_offline_tta(OFFLINE_TTA_NO_PRIOR, prior_contracts={C.C1, C.C2, C.C3})
    assert not any(c.estimand == Estimand.TARGET_PRIOR for c in claims), \
        "a target-prior claim must not be emitted without a prior estimate in the output"
    # the gain claim is still emitted (delta_adapt present)
    assert any(c.estimand == Estimand.TARGET_GAIN for c in claims)


def test_metric_payload_serialized_without_changing_identifiability():
    # the metric payload is serialised but does NOT change the audit verdict
    v_no = check_claim_allowed(Claim("bacc", Regime.R0, Estimand.BALANCED_ACCURACY,
                                     observed=("X_s", "Y_s", "heldout_target_labels"), oracle=True))
    v_yes = check_claim_allowed(Claim("bacc", Regime.R0, Estimand.BALANCED_ACCURACY,
                                      observed=("X_s", "Y_s", "heldout_target_labels"), oracle=True,
                                      metric_payload={"balanced_acc": 0.99}))
    assert (v_no.allowed, v_no.identifiable, v_no.reportable) == \
           (v_yes.allowed, v_yes.identifiable, v_yes.reportable)
    from h2cmi.observability import build_report, report_to_dict
    data = report_to_dict(build_report("p", [
        Claim("m", Regime.R0, Estimand.BALANCED_ACCURACY,
              observed=("X_s", "heldout_target_labels"), oracle=True,
              metric_payload={"balanced_acc": 0.99})]))
    assert data["claims"][0]["metric_payload"] == {"balanced_acc": 0.99}


def test_oracle_target_metric_with_value_still_not_identifiable():
    # even with a concrete value attached, an oracle target metric is NOT identifiable
    v = check_claim_allowed(Claim("g", Regime.R1, Estimand.TARGET_GAIN, observed=("X_T",),
                                  oracle=True, metric_payload={"d_balanced_acc": 0.42}))
    assert v.allowed and v.reportable and not v.identifiable and not v.licenses_target_risk


def test_online_tta_target_bacc_is_oracle_evaluation_only():
    claims = claims_for_online_tta(ONLINE_TTA, has_oracle_target_labels=True)
    assert claims, "online TTA should emit a target bAcc claim"
    v = check_claim_allowed(claims[0])
    assert v.allowed and v.reportable and not v.identifiable and not v.licenses_target_risk
    # without eval labels there is no target metric
    assert claims_for_online_tta(ONLINE_TTA, has_oracle_target_labels=False) == []


def test_leakage_bridge_diagnostic_not_risk():
    for cl in claims_for_leakage(LEAKAGE):
        v = check_claim_allowed(cl)
        assert v.allowed and v.is_diagnostic and not v.licenses_target_risk
        assert cl.estimand == Estimand.LEAKAGE


def test_bridge_fail_loud_when_prior_contracts_undeclared():
    # the guarantee the bridge exists for: an offline-TTA prior WITHOUT C1∧C2∧C3 must fail loud.
    report = build_audited_eval_report("undeclared", offline_tta=OFFLINE_TTA)  # no prior_contracts
    data = report_to_dict(report)
    by_name = {r["name"]: r for r in data["claims"]}
    assert by_name["offline_tta.target_prior"]["allowed"] is False
    assert by_name["offline_tta.target_prior"]["identifiable_estimand"] is None
    assert data["forbidden_claims_violated"] == ["offline_tta.target_prior"]
    raised = False
    try:
        assert_forbidden_claims_not_made(report)
    except ForbiddenClaimViolation:
        raised = True
    assert raised, "undeclared-contract TTA prior must trip the forbidden-claim guard"


def test_bridge_report_has_zero_forbidden_violations():
    report = build_audited_eval_report(
        "bridge", strict_dg=STRICT_DG, offline_tta=OFFLINE_TTA, online_tta=ONLINE_TTA,
        leakage=LEAKAGE, prior_contracts={C.C1, C.C2, C.C3}, has_oracle_target_labels=True)
    data = report_to_dict(report)
    assert data["forbidden_claims_violated"] == [], \
        "a properly-declared eval report must have zero forbidden-claim violations"
    assert data["summary"]["n_claims"] >= 5


def test_bridge_report_identifiable_estimand_none_for_oracle_metrics():
    report = build_audited_eval_report(
        "bridge", strict_dg=STRICT_DG, offline_tta=OFFLINE_TTA, online_tta=ONLINE_TTA,
        leakage=LEAKAGE, prior_contracts={C.C1, C.C2, C.C3})
    data = report_to_dict(report)
    by_name = {r["name"]: r for r in data["claims"]}
    # oracle target metrics: reportable but identifiable_estimand must be None
    for nm in ("strict_dg.target_bacc", "offline_tta.adaptation_gain", "online_tta.target_bacc"):
        assert by_name[nm]["reportable_metric"] is True
        assert by_name[nm]["identifiable_estimand"] is None, \
            f"{nm} is oracle/eval-only and must not be reported as identifiable"
    # the TU-1 prior IS identifiable
    assert by_name["offline_tta.target_prior"]["identifiable_estimand"] == "target_prior"
    # leakage is an identifiable diagnostic, not a risk
    assert by_name["leakage.site"]["is_diagnostic"] is True


ALL_TESTS = [
    test_r0_observed_integrity_rejects_target_coordinates,
    test_r1_observed_integrity_rejects_target_labels_as_adaptation_data,
    test_r1_oracle_target_metric_reportable_but_not_identifiable,
    test_strict_dg_target_bacc_is_oracle_evaluation_only,
    test_offline_tta_gain_is_oracle_evaluation_only_not_identified,
    test_offline_tta_prior_claim_requires_c1_c2_c3_when_payload_present,
    test_offline_tta_prior_claim_not_emitted_without_prior_payload,
    test_metric_payload_serialized_without_changing_identifiability,
    test_oracle_target_metric_with_value_still_not_identifiable,
    test_online_tta_target_bacc_is_oracle_evaluation_only,
    test_leakage_bridge_diagnostic_not_risk,
    test_bridge_fail_loud_when_prior_contracts_undeclared,
    test_bridge_report_has_zero_forbidden_violations,
    test_bridge_report_identifiable_estimand_none_for_oracle_metrics,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} EVAL-BRIDGE TESTS PASSED")


if __name__ == "__main__":
    run()
