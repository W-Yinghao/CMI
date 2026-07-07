"""Project A — anti-overclaim unit tests for the observability audit layer.

These are not formal box-ticking: each asserts that a *forbidden* target claim is actually
REJECTED and a *licensed* claim is ACCEPTED under the OACI rules. Run:

    python -m h2cmi.tests.test_observability_audit      (or: pytest h2cmi/tests/test_observability_audit.py)
"""
from __future__ import annotations

import json
import tempfile

from h2cmi.observability import (Claim, ContractID as C, Estimand, ForbiddenClaimViolation,
                                 REQUIRED_CLAIM_FIELDS, Regime, assert_forbidden_claims_not_made,
                                 build_report, check_claim_allowed, check_monotone_checkability,
                                 write_observability_report_json)


def test_registry_monotone_checkability():
    # MONO-1 invariant: no contract is checkable in R0 but not in R1/R2.
    assert check_monotone_checkability(), "contract checkability must be monotone R0->R1->R2"


def test_r0_source_loso_allowed():
    v = check_claim_allowed(Claim("src-loso", Regime.R0, Estimand.SOURCE_LOSO,
                                  observed=("X_s", "Y_s", "D_s")))
    assert v.allowed, "source LOSO is a source-side quantity, identifiable under R0"


def test_r0_target_gain_rejected_with_ce_r0_2():
    v = check_claim_allowed(Claim("r0-gain", Regime.R0, Estimand.TARGET_GAIN))
    assert v.rejected, "target gain must be non-identifiable under R0 (TOS-1)"
    assert v.failure_certificate == "CE-R0-2"
    # ...but an explicitly declared target-law axiom is a different (labelled) thing
    v_axiom = check_claim_allowed(Claim("r0-gain-axiom", Regime.R0, Estimand.TARGET_GAIN,
                                        target_law_axiom=True))
    assert v_axiom.allowed and "axiom" in v_axiom.reason


def test_r1_target_prior_requires_c1_c2_c3():
    full = check_claim_allowed(Claim("r1-prior", Regime.R1, Estimand.TARGET_PRIOR,
                                     contracts={C.C1, C.C2, C.C3}, observed=("X_T",)))
    assert full.allowed and full.theorem == "TU-1", "prior identifiable under C1∧C2∧C3 (TU-1)"
    partial = check_claim_allowed(Claim("r1-prior-missing", Regime.R1, Estimand.TARGET_PRIOR,
                                        contracts={C.C1, C.C2}, observed=("X_T",)))
    assert partial.rejected and C.C3 in partial.missing_contracts
    assert partial.failure_certificate == "CE-R1-2"


def test_r1_target_concept_rejected_with_ce_r1_1():
    v = check_claim_allowed(Claim("r1-concept", Regime.R1, Estimand.TARGET_CONCEPT,
                                  observed=("X_T",)))
    assert v.rejected, "concept is non-identifiable from unlabeled target (TU-2)"
    assert v.failure_certificate == "CE-R1-1"
    # even the R2 path is rejected without C4/C5/C6 + anchors
    v_r2_bare = check_claim_allowed(Claim("r2-concept-bare", Regime.R2, Estimand.TARGET_CONCEPT))
    assert v_r2_bare.rejected


def test_r2_transport_requires_c8_c11():
    ok = check_claim_allowed(Claim("r2-transport", Regime.R2, Estimand.TARGET_TRANSPORT,
                                   contracts={C.C8, C.C11}, observed=("anchors",)))
    assert ok.allowed and ok.theorem == "MP-1"
    missing = check_claim_allowed(Claim("r2-transport-missing", Regime.R2,
                                        Estimand.TARGET_TRANSPORT, contracts={C.C8}))
    assert missing.rejected and C.C11 in missing.missing_contracts
    assert missing.failure_certificate == "CE-C11-1"
    # transport is not even attemptable outside R2
    not_r2 = check_claim_allowed(Claim("r1-transport", Regime.R1, Estimand.TARGET_TRANSPORT,
                                       contracts={C.C8, C.C11}))
    assert not_r2.rejected and not_r2.failure_certificate == "CE-MP-1"


def test_leakage_not_accuracy_guarantee():
    v = check_claim_allowed(Claim("leak", Regime.R0, Estimand.LEAKAGE, observed=("Z_s",)))
    assert v.allowed and v.is_diagnostic, "leakage is allowed only as a diagnostic"
    assert not v.licenses_target_risk and "guarantee" in v.reason
    # you cannot launder leakage into a target risk claim under R0
    risk = check_claim_allowed(Claim("leak->risk", Regime.R0, Estimand.TARGET_RISK,
                                     observed=("Z_s",)))
    assert risk.rejected, "leakage must not be convertible into a target-risk guarantee under R0"


def test_balanced_accuracy_reporting_rules():
    r0 = check_claim_allowed(Claim("bacc-r0", Regime.R0, Estimand.BALANCED_ACCURACY))
    assert r0.allowed, "R0 bAcc is a source validation metric"
    r1 = check_claim_allowed(Claim("bacc-r1", Regime.R1, Estimand.BALANCED_ACCURACY))
    assert r1.rejected, "R1 unlabeled-target bAcc is not identifiable"
    r1_oracle = check_claim_allowed(Claim("bacc-r1-oracle", Regime.R1,
                                          Estimand.BALANCED_ACCURACY, oracle=True))
    assert r1_oracle.allowed, "R1 bAcc allowed only when explicitly oracle/eval-marked"
    r2 = check_claim_allowed(Claim("bacc-r2", Regime.R2, Estimand.BALANCED_ACCURACY,
                                   has_target_labels=True))
    assert r2.allowed, "R2 bAcc allowed with held-out target labels"
    r2_bare = check_claim_allowed(Claim("bacc-r2-bare", Regime.R2, Estimand.BALANCED_ACCURACY))
    assert r2_bare.rejected and r2_bare.failure_certificate == "CE-R0-2", \
        "R2 bAcc without labels/oracle must be rejected"


def test_forbidden_claim_guard_raises():
    # a rejected claim finalised as a conclusion must raise
    bad = build_report("bad", [Claim("overclaim", Regime.R0, Estimand.TARGET_GAIN)])
    raised = False
    try:
        assert_forbidden_claims_not_made(bad)
    except ForbiddenClaimViolation:
        raised = True
    assert raised, "a rejected conclusion must trip the forbidden-claim guard"
    # ...but the same rejected claim marked as a demonstration (conclusion=False) is exempt
    demo = build_report("demo", [Claim("demo", Regime.R0, Estimand.TARGET_GAIN, conclusion=False)])
    assert_forbidden_claims_not_made(demo)   # no raise


def test_report_contains_required_08_fields():
    report = build_report("fields", [
        Claim("src", Regime.R0, Estimand.SOURCE_LOSO, observed=("X_s", "Y_s")),
        Claim("prior", Regime.R1, Estimand.TARGET_PRIOR, contracts={C.C1, C.C2, C.C3},
              observed=("X_T",)),
        # a rejected DEMO (conclusion=False) -> clean report, but exercises the rejected record
        Claim("concept_demo", Regime.R1, Estimand.TARGET_CONCEPT, conclusion=False),
        # allowed BUT NOT identifiable (oracle/eval-only) -> pins identifiable keyed off .identifiable
        Claim("oracle_bacc", Regime.R0, Estimand.BALANCED_ACCURACY,
              observed=("X_s", "Y_s", "heldout_target_labels"), oracle=True),
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
    data = write_observability_report_json(report, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == data
    assert "forbidden_claims_checked" in loaded and loaded["forbidden_claims_violated"] == []
    for rec in loaded["claims"]:
        for fld in REQUIRED_CLAIM_FIELDS:
            assert fld in rec, f"claim record missing required 08 §5 field: {fld}"
    # identifiable_estimand: the estimand iff allowed, else None (no machine-readable overclaim)
    by_name = {r["name"]: r for r in loaded["claims"]}
    assert by_name["src"]["identifiable_estimand"] == "source_loso"
    assert by_name["prior"]["identifiable_estimand"] == "target_prior"
    assert by_name["concept_demo"]["identifiable_estimand"] is None
    assert by_name["concept_demo"]["allowed"] is False
    # allowed oracle metric: reportable but NOT identifiable (keyed off .identifiable, not .allowed)
    assert by_name["oracle_bacc"]["allowed"] is True
    assert by_name["oracle_bacc"]["reportable_metric"] is True
    assert by_name["oracle_bacc"]["identifiable_estimand"] is None


def test_forbidden_violation_serialized():
    # the NON-EMPTY forbidden-violation path: a rejected claim finalised as a conclusion.
    report = build_report("dirty", [Claim("over", Regime.R0, Estimand.TARGET_GAIN)])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
    data = write_observability_report_json(report, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["forbidden_claims_violated"] == ["over"], \
        "a rejected conclusion must be recorded in forbidden_claims_violated"
    assert loaded["claims"][0]["allowed"] is False
    assert loaded["claims"][0]["identifiable_estimand"] is None


def test_r0_target_prior_rejected_ce_r0_3():
    v = check_claim_allowed(Claim("r0-prior", Regime.R0, Estimand.TARGET_PRIOR))
    assert v.rejected and v.failure_certificate == "CE-R0-3"


def test_target_prior_axiom_monotone():
    # a declared target-law axiom licenses target prior in EVERY regime (MONO-1: R2 ⊒ R0)
    r0 = check_claim_allowed(Claim("axiom-r0", Regime.R0, Estimand.TARGET_PRIOR,
                                   target_law_axiom=True))
    r2 = check_claim_allowed(Claim("axiom-r2", Regime.R2, Estimand.TARGET_PRIOR,
                                   target_law_axiom=True))
    assert r0.allowed and r2.allowed, "declared axiom must not flip allowed->rejected R0->R2"
    assert "axiom" in r0.reason and "axiom" in r2.reason


def test_r2_concept_bounded_residual_allowed():
    # R2 with C4(tested)∧C5∧C6 + anchors -> bounded/tested residual, NOT 'concept detected'
    ok = check_claim_allowed(Claim("r2-concept", Regime.R2, Estimand.TARGET_CONCEPT,
                                   contracts={C.C4, C.C5, C.C6}, has_anchors=True))
    assert ok.allowed and "bounded/tested" in ok.reason
    # missing C6 -> rejected
    miss = check_claim_allowed(Claim("r2-concept-miss", Regime.R2, Estimand.TARGET_CONCEPT,
                                     contracts={C.C4, C.C5}, has_anchors=True))
    assert miss.rejected and C.C6 in miss.missing_contracts
    # C4∧C5∧C6 but NO anchors/labels -> rejected
    no_anchor = check_claim_allowed(Claim("r2-concept-noanchor", Regime.R2,
                                          Estimand.TARGET_CONCEPT, contracts={C.C4, C.C5, C.C6}))
    assert no_anchor.rejected


def test_c14_registered_as_deployment_prior_contract():
    from h2cmi.observability.registry import CONTRACTS
    assert C.C14 in CONTRACTS
    spec = CONTRACTS[C.C14]
    assert "deployment prior" in spec.name.lower() or "utility" in spec.name.lower()
    # declared/external: not source-supported (R0=no), partial at R1, validatable at R2 (monotone)
    assert spec.checkable[Regime.R0] == "no"
    assert check_monotone_checkability()               # C14 must not break MONO-1


def test_c14_does_not_make_target_prior_identifiable_under_r1():
    # a target_prior claim carrying ONLY C14 (no C1∧C2∧C3) is still rejected under R1 — C14 is a
    # declared operating prior, NOT an identification of the actual target prior (must not become TU-1).
    v = check_claim_allowed(Claim("r1-prior-c14", Regime.R1, Estimand.TARGET_PRIOR,
                                  contracts={C.C14}))
    assert v.rejected and not v.identifiable
    assert C.C1 in v.missing_contracts                 # still needs TU-1, C14 does not substitute


def test_prior_weighted_gain_claim_requires_declared_prior_or_tu1():
    # no declared prior and no TU-1 -> rejected
    bare = check_claim_allowed(Claim("pwg-bare", Regime.R2, Estimand.PRIOR_WEIGHTED_GAIN,
                                     oracle=True))
    assert bare.rejected and C.C14 in bare.missing_contracts
    # declared deployment prior (C14) -> allowed as a counterfactual scenario, NOT identifiable
    declared = check_claim_allowed(Claim("pwg-c14", Regime.R2, Estimand.PRIOR_WEIGHTED_GAIN,
                                         contracts={C.C14}, oracle=True))
    assert declared.allowed and not declared.identifiable and "counterfactual" in declared.reason
    # identified target prior (TU-1) -> allowed under TU-1
    tu1 = check_claim_allowed(Claim("pwg-tu1", Regime.R2, Estimand.PRIOR_WEIGHTED_GAIN,
                                    contracts={C.C1, C.C2, C.C3}, has_target_labels=True))
    assert tu1.allowed and tu1.theorem == "TU-1"


def test_c15_registered_as_prior_uncertainty_contract():
    from h2cmi.observability.registry import CONTRACTS
    assert C.C15 in CONTRACTS
    name = CONTRACTS[C.C15].name.lower()
    assert "prior-uncertainty" in name or "robustness" in name
    assert CONTRACTS[C.C15].checkable[Regime.R0] == "no"
    assert check_monotone_checkability()               # C15 must not break MONO-1


def test_c15_does_not_identify_actual_target_prior_under_r1():
    # a target_prior claim carrying only C15 is still rejected under R1 — C15 declares an uncertainty
    # SET, it does not identify the actual target prior (must not become TU-1).
    v = check_claim_allowed(Claim("r1-prior-c15", Regime.R1, Estimand.TARGET_PRIOR, contracts={C.C15}))
    assert v.rejected and not v.identifiable and C.C1 in v.missing_contracts


def test_robust_prior_weighted_gain_requires_c15():
    bare = check_claim_allowed(Claim("rpwg-bare", Regime.R2, Estimand.ROBUST_PRIOR_WEIGHTED_GAIN,
                                     oracle=True))
    assert bare.rejected and C.C15 in bare.missing_contracts
    ok = check_claim_allowed(Claim("rpwg-c15", Regime.R2, Estimand.ROBUST_PRIOR_WEIGHTED_GAIN,
                                   contracts={C.C15}, oracle=True))
    assert ok.allowed and not ok.identifiable and "counterfactual" in ok.reason


def test_c14_point_prior_does_not_imply_c15_robustness():
    # a declared POINT prior (C14) does not certify robustness over a SET — robust gain still needs C15.
    v = check_claim_allowed(Claim("rpwg-c14only", Regime.R2, Estimand.ROBUST_PRIOR_WEIGHTED_GAIN,
                                  contracts={C.C14}, oracle=True))
    assert v.rejected and C.C15 in v.missing_contracts


ALL_TESTS = [
    test_registry_monotone_checkability,
    test_c14_registered_as_deployment_prior_contract,
    test_c14_does_not_make_target_prior_identifiable_under_r1,
    test_prior_weighted_gain_claim_requires_declared_prior_or_tu1,
    test_c15_registered_as_prior_uncertainty_contract,
    test_c15_does_not_identify_actual_target_prior_under_r1,
    test_robust_prior_weighted_gain_requires_c15,
    test_c14_point_prior_does_not_imply_c15_robustness,
    test_r0_source_loso_allowed,
    test_r0_target_gain_rejected_with_ce_r0_2,
    test_r1_target_prior_requires_c1_c2_c3,
    test_r0_target_prior_rejected_ce_r0_3,
    test_target_prior_axiom_monotone,
    test_r1_target_concept_rejected_with_ce_r1_1,
    test_r2_concept_bounded_residual_allowed,
    test_r2_transport_requires_c8_c11,
    test_leakage_not_accuracy_guarantee,
    test_balanced_accuracy_reporting_rules,
    test_forbidden_claim_guard_raises,
    test_report_contains_required_08_fields,
    test_forbidden_violation_serialized,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} OBSERVABILITY-AUDIT TESTS PASSED")


if __name__ == "__main__":
    run()
