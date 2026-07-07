"""Project A Step 20 — tests for the final closeout and claim ledger.

Run:  python -m h2cmi.tests.test_closeout
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.observability.closeout import build_closeout, _LADDER, _FINAL_VERDICT


def _fake_summaries(root):
    d = Path(root)
    (d / "step14_harm_predictor_summary.json").write_text(json.dumps(
        {"feature_sets": {"R0_source_only": {"balanced_acc_harm_prediction": 0.334}},
         "verdict": "above_baseline_but_within_permutation_null_overfitting_artifact"}))
    (d / "step14_real_minimal_label_curves.json").write_text(json.dumps({"k0_status": "not_identified_R1"}))
    (d / "step15_harm_control_summary.json").write_text(json.dumps({"best_deployable_policy": {"policy": None}}))
    (d / "step16_benefit_anatomy.json").write_text(json.dumps({"benefit_rate": 0.0926}))
    (d / "step17_estimand_consistency.json").write_text(json.dumps({"max_abs_gain_difference": 0.0}))
    (d / "step18_prior_stress.json").write_text(json.dumps({"fraction_prior_dependent_sign": 0.963}))
    (d / "step19_prior_robust_policy.json").write_text(json.dumps(
        {"robust_prior_safe_adaptation_exists_any": False}))
    return d


def test_evidence_ledger_pulls_live_metrics_from_digests():
    with tempfile.TemporaryDirectory() as root:
        _fake_summaries(root)
        s = build_closeout(root)
        by_step = {e["step"]: e for e in s["evidence_ledger"]}
        assert by_step["Step 12"]["headline_value"] == 0.334
        assert by_step["Step 17"]["headline_value"] == 0.0        # estimand-invariant
        assert by_step["Step 18"]["headline_value"] == 0.963
        assert by_step["Step 19"]["headline_value"] is False      # no robust safe adaptation
        assert all(e["verdict"] in ("REFUTES", "CHARACTERIZES", "NULL") for e in s["evidence_ledger"])


def test_no_ladder_rung_licenses_deployable_adaptation():
    s = build_closeout("/nonexistent")                            # ladder is static; digests optional
    assert s["no_ladder_rung_licenses_deployable_adaptation"] is True
    assert all(r["licenses_deployable_adaptation"] is False for r in _LADDER)
    # the oracle rung mentions adaptation but is evaluation-only, still not deployable
    oracle = [r for r in _LADDER if "oracle" in r["level"].lower()][0]
    assert oracle["licenses_deployable_adaptation"] is False


def test_forbidden_headline_claims_all_not_made():
    s = build_closeout("/nonexistent")
    assert s["forbidden_headline_claims_all_not_made"] is True
    assert all(f["made"] is False and f["status"] == "FORBIDDEN" for f in s["forbidden_claims"])
    claims = " ".join(f["claim"] for f in s["forbidden_claims"]).lower()
    assert "safe to deploy" in claims and "target prior is identified" in claims and "prior-robust" in claims


def test_terminal_machine_flags_are_all_false():
    s = build_closeout("/nonexistent")
    for k in ("tta_safe_to_deploy_claim", "target_prior_identified_from_r0_r1_claim",
              "prior_robust_benefit_exists_claim", "oracle_gain_ever_deployable",
              "prior_robust_safe_adaptation_certifiable_with_harm_margin"):
        assert s[k] is False, k
    # the only positive is explicitly flagged as a de-emphasised zero-margin sign-level artifact
    assert s["prior_robust_positive_is_zero_margin_sign_level_only"] is True


def test_closeout_registry_forbidden_claims_include_headline_three():
    s = build_closeout("/nonexistent")
    blob = " ".join(s["registry_forbidden_claims"]).lower()
    assert "unlabeled offline-tta is safe to deploy" in blob
    assert "target prior is identified from r0/r1" in blob
    assert "prior-robust adaptation benefit exists" in blob


def test_final_verdict_and_no_new_work_flags():
    s = build_closeout("/nonexistent")
    assert s["final_verdict"] == _FINAL_VERDICT
    assert "cannot be safely controlled under honest prior uncertainty" in s["final_verdict"].lower()
    assert s["no_new_data"] and s["no_retraining"] and s["no_new_rescue_policy"]
    assert s["claim_boundary_ok"] is True
    assert "no sota" in s["claim_boundary"].lower()


ALL_TESTS = [
    test_evidence_ledger_pulls_live_metrics_from_digests,
    test_no_ladder_rung_licenses_deployable_adaptation,
    test_forbidden_headline_claims_all_not_made,
    test_terminal_machine_flags_are_all_false,
    test_closeout_registry_forbidden_claims_include_headline_three,
    test_final_verdict_and_no_new_work_flags,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} CLOSEOUT TESTS PASSED")


if __name__ == "__main__":
    run()
