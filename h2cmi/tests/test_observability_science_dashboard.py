"""Project A Step 12 — tests for the science dashboard combiner.

Run:  python -m h2cmi.tests.test_observability_science_dashboard
"""
from __future__ import annotations

from h2cmi.observability.science_dashboard import build_dashboard, build_step13_dashboard

_HARM_TABLE = {"n_runs": 54, "harm_rate": 0.8333, "oracle_denylist": ["offline_tta_gain_bacc"]}
_HARM_PRED = {"feature_sets": {"R0_source_only": {"balanced_acc_harm_prediction": 0.55},
                               "R1_target_unlabeled": {"balanced_acc_harm_prediction": 0.58}},
              "r1_minus_r0_balanced_acc_delta": 0.03, "majority_baseline_balanced_acc": 0.5,
              "any_predictor_beats_majority_baseline": True, "verdict": "retrospective_signal_present",
              "n_minority_class": 9, "oracle_never_a_feature": True}
_PHASE = {"k0_status": "not_identified_R1", "best_k_overall": 8, "phase_transition_observed": True}
_MULTI = {"all_target_metrics_identifiable_null": True}


def test_dashboard_reports_real_harm_and_predictor():
    d = build_dashboard(_HARM_TABLE, _HARM_PRED, _PHASE, _MULTI)
    m = d["metrics"]
    assert m["n_real_runs"] == 54 and m["real_harm_rate"] == 0.8333
    assert m["R0_harm_predictor_bacc"] == 0.55 and m["R1_harm_predictor_bacc"] == 0.58
    assert m["R0_to_R1_delta"] == 0.03 and m["minimal_paired_best_k"] == 8


def test_dashboard_claim_boundary_ok_requires_oracle_not_feature():
    assert build_dashboard(_HARM_TABLE, _HARM_PRED, _PHASE, _MULTI)["metrics"]["claim_boundary_ok"] is True
    # flip oracle_never_a_feature -> claim boundary fails
    bad = dict(_HARM_PRED, oracle_never_a_feature=False)
    assert build_dashboard(_HARM_TABLE, bad, _PHASE, _MULTI)["metrics"]["claim_boundary_ok"] is False
    # a target metric marked identifiable also fails the boundary
    assert build_dashboard(_HARM_TABLE, _HARM_PRED, _PHASE,
                           {"all_target_metrics_identifiable_null": False})[
        "metrics"]["claim_boundary_ok"] is False


def test_dashboard_lists_learned_and_unknown():
    d = build_dashboard(_HARM_TABLE, _HARM_PRED, _PHASE, _MULTI)
    assert len(d["what_we_learned"]) >= 4 and len(d["what_remains_unknown"]) >= 3
    blob = " ".join(d["what_we_learned"]).lower()
    assert "retrospective" in blob and "not identif" in blob.replace("identifiability", "identif")


def test_dashboard_no_sota_claim():
    d = build_dashboard(_HARM_TABLE, _HARM_PRED, _PHASE, _MULTI)
    assert "not sota" in d["scope"].lower() or "no sota" in d["claim_boundary"].lower()


_HT13 = {"n_runs": 54, "harm_rate": 0.8333, "oracle_denylist": ["offline_tta_gain_bacc"],
         "runs": [{"r1_source": "instrumented_r1_diagnostics"}] * 54}
_HP13_NULL = {"feature_sets": {"R0_source_only": {"balanced_acc_harm_prediction": 0.30},
                               "R1_target_unlabeled": {"balanced_acc_harm_prediction": 0.45,
                                                       "perm_null_p95": 0.62, "perm_null_p99": 0.68,
                                                       "margin_over_perm_null_p95": -0.17}},
              "any_predictor_beats_majority_baseline": False, "oracle_never_a_feature": True}
_REAL = {"k0_status": "not_identified_R1", "best_k_for_0_8_unconditional": None,
         "best_k_for_0_8_conditional": 4, "oracle_labels_used_only_for_r2_slice_and_evaluation": True,
         "per_k": {"256": {"decisive_rate": 0.32, "conditional_accuracy_given_decisive": 0.99}}}


def test_step13_dashboard_reports_r1_availability_and_minimal_label_k():
    m = build_step13_dashboard(_HT13, _HP13_NULL, _REAL, _MULTI)["metrics"]
    assert m["r1_diagnostics_available_rate"] == 1.0
    assert m["real_minimal_label_k256_coverage"] == 0.32
    assert m["real_minimal_label_k256_conditional_accuracy"] == 0.99
    assert m["real_minimal_label_best_k_0_8_conditional"] == 4
    assert m["real_minimal_label_best_k_0_8_unconditional"] is None
    assert m["target_labels_used_in_r1_diagnostics"] is False and m["claim_boundary_ok"] is True


def test_step13_dashboard_stronger_null_when_r1_below_baseline():
    d = build_step13_dashboard(_HT13, _HP13_NULL, _REAL, _MULTI)
    assert d["metrics"]["R1_beats_baseline"] is False
    assert "stronger null" in d["what_changed_from_step12"][0].lower()
    bad_real = dict(_REAL, k0_status="oops")
    assert build_step13_dashboard(_HT13, _HP13_NULL, bad_real, _MULTI)["metrics"]["claim_boundary_ok"] is False


def test_step14_dashboard_uses_harm_power_and_coverage_decomposition():
    power = {"underpowered": True, "minimum_detectable_bacc_approx": 0.67}
    d = build_step13_dashboard(_HT13, _HP13_NULL, _REAL, _MULTI, harm_power=power, step_label="Step 14")
    assert d["step"] == "Step 14"
    m = d["metrics"]
    assert m["harm_power_underpowered"] is True and m["minimum_detectable_bacc_approx"] == 0.67
    assert m["R1_perm_null_p99"] == 0.68


def test_dashboard_does_not_use_deprecated_harm_sign_accuracy():
    m = build_step13_dashboard(_HT13, _HP13_NULL, _REAL, _MULTI)["metrics"]
    assert not any("harm_sign_accuracy" in k for k in m)


ALL_TESTS = [
    test_dashboard_reports_real_harm_and_predictor,
    test_dashboard_claim_boundary_ok_requires_oracle_not_feature,
    test_dashboard_lists_learned_and_unknown,
    test_dashboard_no_sota_claim,
    test_step13_dashboard_reports_r1_availability_and_minimal_label_k,
    test_step13_dashboard_stronger_null_when_r1_below_baseline,
    test_step14_dashboard_uses_harm_power_and_coverage_decomposition,
    test_dashboard_does_not_use_deprecated_harm_sign_accuracy,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} SCIENCE-DASHBOARD TESTS PASSED")


if __name__ == "__main__":
    run()
