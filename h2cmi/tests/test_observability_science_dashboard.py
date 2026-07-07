"""Project A Step 12 — tests for the science dashboard combiner.

Run:  python -m h2cmi.tests.test_observability_science_dashboard
"""
from __future__ import annotations

from h2cmi.observability.science_dashboard import (build_dashboard, build_step13_dashboard,
                                                   build_step15_dashboard, build_step16_dashboard,
                                                   build_step17_dashboard, build_step18_dashboard,
                                                   build_step19_dashboard)

_BEN = {"n_runs": 54, "benefit_rate": 0.1481, "target_sign_consistency_rate": 0.83,
        "per_dataset": {"BNCI2014_001": {"n_beneficial": 4}, "BNCI2014_004": {"n_beneficial": 4}},
        "beneficial_gain_distribution_bacc": {"q90": 0.03},
        "claim_boundary": "oracle-only benefit anatomy; NOT deployment-observable under R0/R1."}
_SEQ16 = {"oracle_policy_selected_as_deployable": False, "r2_iid_sampling_contract_required": True,
          "best_sequential_policy": {"policy": None, "reason": "no policy"}}
_FR16 = {"oracle_excluded": True, "any_policy_meets_harm_0_05": False,
         "any_policy_meets_harm_0_1": True, "any_policy_meets_harm_0_2": True}

_HC = {"n_runs": 54, "always_adapt_harm_rate": 0.85, "harm_constraint": 0.05,
       "claim_boundary_ok": True, "r2_iid_sampling_contract_required": True,
       "oracle_policy_selected_as_deployable": False,
       "best_deployable_policy": {"policy": "ci_three_way", "k": 32, "tau": 0.02,
                                  "adaptation_coverage": 0.12, "decision_coverage": 0.35,
                                  "harm_rate_among_adapt_decisions": 0.03,
                                  "prevented_harm_rate_vs_always_adapt": 0.72, "missed_benefit_rate": 0.4}}

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


def test_step15_dashboard_reports_best_policy_and_coverage_tradeoff():
    d = build_step15_dashboard(_HC, real_curves=_REAL, multi=_MULTI)
    m = d["metrics"]
    assert d["step"] == "Step 15"
    assert m["best_policy_by_harm_control"] == "ci_three_way" and m["best_policy_k"] == 32
    assert m["best_policy_adaptation_coverage"] == 0.12 and m["best_policy_harm_rate_among_adapt"] == 0.03
    assert m["coverage_control_tradeoff_observed"] is True and m["claim_boundary_ok"] is True


def test_deprecated_ci_attempt_key_not_used_by_dashboard():
    m = build_step15_dashboard(_HC, real_curves=_REAL, multi=_MULTI)["metrics"]
    assert "best_label_based_attempt" in m
    assert not any("ci_attempt" in k for k in m)               # deprecated name not surfaced


def test_step15_dashboard_never_selects_oracle_policy():
    m = build_step15_dashboard(_HC, real_curves=_REAL, multi=_MULTI)["metrics"]
    assert m["oracle_policy_selected_as_deployable"] is False
    assert m["best_policy_by_harm_control"] != "oracle_full_target"
    # a summary that (wrongly) marks oracle deployable fails the claim boundary
    bad = dict(_HC, oracle_policy_selected_as_deployable=True)
    assert build_step15_dashboard(bad, real_curves=_REAL, multi=_MULTI)["metrics"]["claim_boundary_ok"] is False


def test_step16_dashboard_reports_benefit_and_sequential():
    d = build_step16_dashboard(_BEN, _SEQ16, _FR16, real_curves=_REAL, multi=_MULTI)
    m = d["metrics"]
    assert d["step"] == "Step 16" and m["benefit_rate"] == 0.1481
    assert m["best_sequential_policy"] is None
    assert m["safe_adaptation_requires_full_or_near_full_labels"] is True
    assert m["policy_frontier_harm_0_05_exists"] is False and m["policy_frontier_harm_0_10_exists"] is True
    assert m["claim_boundary_ok"] is True


def test_step16_dashboard_claim_boundary_fails_if_oracle_deployable():
    bad = dict(_SEQ16, oracle_policy_selected_as_deployable=True)
    m = build_step16_dashboard(_BEN, bad, _FR16, real_curves=_REAL, multi=_MULTI)["metrics"]
    assert m["claim_boundary_ok"] is False


# Scenario A — estimands IDENTICAL on grid (class-balanced targets); control only at k=full (oracle-eq).
_CON17_IDENT = {"n_runs": 54, "accuracy_benefit_rate": 0.1481, "bacc_benefit_rate": 0.1481,
                "cross_estimand_sign_agreement": 1.0, "runs_accuracy_benefit_bacc_harm": 0,
                "runs_bacc_benefit_accuracy_harm": 0, "accuracy_policy_controls_bacc": False,
                "estimand_relationship": "identical_on_grid", "estimands_identical_on_grid": True,
                "all_targets_class_balanced": True, "max_abs_gain_difference": 0.0,
                "estimand_gap_is_sign_disagreement": False, "estimand_gap_is_magnitude_only": False,
                "benefit_eps": 0.005, "accuracy_material_benefit_rate_eps": 0.0926,
                "bacc_material_benefit_rate_eps": 0.0926, "mean_accuracy_gain": -0.0423,
                "mean_bacc_gain": -0.0423, "step16_gap_explanation":
                    "On this grid all 54 target sets are class-balanced; accuracy-gain == balanced-"
                    "accuracy-gain per run (max |diff| = 0.0). The Step-16 gap was a THRESHOLD artifact.",
                "claim_boundary_ok": True}
_FR17_IDENT = {"no_overall_best_across_estimands": True, "accuracy_policy_controls_bacc": False,
               "groups": {
                   "accuracy_gain:iid": {"requires_contract": None, "any_policy_meets_harm_0_1": True,
                                         "best_under_harm_0_1": {"policy": "plugin_sign", "k": "full"}},
                   "balanced_accuracy_gain:class_balanced": {
                       "requires_contract": "C13", "any_policy_meets_harm_0_1": True,
                       "best_under_harm_0_1": {"policy": "plugin_sign", "k": "full"}}}}

# Scenario B — genuine SIGN disagreement; a minimal-label accuracy policy controls, bAcc has none.
_CON17_DISAGREE = dict(_CON17_IDENT, bacc_benefit_rate=0.0926, cross_estimand_sign_agreement=0.87,
                       runs_accuracy_benefit_bacc_harm=4, runs_bacc_benefit_accuracy_harm=1,
                       estimand_relationship="sign_disagreement", estimands_identical_on_grid=False,
                       all_targets_class_balanced=False, max_abs_gain_difference=0.12,
                       estimand_gap_is_sign_disagreement=True, step16_gap_explanation="")
_FR17_DISAGREE = {"no_overall_best_across_estimands": True, "accuracy_policy_controls_bacc": False,
                  "groups": {
                      "accuracy_gain:iid": {"requires_contract": None, "any_policy_meets_harm_0_1": True,
                                            "best_under_harm_0_1": {"policy": "ci_three_way", "k": 32}},
                      "balanced_accuracy_gain:class_balanced": {
                          "requires_contract": "C13", "any_policy_meets_harm_0_1": False,
                          "best_under_harm_0_1": None}}}


def test_step17_dashboard_separates_estimands_and_flags_relationship():
    d = build_step17_dashboard(_CON17_IDENT, _FR17_IDENT)
    m = d["metrics"]
    assert d["step"] == "Step 17"
    assert m["accuracy_benefit_rate"] == 0.1481 and m["bacc_benefit_rate"] == 0.1481
    assert m["accuracy_policy_controls_bacc"] is False
    assert m["no_overall_best_across_estimands"] is True and m["bacc_policy_requires_c13"] is True
    assert m["estimand_relationship"] == "identical_on_grid"
    assert "balanced-accuracy-gain control" in m["estimand_consistency_warning"].lower()
    assert m["claim_boundary_ok"] is True


def test_step17_dashboard_reports_threshold_artifact_when_identical():
    d = build_step17_dashboard(_CON17_IDENT, _FR17_IDENT)
    m = d["metrics"]
    assert m["estimands_identical_on_grid"] is True and m["max_abs_gain_difference"] == 0.0
    blob = " ".join(d["what_we_learned"]).lower()
    assert "threshold artifact" in blob                         # not a real accuracy-vs-bAcc divergence


def test_step17_dashboard_full_budget_control_is_not_minimal_label_success():
    m = build_step17_dashboard(_CON17_IDENT, _FR17_IDENT)["metrics"]
    # both groups "meet" harm<=0.10 only at k=full -> NOT counted as minimal-label control
    assert m["best_accuracy_gain_meets_harm_0_10"] is True
    assert m["best_accuracy_gain_control_at_minimal_labels"] is False
    assert m["best_bacc_gain_control_at_minimal_labels"] is False


def test_step17_dashboard_states_negative_persists_under_bacc_consistent_control():
    blob = " ".join(build_step17_dashboard(_CON17_IDENT, _FR17_IDENT)["what_we_learned"]).lower()
    assert "persists under bacc-consistent control" in blob     # only k=full/oracle-equivalent budgets control


def test_step17_dashboard_reports_per_estimand_best_policies():
    m = build_step17_dashboard(_CON17_DISAGREE, _FR17_DISAGREE)["metrics"]
    assert m["best_accuracy_gain_policy"] == "ci_three_way" and m["best_accuracy_gain_k"] == 32
    assert m["best_accuracy_gain_control_at_minimal_labels"] is True   # k=32 is a real minimal budget
    assert m["best_bacc_gain_policy"] is None                          # no C13 policy meets harm<=0.10
    assert m["best_bacc_gain_meets_harm_0_10"] is False


def test_step17_dashboard_estimand_dependent_when_only_accuracy_controls():
    d = build_step17_dashboard(_CON17_DISAGREE, _FR17_DISAGREE)
    blob = " ".join(d["what_we_learned"]).lower()
    assert "estimand-dependent" in blob and "no sota" in d["claim_boundary"].lower()


def test_step17_dashboard_claim_boundary_fails_if_accuracy_controls_bacc():
    bad_con = dict(_CON17_IDENT, accuracy_policy_controls_bacc=True)
    assert build_step17_dashboard(bad_con, _FR17_IDENT)["metrics"]["claim_boundary_ok"] is False
    bad_fr = dict(_FR17_IDENT, no_overall_best_across_estimands=False)
    assert build_step17_dashboard(_CON17_IDENT, bad_fr)["metrics"]["claim_boundary_ok"] is False


_HM18 = {"n_runs": 54, "mean_lost_correct_rate": 0.1211, "mean_gained_correct_rate": 0.0789,
         "mean_net_gain": -0.0423, "fraction_runs_with_mixed_class_effects": 0.963,
         "worst_classes_by_dataset": {"BNCI2014_001": {"most_common_worst_class": 1},
                                      "BNCI2014_004": {"most_common_worst_class": 0}},
         "oracle_labels_used_only_for_mechanism_and_evaluation": True, "claim_boundary_ok": True}
_PS18 = {"fraction_prior_dependent_sign": 0.963, "fraction_harmful_under_all_priors": 0.037,
         "fraction_beneficial_under_all_priors": 0.0, "fraction_uniform_harm_but_some_prior_benefit": 0.8148,
         "fraction_uniform_benefit_but_some_prior_harm": 0.1481, "mean_prior_sign_width": 0.4423,
         "prior_contract_required": "C14", "deployment_prior_identified_under_R1": False,
         "deployment_prior_identified": False, "claim_boundary_ok": True}


def test_step18_dashboard_reports_harm_channels_and_prior_stress():
    d = build_step18_dashboard(_HM18, _PS18)
    m = d["metrics"]
    assert d["step"] == "Step 18"
    assert m["mean_lost_correct_rate"] == 0.1211 and m["mean_gained_correct_rate"] == 0.0789
    assert m["fraction_prior_dependent_sign"] == 0.963 and m["fraction_harmful_under_all_priors"] == 0.037
    assert m["prior_contract_required"] == "C14" and m["deployment_prior_identified_under_R1"] is False
    assert m["claim_boundary_ok"] is True


def test_step18_dashboard_flags_class_prior_dependence_when_not_global():
    blob = " ".join(build_step18_dashboard(_HM18, _PS18)["what_we_learned"]).lower()
    assert "class/prior-dependent" in blob and "c14" in blob
    assert "masks" in blob or "mask" in blob                   # bAcc hides niche-class benefit/harm


def test_step18_dashboard_flags_global_harm_when_mostly_harmful_all_priors():
    ps = dict(_PS18, fraction_harmful_under_all_priors=0.8, fraction_prior_dependent_sign=0.1)
    blob = " ".join(build_step18_dashboard(_HM18, ps)["what_we_learned"]).lower()
    assert "global" in blob


def test_step18_dashboard_claim_boundary_fails_if_prior_identified():
    bad = dict(_PS18, deployment_prior_identified=True)
    assert build_step18_dashboard(_HM18, bad)["metrics"]["claim_boundary_ok"] is False
    bad_r1 = dict(_PS18, deployment_prior_identified_under_R1=True)
    assert build_step18_dashboard(_HM18, bad_r1)["metrics"]["claim_boundary_ok"] is False


def test_step18_dashboard_no_sota_and_prior_decoupled_boundary():
    d = build_step18_dashboard(_HM18, _PS18)
    assert "no sota" in d["claim_boundary"].lower()
    assert "prior-decoupled" in d["claim_boundary"].lower() or "not identified" in d["claim_boundary"].lower()


_PU19 = {"n_runs": 54, "median_flip_radius_from_uniform": 0.1652, "q25_flip_radius": 0.0731,
         "q75_flip_radius": 0.3042, "n_unflippable_over_simplex": 2,
         "fraction_flip_within_l1_0_10": 0.2778, "fraction_flip_within_l1_0_20": 0.6296,
         "fraction_flip_within_l1_0_50": 0.8704,
         "fraction_ambiguous_by_rho": {"0.1": 0.2778}, "fraction_robust_harm_by_rho": {"0.1": 0.6852},
         "fraction_robust_benefit_by_rho": {"0.1": 0.037},
         "prior_uncertainty_contract_required": "C15", "actual_target_prior_identified": False,
         "deployment_prior_identified_under_R1": False, "claim_boundary_ok": True}
_PP19_NONE = {"best_prior_robust_policy": None, "robust_prior_safe_adaptation_exists_any": False,
              "robust_adapt_never_uniform_harmful": True, "actual_target_prior_identified": False,
              "claim_boundary_ok": True}
_PP19_SAFE = {"best_prior_robust_policy": {"rho": 0.05, "tau": 0.05, "adaptation_coverage": 0.1},
              "robust_prior_safe_adaptation_exists_any": True,
              "robust_adapt_never_uniform_harmful": True, "actual_target_prior_identified": False,
              "claim_boundary_ok": True}


def test_step19_dashboard_reports_flip_radius_and_frontier():
    d = build_step19_dashboard(_PU19, _PP19_NONE)
    m = d["metrics"]
    assert d["step"] == "Step 19"
    assert m["median_l1_flip_radius_from_uniform"] == 0.1652
    assert m["fraction_flip_within_l1_0_20"] == 0.6296
    assert m["fraction_robust_harm_at_rho_0_10"] == 0.6852
    assert m["fraction_robust_benefit_at_rho_0_10"] == 0.037
    assert m["prior_uncertainty_contract_required"] == "C15"
    assert m["deployment_prior_identified_under_R1"] is False and m["claim_boundary_ok"] is True


def test_step19_dashboard_flags_no_harm_margin_safe_adaptation():
    d = build_step19_dashboard(_PU19, _PP19_NONE)
    m = d["metrics"]
    assert m["robust_prior_safe_adaptation_exists_with_harm_margin"] is False
    assert m["best_prior_robust_policy_rho"] is None and m["best_prior_robust_policy_tau"] is None
    blob = " ".join(d["what_we_learned"]).lower()
    assert "fragile" in blob and "cannot be certified" in blob


def test_step19_dashboard_reports_best_policy_when_safe_exists():
    m = build_step19_dashboard(_PU19, _PP19_SAFE)["metrics"]
    assert m["robust_prior_safe_adaptation_exists_with_harm_margin"] is True
    assert m["best_prior_robust_policy_rho"] == 0.05 and m["best_prior_robust_policy_tau"] == 0.05


def test_step19_dashboard_claim_boundary_fails_if_prior_identified():
    bad = dict(_PU19, actual_target_prior_identified=True)
    assert build_step19_dashboard(bad, _PP19_NONE)["metrics"]["claim_boundary_ok"] is False
    bad2 = dict(_PP19_NONE, robust_adapt_never_uniform_harmful=False)
    assert build_step19_dashboard(_PU19, bad2)["metrics"]["claim_boundary_ok"] is False


def test_step19_dashboard_no_sota_and_prior_decoupled_boundary():
    d = build_step19_dashboard(_PU19, _PP19_NONE)
    assert "no sota" in d["claim_boundary"].lower()
    assert "not identified" in d["claim_boundary"].lower() or "prior-decoupled" in d["claim_boundary"].lower()


ALL_TESTS = [
    test_dashboard_reports_real_harm_and_predictor,
    test_dashboard_claim_boundary_ok_requires_oracle_not_feature,
    test_dashboard_lists_learned_and_unknown,
    test_dashboard_no_sota_claim,
    test_step13_dashboard_reports_r1_availability_and_minimal_label_k,
    test_step13_dashboard_stronger_null_when_r1_below_baseline,
    test_step14_dashboard_uses_harm_power_and_coverage_decomposition,
    test_dashboard_does_not_use_deprecated_harm_sign_accuracy,
    test_step15_dashboard_reports_best_policy_and_coverage_tradeoff,
    test_deprecated_ci_attempt_key_not_used_by_dashboard,
    test_step15_dashboard_never_selects_oracle_policy,
    test_step16_dashboard_reports_benefit_and_sequential,
    test_step16_dashboard_claim_boundary_fails_if_oracle_deployable,
    test_step17_dashboard_separates_estimands_and_flags_relationship,
    test_step17_dashboard_reports_threshold_artifact_when_identical,
    test_step17_dashboard_full_budget_control_is_not_minimal_label_success,
    test_step17_dashboard_states_negative_persists_under_bacc_consistent_control,
    test_step17_dashboard_reports_per_estimand_best_policies,
    test_step17_dashboard_estimand_dependent_when_only_accuracy_controls,
    test_step17_dashboard_claim_boundary_fails_if_accuracy_controls_bacc,
    test_step18_dashboard_reports_harm_channels_and_prior_stress,
    test_step18_dashboard_flags_class_prior_dependence_when_not_global,
    test_step18_dashboard_flags_global_harm_when_mostly_harmful_all_priors,
    test_step18_dashboard_claim_boundary_fails_if_prior_identified,
    test_step18_dashboard_no_sota_and_prior_decoupled_boundary,
    test_step19_dashboard_reports_flip_radius_and_frontier,
    test_step19_dashboard_flags_no_harm_margin_safe_adaptation,
    test_step19_dashboard_reports_best_policy_when_safe_exists,
    test_step19_dashboard_claim_boundary_fails_if_prior_identified,
    test_step19_dashboard_no_sota_and_prior_decoupled_boundary,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} SCIENCE-DASHBOARD TESTS PASSED")


if __name__ == "__main__":
    run()
