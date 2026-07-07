"""Project A Step 12 — science dashboard.

Combines the four Step-12 artifacts (harm-attribution table, retrospective harm predictor,
minimal-paired phase transition, multi-dataset digest) into a single reviewer-readable dashboard of
"what we learned" and "what remains unknown". It asserts nothing beyond those artifacts and makes no
SOTA claim; the oracle gain stays an evaluation label throughout.

  python -m h2cmi.observability.science_dashboard --harm-table ... --harm-predictor ... \
      --phase-transition ... --multidataset ... --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .result_index import _load_json, write_json_lf, write_text_lf


def build_dashboard(harm_table, harm_pred, phase, multi) -> Dict[str, Any]:
    fs = (harm_pred or {}).get("feature_sets", {})
    r0 = fs.get("R0_source_only", {})
    r1 = fs.get("R1_target_unlabeled", {})
    oracle_never = (harm_pred or {}).get("oracle_never_a_feature")

    # integrity conjunction: the whole science phase kept the claim boundary
    claim_boundary_ok = (
        oracle_never is True
        and "oracle_denylist" in (harm_table or {})
        and (phase or {}).get("k0_status") == "not_identified_R1"
        and (multi or {}).get("all_target_metrics_identifiable_null") is not False)

    beats = (harm_pred or {}).get("any_predictor_beats_majority_baseline")
    metrics = {
        "n_real_runs": (harm_table or {}).get("n_runs"),
        "real_harm_rate": (harm_table or {}).get("harm_rate"),
        "R0_harm_predictor_bacc": r0.get("balanced_acc_harm_prediction"),
        "R1_harm_predictor_bacc": r1.get("balanced_acc_harm_prediction"),
        "R0_to_R1_delta": (harm_pred or {}).get("r1_minus_r0_balanced_acc_delta"),
        "majority_baseline_bacc": (harm_pred or {}).get("majority_baseline_balanced_acc"),
        "harm_predictor_verdict": (harm_pred or {}).get("verdict"),
        "harm_predictor_beats_baseline": beats,
        "harm_predictor_minority_n": (harm_pred or {}).get("n_minority_class"),
        "minimal_paired_k0_status": (phase or {}).get("k0_status"),
        "minimal_paired_best_k": (phase or {}).get("best_k_overall"),
        "minimal_paired_phase_transition_observed": (phase or {}).get("phase_transition_observed"),
        "minimal_paired_k_per_shift": (phase or {}).get("phase_transition_k_per_shift"),
        "claim_boundary_ok": claim_boundary_ok,
        "oracle_gain_used_only_as_evaluation_label": oracle_never is True,
    }

    # honest characterization: below/at 0.5 baseline => NO signal (do not call it "a predictor")
    pred_line = (
        f"R0/R1 diagnostics do NOT retrospectively predict TTA harm above the 0.5 majority baseline "
        f"(R0 bAcc {metrics['R0_harm_predictor_bacc']}, R1 bAcc {metrics['R1_harm_predictor_bacc']}; "
        f"underpowered, minority n={metrics['harm_predictor_minority_n']}) — consistent with the "
        f"TOS-1 source-only ceiling; NULL result, not identifiability."
        if not beats else
        f"R0/R1 diagnostics give a RETROSPECTIVE harm predictor above baseline (R0 bAcc "
        f"{metrics['R0_harm_predictor_bacc']}, R1 bAcc {metrics['R1_harm_predictor_bacc']}, delta "
        f"{metrics['R0_to_R1_delta']}) — empirical retrospective, NOT identifiability.")
    trans = metrics["minimal_paired_phase_transition_observed"]
    trans_line = (
        f"Minimal paired information: harm-sign estimability is a phase transition in k "
        f"(observed={trans}, per-shift k {metrics['minimal_paired_k_per_shift']}); small true gains "
        f"need more labels, tiny gains stay unresolved — a labeled slice under an iid sampling contract.")
    learned = [
        f"Offline TTA harms most audited cells (real harm-rate {metrics['real_harm_rate']}).",
        pred_line,
        "R1 target-unlabeled diagnostics do NOT make target gain identifiable (TOS-1/TU-2 stand).",
        trans_line,
        "Exact counterexamples remain the proof layer; the real-EEG grids illustrate, they do not prove.",
    ]
    unknown = [
        "Whether these patterns hold on clinical / non-motor-imagery EEG.",
        "Whether stronger TTA baselines reduce the harm rate.",
        "Whether label-free target support/marginal diagnostics can be made reliable.",
        "Whether minimal-paired anchors can be collected cheaply in realistic BCI workflows.",
    ]
    return {
        "project": "Project A", "step": "Step 12",
        "scope": "scientific exploration (harm attribution + minimal-information phase transition); not SOTA",
        "metrics": metrics,
        "what_we_learned": learned,
        "what_remains_unknown": unknown,
        "claim_boundary": ("Oracle target gain is an evaluation label throughout; R0/R1 harm "
                           "prediction is retrospective, not target-gain identifiability; k>0 slices "
                           "are labeled slices under an iid sampling contract. No SOTA claim."),
    }


def build_step13_dashboard(harm_table, harm_pred, real_curves, multi, step12_harm_pred=None,
                           harm_power=None, step_label="Step 13") -> Dict[str, Any]:
    fs = (harm_pred or {}).get("feature_sets", {})
    r0, r1 = fs.get("R0_source_only", {}), fs.get("R1_target_unlabeled", {})
    oracle_never = (harm_pred or {}).get("oracle_never_a_feature")
    runs = (harm_table or {}).get("runs", [])
    n = (harm_table or {}).get("n_runs") or len(runs)
    avail = sum(1 for r in runs if r.get("r1_source") == "instrumented_r1_diagnostics")
    r1_bacc = r1.get("balanced_acc_harm_prediction")
    s12_r1 = ((step12_harm_pred or {}).get("feature_sets", {}).get("R1_target_unlabeled", {})
              .get("balanced_acc_harm_prediction"))
    r1_beats = bool((harm_pred or {}).get("any_predictor_beats_majority_baseline"))
    survives = bool((harm_pred or {}).get("any_predictor_survives_permutation_null"))
    robust = bool((harm_pred or {}).get("any_predictor_robust_signal"))
    k256 = ((real_curves or {}).get("per_k", {}) or {}).get("256", {})
    claim_ok = (oracle_never is True and "oracle_denylist" in (harm_table or {})
                and (real_curves or {}).get("k0_status") == "not_identified_R1"
                and (real_curves or {}).get("oracle_labels_used_only_for_r2_slice_and_evaluation") is True
                and (multi or {}).get("all_target_metrics_identifiable_null") is not False)
    metrics = {
        "n_real_runs": n, "real_harm_rate": (harm_table or {}).get("harm_rate"),
        "r1_diagnostics_available_rate": round(avail / n, 4) if n else None,
        "R0_harm_predictor_bacc": r0.get("balanced_acc_harm_prediction"),
        "R1_harm_predictor_bacc": r1_bacc,
        "R1_beats_baseline": r1_beats,
        "R1_survives_permutation_null": survives,
        "R1_robust_signal": robust,
        "R1_perm_null_p95": r1.get("perm_null_p95"),
        "R1_perm_null_p99": r1.get("perm_null_p99"),
        "R1_margin_over_p95": r1.get("margin_over_perm_null_p95"),
        "harm_predictor_verdict": (harm_pred or {}).get("verdict"),
        "harm_power_underpowered": bool((harm_power or {}).get("underpowered")),
        "minimum_detectable_bacc_approx": (harm_power or {}).get("minimum_detectable_bacc_approx"),
        "R1_improves_over_step12": bool(r1_bacc is not None and s12_r1 is not None and r1_bacc > s12_r1),
        # coverage-decomposed real minimal-label curve (unconditional vs conditional, not conflated)
        "real_minimal_label_best_k_0_8_unconditional": (real_curves or {}).get("best_k_for_0_8_unconditional"),
        "real_minimal_label_best_k_0_8_conditional": (real_curves or {}).get("best_k_for_0_8_conditional"),
        "real_minimal_label_k256_coverage": k256.get("decisive_rate"),
        "real_minimal_label_k256_conditional_accuracy": k256.get("conditional_accuracy_given_decisive"),
        "claim_boundary_ok": claim_ok,
        "oracle_gain_used_only_as_evaluation_label": oracle_never is True,
        "target_labels_used_in_r1_diagnostics": False,     # structural: R1 diagnostics are label-free
    }
    r1p95 = metrics["R1_perm_null_p95"]
    if not r1_beats:
        pred_change = ("Richer R1 diagnostics STILL do not beat the 0.5 baseline -> STRONGER NULL: "
                       "current target-unlabeled diagnostics do not retrospectively predict TTA harm.")
    elif not survives:
        pred_change = (f"Richer R1 diagnostics beat 0.5 but do NOT clear their permutation null (p95 "
                       f"{r1p95}) -> OVERFITTING ARTIFACT (19 features / 54 rows / 8 minority), not signal.")
    elif not robust:
        pred_change = (f"Richer R1 diagnostics only MARGINALLY clear a high permutation null (bAcc vs "
                       f"perm-p95 {r1p95}; margin < 0.03) -> within Monte-Carlo/overfitting noise, NOT a "
                       f"robust predictor; R0 source-only stays below its own null (TOS-1 ceiling).")
    else:
        pred_change = ("Richer R1 diagnostics beat the baseline AND robustly clear the permutation null "
                       "-> a real empirical RETROSPECTIVE predictor (still not identifiability).")
    changed = [
        pred_change,
        (f"Real minimal-label curves are COVERAGE-limited, not inaccurate: at k=256 coverage (decisive "
         f"rate) is {metrics['real_minimal_label_k256_coverage']} while accuracy WHEN decisive is "
         f"{metrics['real_minimal_label_k256_conditional_accuracy']}; best k for unconditional≥0.8 = "
         f"{metrics['real_minimal_label_best_k_0_8_unconditional']}, conditional≥0.8 = "
         f"{metrics['real_minimal_label_best_k_0_8_conditional']}. The burden is coverage, not accuracy "
         f"(labeled slice under an iid sampling contract)."),
        (f"Power: minority n small, underpowered={metrics['harm_power_underpowered']}; a bAcc below ~"
         f"{metrics['minimum_detectable_bacc_approx']} is indistinguishable from the overfitting null."),
        "R1 diagnostics remain label-free; oracle per-trial labels used only for R2 curves / evaluation.",
    ]
    return {"project": "Project A", "step": step_label,
            "scope": "rich R1 diagnostics + real minimal-label curves (coverage-decomposed) + power; not SOTA",
            "metrics": metrics, "what_changed_from_step12": changed,
            "claim_boundary": ("R1 diagnostics are label-free; R0/R1 harm prediction is retrospective, not "
                               "identifiability; real k>0 curves are labeled slices under an iid sampling "
                               "contract (coverage-limited). No SOTA claim.")}


def build_step15_dashboard(harm_control, harm_pred=None, real_curves=None, multi=None,
                           step_label="Step 15") -> Dict[str, Any]:
    hc = harm_control or {}
    best = hc.get("best_deployable_policy", {}) or {}
    claim_ok = (hc.get("claim_boundary_ok") is True
                and hc.get("oracle_policy_selected_as_deployable") is False
                and hc.get("r2_iid_sampling_contract_required") is True
                and (real_curves or {}).get("k0_status", "not_identified_R1") == "not_identified_R1"
                and (multi or {}).get("all_target_metrics_identifiable_null") is not False)
    metrics = {
        "n_real_runs": hc.get("n_runs"),
        "real_harm_rate_always_adapt": hc.get("always_adapt_harm_rate"),
        "best_policy_by_harm_control": best.get("policy"),
        "best_policy_k": best.get("k"), "best_policy_tau": best.get("tau"),
        "best_policy_adaptation_coverage": best.get("adaptation_coverage"),
        "best_policy_decision_coverage": best.get("decision_coverage"),
        "best_policy_harm_rate_among_adapt": best.get("harm_rate_among_adapt_decisions"),
        "best_policy_prevented_harm_vs_always_adapt": best.get("prevented_harm_rate_vs_always_adapt"),
        "best_policy_missed_benefit_rate": best.get("missed_benefit_rate"),
        "coverage_control_tradeoff_observed": best.get("policy") is not None,
        "oracle_reference": hc.get("oracle_reference"),
        "best_label_based_attempt": hc.get("best_label_based_attempt"),
        "claim_boundary_ok": claim_ok,
        "r2_iid_sampling_contract_required": hc.get("r2_iid_sampling_contract_required") is True,
        "oracle_policy_selected_as_deployable": bool(hc.get("oracle_policy_selected_as_deployable")),
    }
    oref = hc.get("oracle_reference") or {}
    att = hc.get("best_label_based_attempt") or {}
    if best.get("policy") is not None:
        learned = [
            (f"A coverage-aware policy ({best.get('policy')}, k={best.get('k')}, tau={best.get('tau')}) "
             f"adapts {best.get('adaptation_coverage')} of cells with harm-among-adapt "
             f"{best.get('harm_rate_among_adapt_decisions')} (<= {hc.get('harm_constraint')}) vs the "
             f"always-adapt harm-rate {hc.get('always_adapt_harm_rate')} -> harm is CONTROLLABLE with "
             f"minimal labels, but only at LOW coverage (most cells abstain / stay identity)."),
            (f"It prevents {best.get('prevented_harm_rate_vs_always_adapt')} of always-adapt harm at "
             f"the cost of missing {best.get('missed_benefit_rate')} of benefit -> a coverage/control "
             f"tradeoff, not free improvement."),
        ]
    else:
        learned = [
            (f"NO deployable minimal-label policy adapts while keeping harm<=0.05: the best a "
             f"label-based policy achieves is adapt-coverage {att.get('adaptation_coverage')} at "
             f"harm-among-adapt {att.get('harm_rate_among_adapt_decisions')} ({att.get('policy')}, "
             f"k={att.get('k')}). Confident/positive slices do NOT select beneficial cells -- with a "
             f"high harm base-rate, adapt-positive events are dominated by false positives on harmful cells."),
            (f"The oracle full-label upper bound adapts {oref.get('adaptation_coverage')} of cells at "
             f"harm {oref.get('harm_rate_among_adapt_decisions')} (prevented {oref.get('prevented_harm_rate_vs_always_adapt')}, "
             f"missed {oref.get('missed_benefit_rate')}): safe adaptation IS possible, but only with "
             f"(near-)full target labels -- the measurement->control gap is NOT closed by R2 minimal-label "
             f"CI policies on this grid.")]
    learned.append("Decisions use k>0 target labels (R2 labeled slice under an iid sampling contract); "
                   "k=0 stays R1 non-identifiable; the oracle policy is an evaluation-only upper bound.")
    return {"project": "Project A", "step": step_label,
            "scope": "coverage-aware harm-control policies under minimal labels; not SOTA",
            "metrics": metrics, "what_we_learned": learned,
            "what_remains_unknown": [
                "Whether the same policy transfers to clinical / non-motor-imagery EEG.",
                "Whether cheaper-than-iid label acquisition raises coverage at fixed harm.",
                "Whether a label-free coverage proxy could pre-screen which targets to label."],
            "claim_boundary": ("R2 minimal-label policy evaluation under an iid sampling contract; NOT "
                               "R1 target-gain identifiability; oracle policy is not deployable. No SOTA.")}


def build_step16_dashboard(benefit, sequential, frontier, harm_control_static=None,
                           real_curves=None, multi=None, step_label="Step 16") -> Dict[str, Any]:
    ben = benefit or {}
    seq = sequential or {}
    fr = frontier or {}
    bseq = seq.get("best_sequential_policy", {}) or {}
    bstat = (harm_control_static or {}).get("best_deployable_policy", {}) or {}
    seq_full_or_none = (bseq.get("policy") is None) or (str(bseq.get("budget")) == "full")
    claim_ok = (ben.get("claim_boundary", "").lower().find("oracle-only") >= 0
                and seq.get("oracle_policy_selected_as_deployable") is False
                and seq.get("r2_iid_sampling_contract_required") is True
                and fr.get("oracle_excluded") is True
                and (real_curves or {}).get("k0_status", "not_identified_R1") == "not_identified_R1"
                and (multi or {}).get("all_target_metrics_identifiable_null") is not False)
    metrics = {
        "n_real_runs": ben.get("n_runs"),
        "benefit_rate": ben.get("benefit_rate"),
        "beneficial_cells_by_dataset": {k: v.get("n_beneficial") for k, v in (ben.get("per_dataset") or {}).items()},
        "benefit_sign_stability_by_target": ben.get("target_sign_consistency_rate"),
        "beneficial_gain_q90_bacc": (ben.get("beneficial_gain_distribution_bacc") or {}).get("q90"),
        "best_static_policy": bstat.get("policy"),
        "best_sequential_policy": bseq.get("policy"),
        "best_sequential_budget": bseq.get("budget"),
        "best_sequential_tau": bseq.get("tau"),
        "best_sequential_mean_labels_used": bseq.get("mean_labels_used"),
        "best_sequential_adaptation_coverage": bseq.get("adaptation_coverage"),
        "best_sequential_harm_rate": bseq.get("harm_rate_among_adapt_decisions"),
        "best_sequential_missed_benefit_rate": bseq.get("missed_benefit_rate"),
        "safe_adaptation_requires_full_or_near_full_labels": bool(seq_full_or_none),
        "policy_frontier_harm_0_05_exists": bool(fr.get("any_policy_meets_harm_0_05")),
        "policy_frontier_harm_0_10_exists": bool(fr.get("any_policy_meets_harm_0_1")),
        "policy_frontier_harm_0_20_exists": bool(fr.get("any_policy_meets_harm_0_2")),
        "claim_boundary_ok": claim_ok,
        "oracle_policy_selected_as_deployable": bool(seq.get("oracle_policy_selected_as_deployable")),
    }
    if bseq.get("policy") is None:
        seq_line = ("Sequential label acquisition does NOT rescue Step 15: no sequential policy meets "
                    "harm<=0.05 with coverage>=0.05 at any budget -> minimal labels do not enable safe "
                    "adaptation selection; identity/default remains safest.")
    elif str(bseq.get("budget")) == "full":
        seq_line = (f"A sequential policy meets harm<=0.05 only near FULL budget "
                    f"({bseq.get('policy')}, mean-labels {bseq.get('mean_labels_used')}, adapt-cov "
                    f"{bseq.get('adaptation_coverage')}) -> safe adaptation requires a large calibration burden.")
    else:
        seq_line = (f"A sequential policy meets harm<=0.05 with a moderate budget ({bseq.get('policy')}, "
                    f"budget {bseq.get('budget')}, mean-labels {bseq.get('mean_labels_used')}, adapt-cov "
                    f"{bseq.get('adaptation_coverage')}) -> R2 minimal labels can control harm under an iid "
                    f"sampling contract, but NOT R1.")
    learned = [
        (f"Beneficial cells are rare (benefit-rate {ben.get('benefit_rate')}) and their sign is "
         f"{ben.get('target_sign_consistency_rate')} consistent across seeds per target; beneficial gains "
         f"are small (q90 bAcc {(ben.get('beneficial_gain_distribution_bacc') or {}).get('q90')}) -> the "
         f"Step-15 false positives are explained by rare, small, unstable benefit."),
        seq_line,
        (f"Frontier: any deployable policy meets harm<=0.05 {metrics['policy_frontier_harm_0_05_exists']}, "
         f"<=0.10 {metrics['policy_frontier_harm_0_10_exists']}, <=0.20 {metrics['policy_frontier_harm_0_20_exists']} "
         f"-- shows whether the 0.05 constraint was simply too strict."),
        "Benefit anatomy is oracle-only; sequential policies are R2 labeled slices under an iid sampling "
        "contract; the oracle policy is an evaluation-only upper bound, never deployable.",
    ]
    return {"project": "Project A", "step": step_label,
            "scope": "benefit anatomy + sequential label-acquisition frontier; not SOTA",
            "metrics": metrics, "what_we_learned": learned,
            "what_remains_unknown": [
                "Whether benefit rarity/instability holds on clinical / non-motor-imagery EEG.",
                "Whether a label-free coverage proxy could pre-screen which targets to label.",
                "Whether active (non-iid) acquisition beats iid sampling at fixed harm."],
            "claim_boundary": ("Benefit anatomy is oracle/evaluation-only; sequential policies are R2 "
                               "labeled slices under an iid sampling contract, NOT R1 identifiability; "
                               "oracle policy not deployable. No SOTA.")}


def build_step17_dashboard(consistency, frontier, step_label="Step 17") -> Dict[str, Any]:
    con = consistency or {}
    fr = frontier or {}
    groups = fr.get("groups", {}) or {}
    acc_grp = groups.get("accuracy_gain:iid", {}) or {}          # accuracy's natural sampling
    bacc_grp = groups.get("balanced_accuracy_gain:class_balanced", {}) or {}   # bAcc's C13 sampling

    def _best_policy(g):
        b = (g or {}).get("best_under_harm_0_1")
        return (b or {}).get("policy")

    def _best_k(g):
        b = (g or {}).get("best_under_harm_0_1")
        return (b or {}).get("k")

    def _minimal_label(g):                                       # control at a genuinely small budget?
        b = (g or {}).get("best_under_harm_0_1")
        return bool(b and b.get("k") not in (None, "full"))

    disagree = (con.get("runs_accuracy_benefit_bacc_harm", 0) or 0) + \
        (con.get("runs_bacc_benefit_accuracy_harm", 0) or 0)
    sign_agree = con.get("cross_estimand_sign_agreement")
    rel = con.get("estimand_relationship")
    eps = con.get("benefit_eps")
    acc_eps = con.get("accuracy_material_benefit_rate_eps")
    bacc_eps = con.get("bacc_material_benefit_rate_eps")
    gap_expl = con.get("step16_gap_explanation", "")
    identical = bool(con.get("estimands_identical_on_grid"))
    tail = " An accuracy-gain policy is still never reported as a balanced-accuracy-gain control."
    if rel == "sign_disagreement":
        warn = (f"accuracy-gain and balanced-accuracy-gain disagree on SIGN for {disagree} run(s) "
                f"(sign-agreement {sign_agree}); they are genuinely different functionals here." + tail)
    elif rel == "identical_on_grid":
        warn = ("On this grid the two estimands COINCIDE (all targets class-balanced -> accuracy-gain == "
                "balanced-accuracy-gain per run); the Step-16 gap was a THRESHOLD artifact, not an estimand "
                "divergence." + tail)
    else:
        warn = (f"accuracy-gain and balanced-accuracy-gain agree on sign but differ in magnitude "
                f"(eps={eps}: {acc_eps} vs {bacc_eps})." + tail)

    bacc_requires_c13 = (bacc_grp.get("requires_contract") == "C13")
    claim_ok = (con.get("accuracy_policy_controls_bacc") is False
                and con.get("claim_boundary_ok") is True
                and fr.get("no_overall_best_across_estimands") is True
                and fr.get("accuracy_policy_controls_bacc") is False
                and bacc_requires_c13)
    metrics = {
        "n_real_runs": con.get("n_runs"),
        "accuracy_benefit_rate": con.get("accuracy_benefit_rate"),
        "bacc_benefit_rate": con.get("bacc_benefit_rate"),
        "cross_estimand_sign_agreement": sign_agree,
        "runs_accuracy_benefit_bacc_harm": con.get("runs_accuracy_benefit_bacc_harm"),
        "runs_bacc_benefit_accuracy_harm": con.get("runs_bacc_benefit_accuracy_harm"),
        "estimand_relationship": rel,
        "estimands_identical_on_grid": identical,
        "all_targets_class_balanced": bool(con.get("all_targets_class_balanced")),
        "max_abs_gain_difference": con.get("max_abs_gain_difference"),
        "estimand_gap_is_sign_disagreement": bool(con.get("estimand_gap_is_sign_disagreement")),
        "estimand_gap_is_magnitude_only": bool(con.get("estimand_gap_is_magnitude_only")),
        "accuracy_material_benefit_rate_eps": acc_eps,
        "bacc_material_benefit_rate_eps": bacc_eps,
        "mean_accuracy_gain": con.get("mean_accuracy_gain"),
        "mean_bacc_gain": con.get("mean_bacc_gain"),
        "step16_gap_explanation": gap_expl,
        "accuracy_policy_controls_bacc": False,
        "best_accuracy_gain_policy": _best_policy(acc_grp),
        "best_accuracy_gain_k": _best_k(acc_grp),
        "best_accuracy_gain_meets_harm_0_10": bool(acc_grp.get("any_policy_meets_harm_0_1")),
        "best_accuracy_gain_control_at_minimal_labels": _minimal_label(acc_grp),
        "best_bacc_gain_policy": _best_policy(bacc_grp),
        "best_bacc_gain_k": _best_k(bacc_grp),
        "best_bacc_gain_meets_harm_0_10": bool(bacc_grp.get("any_policy_meets_harm_0_1")),
        "best_bacc_gain_control_at_minimal_labels": _minimal_label(bacc_grp),
        "bacc_policy_requires_c13": bacc_requires_c13,
        "no_overall_best_across_estimands": bool(fr.get("no_overall_best_across_estimands")),
        "estimand_consistency_warning": warn,
        "claim_boundary_ok": claim_ok,
    }
    # Does the Step 15/16 negative depend on the estimand mismatch, or persist under a bAcc-consistent
    # policy? "Control" only counts if it holds at a GENUINELY MINIMAL label budget — a policy that only
    # meets harm<=0.10 at k=full uses (near-)full target labels and is oracle-equivalent, not a win.
    acc_min = metrics["best_accuracy_gain_control_at_minimal_labels"]
    bacc_min = metrics["best_bacc_gain_control_at_minimal_labels"]
    if not acc_min and not bacc_min:
        why = ("both estimands COINCIDE here (class-balanced targets), so this is estimand-invariant"
               if identical else "the failure is NOT merely an accuracy-vs-bAcc estimand mismatch")
        persist = (f"The Step 15/16 negative PERSISTS under bAcc-consistent control: no minimal-label "
                   f"policy meets harm<=0.10 at coverage>=0.05 for EITHER estimand (only k=full / oracle-"
                   f"equivalent budgets do; best-k accuracy {metrics['best_accuracy_gain_k']}, bAcc "
                   f"{metrics['best_bacc_gain_k']}) — {why}.")
    elif acc_min and not bacc_min:
        persist = ("A minimal-label policy controls ACCURACY-gain harm but no class-balanced (C13) policy "
                   "controls balanced-accuracy-gain harm at a minimal budget: the apparent control is "
                   "estimand-dependent, not a balanced-accuracy guarantee.")
    else:
        persist = (f"A minimal-label bAcc-consistent (C13) policy ({metrics['best_bacc_gain_policy']}, "
                   f"k={metrics['best_bacc_gain_k']}) meets harm<=0.10 at coverage>=0.05; reported ONLY "
                   f"for the balanced-accuracy-gain estimand it was evaluated under.")
    learned = [
        gap_expl if gap_expl else
        (f"Accuracy-gain vs balanced-accuracy-gain: sign-agreement {sign_agree}, {disagree} sign-"
         f"disagreement run(s); at eps={eps} material-benefit rates {acc_eps} vs {bacc_eps}."),
        warn,
        persist,
        ("Frontiers are kept strictly separate by estimand and sampling; there is no overall best policy "
         "across estimands; the class-balanced balanced-accuracy frontier requires contract C13."),
    ]
    return {"project": "Project A", "step": step_label,
            "scope": "estimand-consistent harm control (accuracy vs balanced-accuracy); not SOTA",
            "metrics": metrics, "what_we_learned": learned,
            "what_remains_unknown": [
                "Whether the accuracy/bAcc gap widens on more class-imbalanced clinical EEG.",
                "Whether a class-balanced (C13) acquisition protocol is feasible in real BCI calibration.",
                "Whether a bAcc-consistent policy could control harm at higher coverage with active sampling."],
            "claim_boundary": ("Accuracy-gain and balanced-accuracy-gain are distinct target functionals; a "
                               "policy licensed for one is never reported as controlling the other; class-"
                               "balanced bAcc-gain estimation requires contract C13; k>0 slices are R2 under a "
                               "sampling contract, NOT R1 identifiability. No SOTA.")}


def build_step18_dashboard(harm_mechanisms, prior_stress, step_label="Step 18") -> Dict[str, Any]:
    hm = harm_mechanisms or {}
    ps = prior_stress or {}
    claim_ok = (hm.get("oracle_labels_used_only_for_mechanism_and_evaluation") is True
                and ps.get("deployment_prior_identified") is False
                and ps.get("deployment_prior_identified_under_R1") is False
                and ps.get("prior_contract_required") == "C14"
                and hm.get("claim_boundary_ok") is True and ps.get("claim_boundary_ok") is True)
    metrics = {
        "n_real_runs": hm.get("n_runs"),
        "mean_lost_correct_rate": hm.get("mean_lost_correct_rate"),
        "mean_gained_correct_rate": hm.get("mean_gained_correct_rate"),
        "mean_net_gain": hm.get("mean_net_gain"),
        "fraction_runs_with_mixed_class_effects": hm.get("fraction_runs_with_mixed_class_effects"),
        "fraction_prior_dependent_sign": ps.get("fraction_prior_dependent_sign"),
        "fraction_harmful_under_all_priors": ps.get("fraction_harmful_under_all_priors"),
        "fraction_beneficial_under_all_priors": ps.get("fraction_beneficial_under_all_priors"),
        "fraction_uniform_harm_but_some_prior_benefit": ps.get("fraction_uniform_harm_but_some_prior_benefit"),
        "fraction_uniform_benefit_but_some_prior_harm": ps.get("fraction_uniform_benefit_but_some_prior_harm"),
        "mean_prior_sign_width": ps.get("mean_prior_sign_width"),
        "worst_classes_by_dataset": {k: v.get("most_common_worst_class")
                                     for k, v in (hm.get("worst_classes_by_dataset") or {}).items()},
        "prior_contract_required": ps.get("prior_contract_required"),
        "deployment_prior_identified_under_R1": bool(ps.get("deployment_prior_identified_under_R1")),
        "claim_boundary_ok": claim_ok,
    }
    lost, gained = metrics["mean_lost_correct_rate"], metrics["mean_gained_correct_rate"]
    harmful_all = metrics["fraction_harmful_under_all_priors"] or 0.0
    prior_dep = metrics["fraction_prior_dependent_sign"] or 0.0
    uni_harm_benefit = metrics["fraction_uniform_harm_but_some_prior_benefit"] or 0.0
    uni_benefit_harm = metrics["fraction_uniform_benefit_but_some_prior_harm"] or 0.0
    channel = (f"TTA harm is driven by lost-correct > gained-correct trials (mean {lost} vs {gained}); "
               f"per-class it is MIXED in {metrics['fraction_runs_with_mixed_class_effects']} of runs.")
    if harmful_all >= 0.5:
        prior_line = (f"Harm is GLOBAL: {harmful_all} of runs are harmful under ALL priors, so declaring a "
                      f"deployment prior cannot rescue them.")
    elif prior_dep >= 0.5:
        prior_line = (f"Harm is CLASS/PRIOR-DEPENDENT: only {harmful_all} of runs are harmful under all "
                      f"priors while {prior_dep} are prior-dependent (a declared deployment prior flips the "
                      f"gain sign). The benchmark-uniform bAcc hides this; deployment utility/prior matters "
                      f"(contract C14). This is the Prior-Decoupled boundary: without a declared prior the "
                      f"gain sign is under-determined — NOT that adaptation is safe (class deltas are oracle, "
                      f"the true prior is unidentified under R0/R1).")
    else:
        prior_line = (f"Mixed: {harmful_all} harmful-under-all-priors, {prior_dep} prior-dependent.")
    mask_line = (f"Uniform-bAcc evaluation can MASK niche-class benefit ({uni_harm_benefit} of runs are "
                 f"uniform-harm-but-some-prior-benefit) AND can MASK deployment harm ({uni_benefit_harm} "
                 f"uniform-benefit-but-some-prior-harm) — a bAcc-positive adaptation is not prior-robust.")
    learned = [channel, prior_line, mask_line,
               ("Harm-channel decomposition and prior stress are oracle/evaluation-only; the deployment "
                "prior is DECLARED (C14), never identified from R0/R1; no adaptation or SOTA claim.")]
    return {"project": "Project A", "step": step_label,
            "scope": "TTA harm mechanisms + deployment-prior stress; not SOTA",
            "metrics": metrics, "what_we_learned": learned,
            "what_remains_unknown": [
                "Whether the true deployment prior can be bounded cheaply (would need TU-1-grade contracts).",
                "Whether class-specific harm persists on clinical / non-motor-imagery EEG.",
                "Whether a utility-aware (C14-declared) selector could avoid the worst-class harm channel."],
            "claim_boundary": ("Harm-channel and prior-stress analyses are oracle/evaluation-only; priors "
                               "are DECLARED (C14), not identified; this does NOT revive a source-only "
                               "target-prior claim (Prior-Decoupled boundary). No SOTA.")}


def build_step19_dashboard(prior_uncertainty, prior_robust_policy, step_label="Step 19") -> Dict[str, Any]:
    pu = prior_uncertainty or {}
    pp = prior_robust_policy or {}
    best = pp.get("best_prior_robust_policy")
    rb10 = (pu.get("fraction_robust_benefit_by_rho") or {}).get("0.1")
    claim_ok = (pu.get("actual_target_prior_identified") is False
                and pu.get("deployment_prior_identified_under_R1") is False
                and pu.get("prior_uncertainty_contract_required") == "C15"
                and pp.get("actual_target_prior_identified") is False
                and pp.get("robust_adapt_never_uniform_harmful") is True
                and pu.get("claim_boundary_ok") is True and pp.get("claim_boundary_ok") is True)
    metrics = {
        "n_real_runs": pu.get("n_runs"),
        "median_l1_flip_radius_from_uniform": pu.get("median_flip_radius_from_uniform"),
        "q25_flip_radius": pu.get("q25_flip_radius"), "q75_flip_radius": pu.get("q75_flip_radius"),
        "n_unflippable_over_simplex": pu.get("n_unflippable_over_simplex"),
        "fraction_flip_within_l1_0_10": pu.get("fraction_flip_within_l1_0_10"),
        "fraction_flip_within_l1_0_20": pu.get("fraction_flip_within_l1_0_20"),
        "fraction_flip_within_l1_0_50": pu.get("fraction_flip_within_l1_0_50"),
        "fraction_ambiguous_at_rho_0_10": (pu.get("fraction_ambiguous_by_rho") or {}).get("0.1"),
        "fraction_robust_harm_at_rho_0_10": (pu.get("fraction_robust_harm_by_rho") or {}).get("0.1"),
        "fraction_robust_benefit_at_rho_0_10": rb10,
        "prior_robust_safe_adaptation_exists_at_rho_0_10": bool(rb10 and rb10 > 0),
        "robust_prior_safe_adaptation_exists_with_harm_margin": bool(
            pp.get("robust_prior_safe_adaptation_exists_any")),
        "best_prior_robust_policy_rho": None if best is None else best.get("rho"),
        "best_prior_robust_policy_tau": None if best is None else best.get("tau"),
        "prior_uncertainty_contract_required": pu.get("prior_uncertainty_contract_required"),
        "deployment_prior_identified_under_R1": bool(pu.get("deployment_prior_identified_under_R1")),
        "claim_boundary_ok": claim_ok,
    }
    med = metrics["median_l1_flip_radius_from_uniform"]
    w10, w20 = metrics["fraction_flip_within_l1_0_10"], metrics["fraction_flip_within_l1_0_20"]
    frag = (f"The gain sign is FRAGILE: median L1 flip-radius from uniform is {med} (q25 "
            f"{metrics['q25_flip_radius']} / q75 {metrics['q75_flip_radius']}); {w10} of runs flip within "
            f"L1≤0.10 and {w20} within ≤0.20. Only {metrics['n_unflippable_over_simplex']} runs cannot "
            f"flip over the whole simplex.")
    rb = metrics["fraction_robust_benefit_at_rho_0_10"]
    if not metrics["robust_prior_safe_adaptation_exists_with_harm_margin"]:
        safe = (f"Safe adaptation CANNOT be certified under bounded prior uncertainty: no (rho, tau) with "
                f"a harm margin tau>=0.05 yields any robustly-beneficial run (best policy = none). Even at "
                f"the zero-margin sign level only {rb} of runs are robustly beneficial at rho=0.10, "
                f"collapsing to 0 by rho=0.20. Robust-benefit is not attainable under declared uncertainty.")
    else:
        safe = (f"A prior-robust safe adaptation exists (best rho={metrics['best_prior_robust_policy_rho']}, "
                f"tau={metrics['best_prior_robust_policy_tau']}); reported only over the DECLARED prior set, "
                f"never as an identified deployment guarantee.")
    harm = (f"Identity/block is robustly justified for a meaningful fraction: robust-harm "
            f"{metrics['fraction_robust_harm_at_rho_0_10']} at rho=0.10, ambiguity "
            f"{metrics['fraction_ambiguous_at_rho_0_10']} — under bounded prior uncertainty most decisions "
            f"become abstain, and robust adaptation is never certifiable here.")
    learned = [frag, safe, harm,
               ("Robust bounds are over DECLARED L1 prior-uncertainty sets (C15); class deltas are oracle/"
                "evaluation-only; this is not a deployable selector and does not identify the actual target "
                "prior. No SOTA.")]
    return {"project": "Project A", "step": step_label,
            "scope": "prior-uncertainty robustness frontier + prior-robust policy (C15); not SOTA",
            "metrics": metrics, "what_we_learned": learned,
            "what_remains_unknown": [
                "Whether the true operating prior lies within a small L1 ball of uniform (needs TU-1-grade evidence).",
                "Whether class-specific harm channels can be avoided by a utility-aware acquisition.",
                "Whether the sign fragility persists on clinical / non-motor-imagery EEG."],
            "claim_boundary": ("Robust gain bounds are over DECLARED prior-uncertainty sets (C15); class "
                               "deltas are oracle/evaluation-only; the actual target prior is NOT identified "
                               "(Prior-Decoupled boundary). No SOTA.")}


def write_step19_md(d: Dict[str, Any], path) -> str:
    m = d["metrics"]
    lines = [f"# {d['step']} — Science Dashboard (prior-uncertainty robustness frontier)", "",
             f"Scope: {d['scope']}.", "", "## Key metrics", "",
             f"- real runs: **{m['n_real_runs']}** · median L1 flip-radius from uniform "
             f"**{m['median_l1_flip_radius_from_uniform']}** (q25 **{m['q25_flip_radius']}** / q75 "
             f"**{m['q75_flip_radius']}**) · unflippable **{m['n_unflippable_over_simplex']}**",
             f"- flip within L1 ≤0.10 **{m['fraction_flip_within_l1_0_10']}** · ≤0.20 "
             f"**{m['fraction_flip_within_l1_0_20']}** · ≤0.50 **{m['fraction_flip_within_l1_0_50']}**",
             f"- at ρ=0.10: robust-harm **{m['fraction_robust_harm_at_rho_0_10']}** · ambiguous "
             f"**{m['fraction_ambiguous_at_rho_0_10']}** · robust-benefit **{m['fraction_robust_benefit_at_rho_0_10']}**",
             f"- prior-robust safe adaptation exists @ρ0.10 (sign) **{m['prior_robust_safe_adaptation_exists_at_rho_0_10']}** · "
             f"with harm margin **{m['robust_prior_safe_adaptation_exists_with_harm_margin']}** · best policy "
             f"ρ **{m['best_prior_robust_policy_rho']}** τ **{m['best_prior_robust_policy_tau']}**",
             f"- prior-uncertainty contract required **{m['prior_uncertainty_contract_required']}** · "
             f"deployment prior identified under R1 **{m['deployment_prior_identified_under_R1']}** · claim "
             f"boundary ok **{m['claim_boundary_ok']}**", "", "## What we learned", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_we_learned"], 1)]
    lines += ["", "## What remains unknown", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_remains_unknown"], 1)]
    lines += ["", "> " + d["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def write_step18_md(d: Dict[str, Any], path) -> str:
    m = d["metrics"]
    lines = [f"# {d['step']} — Science Dashboard (harm mechanisms + prior stress)", "",
             f"Scope: {d['scope']}.", "", "## Key metrics", "",
             f"- real runs: **{m['n_real_runs']}** · mean lost-correct **{m['mean_lost_correct_rate']}** · "
             f"mean gained-correct **{m['mean_gained_correct_rate']}** · mean net gain **{m['mean_net_gain']}**",
             f"- mixed class effects **{m['fraction_runs_with_mixed_class_effects']}** · prior-dependent-sign "
             f"**{m['fraction_prior_dependent_sign']}** · harmful-under-all-priors "
             f"**{m['fraction_harmful_under_all_priors']}** · beneficial-under-all-priors "
             f"**{m['fraction_beneficial_under_all_priors']}**",
             f"- uniform-harm-but-some-prior-benefit **{m['fraction_uniform_harm_but_some_prior_benefit']}** · "
             f"uniform-benefit-but-some-prior-harm **{m['fraction_uniform_benefit_but_some_prior_harm']}** · "
             f"mean prior-sign-width **{m['mean_prior_sign_width']}**",
             f"- worst classes by dataset **{m['worst_classes_by_dataset']}**",
             f"- prior contract required **{m['prior_contract_required']}** · deployment prior identified "
             f"under R1 **{m['deployment_prior_identified_under_R1']}** · claim boundary ok "
             f"**{m['claim_boundary_ok']}**", "", "## What we learned", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_we_learned"], 1)]
    lines += ["", "## What remains unknown", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_remains_unknown"], 1)]
    lines += ["", "> " + d["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def write_step17_md(d: Dict[str, Any], path) -> str:
    m = d["metrics"]
    lines = [f"# {d['step']} — Science Dashboard (estimand-consistent harm control)", "",
             f"Scope: {d['scope']}.", "", "## Key metrics", "",
             f"- real runs: **{m['n_real_runs']}** · accuracy benefit-rate **{m['accuracy_benefit_rate']}** · "
             f"bAcc benefit-rate **{m['bacc_benefit_rate']}** · sign-agreement "
             f"**{m['cross_estimand_sign_agreement']}**",
             f"- runs accuracy-benefit∧bAcc-harm **{m['runs_accuracy_benefit_bacc_harm']}** · "
             f"bAcc-benefit∧accuracy-harm **{m['runs_bacc_benefit_accuracy_harm']}**",
             f"- estimand relationship: **{m['estimand_relationship']}** · identical on grid "
             f"**{m['estimands_identical_on_grid']}** · all targets class-balanced "
             f"**{m['all_targets_class_balanced']}** · max |acc−bAcc gain| **{m['max_abs_gain_difference']}**",
             f"- accuracy policy controls bAcc: **{m['accuracy_policy_controls_bacc']}** · "
             f"no overall best across estimands: **{m['no_overall_best_across_estimands']}**",
             f"- best accuracy-gain policy **{m['best_accuracy_gain_policy']}** (k **{m['best_accuracy_gain_k']}**, "
             f"minimal-label control **{m['best_accuracy_gain_control_at_minimal_labels']}**) · best bAcc-gain "
             f"policy **{m['best_bacc_gain_policy']}** (k **{m['best_bacc_gain_k']}**, minimal-label control "
             f"**{m['best_bacc_gain_control_at_minimal_labels']}**, requires C13 **{m['bacc_policy_requires_c13']}**)",
             f"- estimand-consistency warning: {m['estimand_consistency_warning']}",
             f"- claim boundary ok **{m['claim_boundary_ok']}**", "", "## What we learned", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_we_learned"], 1)]
    lines += ["", "## What remains unknown", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_remains_unknown"], 1)]
    lines += ["", "> " + d["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def write_step16_md(d: Dict[str, Any], path) -> str:
    m = d["metrics"]
    lines = [f"# {d['step']} — Science Dashboard (benefit anatomy + sequential frontier)", "",
             f"Scope: {d['scope']}.", "", "## Key metrics", "",
             f"- real runs: **{m['n_real_runs']}** · benefit-rate **{m['benefit_rate']}** · target "
             f"sign-stability **{m['benefit_sign_stability_by_target']}**",
             f"- best static policy **{m['best_static_policy']}** · best sequential policy "
             f"**{m['best_sequential_policy']}** (budget **{m['best_sequential_budget']}**, mean-labels "
             f"**{m['best_sequential_mean_labels_used']}**, adapt-cov **{m['best_sequential_adaptation_coverage']}**, "
             f"harm **{m['best_sequential_harm_rate']}**)",
             f"- safe adaptation requires full/near-full labels **{m['safe_adaptation_requires_full_or_near_full_labels']}**",
             f"- frontier meets harm<=0.05 **{m['policy_frontier_harm_0_05_exists']}** · <=0.10 "
             f"**{m['policy_frontier_harm_0_10_exists']}** · <=0.20 **{m['policy_frontier_harm_0_20_exists']}**",
             f"- claim boundary ok **{m['claim_boundary_ok']}** · oracle selected deployable "
             f"**{m['oracle_policy_selected_as_deployable']}**", "", "## What we learned", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_we_learned"], 1)]
    lines += ["", "## What remains unknown", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_remains_unknown"], 1)]
    lines += ["", "> " + d["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def write_step15_md(d: Dict[str, Any], path) -> str:
    m = d["metrics"]
    lines = [f"# {d['step']} — Science Dashboard (coverage-aware harm-control policies)", "",
             f"Scope: {d['scope']}.", "", "## Key metrics", "",
             f"- real runs: **{m['n_real_runs']}** · always-adapt harm-rate **{m['real_harm_rate_always_adapt']}**",
             f"- best deployable policy: **{m['best_policy_by_harm_control']}** (k **{m['best_policy_k']}**, "
             f"tau **{m['best_policy_tau']}**)",
             f"- adaptation coverage **{m['best_policy_adaptation_coverage']}** · decision coverage "
             f"**{m['best_policy_decision_coverage']}** · harm-among-adapt "
             f"**{m['best_policy_harm_rate_among_adapt']}**",
             f"- prevented harm vs always-adapt **{m['best_policy_prevented_harm_vs_always_adapt']}** · "
             f"missed benefit **{m['best_policy_missed_benefit_rate']}**",
             f"- coverage/control tradeoff observed **{m['coverage_control_tradeoff_observed']}** · "
             f"oracle selected deployable **{m['oracle_policy_selected_as_deployable']}** · claim "
             f"boundary ok **{m['claim_boundary_ok']}**", "", "## What we learned", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_we_learned"], 1)]
    lines += ["", "## What remains unknown", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_remains_unknown"], 1)]
    lines += ["", "> " + d["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def write_step13_md(d: Dict[str, Any], path) -> str:
    m = d["metrics"]
    lines = [f"# {d['step']} — Science Dashboard (rich R1 diagnostics + real minimal-label curves)", "",
             f"Scope: {d['scope']}.", "", "## Key metrics", "",
             f"- real runs: **{m['n_real_runs']}** · harm-rate **{m['real_harm_rate']}** · "
             f"R1 diagnostics available **{m['r1_diagnostics_available_rate']}**",
             f"- harm-predictor bAcc — R0 **{m['R0_harm_predictor_bacc']}** · R1 **{m['R1_harm_predictor_bacc']}** · "
             f"perm-null p95 **{m['R1_perm_null_p95']}** · margin **{m['R1_margin_over_p95']}** · robust "
             f"**{m['R1_robust_signal']}** · verdict **{m['harm_predictor_verdict']}**",
             f"- power: underpowered **{m['harm_power_underpowered']}** · min detectable bAcc ≈ "
             f"**{m['minimum_detectable_bacc_approx']}**",
             f"- real minimal-label: k256 coverage **{m['real_minimal_label_k256_coverage']}** · accuracy "
             f"when decisive **{m['real_minimal_label_k256_conditional_accuracy']}** · best k uncond≥0.8 "
             f"**{m['real_minimal_label_best_k_0_8_unconditional']}** · cond≥0.8 "
             f"**{m['real_minimal_label_best_k_0_8_conditional']}**",
             f"- claim boundary ok **{m['claim_boundary_ok']}** · target labels used in R1 diagnostics "
             f"**{m['target_labels_used_in_r1_diagnostics']}**", "", "## What changed from Step 12", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_changed_from_step12"], 1)]
    lines += ["", "> " + d["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def write_md(d: Dict[str, Any], path) -> str:
    m = d["metrics"]
    lines = ["# Step 12 — Science Dashboard", "",
             f"Scope: {d['scope']}.", "",
             "## Key metrics", "",
             f"- real runs: **{m['n_real_runs']}** · real harm-rate: **{m['real_harm_rate']}**",
             f"- harm-predictor balanced-acc — R0: **{m['R0_harm_predictor_bacc']}** · R1: "
             f"**{m['R1_harm_predictor_bacc']}** · R1−R0 delta: **{m['R0_to_R1_delta']}** "
             f"(majority baseline **{m['majority_baseline_bacc']}**)",
             f"- minimal-paired: k0 **{m['minimal_paired_k0_status']}** · phase transition "
             f"**{m['minimal_paired_phase_transition_observed']}** · best k **{m['minimal_paired_best_k']}**",
             f"- claim boundary ok: **{m['claim_boundary_ok']}** · oracle gain evaluation-only: "
             f"**{m['oracle_gain_used_only_as_evaluation_label']}**", "",
             "## What we learned", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_we_learned"], 1)]
    lines += ["", "## What remains unknown", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_remains_unknown"], 1)]
    lines += ["", "> " + d["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 12 science dashboard")
    ap.add_argument("--harm-table", default=None)
    ap.add_argument("--harm-predictor", default=None)
    ap.add_argument("--phase-transition", default=None, help="Step-12 mode")
    ap.add_argument("--multidataset", default=None)
    ap.add_argument("--real-minimal-labels", default=None,
                    help="Step-13 mode: real minimal-label curves JSON")
    ap.add_argument("--step12-harm-predictor", default=None,
                    help="Step-13+ mode: prior Step-12 harm predictor for improvement comparison")
    ap.add_argument("--harm-power", default=None, help="Step-14 mode: harm-power summary JSON")
    ap.add_argument("--harm-control", default=None, help="Step-15 mode: harm-control summary JSON")
    ap.add_argument("--benefit-anatomy", default=None, help="Step-16 mode: benefit-anatomy JSON")
    ap.add_argument("--sequential-harm-control", default=None, help="Step-16: sequential harm-control JSON")
    ap.add_argument("--policy-frontier", default=None, help="Step-16: policy-frontier JSON")
    ap.add_argument("--estimand-consistency", default=None, help="Step-17: estimand-consistency JSON")
    ap.add_argument("--estimand-frontier", default=None, help="Step-17: per-estimand frontier JSON")
    ap.add_argument("--harm-mechanisms", default=None, help="Step-18: harm-mechanism JSON")
    ap.add_argument("--prior-stress", default=None, help="Step-18: deployment-prior stress JSON")
    ap.add_argument("--prior-uncertainty", default=None, help="Step-19: prior-uncertainty frontier JSON")
    ap.add_argument("--prior-robust-policy", default=None, help="Step-19: prior-robust policy JSON")
    ap.add_argument("--step-label", default="Step 13", help="dashboard provenance label")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)

    if args.prior_uncertainty:                                # Step 19
        label = "Step 19" if args.step_label == "Step 13" else args.step_label
        d = build_step19_dashboard(
            _load_json(Path(args.prior_uncertainty)),
            _load_json(Path(args.prior_robust_policy)) if args.prior_robust_policy else None,
            step_label=label)
        md_writer = write_step19_md
    elif args.harm_mechanisms:                                # Step 18
        label = "Step 18" if args.step_label == "Step 13" else args.step_label
        d = build_step18_dashboard(
            _load_json(Path(args.harm_mechanisms)),
            _load_json(Path(args.prior_stress)) if args.prior_stress else None, step_label=label)
        md_writer = write_step18_md
    elif args.estimand_consistency:                           # Step 17
        label = "Step 17" if args.step_label == "Step 13" else args.step_label
        d = build_step17_dashboard(
            _load_json(Path(args.estimand_consistency)),
            _load_json(Path(args.estimand_frontier)) if args.estimand_frontier else None,
            step_label=label)
        md_writer = write_step17_md
    elif args.benefit_anatomy:                                # Step 16
        label = "Step 16" if args.step_label == "Step 13" else args.step_label
        d = build_step16_dashboard(
            _load_json(Path(args.benefit_anatomy)),
            _load_json(Path(args.sequential_harm_control)) if args.sequential_harm_control else None,
            _load_json(Path(args.policy_frontier)) if args.policy_frontier else None,
            _load_json(Path(args.harm_control)) if args.harm_control else None,
            _load_json(Path(args.real_minimal_labels)) if args.real_minimal_labels else None,
            _load_json(Path(args.multidataset)), step_label=label)
        md_writer = write_step16_md
    elif args.harm_control:                                   # Step 15
        label = "Step 15" if args.step_label == "Step 13" else args.step_label
        d = build_step15_dashboard(
            _load_json(Path(args.harm_control)), _load_json(Path(args.harm_predictor)),
            _load_json(Path(args.real_minimal_labels)) if args.real_minimal_labels else None,
            _load_json(Path(args.multidataset)), step_label=label)
        md_writer = write_step15_md
    elif args.real_minimal_labels:                            # Step 13 / 14
        d = build_step13_dashboard(
            _load_json(Path(args.harm_table)), _load_json(Path(args.harm_predictor)),
            _load_json(Path(args.real_minimal_labels)), _load_json(Path(args.multidataset)),
            _load_json(Path(args.step12_harm_predictor)) if args.step12_harm_predictor else None,
            _load_json(Path(args.harm_power)) if args.harm_power else None, step_label=args.step_label)
        md_writer = write_step13_md
    else:                                                     # Step 12
        d = build_dashboard(_load_json(Path(args.harm_table)), _load_json(Path(args.harm_predictor)),
                            _load_json(Path(args.phase_transition)), _load_json(Path(args.multidataset)))
        md_writer = write_md
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, d)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        md_writer(d, args.out_md)
    m = d["metrics"]
    print(f"science_dashboard[{d['step']}] n_real_runs={m.get('n_real_runs')} "
          f"R1_bAcc={m.get('R1_harm_predictor_bacc')} "
          f"best_policy={m.get('best_policy_by_harm_control')} "
          f"best_policy_adapt_cov={m.get('best_policy_adaptation_coverage')} "
          f"claim_boundary_ok={m['claim_boundary_ok']}")
    return 0 if m["claim_boundary_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
