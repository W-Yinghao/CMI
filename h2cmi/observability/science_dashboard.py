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
        "best_deployable_ci_attempt": hc.get("best_deployable_ci_attempt"),
        "claim_boundary_ok": claim_ok,
        "r2_iid_sampling_contract_required": hc.get("r2_iid_sampling_contract_required") is True,
        "oracle_policy_selected_as_deployable": bool(hc.get("oracle_policy_selected_as_deployable")),
    }
    oref = hc.get("oracle_reference") or {}
    att = hc.get("best_deployable_ci_attempt") or {}
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
    ap.add_argument("--harm-table", required=True)
    ap.add_argument("--harm-predictor", required=True)
    ap.add_argument("--phase-transition", default=None, help="Step-12 mode")
    ap.add_argument("--multidataset", required=True)
    ap.add_argument("--real-minimal-labels", default=None,
                    help="Step-13 mode: real minimal-label curves JSON")
    ap.add_argument("--step12-harm-predictor", default=None,
                    help="Step-13+ mode: prior Step-12 harm predictor for improvement comparison")
    ap.add_argument("--harm-power", default=None, help="Step-14 mode: harm-power summary JSON")
    ap.add_argument("--harm-control", default=None, help="Step-15 mode: harm-control summary JSON")
    ap.add_argument("--step-label", default="Step 13", help="dashboard provenance label")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)

    if args.harm_control:                                     # Step 15
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
