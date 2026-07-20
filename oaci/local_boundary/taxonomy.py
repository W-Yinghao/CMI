"""C33 deterministic B1-B10 local-boundary taxonomy."""
from __future__ import annotations

from . import schema


def classify(boundary, pairs, gradients, plateaus, ladder, robust=None):
    cases, evidence = [], {}
    bs = boundary["summary"]
    ps = pairs["summary"]
    gs = gradients["summary"]
    pls = plateaus["summary"]
    ls = ladder["summary"]

    b1 = bool((bs["pm1_contains_joint_fraction"] or 0) >= 0.70 and (bs["mean_transition_rate"] or 0) >= 0.20)
    evidence["B1"] = {"pm1_contains_joint_fraction": bs["pm1_contains_joint_fraction"],
                      "mean_transition_rate": bs["mean_transition_rate"],
                      "median_selected_boundary_distance": bs["median_selected_boundary_distance"]}
    if b1:
        cases.append(schema.B1)

    b2 = bool((ps["source_flat_fraction"] or 0) >= 0.30)
    evidence["B2"] = {"source_flat_fraction": ps["source_flat_fraction"]}
    if b2:
        cases.append(schema.B2)

    b3 = bool((ps["source_wrong_fraction"] or 0) >= 0.30)
    evidence["B3"] = {"source_wrong_fraction": ps["source_wrong_fraction"]}
    if b3:
        cases.append(schema.B3)

    b4 = bool((ps["gauge_jump_unseen_fraction"] or 0) >= 0.30 or
              ((gs["transition_gauge_jump_fraction"] or 0) >= 0.50 and
               (gs["source_gradient_agreement"] or 1.0) <= 0.55))
    evidence["B4"] = {"pair_gauge_jump_unseen_fraction": ps["gauge_jump_unseen_fraction"],
                      "transition_gauge_jump_fraction": gs["transition_gauge_jump_fraction"],
                      "source_gradient_agreement": gs["source_gradient_agreement"],
                      "interpretation": "gauge jumps are common while source transition alignment is weak"}
    if b4:
        cases.append(schema.B4)

    b5 = bool((gs["source_gradient_agreement"] or 0) >= 0.50 and (gs["mean_abs_source_transition_gradient"] or 0) <= 0.05)
    evidence["B5"] = {"source_gradient_agreement": gs["source_gradient_agreement"],
                      "mean_abs_source_transition_gradient": gs["mean_abs_source_transition_gradient"]}
    if b5:
        cases.append(schema.B5)

    b6 = bool((ls["target_unlabeled_pm1_top1_gain_vs_source"] or 0) > 0.05 or
              (ls["target_unlabeled_pm2_top1_gain_vs_source"] or 0) > 0.05)
    evidence["B6"] = {"target_unlabeled_pm1_top1_gain_vs_source": ls["target_unlabeled_pm1_top1_gain_vs_source"],
                      "target_unlabeled_pm2_top1_gain_vs_source": ls["target_unlabeled_pm2_top1_gain_vs_source"]}
    if b6:
        cases.append(schema.B6)
    else:
        cases.append(schema.B7)

    b9 = bool((pls["mean_plateau_size"] or 0) >= 2.0 and (pls["selected_bad_plateau_has_joint_fraction"] or 0) >= 0.30)
    evidence["B9"] = {"mean_plateau_size": pls["mean_plateau_size"],
                      "selected_bad_plateau_has_joint_fraction": pls["selected_bad_plateau_has_joint_fraction"]}
    if b9:
        cases.append(schema.B9)

    if robust is not None:
        changed = bool(set(robust["taxonomy"]["cases"]) != set(cases)) if "taxonomy" in robust else False
        coverage_drop = ((bs["pm1_contains_joint_fraction"] or 0) - (robust["boundary"]["summary"]["pm1_contains_joint_fraction"] or 0)) if "boundary" in robust else 0
        evidence["B8"] = {"robust_case_change": changed, "pm1_contains_joint_fraction_drop": coverage_drop}
        if changed or coverage_drop >= 0.10:
            cases.append(schema.B8)

    tail = bool((ps["median_order_delta"] or 0) <= 1 and (ps["source_wrong_fraction"] or 0) < 0.30 and
                (bs["pm1_contains_joint_fraction"] or 0) >= 0.70)
    evidence["B10"] = {"median_order_delta": ps["median_order_delta"],
                       "source_wrong_fraction": ps["source_wrong_fraction"],
                       "pm1_contains_joint_fraction": bs["pm1_contains_joint_fraction"]}
    if tail and not (b2 or b3 or b4):
        cases.append(schema.B10)

    if not cases:
        cases.append(schema.B11)
    return {"cases": cases, "evidence": evidence}
