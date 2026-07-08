"""C46 deterministic conditioning-boundary taxonomy."""
from __future__ import annotations

from . import schema


def _vd(vdecomp, outcome, component):
    return vdecomp["summary"][(outcome, component)]["eta_squared"]


def classify(neighbor, condvar, vdecomp, usefulness):
    ns = neighbor["summary"]
    cv = condvar["summary"]
    du = usefulness["summary"]
    within_traj = ns["within_trajectory"]["source_equivalent_q10_target_divergent_rate"]
    within_target = ns["within_target"]["source_equivalent_q10_target_divergent_rate"]
    cross_target = ns["cross_target"]["source_equivalent_q10_target_divergent_rate"]
    same_regime = ns["within_regime"]["source_equivalent_q10_target_divergent_rate"]
    cross_regime = ns["cross_regime"]["source_equivalent_q10_target_divergent_rate"]
    target_eta = _vd(vdecomp, "target_utility_score", "target")
    traj_resid = _vd(vdecomp, "target_utility_score", "residual_within_trajectory")
    target_var_ratio = cv["target"]["target_utility_variance_over_global"]
    traj_var_ratio = cv["trajectory"]["target_utility_variance_over_global"]
    within_dist_signal = du["within_trajectory"]["source_distance_target_utility_gap_spearman"]
    cross_dist_signal = du["cross_target"]["source_distance_target_utility_gap_spearman"]
    established = {
        schema.CB1: (
            within_target <= schema.WITHIN_TARGET_STRONG_GATE and
            within_traj <= schema.WITHIN_HOMOGENEITY_GATE and
            cross_target >= schema.CROSS_TARGET_DIVERGENCE_GATE
        ),
        schema.CB2: (
            cross_target >= schema.CROSS_TARGET_DIVERGENCE_GATE and
            within_target <= schema.WITHIN_TARGET_STRONG_GATE
        ),
        schema.CB3: (
            within_traj <= schema.WITHIN_HOMOGENEITY_GATE and
            traj_var_ratio < 0.50
        ),
        schema.CB4: (
            cross_target >= schema.CROSS_TARGET_DIVERGENCE_GATE and
            cross_dist_signal is not None and abs(cross_dist_signal) < 0.05
        ),
        schema.CB5: target_eta >= schema.TARGET_COMPONENT_GATE,
        schema.CB6: (
            same_regime >= schema.SAME_REGIME_INTERMEDIATE_GATE and
            same_regime < cross_target
        ),
        schema.CB7: False,
    }
    evidence = {
        schema.CB1: (
            f"within_target_q10={within_target}, within_traj_q10={within_traj}, "
            f"cross_target_q10={cross_target}"),
        schema.CB2: (
            f"cross_target_q10={cross_target}, within_target_q10={within_target}, "
            f"target_conditional_variance_ratio={target_var_ratio}"),
        schema.CB3: (
            f"within_traj_q10={within_traj}, trajectory_conditional_variance_ratio={traj_var_ratio}"),
        schema.CB4: (
            f"cross_target_q10={cross_target}, cross_regime_q10={cross_regime}, "
            f"cross_target_distance_gap_spearman={cross_dist_signal}"),
        schema.CB5: (
            f"target_eta2={target_eta}, residual_within_trajectory_fraction={traj_resid}"),
        schema.CB6: (
            f"within_regime_q10={same_regime}, cross_target_q10={cross_target}"),
        schema.CB7: "required inherited artifacts available",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {"cases": [c for c in schema.ALL_CASES if established[c]], "case_rows": rows,
            "established": established, "evidence": evidence}
