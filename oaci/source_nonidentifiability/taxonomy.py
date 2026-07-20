"""C45 deterministic taxonomy."""
from __future__ import annotations

from . import schema


def classify(witness, variance, family, gauge, lower, registry_summary):
    wt = witness["summary"]["within_trajectory"]
    ct = witness["summary"]["cross_target"]
    sr = witness["summary"]["same_regime"]
    q05 = variance["summary"]["q05"]
    q10_var = variance["summary"]["q10"]
    q05_lb = lower["summary"]["q05"]
    q10_lb = lower["summary"]["q10"]
    fam = family["summary"]
    rank = fam["rank_only"]
    reduced = [r for k, r in fam.items() if k != "all_source_objectives"]
    min_reduced_divergent = min(r["source_equivalent_q10_target_divergent_rate"] for r in reduced)
    rank_reduction_vs_baseline = (
        rank["baseline_joint_good_disagreement_rate"] -
        rank["nearest_joint_good_disagreement_rate"]
    )
    gauge_corr = gauge["summary"]["gauge_gap_target_utility_gap_corr"]
    established = {
        schema.N1: (
            max(wt["source_equivalent_q10_target_divergent_rate"],
                ct["source_equivalent_q10_target_divergent_rate"],
                sr["source_equivalent_q10_target_divergent_rate"]) >= schema.FREQUENT_WITNESS_GATE and
            witness["summary"]["source_equivalent_target_divergent_pair_count"] > 0
        ),
        schema.N2: (
            witness["summary"]["trajectories_with_source_equivalent_divergent_pair_fraction"] >=
            schema.TRAJECTORY_WITNESS_GATE
        ),
        schema.N3: (
            q05["target_utility_variance_over_baseline"] >= schema.VARIANCE_PERSIST_RATIO_GATE or
            q05["joint_entropy_over_baseline"] >= schema.VARIANCE_PERSIST_RATIO_GATE
        ),
        schema.N4: (
            q10_var["joint_entropy_over_baseline"] >= schema.NONDISCRIMINATIVE_RATIO_GATE or
            wt["joint_good_disagreement_rate"] >=
            schema.NONDISCRIMINATIVE_RATIO_GATE * wt["baseline_joint_good_disagreement_rate"]
        ),
        schema.N5: (
            gauge["summary"]["source_equivalent_gauge_witness_fraction"] >= schema.GAUGE_WITNESS_RATE_GATE and
            gauge_corr is not None and abs(gauge_corr) >= schema.GAUGE_UTILITY_CORR_GATE
        ),
        schema.N6: (
            min_reduced_divergent >= schema.FAMILY_AMBIGUITY_GATE
        ),
        schema.N7: (
            rank_reduction_vs_baseline >= schema.RANK_REDUCTION_MIN and
            rank["source_equivalent_q10_target_divergent_rate"] >= schema.FAMILY_AMBIGUITY_GATE
        ),
        schema.N8: (
            q05_lb["minimum_unavoidable_ambiguity_rate"] >= schema.LOWER_BOUND_GATE or
            q10_lb["ambiguous_neighborhood_fraction"] >= schema.FREQUENT_WITNESS_GATE
        ),
        schema.N9: not registry_summary["primary_source_objectives_complete"],
    }
    evidence = {
        schema.N1: (
            "within_traj_q10_divergent_rate=%s, cross_target_q10_divergent_rate=%s, "
            "same_regime_q10_divergent_rate=%s, pair_count=%s" % (
                wt["source_equivalent_q10_target_divergent_rate"],
                ct["source_equivalent_q10_target_divergent_rate"],
                sr["source_equivalent_q10_target_divergent_rate"],
                witness["summary"]["source_equivalent_target_divergent_pair_count"])),
        schema.N2: (
            "trajectory_fraction=%s" %
            witness["summary"]["trajectories_with_source_equivalent_divergent_pair_fraction"]),
        schema.N3: (
            "q05_variance_over_baseline=%s, q05_joint_entropy_over_baseline=%s" % (
                q05["target_utility_variance_over_baseline"], q05["joint_entropy_over_baseline"])),
        schema.N4: (
            "q10_joint_entropy_over_baseline=%s, nearest_joint_disagreement=%s, baseline=%s" % (
                q10_var["joint_entropy_over_baseline"], wt["joint_good_disagreement_rate"],
                wt["baseline_joint_good_disagreement_rate"])),
        schema.N5: (
            "gauge_witness_fraction=%s, gauge_utility_corr=%s" % (
                gauge["summary"]["source_equivalent_gauge_witness_fraction"], gauge_corr)),
        schema.N6: (
            "min_reduced_q10_divergent=%s" % min_reduced_divergent),
        schema.N7: (
            "rank_reduction_vs_baseline=%s, rank_q10_divergent=%s" % (
                rank_reduction_vs_baseline, rank["source_equivalent_q10_target_divergent_rate"])),
        schema.N8: (
            "q05_unavoidable=%s, q10_ambiguous_fraction=%s" % (
                q05_lb["minimum_unavoidable_ambiguity_rate"],
                q10_lb["ambiguous_neighborhood_fraction"])),
        schema.N9: "primary_source_objectives_complete=%s" % (
            registry_summary["primary_source_objectives_complete"]),
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {"cases": [c for c in schema.ALL_CASES if established[c]], "case_rows": rows,
            "established": established, "evidence": evidence}
