"""C44 deterministic taxonomy."""
from __future__ import annotations

from . import schema


def classify(nulls, effdim, families, discrim, depth):
    null_summary = nulls["summary"]
    observed = null_summary["observed_front_fraction"]
    gaussian = null_summary["gaussian_null_front_fraction"]
    objective_shuffled = null_summary["objective_shuffled_front_fraction"]
    joint_disc = discrim["summary"]["joint_good"]
    co = discrim["cooccupancy_summary"]
    all_family = families["summary"]["all_families"]
    reduced = [r for k, r in families["summary"].items() if k != "all_families"]
    min_reduced_front = min(r["mean_front_fraction"] for r in reduced)
    max_reduced_joint_enrichment = max(r["mean_front_joint_good_enrichment"] for r in reduced)
    max_reduced_depth_auc = max(r["mean_depth_auc_vs_target_utility"] for r in reduced)
    diagnostic_subsets = [
        r for r in reduced
        if r["mean_front_joint_good_enrichment"] >= schema.WEAK_ENRICHMENT_GATE and
        r["mean_depth_auc_vs_target_utility"] > schema.DEPTH_SIGNAL_LOW
    ]
    best_diagnostic_subset = max(
        diagnostic_subsets,
        key=lambda r: (r["mean_front_joint_good_enrichment"], r["mean_depth_auc_vs_target_utility"]),
        default=None,
    )
    reduced_narrows = min_reduced_front <= schema.REDUCED_FRONT_NARROW_GATE
    reduced_loses_coverage = any(
        all_family["mean_joint_good_front_fraction"] - r["mean_joint_good_front_fraction"] >=
        schema.REDUCED_FRONT_COVERAGE_LOSS for r in reduced)
    depth_auc = depth["summary"]["mean_layer_auc_vs_target_utility"]
    depth_weak = schema.DEPTH_SIGNAL_LOW < depth_auc < schema.DEPTH_SIGNAL_HIGH
    depth_none = depth_auc <= schema.DEPTH_SIGNAL_LOW
    front_delta = abs(joint_disc["mean_p_label_given_front"] - joint_disc["mean_trajectory_baseline"])
    established = {
        schema.PF1: observed >= schema.FRONT_DEGENERATE_FRACTION and
        abs(observed - gaussian) <= schema.FRONT_NULL_CLOSE_DELTA,
        schema.PF2: observed >= schema.FRONT_DEGENERATE_FRACTION and
        front_delta <= schema.FRONT_NONDISCRIM_DELTA,
        schema.PF3: effdim["summary"]["negative_pair_fraction"] >= schema.CONFLICT_NEGATIVE_FRACTION or
        (effdim["summary"]["leakage_rank_mean_spearman"] or 0) <= -0.20,
        schema.PF4: effdim["summary"]["effective_rank"] >= schema.EFFECTIVE_RANK_MIN,
        schema.PF5: reduced_narrows and reduced_loses_coverage,
        schema.PF6: best_diagnostic_subset is not None,
        schema.PF7: depth_none,
        schema.PF8: depth_weak,
        schema.PF9: observed >= schema.FRONT_DEGENERATE_FRACTION and
        front_delta <= schema.FRONT_NONDISCRIM_DELTA and
        max(r["mean_p_joint_good_given_front"] for r in reduced) < 0.70,
        schema.PF10: False,
    }
    evidence = {
        schema.PF1: (
            f"observed_front={observed}, gaussian_null={gaussian}, "
            f"objective_shuffled_null={objective_shuffled}"),
        schema.PF2: (
            f"front_both_good_bad={co['front_contains_both_good_and_bad_fraction']}, "
            f"p_good_front={joint_disc['mean_p_label_given_front']}, "
            f"baseline={joint_disc['mean_trajectory_baseline']}"),
        schema.PF3: (
            f"negative_pair_fraction={effdim['summary']['negative_pair_fraction']}, "
            f"leakage_rank_mean={effdim['summary']['leakage_rank_mean_spearman']}"),
        schema.PF4: (
            f"effective_rank={effdim['summary']['effective_rank']}, "
            f"pca_var1={effdim['summary']['pca_var1']}"),
        schema.PF5: (
            f"min_reduced_front={min_reduced_front}, reduced_loses_coverage={reduced_loses_coverage}"),
        schema.PF6: (
            "best_subset=%s, enrichment=%s, depth_auc=%s" % (
                best_diagnostic_subset["subset"] if best_diagnostic_subset else None,
                best_diagnostic_subset["mean_front_joint_good_enrichment"] if best_diagnostic_subset else None,
                best_diagnostic_subset["mean_depth_auc_vs_target_utility"] if best_diagnostic_subset else None)),
        schema.PF7: f"depth_auc={depth_auc}",
        schema.PF8: f"depth_auc={depth_auc}",
        schema.PF9: "frontier broad + non-discriminative; no reduced-family reliable geometry",
        schema.PF10: "objective availability complete",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {"cases": [c for c in schema.ALL_CASES if established[c]], "case_rows": rows,
            "established": established, "evidence": evidence}
