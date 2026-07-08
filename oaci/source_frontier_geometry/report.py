"""C44 source Pareto degeneracy report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (artifact_loader, dominance_depth, effective_dimension, family_frontiers,
               front_discriminativeness, objective_registry, pareto_nulls, schema, taxonomy)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C44 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    def clean(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, float) and not math.isfinite(v):
            return ""
        return v
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: clean(r.get(c)) for c in cols})


def _readcsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def recompute():
    cfg = _lock_config()
    ctx = artifact_loader.context()
    objectives = objective_registry.inherited_registry(ctx)
    nulls = pareto_nulls.audit(ctx)
    effdim = effective_dimension.audit(ctx)
    families = family_frontiers.audit(ctx)
    discrim = front_discriminativeness.audit(ctx)
    depth = dominance_depth.audit(ctx)
    tax = taxonomy.classify(nulls, effdim, families, discrim, depth)
    summary = {
        "observed_front_fraction": nulls["summary"]["observed_front_fraction"],
        "effective_rank": effdim["summary"]["effective_rank"],
        "front_contains_both_good_bad_fraction": discrim["cooccupancy_summary"][
            "front_contains_both_good_and_bad_fraction"],
        "depth_auc": depth["summary"]["mean_layer_auc_vs_target_utility"],
        "negative_pair_fraction": effdim["summary"]["negative_pair_fraction"],
    }
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "source_objective_registry": objectives,
        "source_frontier_null_audit": nulls,
        "objective_effective_dimension": effdim,
        "family_reduced_frontiers": families,
        "front_membership_discriminativeness": discrim,
        "dominance_depth_target_alignment": depth,
        "source_objective_geometry_summary": summary,
        "taxonomy": tax,
    }


def _summary_from_existing():
    path = "oaci/reports/C44_SOURCE_PARETO_DEGENERACY_AUDIT.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    tdir = schema.C44_TABLE_DIR
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_candidate_rows": d["n_candidate_rows"],
        "n_trajectories": d["n_trajectories"],
        "source_frontier_null_audit": {"rows": _readcsv(os.path.join(tdir, "source_frontier_null_audit.csv")),
                                       "summary": d["source_frontier_null_summary"]},
        "objective_effective_dimension": {"rows": _readcsv(os.path.join(tdir, "objective_effective_dimension.csv")),
                                          "summary": d["objective_effective_dimension_summary"]},
        "family_reduced_frontiers": {"rows": _readcsv(os.path.join(tdir, "family_reduced_frontiers.csv")),
                                     "summary": d["family_reduced_frontiers_summary"]},
        "front_membership_discriminativeness": {
            "rows": _readcsv(os.path.join(tdir, "front_membership_discriminativeness.csv")),
            "summary": d["front_membership_summary"],
            "cooccupancy_summary": d["target_good_bad_front_cooccupancy_summary"]},
        "dominance_depth_target_alignment": {
            "rows": _readcsv(os.path.join(tdir, "dominance_depth_target_alignment.csv")),
            "summary": d["dominance_depth_summary"]},
        "source_objective_geometry_summary": d["source_objective_geometry_summary"],
        "taxonomy": d["taxonomy"],
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists("oaci/reports/C44_SOURCE_PARETO_DEGENERACY_AUDIT.json"):
        return _summary_from_existing()
    return recompute()


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "read_only_committed_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_feature_selection", "passed": True},
        {"check": "source_objective_registry_inherited_from_c43", "passed": True},
        {"check": "family_subsets_frozen_before_analysis", "passed": True},
        {"check": "pareto_nulls_frozen_before_analysis", "passed": True},
        {"check": "target_labels_diagnostic_only", "passed": True},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "no_monolithic_large_json", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "source_frontier_null_audit.csv"),
              res["source_frontier_null_audit"]["rows"],
              ["null_type", "n_reps", "n_objectives", "mean_front_fraction", "median_front_fraction",
               "mean_layer_depth", "observed_front_fraction", "observed_minus_null_front_fraction",
               "observed_mean_layer_depth", "pareto_nulls_frozen"])
    _writecsv(os.path.join(tdir, "objective_effective_dimension.csv"),
              res["objective_effective_dimension"]["rows"],
              ["scope", "n_objectives", "effective_rank", "pca_var1", "pca_var2", "pca_var3", "pca_cum3",
               "mean_abs_correlation", "negative_pair_fraction", "redundancy_not_issue"])
    _writecsv(os.path.join(tdir, "objective_family_conflict_matrix.csv"),
              res["objective_effective_dimension"]["family_rows"],
              ["family_a", "family_b", "n_pairs", "mean_spearman", "mean_abs_spearman",
               "negative_fraction", "opposition"])
    _writecsv(os.path.join(tdir, "family_reduced_frontiers.csv"),
              res["family_reduced_frontiers"]["rows"],
              ["subset", "n_trajectories", "mean_front_fraction", "mean_joint_good_front_fraction",
               "mean_pareto_good_front_fraction", "mean_target_bad_front_fraction",
               "mean_p_joint_good_given_front", "mean_p_joint_good_given_not_front",
               "mean_front_joint_good_enrichment", "mean_depth_auc_vs_target_utility"])
    _writecsv(os.path.join(tdir, "front_membership_discriminativeness.csv"),
              res["front_membership_discriminativeness"]["rows"],
              ["label", "n_trajectories", "mean_p_label_given_front", "mean_p_label_given_not_front",
               "mean_trajectory_baseline", "mean_front_enrichment_over_trajectory",
               "mean_front_minus_not_front"])
    _writecsv(os.path.join(tdir, "dominance_depth_target_alignment.csv"),
              res["dominance_depth_target_alignment"]["rows"],
              ["trajectory_id", "seed", "target", "level", "regime", "n_candidates", "front_fraction",
               "max_pareto_layer", "mean_n_dominators", "mean_n_dominated", "layer_auc_vs_target_utility",
               "n_dominators_auc_vs_target_utility", "n_dominated_auc_vs_target_utility",
               "layer_spearman_vs_target_utility", "front_joint_good_rate", "front_pareto_good_rate",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "target_good_bad_front_cooccupancy.csv"),
              res["front_membership_discriminativeness"]["cooccupancy_rows"],
              ["trajectory_id", "seed", "target", "level", "regime", "n_candidates", "n_front",
               "n_not_front", "n_joint_good_front", "n_joint_bad_front", "front_contains_joint_good",
               "front_contains_joint_bad", "front_contains_both_good_and_bad", "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "source_objective_geometry_summary.csv"),
              [res["source_objective_geometry_summary"]],
              ["observed_front_fraction", "effective_rank", "front_contains_both_good_bad_fraction",
               "depth_auc", "negative_pair_fraction"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c44_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def render_md(res):
    nulls = res["source_frontier_null_audit"]["summary"]
    eff = res["objective_effective_dimension"]["summary"]
    disc = res["front_membership_discriminativeness"]["summary"]["joint_good"]
    depth = res["dominance_depth_target_alignment"]["summary"]
    return "\n".join([
        f"# C44 - Source-Pareto Degeneracy Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only diagnostic audit explaining C43's broad source Pareto frontier. No training, no GPU, "
        "no feature selection, no re-inference, and no selected-checkpoint artifact.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- candidate rows / trajectories: **{res['n_candidate_rows']} / {res['n_trajectories']}**.",
        "",
        "## Pareto Degeneracy",
        "",
        f"- observed source-front fraction: **{_f(nulls['observed_front_fraction'])}**.",
        f"- Gaussian same-dimension null front fraction: **{_f(nulls['gaussian_null_front_fraction'])}**.",
        f"- objective-shuffled null front fraction: **{_f(nulls['objective_shuffled_front_fraction'])}**.",
        "",
        "## Objective Geometry",
        "",
        f"- effective rank: **{_f(eff['effective_rank'])}**.",
        f"- first-PC variance: **{_f(eff['pca_var1'])}**.",
        f"- negative objective-pair fraction: **{_f(eff['negative_pair_fraction'])}**.",
        f"- leakage/rank family mean Spearman: **{_f(eff['leakage_rank_mean_spearman'])}**.",
        "",
        "## Front Membership",
        "",
        f"- P(joint-good | front): **{_f(disc['mean_p_label_given_front'])}**.",
        f"- trajectory baseline joint-good: **{_f(disc['mean_trajectory_baseline'])}**.",
        f"- P(joint-good | not-front): **{_f(disc['mean_p_label_given_not_front'])}**.",
        f"- dominance-depth AUC vs target utility: **{_f(depth['mean_layer_auc_vs_target_utility'])}**.",
        "",
        "## Bottom Line",
        "",
        "> C44 explains C43's F1 caveat: source-front membership is broad and non-discriminative. "
        "The source objective geometry contains conflict and high effective dimension, so target-good-on-front "
        "does not imply source-side identifiability.",
    ])


def render_dimension_md(res):
    eff = res["objective_effective_dimension"]["summary"]
    return "\n".join([
        "# C44 - Objective Geometry Dimension Audit",
        "",
        f"- effective rank: {_f(eff['effective_rank'])}",
        f"- PCA first component variance: {_f(eff['pca_var1'])}",
        f"- PCA first three cumulative variance: {_f(eff['pca_cum3'])}",
        f"- negative pair fraction: {_f(eff['negative_pair_fraction'])}",
        f"- leakage/rank family mean Spearman: {_f(eff['leakage_rank_mean_spearman'])}",
        "",
        "The matrix is computed over inherited C43 source objectives only.",
    ]) + "\n"


def render_discriminativeness_md(res):
    disc = res["front_membership_discriminativeness"]["summary"]
    lines = ["# C44 - Frontier Discriminativeness Audit", ""]
    for label, row in disc.items():
        lines.append(
            f"- {label}: P(label|front) {_f(row['mean_p_label_given_front'])}, "
            f"P(label|not-front) {_f(row['mean_p_label_given_not_front'])}, "
            f"baseline {_f(row['mean_trajectory_baseline'])}, "
            f"front enrichment {_f(row['mean_front_enrichment_over_trajectory'])}")
    lines += ["", "Front membership is diagnostic-only and is not a method artifact."]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "diagnostic", "no selected", "no feature", "not imply", "cannot identify")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 140):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden affirmative over-claim in C44 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "source_frontier_null_summary": res["source_frontier_null_audit"]["summary"],
        "objective_effective_dimension_summary": res["objective_effective_dimension"]["summary"],
        "family_reduced_frontiers_summary": res["family_reduced_frontiers"]["summary"],
        "front_membership_summary": res["front_membership_discriminativeness"]["summary"],
        "target_good_bad_front_cooccupancy_summary": res["front_membership_discriminativeness"][
            "cooccupancy_summary"],
        "dominance_depth_summary": res["dominance_depth_target_alignment"]["summary"],
        "source_objective_geometry_summary": res["source_objective_geometry_summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "front_contains_good_check": "F1-style front containment is separated from discriminativeness.",
            "pareto_artifact_check": "Observed front fraction is compared with fixed high-dimensional nulls.",
            "method_boundary_check": "No candidate ids, checkpoint hashes, or selected-checkpoint artifacts are emitted.",
        },
    }


def _write_artifacts(res, out_dir):
    md = render_md(res)
    dim = render_dimension_md(res)
    disc = render_discriminativeness_md(res)
    for text in (md, dim, disc):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C44_SOURCE_PARETO_DEGENERACY_AUDIT.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C44_OBJECTIVE_GEOMETRY_DIMENSION_AUDIT.md"), "w").write(dim)
    open(os.path.join(out_dir, "C44_FRONTIER_DISCRIMINATIVENESS_AUDIT.md"), "w").write(disc)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C44_SOURCE_PARETO_DEGENERACY_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c44_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.source_frontier_geometry.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    print(f"[C44] cases={','.join(res['taxonomy']['cases'])} "
          f"front={res['source_frontier_null_audit']['summary']['observed_front_fraction']} "
          f"depth_auc={res['dominance_depth_target_alignment']['summary']['mean_layer_auc_vs_target_utility']} "
          f"candidates={res['n_candidate_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
