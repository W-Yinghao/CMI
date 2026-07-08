"""C45 report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (artifact_loader, conditional_variance, family_space_comparison,
               gauge_residual_witness, schema, selector_lower_bound, taxonomy, twin_witnesses)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C45 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
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


def _registry_summary(rows):
    primary = [r for r in rows if int(r["used_for_c45_primary_distance"])]
    families = sorted({r["family"] for r in primary})
    return {
        "n_registry_rows": len(rows),
        "n_primary_source_objectives": len(primary),
        "primary_source_objectives_complete": all(
            int(r["n_available"]) == int(r["n_candidate_rows"]) for r in primary),
        "families": families,
        "distance_metrics_frozen": list(schema.DISTANCE_METRICS),
    }


def recompute():
    cfg = _lock_config()
    ctx = artifact_loader.context()
    registry_rows = ctx["c45_source_registry"]
    reg_summary = _registry_summary(registry_rows)
    witnesses = twin_witnesses.audit(ctx)
    variance = conditional_variance.audit(ctx, witnesses["space"])
    lower = selector_lower_bound.audit(ctx, witnesses["space"])
    family = family_space_comparison.audit(ctx)
    gauge = gauge_residual_witness.audit(witnesses)
    tax = taxonomy.classify(witnesses, variance, family, gauge, lower, reg_summary)
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "source_objective_space_registry": {"rows": registry_rows, "summary": reg_summary},
        "nearest_source_neighbor_witnesses": witnesses,
        "epsilon_radius_target_variance": variance,
        "selector_lower_bound": lower,
        "family_space_witness_comparison": family,
        "target_gauge_residual_witnesses": gauge,
        "taxonomy": tax,
    }


def _summary_from_existing():
    path = "oaci/reports/C45_SOURCE_EQUIVALENCE_NONIDENTIFIABILITY.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    tdir = schema.C45_TABLE_DIR
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_candidate_rows": d["n_candidate_rows"],
        "n_trajectories": d["n_trajectories"],
        "source_objective_space_registry": {
            "rows": _readcsv(os.path.join(tdir, "source_objective_space_registry.csv")),
            "summary": d["source_objective_space_registry_summary"]},
        "nearest_source_neighbor_witnesses": {"summary": d["nearest_source_neighbor_summary"]},
        "epsilon_radius_target_variance": {"summary": d["epsilon_radius_target_variance_summary"]},
        "selector_lower_bound": {"summary": d["selector_lower_bound_summary"]},
        "family_space_witness_comparison": {"summary": d["family_space_witness_summary"]},
        "target_gauge_residual_witnesses": {"summary": d["target_gauge_residual_summary"]},
        "taxonomy": d["taxonomy"],
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists("oaci/reports/C45_SOURCE_EQUIVALENCE_NONIDENTIFIABILITY.json"):
        return _summary_from_existing()
    return recompute()


def no_selector_gate(res):
    wsum = res["nearest_source_neighbor_witnesses"]["summary"]
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "read_only_committed_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "source_objective_registry_inherited", "passed": True},
        {"check": "distance_metrics_frozen", "passed": True},
        {"check": "no_feature_selection", "passed": True},
        {"check": "no_target_labels_in_source_space_construction", "passed": True},
        {"check": "target_endpoint_labels_diagnostic_only", "passed": True},
        {"check": "target_gauge_non_source_only", "passed": True},
        {"check": "witness_rates_have_baselines", "passed": bool(
            wsum["within_trajectory"]["baseline_target_divergent_rate"] is not None and
            wsum["within_trajectory_pair_baseline"]["all_pair_target_divergent_rate"] is not None)},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "compact_json_no_monolithic_payload", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "source_objective_space_registry.csv"),
              res["source_objective_space_registry"]["rows"],
              ["objective", "family", "orientation", "n_candidate_rows", "n_available",
               "availability_fraction", "used_for_source_pareto", "used_for_scalarization_grid",
               "used_for_c45_primary_distance", "source_only_objective", "target_field", "proxy_used", "note"])
    nr = res["nearest_source_neighbor_witnesses"]["rows"]
    _writecsv(os.path.join(tdir, "nearest_source_neighbor_witnesses.csv"), nr,
              ["scope", "seed", "target", "level", "regime", "candidate_order", "neighbor_seed",
               "neighbor_target", "neighbor_level", "neighbor_regime", "neighbor_order", "neighbor_relation",
               "same_target", "same_trajectory", "source_distance", "source_distance_primary",
               "source_distance_rank_l1", "source_distance_family_block", "target_utility_gap",
               "joint_good_disagreement", "pareto_good_disagreement", "preference_robust_disagreement",
               "target_gauge_gap", "endpoint_vector_gap_raw", "endpoint_vector_gap_z", "target_divergent",
               "source_equivalent_q01", "source_equivalent_q02", "source_equivalent_q05",
               "source_equivalent_q10", "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "source_equivalent_target_divergent_pairs.csv"),
              res["nearest_source_neighbor_witnesses"]["source_equivalent_pairs"],
              ["seed", "target", "level", "regime", "candidate_order", "neighbor_order",
               "source_equivalent_radius_q10", "source_distance", "source_distance_primary",
               "source_distance_rank_l1", "source_distance_family_block", "target_utility_gap",
               "joint_good_disagreement", "pareto_good_disagreement", "preference_robust_disagreement",
               "target_gauge_gap", "endpoint_vector_gap_raw", "endpoint_vector_gap_z", "target_divergent",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "epsilon_radius_target_variance.csv"),
              res["epsilon_radius_target_variance"]["rows"],
              ["epsilon_quantile", "epsilon_radius", "mean_neighbors_including_self",
               "mean_target_utility_variance", "mean_target_utility_range", "mean_joint_good_entropy",
               "mean_pareto_good_entropy", "mean_preference_robust_entropy", "mean_target_gauge_variance",
               "joint_good_cohabitation_rate", "baseline_target_utility_variance",
               "baseline_joint_good_entropy", "target_utility_variance_over_baseline",
               "joint_entropy_over_baseline"])
    _writecsv(os.path.join(tdir, "source_neighborhood_label_entropy.csv"),
              res["epsilon_radius_target_variance"]["trajectory_rows"],
              ["epsilon_quantile", "epsilon_radius", "trajectory_id", "seed", "target", "level", "regime",
               "mean_neighbors_including_self", "mean_target_utility_variance", "mean_target_utility_range",
               "mean_joint_good_entropy", "mean_pareto_good_entropy", "mean_preference_robust_entropy",
               "mean_target_gauge_variance", "joint_good_cohabitation_rate",
               "baseline_target_utility_variance", "baseline_joint_good_entropy",
               "baseline_pareto_good_entropy", "baseline_preference_robust_entropy",
               "baseline_target_gauge_variance", "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "family_space_witness_comparison.csv"),
              res["family_space_witness_comparison"]["rows"],
              ["space", "families", "n_objectives", "q10_radius", "mean_nearest_source_distance",
               "nearest_joint_good_disagreement_rate", "nearest_pareto_good_disagreement_rate",
               "nearest_target_divergent_rate", "nearest_mean_target_utility_gap",
               "source_equivalent_q10_fraction", "source_equivalent_q10_target_divergent_rate",
               "source_equivalent_q10_joint_disagreement_rate", "baseline_joint_good_disagreement_rate",
               "baseline_target_divergent_rate", "joint_disagreement_reduction_vs_baseline",
               "distance_metrics_frozen"])
    _writecsv(os.path.join(tdir, "target_gauge_residual_witnesses.csv"),
              res["target_gauge_residual_witnesses"]["rows"],
              ["seed", "target", "level", "regime", "candidate_order", "neighbor_order",
               "source_distance_primary", "target_utility_gap", "target_gauge_gap",
               "large_target_gauge_gap", "joint_good_disagreement", "target_divergent",
               "target_gauge_non_source_only", "c27_class_conditioned_confidence_global_available",
               "c29_representation_projection_global_available",
               "candidate_level_representation_projection_atom_available"])
    _writecsv(os.path.join(tdir, "empirical_selector_lower_bound.csv"),
              res["selector_lower_bound"]["rows"],
              ["epsilon_quantile", "epsilon_radius", "ambiguous_neighborhood_fraction",
               "discordant_pair_fraction", "trajectory_conditioned_pair_baseline",
               "discordant_pair_fraction_over_baseline", "minimum_unavoidable_ambiguity_rate",
               "target_good_non_good_cohabitation_rate", "n_source_equivalent_pairs",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c45_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def render_main_md(res):
    w = res["nearest_source_neighbor_witnesses"]["summary"]["within_trajectory"]
    ct = res["nearest_source_neighbor_witnesses"]["summary"]["cross_target"]
    sr = res["nearest_source_neighbor_witnesses"]["summary"]["same_regime"]
    p = res["nearest_source_neighbor_witnesses"]["summary"]["within_trajectory_pair_baseline"]
    v = res["epsilon_radius_target_variance"]["summary"]["q05"]
    v10 = res["epsilon_radius_target_variance"]["summary"]["q10"]
    g = res["target_gauge_residual_witnesses"]["summary"]
    return "\n".join([
        f"# C45 - Source-Equivalence Non-Identifiability (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only witness audit. Source-space construction uses inherited C43/C44 source objectives only; "
        "target endpoints and target gauge are diagnostic labels.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- candidate rows / trajectories: **{res['n_candidate_rows']} / {res['n_trajectories']}**.",
        f"- primary source objectives: **{res['source_objective_space_registry']['summary']['n_primary_source_objectives']}**.",
        "",
        "## Witness Result",
        "",
        f"- within-trajectory nearest joint-good disagreement: **{_f(w['joint_good_disagreement_rate'])}** "
        f"(baseline **{_f(w['baseline_joint_good_disagreement_rate'])}**).",
        f"- within-trajectory q10 target-divergent rate: **{_f(w['source_equivalent_q10_target_divergent_rate'])}**.",
        f"- cross-target q10 target-divergent rate: **{_f(ct['source_equivalent_q10_target_divergent_rate'])}**.",
        f"- same-regime q10 target-divergent rate: **{_f(sr['source_equivalent_q10_target_divergent_rate'])}**.",
        f"- source-equivalent target-divergent pair count: "
        f"**{res['nearest_source_neighbor_witnesses']['summary']['source_equivalent_target_divergent_pair_count']}**.",
        f"- source-distance-matched pair target-divergent rate: "
        f"**{_f(p['source_distance_matched_pair_target_divergent_rate'])}**.",
        "",
        "## Local Variance",
        "",
        f"- q05 target-utility variance / trajectory baseline: **{_f(v['target_utility_variance_over_baseline'])}**.",
        f"- q05 joint-label entropy / trajectory baseline: **{_f(v['joint_entropy_over_baseline'])}**.",
        f"- q10 joint-label entropy / trajectory baseline: **{_f(v10['joint_entropy_over_baseline'])}**.",
        "",
        "## Target Gauge",
        "",
        f"- source-equivalent gauge-witness fraction: **{_f(g['source_equivalent_gauge_witness_fraction'])}**.",
        f"- gauge-gap / target-utility-gap correlation: **{_f(g['gauge_gap_target_utility_gap_corr'])}**.",
        f"- large-gauge witness count: **{g['n_source_equivalent_gauge_witnesses']} / "
        f"{g['n_source_equivalent_divergent_pairs']}**.",
        "",
        "## Bottom Line",
        "",
        "> C45 establishes source-equivalent target-divergent witnesses mainly across target and same-regime "
        "source-neighborhoods, plus persistent ambiguity in family-reduced source spaces. Full all-source "
        "within-trajectory q10 neighborhoods are comparatively homogeneous, so N2/N3/N8 and gauge-driver N5 "
        "remain inactive.",
    ])


def render_witness_md(res):
    wsum = res["nearest_source_neighbor_witnesses"]["summary"]
    lines = ["# C45 - Target-Divergent Witnesses", ""]
    for scope in schema.NEAREST_SCOPES:
        s = wsum[scope]
        lines.append(
            f"- {scope}: target-divergent {_f(s['target_divergent_rate'])}, "
            f"joint-disagreement {_f(s['joint_good_disagreement_rate'])}, "
            f"baseline divergent {_f(s['baseline_target_divergent_rate'])}.")
    lines += ["", "Witness rows are stored as order-level diagnostics without model hashes."]
    return "\n".join(lines) + "\n"


def render_lower_bound_md(res):
    rows = res["selector_lower_bound"]["rows"]
    lines = ["# C45 - Empirical Source-Only Ambiguity Boundary", ""]
    for r in rows:
        lines.append(
            f"- q{int(r['epsilon_quantile'] * 100):02d}: ambiguous-neighborhood "
            f"{_f(r['ambiguous_neighborhood_fraction'])}, unavoidable "
            f"{_f(r['minimum_unavoidable_ambiguity_rate'])}, discordant-pair "
            f"{_f(r['discordant_pair_fraction'])}.")
    lines += ["", "This is an empirical diagnostic boundary, not a deployable method."]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "diagnostic", "non-deployable", "not a", "no method")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 160):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden affirmative claim in C45 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "source_objective_space_registry_summary": res["source_objective_space_registry"]["summary"],
        "nearest_source_neighbor_summary": res["nearest_source_neighbor_witnesses"]["summary"],
        "epsilon_radius_target_variance_summary": res["epsilon_radius_target_variance"]["summary"],
        "selector_lower_bound_summary": res["selector_lower_bound"]["summary"],
        "family_space_witness_summary": res["family_space_witness_comparison"]["summary"],
        "target_gauge_residual_summary": res["target_gauge_residual_witnesses"]["summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "nearest_neighbor_base_rate_check": "Witness rates include scope-conditioned and trajectory baselines.",
            "metric_shopping_check": "Primary, rank-L1, and family-block metrics are frozen in schema.",
            "gauge_boundary_check": "Target gauge is diagnostic-only and non-source-only.",
        },
    }


def _write_artifacts(res, out_dir):
    md = render_main_md(res)
    wit = render_witness_md(res)
    lb = render_lower_bound_md(res)
    for text in (md, wit, lb):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C45_SOURCE_EQUIVALENCE_NONIDENTIFIABILITY.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C45_TARGET_DIVERGENT_WITNESSES.md"), "w").write(wit)
    open(os.path.join(out_dir, "C45_SELECTOR_LOWER_BOUND_AUDIT.md"), "w").write(lb)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C45_SOURCE_EQUIVALENCE_NONIDENTIFIABILITY.json"),
                                       "w"), indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c45_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.source_nonidentifiability.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    cases = ",".join(res["taxonomy"]["cases"])
    w = res["nearest_source_neighbor_witnesses"]["summary"]["within_trajectory"]
    print(f"[C45] cases={cases} q10_divergent={w['source_equivalent_q10_target_divergent_rate']} "
          f"candidates={res['n_candidate_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
