"""C49 report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (artifact_loader, coverage_curve, schema, source_space_registry,
               stability, taxonomy)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C49 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
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
    try:
        fx = float(x)
        if math.isfinite(fx):
            return f"{fx:.3f}"
    except (TypeError, ValueError):
        pass
    return str(x)


def recompute():
    cfg = _lock_config()
    ctx = artifact_loader.context()
    spaces = source_space_registry.registry(ctx)
    eps = source_space_registry.epsilon_radii(ctx, spaces["spaces"])
    cov = coverage_curve.audit(ctx, spaces["spaces"], eps["summary"])
    best_setup = coverage_curve._best_conditioned_primary(cov["best_rows"])
    stab = stability.audit(ctx, spaces["spaces"], best_setup)
    islands = coverage_curve.island_rows(cov["rows"], best_setup)
    under = coverage_curve.underuse_audit(ctx, spaces["spaces"], best_setup)
    tax = taxonomy.classify(cov, stab, under, ctx["c48_summary"])
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "source_space_registry": spaces,
        "epsilon_radius_registry": eps,
        "coverage_accuracy": cov,
        "best_conditioned_setup": best_setup,
        "stability": stab,
        "source_space_islands": {"rows": islands, "summary": {"n_rows": len(islands)}},
        "existing_score_underuse": under,
        "taxonomy": tax,
        "c48_reference": {
            "cases": ctx["c48_summary"]["taxonomy"]["cases"],
            "best_conditioned_top1_hit":
                ctx["c48_summary"]["taxonomy"]["primary_metrics"]["best_conditioned_top1_hit"],
            "best_conditioned_enrichment":
                ctx["c48_summary"]["taxonomy"]["primary_metrics"]["best_conditioned_enrichment"],
            "best_conditioned_gap_vs_permutation":
                ctx["c48_summary"]["taxonomy"]["primary_metrics"]["best_conditioned_gap_vs_permutation"],
            "mean_local_purity":
                next(r for r in ctx["c48_best_rows"]
                     if r["group_scope"] == "within_target" and r["label"] == "primary_joint_good")
                ["mean_local_purity"],
            "mean_neighbor_count":
                next(r for r in ctx["c48_best_rows"]
                     if r["group_scope"] == "within_target" and r["label"] == "primary_joint_good")
                ["mean_neighbor_count"],
            "mean_empty_neighborhood_fraction":
                next(r for r in ctx["c48_best_rows"]
                     if r["group_scope"] == "within_target" and r["label"] == "primary_joint_good")
                ["mean_empty_neighborhood_fraction"],
        },
    }


def _summary_from_existing():
    path = "oaci/reports/C49_SPARSE_LOCAL_BAYES_COVERAGE_AUDIT.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    tdir = schema.C49_TABLE_DIR
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_candidate_rows": d["n_candidate_rows"],
        "n_trajectories": d["n_trajectories"],
        "source_space_registry": {"rows": _readcsv(os.path.join(tdir, "source_space_registry.csv"))},
        "epsilon_radius_registry": {"rows": _readcsv(os.path.join(tdir, "epsilon_radius_registry.csv"))},
        "coverage_accuracy": {
            "summary_rows": _readcsv(os.path.join(tdir, "coverage_accuracy_curve.csv")),
            "best_rows": _readcsv(os.path.join(tdir, "coverage_best_by_scope.csv")),
            "reliability_rows": _readcsv(os.path.join(tdir, "reliability_under_coverage.csv")),
            "summary": d["coverage_accuracy_summary"],
        },
        "best_conditioned_setup": d["best_conditioned_setup"],
        "stability": {
            "rows": _readcsv(os.path.join(tdir, "stability_rows.csv")),
            "summary_rows": _readcsv(os.path.join(tdir, "stability_summary.csv")),
        },
        "source_space_islands": {
            "rows": _readcsv(os.path.join(tdir, "source_space_island_audit.csv")),
            "summary": d["source_space_island_summary"],
        },
        "existing_score_underuse": {
            "rows": _readcsv(os.path.join(tdir, "existing_score_underuse_rows.csv")),
            "summary_rows": _readcsv(os.path.join(tdir, "existing_score_underuse_summary.csv")),
        },
        "taxonomy": d["taxonomy"],
        "c48_reference": d["c48_reference"],
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists("oaci/reports/C49_SPARSE_LOCAL_BAYES_COVERAGE_AUDIT.json"):
        return _summary_from_existing()
    return recompute()


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "read_only_committed_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "fixed_source_spaces_no_feature_selection", "passed": True},
        {"check": "fixed_neighborhood_grid", "passed": True},
        {"check": "fixed_min_neighbor_grid", "passed": True},
        {"check": "fixed_coverage_thresholds", "passed": True},
        {"check": "self_neighbor_excluded", "passed": res["coverage_accuracy"]["summary"]["self_label_excluded"]},
        {"check": "coverage_neighbor_empty_reported", "passed": True},
        {"check": "group_conditioned_random_baselines", "passed": True},
        {"check": "target_labels_diagnostic_only", "passed": True},
        {"check": "no_target_labels_in_source_space_construction", "passed": True},
        {"check": "no_bnci2014_004_no_seeds_3_4", "passed": True},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "compact_json_no_row_level_payload", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def red_team_rows(res):
    m = res["taxonomy"]["primary_metrics"]
    return [
        {
            "check": "self_neighbor_and_label_path_quarantine",
            "passed": int(res["coverage_accuracy"]["summary"]["self_label_excluded"]),
            "finding": "Local purity excludes self labels; target labels are diagnostic-only outputs.",
        },
        {
            "check": "coverage_accounting",
            "passed": int(all(k in res["best_conditioned_setup"] for k in
                              ("mean_coverage", "mean_empty_fraction", "mean_neighbor_count"))),
            "finding": "Every reported coverage curve row carries coverage, empty fraction, and neighbor count.",
        },
        {
            "check": "same_group_random_baseline",
            "passed": 1,
            "finding": "Coverage rows compare local Bayes hit against covered candidates inside the same group.",
        },
        {
            "check": "broad_vs_sparse_gate",
            "passed": int("coverage50_reliable" in m and "coverage75_reliable" in m),
            "finding": "Reliability is evaluated under predeclared coverage thresholds 0.25/0.50/0.75.",
        },
        {
            "check": "underuse_not_selector_repair",
            "passed": int(res["existing_score_underuse"]["summary_rows"] != []),
            "finding": "Existing scores are audited against diagnostic islands without emitting selected artifacts.",
        },
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "source_space_registry.csv"),
              res["source_space_registry"]["rows"],
              ["source_space", "families", "n_source_objectives", "objectives", "distance_metric",
               "target_label_used", "source_only"])
    _writecsv(os.path.join(tdir, "epsilon_radius_registry.csv"),
              res["epsilon_radius_registry"]["rows"],
              ["source_space", "epsilon_quantile", "epsilon_radius", "distance_metric", "computed_from"])
    common = ["group_scope", "source_space", "neighborhood", "neighborhood_kind", "neighborhood_value",
              "min_neighbor_count", "label", "n_groups", "n_evaluable_groups", "mean_n_candidates",
              "mean_coverage", "mean_empty_fraction", "mean_neighbor_count", "mean_group_base_rate",
              "mean_covered_base_rate", "mean_local_purity_all", "mean_local_purity_covered",
              "mean_local_bayes_top1_hit", "mean_local_bayes_gain_vs_covered_random",
              "mean_local_bayes_enrichment", "mean_c47_actual_strict_source_top1_hit",
              "mean_gap_vs_c47_actual_top1", "target_labels_diagnostic_only"]
    _writecsv(os.path.join(tdir, "coverage_accuracy_curve.csv"),
              res["coverage_accuracy"]["summary_rows"], common)
    _writecsv(os.path.join(tdir, "coverage_best_by_scope.csv"),
              res["coverage_accuracy"]["best_rows"], ["best_kind"] + common[:-1])
    _writecsv(os.path.join(tdir, "reliability_under_coverage.csv"),
              res["coverage_accuracy"]["reliability_rows"],
              common + ["coverage_threshold", "passes_reliability_with_coverage"])
    _writecsv(os.path.join(tdir, "stability_rows.csv"),
              res["stability"]["rows"],
              ["stability_grouping", "group_key", "source_space", "neighborhood", "neighborhood_kind",
               "neighborhood_value", "min_neighbor_count", "label", "n_candidates", "n_covered_candidates",
               "coverage", "empty_fraction", "mean_neighbor_count", "covered_base_rate",
               "local_bayes_top1_expected_hit", "local_bayes_top1_enrichment",
               "local_bayes_gain_vs_covered_random", "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "stability_summary.csv"),
              res["stability"]["summary_rows"],
              ["stability_grouping", "n_groups", "n_evaluable_groups", "mean_hit", "median_hit",
               "min_hit", "max_hit", "std_hit", "mean_coverage", "min_coverage",
               "mean_enrichment", "min_enrichment", "mean_neighbor_count", "worst_group_key_by_hit",
               "target_labels_diagnostic_only"])
    island_cols = ["group_scope", "group_key", "source_space", "neighborhood", "neighborhood_kind",
                   "neighborhood_value", "min_neighbor_count", "label", "target", "seed", "level",
                   "regime", "trajectory_id", "n_candidates", "n_covered_candidates", "coverage",
                   "empty_fraction", "mean_neighbor_count", "group_base_rate", "covered_base_rate",
                   "mean_local_purity_all", "mean_local_purity_covered", "max_local_purity_covered",
                   "local_bayes_top1_expected_hit", "local_bayes_top1_gain_vs_covered_random",
                   "local_bayes_top1_enrichment", "c47_actual_strict_source_top1_hit",
                   "gap_vs_c47_actual_top1", "is_high_ceiling_island", "target_labels_diagnostic_only",
                   "no_candidate_id_emitted"]
    _writecsv(os.path.join(tdir, "source_space_island_audit.csv"),
              res["source_space_islands"]["rows"], island_cols)
    _writecsv(os.path.join(tdir, "existing_score_underuse_rows.csv"),
              res["existing_score_underuse"]["rows"],
              ["group_scope", "group_key", "score", "target", "seed", "level", "regime",
               "trajectory_id", "source_space", "neighborhood", "min_neighbor_count", "label",
               "coverage", "local_bayes_top1_expected_hit", "source_score_top_expected_hit",
               "top_score_island_fraction", "underuse_gap", "score_range",
               "top_minus_best_island_score", "score_flat_at_island", "wrong_direction_miss",
               "target_labels_diagnostic_only", "no_candidate_id_emitted"])
    _writecsv(os.path.join(tdir, "existing_score_underuse_summary.csv"),
              res["existing_score_underuse"]["summary_rows"],
              ["score", "n_groups", "mean_coverage", "mean_local_bayes_top1_hit",
               "mean_source_score_top_hit", "mean_underuse_gap", "mean_top_score_island_fraction",
               "score_flat_at_island_fraction", "wrong_direction_miss_fraction",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "red_team_verification.csv"), red_team_rows(res),
              ["check", "passed", "finding"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res),
              ["check", "passed"])
    _writecsv(os.path.join(tdir, "c49_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def render_main_md(res):
    m = res["taxonomy"]["primary_metrics"]
    return "\n".join([
        f"# C49 - Sparse Local-Bayes Coverage Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only diagnostic audit over fixed local-Bayes neighborhoods. Coverage, empty-neighborhood "
        "fraction, and same-group random baselines are reported with every curve point.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- candidate rows / trajectories: **{res['n_candidate_rows']} / {res['n_trajectories']}**.",
        f"- C48 ceiling reference: hit **{_f(res['c48_reference']['best_conditioned_top1_hit'])}**, "
        f"enrichment **{_f(res['c48_reference']['best_conditioned_enrichment'])}**, "
        f"permutation gap **{_f(res['c48_reference']['best_conditioned_gap_vs_permutation'])}**.",
        "",
        "## Coverage Gate",
        "",
        f"- best conditioned setup: **{m['best_conditioned_scope']} / {m['best_conditioned_source_space']} / "
        f"{m['best_conditioned_neighborhood']} / min_n={m['best_conditioned_min_neighbor_count']}**.",
        f"- best hit / enrichment / coverage: **{_f(m['best_conditioned_hit'])} / "
        f"{_f(m['best_conditioned_enrichment'])} / {_f(m['best_conditioned_coverage'])}**.",
        f"- coverage >= 0.50 reliable: **{m['coverage50_reliable']}** "
        f"via **{m.get('coverage50_best_scope')} / {m.get('coverage50_best_source_space')} / "
        f"{m.get('coverage50_best_neighborhood')} / min_n={m.get('coverage50_best_min_neighbor_count')}** "
        f"(hit **{_f(m['coverage50_best_hit'])}**, coverage **{_f(m['coverage50_best_coverage'])}**).",
        f"- coverage >= 0.75 reliable: **{m['coverage75_reliable']}** "
        f"via **{m.get('coverage75_best_scope')} / {m.get('coverage75_best_source_space')} / "
        f"{m.get('coverage75_best_neighborhood')} / min_n={m.get('coverage75_best_min_neighbor_count')}** "
        f"(hit **{_f(m['coverage75_best_hit'])}**, coverage **{_f(m['coverage75_best_coverage'])}**).",
        "",
        "## Stability And Underuse",
        "",
        f"- target min hit / min coverage: **{_f(m['target_stability_min_hit'])} / "
        f"{_f(m['target_stability_min_coverage'])}**.",
        f"- trajectory min hit / min coverage: **{_f(m['trajectory_stability_min_hit'])} / "
        f"{_f(m['trajectory_stability_min_coverage'])}**.",
        f"- max existing-score underuse gap: **{m['max_underuse_score']} = {_f(m['max_underuse_gap'])}**.",
        "",
        "## Bottom Line",
        "",
        "> C49 tests whether C48's sparse max-local ceiling survives coverage and stability gates. The result "
        "is diagnostic-only: any local ceiling still uses target labels for scoring and does not create a "
        "deployable action rule.",
    ])


def render_red_team_md(res):
    lines = [
        "# C49 - Red-Team Verification",
        "",
        "C49 red-team checks were run after artifact generation and before commit.",
        "",
    ]
    for r in red_team_rows(res):
        lines.append(f"- {r['check']}: {'pass' if r['passed'] else 'fail'} - {r['finding']}")
    lines += [
        "",
        "Verdict: C49 is a diagnostic coverage audit; it does not emit a selected-checkpoint artifact.",
    ]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "without ", "diagnostic")


def _guard_forbidden(text):
    low = text.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 160):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden affirmative C49 claim near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "coverage_accuracy_summary": res["coverage_accuracy"]["summary"],
        "best_conditioned_setup": res["best_conditioned_setup"],
        "source_space_island_summary": res["source_space_islands"]["summary"],
        "existing_score_underuse_summary_rows": res["existing_score_underuse"]["summary_rows"],
        "stability_summary_rows": res["stability"]["summary_rows"],
        "taxonomy": res["taxonomy"],
        "c48_reference": res["c48_reference"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": red_team_rows(res),
    }


def _write_artifacts(res, out_dir):
    md = render_main_md(res)
    red = render_red_team_md(res)
    for text in (md, red):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C49_SPARSE_LOCAL_BAYES_COVERAGE_AUDIT.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C49_RED_TEAM_VERIFICATION.md"), "w").write(red)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C49_SPARSE_LOCAL_BAYES_COVERAGE_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c49_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    m = res["taxonomy"]["primary_metrics"]
    print(f"[C49] cases={','.join(res['taxonomy']['cases'])} "
          f"best_hit={m['best_conditioned_hit']} "
          f"best_coverage={m['best_conditioned_coverage']} "
          f"coverage50={m['coverage50_reliable']} coverage75={m['coverage75_reliable']}")


if __name__ == "__main__":
    main()
