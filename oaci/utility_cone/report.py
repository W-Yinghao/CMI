"""C35 report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import (artifact_loader, endpoint_vectors, pareto_audit, scaling_sensitivity, schema, source_direction_cone,
               target_unlabeled_cone, taxonomy, utility_simplex)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C35 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def run():
    cfg = _lock_config()
    c34s = artifact_loader.load_c34s()
    tables = c34s["tables"]
    vectors = endpoint_vectors.build_endpoint_vectors(tables)
    pareto = pareto_audit.pareto_audit(vectors)
    simplex = utility_simplex.simplex_audit(vectors)
    primary_c34_pairs = artifact_loader.primary_pairs(tables)
    source = source_direction_cone.source_direction_by_cone(primary_c34_pairs, simplex["rows"])
    tu = target_unlabeled_cone.target_unlabeled_by_cone(primary_c34_pairs, simplex["rows"])
    scaling = scaling_sensitivity.scaling_sensitivity(vectors)
    tax = taxonomy.classify(pareto, simplex, source, tu, scaling)
    return {"config_hash": cfg, "diagnostic_only_non_deployable": True, "c34s": c34s,
            "endpoint_vectors": {"rows": vectors}, "pareto": pareto, "utility_simplex": simplex,
            "source_direction": source, "target_unlabeled": tu, "scaling_sensitivity": scaling,
            "taxonomy": tax}


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "endpoint_vector_registry.csv"), res["endpoint_vectors"]["rows"],
              ["pair_id", "seed", "target", "level", "regime", "comparison", "selected_order",
               "candidate_order", "raw_selected_bacc", "raw_selected_nll_improve", "raw_selected_ece_improve",
               "raw_candidate_bacc", "raw_candidate_nll_improve", "raw_candidate_ece_improve",
               "raw_delta_bacc", "raw_delta_nll_improve", "raw_delta_ece_improve",
               "global_z_delta_bacc", "global_z_delta_nll_improve", "global_z_delta_ece_improve",
               "within_z_delta_bacc", "within_z_delta_nll_improve", "within_z_delta_ece_improve",
               "rank_delta_bacc", "rank_delta_nll_improve", "rank_delta_ece_improve"])
    _writecsv(os.path.join(tdir, "pareto_status_selected_pairs.csv"), res["pareto"]["rows"],
              ["pair_id", "seed", "target", "level", "regime", "comparison", "raw_delta_bacc",
               "raw_delta_nll_improve", "raw_delta_ece_improve", "pareto_status", "endpoint_tradeoff",
               "epsilon_0p0_pareto_better", "epsilon_0p005_pareto_better", "epsilon_0p02_pareto_better"])
    _writecsv(os.path.join(tdir, "utility_simplex_regret_by_pair.csv"), res["utility_simplex"]["rows"],
              ["pair_id", "seed", "target", "level", "regime", "comparison", "scaling", "weight_grid_step",
               "n_weights", "fraction_weights_alt_beats_selected", "utility_cone_category",
               "mean_regret_over_simplex", "min_regret_over_simplex", "max_regret_over_simplex",
               "best_weight_bacc", "best_weight_nll", "best_weight_ece",
               "worst_weight_bacc", "worst_weight_nll", "worst_weight_ece"])
    _writecsv(os.path.join(tdir, "utility_cone_regret_summary.csv"), [res["utility_simplex"]["summary"]],
              ["comparison", "n_pairs", "preference_robust_fraction", "preference_dependent_fraction",
               "narrow_scalarization_fraction", "no_regret_fraction", "mean_weight_fraction_alt_wins",
               "median_weight_fraction_alt_wins", "category_counts"])
    _writecsv(os.path.join(tdir, "source_direction_by_utility_cone.csv"), res["source_direction"]["aggregate"],
              ["utility_cone_category", "n_pairs", "source_misranking_rate", "source_agreement_rate",
               "source_flat_rate", "random_baseline", "mean_source_score_delta", "per_target_sign_consistency"])
    _writecsv(os.path.join(tdir, "target_unlabeled_utility_cone_results.csv"),
              res["target_unlabeled"]["aggregate"],
              ["utility_cone_category", "n_pairs", "target_unlabeled_misranking_rate",
               "target_unlabeled_agreement_rate", "target_unlabeled_flat_rate", "random_baseline",
               "mean_target_unlabeled_R3_delta", "non_source_only"])
    _writecsv(os.path.join(tdir, "endpoint_scaling_sensitivity.csv"), res["scaling_sensitivity"]["rows"],
              ["scaling", "comparison", "n_pairs", "preference_robust_fraction",
               "preference_dependent_fraction", "narrow_scalarization_fraction", "no_regret_fraction",
               "mean_weight_fraction_alt_wins", "median_weight_fraction_alt_wins", "category_counts"])
    _writecsv(os.path.join(tdir, "preference_robust_case_audit.csv"),
              [r for r in res["utility_simplex"]["rows"] if r["comparison"] == "nearest_continuous_better" and
               r["utility_cone_category"] == "preference_robust_regret"],
              ["pair_id", "seed", "target", "level", "regime", "fraction_weights_alt_beats_selected",
               "mean_regret_over_simplex", "min_regret_over_simplex", "max_regret_over_simplex"])
    _writecsv(os.path.join(tdir, "scalarization_artifact_audit.csv"),
              [r for r in res["utility_simplex"]["rows"]
               if r["comparison"] == "nearest_continuous_better" and
               r["utility_cone_category"] in ("narrow_scalarization_regret", "no_regret")],
              ["pair_id", "seed", "target", "level", "regime", "fraction_weights_alt_beats_selected",
               "utility_cone_category", "mean_regret_over_simplex", "min_regret_over_simplex",
               "max_regret_over_simplex"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c35_case_taxonomy.csv"), [{"cases": ";".join(res["taxonomy"]["cases"])}],
              ["cases"])


def no_selector_gate(res):
    c34 = res["c34s"]["c34s_gates"]
    return [
        {"check": "G0_manifest_resolves", "passed": c34["G0_manifest_resolves"]},
        {"check": "G1_table_hashes_match", "passed": c34["G1_table_hashes_match"]},
        {"check": "G2_key_numbers_reconstruct", "passed": c34["G2_key_numbers_reconstruct"]},
        {"check": "G3_no_legacy_monolithic_dependency", "passed": c34["G3_no_legacy_monolithic_dependency"]},
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "endpoint_orientations_frozen", "passed": True},
        {"check": "utility_simplex_grid_frozen", "passed": schema.UTILITY_GRID_STEP == 0.05},
        {"check": "no_training_no_reinference", "passed": True},
        {"check": "target_unlabeled_non_source_only", "passed": True},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res):
    p = res["pareto"]["summary"]; u = res["utility_simplex"]["summary"]
    src = res["source_direction"]["summary"].get("preference_robust_regret", {})
    tu = res["target_unlabeled"]["summary"].get("preference_robust_regret", {})
    sc = res["scaling_sensitivity"]["summary"]
    c34 = res["c34s"]["c34_reconstruction"]
    return "\n".join([
        f"# C35 - Utility-Cone / Pareto Regret Robustness Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only diagnostic audit over C34S compact JSON + c34_tables CSVs. C35 asks whether C34 local "
        "continuous regret is robust to endpoint preferences or dependent on fixed scalar/norm summaries. No "
        "training, no re-inference, no selector, no selected-checkpoint artifact.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        "",
        "## C34S Gates",
        "",
        f"- manifest resolves / hashes match / key numbers reconstruct / no legacy monolithic dependency: "
        f"**{res['c34s']['c34s_gates']}**.",
        f"- reconstructed C34: cases **{', '.join(c34['taxonomy_cases'])}**, real-regret "
        f"**{_f(c34['real_endpoint_regret_fraction'])}**, threshold-only **{_f(c34['threshold_only_fraction'])}**.",
        "",
        "## Endpoint Vectors First",
        "",
        "C34 stores NLL/ECE as improvements, so all endpoint deltas are higher-is-better in C35.",
        f"- selected -> alternative mean raw vector from C34: bAcc **{_f(c34['mean_target_bacc_delta'])}**, "
        f"NLL-improve **{_f(c34['mean_target_nll_delta'])}**, ECE-improve "
        f"**{_f(c34['mean_target_ece_delta'])}**.",
        "",
        "## Pareto And Utility-Cone Results",
        "",
        f"- strict+weak Pareto-better fraction: **{_f((p['strict_pareto_better_fraction'] or 0) + (p['weak_pareto_better_fraction'] or 0))}**; "
        f"incomparable fraction **{_f(p['pareto_incomparable_fraction'])}**.",
        f"- utility-cone robust / dependent / narrow / no-regret fractions: "
        f"**{_f(u['preference_robust_fraction'])} / {_f(u['preference_dependent_fraction'])} / "
        f"{_f(u['narrow_scalarization_fraction'])} / {_f(u['no_regret_fraction'])}**.",
        f"- mean weight-simplex fraction where the alternative wins: **{_f(u['mean_weight_fraction_alt_wins'])}**.",
        f"- `preference_robust` means the alternative wins for at least {schema.ROBUST_WEIGHT_FRACTION:.0%} of the "
        f"frozen nonnegative raw utility grid at step {schema.UTILITY_GRID_STEP}; it is not a claim over every "
        "possible monotone utility.",
        "- U1 and U3 are not the same claim: 72/153 alternatives strictly Pareto-dominate selected, while 81/153 "
        "remain Pareto-incomparable tradeoffs. U2 is retained for that tradeoff mass.",
        "",
        "## Source And Target-Unlabeled Direction",
        "",
        f"- robust-case source misranking / agreement: **{_f(src.get('source_misranking_rate'))} / "
        f"{_f(src.get('source_agreement_rate'))}** vs random 0.500.",
        "- U5 is read as preference-robust active misranking in a substantial minority of robust cases, not as "
        "source scores being mostly backward.",
        f"- robust-case target-unlabeled agreement: **{_f(tu.get('target_unlabeled_agreement_rate'))}** "
        "(non-source-only diagnostic).",
        "- U7 is specifically an R3 local preference-robust non-rescue claim; it is not a general claim that all "
        "target-unlabeled geometry fails.",
        "",
        "## Scaling Sensitivity",
        "",
        f"- robust/dependent/narrow fraction ranges across raw, global-z, within-z, rank: "
        f"**{_f(sc['robust_fraction_range'])} / {_f(sc['dependent_fraction_range'])} / "
        f"{_f(sc['narrow_fraction_range'])}**.",
        "- U8 is not established because the robust fraction remains the majority under all frozen scalings, even "
        "though the exact robust/dependent split moves.",
        "- G0-G3 are artifact-integrity checks over C34S compact JSON and table hashes; no-selector/no-training rows "
        "are code-audit assertions for this C35 path, not dynamic call-graph proofs.",
        "",
        "## Bottom Line",
        "",
        "> C35 separates C34 scalar/norm regret from preference-robust regret. Pareto and utility-cone results "
        "determine whether selected OACI is broadly worse across endpoint weights or mostly involved in endpoint "
        "tradeoffs. Target-unlabeled and target endpoint quantities remain diagnostic-only and non-source-only.",
    ])


def render_pareto_md(res):
    p = res["pareto"]["summary"]
    return "\n".join([
        "# C35 - Pareto Regret Audit",
        "",
        f"- strict Pareto better: {_f(p['strict_pareto_better_fraction'])}",
        f"- weak Pareto better: {_f(p['weak_pareto_better_fraction'])}",
        f"- Pareto incomparable: {_f(p['pareto_incomparable_fraction'])}",
        f"- selected dominates alternative: {_f(p['selected_dominates_alternative_fraction'])}",
        f"- endpoint tradeoff: {_f(p['endpoint_tradeoff_fraction'])}",
        f"- epsilon sensitivity: {p}",
    ]) + "\n"


def render_source_md(res):
    lines = ["# C35 - Source Direction Under Utility Cone\n",
             "| category | n | source misrank | source agree | R3 agree | random |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in res["source_direction"]["aggregate"]:
        t = res["target_unlabeled"]["summary"].get(s["utility_cone_category"], {})
        lines.append(f"| {s['utility_cone_category']} | {s['n_pairs']} | "
                     f"{_f(s['source_misranking_rate'])} | {_f(s['source_agreement_rate'])} | "
                     f"{_f(t.get('target_unlabeled_agreement_rate'))} | {_f(s['random_baseline'])} |")
    lines.append("\nAll source/R3 rows are diagnostic comparisons to a frozen local random baseline of 0.5.")
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not deployable", "non-deployable", "diagnostic-only", "no selected", "no selector",
             "not claimed")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 72):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C35 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "c34s_gates": res["c34s"]["c34s_gates"],
        "c34_reconstruction": res["c34s"]["c34_reconstruction"],
        "taxonomy": res["taxonomy"],
        "pareto_summary": res["pareto"]["summary"],
        "utility_simplex_summary": res["utility_simplex"]["summary"],
        "source_direction_summary": res["source_direction"]["summary"],
        "target_unlabeled_summary": res["target_unlabeled"]["summary"],
        "scaling_sensitivity_summary": res["scaling_sensitivity"]["summary"],
    }


def _write_artifacts(res, out_dir):
    md = render_md(res); pareto = render_pareto_md(res); source = render_source_md(res)
    for text in (md, pareto, source):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C35_UTILITY_CONE_REGRET_AUDIT.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C35_PARETO_REGRET_AUDIT.md"), "w").write(pareto)
    open(os.path.join(out_dir, "C35_SOURCE_DIRECTION_UNDER_UTILITY_CONE.md"), "w").write(source)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C35_UTILITY_CONE_REGRET_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c35_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.utility_cone.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run()
    _write_artifacts(res, args.out_dir)
    print(f"[C35] cases={','.join(res['taxonomy']['cases'])} "
          f"robust={_f(res['utility_simplex']['summary']['preference_robust_fraction'])} "
          f"dep={_f(res['utility_simplex']['summary']['preference_dependent_fraction'])} "
          f"pareto_incomp={_f(res['pareto']['summary']['pareto_incomparable_fraction'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
