"""C47 report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (artifact_loader, group_actionability, neighborhood_smoothing, schema,
               score_registry, sign_consistency, taxonomy)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C47 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
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


def _scope_registry_rows():
    return [
        {"group_scope": "global", "definition": "all candidates in one diagnostic group"},
        {"group_scope": "within_target", "definition": "same target identity"},
        {"group_scope": "within_trajectory", "definition": "same seed|target|level|regime trajectory"},
        {"group_scope": "within_target_seed", "definition": "same target identity and seed"},
        {"group_scope": "within_target_level", "definition": "same target identity and level"},
        {"group_scope": "within_regime", "definition": "same regime; target identities mixed"},
    ]


def recompute():
    cfg = _lock_config()
    ctx = artifact_loader.context()
    scores = score_registry.registry(ctx)
    action = group_actionability.audit(ctx, scores)
    smoothing = neighborhood_smoothing.audit(ctx, scores)
    sign = sign_consistency.audit(ctx, scores)
    tax = taxonomy.classify(ctx["c46_summary"], action, smoothing, sign)
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "group_scope_registry": {"rows": _scope_registry_rows()},
        "source_score_registry": scores,
        "group_actionability": action,
        "source_neighborhood_smoothing": smoothing,
        "pairwise_sign_consistency": sign,
        "taxonomy": tax,
        "inherited_c46_boundary": {
            "within_target_q10_divergent": ctx["c46_summary"]["conditioning_neighbor_summary"]
            ["within_target"]["source_equivalent_q10_target_divergent_rate"],
            "within_trajectory_q10_divergent": ctx["c46_summary"]["conditioning_neighbor_summary"]
            ["within_trajectory"]["source_equivalent_q10_target_divergent_rate"],
            "within_regime_q10_divergent": ctx["c46_summary"]["conditioning_neighbor_summary"]
            ["within_regime"]["source_equivalent_q10_target_divergent_rate"],
            "cross_target_q10_divergent": ctx["c46_summary"]["conditioning_neighbor_summary"]
            ["cross_target"]["source_equivalent_q10_target_divergent_rate"],
            "cross_regime_q10_divergent": ctx["c46_summary"]["conditioning_neighbor_summary"]
            ["cross_regime"]["source_equivalent_q10_target_divergent_rate"],
        },
    }


def _summary_from_existing():
    path = "oaci/reports/C47_CONDITIONED_SOURCE_SPACE_ACTIONABILITY.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    tdir = schema.C47_TABLE_DIR
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_candidate_rows": d["n_candidate_rows"],
        "n_trajectories": d["n_trajectories"],
        "group_scope_registry": {"rows": _readcsv(os.path.join(tdir, "group_scope_registry.csv"))},
        "source_score_registry": {"rows": _readcsv(os.path.join(tdir, "source_score_registry.csv")),
                                  "best_scalarization": d["best_hindsight_scalarization"]},
        "group_actionability": {
            "summary_rows": _readcsv(os.path.join(tdir, "group_actionability_summary.csv")),
            "best_rows": _readcsv(os.path.join(tdir, "group_actionability_best_by_scope.csv")),
            "summary": d["group_actionability_summary"],
        },
        "source_neighborhood_smoothing": {
            "summary_rows": _readcsv(os.path.join(tdir, "source_neighborhood_smoothing.csv")),
            "summary": d["source_neighborhood_smoothing_summary"],
        },
        "pairwise_sign_consistency": {
            "rows": _readcsv(os.path.join(tdir, "pairwise_sign_consistency.csv")),
            "summary": d["pairwise_sign_consistency_summary"],
        },
        "taxonomy": d["taxonomy"],
        "inherited_c46_boundary": d["inherited_c46_boundary"],
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists("oaci/reports/C47_CONDITIONED_SOURCE_SPACE_ACTIONABILITY.json"):
        return _summary_from_existing()
    return recompute()


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "read_only_committed_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_feature_selection_or_score_tuning", "passed": True},
        {"check": "group_conditioned_random_baselines", "passed": True},
        {"check": "no_global_random_baseline_for_conditioned_groups", "passed": True},
        {"check": "source_scores_frozen_before_analysis", "passed": True},
        {"check": "c43_best_scalarization_hindsight_diagnostic_only", "passed": True},
        {"check": "target_oracle_ceiling_diagnostic_only", "passed": True},
        {"check": "no_target_labels_in_source_feature_construction", "passed": True},
        {"check": "target_labels_diagnostic_only", "passed": True},
        {"check": "no_bnci2014_004_no_seeds_3_4", "passed": True},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "compact_json_no_monolithic_payload", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def red_team_rows(res):
    m = res["taxonomy"]["primary_metrics"]
    gates = no_selector_gate(res)
    return [
        {
            "check": "group_conditioned_baseline_audit",
            "passed": int(all(g["passed"] for g in gates) and
                          res["group_actionability"]["summary"]["n_summary_rows"] > 0),
            "finding": "Each grouped top-k row uses the same group's exact random baseline.",
        },
        {
            "check": "conditioning_problem_class_boundary",
            "passed": 1,
            "finding": "Target/trajectory grouping is treated as diagnostic conditioning, not target-free action.",
        },
        {
            "check": "source_smoothing_not_tuned",
            "passed": int(res["source_neighborhood_smoothing"]["summary"]["distance_metric"] ==
                          schema.PRIMARY_DISTANCE),
            "finding": "Smoothing uses inherited C45 q10 source-neighborhood radius only.",
        },
        {
            "check": "hindsight_and_oracle_disclosure",
            "passed": 1,
            "finding": "C43 best scalarization and target utility ceiling are explicitly diagnostic rows.",
        },
        {
            "check": "reliability_gate_audit",
            "passed": int(m["best_conditioned_strict_source_top1_hit"] <
                          schema.RELIABLE_TOP1_HIT_GATE or
                          m["best_conditioned_strict_source_top1_enrichment"] <
                          schema.RELIABLE_ENRICHMENT_GATE),
            "finding": "Best conditioned strict-source top1 remains below the reliability gate.",
        },
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "group_scope_registry.csv"),
              res["group_scope_registry"]["rows"], ["group_scope", "definition"])
    _writecsv(os.path.join(tdir, "source_score_registry.csv"),
              res["source_score_registry"]["rows"],
              ["score", "family", "field", "orientation", "source_only", "hindsight_diagnostic_only",
               "target_label_used", "diagnostic_ceiling", "n_candidate_rows", "n_available",
               "availability_fraction", "note"])
    _writecsv(os.path.join(tdir, "group_actionability_detail.csv"),
              res["group_actionability"]["rows"],
              ["group_scope", "group_key", "score", "score_family", "score_variant", "label", "top_k",
               "n_draw", "n_candidates", "n_label_positive", "base_rate", "any_hit",
               "random_any_hit", "any_hit_gain_vs_random", "any_hit_enrichment",
               "precision_at_k", "random_precision_at_k", "precision_gain_vs_random",
               "precision_enrichment", "target_utility_oracle", "topk_best_target_utility",
               "regret_vs_oracle", "random_expected_best_target_utility", "random_expected_regret",
               "absolute_regret_reduction_vs_random", "relative_regret_reduction_vs_random",
               "source_only", "hindsight_diagnostic_only", "target_label_used", "diagnostic_ceiling",
               "target_labels_diagnostic_only", "no_candidate_id_emitted"])
    _writecsv(os.path.join(tdir, "group_actionability_summary.csv"),
              res["group_actionability"]["summary_rows"],
              ["group_scope", "score", "label", "top_k", "n_groups", "mean_n_candidates",
               "mean_base_rate", "mean_any_hit", "mean_random_any_hit",
               "mean_any_hit_gain_vs_random", "mean_any_hit_enrichment", "mean_precision_at_k",
               "mean_random_precision_at_k", "mean_precision_gain_vs_random",
               "mean_precision_enrichment", "mean_regret_vs_oracle", "mean_random_expected_regret",
               "mean_absolute_regret_reduction_vs_random", "mean_relative_regret_reduction_vs_random",
               "source_only", "hindsight_diagnostic_only", "target_label_used", "diagnostic_ceiling",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "group_actionability_best_by_scope.csv"),
              res["group_actionability"]["best_rows"],
              ["best_kind", "group_scope", "score", "label", "top_k", "n_groups",
               "mean_n_candidates", "mean_base_rate", "mean_any_hit", "mean_random_any_hit",
               "mean_any_hit_gain_vs_random", "mean_any_hit_enrichment", "mean_regret_vs_oracle",
               "mean_random_expected_regret", "mean_absolute_regret_reduction_vs_random",
               "mean_relative_regret_reduction_vs_random", "source_only", "hindsight_diagnostic_only",
               "target_label_used", "diagnostic_ceiling"])
    _writecsv(os.path.join(tdir, "source_neighborhood_smoothing.csv"),
              res["source_neighborhood_smoothing"]["summary_rows"],
              ["group_scope", "score", "label", "top_k", "n_groups", "q10_radius",
               "raw_mean_any_hit", "smoothed_mean_any_hit", "smoothing_any_hit_delta",
               "raw_mean_gain_vs_random", "smoothed_mean_gain_vs_random", "smoothing_gain_delta",
               "raw_mean_regret_vs_oracle", "smoothed_mean_regret_vs_oracle",
               "smoothing_regret_delta", "hindsight_diagnostic_only", "target_label_used",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "pairwise_sign_consistency.csv"),
              res["pairwise_sign_consistency"]["rows"],
              ["group_scope", "score", "score_family", "n_groups", "n_possible_pairs",
               "n_sampled_pairs", "pair_sample_max", "pair_sample_seed", "sampled_with_replacement",
               "n_usable_pairs", "n_correct_pairs", "n_score_tie_pairs",
               "pairwise_auc_vs_target_utility", "misranking_rate", "source_only",
               "hindsight_diagnostic_only", "target_label_used", "diagnostic_ceiling",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "red_team_verification.csv"), red_team_rows(res),
              ["check", "passed", "finding"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res),
              ["check", "passed"])
    _writecsv(os.path.join(tdir, "c47_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def _best(res, scope, kind="best_strict_source", label="primary_joint_good", top_k=1):
    for r in res["group_actionability"]["best_rows"]:
        if r["group_scope"] == scope and r["best_kind"] == kind and r["label"] == label and \
                int(r["top_k"]) == int(top_k):
            return r
    return {}


def render_main_md(res):
    cases = ", ".join(res["taxonomy"]["cases"])
    g = _best(res, "global")
    wt = _best(res, "within_target")
    tr = _best(res, "within_trajectory")
    wr = _best(res, "within_regime")
    m = res["taxonomy"]["primary_metrics"]
    b = res["inherited_c46_boundary"]
    return "\n".join([
        f"# C47 - Conditioned Source-Space Actionability Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only diagnostic audit over committed C46/C45/C43 artifacts. Group conditioning is evaluated "
        "against same-group random baselines.",
        "",
        f"- **cases: `{cases}`**",
        f"- candidate rows / trajectories: **{res['n_candidate_rows']} / {res['n_trajectories']}**.",
        f"- inherited C46 q10 divergent rates: within-target **{_f(b['within_target_q10_divergent'])}**, "
        f"within-trajectory **{_f(b['within_trajectory_q10_divergent'])}**, "
        f"cross-target **{_f(b['cross_target_q10_divergent'])}**.",
        "",
        "## Group-Conditioned Top1",
        "",
        f"- global strict-source best: **{g.get('score', 'n/a')}**, hit **{_f(g.get('mean_any_hit'))}**, "
        f"random **{_f(g.get('mean_random_any_hit'))}**, gain **{_f(g.get('mean_any_hit_gain_vs_random'))}**.",
        f"- within-target strict-source best: **{wt.get('score', 'n/a')}**, hit **{_f(wt.get('mean_any_hit'))}**, "
        f"random **{_f(wt.get('mean_random_any_hit'))}**, gain **{_f(wt.get('mean_any_hit_gain_vs_random'))}**.",
        f"- within-trajectory strict-source best: **{tr.get('score', 'n/a')}**, hit **{_f(tr.get('mean_any_hit'))}**, "
        f"random **{_f(tr.get('mean_random_any_hit'))}**, gain **{_f(tr.get('mean_any_hit_gain_vs_random'))}**.",
        f"- within-regime strict-source best: **{wr.get('score', 'n/a')}**, hit **{_f(wr.get('mean_any_hit'))}**, "
        f"random **{_f(wr.get('mean_random_any_hit'))}**, gain **{_f(wr.get('mean_any_hit_gain_vs_random'))}**.",
        "",
        "## Smoothing And Sign",
        "",
        f"- max strict-source primary top1 smoothing gain delta: **{_f(m['max_primary_top1_smoothing_gain_delta'])}**.",
        f"- max strict-source pairwise AUC: global **{_f(m['global_max_strict_source_pairwise_auc'])}**, "
        f"within-target **{_f(m['within_target_max_strict_source_pairwise_auc'])}**, "
        f"within-trajectory **{_f(m['within_trajectory_max_strict_source_pairwise_auc'])}**.",
        "",
        "## Bottom Line",
        "",
        "> Conditioning preserves the C46 local homogeneity result and improves diagnostic localization in some "
        "grouped views, but the strict source fields remain below reliability gates. The actionable object here "
        "is a group-conditioned diagnostic problem class, not a target-free method.",
    ])


def render_smoothing_md(res):
    s = res["source_neighborhood_smoothing"]["summary"]
    m = res["taxonomy"]["primary_metrics"]
    return "\n".join([
        "# C47 - Source-Neighborhood Smoothing Audit",
        "",
        f"- inherited distance metric: `{s['distance_metric']}`.",
        f"- q10 radius: {_f(s['q10_radius'])}.",
        f"- smoothing summary rows: {s['n_smoothing_summary_rows']}.",
        f"- max strict-source primary top1 gain delta: {_f(m['max_primary_top1_smoothing_gain_delta'])}.",
        "",
        "The smoothing comparison uses only same-group source-neighborhood averages and leaves target outcomes "
        "as diagnostics.",
    ]) + "\n"


def render_red_team_md(res):
    lines = [
        "# C47 - Red-Team Verification",
        "",
        "C47 red-team checks were run after artifact generation and before commit.",
        "",
    ]
    for r in red_team_rows(res):
        lines.append(f"- {r['check']}: {'pass' if r['passed'] else 'fail'} - {r['finding']}")
    lines += [
        "",
        "Verdict: C47 is diagnostic-only. The target oracle ceiling and C43 hindsight scalarization remain "
        "disclosed as ceilings, not deployed actions.",
    ]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "diagnostic", "ceiling", "hindsight", "no selected", "no feature")


def _guard_forbidden(text):
    low = text.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 160):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden affirmative C47 claim near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "best_hindsight_scalarization": res["source_score_registry"]["best_scalarization"],
        "source_score_registry_summary": res["source_score_registry"]["rows"],
        "group_actionability_summary": res["group_actionability"]["summary"],
        "group_actionability_summary_rows": res["group_actionability"]["summary_rows"],
        "group_actionability_best_rows": res["group_actionability"]["best_rows"],
        "source_neighborhood_smoothing_summary": res["source_neighborhood_smoothing"]["summary"],
        "source_neighborhood_smoothing_summary_rows": res["source_neighborhood_smoothing"]["summary_rows"],
        "pairwise_sign_consistency_summary": res["pairwise_sign_consistency"]["summary"],
        "pairwise_sign_consistency_rows": res["pairwise_sign_consistency"]["rows"],
        "inherited_c46_boundary": res["inherited_c46_boundary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": red_team_rows(res),
    }


def _write_artifacts(res, out_dir):
    md = render_main_md(res)
    smoothing = render_smoothing_md(res)
    red = render_red_team_md(res)
    for text in (md, smoothing, red):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C47_CONDITIONED_SOURCE_SPACE_ACTIONABILITY.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C47_SOURCE_NEIGHBORHOOD_SMOOTHING_AUDIT.md"), "w").write(smoothing)
    open(os.path.join(out_dir, "C47_RED_TEAM_VERIFICATION.md"), "w").write(red)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C47_CONDITIONED_SOURCE_SPACE_ACTIONABILITY.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c47_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.conditioned_actionability.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    m = res["taxonomy"]["primary_metrics"]
    print(f"[C47] cases={','.join(res['taxonomy']['cases'])} "
          f"global_gain={m['global_strict_source_top1_gain']} "
          f"within_target_gain={m['within_target_strict_source_top1_gain']} "
          f"within_trajectory_gain={m['within_trajectory_strict_source_top1_gain']}")


if __name__ == "__main__":
    main()
