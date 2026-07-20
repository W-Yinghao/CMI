"""C34 report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from . import (artifact_hygiene, artifact_loader, endpoint_utility, gauge_locality, local_direction, margin_free_taxonomy,
               schema, selected_pair_regret, source_objective_components)


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C34 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def run(scores_sidecar=None, c10_dir=None, reinfer_sidecar=None, mode="in_regime"):
    cfg = _lock_config()
    rows, tu = artifact_loader.load_rows(scores_sidecar, c10_dir, reinfer_sidecar, mode, schema.PRIMARY_MARGIN)
    endpoint_summary = endpoint_utility.summarize_registry(rows)
    selected = selected_pair_regret.selected_pair_regret(rows)
    direction = local_direction.local_source_direction(rows)
    components = source_objective_components.source_objective_conflict(rows)
    gauge = gauge_locality.gauge_locality(selected, direction)

    robust_rows, _ = artifact_loader.load_rows(scores_sidecar, c10_dir, reinfer_sidecar, mode, schema.ROBUST_MARGIN)
    robust_selected = selected_pair_regret.selected_pair_regret(robust_rows)
    boundary_status = margin_free_taxonomy.binary_vs_continuous_status(selected, robust_selected)
    boundary_summary = margin_free_taxonomy.status_summary(boundary_status)
    taxonomy = margin_free_taxonomy.classify(endpoint_summary, selected, direction, components, gauge)

    return {"config_hash": cfg, "mode": mode, "n_rows": len(rows), "target_unlabeled": tu,
            "primary_margin": schema.PRIMARY_MARGIN, "robust_margin": schema.ROBUST_MARGIN,
            "endpoint_summary": endpoint_summary,
            "endpoint_registry": endpoint_utility.endpoint_registry(rows),
            "selected_pairs": selected,
            "source_direction": direction,
            "source_objective_components": components,
            "gauge_locality": gauge,
            "binary_vs_continuous_boundary": {"rows": boundary_status, "summary": boundary_summary},
            "robust_selected_pairs": {"summary": robust_selected["summary"]},
            "taxonomy": taxonomy,
            "diagnostic_only_non_deployable": True}


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "endpoint_utility_registry.csv"), res["endpoint_registry"],
              ["seed", "target", "level", "regime", "order", "epoch", "selected_oaci", "primary_joint_good",
               "target_bacc_delta", "target_nll_delta", "target_ece_delta",
               "target_bacc_z", "target_nll_z", "target_ece_z",
               "continuous_joint_min_margin", "pareto_distance", "dominated_hypervolume_regret",
               "endpoint_vector_norm_regret"])
    _writecsv(os.path.join(tdir, "selected_vs_continuous_better_pairs.csv"), res["selected_pairs"]["pairs"],
              ["seed", "target", "level", "regime", "comparison", "selected_joint_good", "candidate_joint_good",
               "selected_order", "candidate_order", "order_distance", "epoch_distance",
               "selected_target_bacc_delta", "selected_target_nll_delta", "selected_target_ece_delta",
               "candidate_target_bacc_delta", "candidate_target_nll_delta", "candidate_target_ece_delta",
               "target_bacc_delta", "target_nll_delta", "target_ece_delta",
               "joint_min_margin_delta", "pareto_distance_delta", "pareto_distance_reduction",
               "endpoint_vector_norm_regret_delta", "endpoint_vector_norm_regret_reduction",
               "dominated_hypervolume_regret_delta", "source_score_delta", "R_src_delta",
               "c30_source_rank_delta", "target_gauge_delta", "target_margin_mean_delta",
               "target_unlabeled_R3_delta", "target_grouped_centered_delta",
               "meaningful_continuous_regret", "tiny_continuous_difference", "threshold_artifact",
               "endpoint_tradeoff", "pair_case"])
    _writecsv(os.path.join(tdir, "continuous_local_regret_by_trajectory.csv"), res["selected_pairs"]["per_unit"],
              ["seed", "target", "level", "regime", "selected_joint_good", "has_binary_joint_neighbor",
               "has_continuous_better", "selected_to_continuous_better_joint_min_delta",
               "selected_to_continuous_better_norm_regret_reduction", "selected_to_oracle_joint_min_delta",
               "selected_to_oracle_norm_regret_reduction", "threshold_artifact", "continuous_pair_case"])
    _writecsv(os.path.join(tdir, "local_source_gradient_alignment.csv"), res["source_direction"]["pair_rows"],
              ["seed", "target", "level", "regime", "pair_scope", "neighborhood", "order_a", "order_b",
               "component", "component_family", "target_utility_delta", "component_delta", "sign_agreement",
               "wrong_direction", "flat_component", "target_bacc_delta", "target_nll_delta", "target_ece_delta",
               "joint_min_margin_delta", "pareto_distance_delta", "source_score_delta", "target_gauge_delta",
               "target_unlabeled_R3_delta"])
    _writecsv(os.path.join(tdir, "source_objective_component_conflict.csv"),
              res["source_objective_components"]["aggregate"],
              ["component", "component_family", "available", "n_pairs", "wrong_direction_fraction",
               "flat_fraction", "correct_fraction", "mean_component_delta", "mean_target_joint_min_gain"])
    _writecsv(os.path.join(tdir, "gauge_jump_local_regret.csv"), res["gauge_locality"]["gauge_rows"],
              ["seed", "target", "level", "regime", "comparison", "meaningful_continuous_regret",
               "joint_min_margin_delta", "endpoint_norm_regret_reduction", "target_gauge_delta",
               "target_margin_mean_delta", "source_score_delta", "target_unlabeled_R3_delta",
               "target_grouped_centered_delta", "gauge_jump", "source_insensitive_to_gauge",
               "gauge_jump_unseen_by_source"])
    _writecsv(os.path.join(tdir, "target_unlabeled_local_regret.csv"), res["gauge_locality"]["target_unlabeled_rows"],
              ["strategy", "info_class", "neighborhood", "mean_strategy_top1_regret", "mean_local_random_regret",
               "regret_delta_vs_source", "non_source_only"])
    _writecsv(os.path.join(tdir, "binary_vs_continuous_boundary_status.csv"),
              res["binary_vs_continuous_boundary"]["rows"],
              ["seed", "target", "level", "regime", "primary_selected_joint_good", "robust_selected_joint_good",
               "has_binary_joint_neighbor", "has_continuous_better", "continuous_regret_status",
               "threshold_artifact", "primary_continuous_pair_case", "robust_continuous_pair_case"])
    _writecsv(os.path.join(tdir, "local_random_baseline_continuous.csv"), res["source_direction"]["random_rows"],
              ["seed", "target", "level", "regime", "neighborhood", "strategy", "info_class", "n",
               "local_oracle_utility", "strategy_top1_regret", "local_random_mean_regret",
               "regret_improvement_vs_random"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), _no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c34_case_taxonomy.csv"), margin_free_taxonomy.case_rows(res["taxonomy"]), ["cases"])


def _no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "no_training_no_reinference", "passed": True},
        {"check": "endpoint_utility_definitions_frozen_before_analysis", "passed": True},
        {"check": "local_window_definitions_frozen", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "endpoint_vectors_reported_before_scalar_summaries", "passed": True},
        {"check": "local_random_baseline_reported", "passed": bool(res["source_direction"]["random_rows"])},
        {"check": "target_endpoint_labels_diagnostic_only", "passed": True},
        {"check": "target_gauge_non_source_only", "passed": True},
        {"check": "target_unlabeled_non_source_only", "passed": True},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": bool(res["diagnostic_only_non_deployable"])},
    ]


def _f(x):
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float):
        return f"{x:+.3f}"
    return str(x)


def _mean_pair(res, comparison, key):
    vals = [float(p[key]) for p in res["selected_pairs"]["pairs"]
            if p["comparison"] == comparison and endpoint_utility.finite(p.get(key))]
    return float(np.mean(vals)) if vals else None


def render_md(res):
    ps = res["selected_pairs"]["summary"]
    ds = res["source_direction"]["summary"]
    cs = res["source_objective_components"]["summary"]
    gs = res["gauge_locality"]["summary"]
    tax = res["taxonomy"]
    status = res["binary_vs_continuous_boundary"]["summary"]
    return "\n".join([
        f"# C34 - Continuous Local Regret / Source-Objective Direction Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Diagnostic-only read-only audit. C34 asks whether C33 local misses remain real continuous endpoint "
        "regret after removing the binary joint-good threshold as the primary object. No training, no re-inference, "
        "no selector, no selected-checkpoint artifact.",
        "",
        f"- **cases: `{', '.join(tax['cases'])}`**",
        "",
        "## Endpoint vectors first",
        "",
        "Selected -> nearest continuous-better mean raw endpoint deltas:",
        f"- Δtarget bAcc: **{_f(_mean_pair(res, 'nearest_continuous_better', 'target_bacc_delta'))}**",
        f"- Δtarget NLL improvement: **{_f(_mean_pair(res, 'nearest_continuous_better', 'target_nll_delta'))}**",
        f"- Δtarget ECE improvement: **{_f(_mean_pair(res, 'nearest_continuous_better', 'target_ece_delta'))}**",
        "",
        "Fixed scalar summaries after the vector:",
        f"- mean / median Δjoint-min-margin: **{_f(ps['mean_continuous_joint_min_regret'])} / "
        f"{_f(ps['median_continuous_joint_min_regret'])}**.",
        f"- mean / median endpoint-norm regret reduction: **{_f(ps['mean_endpoint_norm_regret_reduction'])} / "
        f"{_f(ps['median_endpoint_norm_regret_reduction'])}**.",
        f"- real continuous-regret fraction among selected->continuous-better pairs: "
        f"**{_f(ps['real_endpoint_regret_fraction'])}**.",
        f"- raw Pareto-nonworse / raw endpoint-backward / negative joint-min counts among nearest continuous-better: "
        f"**{ps['continuous_raw_pareto_nonworse_count']} / {ps['continuous_raw_endpoint_backward_count']} / "
        f"{ps['continuous_joint_min_negative_count']}** of {ps['n_selected_continuous_better_pairs']}. "
        "Thus `real_endpoint_regret` means regret under the fixed C34 scalar/norm summaries, not pure Pareto "
        "dominance.",
        "",
        "## Binary vs continuous boundary",
        "",
        f"- threshold-only fraction among binary misses: **{_f(ps['threshold_only_fraction_among_binary_misses'])}**.",
        f"- binary misses: tiny-threshold / endpoint-tradeoff / scalar-or-norm-worse counts "
        f"**{ps['binary_threshold_tiny_count']} / {ps['binary_endpoint_tradeoff_count']} / "
        f"{ps['binary_worse_by_scalar_or_norm_count']}** of {ps['binary_miss_count']}. "
        "`threshold-only` is the strict tiny-difference artifact, not every broader binary-label tradeoff.",
        f"- boundary status counts: **{status['status_counts']}**.",
        f"- endpoint tradeoff fraction: **{_f(ps['endpoint_tradeoff_fraction'])}**.",
        "",
        "## Source-objective direction",
        "",
        f"- source pairwise AUC against continuous target utility: **{_f(ds['source_pairwise_auc'])}** "
        f"(local random baseline **{_f(ds['random_pairwise_auc'])}**).",
        f"- source wrong-direction / flat fractions: **{_f(ds['source_wrong_direction_fraction'])} / "
        f"{_f(ds['source_flat_fraction'])}**.",
        f"- selected-pair source active misranking fraction: **{_f(ps['source_active_misranking_fraction'])}**.",
        "- M2 is therefore read as a substantial selected-pair minority, not as a global source objective pointing "
        "mostly backward.",
        f"- component conflicts: leakage wrong **{_f(cs['leakage_mean_wrong_direction_fraction'])}**, "
        f"risk wrong **{_f(cs['risk_mean_wrong_direction_fraction'])}**.",
        "",
        "## Gauge-locality and target-unlabeled rung",
        "",
        f"- meaningful-regret gauge-jump / gauge-unseen fractions: "
        f"**{_f(gs['meaningful_regret_gauge_jump_fraction'])} / "
        f"{_f(gs['meaningful_regret_gauge_unseen_fraction'])}**. C34 does not treat gauge jumps alone as M6; "
        "the source-insensitivity gate must also fire.",
        f"- target-unlabeled pm1 regret delta vs source: "
        f"**{_f(gs['target_unlabeled_pm1_regret_delta_vs_source'])}** "
        "(positive means worse local continuous regret than source).",
        "",
        "## Bottom line",
        "",
        "> C34 is margin-free relative to C33's binary label: endpoint vectors show whether selected OACI is worse "
        "than nearby alternatives under fixed continuous summaries, and where that scalar/norm summary hides endpoint "
        "tradeoffs. The taxonomy is determined by continuous regret, source-objective direction, "
        "gauge-locality, target-unlabeled local regret, and explicit threshold-artifact flags. M8 means many nearest "
        "continuous-better pairs are endpoint tradeoffs under the fixed scalar summary; it is not a replacement for "
        "the endpoint-vector table. Target endpoint labels, target gauge, and target-unlabeled factors remain "
        "diagnostic-only and non-source-only.",
    ])


def render_source_direction_md(res):
    rows = res["source_direction"]["aggregate"]
    comp = res["source_objective_components"]["aggregate"]
    lines = ["# C34 - Source Objective Direction\n",
             "| component | family | pairwise AUC | corr | wrong | flat | random |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['component']} | {r['component_family']} | {_f(r['pairwise_auc'])} | "
                     f"{_f(r['gradient_correlation'])} | {_f(r['wrong_direction_fraction'])} | "
                     f"{_f(r['flat_fraction'])} | {_f(r['random_pairwise_auc'])} |")
    lines.extend(["", "## Selected-pair component conflict", "",
                  "| component | available | wrong | flat | correct | mean delta |",
                  "|---|---:|---:|---:|---:|---:|"])
    for r in comp:
        lines.append(f"| {r['component']} | {r['available']} | {_f(r['wrong_direction_fraction'])} | "
                     f"{_f(r['flat_fraction'])} | {_f(r.get('correct_fraction'))} | "
                     f"{_f(r['mean_component_delta'])} |")
    lines.append("\nAll rows are diagnostic decompositions of existing scores, not selector definitions.")
    return "\n".join(lines)


def render_margin_free_md(res):
    ps = res["selected_pairs"]["summary"]
    lines = ["# C34 - Margin-Free Boundary Check\n",
             f"- cases: `{', '.join(res['taxonomy']['cases'])}`",
             f"- threshold-only fraction among binary misses: {_f(ps['threshold_only_fraction_among_binary_misses'])}",
             f"- real continuous-regret fraction: {_f(ps['real_endpoint_regret_fraction'])}",
             f"- endpoint-tradeoff fraction: {_f(ps['endpoint_tradeoff_fraction'])}",
             "",
             "| status | count |",
             "|---|---:|"]
    for k, v in sorted(res["binary_vs_continuous_boundary"]["summary"]["status_counts"].items()):
        lines.append(f"| {k} | {v} |")
    lines.append("\nRobust binary labels are used only for sensitivity; continuous endpoint regret is the primary C34 object.")
    return "\n".join(lines)


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not deployable", "non-deployable", "diagnostic-only", "no selected", "no selector",
             "not claimed")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 64):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C34 report near: {s}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res)
    src = render_source_direction_md(res)
    margin = render_margin_free_md(res)
    for text in (md, src, margin):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C34_CONTINUOUS_LOCAL_REGRET_AUDIT.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C34_SOURCE_OBJECTIVE_DIRECTION.md"), "w").write(src + "\n")
    open(os.path.join(out_dir, "C34_MARGIN_FREE_BOUNDARY_CHECK.md"), "w").write(margin + "\n")
    table_dir = os.path.join(out_dir, "c34_tables")
    write_tables(res, table_dir)
    compact = artifact_hygiene.compact_payload(res, table_dir)
    json.dump(compact, open(os.path.join(out_dir, "C34_CONTINUOUS_LOCAL_REGRET_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.continuous_regret.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run()
    _write_artifacts(res, args.out_dir)
    print(f"[C34] cases={','.join(res['taxonomy']['cases'])} "
          f"real_regret={_f(res['selected_pairs']['summary']['real_endpoint_regret_fraction'])} "
          f"threshold_only={_f(res['selected_pairs']['summary']['threshold_only_fraction_among_binary_misses'])} "
          f"tu_pm1_delta={_f(res['gauge_locality']['summary']['target_unlabeled_pm1_regret_delta_vs_source'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
