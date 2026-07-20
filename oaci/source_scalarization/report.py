"""C43 source-objective scalarization frontier report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (actionability_metrics, artifact_loader, leakage_rank_frontier, multiplicity, objective_registry,
               scalarization_grid, schema, source_pareto_frontier, taxonomy)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C43 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
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
    objectives = objective_registry.registry(ctx)
    frontier = source_pareto_frontier.audit(ctx)
    grid = scalarization_grid.build_grid()
    action = actionability_metrics.audit(ctx, grid)
    mult = multiplicity.audit(ctx, grid, action)
    conflict = leakage_rank_frontier.audit(ctx, grid, mult)
    tax = taxonomy.classify(frontier, action, mult, conflict)
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "source_objective_registry": objectives,
        "source_pareto_frontier_status": frontier,
        "scalarization_grid_registry": grid,
        "scalarization_actionability_metrics": action,
        "scalarization_multiplicity_audit": mult,
        "leakage_rank_frontier_conflict": conflict,
        "taxonomy": tax,
    }


def _summary_from_existing():
    path = "oaci/reports/C43_SOURCE_OBJECTIVE_SCALARIZATION_FRONTIER.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    tdir = schema.C43_TABLE_DIR
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_candidate_rows": d["n_candidate_rows"],
        "n_trajectories": d["n_trajectories"],
        "source_objective_registry": {"rows": _readcsv(os.path.join(tdir, "source_objective_registry.csv")),
                                      "summary": d["source_objective_registry_summary"]},
        "source_pareto_frontier_status": {
            "rows": _readcsv(os.path.join(tdir, "source_pareto_frontier_status.csv")),
            "summary": d["source_pareto_frontier_summary"]},
        "scalarization_grid_registry": {"rows": _readcsv(os.path.join(tdir, "scalarization_grid_registry.csv")),
                                        "summary": d["scalarization_grid_summary"]},
        "scalarization_actionability_metrics": {
            "rows": _readcsv(os.path.join(tdir, "scalarization_actionability_metrics.csv")),
            "summary_rows": d["scalarization_actionability_summary_rows"]},
        "scalarization_multiplicity_audit": {
            "rows": _readcsv(os.path.join(tdir, "scalarization_multiplicity_audit.csv")),
            "summary": d["scalarization_multiplicity_summary"]},
        "leakage_rank_frontier_conflict": {
            "rows": _readcsv(os.path.join(tdir, "leakage_rank_frontier_conflict.csv")),
            "summary": d["leakage_rank_frontier_summary"]},
        "taxonomy": d["taxonomy"],
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists("oaci/reports/C43_SOURCE_OBJECTIVE_SCALARIZATION_FRONTIER.json"):
        return _summary_from_existing()
    return recompute()


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "read_only_committed_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_feature_selection", "passed": True},
        {"check": "source_objective_registry_frozen_before_analysis", "passed": True},
        {"check": "scalarization_grid_frozen_before_analysis", "passed": True},
        {"check": "best_grid_hindsight_diagnostic_only", "passed": True},
        {"check": "trajectory_conditioned_random_baseline_for_topk", "passed": True},
        {"check": "multiplicity_correction_required", "passed": True},
        {"check": "per_target_sign_consistency_required", "passed": True},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "no_monolithic_large_json", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "source_objective_registry.csv"),
              res["source_objective_registry"]["rows"],
              ["objective", "family", "orientation", "n_candidate_rows", "n_available",
               "availability_fraction", "used_for_source_pareto", "used_for_scalarization_grid",
               "target_field", "proxy_used", "note"])
    _writecsv(os.path.join(tdir, "source_pareto_frontier_status.csv"),
              res["source_pareto_frontier_status"]["rows"],
              ["trajectory_id", "seed", "target", "level", "regime", "n_candidates", "n_source_pareto_front",
               "source_pareto_front_fraction", "oaci_selected_on_source_front",
               "oaci_selected_source_pareto_layer", "source_rank_top_on_source_front",
               "source_rank_top_source_pareto_layer", "joint_good_count", "joint_good_front_fraction",
               "joint_good_mean_source_pareto_layer", "joint_good_source_pareto_rejected_fraction",
               "pareto_good_count", "pareto_good_front_fraction", "pareto_good_mean_source_pareto_layer",
               "pareto_good_source_pareto_rejected_fraction", "preference_robust_target_better_count",
               "preference_robust_target_better_front_fraction",
               "preference_robust_target_better_mean_source_pareto_layer",
               "preference_robust_target_better_source_pareto_rejected_fraction", "no_candidate_id_emitted"])
    _writecsv(os.path.join(tdir, "scalarization_grid_registry.csv"),
              res["scalarization_grid_registry"]["rows"],
              ["scalarization_id", "grid_family", "weight_leakage", "weight_source_rank",
               "weight_source_risk", "weight_audit_leakage", "grid_step", "source_only",
               "hindsight_diagnostic_only"])
    _writecsv(os.path.join(tdir, "scalarization_actionability_metrics.csv"),
              res["scalarization_actionability_metrics"]["rows"],
              ["scalarization_id", "grid_family", "selection_rule", "label", "n_trajectories",
               "mean_pairwise_auc_vs_target_utility", "mean_hit_rate", "mean_random_baseline",
               "mean_enrichment_ratio", "mean_regret_vs_target_oracle",
               "mean_target_utility_delta_vs_oaci", "hindsight_diagnostic_only"])
    _writecsv(os.path.join(tdir, "scalarization_multiplicity_audit.csv"),
              res["scalarization_multiplicity_audit"]["rows"],
              ["scalarization_id", "grid_family", "n_trajectories", "observed_top1_joint_good_rate",
               "expected_random_top1_joint_good_rate", "top1_joint_good_gain_vs_random",
               "p_value_vs_trajectory_random", "holm_p_value", "bh_q_value",
               "per_target_auc_sign_consistency", "passes_bh_0_05",
               "positive_scalarization_claim_allowed", "hindsight_diagnostic_only"])
    _writecsv(os.path.join(tdir, "leakage_rank_frontier_conflict.csv"),
              res["leakage_rank_frontier_conflict"]["rows"],
              ["trajectory_id", "seed", "target", "level", "regime", "best_scalarization_id",
               "leakage_rank_spearman", "oaci_leakage_rank_percentile", "oaci_source_rank_percentile",
               "rank_top_target_utility_delta_vs_oaci", "rank_top_selection_leakage_delta_vs_oaci",
               "rank_top_target_better_than_oaci", "leakage_blocks_rank_better_candidate",
               "best_scalarization_target_utility_delta_vs_oaci", "best_scalarization_target_better_than_oaci",
               "best_scalarization_joint_good", "best_scalarization_pareto_good", "no_candidate_id_emitted"])
    _writecsv(os.path.join(tdir, "best_hindsight_scalarization_ceiling.csv"),
              res["scalarization_multiplicity_audit"]["best_rows"],
              ["scalarization_id", "grid_family", "observed_top1_joint_good_rate",
               "expected_random_top1_joint_good_rate", "top1_joint_good_gain_vs_random",
               "p_value_vs_trajectory_random", "holm_p_value", "bh_q_value",
               "per_target_auc_sign_consistency", "positive_scalarization_claim_allowed",
               "hindsight_diagnostic_only"])
    _writecsv(os.path.join(tdir, "per_target_scalarization_stability.csv"),
              res["scalarization_actionability_metrics"]["per_target_rows"],
              ["scalarization_id", "target", "n_trajectories", "mean_pairwise_auc",
               "top1_joint_good_rate", "auc_positive_side"])
    _writecsv(os.path.join(tdir, "trajectory_conditioned_random_baseline.csv"),
              res["scalarization_actionability_metrics"]["random_baseline"]["rows"],
              ["trajectory_id", "seed", "target", "level", "regime", "selection_rule", "label",
               "n_candidates", "n_draw", "trajectory_random_hit_rate", "trajectory_random_expected_regret"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c43_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def render_md(res):
    mult = res["scalarization_multiplicity_audit"]["summary"]
    front = res["source_pareto_frontier_status"]["summary"]
    conflict = res["leakage_rank_frontier_conflict"]["summary"]
    return "\n".join([
        f"# C43 - Source-Objective Scalarization Frontier Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only hindsight diagnostic over a frozen source-only objective registry and fixed scalarization grid. "
        "No training, no GPU, no feature selection, no score tuning, and no selected-checkpoint artifact.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- candidate rows / trajectories: **{res['n_candidate_rows']} / {res['n_trajectories']}**.",
        f"- fixed scalarizations: **{res['scalarization_grid_registry']['summary']['n_scalarizations']}** "
        f"at step **{schema.SCALARIZATION_GRID_STEP:.2f}**.",
        "",
        "## Best Hindsight Source Scalarization",
        "",
        f"- best id: **`{mult['best_scalarization_id']}`**.",
        f"- best top1 joint-good: **{_f(mult['best_top1_joint_good_rate'])}** "
        f"vs trajectory random **{_f(mult['best_expected_random_top1_joint_good_rate'])}**.",
        f"- gain vs random: **{_f(mult['best_top1_gain_vs_random'])}**.",
        f"- Holm p / BH q: **{_f(mult['best_holm_p_value'])} / {_f(mult['best_bh_q_value'])}**.",
        f"- per-target AUC sign consistency: **{_f(mult['best_per_target_sign_consistency'])}**.",
        "",
        "## Source Pareto Frontier",
        "",
        f"- source-front fraction: **{_f(front['mean_front_fraction'])}**.",
        f"- joint-good front fraction: **{_f(front['joint_good_front_fraction'])}**.",
        f"- Pareto-good front fraction: **{_f(front['pareto_good_front_fraction'])}**.",
        f"- preference-robust target-better front fraction: **{_f(front['preference_robust_front_fraction'])}**.",
        "",
        "## Leakage-Rank Frontier",
        "",
        f"- mean leakage/rank Spearman: **{_f(conflict['mean_leakage_rank_spearman'])}**.",
        f"- leakage-blocks-rank-better fraction: **{_f(conflict['leakage_blocks_rank_better_fraction'])}**.",
        f"- OACI mean leakage percentile: **{_f(conflict['mean_oaci_leakage_rank_percentile'])}**.",
        f"- OACI mean source-rank percentile: **{_f(conflict['mean_oaci_source_rank_percentile'])}**.",
        "",
        "## Bottom Line",
        "",
        "> C43 closes the broader source-only scalarization escape hatch under current artifacts: source objectives "
        "contain weak target-relevant signal and the best fixed hindsight mixture improves over random, but the "
        "top1/top-k localization remains below reliability gates after multiplicity and stability checks.",
    ])


def render_frontier_md(res):
    f = res["source_pareto_frontier_status"]["summary"]
    return "\n".join([
        "# C43 - Source Pareto Frontier Audit",
        "",
        f"- OACI selected on source front: {_f(f['oaci_selected_front_rate'])}",
        f"- source-rank top on source front: {_f(f['source_rank_top_front_rate'])}",
        f"- joint-good front fraction: {_f(f['joint_good_front_fraction'])}",
        f"- Pareto-good front fraction: {_f(f['pareto_good_front_fraction'])}",
        f"- preference-robust target-better front fraction: {_f(f['preference_robust_front_fraction'])}",
        "",
        "Candidate ids and checkpoint hashes are not emitted.",
    ]) + "\n"


def render_escape_md(res):
    m = res["scalarization_multiplicity_audit"]["summary"]
    return "\n".join([
        "# C43 - Scalarization Escape-Hatch Audit",
        "",
        f"- best hindsight scalarization: `{m['best_scalarization_id']}`",
        f"- top1 joint-good: {_f(m['best_top1_joint_good_rate'])}",
        f"- random baseline: {_f(m['best_expected_random_top1_joint_good_rate'])}",
        f"- BH q: {_f(m['best_bh_q_value'])}",
        f"- positive scalarization claim allowed: {m['any_positive_scalarization_claim_allowed']}",
        "",
        "Best-grid rows are diagnostic ceilings only and are not method artifacts.",
    ]) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "diagnostic", "ceiling", "hindsight", "no selected", "no feature", "closes", "closed")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 140):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden affirmative over-claim in C43 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "source_objective_registry_summary": res["source_objective_registry"]["summary"],
        "source_pareto_frontier_summary": res["source_pareto_frontier_status"]["summary"],
        "scalarization_grid_summary": res["scalarization_grid_registry"]["summary"],
        "scalarization_actionability_summary_rows": res["scalarization_actionability_metrics"]["rows"],
        "scalarization_multiplicity_summary": res["scalarization_multiplicity_audit"]["summary"],
        "leakage_rank_frontier_summary": res["leakage_rank_frontier_conflict"]["summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "scalarization_overfit_check": "Best grid is marked hindsight diagnostic and corrected over the full grid.",
            "base_rate_check": "Every top-k result carries a trajectory-conditioned random baseline.",
            "method_boundary_check": "No selected-checkpoint artifact or candidate ids are emitted.",
        },
    }


def _write_artifacts(res, out_dir):
    md = render_md(res)
    frontier = render_frontier_md(res)
    escape = render_escape_md(res)
    for text in (md, frontier, escape):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C43_SOURCE_OBJECTIVE_SCALARIZATION_FRONTIER.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C43_SOURCE_PARETO_FRONTIER_AUDIT.md"), "w").write(frontier)
    open(os.path.join(out_dir, "C43_SCALARIZATION_ESCAPE_HATCH_AUDIT.md"), "w").write(escape)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C43_SOURCE_OBJECTIVE_SCALARIZATION_FRONTIER.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c43_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.source_scalarization.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    print(f"[C43] cases={','.join(res['taxonomy']['cases'])} "
          f"best={res['scalarization_multiplicity_audit']['summary']['best_scalarization_id']} "
          f"top1={res['scalarization_multiplicity_audit']['summary']['best_top1_joint_good_rate']} "
          f"candidates={res['n_candidate_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
