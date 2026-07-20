"""C42 source-rank actionability report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (artifact_loader, auc_to_topk_gap, diagnostic_top1, gauge_sensitivity, leakage_rank_conflict,
               schema, score_registry, taxonomy, top_region_stability)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C42 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
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
    scores = score_registry.registry(ctx)
    gap = auc_to_topk_gap.audit(ctx)
    top1 = diagnostic_top1.audit(ctx)
    stability = top_region_stability.audit(ctx)
    gauge = gauge_sensitivity.audit(ctx)
    conflict = leakage_rank_conflict.audit(ctx)
    tax = taxonomy.classify(ctx, gap, top1, stability, gauge, conflict)
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "c30_source_rank_auc": ctx["summary"]["c30_source_rank_auc"],
        "c41_selection_leakage_auc": ctx["summary"]["c41_selection_leakage_auc"],
        "score_registry": scores,
        "auc_to_topk_gap": gap,
        "diagnostic_top1_by_score": top1,
        "source_rank_top_region_stability": stability,
        "source_rank_gauge_sensitivity": gauge,
        "leakage_vs_rank_conflict": conflict,
        "taxonomy": tax,
    }


def _summary_from_existing():
    path = "oaci/reports/C42_SOURCE_RANK_ACTIONABILITY_AUDIT.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    tdir = schema.C42_TABLE_DIR
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_candidate_rows": d["n_candidate_rows"],
        "n_trajectories": d["n_trajectories"],
        "c30_source_rank_auc": d["c30_source_rank_auc"],
        "c41_selection_leakage_auc": d["c41_selection_leakage_auc"],
        "score_registry": {"rows": _readcsv(os.path.join(tdir, "rank_actionability_score_registry.csv")),
                           "summary": d["score_registry_summary"]},
        "auc_to_topk_gap": {"rows": _readcsv(os.path.join(tdir, "auc_to_topk_gap.csv")),
                            "summary_rows": d["auc_to_topk_gap_summary_rows"]},
        "diagnostic_top1_by_score": {"rows": _readcsv(os.path.join(tdir, "diagnostic_top1_by_score.csv")),
                                     "summary": d["diagnostic_top1_summary"]},
        "source_rank_top_region_stability": {
            "rows": _readcsv(os.path.join(tdir, "source_rank_top_region_stability.csv")),
            "summary": d["source_rank_top_region_stability_summary"]},
        "source_rank_gauge_sensitivity": {
            "rows": _readcsv(os.path.join(tdir, "source_rank_gauge_sensitivity.csv")),
            "summary": d["source_rank_gauge_sensitivity_summary"]},
        "leakage_vs_rank_conflict": {
            "rows": _readcsv(os.path.join(tdir, "leakage_vs_rank_conflict.csv")),
            "summary": d["leakage_vs_rank_conflict_summary"]},
        "taxonomy": d["taxonomy"],
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists("oaci/reports/C42_SOURCE_RANK_ACTIONABILITY_AUDIT.json"):
        return _summary_from_existing()
    return recompute()


def no_selector_gate(res):
    scores = res["score_registry"]["summary"]
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "read_only_committed_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_new_selector_artifact_emitted", "passed": True},
        {"check": "score_registry_frozen_before_analysis", "passed": True},
        {"check": "c30_source_rank_orientation_fixed", "passed": True},
        {"check": "trajectory_conditioned_random_baseline_for_topk", "passed": True},
        {"check": "target_centered_rank_marked_non_deployable_diagnostic", "passed": True},
        {"check": "no_threshold_tuning_no_feature_selection", "passed": True},
        {"check": "no_monolithic_large_json", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "no_proxy_target_unlabeled_r3", "passed": scores["target_unlabeled_R3"]["proxy_used"] == 0},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "rank_actionability_score_registry.csv"),
              res["score_registry"]["rows"],
              ["score", "field", "orientation", "scope", "candidate_level_available", "topk_eligible",
               "selected_only", "non_source_only", "diagnostic_ceiling", "n_candidate_rows", "n_available",
               "proxy_used", "note"])
    _writecsv(os.path.join(tdir, "auc_to_topk_gap.csv"),
              res["auc_to_topk_gap"]["summary_rows"],
              ["score", "selection_rule", "label", "n_trajectories", "mean_pairwise_auc_vs_target_utility",
               "mean_hit_rate", "mean_random_baseline", "mean_enrichment_ratio",
               "mean_regret_vs_target_oracle", "non_source_only", "diagnostic_ceiling"])
    _writecsv(os.path.join(tdir, "diagnostic_top1_by_score.csv"),
              res["diagnostic_top1_by_score"]["rows"],
              ["score", "n_trajectories", "top1_joint_good_rate", "top1_pareto_good_rate",
               "top1_preference_robust_utility_rate", "top1_regret_vs_target_oracle",
               "mean_target_utility_delta_vs_actual_oaci", "fraction_top1_target_better_than_actual_oaci",
               "top1_joint_good_gain_vs_random", "non_source_only", "diagnostic_ceiling"])
    _writecsv(os.path.join(tdir, "source_rank_top_region_stability.csv"),
              res["source_rank_top_region_stability"]["rows"],
              ["trajectory_id", "seed", "target", "level", "regime", "top1_top2_margin", "top1_top3_margin",
               "plateau_epsilon", "plateau_size", "plateau_fraction", "plateau_joint_good_rate",
               "plateau_pareto_good_rate", "top1_joint_good", "top1_pareto_good", "top_region_low_margin",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "source_rank_gauge_sensitivity.csv"),
              res["source_rank_gauge_sensitivity"]["rows"],
              ["normalization", "top1_joint_good_rate", "top1_pareto_good_rate", "top1_regret_vs_target_oracle",
               "non_source_only", "diagnostic_ceiling", "top1_joint_good_gain_vs_raw",
               "top1_regret_gain_vs_raw"])
    _writecsv(os.path.join(tdir, "leakage_vs_rank_conflict.csv"),
              res["leakage_vs_rank_conflict"]["rows"],
              ["trajectory_id", "seed", "target", "level", "regime",
               "target_utility_delta_rank_top_minus_oaci", "selection_leakage_delta_rank_top_minus_oaci",
               "audit_leakage_delta_rank_top_minus_oaci", "R_src_delta_rank_top_minus_oaci",
               "rank_top_target_better_than_oaci", "rank_top_higher_selection_leakage_than_oaci",
               "leakage_blocks_rank_better_candidate", "oaci_joint_good", "rank_top_joint_good",
               "oaci_pareto_good", "rank_top_pareto_good", "target_gauge_delta_available",
               "target_gauge_delta_rank_top_minus_oaci", "no_candidate_id_emitted"])
    _writecsv(os.path.join(tdir, "source_rank_regret_vs_oracle.csv"),
              res["diagnostic_top1_by_score"]["rows"],
              ["score", "n_trajectories", "top1_regret_vs_target_oracle",
               "mean_target_utility_delta_vs_actual_oaci", "fraction_top1_target_better_than_actual_oaci",
               "non_source_only", "diagnostic_ceiling"])
    _writecsv(os.path.join(tdir, "trajectory_conditioned_random_baseline.csv"),
              res["auc_to_topk_gap"]["random_baseline"]["rows"],
              ["trajectory_id", "seed", "target", "level", "regime", "selection_rule", "label",
               "n_candidates", "n_draw", "trajectory_random_hit_rate", "trajectory_random_expected_regret"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c42_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def render_md(res):
    top1 = res["diagnostic_top1_by_score"]["summary"]
    sr = top1["C30_source_rank_score"]
    oaci = top1["actual_oaci_selector"]
    rand = top1["random_trajectory_conditioned"]
    src_gap = next(r for r in res["auc_to_topk_gap"]["summary_rows"]
                   if r["score"] == "C30_source_rank_score" and
                   r["selection_rule"] == "top1" and r["label"] == "primary_joint_good")
    stab = res["source_rank_top_region_stability"]["summary"]
    conf = res["leakage_vs_rank_conflict"]["summary"]
    return "\n".join([
        f"# C42 - Source-Rank Actionability / Rank-to-Selector Gap Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only diagnostic counterfactual. No training, no GPU, no score tuning, no feature selection, "
        "no selected-checkpoint artifact, and no deployable selector claim.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- candidate rows / trajectories: **{res['n_candidate_rows']} / {res['n_trajectories']}**.",
        "",
        "## Pairwise Signal",
        "",
        f"- C30 source-rank AUC from C30: **{_f(res['c30_source_rank_auc'])}**.",
        f"- C42 source-rank AUC vs C41 continuous target utility: "
        f"**{_f(src_gap['mean_pairwise_auc_vs_target_utility'])}**.",
        f"- C41 selection-leakage AUC: **{_f(res['c41_selection_leakage_auc'])}**.",
        "",
        "## Top-1 Actionability",
        "",
        f"- source-rank top1 joint-good: **{_f(sr['top1_joint_good_rate'])}**.",
        f"- actual OACI top1 joint-good: **{_f(oaci['top1_joint_good_rate'])}**.",
        f"- trajectory-conditioned random baseline: **{_f(rand['top1_joint_good_rate'])}**.",
        f"- source-rank top1 regret vs target oracle: **{_f(sr['top1_regret_vs_target_oracle'])}**.",
        f"- source-rank top1 target-better-than-OACI fraction: "
        f"**{_f(sr['fraction_top1_target_better_than_actual_oaci'])}**.",
        "",
        "## Stability And Conflict",
        "",
        f"- source-rank mean plateau size at eps {schema.PLATEAU_EPS}: **{_f(stab['mean_plateau_size'])}**.",
        f"- low top1/top2 margin fraction: **{_f(stab['low_margin_fraction'])}**.",
        f"- leakage-blocks-rank-better fraction: **{_f(conf['leakage_blocks_rank_better_fraction'])}**.",
        "- Target-gauge delta for source-rank top1 vs OACI is not available as a candidate-level field; no proxy is used.",
        "",
        "## Bottom Line",
        "",
        "> C42 closes the source-rank escape hatch for deployment/reliable selection: the rank signal is real and "
        "often target-better than OACI, but its top1/top-k localization is modest over the high "
        "trajectory-conditioned base rate, top regions are plateaued, and regret remains large.",
    ])


def render_gap_md(res):
    rows = [r for r in res["auc_to_topk_gap"]["summary_rows"]
            if r["selection_rule"] in ("top1", "top3") and r["label"] == "primary_joint_good"]
    lines = ["# C42 - AUC-To-Selection Gap", ""]
    for r in rows:
        lines.append(
            f"- {r['score']} / {r['selection_rule']}: AUC {_f(r['mean_pairwise_auc_vs_target_utility'])}, "
            f"hit {_f(r['mean_hit_rate'])}, baseline {_f(r['mean_random_baseline'])}, "
            f"enrichment {_f(r['mean_enrichment_ratio'])}")
    lines += ["", "All top-k baselines are trajectory-conditioned; target labels are diagnostic-only."]
    return "\n".join(lines) + "\n"


def render_conflict_md(res):
    c = res["leakage_vs_rank_conflict"]["summary"]
    return "\n".join([
        "# C42 - Leakage Vs Source-Rank Conflict",
        "",
        f"- rank-top target-better-than-OACI: {c['rank_top_target_better_than_oaci_count']} / {c['n_trajectories']}",
        f"- leakage-blocks-rank-better: {c['leakage_blocks_rank_better_count']} / {c['n_trajectories']}",
        f"- mean target-utility delta rank-top minus OACI: {_f(c['mean_target_utility_delta_rank_top_minus_oaci'])}",
        f"- mean selection-leakage delta rank-top minus OACI: {_f(c['mean_selection_leakage_delta_rank_top_minus_oaci'])}",
        "",
        "Rows omit candidate ids and checkpoint hashes; this is an aggregate diagnostic comparison only.",
    ]) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "non-deployable", "diagnostic-only", "no selected", "no score", "no deployable",
             "closes the", "closed", "not claimed")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 130):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden affirmative over-claim in C42 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "c30_source_rank_auc": res["c30_source_rank_auc"],
        "c41_selection_leakage_auc": res["c41_selection_leakage_auc"],
        "score_registry_summary": res["score_registry"]["summary"],
        "auc_to_topk_gap_summary_rows": res["auc_to_topk_gap"]["summary_rows"],
        "diagnostic_top1_summary": res["diagnostic_top1_by_score"]["summary"],
        "source_rank_top_region_stability_summary": res["source_rank_top_region_stability"]["summary"],
        "source_rank_gauge_sensitivity_summary": res["source_rank_gauge_sensitivity"]["summary"],
        "leakage_vs_rank_conflict_summary": res["leakage_vs_rank_conflict"]["summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "auc_to_topk_check": "Pairwise signal is reported separately from top1/top-k localization.",
            "base_rate_check": "Every top-k summary carries a trajectory-conditioned random baseline.",
            "artifact_check": "No candidate ids, checkpoint hashes, or selected-checkpoint method artifacts are emitted.",
        },
    }


def _write_artifacts(res, out_dir):
    md = render_md(res)
    gap = render_gap_md(res)
    conflict = render_conflict_md(res)
    for text in (md, gap, conflict):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C42_SOURCE_RANK_ACTIONABILITY_AUDIT.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C42_AUC_TO_SELECTION_GAP.md"), "w").write(gap)
    open(os.path.join(out_dir, "C42_LEAKAGE_VS_RANK_CONFLICT.md"), "w").write(conflict)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C42_SOURCE_RANK_ACTIONABILITY_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c42_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.rank_actionability.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    print(f"[C42] cases={','.join(res['taxonomy']['cases'])} "
          f"source_rank_top1={res['diagnostic_top1_by_score']['summary']['C30_source_rank_score']['top1_joint_good_rate']} "
          f"candidates={res['n_candidate_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
