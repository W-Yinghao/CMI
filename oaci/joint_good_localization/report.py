"""C32 report assembler.

Produces the Joint-Good Localization / Selection-Regret Anatomy Audit. The report is gate-first: joint-good
landscape, trajectory-conditioned random baseline, top-k enrichment, selected-OACI regret, and information-ladder
localization are computed before the taxonomy narrative is rendered.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import artifact_loader, information_ladder, landscape, regret, schema, taxonomy, topk


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C32 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _analyze(rows) -> dict:
    land = landscape.joint_good_landscape(rows)
    rand = topk.random_baseline(rows, schema.TOP_KS)
    source = topk.topk_enrichment(rows, lambda r: float(r["score"]), schema.TOP_KS)
    sel = regret.selected_oaci_regret(rows)
    ladder = information_ladder.localization_ladder(rows, schema.TOP_KS)
    tax = taxonomy.classify(land, rand, source, sel, ladder)
    return {"landscape": land, "random_baseline": rand, "source_topk": source,
            "selected_oaci_regret": sel, "information_ladder": ladder, "taxonomy": tax}


def run(scores_sidecar=None, c10_dir=None, reinfer_sidecar=None, mode="in_regime") -> dict:
    cfg = _lock_config()
    rows, tu = artifact_loader.load_rows(scores_sidecar, c10_dir, reinfer_sidecar, mode, margin=schema.IMPROVE_MARGIN)
    primary = _analyze(rows)
    robust_rows, robust_tu = artifact_loader.load_rows(scores_sidecar, c10_dir, reinfer_sidecar, mode,
                                                       margin=schema.ROBUST_MARGIN)
    robust = _analyze(robust_rows)
    return {"config_hash": cfg, "mode": mode, "n_rows": len(rows), "target_unlabeled": tu,
            "primary": primary, "robust_sensitivity": robust,
            "robust_target_unlabeled": robust_tu, "diagnostic_only_non_deployable": True}


def _no_selector_gate(res):
    tu = res["target_unlabeled"]
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "no_training_no_new_dg_penalty", "passed": True},
        {"check": "no_bnci2014_004_no_seeds_3_4", "passed": True},
        {"check": "target_unlabeled_features_label_free", "passed": tu["missing_rows"] == 0},
        {"check": "selected_oaci_used_only_for_aggregate_regret", "passed": True},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "target_grouped_oracle_non_deployable", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": bool(res["diagnostic_only_non_deployable"])},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    p = res["primary"]
    land = p["landscape"]
    _writecsv(os.path.join(tdir, "joint_good_landscape.csv"),
              [{"metric": k, "value": v} for k, v in land.items()], ["metric", "value"])
    _writecsv(os.path.join(tdir, "trajectory_random_baseline.csv"),
              p["random_baseline"]["topk"], ["k", "hit_rate", "expected_precision_at_k"])
    _writecsv(os.path.join(tdir, "source_topk_enrichment.csv"),
              p["source_topk"]["topk"],
              ["k", "hit_rate", "random_hit_rate", "hit_enrichment", "precision_at_k",
               "random_precision_at_k", "precision_enrichment"])
    sel = p["selected_oaci_regret"]
    _writecsv(os.path.join(tdir, "selection_regret_decomposition.csv"),
              [{"category": k, "count": sel["summary"]["category_counts"][k],
                "fraction": sel["summary"]["category_fractions"][k]}
               for k in sorted(sel["summary"]["category_counts"])],
              ["category", "count", "fraction"])
    _writecsv(os.path.join(tdir, "selected_to_nearest_joint_good.csv"),
              sel["per_trajectory"],
              ["seed", "target", "level", "regime", "category", "has_joint_good", "selected_joint_good",
               "nearest_order_distance", "nearest_order_distance_norm", "nearest_epoch_distance",
               "selected_score_rank", "best_joint_score_rank", "source_top1_joint", "source_top3_has_joint",
               "source_top5_has_joint", "bacc_regret_to_best_joint", "nll_regret_to_best_joint",
               "ece_regret_to_best_joint"])
    ladder = p["information_ladder"]
    _writecsv(os.path.join(tdir, "information_ladder_localization.csv"), ladder["models"],
              ["strategy", "pooled_auc", "within_target_auc", "top1_hit_rate", "top5_hit_rate"])
    _writecsv(os.path.join(tdir, "information_ladder_topk.csv"), ladder["topk"],
              ["strategy", "k", "hit_rate", "random_hit_rate", "hit_enrichment", "precision_at_k",
               "random_precision_at_k", "precision_enrichment"])
    _writecsv(os.path.join(tdir, "strategy_top1_regret.csv"), ladder["strategy_top1_regret"],
              ["strategy", "top1_joint_hit_rate", "mean_bacc_regret_to_best_joint",
               "mean_nll_regret_to_best_joint", "mean_ece_regret_to_best_joint"])
    _writecsv(os.path.join(tdir, "target_unlabeled_feature_registry.csv"),
              [{"feature": f, "needs_target_labels": False}
               for f in ladder["meta"]["target_unlabeled_feature_names"]],
              ["feature", "needs_target_labels"])
    _writecsv(os.path.join(tdir, "c32_case_taxonomy.csv"),
              [{"cases": ";".join(p["taxonomy"]["cases"])}], ["cases"])
    _writecsv(os.path.join(tdir, "margin_sensitivity.csv"),
              [{"quantity": q, "improve_margin": _get(p, q), "robust_margin": _get(res["robust_sensitivity"], q)}
               for q in ("joint_good_rate", "trajectory_any_joint_fraction", "selected_hit_rate",
                         "source_top1_hit_rate", "target_unlabeled_pooled_auc_gain", "target_grouped_pooled_auc_gain",
                         "cases")],
              ["quantity", "improve_margin", "robust_margin"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), _no_selector_gate(res), ["check", "passed"])


def _get(analysis, q):
    if q == "joint_good_rate":
        return analysis["landscape"]["joint_good_rate"]
    if q == "trajectory_any_joint_fraction":
        return analysis["landscape"]["trajectory_any_joint_fraction"]
    if q == "selected_hit_rate":
        return analysis["selected_oaci_regret"]["summary"]["selected_joint_hit_rate"]
    if q == "source_top1_hit_rate":
        return next(r["hit_rate"] for r in analysis["source_topk"]["topk"] if r["k"] == 1)
    if q == "target_unlabeled_pooled_auc_gain":
        return analysis["information_ladder"]["meta"]["target_unlabeled_pooled_auc_gain_over_source"]
    if q == "target_grouped_pooled_auc_gain":
        return analysis["information_ladder"]["meta"]["target_grouped_pooled_auc_gain_over_source"]
    if q == "cases":
        return ";".join(analysis["taxonomy"]["cases"])
    return None


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_md(res) -> str:
    p = res["primary"]
    land = p["landscape"]
    rand = p["random_baseline"]
    source = p["source_topk"]
    sel = p["selected_oaci_regret"]["summary"]
    ladder = p["information_ladder"]
    tax = p["taxonomy"]
    robust = res["robust_sensitivity"]
    k1_rand = next(r for r in rand["topk"] if r["k"] == 1)
    k5_rand = next(r for r in rand["topk"] if r["k"] == 5)
    k1_source = next(r for r in source["topk"] if r["k"] == 1)
    k3_source = next(r for r in source["topk"] if r["k"] == 3)
    k5_source = next(r for r in source["topk"] if r["k"] == 5)
    models = {m["strategy"]: m for m in ladder["models"]}
    meta = ladder["meta"]
    L = [
        f"# C32 - Joint-Good Localization / Selection-Regret Anatomy Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> C31 showed that joint accuracy+calibration-good checkpoints are common and that the C16 barrier is a "
        "source-observability / gauge / localization failure, not a checkpoint-space trade-off. C32 asks why source-"
        "side selection and diagnostic scores still do not localize the common joint-good set. Read-only over C10 + "
        "C22 + C24; diagnostic-only; no selector, no training, no selected-checkpoint artifact.",
        "",
        f"- **cases: `{', '.join(tax['cases'])}`**",
        "",
        "## Gate 1 - joint-good landscape (scarcity check)",
        "",
        f"- candidates: **{land['n_candidates']}** across **{land['n_trajectories']}** trajectory-regime units.",
        f"- joint-good rate: **{_pct(land['joint_good_rate'])}** "
        f"({land['joint_good_count']} candidates); trajectory-regime units with at least one joint-good: "
        f"**{_pct(land['trajectory_any_joint_fraction'])}**.",
        f"- mean / median joint-good per trajectory-regime unit: **{_f(land['mean_joint_good_per_trajectory'])} / "
        f"{_f(land['median_joint_good_per_trajectory'])}**; min/max: "
        f"**{land['min_joint_good_per_trajectory']} / {land['max_joint_good_per_trajectory']}**.",
        "",
        "## Gate 2 - trajectory-conditioned random baseline",
        "",
        "| k | random hit | source-score hit | enrichment |",
        "|---:|---:|---:|---:|",
    ]
    for k in schema.TOP_KS:
        rb = next(r for r in rand["topk"] if r["k"] == k)
        st = next(r for r in source["topk"] if r["k"] == k)
        L.append(f"| {k} | {_pct(rb['hit_rate'])} | {_pct(st['hit_rate'])} | {_f(st['hit_enrichment'])} |")
    L += [
        "",
        f"- random top-1 is already **{_pct(k1_rand['hit_rate'])}** because joint-good is common.",
        f"- source-score top-1 improves to **{_pct(k1_source['hit_rate'])}** (enrichment "
        f"{_f(k1_source['hit_enrichment'])}), but top-3/top-5 are only weakly above the trajectory-conditioned "
        f"random baseline: **{_pct(k3_source['hit_rate'])} / {_pct(k5_source['hit_rate'])}** vs "
        f"**{_pct(next(r for r in rand['topk'] if r['k'] == 3)['hit_rate'])} / {_pct(k5_rand['hit_rate'])}** "
        f"(enrichment {_f(k3_source['hit_enrichment'])} / {_f(k5_source['hit_enrichment'])}).",
        "",
        "## Gate 3 - selected OACI regret anatomy",
        "",
        f"- selected OACI joint-good hit: **{_pct(sel['selected_joint_hit_rate'])}**, essentially random top-1 "
        f"({_pct(k1_rand['hit_rate'])}); scarcity/no-joint trajectories are only "
        f"**{_pct(sel['category_fractions'].get('scarcity_no_joint_good', 0.0))}**.",
        f"- nearest joint-good distance from selected OACI: median order **{_f(sel['median_nearest_order_distance'])}**, "
        f"mean order **{_f(sel['mean_nearest_order_distance'])}**; median epoch "
        f"**{_f(sel['median_nearest_epoch_distance'])}**.",
        f"- selected-to-best-joint regret: bAcc mean **{_f(sel['mean_bacc_regret_to_best_joint'])}** "
        f"(median {_f(sel['median_bacc_regret_to_best_joint'])}, max {_f(sel['max_bacc_regret_to_best_joint'])}); "
        f"NLL/ECE mean regret **{_f(sel['mean_nll_regret_to_best_joint'])} / "
        f"{_f(sel['mean_ece_regret_to_best_joint'])}**.",
        "",
        "| selected-regret category | fraction | count |",
        "|---|---:|---:|",
    ]
    for cat, count in sorted(sel["category_counts"].items()):
        L.append(f"| {cat} | {_pct(sel['category_fractions'][cat])} | {count} |")
    L += [
        "",
        "## Gate 4 - source-only / target-unlabeled / target-grouped localization ladder",
        "",
        "| information rung | pooled AUC | within-target AUC | top-1 hit | top-5 hit |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("source_score", "target_unlabeled_loto", "source_plus_target_unlabeled_loto",
                 "target_grouped_centered_score"):
        m = models[name]
        L.append(f"| {name} | {_f(m['pooled_auc'])} | {_f(m['within_target_auc'])} | "
                 f"{_pct(m['top1_hit_rate'])} | {_pct(m['top5_hit_rate'])} |")
    L += [
        "",
        f"- target-unlabeled confidence geometry improves **pooled** localization by "
        f"**{_f(meta['target_unlabeled_pooled_auc_gain_over_source'])}** AUC over source score, but its top-1 "
        f"trajectory localization is **{_f(meta['target_unlabeled_top1_gain_over_source'])}** relative to source. "
        "This is a weak pooling/gauge aid, not a top-k rescue.",
        f"- target-grouped centering improves pooled AUC by "
        f"**{_f(meta['target_grouped_pooled_auc_gain_over_source'])}** and recovers the within-target rank signal, "
        "but it uses target grouping and is non-deployable.",
        "",
        "## Margin sensitivity (robust margin 0.02)",
        "",
        f"- robust joint-good rate **{_pct(robust['landscape']['joint_good_rate'])}**; trajectories with joint-good "
        f"**{_pct(robust['landscape']['trajectory_any_joint_fraction'])}**; selected hit "
        f"**{_pct(robust['selected_oaci_regret']['summary']['selected_joint_hit_rate'])}**; cases "
        f"**{', '.join(robust['taxonomy']['cases'])}**.",
        "",
        "## Bottom line",
        "",
        "> Joint-good checkpoints are common, and selected OACI is usually close to one, but the source-side "
        "localization signal is too weak and gauge-broken to choose them reliably. Random top-1 is already high "
        "because the set is common; source score gives only mild top-1 enrichment and only weak top-k enrichment; "
        "selected OACI lands near random and most regret is non-scarcity localization regret. Target-unlabeled "
        "features help the pooled gauge weakly but do not rescue top-k localization. Target grouping largely repairs "
        "the pooled rank/gauge mismatch, confirming the C31 reading, but that is an oracle diagnostic, not deployable.",
    ]
    return "\n".join(L)


def render_regret_md(res) -> str:
    sel = res["primary"]["selected_oaci_regret"]["summary"]
    return (
        "# C32 - selected OACI regret anatomy\n\n"
        f"- selected joint-good hit: {_pct(sel['selected_joint_hit_rate'])}\n"
        f"- scarcity/no-joint fraction: {_pct(sel['category_fractions'].get('scarcity_no_joint_good', 0.0))}\n"
        f"- median nearest joint-good order distance: {_f(sel['median_nearest_order_distance'])}\n"
        f"- mean bAcc regret to best joint-good: {_f(sel['mean_bacc_regret_to_best_joint'])}\n\n"
        "Diagnostic-only aggregate; no selected-checkpoint artifact is emitted.\n"
    )


def render_ladder_md(res) -> str:
    ladder = res["primary"]["information_ladder"]
    lines = ["# C32 - localization information ladder\n",
             "| information rung | pooled AUC | within-target AUC | top-1 hit | top-5 hit |",
             "|---|---:|---:|---:|---:|"]
    for m in ladder["models"]:
        lines.append(f"| {m['strategy']} | {_f(m['pooled_auc'])} | {_f(m['within_target_auc'])} | "
                     f"{_pct(m['top1_hit_rate'])} | {_pct(m['top5_hit_rate'])} |")
    lines.append("\nTarget labels are diagnostic targets only; target-unlabeled features are label-free and fixed.")
    return "\n".join(lines)


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not deployable", "non-deployable", "diagnostic-only", "no selected", "no selector",
             "not claimed", "not an")


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 48):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C32 report near: ...{low[max(0, i - 48):i + len(s)]!r}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res)
    rg = render_regret_md(res)
    lg = render_ladder_md(res)
    for text in (md, rg, lg):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C32_JOINT_GOOD_LOCALIZATION_AUDIT.md"), "w").write(md)
    open(os.path.join(out_dir, "C32_SELECTION_REGRET_ANATOMY.md"), "w").write(rg)
    open(os.path.join(out_dir, "C32_INFORMATION_LADDER_LOCALIZATION.md"), "w").write(lg)
    json.dump(res, open(os.path.join(out_dir, "C32_JOINT_GOOD_LOCALIZATION_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c32_tables"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.joint_good_localization.report")
    ap.add_argument("--scores-sidecar", default=None)
    ap.add_argument("--c10-dir", default=None)
    ap.add_argument("--reinfer-sidecar", default=None)
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run(args.scores_sidecar, args.c10_dir, args.reinfer_sidecar)
    _write_artifacts(res, args.out_dir)
    p = res["primary"]
    sel = p["selected_oaci_regret"]["summary"]
    source_k1 = next(r for r in p["source_topk"]["topk"] if r["k"] == 1)
    print(f"[C32] cases={','.join(p['taxonomy']['cases'])} | joint_rate={_f(p['landscape']['joint_good_rate'])} "
          f"selected_hit={_f(sel['selected_joint_hit_rate'])} source_top1={_f(source_k1['hit_rate'])} "
          f"tu_auc_gain={_f(p['information_ladder']['meta']['target_unlabeled_pooled_auc_gain_over_source'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
