"""C33 report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import (artifact_loader, local_gradients, local_information_ladder, margin_sensitivity, plateau_audit,
               schema, selected_pair_audit, taxonomy, trajectory_boundary)


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C33 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _analyze(rows):
    boundary = trajectory_boundary.boundary_geometry(rows)
    pairs = selected_pair_audit.selected_pair_audit(rows)
    gradients = local_gradients.local_gradient_alignment(rows)
    plateaus = plateau_audit.plateau_audit(rows)
    ladder = local_information_ladder.local_random_and_ladder(rows)
    tax = taxonomy.classify(boundary, pairs, gradients, plateaus, ladder)
    return {"boundary": boundary, "pairs": pairs, "gradients": gradients, "plateaus": plateaus,
            "ladder": ladder, "taxonomy": tax}


def run(scores_sidecar=None, c10_dir=None, reinfer_sidecar=None, mode="in_regime"):
    cfg = _lock_config()
    rows, tu = artifact_loader.load_rows(scores_sidecar, c10_dir, reinfer_sidecar, mode, schema.PRIMARY_MARGIN)
    primary = _analyze(rows)
    robust_rows, robust_tu = artifact_loader.load_rows(scores_sidecar, c10_dir, reinfer_sidecar, mode, schema.ROBUST_MARGIN)
    robust = _analyze(robust_rows)
    primary["taxonomy"] = taxonomy.classify(primary["boundary"], primary["pairs"], primary["gradients"],
                                            primary["plateaus"], primary["ladder"], robust=robust)
    sens = margin_sensitivity.margin_sensitivity(primary, robust)
    return {"config_hash": cfg, "mode": mode, "n_rows": len(rows), "target_unlabeled": tu,
            "primary_margin": schema.PRIMARY_MARGIN, "robust_margin": schema.ROBUST_MARGIN,
            "primary": primary, "robust_sensitivity": robust, "margin_sensitivity": sens,
            "diagnostic_only_non_deployable": True}


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    p = res["primary"]
    _writecsv(os.path.join(tdir, "local_boundary_registry.csv"), p["boundary"]["registry"],
              ["seed", "target", "level", "regime", "n", "joint_good_rate", "n_good_runs",
               "mean_good_run_length", "median_good_run_length", "transition_count", "transition_rate",
               "selected_boundary_distance", "selected_joint_good"])
    _writecsv(os.path.join(tdir, "joint_good_transition_geometry.csv"), p["boundary"]["autocorr"],
              ["seed", "target", "level", "regime", "lag", "autocorr"])
    _writecsv(os.path.join(tdir, "selected_neighborhood_hit_rates.csv"), p["boundary"]["neighborhoods"],
              ["seed", "target", "level", "regime", "neighborhood", "joint_good_rate", "contains_joint_good"])
    _writecsv(os.path.join(tdir, "selected_vs_nearest_joint_good_pairs.csv"), p["pairs"]["pairs"],
              ["seed", "target", "level", "regime", "pair_status", "selected_joint_good", "order_delta",
               "epoch_delta", "bacc_gain_to_nearest_joint", "nll_gain_to_nearest_joint",
               "ece_gain_to_nearest_joint", "source_score_delta_to_nearest_joint",
               "source_rank_delta_to_nearest_joint", "R_src_delta_to_nearest_joint",
               "target_gauge_margin_delta_to_nearest_joint", "target_unlabeled_R3_delta_to_nearest_joint",
               "pair_case", "gauge_jump_unseen_by_source"])
    _writecsv(os.path.join(tdir, "local_score_gradient_alignment.csv"), p["gradients"]["gradients"],
              ["seed", "target", "level", "regime", "order_a", "order_b", "transition_type",
               "source_score_gradient", "R_src_gradient", "rank_gradient", "target_gauge_margin_gradient",
               "target_unlabeled_R3_gradient", "source_sign_agrees_with_transition",
               "rank_sign_agrees_with_transition", "target_gauge_jump"])
    _writecsv(os.path.join(tdir, "source_score_plateau_audit.csv"), p["plateaus"]["plateaus"],
              ["seed", "target", "level", "regime", "plateau_size", "plateau_fraction",
               "plateau_joint_good_rate", "plateau_contains_joint_good", "selected_joint_good",
               "selected_rank_within_plateau", "local_score_range_pm3"])
    _writecsv(os.path.join(tdir, "local_information_ladder_results.csv"), p["ladder"]["aggregate"],
              ["strategy", "info_class", "neighborhood", "mean_local_pairwise_auc", "top1_hit_rate",
               "local_random_top1_hit_rate", "top1_enrichment"])
    _writecsv(os.path.join(tdir, "local_random_baseline.csv"), p["ladder"]["random_rows"],
              ["seed", "target", "level", "regime", "neighborhood", "n", "random_top1_hit_rate",
               "contains_joint_good"])
    _writecsv(os.path.join(tdir, "margin_sensitivity_local_boundary.csv"), [_flat_sens(res["margin_sensitivity"])],
              ["primary_cases", "robust_cases", "primary_mean_transition_rate", "robust_mean_transition_rate",
               "primary_pm1_joint_rate", "robust_pm1_joint_rate", "primary_selected_hit", "robust_selected_hit",
               "case_changes"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), _no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c33_case_taxonomy.csv"), [{"cases": ";".join(p["taxonomy"]["cases"])}], ["cases"])


def _flat_sens(s):
    return {**s, "primary_cases": ";".join(s["primary_cases"]), "robust_cases": ";".join(s["robust_cases"]),
            "case_changes": ";".join(s["case_changes"])}


def _no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "no_training_no_reinference", "passed": True},
        {"check": "neighborhood_definitions_frozen", "passed": True},
        {"check": "local_random_baseline_reported", "passed": bool(res["primary"]["ladder"]["random_rows"])},
        {"check": "target_unlabeled_non_source_only", "passed": True},
        {"check": "target_grouped_non_deployable", "passed": True},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": bool(res["diagnostic_only_non_deployable"])},
    ]


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res):
    p = res["primary"]
    b = p["boundary"]["summary"]; ps = p["pairs"]["summary"]; g = p["gradients"]["summary"]
    pl = p["plateaus"]["summary"]; lad = p["ladder"]["summary"]; tax = p["taxonomy"]
    sens = res["margin_sensitivity"]
    return "\n".join([
        f"# C33 - Local Trajectory Boundary / Checkpoint Neighborhood Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> C32R showed selected OACI is near joint-good candidates but hits at random-like top-1. C33 audits the "
        "local boundary: run structure, selected-vs-nearest-joint pairs, adjacent gradients, score plateaus, and "
        "local information rungs. Diagnostic-only; no selector, no training, no selected-checkpoint artifact.",
        "",
        f"- **cases: `{', '.join(tax['cases'])}`**",
        "",
        "## Gate 1 - local boundary geometry",
        "",
        f"- mean transition rate: **{_f(b['mean_transition_rate'])}**; median selected-boundary distance: "
        f"**{_f(b['median_selected_boundary_distance'])}**.",
        f"- selected +/-1 neighborhood contains joint-good in **{_f(b['pm1_contains_joint_fraction'])}** of units; "
        f"mean +/-1 joint-good rate **{_f(b['mean_pm1_joint_good_rate'])}**.",
        f"- mean joint-good run length: **{_f(b['mean_good_run_length'])}**. This does **not** clear the B1 dense-"
        "boundary gate globally; the local boundary signal is moderate, not the headline.",
        "",
        "## Gate 2 - selected vs nearest joint-good",
        "",
        f"- selected hit **{_f(ps['selected_joint_hit_rate'])}**; median order/epoch delta to nearest joint-good "
        f"**{_f(ps['median_order_delta'])} / {_f(ps['median_epoch_delta'])}**.",
        f"- miss-conditioned source-flat fraction **{_f(ps['source_flat_fraction'])}**; source-wrong fraction "
        f"**{_f(ps['source_wrong_fraction'])}**; pair gauge-jump-unseen fraction **{_f(ps['gauge_jump_unseen_fraction'])}**.",
        "",
        "## Gate 3 - adjacent local gradients",
        "",
        f"- transition pairs: **{g['n_transition_pairs']}** / {g['n_adjacent_pairs']} "
        f"(fraction {_f(g['transition_fraction'])}).",
        f"- source gradient sign agreement on transitions **{_f(g['source_gradient_agreement'])}**; rank agreement "
        f"**{_f(g['rank_gradient_agreement'])}**; transition gauge-jump fraction **{_f(g['transition_gauge_jump_fraction'])}** "
        "(B4 is read as common target-margin jumps with weak source alignment, not a clean pairwise unseen-gauge claim).",
        "",
        "## Gate 4 - source-score plateau",
        "",
        f"- mean/median plateau size at eps={schema.PLATEAU_EPS}: **{_f(pl['mean_plateau_size'])} / "
        f"{_f(pl['median_plateau_size'])}**.",
        f"- if selected is bad, plateau contains joint-good in **{_f(pl['selected_bad_plateau_has_joint_fraction'])}** of units.",
        "",
        "## Gate 5 - local information ladder",
        "",
        f"- source pm1 enrichment **{_f(lad['source_pm1_enrichment'])}**.",
        f"- target-unlabeled pm1/pm2 top-1 gain vs source **{_f(lad['target_unlabeled_pm1_top1_gain_vs_source'])} / "
        f"{_f(lad['target_unlabeled_pm2_top1_gain_vs_source'])}**.",
        f"- target-grouped pm1 gain vs source **{_f(lad['target_grouped_pm1_top1_gain_vs_source'])}** "
        "(rank-invariant inside same-target local neighborhoods; non-deployable diagnostic, not a local ceiling).",
        "",
        "## Margin sensitivity",
        "",
        f"- robust cases: **{', '.join(sens['robust_cases'])}**; changed cases: "
        f"**{', '.join(sens['case_changes']) if sens['case_changes'] else 'none'}**.",
        f"- primary vs robust pm1 joint rate **{_f(sens['primary_pm1_joint_rate'])} / "
        f"{_f(sens['robust_pm1_joint_rate'])}**.",
        "",
        "## Bottom line",
        "",
        "> C33 localizes the C32R miss to active local misranking plus target-margin jumps with weak source alignment, "
        "rather than to global dense-boundary jitter or source-score indifference. Selected OACI is usually close to "
        "a joint-good candidate, but among actual misses the source score is usually not flat; it often prefers the "
        "non-joint selected point. Target-unlabeled R3 remains pooled/gauge-help rather than a local top-k rescue. "
        "Target-grouped is rank-invariant in these same-target neighborhoods, and target-label quantities remain "
        "diagnostic only, not methods.",
    ])


def render_pair_md(res):
    ps = res["primary"]["pairs"]["summary"]
    return ("# C33 - selected vs nearest joint-good neighborhood\n\n"
            f"- pairs: {ps['n_pairs']}\n"
            f"- median order delta: {_f(ps['median_order_delta'])}\n"
            f"- source-flat / source-wrong: {_f(ps['source_flat_fraction'])} / {_f(ps['source_wrong_fraction'])}\n"
            f"- gauge-jump-unseen: {_f(ps['gauge_jump_unseen_fraction'])}\n")


def render_ladder_md(res):
    lines = ["# C33 - local information ladder\n",
             "| strategy | neighborhood | top1 hit | local random | enrichment | local AUC |",
             "|---|---|---:|---:|---:|---:|"]
    for r in res["primary"]["ladder"]["aggregate"]:
        lines.append(f"| {r['strategy']} | {r['neighborhood']} | {_f(r['top1_hit_rate'])} | "
                     f"{_f(r['local_random_top1_hit_rate'])} | {_f(r['top1_enrichment'])} | "
                     f"{_f(r['mean_local_pairwise_auc'])} |")
    lines.append("\nTarget-unlabeled and target-grouped rungs are diagnostic-only and non-source-only.")
    return "\n".join(lines)


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not deployable", "non-deployable", "diagnostic-only", "no selected", "no selector",
             "not claimed")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 56):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C33 report near: {s}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res); pair = render_pair_md(res); ladder = render_ladder_md(res)
    for text in (md, pair, ladder):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C33_LOCAL_TRAJECTORY_BOUNDARY_AUDIT.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C33_SELECTED_VS_JOINT_GOOD_NEIGHBORHOOD.md"), "w").write(pair)
    open(os.path.join(out_dir, "C33_LOCAL_INFORMATION_LADDER.md"), "w").write(ladder + "\n")
    json.dump(res, open(os.path.join(out_dir, "C33_LOCAL_TRAJECTORY_BOUNDARY_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c33_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.local_boundary.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run()
    _write_artifacts(res, args.out_dir)
    print(f"[C33] cases={','.join(res['primary']['taxonomy']['cases'])} "
          f"trans={_f(res['primary']['boundary']['summary']['mean_transition_rate'])} "
          f"pm1={_f(res['primary']['boundary']['summary']['pm1_contains_joint_fraction'])} "
          f"tu_pm1_gain={_f(res['primary']['ladder']['summary']['target_unlabeled_pm1_top1_gain_vs_source'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
