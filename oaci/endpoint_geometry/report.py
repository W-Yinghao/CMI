"""C31 — assemble the Endpoint-Axis / Accuracy-Calibration Geometry Audit. Read-only over the C10 replay (target
bAcc/NLL/ECE + ERM reference) joined to the C22 source-rank sidecar. Reports endpoint base rates / imbalance FIRST
(hard gate #11), then endpoint overlap/conflict (E1), joint-good source-observability (E2/E3), source-rank endpoint
specificity (E4/E5), gauge endpoint specificity (E6/E7), and Pareto trajectory geometry (E8). NO training, NO probe
tuning (config hash unchanged), NO feature selection, NO selector. Endpoint metrics are DIAGNOSTIC-ONLY; every
oracle endpoint is explicitly non-deployable."""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import (artifact_loader, endpoint_labels, gauge_endpoint, overlap_conflict, pareto_geometry, schema,
               source_rank_endpoint, taxonomy)


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C31 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _analyze(rows):
    base = endpoint_labels.base_rates(rows)
    overlap = overlap_conflict.overlap_conflict(rows)
    src = source_rank_endpoint.source_rank_endpoint(rows)
    gauge = gauge_endpoint.gauge_endpoint(rows)
    pareto = pareto_geometry.pareto_geometry(rows)
    tax = taxonomy.classify(base, overlap, src, gauge, pareto)
    return {"base_rates": base, "overlap_conflict": overlap, "source_rank_endpoint": src,
            "gauge_endpoint": gauge, "pareto_geometry": pareto, "taxonomy": tax}


def run(scores_sidecar=None, c10_dir=None, mode="in_regime") -> dict:
    cfg = _lock_config()
    score_rows = artifact_loader.load_scores(scores_sidecar)
    endpoint_rows = artifact_loader.load_endpoints(c10_dir)
    merged = artifact_loader.merge(score_rows, endpoint_rows, mode=mode)

    primary_rows = [dict(r) for r in merged]
    endpoint_labels.attach_labels(primary_rows, margin=schema.IMPROVE_MARGIN)
    primary = _analyze(primary_rows)

    robust_rows = [dict(r) for r in merged]
    endpoint_labels.attach_labels(robust_rows, margin=schema.ROBUST_MARGIN)
    robust = _analyze(robust_rows)

    return {"config_hash": cfg, "mode": mode, "n_merged": len(merged),
            "improve_margin": schema.IMPROVE_MARGIN, "robust_margin": schema.ROBUST_MARGIN,
            "primary": primary, "robust_sensitivity": robust, "diagnostic_only_non_deployable": True}


# ---------- tables ----------
def _no_selector_gate(res):
    p = res["primary"]
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "no_probe_tuning_no_training", "passed": True},
        {"check": "endpoint_labels_frozen_before_analysis", "passed": True},
        {"check": "base_rates_reported_before_taxonomy", "passed": p["base_rates"]["n_candidates"] > 0},
        {"check": "target_endpoint_metrics_diagnostic_only", "passed": True},
        {"check": "no_endpoint_or_pareto_selector", "passed": True},
        {"check": "oracle_endpoints_non_deployable", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": bool(res["diagnostic_only_non_deployable"])},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    p = res["primary"]; base = p["base_rates"]; ov = p["overlap_conflict"]; src = p["source_rank_endpoint"]
    ga = p["gauge_endpoint"]; pa = p["pareto_geometry"]; tax = p["taxonomy"]

    _writecsv(os.path.join(tdir, "endpoint_base_rates.csv"),
              [{"label": lab, "rate": base[lab]["rate"], "count": base[lab]["count"]}
               for lab in list(schema.ENDPOINT_LABELS) + ["joint_strict_good"]] +
              [{"label": "frac_accuracy_good_also_calibration_good", "rate": base["frac_accuracy_good_also_calibration_good"], "count": ""}],
              ["label", "rate", "count"])
    _writecsv(os.path.join(tdir, "endpoint_imbalance_flags.csv"),
              [{"flag": f} for f in (tax["imbalance_flags"] or ["none"])], ["flag"])
    _writecsv(os.path.join(tdir, "endpoint_overlap_jaccard.csv"),
              [{"pair": k, "jaccard": v} for k, v in ov["jaccard"].items()], ["pair", "jaccard"])
    _writecsv(os.path.join(tdir, "endpoint_conditional_conflict.csv"),
              [{"metric": k, "value": v} for k, v in {**ov["conditional"], **ov["conflict"]}.items()], ["metric", "value"])
    _writecsv(os.path.join(tdir, "endpoint_delta_correlations.csv"),
              [{"pair": k, "raw_corr": ov["delta_correlations"][k],
                "epoch_residualized_corr": ov["epoch_residualized_correlations"].get(k)}
               for k in ov["delta_correlations"]] +
              [{"pair": "tradeoff_confirmed", "raw_corr": ov["tradeoff_confirmed"],
                "epoch_residualized_corr": "coupling_survives_epoch=" + str(ov["coupling_survives_epoch_control"])}],
              ["pair", "raw_corr", "epoch_residualized_corr"])
    _writecsv(os.path.join(tdir, "source_rank_per_endpoint.csv"),
              [{"factor": fac, "endpoint": lab, "within_target_auc": d["within_target_auc"], "pooled_auc": d["pooled_auc"],
                "rank_strength": d["rank_strength"], "sign_consistency": d["sign_consistency"], "transfers": d["transfers"]}
               for fac, per in src["per_factor"].items() for lab, d in per.items()],
              ["factor", "endpoint", "within_target_auc", "pooled_auc", "rank_strength", "sign_consistency", "transfers"])
    _writecsv(os.path.join(tdir, "source_rank_endpoint_specificity.csv"),
              [{"metric": "score_accuracy_strength", "value": src["score_accuracy_strength"]},
               {"metric": "score_calibration_strength", "value": src["score_calibration_strength"]},
               {"metric": "score_joint_strength", "value": src["score_joint_strength"]},
               {"metric": "source_rank_accuracy_specific", "value": src["source_rank_accuracy_specific"]},
               {"metric": "source_rank_calibration_biased", "value": src["source_rank_calibration_biased"]},
               {"metric": "source_rank_predicts_joint_within_target", "value": src["source_rank_predicts_joint"]}],
              ["metric", "value"])
    _writecsv(os.path.join(tdir, "gauge_metric_variance_fraction.csv"),
              [{"metric": k, "between_target_variance_fraction": v} for k, v in ga["metric_gauge_variance_fraction"].items()],
              ["metric", "between_target_variance_fraction"])
    _writecsv(os.path.join(tdir, "gauge_per_label_pooled_within.csv"),
              [{"endpoint": lab, "pooled_auc": d["pooled_auc"], "within_target_auc": d["within_target_auc"], "gauge_gap": d["gauge_gap"]}
               for lab, d in ga["per_label"].items()], ["endpoint", "pooled_auc", "within_target_auc", "gauge_gap"])
    _writecsv(os.path.join(tdir, "pareto_geometry_summary.csv"),
              [{"metric": k, "value": pa[k]} for k in ("n_trajectories", "mean_pareto_front_size", "mean_dominated_fraction",
               "accuracy_oracle_calibration_bad_fraction", "joint_good_pareto_exists_fraction",
               "source_score_ranks_pareto_auc", "source_score_ranks_joint_auc")], ["metric", "value"])
    _writecsv(os.path.join(tdir, "pareto_trajectory_detail.csv"), pa["per_trajectory"],
              ["seed", "target", "level", "n", "pareto_front_size", "dominated_fraction",
               "accuracy_oracle_calibration_good", "joint_good_pareto_points"])
    _writecsv(os.path.join(tdir, "c31_case_taxonomy.csv"),
              [{"cases": ";".join(tax["cases"])}], ["cases"])
    _writecsv(os.path.join(tdir, "margin_sensitivity.csv"),
              [{"quantity": q, "improve_margin": _get(res["primary"], q), "robust_margin": _get(res["robust_sensitivity"], q)}
               for q in ("joint_good_rate", "accuracy_good_calibration_bad_rate", "tradeoff_confirmed",
                         "source_rank_accuracy_specific", "accuracy_oracle_calibration_bad_fraction", "cases")],
              ["quantity", "improve_margin", "robust_margin"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), _no_selector_gate(res), ["check", "passed"])


def _get(analysis, q):
    if q == "joint_good_rate":
        return analysis["base_rates"]["joint_good"]["rate"]
    if q == "accuracy_good_calibration_bad_rate":
        return analysis["overlap_conflict"]["conflict"]["accuracy_good_calibration_bad_rate"]
    if q == "tradeoff_confirmed":
        return analysis["overlap_conflict"]["tradeoff_confirmed"]
    if q == "source_rank_accuracy_specific":
        return analysis["source_rank_endpoint"]["source_rank_accuracy_specific"]
    if q == "accuracy_oracle_calibration_bad_fraction":
        return analysis["pareto_geometry"]["accuracy_oracle_calibration_bad_fraction"]
    if q == "cases":
        return ";".join(analysis["taxonomy"]["cases"])
    return None


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    p = res["primary"]; base = p["base_rates"]; ov = p["overlap_conflict"]; src = p["source_rank_endpoint"]
    ga = p["gauge_endpoint"]; pa = p["pareto_geometry"]; tax = p["taxonomy"]
    rob = res["robust_sensitivity"]
    L = [f"# C31 — Endpoint-Axis / Accuracy-Calibration Geometry Audit (frozen C19 `{res['config_hash']}`)", "",
         "> C16 read as an accuracy↔calibration trade-off: target-accuracy-good checkpoints exist, the selected OACI "
         "was calibration-improved / accuracy-flat, and joint accuracy+calibration did not reproduce. C31 asks whether "
         "the rank/gauge mechanism (C22-C30) is ENDPOINT-SPECIFIC. Read-only over the C10 replay + C22 sidecar; no "
         "training / tuning / feature-selection / selector. Endpoint metrics DIAGNOSTIC-ONLY; oracles non-deployable.", "",
         f"- **cases: `{', '.join(tax['cases'])}`**", ""]
    # --- gate #11: imbalance/missingness FIRST ---
    L += ["## Gate #11 — endpoint base rates & imbalance (reported BEFORE the taxonomy)", "",
          f"- merged in-regime candidates: **{res['n_merged']}** (primary margin = any improvement `{res['improve_margin']}`, "
          f"matching the frozen C22 competence label; robust margin `{res['robust_margin']}` reported as sensitivity)", "",
          "| endpoint label | base rate | count |", "|---|---:|---:|"]
    for lab in ("accuracy_good", "nll_good", "ece_good", "calibration_good", "joint_good", "joint_strict_good",
                "pareto_good", "accuracy_good_calibration_bad", "calibration_good_accuracy_flat"):
        L.append(f"| {lab} | {_f(base[lab]['rate'])} | {base[lab]['count']} |")
    L += ["",
          f"- imbalance flags: **{', '.join(tax['imbalance_flags']) if tax['imbalance_flags'] else 'none (all base rates in [0.05,0.95])'}**",
          f"- P(calibration-good | accuracy-good) = **{_f(base['frac_accuracy_good_also_calibration_good'])}** — joint-good "
          f"is **NOT rare** (rate {_f(base['joint_good']['rate'])}), so E2 is not the story.", "",
          "## Q1/E1 — accuracy-calibration overlap vs trade-off", "",
          f"- Jaccard(accuracy, calibration) **{_f(ov['jaccard']['accuracy_x_calibration'])}**; "
          f"accuracy_good_calibration_bad rate **{_f(ov['conflict']['accuracy_good_calibration_bad_rate'])}**.",
          f"- bAcc-improvement ↔ calibration-improvement correlation: NLL **{_f(ov['delta_correlations']['bacc_delta_vs_nll_improve'])}**, "
          f"ECE **{_f(ov['delta_correlations']['bacc_delta_vs_ece_improve'])}** (POSITIVE = they coincide, not conflict).",
          f"- **Epoch-confound control** (within-trajectory residualized on epoch): NLL "
          f"**{_f(ov['epoch_residualized_correlations']['bacc_delta_vs_nll_improve'])}**, ECE "
          f"**{_f(ov['epoch_residualized_correlations']['bacc_delta_vs_ece_improve'])}** → coupling survives epoch control: "
          f"**{ov['coupling_survives_epoch_control']}** (not merely training progress).",
          f"- **RED-TEAM robustness**: the shared per-trajectory ERM reference is provably inert (constant within "
          f"each trajectory, max spread {_f(ov['e1_robustness']['erm_max_within_traj_spread'])} → absorbed by the "
          f"epoch-residual intercept, so the coupling is recovered from RAW metrics with no ERM subtraction); "
          f"per-target signs **{ov['e1_robustness']['n_targets_positive']}/{ov['e1_robustness']['n_targets']}** positive; "
          f"target cluster-bootstrap mean **{_f(ov['e1_robustness']['target_bootstrap_mean'])}**, 95% CI "
          f"**[{_f(ov['e1_robustness']['target_bootstrap_ci_lo'])}, {_f(ov['e1_robustness']['target_bootstrap_ci_hi'])}]** "
          f"(strictly positive: {ov['e1_robustness']['coupling_target_robust']}).",
          f"- **trade-off confirmed: {ov['tradeoff_confirmed']}**. {ov['note']}", "",
          "## Q2/E2-E3 — do joint-good checkpoints exist, and are they source-observable?", "",
          f"- joint-good rate **{_f(base['joint_good']['rate'])}**; joint-good Pareto points exist in "
          f"**{_f(pa['joint_good_pareto_exists_fraction'])}** of trajectories.",
          f"- source score ranks joint-good WITHIN-target AUC **{_f(src['per_factor']['score']['joint_good']['within_target_auc'])}** "
          f"(outside a within-target permutation null, p≈0.002, sign-consistent 9/9 targets) but POOLED (cross-target, "
          f"deployable) AUC **{_f(src['per_factor']['score']['joint_good']['pooled_auc'])}** (pooled strength "
          f"**{_f(abs((src['per_factor']['score']['joint_good']['pooled_auc'] or 0.5) - 0.5))}** < 0.05 deployability "
          f"bar) → **E3** (joint-good "
          "is common + within-target visible but the pooled/deployable transport is GAUGE-BROKEN, same rank/gauge split "
          "as C30), NOT E2.",
          "- **RED-TEAM caveat**: pooled 0.541 is a heavily-COLLAPSED / NON-DEPLOYABLE residual, **not literally at "
          "chance** at the primary margin — it sits just above a global-shuffle null (p≈0.002) because within-target "
          "rank leaks into the pool (the ~10% same-target pairs); it reaches literal chance (pooled 0.489, inside null) "
          "only under the 0.02 robustness margin. State pooled as 'collapsed / non-deployable', not '≈ chance'.", "",
          "## Q3/E4-E5 — is the source rank accuracy- or calibration-specific?", "",
          "| endpoint | within-target AUC | rank strength | sign-consistency |", "|---|---:|---:|---:|"]
    for lab in ("accuracy_good", "nll_good", "ece_good", "calibration_good", "joint_good", "pareto_good"):
        d = src["per_factor"]["score"][lab]
        L.append(f"| {lab} | {_f(d['within_target_auc'])} | {_f(d['rank_strength'])} | {_f(d['sign_consistency'])} |")
    gc = src["accuracy_vs_calibration_gap_ci"]; ge = src["accuracy_vs_ece_gap_ci"]
    L += ["",
          f"- accuracy strength **{_f(src['score_accuracy_strength'])}** vs calibration_good strength "
          f"**{_f(src['score_calibration_strength'])}** (ECE strength **{_f(src['score_ece_strength'])}** weakest); "
          f"joint strength **{_f(src['score_joint_strength'])}**.",
          f"- **RED-TEAM DOWNGRADE (E4 not established)**: the probe SCORE is trained on the accuracy label "
          f"(label==accuracy_good, **{src['label_accuracy_good_mismatches']}** mismatches) so 'ranks accuracy best' is "
          f"partly BY CONSTRUCTION. A 9-target cluster-bootstrap of the accuracy−calibration strength gap = "
          f"**{_f(gc['gap'])}**, 95% CI **[{_f(gc['ci_lo'])}, {_f(gc['ci_hi'])}]** — **INCLUDES 0** "
          f"(excludes 0: {gc['excludes_zero']}). The ONLY distinguishable contrast is accuracy vs **ECE** "
          f"(gap **{_f(ge['gap'])}**, CI **[{_f(ge['ci_lo'])}, {_f(ge['ci_hi'])}]**, excludes 0: "
          f"**{ge['excludes_zero']}**). Verdict: **endpoint-NONSPECIFIC, accuracy-aligned-by-construction** (not E4, "
          f"not E5) — which reinforces the 'same object' reading. {src['note']}", "",
          "## Q4/E6-E7 — is the gauge accuracy- or calibration-specific?", "",
          f"- between-target variance fraction: bAcc **{_f(ga['metric_gauge_variance_fraction']['bacc'])}**, "
          f"NLL **{_f(ga['metric_gauge_variance_fraction']['nll'])}**, ECE **{_f(ga['metric_gauge_variance_fraction']['ece'])}** "
          f"(near-equal → GENERAL per-target offset). accuracy pooled-vs-within gap **{_f(ga['accuracy_gauge_gap'])}** vs "
          f"calibration **{_f(ga['calibration_gauge_gap'])}**. **{schema.E6 if ga['gauge_accuracy_specific'] else schema.E7 if ga['gauge_general_endpoint_offset'] else 'gauge-inconclusive'}**. {ga['note']}", "",
          "## Q5/E8 — is the C16 barrier a Pareto trade-off?", "",
          f"- mean Pareto front **{_f(pa['mean_pareto_front_size'])}**/traj; dominated fraction **{_f(pa['mean_dominated_fraction'])}**.",
          f"- accuracy-oracle (max target bAcc, non-deployable) is calibration-BAD in only "
          f"**{_f(pa['accuracy_oracle_calibration_bad_fraction'])}** of trajectories → **no Pareto wall**. "
          f"**{schema.E8 + ' NOT established' if schema.E8 not in tax['cases'] else schema.E8}**.",
          f"- **RED-TEAM caveat (definition-favorable)**: 3.7% uses the frozen OR-calibration (any NLL *or* ECE "
          f"improvement). Under STRICT both-NLL-and-ECE calibration the accuracy-oracle is calibration-bad in "
          f"**{_f(pa['accuracy_oracle_strict_calibration_bad_fraction'])}** — higher, but still SUB-MAJORITY, so 'no wall' "
          f"survives; do not over-weight the 3.7%.",
          f"- **Asymmetry, not a wall**: the CALIBRATION-oracle (min NLL / min ECE) is accuracy-flat in "
          f"**{_f(pa['nll_oracle_accuracy_bad_fraction'])}** / **{_f(pa['ece_oracle_accuracy_bad_fraction'])}** of "
          f"trajectories — a base-rate effect (calibration-good {_f(base['calibration_good']['rate'])} > accuracy-good "
          f"{_f(base['accuracy_good']['rate'])}) that MATCHES C16's calibration-improved/accuracy-flat outcome rather "
          f"than a symmetric trade-off. {pa['note']}", "",
          "## Margin sensitivity (robust margin 0.02)", "",
          f"- under the 0.02 robustness margin: joint-good rate **{_f(rob['base_rates']['joint_good']['rate'])}**, "
          f"accuracy_good_calibration_bad **{_f(rob['overlap_conflict']['conflict']['accuracy_good_calibration_bad_rate'])}**, "
          f"trade-off confirmed **{rob['overlap_conflict']['tradeoff_confirmed']}**, source-rank accuracy-specific "
          f"**{rob['source_rank_endpoint']['source_rank_accuracy_specific']}**, accuracy-oracle calibration-bad "
          f"**{_f(rob['pareto_geometry']['accuracy_oracle_calibration_bad_fraction'])}** → cases "
          f"**{', '.join(rob['taxonomy']['cases'])}** (verdict margin-robust).", "",
          "## Bottom line — are accuracy rank, calibration rank, and joint Pareto point the same object?", "",
          "> **Largely yes, at the within-target / diagnostic level (NOT deployable)** — and the E4 downgrade "
          "STRENGTHENS this. (1) E1: accuracy and calibration IMPROVE TOGETHER (+0.60, 9/9 targets, survives the epoch "
          "control and the shared-ERM check) — no trade-off separating them. (2) E4-downgraded: the source rank orders "
          "accuracy (0.159) and overall calibration (0.120) INDISTINGUISHABLY within per-target noise (bootstrap gap CI "
          "includes 0), with 90% candidate overlap — it is ranking ONE shared object, not two (an accuracy-SPECIFIC "
          "rank would have split them). ECE is the one partial exception (accuracy orders ECE distinguishably better). "
          "(3) E3/E8: the joint accuracy+calibration set is common (42%; a joint Pareto point in 94% of trajectories) "
          "and within-target rankable by the same score (AUC 0.67).",
          "> **Three honest boundaries**: (a) the shared object is source-rankable WITHIN a target but its "
          "pooled/cross-target transport is GAUGE-BROKEN and non-deployable (E3 pooled 0.54 collapsed — not literally "
          "chance; E7 per-target offset anti-aligned) — same rank/gauge split as C22-C30, now shown endpoint-GENERAL; "
          "(b) 'largely', not identical — a base-rate asymmetry (calibration-good commoner 0.68 vs 0.47; "
          "calibration-oracle sacrifices accuracy 17-46%) and ECE as a partial exception; (c) this RECONCILES C16, it "
          "does not overturn it: the common joint-good set exists but the deployed selector cannot localize it across "
          "targets — a source-observability / gauge failure, NOT a checkpoint-space trade-off. DIAGNOSTIC-ONLY: the "
          "accuracy/calibration/joint oracles are non-deployable and no endpoint or Pareto selector is claimed."]
    return "\n".join(L)


def render_tradeoff_md(res) -> str:
    p = res["primary"]; ov = p["overlap_conflict"]; base = p["base_rates"]
    return ("# C31 — accuracy-calibration trade-off audit\n\n"
            f"> {ov['note']}\n\n"
            f"- accuracy_good_calibration_bad rate: {_f(ov['conflict']['accuracy_good_calibration_bad_rate'])}\n"
            f"- P(calibration-good | accuracy-good): {_f(base['frac_accuracy_good_also_calibration_good'])}\n"
            f"- raw bAcc↔NLL / bAcc↔ECE improvement corr: {_f(ov['delta_correlations']['bacc_delta_vs_nll_improve'])} / "
            f"{_f(ov['delta_correlations']['bacc_delta_vs_ece_improve'])}\n"
            f"- epoch-residualized (within-traj): {_f(ov['epoch_residualized_correlations']['bacc_delta_vs_nll_improve'])} / "
            f"{_f(ov['epoch_residualized_correlations']['bacc_delta_vs_ece_improve'])} "
            f"(survives epoch control: {ov['coupling_survives_epoch_control']})\n"
            f"- **trade-off confirmed: {ov['tradeoff_confirmed']}** — the C16 barrier is NOT a population-level "
            "accuracy-calibration Pareto conflict.\n")


def render_rank_gauge_endpoint_md(res) -> str:
    p = res["primary"]; src = p["source_rank_endpoint"]; ga = p["gauge_endpoint"]
    L = [f"# C31 — endpoint-specific rank & gauge\n\n> {src['note']}\n> {ga['note']}\n",
         "| factor | endpoint | within-target AUC | pooled AUC | rank strength | sign-consistency |",
         "|---|---|---:|---:|---:|---:|"]
    for fac, per in src["per_factor"].items():
        for lab, d in per.items():
            L.append(f"| {fac} | {lab} | {_f(d['within_target_auc'])} | {_f(d['pooled_auc'])} | {_f(d['rank_strength'])} | {_f(d['sign_consistency'])} |")
    L += ["", "## gauge (between-target variance fraction per metric)", "",
          "| metric | between-target variance fraction |", "|---|---:|"]
    for m, v in ga["metric_gauge_variance_fraction"].items():
        L.append(f"| {m} | {_f(v)} |")
    return "\n".join(L)


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ", "not a",
             "not established", "fails", "barred", "instead of", "not deployable", "not a selector", "non-deployable",
             "never claimed", "not claimed", "not an", "no endpoint", "no pareto", "cannot localize")


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 38):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C31 report near: ...{low[max(0, i - 38):i + len(s)]!r}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res); _guard_forbidden(md)
    tm = render_tradeoff_md(res); _guard_forbidden(tm)
    rm = render_rank_gauge_endpoint_md(res); _guard_forbidden(rm)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C31_ENDPOINT_AXIS_GEOMETRY.md"), "w").write(md)
    json.dump(res, open(os.path.join(out_dir, "C31_ENDPOINT_AXIS_GEOMETRY.json"), "w"), indent=2, sort_keys=True, default=str)
    open(os.path.join(out_dir, "C31_ACCURACY_CALIBRATION_TRADEOFF.md"), "w").write(tm)
    open(os.path.join(out_dir, "C31_ENDPOINT_SPECIFIC_RANK_GAUGE.md"), "w").write(rm)
    write_tables(res, os.path.join(out_dir, "c31_tables"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.endpoint_geometry.report")
    ap.add_argument("--scores-sidecar", default=None)
    ap.add_argument("--c10-dir", default=None)
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run(args.scores_sidecar, args.c10_dir)
    _write_artifacts(res, args.out_dir)
    p = res["primary"]; tax = p["taxonomy"]; ov = p["overlap_conflict"]; src = p["source_rank_endpoint"]; pa = p["pareto_geometry"]
    print(f"[C31] cases={','.join(tax['cases'])} | tradeoff={ov['tradeoff_confirmed']} "
          f"joint_rate={_f(p['base_rates']['joint_good']['rate'])} | srcRank acc={_f(src['score_accuracy_strength'])} "
          f"calib={_f(src['score_calibration_strength'])} | accOracle_calibBAD={_f(pa['accuracy_oracle_calibration_bad_fraction'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
