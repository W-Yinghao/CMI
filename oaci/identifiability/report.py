"""C17 report — assembles the source-signal identifiability audit (atlas + univariate + multivariate probe +
axis decomposition + class-boundary + case taxonomy) into Markdown + canonical JSON + CSV tables. The
multivariate probe is slow (permutation baseline); run this as a background/CPU job. Everything is labeled
diagnostic-only / non-deployable."""
from __future__ import annotations

import argparse
import csv
import os

from ..artifacts.canonical_json import canonical_json_bytes
from .axis_decomposition import axis_decomposition
from .class_boundary import class_boundary_identifiability
from .multivariate_probe import multivariate_probe
from .signal_atlas import SOURCE_SIGNALS, build_atlas, load_replay
from .target_labels import assert_diagnostic_only
from .taxonomy import case_taxonomy
from .univariate import univariate_identifiability


def _f(x, nd=3):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def _wcsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)


def build(replay_dir, loso_root) -> dict:
    rows = build_atlas(load_replay(replay_dir))
    assert_diagnostic_only(rows)
    univ = univariate_identifiability(rows)
    multi = multivariate_probe(rows)
    axis = axis_decomposition(univ)
    cb = class_boundary_identifiability(loso_root)
    tax = case_taxonomy(univ, multi, axis)
    # subject heterogeneity of identifiability: per target base rate + probe AUC
    by_t = {}
    for r in rows:
        b = by_t.setdefault(str(r["target"]), {"n": 0, "good": 0})
        b["n"] += 1; b["good"] += int(r["tgt__target_bacc_good"])
    subj = {t: {"n": v["n"], "target_good_rate": v["good"] / v["n"],
                "probe_auc": multi["per_target_auc"].get(t)} for t, v in sorted(by_t.items(), key=lambda kv: int(kv[0]))}
    return {"atlas_rows": rows, "univariate": univ, "multivariate": multi, "axis": axis,
            "class_boundary": cb, "subject_heterogeneity": subj, "taxonomy": tax}


def write_tables(res, outdir) -> int:
    rows, univ, multi, axis, cb = res["atlas_rows"], res["univariate"], res["multivariate"], res["axis"], res["class_boundary"]
    src_cols = [f"src__{s}" for s in SOURCE_SIGNALS]
    _wcsv(os.path.join(outdir, "source_signal_atlas.csv"),
          ["seed", "target", "level", "model_hash"] + src_cols + ["tgt__target_bacc_delta", "tgt__target_nll_delta", "tgt__target_bacc_good", "tgt__target_oracle_rank"],
          [[r["seed"], r["target"], r["level"], r["model_hash"][:12]] + [r[c] for c in src_cols]
           + [r["tgt__target_bacc_delta"], r["tgt__target_nll_delta"], r["tgt__target_bacc_good"], r["tgt__target_oracle_rank"]] for r in rows])
    _wcsv(os.path.join(outdir, "target_oracle_checkpoint_labels.csv"),
          ["seed", "target", "level", "model_hash", "target_bacc_good", "target_joint_good", "target_oracle_rank", "diagnostic_only_non_deployable"],
          [[r["seed"], r["target"], r["level"], r["model_hash"][:12], r["tgt__target_bacc_good"], r["tgt__target_joint_good"], r["tgt__target_oracle_rank"], True] for r in rows])
    _wcsv(os.path.join(outdir, "univariate_signal_identifiability.csv"),
          ["signal", "axis", "rho_bacc", "rho_nll", "perm_p_bacc", "strong_accuracy", "weak_accuracy", "identifies_nll"],
          [[s, v["axis"], v["mean_within_fold_spearman_bacc"], v["mean_within_fold_spearman_nll"], v["perm_p_bacc"],
            v["strong_accuracy_signal"], v["weak_accuracy_signal"], v["identifies_target_nll"]] for s, v in univ["per_signal"].items()])
    _wcsv(os.path.join(outdir, "topk_enrichment_by_signal.csv"), ["signal", "topk_good_rate", "base_rate", "k"],
          [[s, v["topk_enrichment"]["topk_good_rate"], v["topk_enrichment"]["base_rate"], v["topk_enrichment"]["k"]] for s, v in univ["per_signal"].items()])
    _wcsv(os.path.join(outdir, "source_target_rank_correlation.csv"), ["signal", "rho_target_bacc", "rho_target_nll"],
          [[s, v["mean_within_fold_spearman_bacc"], v["mean_within_fold_spearman_nll"]] for s, v in univ["per_signal"].items()])
    _wcsv(os.path.join(outdir, "multivariate_probe_loto.csv"), ["metric", "value"],
          [["loto_auc", multi["loto_auc"]], ["loso_auc", multi["loso_auc"]], ["base_rate", multi["base_rate"]],
           ["n_used", multi["n_used"]], ["non_deployable", multi["non_deployable"]]]
          + [[f"per_target_auc_{t}", a] for t, a in multi["per_target_auc"].items()])
    _wcsv(os.path.join(outdir, "multivariate_probe_permutation_baseline.csv"), ["metric", "value"],
          [["permutation_mean_auc", multi["permutation_mean_auc"]], ["permutation_p", multi["permutation_p"]],
           ["beats_permutation", multi["beats_permutation"]]])
    _wcsv(os.path.join(outdir, "calibration_accuracy_axis_decomposition.csv"),
          ["axis", "n_signals", "mean_abs_rho_target_bacc", "mean_abs_rho_target_nll"],
          [[ax, v["n"], v["mean_abs_rho_target_bacc"], v["mean_abs_rho_target_nll"]] for ax, v in axis["by_axis"].items()])
    _wcsv(os.path.join(outdir, "source_visible_vs_target_visible_axes.csv"), ["quantity", "value"],
          [["calibration_axis_target_nll_visibility", axis["calibration_axis_target_nll_visibility"]],
           ["accuracy_axis_target_bacc_visibility", axis["accuracy_axis_target_bacc_visibility"]],
           ["risk_axis_target_bacc_visibility", axis["risk_axis_target_bacc_visibility"]],
           ["source_signals_see_calibration_more_than_accuracy", axis["source_signals_see_calibration_more_than_accuracy"]]])
    _wcsv(os.path.join(outdir, "class_boundary_rotation_identifiability.csv"),
          ["class", "n", "mean_src_recall_delta", "mean_tgt_recall_delta", "src_tgt_corr"],
          [[c, v["n"], v["mean_src_recall_delta"], v["mean_tgt_recall_delta"], v["src_tgt_corr"]] for c, v in cb["per_class"].items()])
    _wcsv(os.path.join(outdir, "subject_heterogeneity_identifiability.csv"), ["target", "n", "target_good_rate", "probe_auc"],
          [[t, v["n"], v["target_good_rate"], v["probe_auc"]] for t, v in res["subject_heterogeneity"].items()])
    _wcsv(os.path.join(outdir, "c17_case_taxonomy.csv"), ["quantity", "value"],
          [["case_label", res["taxonomy"]["case_label"]], ["univariate_verdict", univ["univariate_verdict"]],
           ["multivariate_loto_auc", multi["loto_auc"]], ["multivariate_beats_permutation", multi["beats_permutation"]],
           ["source_signals_calibration_biased", axis["source_signals_see_calibration_more_than_accuracy"]],
           ["next_science", res["taxonomy"]["next_science"]]])
    return 12


def render_md(res) -> str:
    univ, multi, axis, cb, tax = res["univariate"], res["multivariate"], res["axis"], res["class_boundary"], res["taxonomy"]
    L = ["# C17 — Source-signal identifiability audit", "",
         "> Diagnostic study: are the target-accuracy-good OACI checkpoints (C16) identifiable from SOURCE-ONLY "
         "checkpoint observables? Target labels are used POST HOC only (diagnostic_only_non_deployable); no "
         "deployable selector is produced.", "",
         f"- **CASE: `{tax['case_label']}`** — {tax['interpretation']}",
         f"- next science: {tax['next_science']}", "",
         "## Univariate identifiability (within-fold-level Spearman; permutation p)", "",
         f"- verdict `{univ['univariate_verdict']}`; strong accuracy signals {univ['n_strong_accuracy_signals']}, "
         f"weak {univ['n_weak_accuracy_signals']}, NLL-identifying {univ['n_signals_identify_nll']}",
         f"- C10's oracle signal (source_audit_worst_bacc) within-fold ρ(target bAcc) = "
         f"{_f(univ['oracle_signal_spearman_bacc'])}; best |ρ| any signal = {_f(univ['max_abs_accuracy_spearman'])}; "
         f"accuracy signal families = {univ['accuracy_signal_families']}", "",
         "| signal | axis | ρ(tgt bAcc) | ρ(tgt NLL) | perm p | strong | weak |", "|---|---|---:|---:|---:|:--:|:--:|"]
    for s, v in univ["per_signal"].items():
        L.append(f"| {s} | {v['axis']} | {_f(v['mean_within_fold_spearman_bacc'])} | {_f(v['mean_within_fold_spearman_nll'])} | "
                 f"{_f(v['perm_p_bacc'])} | {v['strong_accuracy_signal']} | {v['weak_accuracy_signal']} |")
    L += ["", "## Multivariate competence probe (DIAGNOSTIC-ONLY, leave-one-target-out)", "",
          f"- LOTO AUC **{_f(multi['loto_auc'])}** vs permutation mean {_f(multi['permutation_mean_auc'])} "
          f"(p {_f(multi['permutation_p'])}); LOSO AUC {_f(multi['loso_auc'])}; base rate {_f(multi['base_rate'])}",
          f"- **beats permutation: {multi['beats_permutation']}** · non_deployable = {multi['non_deployable']}", "",
          "## Calibration-vs-accuracy axis decomposition", "",
          f"- calibration-axis→target-NLL visibility {_f(axis['calibration_axis_target_nll_visibility'])} vs "
          f"accuracy-axis→target-bAcc visibility {_f(axis['accuracy_axis_target_bacc_visibility'])} → "
          f"**source signals calibration-biased: {axis['source_signals_see_calibration_more_than_accuracy']}**", "",
          "## Class-boundary rotation identifiability (selected checkpoints)", "",
          f"- source↔target per-class recall-delta correlation {_f(cb['source_target_recall_delta_corr'])} → "
          f"class-boundary source-identifiable: {cb['class_boundary_source_identifiable']} ({cb['scope']})", "",
          f"> {tax['interpretation']}"]
    return "\n".join(L)


def render_taxonomy_md(res) -> str:
    tax = res["taxonomy"]
    L = ["# C17 — Identifiability case taxonomy", "", f"- **CASE: `{tax['case_label']}`**", "",
         "## Inputs", ""]
    for k, v in tax["inputs"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## Interpretation", f"> {tax['interpretation']}", "", "## Next science", f"> {tax['next_science']}"]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.identifiability.report")
    ap.add_argument("--replay-dir", default="/projects/EEG-foundation-model/yinghao/oaci-c10-replay")
    ap.add_argument("--loso-root", default="/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012")
    ap.add_argument("--reports-dir", default="oaci/reports")
    ap.add_argument("--tables-dir", default="oaci/reports/c17_tables")
    args = ap.parse_args(argv)
    res = build(args.replay_dir, args.loso_root)
    os.makedirs(args.reports_dir, exist_ok=True)
    write_tables(res, args.tables_dir)
    serial = {k: v for k, v in res.items() if k != "atlas_rows"}      # atlas is in the CSV, keep json compact
    serial["n_atlas_rows"] = len(res["atlas_rows"])
    with open(os.path.join(args.reports_dir, "C17_SOURCE_SIGNAL_IDENTIFIABILITY.json"), "wb") as f:
        f.write(canonical_json_bytes(serial))
    with open(os.path.join(args.reports_dir, "C17_SOURCE_SIGNAL_IDENTIFIABILITY.md"), "w") as f:
        f.write(render_md(res))
    with open(os.path.join(args.reports_dir, "C17_IDENTIFIABILITY_CASE_TAXONOMY.md"), "w") as f:
        f.write(render_taxonomy_md(res))
    print(f"C17 case={res['taxonomy']['case_label']}; univariate={res['univariate']['univariate_verdict']}; "
          f"multivariate LOTO AUC={res['multivariate']['loto_auc']:.3f} beats_perm={res['multivariate']['beats_permutation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
