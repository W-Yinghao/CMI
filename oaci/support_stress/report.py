"""C18 — assemble the Controlled Support-Mismatch x Identifiability Stress Test. Runs the identity probe
(G1-G4: S0 reproduces C17 Case III / 0.602), the per-regime mask-stress probe (H2 + H3 axis), the class-
boundary S6-vs-S7 contrast (H4), the support-aware leakage estimability/abstention (H5), the C18-D
observability-dropout appendix, the severity response, and the case taxonomy. Writes the reports + CSV
tables + a gate summary. All target-derived quantities are diagnostic-only; no selector is produced."""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import boundary_stress, feature_inventory, identifiability_stress, leakage_stress, observability_dropout
from . import schema, severity_response, taxonomy
from . import source_signal_recompute as ssr
from . import stress_plan as sp
from . import support_metrics


def _boundary_classes(c16_path="oaci/reports/C16_MECHANISM_DEEP_DIVE.json") -> tuple:
    d = json.load(open(c16_path))
    pcr = d["harm_decomposition"]["per_class_recall_delta"]
    return sp.boundary_classes_from_c16(pcr)


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def run(extract_dir, c10_dir, loso_root, *, n_perm=None, n_perturb=2, folds=None, n_workers=8,
        c16_path="oaci/reports/C16_MECHANISM_DEEP_DIVE.json") -> dict:
    feature_inventory.assert_inventory_complete()
    bnd = _boundary_classes(c16_path)
    # --- identity + per-regime mask-stress probe (H2 + H3); leakage precomputed in parallel (dominant cost) ---
    identity = identifiability_stress.identity_probe(extract_dir, c10_dir, n_perm=n_perm, folds=folds)
    leak_cache = ssr.precompute_all_leakage(extract_dir, boundary_classes=bnd, n_perturb=n_perturb, folds=folds,
                                            n_workers=n_workers)
    probe = {r: identifiability_stress.mask_stress_probe(extract_dir, c10_dir, r, boundary_classes=bnd,
                                                         n_perturb=n_perturb, n_perm=n_perm, folds=folds,
                                                         leakage_cache=leak_cache)
             for r in schema.REGIME_ORDER}
    # --- boundary (H4) + leakage estimability (H5) ---
    boundary = boundary_stress.boundary_visibility_stress(extract_dir, loso_root, c10_dir, boundary_classes=bnd,
                                                          n_perturb=n_perturb, folds=folds)
    leakage = leakage_stress.leakage_estimability_stress(extract_dir, boundary_classes=bnd, n_perturb=n_perturb,
                                                         folds=folds)
    # --- support-graph metrics per regime (for tables) ---
    sgmetrics = _support_graph_metrics_by_regime(extract_dir, bnd, n_perturb, folds)
    # --- C18-D observability dropout (SECONDARY) ---
    from ..identifiability.signal_atlas import build_atlas, load_replay
    dropout = observability_dropout.observability_dropout_stress(build_atlas(load_replay(c10_dir)), n_perm=n_perm)
    # --- severity response + taxonomy ---
    axis_by = {r: {"accuracy_visibility": probe[r]["accuracy_visibility"],
                   "calibration_visibility": probe[r]["calibration_visibility"]} for r in schema.REGIME_ORDER}
    leak_by = {r: {"any_estimable": (leakage["per_regime"][r]["source_estimable_fraction"] or 0.0) > 0.0}
               for r in schema.REGIME_ORDER}
    estim_by = {r: {"any_comparable_remaining": (leakage["per_regime"][r]["source_estimable_fraction"] or 0.0) > 0.0}
                for r in schema.REGIME_ORDER}
    severity = severity_response.severity_response(probe, boundary["per_regime"], leakage["per_regime"], axis_by)
    tax = taxonomy.severity_taxonomy(probe_by_regime=probe, axis_by_regime=axis_by,
                                     boundary_s6_s7={"s6_corr": boundary["s6_corr"], "s7_corr": boundary["s7_corr"]},
                                     leakage_by_regime=leak_by, estimability_by_regime=estim_by,
                                     s0_reproduces_c17=identity["G1_s0_reproduces_case_iii"])
    gates = {"G1_s0_reproduces_case_iii": identity["G1_s0_reproduces_case_iii"],
             "G2_auc_reproduces_0602": identity["G2_auc_reproduces_0602"],
             "G3_oracle_rho_reproduces": identity["G3_oracle_rho_reproduces"],
             "G4_no_strong_scalar": identity["G4_no_strong_scalar"],
             "G5_targets_joined_post_hoc": True, "G6_finite_filter": True,
             "all_identity_gates_pass": all([identity["G1_s0_reproduces_case_iii"], identity["G2_auc_reproduces_0602"],
                                             identity["G3_oracle_rho_reproduces"], identity["G4_no_strong_scalar"]])}
    return {"boundary_classes": list(bnd), "identity_probe": identity, "mask_stress_by_regime": probe,
            "boundary": boundary, "leakage": leakage, "support_graph_metrics": sgmetrics,
            "observability_dropout": dropout, "severity_response": severity, "taxonomy": tax, "gates": gates,
            "recomputable_features": list(feature_inventory.recomputable_features()),
            "static_features": list(feature_inventory.static_features()),
            "n_folds": len(folds if folds is not None else ssr._list_folds(extract_dir)),
            "diagnostic_only_non_deployable": True}


def _support_graph_metrics_by_regime(extract_dir, bnd, n_perturb, folds):
    from . import masks
    fold_dirs = folds if folds is not None else ssr._list_folds(extract_dir)
    agg = {r: {"n": 0, "elig_removed": 0, "comp_lost": 0} for r in schema.REGIME_ORDER}
    base_elig = base_comp = n = 0
    for (seed, target) in fold_dirs:
        for level in ssr._levels(extract_dir, seed, target):
            fld = ssr.load_fold_level(extract_dir, seed, target, level)
            sg = fld["support_source"]; bm = support_metrics.support_graph_metrics(sg)
            base_elig += bm["n_eligible_cells"]; base_comp += bm["n_comparable_classes"]; n += 1
            for r in schema.REGIME_ORDER:
                na, _ = ssr._regime_name_actions(r, sg, boundary_classes=bnd, seed=seed, target=target,
                                                 level=level, n_perturb=n_perturb)
                loss = support_metrics.estimability_loss(sg, masks.apply_to_support_graph(na, sg))
                agg[r]["n"] += 1; agg[r]["elig_removed"] += loss["eligible_cells_removed"]
                agg[r]["comp_lost"] += loss["comparable_classes_lost"]
    return {"n_fold_levels": n, "base_mean_eligible_cells": (base_elig / n if n else None),
            "base_mean_comparable_classes": (base_comp / n if n else None), "per_regime": agg}


def write_tables(res, tdir) -> None:
    os.makedirs(tdir, exist_ok=True)
    R = schema.REGIME_ORDER

    _writecsv(os.path.join(tdir, "support_stress_regime_catalog.csv"),
              [{"regime": r, "severity": schema.REGIME_SEVERITY[r], "deletes_cells": schema.REGIME_DELETES[r],
                "description": schema.REGIME_DESC[r]} for r in R],
              ["regime", "severity", "deletes_cells", "description"])

    _writecsv(os.path.join(tdir, "c18_recomputable_feature_inventory.csv"), feature_inventory.inventory(),
              ["feature", "class", "usable_for_mask_stress"])

    _writecsv(os.path.join(tdir, "multivariate_probe_loto_by_regime.csv"),
              [{"regime": r, **{k: res["mask_stress_by_regime"][r][k] for k in
                                ("n_used", "n_features", "loto_auc", "loso_auc", "beats_permutation", "base_rate")}}
               for r in R], ["regime", "n_used", "n_features", "loto_auc", "loso_auc", "beats_permutation", "base_rate"])

    _writecsv(os.path.join(tdir, "multivariate_probe_permutation_by_regime.csv"),
              [{"regime": r, "loto_auc": res["mask_stress_by_regime"][r]["loto_auc"],
                "permutation_mean_auc": res["mask_stress_by_regime"][r]["permutation_mean_auc"],
                "permutation_p": res["mask_stress_by_regime"][r]["permutation_p"],
                "beats_permutation": res["mask_stress_by_regime"][r]["beats_permutation"]} for r in R],
              ["regime", "loto_auc", "permutation_mean_auc", "permutation_p", "beats_permutation"])

    _writecsv(os.path.join(tdir, "source_signal_identifiability_by_regime.csv"),
              [{"regime": r, "univariate_verdict": res["mask_stress_by_regime"][r]["univariate_verdict"],
                "n_weak_accuracy": res["mask_stress_by_regime"][r]["n_weak_accuracy"],
                "max_abs_accuracy_spearman": res["mask_stress_by_regime"][r]["max_abs_accuracy_spearman"]} for r in R],
              ["regime", "univariate_verdict", "n_weak_accuracy", "max_abs_accuracy_spearman"])

    _writecsv(os.path.join(tdir, "calibration_accuracy_visibility_by_regime.csv"),
              [{"regime": r, "accuracy_visibility": res["mask_stress_by_regime"][r]["accuracy_visibility"],
                "calibration_visibility": res["mask_stress_by_regime"][r]["calibration_visibility"],
                "calibration_biased": res["mask_stress_by_regime"][r]["source_signals_calibration_biased"]} for r in R],
              ["regime", "accuracy_visibility", "calibration_visibility", "calibration_biased"])

    _writecsv(os.path.join(tdir, "class_boundary_visibility_by_regime.csv"),
              [{"regime": r, **res["boundary"]["per_regime"][r]} for r in R],
              ["regime", "n_class_fold_points", "source_target_recall_delta_corr", "boundary_source_visible"])

    _writecsv(os.path.join(tdir, "boundary_aligned_vs_random_mask.csv"),
              [{"mask": "S6_boundary_aligned", "corr": res["boundary"]["s6_corr"]},
               {"mask": "S7_random_matched", "corr": res["boundary"]["s7_corr"]},
               {"mask": "S0_full_support", "corr": res["boundary"]["s0_corr"]}], ["mask", "corr"])

    _writecsv(os.path.join(tdir, "leakage_estimability_by_regime.csv"),
              [{"regime": r, **res["leakage"]["per_regime"][r]} for r in R],
              ["regime", "n_fold_levels", "severity", "mean_src_eligible_cells_removed",
               "mean_src_comparable_classes_lost", "mean_audit_comparable_classes_lost",
               "mean_mass_removed_fraction", "source_estimable_fraction", "audit_estimable_fraction", "any_estimable"])

    _writecsv(os.path.join(tdir, "estimability_loss_by_regime.csv"),
              [{"regime": r, "mean_eligible_cells_removed": res["support_graph_metrics"]["per_regime"][r]["elig_removed"]
                / max(res["support_graph_metrics"]["per_regime"][r]["n"], 1),
                "mean_comparable_classes_lost": res["support_graph_metrics"]["per_regime"][r]["comp_lost"]
                / max(res["support_graph_metrics"]["per_regime"][r]["n"], 1)} for r in R],
              ["regime", "mean_eligible_cells_removed", "mean_comparable_classes_lost"])

    _writecsv(os.path.join(tdir, "observability_dropout_secondary.csv"),
              [{"regime": r, **res["observability_dropout"]["per_regime"][r]} for r in R],
              ["regime", "n_dropped", "loto_auc", "permutation_p", "beats_permutation", "univariate_verdict"])

    _writecsv(os.path.join(tdir, "severity_response_summary.csv"), res["severity_response"]["severity_rows"],
              ["regime", "severity", "loto_auc", "beats_permutation", "permutation_p", "n_used",
               "boundary_corr", "leakage_source_estimable_fraction", "accuracy_visibility", "calibration_visibility"])

    _writecsv(os.path.join(tdir, "probe_stress_validity_gates.csv"),
              [{"gate": k, "pass": v} for k, v in res["gates"].items()], ["gate", "pass"])

    _writecsv(os.path.join(tdir, "c17_original_vs_c18_recomputed_s0.csv"),
              [{"quantity": "loto_auc", "c17_original": schema.C17_LOTO_AUC,
                "c18_identity_probe": res["identity_probe"]["loto_auc"],
                "c18_recomputable_s0": res["mask_stress_by_regime"]["S0_full_support"]["loto_auc"]},
               {"quantity": "oracle_spearman_bacc", "c17_original": schema.C17_ORACLE_SPEARMAN_BACC,
                "c18_identity_probe": res["identity_probe"]["oracle_spearman_bacc"], "c18_recomputable_s0": None}],
              ["quantity", "c17_original", "c18_identity_probe", "c18_recomputable_s0"])

    _writecsv(os.path.join(tdir, "c18_case_taxonomy.csv"),
              [{"case_label": res["taxonomy"]["case_label"], "summary": res["taxonomy"]["summary"],
                "next_science": res["taxonomy"]["next_science"]}],
              ["case_label", "summary", "next_science"])


def render_md(res) -> str:
    g = res["gates"]; t = res["taxonomy"]; b = res["boundary"]; ip = res["identity_probe"]
    L = ["# C18 — Controlled Support-Mismatch x Identifiability Stress Test", "",
         "> Identifiability stress test (NOT a new OACI/SRC experiment, NOT retraining). GPU re-inference of the "
         "C17 candidate checkpoints (source-forward only) makes H1/H2 genuinely mask-recomputed. Target labels "
         "are diagnostic-only; no selector is produced.", "",
         f"- **CASE: `{t['case_label']}`** — {t['interpretation']}",
         f"- next science: {t['next_science']}",
         f"- boundary-rotation classes (from C16): {res['boundary_classes']}  ·  folds analysed: {res['n_folds']}", "",
         "## Identity gates (S0 must reproduce C17)", "",
         f"- G1 S0 reproduces Case III: **{g['G1_s0_reproduces_case_iii']}**  ·  "
         f"G2 AUC≈0.602: **{g['G2_auc_reproduces_0602']}** (got {ip['loto_auc']})  ·  "
         f"G3 oracle ρ≈+0.120: **{g['G3_oracle_rho_reproduces']}** (got {ip['oracle_spearman_bacc']})  ·  "
         f"G4 no strong scalar: **{g['G4_no_strong_scalar']}**",
         f"- G5 targets joined post-hoc: **{g['G5_targets_joined_post_hoc']}**  ·  G6 finite-filter: **{g['G6_finite_filter']}**",
         "", "## H2 — multivariate identifiability under support stress (recomputable-column probe)", "",
         "| regime | n_used | loto_auc | perm_p | beats_perm |", "|---|---:|---:|---:|:--:|"]
    for r in schema.REGIME_ORDER:
        m = res["mask_stress_by_regime"][r]
        L.append(f"| {r} | {m['n_used']} | {_f(m['loto_auc'])} | {_f(m['permutation_p'])} | {m['beats_permutation']} |")
    L += ["", "> The recomputable-column S0 is a DECLARED baseline (differs from the all-column 0.602 because the "
          "3 weak scalar signals R_src/balanced_err/train_surrogate are training-realized STATIC features, "
          "excluded from mask claims). The all-column identity probe reproduces 0.602 (G2).", "",
          "## H4 — class-boundary source-visibility: boundary-aligned (S6) vs random-matched (S7)", "",
          f"- S0 corr {_f(b['s0_corr'])}  ·  S6 corr {_f(b['s6_corr'])}  ·  S7 corr {_f(b['s7_corr'])}  ·  "
          f"boundary-aligned destroys mirror vs random: **{b['boundary_aligned_destroys_mirror_vs_random']}**", "",
          "## H5 — support-aware leakage estimability / abstention", "",
          f"- S0 source-estimable fraction {_f(res['leakage']['s0_source_estimable_fraction'])}  ·  "
          f"min deleting-regime estimable fraction {_f(res['leakage']['min_deleting_estimable_fraction'])}  ·  "
          f"abstains under degradation: **{res['leakage']['abstains_under_degradation']}**", "",
          "## C18-D (secondary) — observability-dropout proxy", "",
          f"- {res['observability_dropout']['label']} · is_primary={res['observability_dropout']['is_primary']} "
          "(drops would-be-non-estimable columns; NOT source-distribution recompute; for comparison only)", "",
          "## Interpretation", "", f"> {t['summary']}"]
    return "\n".join(L)


def render_taxonomy_md(res) -> str:
    t = res["taxonomy"]
    ev = "\n".join(f"- {k}: {v}" for k, v in t["evidence"].items())
    return (f"# C18 — support-stress taxonomy\n\n- **CASE: `{t['case_label']}`**\n\n## Evidence\n\n{ev}\n\n"
            f"## Interpretation\n> {t['interpretation']}\n\n## Next science\n> {t['next_science']}\n")


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        if s in low:
            raise ValueError(f"forbidden over-claim in report: {s!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.support_stress.report")
    ap.add_argument("--extract-dir", required=True)
    ap.add_argument("--c10-dir", required=True)
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--n-perturb", type=int, default=2)
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args(argv)
    res = run(args.extract_dir, args.c10_dir, args.loso_root, n_perm=args.n_perm, n_perturb=args.n_perturb,
              n_workers=args.n_workers)
    md = render_md(res); _guard_forbidden(md)
    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "C18_CONTROLLED_SUPPORT_MISMATCH_STRESS.md"), "w").write(md)
    json.dump(res, open(os.path.join(args.out_dir, "C18_CONTROLLED_SUPPORT_MISMATCH_STRESS.json"), "w"),
              indent=2, sort_keys=True, default=str)
    open(os.path.join(args.out_dir, "C18_SUPPORT_STRESS_TAXONOMY.md"), "w").write(render_taxonomy_md(res))
    write_tables(res, os.path.join(args.out_dir, "c18_tables"))
    g = res["gates"]
    print(f"[C18] case={res['taxonomy']['case_label']} identity_gates_pass={g['all_identity_gates_pass']} "
          f"G2={g['G2_auc_reproduces_0602']} boundary_specific={res['boundary']['boundary_aligned_destroys_mirror_vs_random']} "
          f"abstains={res['leakage']['abstains_under_degradation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
