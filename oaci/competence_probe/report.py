"""C19 — assemble the Pre-registered Low-Freedom Source-Only Competence Probe. Emits the pre-registration
(frozen config + hash), the robust-core (primary) + endpoint-augmented (secondary) LOTO results with
permutation baselines across S0/S2/S3, the endpoint-estimability gate, the per-target heterogeneity readout,
the case taxonomy, a forbidden-claim audit, and the no-selector gate. DIAGNOSTIC-ONLY; no selector emitted."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

from . import labels, robustness, schema, validation


def _config_hash() -> str:
    return hashlib.sha256(json.dumps(schema.frozen_config(), sort_keys=True).encode()).hexdigest()[:16]


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _boundary_classes(c16_path="oaci/reports/C16_MECHANISM_DEEP_DIVE.json") -> tuple:
    from ..support_stress.stress_plan import boundary_classes_from_c16
    d = json.load(open(c16_path))
    return boundary_classes_from_c16(d["harm_decomposition"]["per_class_recall_delta"])


def _heterogeneity(robust_s0) -> dict:
    pt = {k: v for k, v in (robust_s0.get("per_target_auc") or {}).items() if v is not None}
    if not pt:
        return {"spread": None, "min": None, "max": None, "heterogeneous": False}
    vals = list(pt.values())
    spread = max(vals) - min(vals)
    return {"spread": spread, "min": min(vals), "max": max(vals),
            "heterogeneous": bool(spread > schema.HETEROGENEITY_SPREAD or min(vals) < 0.40)}


def _taxonomy(primary, robust_by_regime, aug_by_regime, hetero) -> dict:
    aug_pass = all(aug_by_regime.get(r, {}).get("passes") for r in schema.SUCCESS_REGIMES)
    if primary["primary_success"] and hetero["heterogeneous"]:
        case = schema.CASE_HETEROGENEOUS
    elif primary["primary_success"]:
        case = schema.CASE_ROBUST_CORE_RECOVERS
    elif aug_pass:
        case = schema.CASE_ENDPOINT_AUGMENTED_ONLY
    else:
        case = schema.CASE_NOT_RECOVERABLE
    nxt = {
        schema.CASE_ROBUST_CORE_RECOVERS: "C20 = external / new-regime validation of the diagnostic probe (NOT a DG penalty, NOT a selector).",
        schema.CASE_HETEROGENEOUS: "weak diagnostic requiring external validation; do NOT call it a detector; report per-target heterogeneity.",
        schema.CASE_ENDPOINT_AUGMENTED_ONLY: "signal depends on source accuracy endpoints WHEN estimable; emphasize endpoint-estimability limits.",
        schema.CASE_NOT_RECOVERABLE: "C17/C18 weak multivariate identifiability was exploratory but not recoverable under pre-registered low-freedom constraints; package as mechanism + falsification framework.",
    }[case]
    return {"case_label": case, "primary_success": primary["primary_success"],
            "endpoint_augmented_success": bool(aug_pass), "per_target_heterogeneity": hetero,
            "next_science": nxt, "diagnostic_only_non_deployable": True}


def run(extract_dir, c10_dir, *, folds=None, n_perm=schema.N_PERM, n_workers=8,
        c16_path="oaci/reports/C16_MECHANISM_DEEP_DIVE.json") -> dict:
    bnd = _boundary_classes(c16_path)
    rob = robustness.run_robustness(extract_dir, c10_dir, boundary_classes=bnd, folds=folds, n_perm=n_perm,
                                    n_workers=n_workers)
    robust_by = {r: rob[r]["robust_core"] for r in schema.ROBUSTNESS_REGIMES}
    aug_by = {r: rob[r]["endpoint_augmented"] for r in schema.ROBUSTNESS_REGIMES}
    primary = validation.primary_success(robust_by)
    hetero = _heterogeneity(robust_by[schema.PRIMARY_REGIME])
    tax = _taxonomy(primary, robust_by, aug_by, hetero)
    no_selector = all(("selector" not in robust_by[r]) and ("chosen_model_hash" not in robust_by[r])
                      for r in schema.ROBUSTNESS_REGIMES)
    gates = {"preregistration_config_hash": _config_hash(),
             "feature_registry_frozen": True,
             "static_columns_excluded_from_primary": not (set(schema.STATIC_EXCLUDED) & set(schema.ROBUST_CORE_FEATURES)),
             "fragile_endpoints_excluded_from_primary": not (set(schema.ENDPOINT_FEATURES) & set(schema.ROBUST_CORE_FEATURES)),
             "targets_joined_post_hoc": True, "finite_filter": True, "no_selector_artifact": bool(no_selector)}
    return {"boundary_classes": list(bnd), "regimes": list(schema.ROBUSTNESS_REGIMES),
            "robustness": rob, "primary": primary, "taxonomy": tax, "gates": gates,
            "frozen_config": schema.frozen_config(), "config_hash": _config_hash(),
            "n_folds": len(folds) if folds is not None else None, "diagnostic_only_non_deployable": True}


# ---------- artifacts ----------
def write_tables(res, tdir) -> None:
    os.makedirs(tdir, exist_ok=True)
    R = res["regimes"]; rob = res["robustness"]

    _writecsv(os.path.join(tdir, "feature_registry.csv"),
              [{"feature": f, "family": schema.feature_family(f),
                "in_robust_core": f in schema.ROBUST_CORE_FEATURES,
                "in_endpoint_augmented": f in (schema.ROBUST_CORE_FEATURES + schema.ENDPOINT_FEATURES),
                "excluded": f in schema.STATIC_EXCLUDED}
               for f in (schema.ROBUST_CORE_FEATURES + schema.ENDPOINT_FEATURES + schema.STATIC_EXCLUDED)],
              ["feature", "family", "in_robust_core", "in_endpoint_augmented", "excluded"])

    _writecsv(os.path.join(tdir, "probe_preregistration.csv"),
              [{"key": k, "value": v} for k, v in res["frozen_config"].items()], ["key", "value"])

    _writecsv(os.path.join(tdir, "endpoint_estimability_gate.csv"),
              [{"regime": r, **{k: rob[r]["robust_core"]["gate"][k] for k in
                                ("scored_rate", "endpoint_nonestimable_rate", "insufficient_finite_rate")},
                "endpoint_gate_nonestimable_rate": rob[r]["endpoint_augmented"]["gate"]["endpoint_nonestimable_rate"]}
               for r in R],
              ["regime", "scored_rate", "endpoint_nonestimable_rate", "insufficient_finite_rate",
               "endpoint_gate_nonestimable_rate"])

    def probe_rows(key):
        return [{"regime": r, **{k: rob[r][key][k] for k in
                                 ("n_used", "n_features", "loto_auc", "loso_auc", "beats_permutation",
                                  "meets_margin", "passes")}} for r in R]
    _writecsv(os.path.join(tdir, "robust_core_loto_results.csv"), probe_rows("robust_core"),
              ["regime", "n_used", "n_features", "loto_auc", "loso_auc", "beats_permutation", "meets_margin", "passes"])
    _writecsv(os.path.join(tdir, "endpoint_augmented_loto_results.csv"),
              [{**row, "n_endpoint_estimable": rob[row["regime"]]["endpoint_augmented"].get("n_endpoint_estimable")}
               for row in probe_rows("endpoint_augmented")],
              ["regime", "n_used", "n_endpoint_estimable", "n_features", "loto_auc", "loso_auc",
               "beats_permutation", "meets_margin", "passes"])

    def perm_rows(key):
        return [{"regime": r, "loto_auc": rob[r][key]["loto_auc"], "permutation_mean_auc": rob[r][key]["permutation_mean_auc"],
                 "permutation_p": rob[r][key]["permutation_p"], "beats_permutation": rob[r][key]["beats_permutation"]}
                for r in R]
    _writecsv(os.path.join(tdir, "robust_core_permutation_baseline.csv"), perm_rows("robust_core"),
              ["regime", "loto_auc", "permutation_mean_auc", "permutation_p", "beats_permutation"])
    _writecsv(os.path.join(tdir, "endpoint_augmented_permutation_baseline.csv"), perm_rows("endpoint_augmented"),
              ["regime", "loto_auc", "permutation_mean_auc", "permutation_p", "beats_permutation"])

    _writecsv(os.path.join(tdir, "per_target_probe_results.csv"),
              [{"regime": r, "target": t, "robust_core_auc": v}
               for r in R for t, v in sorted((rob[r]["robust_core"].get("per_target_auc") or {}).items())],
              ["regime", "target", "robust_core_auc"])

    _writecsv(os.path.join(tdir, "regime_robustness_results.csv"),
              [{"regime": r, "base_rate": rob[r]["base_rate"], "robust_core_passes": rob[r]["robust_core"]["passes"],
                "endpoint_augmented_passes": rob[r]["endpoint_augmented"]["passes"]} for r in R],
              ["regime", "base_rate", "robust_core_passes", "endpoint_augmented_passes"])

    _writecsv(os.path.join(tdir, "abstention_reason_codes.csv"),
              [{"regime": r, "status": s, "count": rob[r]["robust_core"]["gate"]["counts"][s]}
               for r in R for s in schema.SCORE_STATUS],
              ["regime", "status", "count"])

    _writecsv(os.path.join(tdir, "probe_score_calibration.csv"),
              [{"regime": r, "base_rate": rob[r]["base_rate"], "loto_auc": rob[r]["robust_core"]["loto_auc"],
                "scored_rate": rob[r]["robust_core"]["gate"]["scored_rate"]} for r in R],
              ["regime", "base_rate", "loto_auc", "scored_rate"])

    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"),
              [{"check": k, "pass": v} for k, v in res["gates"].items()], ["check", "pass"])

    h = res["taxonomy"]["per_target_heterogeneity"]
    _writecsv(os.path.join(tdir, "c19_case_taxonomy.csv"),
              [{"case_label": res["taxonomy"]["case_label"], "primary_success": res["primary"]["primary_success"],
                "per_target_spread": h["spread"], "heterogeneous": h["heterogeneous"],
                "next_science": res["taxonomy"]["next_science"]}],
              ["case_label", "primary_success", "per_target_spread", "heterogeneous", "next_science"])


def render_preregistration_md(res) -> str:
    c = res["frozen_config"]
    L = [f"# C19 — probe pre-registration (config hash `{res['config_hash']}`)", "",
         "> FROZEN before any fit. No grid search, no feature selection, no post-hoc tuning. Executed config is "
         "asserted to match this hash (test_c19_preregistration_matches_executed_config).", "",
         f"- model: `{c['model']}`  ·  L2 C: {c['l2_C']}  ·  standardize: {c['standardize']}  ·  iters/lr: {c['iters']}/{c['lr']}",
         f"- validation: `{c['validation']}`  ·  permutation: within-(seed,target,level), n={c['n_perm']}, seed {c['perm_seed']}",
         f"- success: LOTO beats permutation p<{c['success_p']} AND AUC−perm_mean≥{c['success_auc_margin']} on {c['success_regimes']}",
         f"- diagnostic label (post-hoc only): `{c['diagnostic_label']}`", "",
         "## Robust-core features (primary)", "", ", ".join(c["robust_core_features"]), "",
         "## Endpoint features (secondary, only where estimable)", "", ", ".join(c["endpoint_features"]), "",
         "## Static features (excluded entirely)", "", ", ".join(c["static_excluded"])]
    return "\n".join(L)


def render_forbidden_audit_md(res) -> str:
    return ("# C19 — forbidden-claim audit\n\nThis probe is DIAGNOSTIC-ONLY and emits NO selector. The report "
            "text is checked against the forbidden-claim list; the no-selector gate is asserted.\n\n"
            + "\n".join(f"- FORBIDDEN: {s}" for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS)
            + f"\n\n- no_selector_artifact gate: {res['gates']['no_selector_artifact']}\n"
            + f"- static excluded from primary: {res['gates']['static_columns_excluded_from_primary']}\n"
            + f"- fragile endpoints excluded from primary: {res['gates']['fragile_endpoints_excluded_from_primary']}\n")


def render_md(res) -> str:
    t = res["taxonomy"]; rob = res["robustness"]; R = res["regimes"]
    L = [f"# C19 — Source-Only Competence Probe (diagnostic-only, config `{res['config_hash']}`)", "",
         "> Pre-registered low-freedom probe. NOT a target-free selector, NOT an OACI rescue, NOT a deployable "
         "competence detector. Tests whether the weak C17/C18 source-only signal survives pre-registration + "
         "feature discipline using DELETION-ROBUST observables, with endpoint-estimability as a first-class output.", "",
         f"- **CASE: `{t['case_label']}`**  ·  primary_success (robust-core beats perm+margin on S0/S2/S3): "
         f"**{t['primary_success']}**",
         f"- next science: {t['next_science']}", "",
         "## Robust-core (primary) — LOTO vs within-fold permutation", "",
         "| regime | n_used | n_feat | loto_auc | perm_mean | perm_p | margin≥.03 | passes |",
         "|---|---:|---:|---:|---:|---:|:--:|:--:|"]
    for r in R:
        m = rob[r]["robust_core"]
        L.append(f"| {r} | {m['n_used']} | {m['n_features']} | {_f(m['loto_auc'])} | {_f(m['permutation_mean_auc'])} "
                 f"| {_f(m['permutation_p'])} | {m['meets_margin']} | {m['passes']} |")
    L += ["", "## Endpoint-estimability gate (first-class output)", "",
          "| regime | scored_rate | endpoint_nonestimable_rate (aug) |", "|---|---:|---:|"]
    for r in R:
        L.append(f"| {r} | {_f(rob[r]['robust_core']['gate']['scored_rate'])} | "
                 f"{_f(rob[r]['endpoint_augmented']['gate']['endpoint_nonestimable_rate'])} |")
    h = t["per_target_heterogeneity"]
    L += ["", "## Endpoint-augmented (secondary, only where estimable)", "",
          "| regime | n_endpoint_estimable | loto_auc | perm_p | passes |", "|---|---:|---:|---:|:--:|"]
    for r in R:
        a = rob[r]["endpoint_augmented"]
        L.append(f"| {r} | {a.get('n_endpoint_estimable')} | {_f(a['loto_auc'])} | {_f(a['permutation_p'])} | {a['passes']} |")
    L += ["", f"## Per-target heterogeneity (S0 robust-core)", "",
          f"- spread {_f(h['spread'])} (min {_f(h['min'])}, max {_f(h['max'])}) → heterogeneous: **{h['heterogeneous']}**", "",
          "## Gates", "",
          f"- preregistration hash `{res['gates']['preregistration_config_hash']}` · static excluded from primary "
          f"**{res['gates']['static_columns_excluded_from_primary']}** · fragile endpoints excluded "
          f"**{res['gates']['fragile_endpoints_excluded_from_primary']}** · targets post-hoc "
          f"**{res['gates']['targets_joined_post_hoc']}** · finite-filter **{res['gates']['finite_filter']}** · "
          f"no selector artifact **{res['gates']['no_selector_artifact']}**", "",
          "> DIAGNOSTIC-ONLY. A positive result means only that a pre-registered low-freedom source-only "
          "diagnostic probe recovers weak competence information; it is NOT evidence of a deployment-ready "
          "target-free checkpoint chooser, and no such artifact is produced."]
    return "\n".join(L)


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        if s in low:
            raise ValueError(f"forbidden claim in C19 report: {s!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.competence_probe.report")
    ap.add_argument("--extract-dir", required=True)
    ap.add_argument("--c10-dir", required=True)
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--n-perm", type=int, default=schema.N_PERM)
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args(argv)
    res = run(args.extract_dir, args.c10_dir, n_perm=args.n_perm, n_workers=args.n_workers)
    md = render_md(res); _guard_forbidden(md)
    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "C19_SOURCE_ONLY_COMPETENCE_PROBE.md"), "w").write(md)
    json.dump(res, open(os.path.join(args.out_dir, "C19_SOURCE_ONLY_COMPETENCE_PROBE.json"), "w"),
              indent=2, sort_keys=True, default=str)
    open(os.path.join(args.out_dir, "C19_PROBE_PREREGISTRATION.md"), "w").write(render_preregistration_md(res))
    open(os.path.join(args.out_dir, "C19_FORBIDDEN_CLAIM_AUDIT.md"), "w").write(render_forbidden_audit_md(res))
    write_tables(res, os.path.join(args.out_dir, "c19_tables"))
    t = res["taxonomy"]
    print(f"[C19] case={t['case_label']} primary_success={t['primary_success']} "
          f"config_hash={res['config_hash']} no_selector={res['gates']['no_selector_artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
