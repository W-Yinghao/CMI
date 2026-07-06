"""C20 — assemble the frozen-probe new-regime validation. Locks to the C19 config (hash 664007686afb520f) +
feature set, builds the robust-core feature atlas for the development (S0/S2/S3) and held-out (S4/S5/S6/S7)
regimes from the committed C18 replay, runs the cross-regime LOTO of the frozen probe on each held-out regime
with a cross-regime permutation baseline, reports feature availability / abstention + the secondary endpoint-
augmented check, chooses a case, and emits the C20-B external-dataset PROTOCOL (document only) + a claim-
boundary audit. DIAGNOSTIC-ONLY; no selector; no second-dataset execution."""
from __future__ import annotations

import argparse
import csv
import json
import os

from ..competence_probe import feature_registry
from ..competence_probe import schema as c19
from ..support_stress import source_signal_recompute as ssr
from ..support_stress.stress_plan import boundary_classes_from_c16
from . import (abstention, cross_regime_validation, endpoint_checks, feature_lock, frozen_config, regime_splits,
               schema)

_ALL_REGIMES = tuple(schema.DEVELOPMENT_REGIMES) + tuple(schema.HELD_OUT_REGIMES)


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _boundary_classes(c16_path):
    return boundary_classes_from_c16(json.load(open(c16_path))["harm_decomposition"]["per_class_recall_delta"])


def _atlas_by_regime(extract_dir, c10_dir, bnd, folds, n_workers):
    cache = ssr.precompute_all_leakage(extract_dir, boundary_classes=bnd, folds=folds, n_workers=n_workers,
                                       regimes=list(_ALL_REGIMES))
    out = {}
    for regime in _ALL_REGIMES:
        lk = (lambda s, t, l, mh, rg=regime: cache.get((s, t, l, rg, mh), (None, None)))
        out[regime] = feature_registry.build_atlas(extract_dir, c10_dir, regime, boundary_classes=bnd,
                                                    leakage_lookup=lk, folds=folds)
    return out


def _taxonomy(heldout, availability) -> dict:
    passes = {r: bool(heldout[r]["primary"]["passes"]) for r in schema.HELD_OUT_REGIMES}
    n_pass = sum(passes.values())
    del_pass = [passes[r] for r in schema.DELETION_HELD_OUT]
    s5_pass = passes.get("S5_block_class_by_domain", False)
    failing = [r for r in schema.HELD_OUT_REGIMES if not passes[r]]
    # availability is the LIMITER only if the ROBUST-CORE features are themselves unavailable on failing regimes
    low_avail = [r for r in failing if (availability[r]["robust_core_scored_rate"] or 0.0) < schema.AVAILABILITY_FLOOR]
    avail_limited = bool(failing) and len(low_avail) == len(failing)      # ALL failing regimes robust-unavailable
    if n_pass == len(schema.HELD_OUT_REGIMES):
        case = schema.CASE_GENERALIZES
    elif avail_limited:
        case = schema.CASE_AVAILABILITY_LIMITED
    elif s5_pass and not any(del_pass):
        case = schema.CASE_SURVIVES_NONDELETION
    elif n_pass >= 1:
        case = schema.CASE_PARTIAL                                       # some pass with robust core AVAILABLE
    else:
        case = schema.CASE_REGIME_LOCAL
    passing_regimes = [r for r in schema.HELD_OUT_REGIMES if passes[r]]
    nxt = {
        schema.CASE_GENERALIZES: "C20-B external-dataset protocol may proceed to C21 execution (still diagnostic-only).",
        schema.CASE_PARTIAL: (f"cross-regime generalization is PARTIAL/marginal (passes {passing_regimes} at thin "
                              "margins; fails the rest). Robust-core features ARE available, so failures are "
                              "relationship-level not availability-level. Scope the claim to the passing regimes; "
                              "do NOT claim general external generalization. Treat marginal passes as fragile "
                              "(multiple-comparisons across held-out regimes)."),
        schema.CASE_SURVIVES_NONDELETION: "signal robust to block/rare/nonestimable but not deletion; scope the claim to non-deletion support stress.",
        schema.CASE_REGIME_LOCAL: "C19 positive is regime-local; report the boundary; do NOT externally generalize the claim.",
        schema.CASE_AVAILABILITY_LIMITED: "held-out validity is limited by observable AVAILABILITY (robust-core non-estimability), not the source-target relationship.",
    }[case]
    return {"case_label": case, "held_out_pass": passes, "n_pass": n_pass, "passing_regimes": passing_regimes,
            "robust_core_available_on_failing": {r: availability[r]["robust_core_scored_rate"] for r in failing},
            "next_science": nxt, "diagnostic_only_non_deployable": True}


def run(extract_dir, c10_dir, *, folds=None, n_perm=schema.N_PERM, n_workers=8,
        c16_path="oaci/reports/C16_MECHANISM_DEEP_DIVE.json") -> dict:
    frozen_config.assert_locked(); feature_lock.assert_locked(); regime_splits.assert_no_leakage_between_splits()
    bnd = _boundary_classes(c16_path)
    atlas = _atlas_by_regime(extract_dir, c10_dir, bnd, folds, n_workers)
    dev_by = {r: atlas[r] for r in schema.DEVELOPMENT_REGIMES}
    cols = list(c19.ROBUST_CORE_FEATURES)
    heldout, avail = {}, {}
    for r in schema.HELD_OUT_REGIMES:
        heldout[r] = {"primary": cross_regime_validation.cross_regime_loto(dev_by, atlas[r], cols, n_perm=n_perm),
                      "endpoint_secondary": endpoint_checks.endpoint_augmented_cross_regime(dev_by, atlas[r], n_perm=n_perm)}
        avail[r] = abstention.availability(atlas[r])
    tax = _taxonomy(heldout, avail)
    no_selector = all(("selector" not in heldout[r]["primary"]) and ("chosen_model_hash" not in heldout[r]["primary"])
                      for r in schema.HELD_OUT_REGIMES)
    gates = {"locked_c19_config_hash": frozen_config.assert_locked(),
             "feature_lock": feature_lock.lock_audit()["robust_core_matches_c19"],
             "endpoints_excluded_from_primary": feature_lock.lock_audit()["endpoints_excluded_from_primary"],
             "development_heldout_disjoint": regime_splits.split_plan()["disjoint"],
             "targets_joined_post_hoc": True, "finite_filter": True, "no_selector_artifact": bool(no_selector),
             "success_margin_vs_strict_chance": schema.SUCCESS_AUC_MARGIN_VS_CHANCE}
    return {"boundary_classes": list(bnd), "split_plan": regime_splits.split_plan(),
            "feature_lock": feature_lock.lock_audit(), "held_out": heldout, "availability": avail,
            "taxonomy": tax, "gates": gates, "config_hash": frozen_config.assert_locked(),
            "n_folds": len(folds) if folds is not None else None, "diagnostic_only_non_deployable": True}


# ---------- artifacts ----------
def write_tables(res, tdir) -> None:
    os.makedirs(tdir, exist_ok=True)
    H = schema.HELD_OUT_REGIMES; ho = res["held_out"]

    _writecsv(os.path.join(tdir, "frozen_config_manifest.csv"),
              [{"key": k, "value": v} for k, v in c19.frozen_config().items()], ["key", "value"])
    fl = res["feature_lock"]
    _writecsv(os.path.join(tdir, "c19_to_c20_lock_audit.csv"),
              [{"check": "locked_config_hash", "value": res["config_hash"]},
               {"check": "robust_core_matches_c19", "value": fl["robust_core_matches_c19"]},
               {"check": "n_robust_core", "value": fl["n_robust_core"]},
               {"check": "endpoints_excluded_from_primary", "value": fl["endpoints_excluded_from_primary"]},
               {"check": "static_excluded_from_primary", "value": fl["static_excluded_from_primary"]}],
              ["check", "value"])
    _writecsv(os.path.join(tdir, "new_regime_split_plan.csv"),
              [{"role": "development", "regime": r} for r in schema.DEVELOPMENT_REGIMES]
              + [{"role": "held_out", "regime": r} for r in schema.HELD_OUT_REGIMES]
              + [{"role": "noop_negative_control", "regime": schema.NOOP_REGIME}], ["role", "regime"])

    _writecsv(os.path.join(tdir, "cross_regime_loto_results.csv"),
              [{"held_out_regime": r, **{k: ho[r]["primary"].get(k) for k in
                                         ("n_used", "loto_auc", "margin_vs_chance", "beats_permutation",
                                          "meets_margin_vs_chance", "passes")}} for r in H],
              ["held_out_regime", "n_used", "loto_auc", "margin_vs_chance", "beats_permutation",
               "meets_margin_vs_chance", "passes"])
    _writecsv(os.path.join(tdir, "cross_regime_permutation_baseline.csv"),
              [{"held_out_regime": r, "loto_auc": ho[r]["primary"].get("loto_auc"),
                "permutation_mean_auc": ho[r]["primary"].get("permutation_mean_auc"),
                "permutation_p": ho[r]["primary"].get("permutation_p")} for r in H],
              ["held_out_regime", "loto_auc", "permutation_mean_auc", "permutation_p"])
    _writecsv(os.path.join(tdir, "heldout_regime_results.csv"),
              [{"held_out_regime": r, "deletion": r in schema.DELETION_HELD_OUT, "passes": ho[r]["primary"]["passes"],
                "loto_auc": ho[r]["primary"].get("loto_auc"), "per_target_spread": ho[r]["primary"].get("per_target_spread")}
               for r in H],
              ["held_out_regime", "deletion", "passes", "loto_auc", "per_target_spread"])
    _writecsv(os.path.join(tdir, "per_target_new_regime_results.csv"),
              [{"held_out_regime": r, "target": t, "auc": v}
               for r in H for t, v in sorted((ho[r]["primary"].get("per_target_auc") or {}).items())],
              ["held_out_regime", "target", "auc"])
    _writecsv(os.path.join(tdir, "feature_availability_by_regime.csv"),
              [{"held_out_regime": r, "robust_core_scored_rate": res["availability"][r]["robust_core_scored_rate"],
                "robust_core_insufficient_rate": res["availability"][r]["robust_core_insufficient_rate"],
                "endpoint_available_rate": res["availability"][r]["endpoint_available_rate"],
                "endpoint_nonestimable_rate": res["availability"][r]["endpoint_nonestimable_rate"]} for r in H],
              ["held_out_regime", "robust_core_scored_rate", "robust_core_insufficient_rate",
               "endpoint_available_rate", "endpoint_nonestimable_rate"])
    _writecsv(os.path.join(tdir, "abstention_by_regime.csv"),
              [{"held_out_regime": r, "status": s, "count": res["availability"][r]["counts"][s]}
               for r in H for s in c19.SCORE_STATUS], ["held_out_regime", "status", "count"])
    _writecsv(os.path.join(tdir, "endpoint_augmented_secondary_results.csv"),
              [{"held_out_regime": r, "n_endpoint_estimable_val": ho[r]["endpoint_secondary"].get("n_endpoint_estimable_val"),
                "loto_auc": ho[r]["endpoint_secondary"].get("loto_auc"),
                "permutation_p": ho[r]["endpoint_secondary"].get("permutation_p"),
                "passes": ho[r]["endpoint_secondary"].get("passes"), "is_secondary": True} for r in H],
              ["held_out_regime", "n_endpoint_estimable_val", "loto_auc", "permutation_p", "passes", "is_secondary"])
    _writecsv(os.path.join(tdir, "external_dataset_candidate_matrix.csv"),
              [{"dataset": "BNCI2014_001 (this study)", "role": "internal_development", "status": "used", "note": "not external"},
               {"dataset": "BNCI2014_004", "role": "external_candidate", "status": "BARRED_pending_explicit_approval",
                "note": "requires a new protocol + approval; not unbarred by C20"},
               {"dataset": "other MOABB MI cohorts", "role": "external_candidate", "status": "protocol_only",
                "note": "specify in C20_EXTERNAL_DATASET_PROTOCOL before any execution"}],
              ["dataset", "role", "status", "note"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"),
              [{"check": k, "pass": v} for k, v in res["gates"].items()], ["check", "pass"])
    _writecsv(os.path.join(tdir, "c20_case_taxonomy.csv"),
              [{"case_label": res["taxonomy"]["case_label"], "next_science": res["taxonomy"]["next_science"]}],
              ["case_label", "next_science"])


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    t = res["taxonomy"]; ho = res["held_out"]; g = res["gates"]
    L = [f"# C20 — Frozen-probe new-regime validation (locked to C19 `{res['config_hash']}`)", "",
         "> Validates the FROZEN C19 robust-core diagnostic probe by cross-regime LOTO: train on development "
         "regimes S0/S2/S3 (non-held-out targets), evaluate on the held-out target in DELETION regimes "
         "S4/S5/S6/S7. Nothing about the probe changed. DIAGNOSTIC-ONLY; no selector.", "",
         f"- **CASE: `{t['case_label']}`**  ·  next: {t['next_science']}",
         f"- locked C19 config hash: `{g['locked_c19_config_hash']}` · feature-lock: {g['feature_lock']} · "
         f"dev/held-out disjoint: {g['development_heldout_disjoint']} · no-selector: {g['no_selector_artifact']}", "",
         "## Cross-regime LOTO (frozen probe, held-out deletion regimes)", "",
         "| held-out regime | deletion | n_used | loto_auc | perm_p | margin vs 0.5 | passes |",
         "|---|:--:|---:|---:|---:|---:|:--:|"]
    for r in schema.HELD_OUT_REGIMES:
        p = ho[r]["primary"]
        L.append(f"| {r} | {r in schema.DELETION_HELD_OUT} | {p.get('n_used')} | {_f(p.get('loto_auc'))} | "
                 f"{_f(p.get('permutation_p'))} | {_f(p.get('margin_vs_chance'))} | {p.get('passes')} |")
    L += ["", "> Success = beats cross-regime permutation p<0.05 AND margin vs STRICT chance 0.5 >= 0.03 "
          "(stricter than C19, which used margin-vs-empirical-null; C20 must clear real chance).", "",
          "## Feature availability / abstention per held-out regime (first-class output)", "",
          "| held-out regime | robust-core scored_rate | endpoint_nonestimable_rate |", "|---|---:|---:|"]
    for r in schema.HELD_OUT_REGIMES:
        a = res["availability"][r]
        L.append(f"| {r} | {_f(a['robust_core_scored_rate'])} | {_f(a['endpoint_nonestimable_rate'])} |")
    L += ["", "## Endpoint-augmented (SECONDARY — cannot rescue primary)", "",
          "| held-out regime | n_endpoint_estimable | loto_auc | passes |", "|---|---:|---:|:--:|"]
    for r in schema.HELD_OUT_REGIMES:
        e = ho[r]["endpoint_secondary"]
        L.append(f"| {r} | {e.get('n_endpoint_estimable_val')} | {_f(e.get('loto_auc'))} | {e.get('passes')} |")
    L += ["", "## Boundary of the claim", "",
          "> DIAGNOSTIC-ONLY. A held-out pass would read only as: the frozen low-freedom source-only diagnostic "
          "probe generalizes across held-out support-stress regimes. It is NOT a deployment-ready target-free "
          "chooser (none produced), and endpoint-augmented results can never rescue a failed robust-core primary."]
    return "\n".join(L)


def render_protocol_md(res) -> str:
    return ("# C20-B — External-dataset validation PROTOCOL (document only; NO execution in C20)\n\n"
            "This is a protocol skeleton. C20 executes NO second dataset. BNCI2014_004 stays BARRED pending "
            "explicit future approval + a dedicated protocol; it is NOT unbarred here.\n\n"
            "## Questions the protocol must answer before any C21 execution\n"
            "- which dataset, and WHY it counts as external (different site/subjects/acquisition, not a re-split);\n"
            "- what splits (leakage-safe, recording-grouped), and which target labels are used POST HOC only;\n"
            "- exact frozen artifacts carried over (config hash `" + res["config_hash"] + "`, robust-core features);\n"
            "- required artifacts (identity gate, feature availability, abstention, permutation baseline);\n"
            "- failure modes (feature non-availability vs relationship absence) and a pre-registered STOP RULE;\n"
            "- forbidden: selector emission, target-metric selection, deployment language.\n\n"
            "## Sequencing\nC20-A (this) -> C20-B protocol (this) -> C21 external execution ONLY after approval.\n\n"
            "## Candidate matrix\nSee c20_tables/external_dataset_candidate_matrix.csv. BNCI2014_004 = "
            "BARRED_pending_explicit_approval.\n")


def render_claim_boundary_md(res) -> str:
    t = res["taxonomy"]
    return (f"# C20 — claim-boundary audit\n\n- CASE: `{t['case_label']}`\n\n"
            "## Allowed (per case)\n"
            "- generalizes: the frozen diagnostic probe generalizes across held-out support-stress regimes.\n"
            "- survives_nondeletion: robust to block/rare/nonestimable but not deletion; scope accordingly.\n"
            "- regime_local: C19 positive is regime-local; report the boundary.\n"
            "- availability_limited: held-out validity limited by observable AVAILABILITY, not the relationship.\n\n"
            "## Forbidden (guarded)\n" + "\n".join(f"- {s}" for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS)
            + f"\n\n- no_selector_artifact gate: {res['gates']['no_selector_artifact']}\n"
            + "- endpoint-augmented is SECONDARY and cannot rescue a failed robust-core primary.\n")


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        if s in low:
            raise ValueError(f"forbidden claim in C20 report: {s!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.probe_validation.report")
    ap.add_argument("--extract-dir", required=True)
    ap.add_argument("--c10-dir", required=True)
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--n-perm", type=int, default=schema.N_PERM)
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args(argv)
    res = run(args.extract_dir, args.c10_dir, n_perm=args.n_perm, n_workers=args.n_workers)
    md = render_md(res); _guard_forbidden(md)
    prot = render_protocol_md(res); _guard_forbidden(prot)
    cb = render_claim_boundary_md(res)
    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "C20_FROZEN_PROBE_NEW_REGIME_VALIDATION.md"), "w").write(md)
    json.dump(res, open(os.path.join(args.out_dir, "C20_FROZEN_PROBE_NEW_REGIME_VALIDATION.json"), "w"),
              indent=2, sort_keys=True, default=str)
    open(os.path.join(args.out_dir, "C20_EXTERNAL_DATASET_PROTOCOL.md"), "w").write(prot)
    open(os.path.join(args.out_dir, "C20_CLAIM_BOUNDARY_AUDIT.md"), "w").write(cb)
    write_tables(res, os.path.join(args.out_dir, "c20_tables"))
    t = res["taxonomy"]
    print(f"[C20] case={t['case_label']} held_out_pass={t['held_out_pass']} "
          f"locked_hash={res['config_hash']} no_selector={res['gates']['no_selector_artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
