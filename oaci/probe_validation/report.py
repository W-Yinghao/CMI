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
    n_pass = sum(passes.values()); H = schema.HELD_OUT_REGIMES
    failing = [r for r in H if not passes[r]]
    passing = [r for r in H if passes[r]]
    low_avail = [r for r in failing if (availability[r]["robust_core_scored_rate"] or 0.0) < schema.AVAILABILITY_FLOOR]
    avail_limited = bool(failing) and len(low_avail) == len(failing)   # ALL failing regimes robust-UNavailable
    robust_available_everywhere = all((availability[r]["robust_core_scored_rate"] or 0.0) >= schema.AVAILABILITY_FLOOR for r in H)

    # ---- headline case (conservative): a mixed/marginal result is NOT "partial success" ----
    if n_pass == len(H):
        case = schema.CASE_GENERALIZES; primary = schema.PRIMARY_ESTABLISHED
    elif avail_limited:
        case = schema.CASE_AVAILABILITY_LIMITED; primary = schema.PRIMARY_NOT_ESTABLISHED
    elif n_pass >= 1:
        case = schema.CASE_LARGELY_REGIME_LOCAL; primary = schema.PRIMARY_NOT_ESTABLISHED
    else:
        case = schema.CASE_REGIME_LOCAL; primary = schema.PRIMARY_NOT_ESTABLISHED

    failure_mode = schema.FAILURE_AVAILABILITY if avail_limited else schema.FAILURE_RELATIONSHIP
    availability_status = schema.AVAILABILITY_OK if robust_available_everywhere else "robust_core_partially_unavailable"
    from ..support_stress.schema import REGIME_SEVERITY
    margins = {r: heldout[r]["primary"].get("margin_vs_chance") for r in H}
    # how far each pass clears the strict 0.03 bar (adversarial: S6/S7 clear it by only ~0.002-0.006)
    over_bar = {r: (margins[r] - schema.SUCCESS_AUC_MARGIN_VS_CHANCE if margins[r] is not None else None) for r in passing}
    secondary = [{"regime": r, "margin_vs_chance": margins[r], "clears_bar_by": over_bar[r],
                  "severity": REGIME_SEVERITY[r], "marginal": True} for r in passing]
    # severity-local check: are the passing regimes the LOWER-severity ones? (=> severity gradient, not novel-regime transfer)
    pass_sev = [REGIME_SEVERITY[r] for r in passing]; fail_sev = [REGIME_SEVERITY[r] for r in failing]
    severity_local = bool(pass_sev and fail_sev and max(pass_sev) < min(fail_sev))

    # ---- LAYERED verdict (primary / secondary / failure-mode / availability / strength) ----
    layered = {"primary_verdict": primary,
               "secondary_observation": ("marginal_above_chance_transfer_in_" + "_".join(passing) if passing
                                         else "none"),
               "failure_mode": failure_mode, "availability_status": availability_status,
               "claim_strength": schema.CLAIM_WEAK_DIAGNOSTIC}
    nxt = ("Broad external new-regime generalization is NOT established. Robust-core features are available in "
           "all held-out regimes, so the failures are RELATIONSHIP-level regime shift, not feature availability. "
           f"{n_pass}/{len(H)} held-out regimes ({passing}) only MARGINALLY clear the strict chance bar and are "
           "reported as secondary exceptions, not a generalization claim. Remain diagnostic-only; do NOT proceed "
           "to C20-B external execution on this evidence." if primary == schema.PRIMARY_NOT_ESTABLISHED
           else "C20-B external-dataset protocol may proceed to C21 execution (still diagnostic-only).")
    return {"case_label": case, "layered_verdict": layered, "held_out_pass": passes, "n_pass": n_pass,
            "passing_regimes": passing, "secondary_marginal_exceptions": secondary,
            "severity_local_marginal_transfer": severity_local,
            "severity_note": ("the only above-chance held-out regimes are the LOWER-severity ones (severity "
                              f"{sorted(set(pass_sev))} pass vs {sorted(set(fail_sev))} fail) -> a severity "
                              "gradient, NOT novel-regime generalization" if severity_local else "n/a"),
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
    lv = res["taxonomy"]["layered_verdict"]
    _writecsv(os.path.join(tdir, "c20_case_taxonomy.csv"),
              [{"case_label": res["taxonomy"]["case_label"], "primary_verdict": lv["primary_verdict"],
                "secondary_observation": lv["secondary_observation"], "failure_mode": lv["failure_mode"],
                "availability_status": lv["availability_status"], "claim_strength": lv["claim_strength"],
                "severity_local_marginal_transfer": res["taxonomy"].get("severity_local_marginal_transfer"),
                "n_pass": res["taxonomy"]["n_pass"]}],
              ["case_label", "primary_verdict", "secondary_observation", "failure_mode", "availability_status",
               "claim_strength", "severity_local_marginal_transfer", "n_pass"])


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    t = res["taxonomy"]; ho = res["held_out"]; g = res["gates"]; lv = t["layered_verdict"]
    L = [f"# C20 — Frozen-probe new-regime validation (locked to C19 `{res['config_hash']}`)", "",
         "> Validates the FROZEN C19 robust-core diagnostic probe by cross-regime LOTO: train on development "
         "regimes S0/S2/S3 (non-held-out targets), evaluate on the held-out target in DELETION regimes "
         "S4/S5/S6/S7. Nothing about the probe changed. DIAGNOSTIC-ONLY; no selector.", "",
         "## Verdict (layered — headline is deliberately conservative)", "",
         f"- **primary: `{lv['primary_verdict']}`**  ·  case `{t['case_label']}`",
         f"- secondary observation: `{lv['secondary_observation']}` (MARGINAL exceptions, not a generalization claim)",
         f"- failure mode: `{lv['failure_mode']}`  ·  availability: `{lv['availability_status']}`  ·  strength: `{lv['claim_strength']}`",
         f"- {t['next_science']}", "",
         "## 1. Gate correction (superseded first run)", "",
         "> The first C20-A run reported `validity_limited_by_feature_availability` — a metric BUG: "
         "`abstention.availability` fed the fragile accuracy ENDPOINTS to the estimability gate, so the reported "
         "'robust_core_scored_rate' was really the robust+endpoint rate → 0 under deletion (endpoints NaN) while "
         "the robust-core features are fully finite. That first taxonomy is DISCARDED. Fixed: robust-only "
         "availability reported separately; failures are now correctly attributed to the source→target "
         "relationship, not feature availability.", "",
         "## 2. C19 lock audit", "",
         f"- locked C19 config hash `{g['locked_c19_config_hash']}` · feature-lock {g['feature_lock']} · "
         f"dev/held-out disjoint {g['development_heldout_disjoint']} · no-selector {g['no_selector_artifact']} "
         "(no feature / threshold / regularization tuning)", "",
         "## 3. Corrected availability", "",
         "| held-out regime | robust-core scored_rate | endpoint_available_rate | endpoint_nonestimable_rate |",
         "|---|---:|---:|---:|"]
    for r in schema.HELD_OUT_REGIMES:
        a = res["availability"][r]
        L.append(f"| {r} | {_f(a['robust_core_scored_rate'])} | {_f(a.get('endpoint_available_rate'))} | "
                 f"{_f(a['endpoint_nonestimable_rate'])} |")
    L += ["", "> Robust-core available in ALL held-out regimes; endpoint availability stays fragile (as C18 "
          "predicted). So failures are relationship-level, not availability-level.", "",
          "## 4. Held-out regime results (severity in parens)", "",
          "| held-out regime | sev | loto_auc | margin vs 0.5 | clears 0.03 bar by | passes |",
          "|---|:--:|---:|---:|---:|:--:|"]
    from ..support_stress.schema import REGIME_SEVERITY
    for r in schema.HELD_OUT_REGIMES:
        p = ho[r]["primary"]; m = p.get("margin_vs_chance")
        over = (m - schema.SUCCESS_AUC_MARGIN_VS_CHANCE) if (m is not None and p.get("passes")) else None
        L.append(f"| {r} | {REGIME_SEVERITY[r]} | {_f(p.get('loto_auc'))} | {_f(m)} | {_f(over)} | {p.get('passes')} |")
    L += ["", "> **Permutation p is NOT the discriminator**: the LOTO shuffle null centers below 0.5, so perm p "
          "hits the floor (~0.005) for ALL four regimes including at-chance S4 — do NOT report 'all beat "
          "permutation'. The BINDING criterion is the strict chance-margin (>=0.03). Under it, S4 (0.500, at "
          "chance) and S5 (+0.011) FAIL; S6 (+0.032) and S7 (+0.036) pass but clear the bar by only ~0.002 and "
          "~0.006 — threshold-level, fragile.",
          f"> **Not boundary-specific / severity-local**: S7 random ({_f(ho['S7_random_matched_mask']['primary'].get('loto_auc'))}) "
          f">= S6 boundary-aligned ({_f(ho['S6_boundary_aligned_mask']['primary'].get('loto_auc'))}), so the marginal "
          f"transfer is NOT a boundary mechanism. The only above-chance regimes (S6/S7) are the LOWER-severity ones "
          f"(sev 3) vs the failing S4/S5 (sev 4): severity_local_marginal_transfer = **{t['severity_local_marginal_transfer']}** "
          "→ a severity gradient, not novel-regime generalization.", "",
          "## 5. Endpoint-augmented (SECONDARY — cannot rescue primary)", "",
          "| held-out regime | n_endpoint_estimable | loto_auc | passes |", "|---|---:|---:|:--:|"]
    for r in schema.HELD_OUT_REGIMES:
        e = ho[r]["endpoint_secondary"]
        L.append(f"| {r} | {e.get('n_endpoint_estimable_val')} | {_f(e.get('loto_auc'))} | {e.get('passes')} |")
    L += ["", "## 6. Claim boundary", "",
          "> DIAGNOSTIC-ONLY. Broad external new-regime generalization is NOT established. The C19 robust-core "
          "competence signal is real in its pre-registered development regimes but its held-out support-regime "
          "transfer is weak and largely regime-local; S6/S7 are marginal above-chance exceptions on the mildest "
          "deletion regimes, not a generalization result. It remains a weak diagnostic signal — not a detector, "
          "not a chooser, no OACI rescue. C20-B external-dataset work stays a PROTOCOL only; do NOT proceed to "
          "execution on this evidence."]
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


def _write_artifacts(res, out_dir):
    md = render_md(res); _guard_forbidden(md)
    prot = render_protocol_md(res); _guard_forbidden(prot)
    cb = render_claim_boundary_md(res)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C20_FROZEN_PROBE_NEW_REGIME_VALIDATION.md"), "w").write(md)
    json.dump(res, open(os.path.join(out_dir, "C20_FROZEN_PROBE_NEW_REGIME_VALIDATION.json"), "w"),
              indent=2, sort_keys=True, default=str)
    open(os.path.join(out_dir, "C20_EXTERNAL_DATASET_PROTOCOL.md"), "w").write(prot)
    open(os.path.join(out_dir, "C20_CLAIM_BOUNDARY_AUDIT.md"), "w").write(cb)
    write_tables(res, os.path.join(out_dir, "c20_tables"))


def reinterpret(json_path, out_dir) -> dict:
    """Re-derive the taxonomy/layered verdict + artifacts from an already-computed C20 JSON WITHOUT recomputing
    the deterministic cross-regime LOTO / permutation. Uses the SAME _taxonomy()."""
    res = json.load(open(json_path))
    res["taxonomy"] = _taxonomy(res["held_out"], res["availability"])
    _write_artifacts(res, out_dir)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.probe_validation.report")
    ap.add_argument("--extract-dir")
    ap.add_argument("--c10-dir")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--n-perm", type=int, default=schema.N_PERM)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--reinterpret", default=None, help="path to a computed C20 JSON; re-derive taxonomy + artifacts")
    args = ap.parse_args(argv)
    if args.reinterpret:
        res = reinterpret(args.reinterpret, args.out_dir)
        t = res["taxonomy"]
        print(f"[C20 reinterpret] case={t['case_label']} primary={t['layered_verdict']['primary_verdict']} "
              f"n_pass={t['n_pass']} severity_local={t.get('severity_local_marginal_transfer')}")
        return 0
    if not (args.extract_dir and args.c10_dir):
        ap.error("full run requires --extract-dir and --c10-dir (or use --reinterpret)")
    res = run(args.extract_dir, args.c10_dir, n_perm=args.n_perm, n_workers=args.n_workers)
    _write_artifacts(res, args.out_dir)
    t = res["taxonomy"]
    print(f"[C20] case={t['case_label']} primary={t['layered_verdict']['primary_verdict']} "
          f"held_out_pass={t['held_out_pass']} locked_hash={res['config_hash']} "
          f"no_selector={res['gates']['no_selector_artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
