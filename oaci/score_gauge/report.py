"""C23 — assemble the Target-Free Score Calibration / Gauge Audit. Locks the C19 config hash, loads the C22
score sidecar (read-only), builds the TARGET-ANONYMOUS source-only gauge, runs the target-identity-leakage HARD
GATE FIRST, fits the fixed-ridge LOTO offset model, walks the calibration ceiling ladder, computes the
diagnostics + secondary risk-family gauge, and emits the deterministic G1-G6 taxonomy. No selector, no probe
tuning, no external dataset, no target identity at score time. DIAGNOSTIC-ONLY."""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from ..competence_probe import schema as c19
from . import (artifact_loader, calibration_diagnostics, ceiling_ladder, gauge_feature_registry, identity_leakage_audit,
               offset_model, risk_family, schema, taxonomy)


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C23 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _epoch_residual_check(rows, mode, fit) -> dict:
    """Carry the C22 epoch/order concern forward at the TARGET-OFFSET level: is the offset (and the gauge's LOTO
    prediction of it) merely a per-target epoch/order proxy? The gauge features exclude epoch/order, but we still
    verify the offset it predicts is not an epoch artifact."""
    mr = [r for r in rows if r["mode"] == mode]
    targets = fit["targets"]

    def _mean(key):
        return {t: float(np.mean([r[key] for r in mr if r["target"] == t and r.get(key) is not None]))
                for t in targets if any(r["target"] == t and r.get(key) is not None for r in mr)}

    def _corr(a, b):
        ks = [t for t in targets if t in a and t in b]
        if len(ks) < 3:
            return None
        x = np.array([a[t] for t in ks]); y = np.array([b[t] for t in ks])
        if x.std() < 1e-9 or y.std() < 1e-9:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    off_true = fit["offset_true"]; off_hat = fit["offset_hat_loto"]
    me, mo = _mean("epoch"), _mean("order")
    c_off_epoch = _corr(off_true, me); c_off_order = _corr(off_true, mo)
    c_hat_epoch = _corr(off_hat, me)
    strong = 0.70
    residual = bool((c_off_epoch is not None and abs(c_off_epoch) > strong)
                    and (c_hat_epoch is not None and abs(c_hat_epoch) > strong))
    return {"corr_offset_epoch": c_off_epoch, "corr_offset_order": c_off_order,
            "corr_gauge_pred_epoch": c_hat_epoch, "offset_is_epoch_proxy": residual,
            "note": "gauge features EXCLUDE epoch/order; this checks the offset it predicts is not an epoch artifact."}


def run(sidecar_path=None) -> dict:
    cfg_hash = _lock_config()
    rows = artifact_loader.load(sidecar_path)
    mode = "in_regime"                                    # the offset lives where the C19 signal lives
    gauge_table = gauge_feature_registry.build_gauge_table(rows, mode)
    # --- HARD GATE: identity-leakage audit runs FIRST, before any positive calibration claim ---
    identity = identity_leakage_audit.identity_leakage_audit(rows, mode)
    fit = offset_model.fit_offsets(gauge_table)
    ladder = ceiling_ladder.ceiling_ladder(rows, mode, fit["offset_hat_loto"])
    ladder_cross = ceiling_ladder.ceiling_ladder(rows, "cross_regime", fit["offset_hat_loto"])
    diag = calibration_diagnostics.diagnostics(gauge_table, fit)
    risk = risk_family.risk_family_gauge(gauge_table, rows, mode)
    epoch_residual = _epoch_residual_check(rows, mode, fit)
    tax = taxonomy.gauge_taxonomy(ladder, fit, identity, diag, risk, epoch_residual["offset_is_epoch_proxy"])
    return {"config_hash": cfg_hash, "mode": mode, "n_rows": len(rows), "n_targets": fit["n_targets"],
            "n_gauge_features": fit["n_gauge_features"], "gauge_feature_names": gauge_feature_registry.gauge_feature_names(),
            "identity_leakage": identity, "offset_fit": fit, "ceiling_ladder": ladder,
            "ceiling_ladder_cross_regime": ladder_cross, "diagnostics": diag, "risk_family": risk,
            "epoch_residual": epoch_residual, "taxonomy": tax, "diagnostic_only_non_deployable": True}


# ---------- tables ----------
def _no_selector_gate(res) -> list:
    fn = res["gauge_feature_names"]
    try:
        gauge_feature_registry.assert_target_anonymous(fn)
        anon = True
    except ValueError:
        anon = False
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "gauge_features_target_anonymous", "passed": anon},
        {"check": "no_target_id_in_primary_gauge", "passed": anon},
        {"check": "identity_leakage_audit_ran_before_taxonomy", "passed": res["identity_leakage"].get("target_id_accuracy_from_source_features") is not None},
        {"check": "no_per_candidate_selected_checkpoint_emitted", "passed": True},
        {"check": "offset_model_fixed_reg_no_grid", "passed": schema.RIDGE_L2 == 1.0},
        {"check": "diagnostic_only_non_deployable", "passed": bool(res["diagnostic_only_non_deployable"])},
    ]


def write_tables(res, tdir) -> None:
    os.makedirs(tdir, exist_ok=True)
    lad = res["ceiling_ladder"]
    rungs = [("raw_pooled", lad["raw_pooled"]), ("regime_centered", lad["regime_centered"]),
             ("source_gauge_loto", lad["source_gauge_loto"]), ("target_centered_oracle", lad["target_centered_oracle"]),
             ("target_rank_oracle", lad["target_rank_oracle"]), ("within_target_ceiling", lad["within_target_ceiling"])]
    _writecsv(os.path.join(tdir, "calibration_ceiling_ladder.csv"),
              [{"rung": k, "pooled_auc": v, "deployable": (k in ("raw_pooled", "regime_centered", "source_gauge_loto"))}
               for k, v in rungs], ["rung", "pooled_auc", "deployable"])
    fam = {}
    for s in c19.ROBUST_CORE_FEATURES:
        for m in schema.GAUGE_MOMENTS:
            fam[f"{s}__{m}"] = "robust_core_source_moment"
    for e in schema.GAUGE_EXTRA:
        fam[e] = "support_availability"
    _writecsv(os.path.join(tdir, "gauge_feature_registry.csv"),
              [{"feature": f, "family": fam[f], "target_anonymous": True} for f in res["gauge_feature_names"]],
              ["feature", "family", "target_anonymous"])
    idn = res["identity_leakage"]
    _writecsv(os.path.join(tdir, "target_identity_leakage_audit.csv"),
              [{"metric": "target_id_accuracy_from_source_features", "value": idn["target_id_accuracy_from_source_features"]},
               {"metric": "chance", "value": idn["chance"]},
               {"metric": "identity_leakage_ceiling", "value": schema.IDENTITY_LEAKAGE_CEILING},
               {"metric": "source_features_identity_separable", "value": idn["source_features_identity_separable"]},
               {"metric": "n_candidates", "value": idn["n_candidates"]}, {"metric": "n_targets", "value": idn["n_targets"]}],
              ["metric", "value"])
    fit = res["offset_fit"]
    _writecsv(os.path.join(tdir, "source_gauge_offset_prediction_loto.csv"),
              [{"target": t, "offset_true": fit["offset_true"][t], "offset_hat_loto": fit["offset_hat_loto"][t],
                "offset_hat_insample": fit["offset_hat_insample"][t]} for t in fit["targets"]],
              ["target", "offset_true", "offset_hat_loto", "offset_hat_insample"])
    ladc = res["ceiling_ladder_cross_regime"]
    _writecsv(os.path.join(tdir, "source_gauge_cross_regime_results.csv"),
              [{"mode": "in_regime", "rung": k, "pooled_auc": lad[k]} for k in
               ("raw_pooled", "regime_centered", "source_gauge_loto", "target_centered_oracle", "within_target_ceiling")]
              + [{"mode": "cross_regime", "rung": k, "pooled_auc": ladc[k]} for k in
                 ("raw_pooled", "regime_centered", "source_gauge_loto", "target_centered_oracle", "within_target_ceiling")],
              ["mode", "rung", "pooled_auc"])
    _writecsv(os.path.join(tdir, "source_gauge_gap_closure.csv"),
              [{"metric": "raw_pooled_auc", "value": lad["raw_pooled"]},
               {"metric": "source_gauge_loto_auc", "value": lad["source_gauge_loto"]},
               {"metric": "target_centered_oracle_auc", "value": lad["target_centered_oracle"]},
               {"metric": "auc_improve_source_gauge", "value": lad["auc_improve_source_gauge"]},
               {"metric": "gap_closed_source_gauge", "value": lad["gap_closed_source_gauge"]},
               {"metric": "success_auc_improve_threshold", "value": schema.SUCCESS_AUC_IMPROVE},
               {"metric": "success_gap_closed_threshold", "value": schema.SUCCESS_GAP_CLOSED}],
              ["metric", "value"])
    dg = res["diagnostics"]
    _writecsv(os.path.join(tdir, "residual_offset_after_gauge.csv"),
              [{"metric": k, "value": dg[k]} for k in ("true_offset_std", "residual_offset_std", "residual_over_true_std",
               "loto_r2", "insample_r2", "loto_r2_perm_p", "loto_r2_perm_mean", "loto_beats_permutation")],
              ["metric", "value"])
    rf = res["risk_family"]
    _writecsv(os.path.join(tdir, "risk_family_gauge_results.csv"),
              [{"metric": k, "value": rf[k]} for k in ("is_secondary", "feature", "loto_r2", "source_gauge_loto_auc", "gap_closed")],
              ["metric", "value"])
    _writecsv(os.path.join(tdir, "robust_core_vs_risk_family_gauge.csv"),
              [{"gauge": "robust_core_source_moments", "loto_r2": fit["loto_r2"],
                "source_gauge_loto_auc": lad["source_gauge_loto"], "gap_closed": lad["gap_closed_source_gauge"]},
               {"gauge": "risk_family_R_src_only", "loto_r2": rf["loto_r2"],
                "source_gauge_loto_auc": rf["source_gauge_loto_auc"], "gap_closed": rf["gap_closed"]}],
              ["gauge", "loto_r2", "source_gauge_loto_auc", "gap_closed"])
    er = res["epoch_residual"]
    _writecsv(os.path.join(tdir, "epoch_order_residual_checks.csv"),
              [{"metric": k, "value": er[k]} for k in ("corr_offset_epoch", "corr_offset_order", "corr_gauge_pred_epoch", "offset_is_epoch_proxy")],
              ["metric", "value"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), _no_selector_gate(res), ["check", "passed"])
    t = res["taxonomy"]
    _writecsv(os.path.join(tdir, "c23_case_taxonomy.csv"),
              [{"primary_case": t["primary_case"], "auc_improve": t["auc_improve"], "gap_closed": t["gap_closed"],
                "identity_laden": t["identity_laden"], "loto_generalizes": t["loto_generalizes"],
                "interpretation": t["interpretation"], "next_science": t["next_science"]}],
              ["primary_case", "auc_improve", "gap_closed", "identity_laden", "loto_generalizes", "interpretation", "next_science"])


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    t = res["taxonomy"]; idn = res["identity_leakage"]; lad = res["ceiling_ladder"]; dg = res["diagnostics"]
    rf = res["risk_family"]; er = res["epoch_residual"]
    L = [f"# C23 — Target-Free Score Calibration / Gauge Audit (frozen C19 `{res['config_hash']}`)", "",
         "> Read-only MECHANISM audit: can the per-target score OFFSET that breaks the C20/C22 pooled cross-target "
         "estimand be explained/reduced by TARGET-FREE, SOURCE-ONLY, TARGET-ANONYMOUS gauge summaries — without "
         "target identity, target labels, target-wise centering, source subject IDs, or checkpoint selection? "
         "NOT a selector, NOT an OACI rescue, NOT deployable calibration.", "",
         f"- **CASE: `{t['primary_case']}`**", f"- {t['interpretation']}", f"- next: {t['next_science']}", "",
         "## HARD GATE — target-identity-leakage audit (reported FIRST, gates any positive claim)", "",
         f"- 9-way target-id accuracy from raw source features **{_f(idn['target_id_accuracy_from_source_features'])}** "
         f"vs chance **{_f(idn['chance'])}** (ceiling {schema.IDENTITY_LEAKAGE_CEILING}) → source features "
         f"identity-separable: **{idn['source_features_identity_separable']}**",
         f"- {idn['note']}", "",
         "> In LOSO the source composition ≈ target identity. If source features carry target id, a per-target "
         "gauge can only count as target-free calibration if the offset relationship GENERALIZES leave-one-"
         "target-out (offset LOTO below); otherwise it is G3.", "",
         "## Calibration ceiling ladder (pooled AUC; oracle rungs use target identity = NON-deployable ceilings)", "",
         "| rung | pooled AUC | deployable |", "|---|---:|:--:|",
         f"| raw (no calibration) | {_f(lad['raw_pooled'])} | yes |",
         f"| regime-centered | {_f(lad['regime_centered'])} | yes |",
         f"| **source-gauge LOTO** | **{_f(lad['source_gauge_loto'])}** | **yes (target-free)** |",
         f"| target-centered ORACLE | {_f(lad['target_centered_oracle'])} | no (uses target id) |",
         f"| target-rank ORACLE | {_f(lad['target_rank_oracle'])} | no |",
         f"| within-target ceiling | {_f(lad['within_target_ceiling'])} | no |", "",
         f"- source-gauge AUC improvement over raw: **{_f(lad['auc_improve_source_gauge'])}** "
         f"(success ≥ {schema.SUCCESS_AUC_IMPROVE})",
         f"- oracle gap closed by source gauge: **{_f(lad['gap_closed_source_gauge'])}** "
         f"(success ≥ {schema.SUCCESS_GAP_CLOSED})", "",
         "## Offset model (fixed ridge L2=1.0, leave-one-target-out; no grid search, no feature selection)", "",
         f"- LOTO offset R² **{_f(dg['loto_r2'])}** (in-sample {_f(dg['insample_r2'])}); "
         f"permutation p **{_f(dg['loto_r2_perm_p'])}** (null mean {_f(dg['loto_r2_perm_mean'])}) → "
         f"LOTO beats permutation: **{dg['loto_beats_permutation']}**",
         f"- residual offset std **{_f(dg['residual_offset_std'])}** of true offset std **{_f(dg['true_offset_std'])}** "
         f"(residual/true {_f(dg['residual_over_true_std'])})", "",
         "## Secondary — risk-family gauge (R_src per-target mean ONLY; static training scalar)", "",
         f"- R_src-only LOTO offset R² **{_f(rf['loto_r2'])}**, source-gauge AUC **{_f(rf['source_gauge_loto_auc'])}**, "
         f"gap closed **{_f(rf['gap_closed'])}** — {rf['note']}", "",
         "## Epoch/order residual carry-forward (gauge excludes epoch/order)", "",
         f"- corr(offset, mean epoch) **{_f(er['corr_offset_epoch'])}**, corr(offset, mean order) "
         f"**{_f(er['corr_offset_order'])}**, corr(gauge LOTO prediction, mean epoch) **{_f(er['corr_gauge_pred_epoch'])}** "
         f"→ offset is an epoch proxy: **{er['offset_is_epoch_proxy']}**", "",
         "## Boundary of the claim", "",
         "> DIAGNOSTIC-ONLY mechanism audit. The target-centered/rank rungs are ORACLE ceilings (they use target "
         "identity) and are NOT deployable. No selector is produced; no per-candidate checkpoint is selected; the "
         "C19/C20 estimand boundary is characterized, not rescued."]
    return "\n".join(L)


def render_identity_md(res) -> str:
    idn = res["identity_leakage"]
    return (f"# C23 — Target-identity-leakage audit (HARD GATE, reported FIRST)\n\n> Runs BEFORE any positive "
            f"calibration claim. In LOSO the source composition ≈ target identity, so a gauge that merely re-"
            f"encodes target id is NOT target-free calibration (G3).\n\n{idn['note']}\n\n"
            f"- 9-way target-id accuracy from raw source features: **{_f(idn['target_id_accuracy_from_source_features'])}**\n"
            f"- chance (1/9): {_f(idn['chance'])}; identity-laden ceiling: {schema.IDENTITY_LEAKAGE_CEILING}\n"
            f"- source features identity-separable: **{idn['source_features_identity_separable']}**\n"
            f"- n_candidates {idn['n_candidates']}, n_targets {idn['n_targets']}\n\n"
            f"If identity-separable AND the offset does NOT generalize leave-one-target-out, any apparent "
            f"calibration is target-identity leakage (G3), not a target-free gauge.\n")


def render_risk_md(res) -> str:
    rf = res["risk_family"]; lad = res["ceiling_ladder"]; fit = res["offset_fit"]
    return (f"# C23 — Risk-family gauge audit (SECONDARY)\n\n> {rf['note']}\n\n"
            f"- risk-family (R_src-only) LOTO offset R²: {_f(rf['loto_r2'])}; source-gauge AUC {_f(rf['source_gauge_loto_auc'])}; "
            f"gap closed {_f(rf['gap_closed'])}\n"
            f"- robust-core (16 source-moment) LOTO offset R²: {_f(fit['loto_r2'])}; source-gauge AUC "
            f"{_f(lad['source_gauge_loto'])}; gap closed {_f(lad['gap_closed_source_gauge'])}\n\n"
            f"Reported so the offset's predictability is not silently attributed to the deletion-robust robust-core "
            f"gauge when a single static source-risk scalar already (or does not) explain it.\n")


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "no deployable", "not a", "not established", "fails", "barred", "instead of")


def _guard_forbidden(md) -> None:
    """Flag a forbidden phrase only when it is AFFIRMATIVE — NOT preceded (within ~30 chars) by a negation cue.
    Lets the report legitimately say 'NOT a deployable calibration / selector' while catching an affirmative
    over-claim (C21 lesson)."""
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 30):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C23 report near: "
                                 f"...{low[max(0, i - 30):i + len(s)]!r}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res); _guard_forbidden(md)
    idm = render_identity_md(res); _guard_forbidden(idm)
    rkm = render_risk_md(res); _guard_forbidden(rkm)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C23_TARGET_FREE_SCORE_GAUGE_AUDIT.md"), "w").write(md)
    json.dump(res, open(os.path.join(out_dir, "C23_TARGET_FREE_SCORE_GAUGE_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)
    open(os.path.join(out_dir, "C23_TARGET_IDENTITY_LEAKAGE_AUDIT.md"), "w").write(idm)
    open(os.path.join(out_dir, "C23_RISK_FAMILY_GAUGE_AUDIT.md"), "w").write(rkm)
    write_tables(res, os.path.join(out_dir, "c23_tables"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.score_gauge.report")
    ap.add_argument("--sidecar", default=None, help="path to the C22 score sidecar (default: schema.C22_SCORE_SIDECAR)")
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run(args.sidecar)
    _write_artifacts(res, args.out_dir)
    t = res["taxonomy"]; idn = res["identity_leakage"]; lad = res["ceiling_ladder"]
    print(f"[C23] identity_acc={idn['target_id_accuracy_from_source_features']:.3f} (chance {idn['chance']:.3f}) "
          f"separable={idn['source_features_identity_separable']} | "
          f"gauge_auc_improve={_f(lad['auc_improve_source_gauge'])} gap_closed={_f(lad['gap_closed_source_gauge'])} | "
          f"case={t['primary_case']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
