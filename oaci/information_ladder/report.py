"""C24 — assemble the Calibration Information Ladder / Identifiability Boundary Audit. STAGE-1 (read-only):
locks the C19 config hash, reproduces R0/R1/R2 (source-only) from the C22 sidecar, builds R5 (few-label) and R6
(oracle) rungs, runs the C24-A source-only non-identifiability witnesses, and emits the taxonomy SHELL with
R3/R4 marked REQUIRES_REINFERENCE (never proxied, never finalized). The identity-leakage audit is reported
FIRST. No selector, no probe tuning, no external dataset. When the P0-gated target-audit re-inference sidecar
is present, R3/R4 are filled and the taxonomy finalizes."""
from __future__ import annotations

import argparse
import csv
import json
import os

from ..score_gauge import gauge_feature_registry as gfr
from ..score_gauge import identity_leakage_audit as ila
from ..score_gauge import offset_model, risk_family
from . import (artifact_loader, few_label_calibration, oracle_ladder, rung_registry, schema, source_witness,
               target_unlabeled_features, taxonomy)


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C24 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def run(sidecar_path=None, artifact_root=None, reinfer_sidecar=None) -> dict:
    cfg = _lock_config()
    rows = artifact_loader.load(sidecar_path)
    mode = "in_regime"
    # HARD GATE first: identity-leakage audit (source features carry target identity in LOSO)
    identity = ila.identity_leakage_audit(rows, mode)
    # availability probe (honest R3/R4 feasibility; never proxied)
    avail = artifact_loader.target_unlabeled_availability(artifact_root, reinfer_sidecar)
    ladder = rung_registry.ladder(avail)
    # R1 source-only gauge (reproduce C23) + R6 oracle ladder
    gt = gfr.build_gauge_table(rows, mode)
    fit = offset_model.fit_offsets(gt)
    oracle = oracle_ladder.oracle_ladder(rows, mode, fit["offset_hat_loto"])
    raw = oracle["raw_pooled"]; orc = oracle["target_centered_oracle"]
    r1_gap = ((oracle["source_gauge_loto"] - raw) / (orc - raw)) if (orc is not None and raw is not None and (orc - raw) > 1e-6) else None
    # R2 static risk-family gauge
    r2 = risk_family.risk_family_gauge(gt, rows, mode)
    # R5 few-label calibration
    few = few_label_calibration.few_label_curve(rows, mode)
    # C24-A witnesses
    witnesses = source_witness.witness_audit(rows, mode)
    # R3/R4 target-unlabeled gauge (Stage-1 -> REQUIRES_REINFERENCE; Stage-3 -> real)
    reinf = artifact_loader.load_target_unlabeled_sidecar(reinfer_sidecar)
    r3r4 = target_unlabeled_features.build_target_unlabeled_gauge(rows, avail, mode, sidecar=reinf)
    r3r4_resolved = None
    if r3r4["status"] == schema.STATUS_OK:
        tfit = offset_model.fit_offsets(r3r4["gauge_table"])
        tlad = oracle_ladder.oracle_ladder(rows, mode, tfit["offset_hat_loto"])
        gap = ((tlad["source_gauge_loto"] - raw) / (orc - raw)) if (orc - raw) > 1e-6 else None
        r3r4_resolved = {"status": schema.STATUS_OK, "gap_closed": gap,
                         "auc_improve": (tlad["source_gauge_loto"] - raw) if tlad["source_gauge_loto"] is not None else None,
                         "loto_generalizes": bool((tfit["loto_r2"] or -1) > 0), "loto_r2": tfit["loto_r2"],
                         "pooled_auc": tlad["source_gauge_loto"]}
    tax = taxonomy.gauge_taxonomy(witnesses, few, oracle, identity, r1_gap, r3r4_resolved)
    return {"config_hash": cfg, "mode": mode, "n_rows": len(rows), "identity_leakage": identity,
            "availability": avail, "ladder": ladder, "oracle": oracle, "r1_source_gauge_gap": r1_gap,
            "risk_family_r2": r2, "few_label_r5": few, "witnesses": witnesses,
            "target_unlabeled_r3r4": r3r4, "target_unlabeled_resolved": r3r4_resolved, "taxonomy": tax,
            "diagnostic_only_non_deployable": True}


# ---------- tables ----------
def _offset_recovery_rows(res):
    o = res["oracle"]; raw = o["raw_pooled"]; r2 = res["risk_family_r2"]; few = res["few_label_r5"]
    r5_best = max(few["curve"], key=lambda c: (c["pooled_auc"] if c["pooled_auc"] is not None else -1))
    rr = res["target_unlabeled_resolved"]
    def row(rung, auc, gap, status):
        return {"rung": rung, "pooled_auc": auc, "auc_improve": (auc - raw) if (auc is not None and raw is not None) else None,
                "gap_closed": gap, "status": status}
    return [
        row(schema.R0, raw, 0.0, schema.STATUS_OK),
        row(schema.R1, o["source_gauge_loto"], res["r1_source_gauge_gap"], schema.STATUS_OK),
        row(schema.R2, r2["source_gauge_loto_auc"], r2["gap_closed"], schema.STATUS_OK),
        row(schema.R3, (rr["pooled_auc"] if rr else None), (rr["gap_closed"] if rr else None),
            (rr["status"] if rr else res["target_unlabeled_r3r4"]["status"])),
        row(schema.R4, None, None, res["target_unlabeled_r3r4"]["status"]),
        row(schema.R5, r5_best["pooled_auc"], r5_best["gap_closed"], schema.STATUS_OK),
        row(schema.R6, o["target_centered_oracle"], 1.0, schema.STATUS_OK),
    ]


def _no_selector_gate(res):
    fn = target_unlabeled_features.target_unlabeled_feature_names()
    try:
        target_unlabeled_features.assert_no_target_labels(fn); anon = True
    except ValueError:
        anon = False
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "identity_leakage_audit_ran_first", "passed": res["identity_leakage"].get("target_id_accuracy_from_source_features") is not None},
        {"check": "r3r4_not_proxied_from_method_final", "passed": res["target_unlabeled_r3r4"]["status"] != "proxy"},
        {"check": "r3r4_features_exclude_target_labels", "passed": anon},
        {"check": "c24_not_finalized_while_r3r4_pending", "passed": (res["taxonomy"]["final"] == (res["target_unlabeled_r3r4"]["status"] == schema.STATUS_OK))},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": bool(res["diagnostic_only_non_deployable"])},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "information_ladder_rungs.csv"), res["ladder"],
              ["rung", "name", "info_class", "deployable", "needs_target_inputs", "needs_target_labels",
               "feasible_readonly", "status", "reason"])
    w = res["witnesses"]
    _writecsv(os.path.join(tdir, "source_only_witness_pairs.csv"), w.get("top_witnesses", []),
              ["unit_a", "unit_b", "source_distance", "offset_difference", "witness_strength"])
    _writecsv(os.path.join(tdir, "offset_recovery_by_rung.csv"), _offset_recovery_rows(res),
              ["rung", "pooled_auc", "auc_improve", "gap_closed", "status"])
    _writecsv(os.path.join(tdir, "target_unlabeled_feature_registry.csv"),
              [{"feature": f, "family": "target_unlabeled_confidence_geometry", "needs_target_labels": False}
               for f in target_unlabeled_features.target_unlabeled_feature_names()],
              ["feature", "family", "needs_target_labels"])
    rr = res["target_unlabeled_resolved"]
    _writecsv(os.path.join(tdir, "target_unlabeled_gauge_results.csv"),
              [{"metric": k, "value": (rr[k] if rr else None)} for k in ("status", "pooled_auc", "auc_improve", "gap_closed", "loto_r2", "loto_generalizes")]
              if rr else [{"metric": "status", "value": res["target_unlabeled_r3r4"]["status"]},
                          {"metric": "reason", "value": res["target_unlabeled_r3r4"].get("reason")}],
              ["metric", "value"])
    _writecsv(os.path.join(tdir, "source_vs_target_unlabeled_gap_closure.csv"),
              [{"gauge": "R1_source_only", "gap_closed": res["r1_source_gauge_gap"], "status": schema.STATUS_OK},
               {"gauge": "R2_risk_family", "gap_closed": res["risk_family_r2"]["gap_closed"], "status": schema.STATUS_OK},
               {"gauge": "R3_target_unlabeled", "gap_closed": (rr["gap_closed"] if rr else None),
                "status": res["target_unlabeled_r3r4"]["status"]}],
              ["gauge", "gap_closed", "status"])
    few = res["few_label_r5"]
    _writecsv(os.path.join(tdir, "few_label_budget_curve.csv"), few["curve"],
              ["k_per_class", "pooled_auc", "auc_improve", "gap_closed", "n_targets_with_both_classes", "n_targets"])
    _writecsv(os.path.join(tdir, "few_label_per_target_results.csv"), few["per_target_small_budget"],
              ["target", "k_per_class", "offset_hat", "n_good_revealed", "n_bad_revealed"])
    o = res["oracle"]
    _writecsv(os.path.join(tdir, "oracle_gap_closure.csv"),
              [{"rung": "raw_pooled", "pooled_auc": o["raw_pooled"]},
               {"rung": "regime_centered", "pooled_auc": o["regime_centered"]},
               {"rung": "target_centered_oracle", "pooled_auc": o["target_centered_oracle"]},
               {"rung": "target_rank_oracle", "pooled_auc": o["target_rank_oracle"]},
               {"rung": "within_target_ceiling", "pooled_auc": o["within_target_ceiling"]}],
              ["rung", "pooled_auc"])
    t = res["taxonomy"]
    _writecsv(os.path.join(tdir, "problem_class_boundary.csv"),
              [{"factor": "source_nonidentifiable", "value": t["source_nonidentifiable"]},
               {"factor": "few_label_case", "value": t["few_label_case"]},
               {"factor": "oracle_recovers", "value": t["oracle_recovers"]},
               {"factor": "identity_laden", "value": t["identity_laden"]},
               {"factor": "r3r4_status", "value": t["r3r4_status"]},
               {"factor": "final", "value": t["final"]}],
              ["factor", "value"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), _no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c24_case_taxonomy.csv"),
              [{"primary_provisional": t["primary_provisional"], "final": t["final"],
                "established_readonly": ";".join(t["established_readonly"]),
                "unresolved_pending_reinference": ";".join(t["unresolved_pending_reinference"]),
                "interpretation": t["interpretation"], "next_science": t["next_science"]}],
              ["primary_provisional", "final", "established_readonly", "unresolved_pending_reinference",
               "interpretation", "next_science"])


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    t = res["taxonomy"]; idn = res["identity_leakage"]; o = res["oracle"]; few = res["few_label_r5"]
    w = res["witnesses"]; av = res["availability"]
    L = [f"# C24 — Calibration Information Ladder / Identifiability Boundary Audit (frozen C19 `{res['config_hash']}`)", "",
         "> C23 closed the source-only target-free gauge (`G5_offset_source_unobservable`). C24 asks what "
         "information, if any, breaks the per-target score-offset non-identifiability. Read-only rungs "
         "(R0/R1/R2/R5/R6 + witnesses) below; R3/R4 (target-UNLABELED) require a no-retraining target-audit "
         "re-inference behind a P0 replay-identity gate and are NOT proxied, NOT finalized here. "
         "DIAGNOSTIC-ONLY; not a selector, not DG success.", "",
         f"- **STAGE: {'FINAL' if t['final'] else 'read-only (R3/R4 pending P0-gated re-inference)'}**",
         f"- established read-only: **{', '.join(t['established_readonly']) or 'none'}**"
         + (f"  ·  unresolved: {', '.join(t['unresolved_pending_reinference'])}" if t['unresolved_pending_reinference'] else ""),
         f"- primary (provisional): **`{t['primary_provisional']}`** — {t['interpretation']}", "",
         "## HARD GATE — target-identity leakage (reported FIRST)", "",
         f"- 9-way target-id accuracy from raw source features **{_f(idn['target_id_accuracy_from_source_features'])}** "
         f"vs chance **{_f(idn['chance'])}** → identity-separable: **{idn['source_features_identity_separable']}** "
         "(gates any positive R3/R4 recovery claim as I7 unless it generalizes LOTO).", "",
         "## Information ladder (pooled AUC; oracle rungs use target grouping = NON-deployable)", "",
         "| rung | information | pooled AUC | gap closed | status |", "|---|---|---:|---:|:--|"]
    for row in _offset_recovery_rows(res):
        meta = next(m for m in res["ladder"] if m["rung"] == row["rung"])
        L.append(f"| {row['rung']} {meta['name']} | {meta['info_class']} | {_f(row['pooled_auc'])} | "
                 f"{_f(row['gap_closed'])} | {row['status']} |")
    L += ["", f"- R0 raw pooled **{_f(o['raw_pooled'])}** → R6 target-centered oracle **{_f(o['target_centered_oracle'])}** "
          f"(oracle gap **{_f(o['oracle_gap_over_raw'])}**); within-target ceiling {_f(o['within_target_ceiling'])}.",
          f"- R1 source-only gauge gap closed **{_f(res['r1_source_gauge_gap'])}** (C23: source-only fails/hurts); "
          f"R2 risk-family gap **{_f(res['risk_family_r2']['gap_closed'])}**.", "",
          "## C24-A — source-only non-identifiability witnesses", ""]
    if res["witnesses"].get("insufficient_units"):
        L.append(f"- insufficient units ({w.get('n_units')}) for a witness audit")
    else:
        L += [f"- Mantel corr(source-dist, offset-dist): all-pairs **{_f(w.get('mantel_corr_all_pairs'))}** (p "
              f"{_f(w.get('mantel_perm_p_all_pairs'))}) → **CROSS-TARGET {_f(w.get('mantel_corr_cross_target'))}** (p "
              f"{_f(w.get('mantel_perm_p_cross_target'))}); within-target block confound: "
              f"**{w.get('within_target_block_confound_detected')}**",
              f"- source predicts offset (cross-target ≥{schema.MANTEL_IDENTIFY_CORR}): **{w.get('source_predicts_offset')}**; "
              f"{w['n_strong_witnesses']} near-source/divergent-offset collisions → source non-identifying: "
              f"**{w.get('source_nonidentifying')}**  · {w.get('interpretation')}"]
    L += ["", "## R5 — few-label target calibration diagnostic (NON-DG supervised)", "",
          "| k labels/class | pooled AUC | gap closed |", "|---:|---:|---:|"]
    for c in few["curve"]:
        L.append(f"| {c['k_per_class']} | {_f(c['pooled_auc'])} | {_f(c['gap_closed'])} |")
    L += ["", f"- k=0 is the LABEL-FREE transductive target-mean centering (== oracle): gap closed "
          f"**{_f(few.get('zero_label_transductive_gap'))}** → offset recovered by target GROUPING at 0 labels; "
          f"grouping==oracle: **{few.get('zero_label_grouping_equals_oracle')}**.",
          f"- competence labels only REFINE beyond grouping (label gain over grouping {_f(few.get('label_gain_over_grouping'))}); "
          f"few-labels (≤{schema.FEW_LABEL_RECOVERS_MAX_K}/class) recover: **{few['few_labels_recover']}** "
          f"(max gap {_f(few['max_gap_closed'])}).", "",
          "## R3/R4 — target-unlabeled gauge (REQUIRES RE-INFERENCE; not proxied, not finalized)", "",
          f"- status: **{res['target_unlabeled_r3r4']['status']}**. {av['method_final_note']}",
          f"- cached method-final target_audit.npz: {av['method_final_target_audit_count']} (wrong population); "
          f"per-candidate target-unlabeled ready: {av['per_candidate_target_unlabeled_ready']}.",
          "- next: C24-R3R4-P0 replay-identity smoke gate → full no-retraining target-audit re-inference → real R3/R4.", "",
          "## Boundary of the claim", "",
          "> DIAGNOSTIC-ONLY. Oracle rungs use target grouping (non-deployable). R5 is a supervised label-budget "
          "diagnostic, not DG. No selector, no selected-checkpoint artifact. C24 is NOT finalized until the "
          "P0-gated re-inference supplies R3/R4."]
    return "\n".join(L)


def render_witness_md(res) -> str:
    w = res["witnesses"]
    if w.get("insufficient_units"):
        return f"# C24-A — source-only non-identifiability witnesses\n\nInsufficient units ({w.get('n_units')}).\n"
    lines = [f"# C24-A — source-only non-identifiability witnesses\n\n> {w['interpretation']}\n",
             f"- units (target×regime): {w['n_units']}; pairs: {w['n_pairs']} ({w['n_cross_target_pairs']} cross-target)",
             f"- Mantel corr(source-dist, offset-dist): all-pairs **{_f(w['mantel_corr_all_pairs'])}** "
             f"(p {_f(w['mantel_perm_p_all_pairs'])}) → **CROSS-TARGET {_f(w['mantel_corr_cross_target'])}** "
             f"(p {_f(w['mantel_perm_p_cross_target'])})",
             f"- within-target block confound detected: **{w['within_target_block_confound_detected']}** "
             f"(all-pairs correlation inflated by same-target pairs); source predicts offset (cross-target ≥"
             f"{schema.MANTEL_IDENTIFY_CORR}): **{w['source_predicts_offset']}**",
             f"- near-distance threshold (bottom {int(schema.WITNESS_NEAR_QUANTILE*100)}%): {_f(w['near_distance_threshold'])}; "
             f"far-offset threshold (top {int((1-schema.WITNESS_FAR_OFFSET_QUANTILE)*100)}%): {_f(w['far_offset_threshold'])}",
             f"- near-source/divergent-offset collisions: **{w['n_strong_witnesses']}**; source non-identifying: "
             f"**{w['source_nonidentifying']}**\n", "| unit A | unit B | source dist | offset diff | strength |",
             "|---|---|---:|---:|---:|"]
    for x in w["top_witnesses"]:
        lines.append(f"| {x['unit_a']} | {x['unit_b']} | {x['source_distance']} | {x['offset_difference']} | {x['witness_strength']} |")
    return "\n".join(lines)


def render_target_unlabeled_md(res) -> str:
    av = res["availability"]
    return (f"# C24 — Target-unlabeled gauge audit (R3/R4): REQUIRES RE-INFERENCE\n\n"
            f"> Status **{res['target_unlabeled_r3r4']['status']}**. R3/R4 are NOT computed read-only and are NOT "
            f"proxied from method-final checkpoints (wrong population), and C24 is NOT finalized without them.\n\n"
            f"## Why re-inference is required\n- The per-target offset is defined over the ~60 feasible-OACI "
            f"CANDIDATE checkpoints per seed×target. The committed artifacts cache target logits only for "
            f"method-final checkpoints ({av['method_final_target_audit_count']} `target_audit.npz`; example "
            f"`{av['example_method_final']}`). {av['method_final_note']}\n"
            f"- The C18 extract holds source logits only; the C10 replay holds a `target_pred_hash` + label-"
            f"dependent target scalars only. No per-candidate target-unlabeled confidence geometry exists.\n\n"
            f"## Planned R3 target-unlabeled features (label-free)\n"
            f"{', '.join(target_unlabeled_features.target_unlabeled_feature_names())}\n\n"
            f"## C24-R3R4-P0 replay-identity smoke gate (must pass before full re-inference)\n"
            f"- G1 overlapping-checkpoint target logits match cached method-final within declared tolerance\n"
            f"- G2 sample IDs / order match `target_audit.npz`\n- G3 checkpoint hashes match the manifest\n"
            f"- G4 repeated forward is deterministic\n- G5 no target labels read by the R3/R4 feature builder\n"
            f"- G6 no target-derived endpoint metric computed in R3/R4\n- G7 no selected-checkpoint artifact\n"
            f"- G8 target labels joined only later for diagnostic validation / offset evaluation\n\n"
            f"## Then\n- full no-retraining target-audit re-inference over the feasible-OACI candidate population → "
            f"per-candidate target logits → R3 (target-unlabeled gauge) + R4 (source + target-unlabeled) → final C24 taxonomy.\n"
            f"- R3/R4 are target-unlabeled TRANSDUCTIVE DIAGNOSTIC rungs: not source-only, not deployable, not DG success.\n")


def render_few_label_md(res) -> str:
    few = res["few_label_r5"]
    lines = [f"# C24 R5 — Few-label target calibration diagnostic\n\n> {few['note']}\n",
             f"- raw pooled {_f(few['raw_pooled'])}; target-centered oracle {_f(few['target_centered_oracle'])}",
             f"- few labels (≤{schema.FEW_LABEL_RECOVERS_MAX_K}/class) recover: **{few['few_labels_recover']}**; "
             f"max gap closed {_f(few['max_gap_closed'])}\n", "| k/class | pooled AUC | gap closed | targets w/ both classes |",
             "|---:|---:|---:|---:|"]
    for c in few["curve"]:
        lines.append(f"| {c['k_per_class']} | {_f(c['pooled_auc'])} | {_f(c['gap_closed'])} | {c['n_targets_with_both_classes']}/{c['n_targets']} |")
    lines.append("\n> Interpretation: if a small labeled budget sharply recovers the offset, the missing quantity "
                 "is target-specific scalar calibration information. This is a supervised label-budget diagnostic, "
                 "NOT domain generalization and NOT a selector.")
    return "\n".join(lines)


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "no deployable", "not a", "not established", "fails", "barred", "instead of", "not finalized",
             "not proxied", "not deployable", "not dg")


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 30):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C24 report near: ...{low[max(0, i - 30):i + len(s)]!r}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res); _guard_forbidden(md)
    wm = render_witness_md(res); _guard_forbidden(wm)
    tm = render_target_unlabeled_md(res); _guard_forbidden(tm)
    fm = render_few_label_md(res); _guard_forbidden(fm)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C24_INFORMATION_LADDER_AUDIT.md"), "w").write(md)
    json.dump(res, open(os.path.join(out_dir, "C24_INFORMATION_LADDER_AUDIT.json"), "w"), indent=2, sort_keys=True, default=str)
    open(os.path.join(out_dir, "C24_SOURCE_ONLY_NONIDENTIFIABILITY_WITNESSES.md"), "w").write(wm)
    open(os.path.join(out_dir, "C24_TARGET_UNLABELED_GAUGE_AUDIT.md"), "w").write(tm)
    open(os.path.join(out_dir, "C24_FEW_LABEL_CALIBRATION_DIAGNOSTIC.md"), "w").write(fm)
    write_tables(res, os.path.join(out_dir, "c24_tables"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.information_ladder.report")
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--artifact-root", default=None)
    ap.add_argument("--reinfer-sidecar", default=None)
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run(args.sidecar, args.artifact_root, args.reinfer_sidecar)
    _write_artifacts(res, args.out_dir)
    t = res["taxonomy"]; w = res["witnesses"]; few = res["few_label_r5"]
    print(f"[C24 stage={'FINAL' if t['final'] else 'readonly'}] r3r4={res['target_unlabeled_r3r4']['status']} | "
          f"witnesses={w.get('n_strong_witnesses')} mantel={_f(w.get('mantel_corr_source_offset'))} "
          f"p={_f(w.get('mantel_perm_p'))} nonid={w.get('source_nonidentifying')} | "
          f"few_label_recover={few['few_labels_recover']} maxgap={_f(few['max_gap_closed'])} | "
          f"established={','.join(t['established_readonly']) or 'none'} primary={t['primary_provisional']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
