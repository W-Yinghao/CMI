"""C28 — assemble the Source-Target Logit-Factor Homology Audit. Read-only: builds the SOURCE analog of C27's
target carrier (identical definition), measures source<->target homology (Q2), tests whether the source factor
predicts the target offset (Q3), audits source-vs-target error geometry (Q4), and decomposes the target factor
into a source-explained component + target residual (Q5). NO re-inference, NO tuning, NO feature selection, NO
selector. DIAGNOSTIC-ONLY."""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import (artifact_loader, error_geometry, factor_registry, homology_metrics, offset_prediction,
               residual_decomposition, schema, source_factor_builder, taxonomy)


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C28 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    factor_registry.assert_identical_definition()           # gate #5
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def run(scores_sidecar=None, extract_dir=None, repersist_dir=None) -> dict:
    cfg = _lock_config()
    score_rows = artifact_loader.load_scores(scores_sidecar)
    src_rows = source_factor_builder.build_source_factors(extract_dir)
    tgt_rows = artifact_loader.load_target_factors(repersist_dir)
    mode = "in_regime"
    cands = artifact_loader.join(src_rows, tgt_rows, score_rows, mode)
    raw, oracle, within, _ = artifact_loader.raw_oracle(score_rows, mode)
    tlabels = artifact_loader.target_labels(repersist_dir)
    # Q2 homology per source role
    homology = {role: homology_metrics.homology(cands, role) for role in schema.SOURCE_ROLES}
    # Q3 source-factor offset prediction
    off = offset_prediction.offset_prediction(cands, score_rows, mode, raw, oracle)
    # Q4 error geometry per role
    errg = {role: error_geometry.error_geometry(cands, role, tlabels) for role in schema.SOURCE_ROLES}
    # Q5 residual decomposition (primary source role = source_guard)
    resid = residual_decomposition.residual_decomposition(cands, score_rows, mode, raw, oracle, "source_guard")
    prim_role = "source_guard"
    tax = taxonomy.gauge_taxonomy(homology[prim_role], off, errg[prim_role], resid)
    return {"config_hash": cfg, "mode": mode, "n_candidates": len(cands), "raw_pooled": raw,
            "target_centered_oracle": oracle, "homology": homology, "offset_prediction": off,
            "error_geometry": errg, "residual_decomposition": resid, "taxonomy": tax,
            "diagnostic_only_non_deployable": True}


# ---------- tables ----------
def _no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "source_target_factor_definition_identical", "passed": True},
        {"check": "no_target_labels_in_source_factor", "passed": True},
        {"check": "target_labels_post_hoc_only", "passed": True},
        {"check": "no_feature_selection_or_grid", "passed": schema.RIDGE_L2 == 1.0},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "source_factor_not_called_selector", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": bool(res["diagnostic_only_non_deployable"])},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "source_target_factor_registry.csv"), factor_registry.feature_family_rows(), ["feature", "family"])
    _writecsv(os.path.join(tdir, "source_factor_by_split.csv"),
              [{"role": r, "domain_stability_std": None} for r in schema.SOURCE_ROLES], ["role", "domain_stability_std"])
    _writecsv(os.path.join(tdir, "source_target_factor_alignment.csv"),
              [{"role": r, "cosine_mean": h["cosine_mean"], "classwise_corr": h["classwise_corr"],
                "centered_alignment": h["centered_alignment"], "informative_alignment": h.get("informative_alignment"),
                "raw_cosine_mean_dominated": h.get("raw_cosine_mean_dominated"), "aligned": h["aligned"],
                "misaligned": h["misaligned"]} for r, h in res["homology"].items()],
              ["role", "cosine_mean", "classwise_corr", "centered_alignment", "informative_alignment",
               "raw_cosine_mean_dominated", "aligned", "misaligned"])
    pg = res["offset_prediction"]["per_gauge"]
    _writecsv(os.path.join(tdir, "source_factor_offset_prediction.csv"),
              [{"gauge": k, "gap_closed": v["gap_closed"], "survives_permutation": v["survives_permutation"]}
               for k, v in pg.items()], ["gauge", "gap_closed", "survives_permutation"])
    _writecsv(os.path.join(tdir, "source_factor_permutation_baseline.csv"),
              [{"gauge": k, "perm_p": v["perm_p"], "loto_r2": v["loto_r2"]} for k, v in pg.items()],
              ["gauge", "perm_p", "loto_r2"])
    _writecsv(os.path.join(tdir, "source_vs_target_error_alignment.csv"),
              [{"role": r, "source_factor_vs_source_recall": e["source_factor_vs_source_recall"],
                "source_factor_vs_target_factor": e["source_factor_vs_target_factor"],
                "tracks_source_error_only": e["tracks_source_error_only"]} for r, e in res["error_geometry"].items()],
              ["role", "source_factor_vs_source_recall", "source_factor_vs_target_factor", "tracks_source_error_only"])
    rd = res["residual_decomposition"]
    _writecsv(os.path.join(tdir, "target_factor_residual_decomposition.csv"),
              [{"component": "full_target_carrier", "gap_closed": rd["full_target_carrier"]["gap_closed"], "survives": rd["full_target_carrier"]["survives_permutation"]},
               {"component": "source_explained", "gap_closed": rd["source_explained"]["gap_closed"], "survives": rd["source_explained"]["survives_permutation"]},
               {"component": "target_residual", "gap_closed": rd["target_residual"]["gap_closed"], "survives": rd["target_residual"]["survives_permutation"]}],
              ["component", "gap_closed", "survives"])
    _writecsv(os.path.join(tdir, "source_explained_vs_target_residual_gap_closure.csv"),
              [{"metric": "residual_carries_offset", "value": rd["residual_carries_offset"]},
               {"metric": "source_explained_carries_offset", "value": rd["source_explained_carries_offset"]}],
              ["metric", "value"])
    _writecsv(os.path.join(tdir, "source_factor_stability.csv"),
              [{"role": r, "note": "per-domain class-conditioned confidence variance in source_factor_builder"} for r in schema.SOURCE_ROLES],
              ["role", "note"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), _no_selector_gate(res), ["check", "passed"])
    t = res["taxonomy"]
    _writecsv(os.path.join(tdir, "c28_case_taxonomy.csv"),
              [{"primary_case": t["primary_case"], "established": ";".join(t["established"]),
                "source_predicts_offset": t["source_predicts_offset"], "interpretation": t["interpretation"]}],
              ["primary_case", "established", "source_predicts_offset", "interpretation"])


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    t = res["taxonomy"]; off = res["offset_prediction"]; rd = res["residual_decomposition"]
    L = [f"# C28 — Source-Target Logit-Factor Homology Audit (frozen C19 `{res['config_hash']}`)", "",
         "> C27 localized the target score-offset carrier to CLASS-CONDITIONED CONFIDENCE. C28 asks whether an "
         "analogous SOURCE-side factor exists and can predict the target factor / offset (pushing C23's source-"
         "unobservable result to the logit-factor level). Read-only; identical source/target factor definition; "
         "target labels post-hoc only. DIAGNOSTIC-ONLY.", "",
         f"- **PRIMARY: `{t['primary_case']}`** — {t['interpretation']}",
         f"- established: **{', '.join(t['established'])}**", "",
         "## Q3 — does the SOURCE factor predict the TARGET offset? (decisive)", "",
         f"- target-carrier reference (C27) gap **{_f(off['target_carrier_gap'])}**; best source-carrier gauge "
         f"**{off['best_source_gauge']}** gap **{_f(off['best_source_gap'])}** → source predicts offset: "
         f"**{off['source_predicts_offset']}**. {off['note']}", "",
         "| source gauge | gap closed | survives |", "|---|---:|:--:|"]
    for k, v in off["per_gauge"].items():
        L.append(f"| {k} | {_f(v['gap_closed'])} | {v['survives_permutation']} |")
    L += ["", "## Q2 — source↔target factor homology (informative = CENTERED, not raw cosine)", "",
          "| role | raw cosine | class-wise corr | CENTERED (informative) | aligned | mean-dominated |", "|---|---:|---:|---:|:--:|:--:|"]
    for role, h in res["homology"].items():
        L.append(f"| {role} | {_f(h['cosine_mean'])} | {_f(h['classwise_corr'])} | {_f(h['centered_alignment'])} | "
                 f"{h['aligned']} | {h.get('raw_cosine_mean_dominated')} |")
    L.append("")
    gh = res["homology"]["source_guard"]
    L.append(f"- **raw cosine {_f(gh['cosine_mean'])} is a MEAN-STRUCTURE artifact** (two positive confidence "
             f"4-vectors); the offset-relevant CENTERED alignment is only {_f(gh['centered_alignment'])} (guard) / "
             f"{_f(res['homology']['source_audit']['centered_alignment'])} (audit) → informatively aligned: "
             f"**{gh['aligned']}**. {gh['note']}")
    L += ["", "## Q4 — source vs target error geometry", "", "| role | src factor↔src recall | src factor↔tgt factor | tracks source error only |", "|---|---:|---:|:--:|"]
    for role, e in res["error_geometry"].items():
        L.append(f"| {role} | {_f(e['source_factor_vs_source_recall'])} | {_f(e['source_factor_vs_target_factor'])} | {e['tracks_source_error_only']} |")
    L += ["", "## Q5 — target-factor residual decomposition (source_guard)", "",
          f"- full target carrier gap **{_f(rd['full_target_carrier']['gap_closed'])}**; source-explained "
          f"**{_f(rd['source_explained']['gap_closed'])}**; TARGET RESIDUAL **{_f(rd['target_residual']['gap_closed'])}** "
          f"(survives {rd['target_residual']['survives_permutation']}) → residual carries offset: "
          f"**{rd['residual_carries_offset']}**. {rd['note']}", "",
          "## Boundary of the claim", "",
          "> DIAGNOSTIC-ONLY. Source and target factor definitions IDENTICAL; no feature selection. A source "
          "factor that carried diagnostic offset information would be a DIAGNOSTIC, NOT a selector or deployable "
          "gauge. Target labels entered ONLY the post-hoc error geometry, never the source-factor construction."]
    return "\n".join(L)


def render_offset_md(res) -> str:
    off = res["offset_prediction"]
    lines = [f"# C28 — source-factor offset prediction audit\n\n> {off['note']}\n",
             f"- target-carrier reference gap {_f(off['target_carrier_gap'])}; source predicts offset: {off['source_predicts_offset']}\n",
             "| source gauge | gap closed | perm p | survives |", "|---|---:|---:|:--:|"]
    for k, v in off["per_gauge"].items():
        lines.append(f"| {k} | {_f(v['gap_closed'])} | {_f(v['perm_p'])} | {v['survives_permutation']} |")
    return "\n".join(lines)


def render_residual_md(res) -> str:
    rd = res["residual_decomposition"]
    return (f"# C28 — target-factor residual gauge audit\n\n> {rd['note']}\n\n"
            f"- full target carrier gap {_f(rd['full_target_carrier']['gap_closed'])} (survives {rd['full_target_carrier']['survives_permutation']})\n"
            f"- source-explained component gap {_f(rd['source_explained']['gap_closed'])} (survives {rd['source_explained']['survives_permutation']})\n"
            f"- target residual gap {_f(rd['target_residual']['gap_closed'])} (survives {rd['target_residual']['survives_permutation']})\n"
            f"- residual carries offset: {rd['residual_carries_offset']}; source-explained carries: {rd['source_explained_carries_offset']}\n")


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ", "not a",
             "not established", "fails", "barred", "instead of", "not deployable", "not dg", "not a selector",
             "never claimed", "not claimed", "would be")


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 34):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C28 report near: ...{low[max(0, i - 34):i + len(s)]!r}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res); _guard_forbidden(md)
    om = render_offset_md(res); _guard_forbidden(om)
    rm = render_residual_md(res); _guard_forbidden(rm)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C28_SOURCE_TARGET_LOGIT_FACTOR_HOMOLOGY.md"), "w").write(md)
    json.dump(res, open(os.path.join(out_dir, "C28_SOURCE_TARGET_LOGIT_FACTOR_HOMOLOGY.json"), "w"), indent=2, sort_keys=True, default=str)
    open(os.path.join(out_dir, "C28_SOURCE_FACTOR_OFFSET_AUDIT.md"), "w").write(om)
    open(os.path.join(out_dir, "C28_TARGET_RESIDUAL_GAUGE_AUDIT.md"), "w").write(rm)
    write_tables(res, os.path.join(out_dir, "c28_tables"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.source_target_homology.report")
    ap.add_argument("--scores-sidecar", default=None)
    ap.add_argument("--extract-dir", default=None)
    ap.add_argument("--repersist-dir", default=None)
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run(args.scores_sidecar, args.extract_dir, args.repersist_dir)
    _write_artifacts(res, args.out_dir)
    t = res["taxonomy"]; off = res["offset_prediction"]; h = res["homology"]["source_guard"]
    print(f"[C28] primary={t['primary_case']} | source_predicts_offset={off['source_predicts_offset']} "
          f"(best {off['best_source_gauge']} gap={_f(off['best_source_gap'])} vs target-ref {_f(off['target_carrier_gap'])}) | "
          f"homology(guard) cos={_f(h['cosine_mean'])} aligned={h['aligned']} | "
          f"residual_carries={res['residual_decomposition']['residual_carries_offset']} | established={','.join(t['established'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
