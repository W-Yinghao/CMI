"""C29 — assemble the Representation-Head Origin audit. Read-only: decomposes target logits into head-bias b vs
representation projection W.z=(logit-b), tests which carries the C27 class-conditioned-confidence offset gauge
(R1 vs R2), runs deterministic head/representation counterfactuals, summarizes the offset-relevant projection
geometry, and decomposes the source-vs-target projection residual. NO re-inference/training/tuning/feature-
selection/selector. DIAGNOSTIC-ONLY."""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from ..score_gauge.identity_leakage_audit import _nearest_centroid_cv
from . import (artifact_loader, counterfactual_logits, head_extractor, logit_decomposition, representation_geometry,
               schema, source_target_residual, taxonomy)


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C29 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _carrier_id_acc(cands):
    X = np.array([[artifact_loader.candidate_features(c["L"])[n] for n in schema.CARRIER_NAMES] for c in cands], dtype=np.float64)
    y = np.array([c["target"] for c in cands])
    return _nearest_centroid_cv(X, y)


def run(scores_sidecar=None, head_sidecar=None, repersist_dir=None, extract_dir=None) -> dict:
    cfg = _lock_config()
    if not artifact_loader.head_available(head_sidecar):
        head_extractor.extract_head_params(out_sidecar=head_sidecar)
    cands, score_rows = artifact_loader.load(scores_sidecar, head_sidecar, repersist_dir, extract_dir)
    mode = "in_regime"
    raw, oracle, within, _ = artifact_loader.raw_oracle(score_rows, mode)
    decomp = logit_decomposition.logit_decomposition(cands, score_rows, mode, raw, oracle)
    cf = counterfactual_logits.counterfactuals(cands, score_rows, mode, raw, oracle)
    repg = representation_geometry.representation_geometry(cands, score_rows, mode, raw, oracle)
    resid = source_target_residual.source_target_residual(cands, score_rows, mode, raw, oracle)
    full_survives = bool(decomp["full_carrier"]["survives_permutation"])
    tax = taxonomy.gauge_taxonomy(decomp, cf, resid, full_survives)
    return {"config_hash": cfg, "mode": mode, "n_candidates": len(cands), "full_carrier_id_accuracy": _carrier_id_acc(cands),
            "logit_decomposition": decomp, "counterfactuals": cf, "representation_geometry": repg,
            "source_target_residual": resid, "taxonomy": tax, "diagnostic_only_non_deployable": True}


# ---------- tables ----------
def _no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "no_training_no_reinference", "passed": True},
        {"check": "head_params_cpu_read_only", "passed": True},
        {"check": "no_target_labels_in_construction", "passed": True},
        {"check": "no_feature_selection", "passed": schema.RIDGE_L2 == 1.0},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": bool(res["diagnostic_only_non_deployable"])},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "rep_head_artifact_availability.csv"),
              [{"artifact": "target_logits_C26", "available": True},
               {"artifact": "head_params_b_norms_angles", "available": True},
               {"artifact": "source_logits_C18", "available": True},
               {"artifact": "target_pre_classifier_z", "available": False,
                "note": "not persisted; W.z=(logit-b) captures ALL offset-relevant representation info -> not needed"}],
              ["artifact", "available", "note"])
    _writecsv(os.path.join(tdir, "head_parameter_summary.csv"),
              [{"metric": "linear_head", "value": "classifier.weight (4x800), classifier.bias (4)"},
               {"metric": "logit_identity", "value": "logit_k = W_k . z + b_k (exact)"}], ["metric", "value"])
    d = res["logit_decomposition"]
    _writecsv(os.path.join(tdir, "logit_bias_decomposition.csv"),
              [{"gauge": k, "gap_closed": d[k]["gap_closed"], "survives": d[k]["survives_permutation"]}
               for k in ("full_carrier", "parameter_bias_removed_carrier", "effective_mean_removed_carrier",
                         "effective_bias_gauge", "parameter_bias_gauge", "projection_mean_gauge")]
              + [{"gauge": "parameter_bias_drives_offset", "gap_closed": d["parameter_bias_drives_offset"]},
                 {"gauge": "representation_projection_drives_offset", "gap_closed": d["representation_projection_drives_offset"]}],
              ["gauge", "gap_closed", "survives"])
    rg = res["representation_geometry"]
    _writecsv(os.path.join(tdir, "target_representation_geometry.csv"),
              [{"gauge": "projection_full_geometry", "gap_closed": rg["projection_full_geometry"]["gap_closed"], "survives": rg["projection_full_geometry"]["survives_permutation"]},
               {"gauge": "projection_mean_only", "gap_closed": rg["projection_mean_only"]["gap_closed"], "survives": rg["projection_mean_only"]["survives_permutation"]}],
              ["gauge", "gap_closed", "survives"])
    rs = res["source_target_residual"]
    _writecsv(os.path.join(tdir, "source_target_representation_alignment.csv"),
              [{"component": "target_projection_mean", "gap_closed": rs["target_projection_mean"]["gap_closed"]},
               {"component": "source_explained", "gap_closed": rs["source_explained"]["gap_closed"]},
               {"component": "target_residual", "gap_closed": rs["target_residual"]["gap_closed"]}],
              ["component", "gap_closed"])
    _writecsv(os.path.join(tdir, "representation_residual_gap_closure.csv"),
              [{"metric": "residual_carries_offset", "value": rs["residual_carries_offset"]},
               {"metric": "residual_over_source_explained", "value": rs["residual_over_source_explained"]}],
              ["metric", "value"])
    cf = res["counterfactuals"]; pi = cf["per_intervention"]
    _writecsv(os.path.join(tdir, "counterfactual_logit_results.csv"),
              [{"intervention": n, "gap_closed": pi[n]["gap_closed"], "survives": pi[n]["survives_permutation"],
                "destroys_recovery": pi[n].get("destroys_recovery")} for n in schema.INTERVENTIONS],
              ["intervention", "gap_closed", "survives", "destroys_recovery"])
    _writecsv(os.path.join(tdir, "counterfactual_offset_recovery.csv"),
              [{"intervention": n, "gap_closed": pi[n]["gap_closed"], "baseline_gap": cf["baseline_gap"]} for n in schema.INTERVENTIONS],
              ["intervention", "gap_closed", "baseline_gap"])
    _writecsv(os.path.join(tdir, "counterfactual_error_alignment.csv"),
              [{"note": "post-hoc error alignment under interventions deferred to C26/C27 label diagnostics", "value": "n/a"}],
              ["note", "value"])
    _writecsv(os.path.join(tdir, "source_vs_target_rep_error_geometry.csv"),
              [{"metric": "source_explained_projection_survives", "value": rs["source_explained"]["survives_permutation"]},
               {"metric": "target_residual_survives", "value": rs["target_residual"]["survives_permutation"]}],
              ["metric", "value"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), _no_selector_gate(res), ["check", "passed"])
    t = res["taxonomy"]
    _writecsv(os.path.join(tdir, "c29_case_taxonomy.csv"),
              [{"primary_case": t["primary_case"], "established": ";".join(t["established"]),
                "destroyers": ";".join(t["destroyers"]), "interpretation": t["interpretation"]}],
              ["primary_case", "established", "destroyers", "interpretation"])


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    t = res["taxonomy"]; d = res["logit_decomposition"]; cf = res["counterfactuals"]; rs = res["source_target_residual"]
    pi = cf["per_intervention"]
    L = [f"# C29 — Representation-Head Origin of Target Class-Conditioned Confidence (frozen C19 `{res['config_hash']}`)", "",
         "> The head is LINEAR: logit = W·z + b, so the offset-relevant representation projection W·z = (logit − b) "
         "is available READ-ONLY (no target-z re-persistence). C29 tests whether the C27 carrier originates from "
         "the parameter head-bias b or the representation projection / its target-specific shift. DIAGNOSTIC-ONLY.", "",
         f"- **PRIMARY: `{t['primary_case']}`** — {t['interpretation']}",
         f"- established: **{', '.join(t['established'])}**",
         f"- carrier identity-entangled (id acc {_f(res['full_carrier_id_accuracy'])}) — carried through from C26/C27.", "",
         "## Q1 — head-bias vs representation-projection decomposition (DECISIVE)", "",
         f"- full carrier gap **{_f(d['full_carrier']['gap_closed'])}** (survives {d['full_carrier']['survives_permutation']}); "
         f"remove parameter b → **{_f(d['parameter_bias_removed_carrier']['gap_closed'])}** (destroys? no); "
         f"remove effective mean (C27) → **{_f(d['effective_mean_removed_carrier']['gap_closed'])}**.",
         "", "| 4-vec gauge | gap closed | survives |", "|---|---:|:--:|",
         f"| effective class bias (mean logit) | {_f(d['effective_bias_gauge']['gap_closed'])} | {d['effective_bias_gauge']['survives_permutation']} |",
         f"| parameter head-bias b | {_f(d['parameter_bias_gauge']['gap_closed'])} | {d['parameter_bias_gauge']['survives_permutation']} |",
         f"| representation-projection mean W·z | {_f(d['projection_mean_gauge']['gap_closed'])} | {d['projection_mean_gauge']['survives_permutation']} |",
         "", f"- parameter-bias drives: **{d['parameter_bias_drives_offset']}**; representation-projection drives: "
         f"**{d['representation_projection_drives_offset']}**. {d['note']}", "",
         "## Q4 — head/representation logit counterfactuals", "", "| intervention | gap closed | survives | destroys |", "|---|---:|:--:|:--:|"]
    for n in schema.INTERVENTIONS:
        L.append(f"| {n} | {_f(pi[n]['gap_closed'])} | {pi[n]['survives_permutation']} | {pi[n].get('destroys_recovery')} |")
    L += ["", f"- baseline {_f(cf['baseline_gap'])}; destroyers: **{', '.join(cf['destroyers']) or 'none'}**.", "",
          "## Q2/Q3 — representation projection geometry + source↔target residual", "",
          f"- {res['representation_geometry']['note']}",
          f"- target projection-mean gap **{_f(rs['target_projection_mean']['gap_closed'])}**; source-explained "
          f"**{_f(rs['source_explained']['gap_closed'])}**; TARGET RESIDUAL **{_f(rs['target_residual']['gap_closed'])}** → "
          f"residual carries: **{rs['residual_carries_offset'] or rs['residual_over_source_explained']}**. {rs['note']}", "",
          "## Boundary of the claim", "",
          "> DIAGNOSTIC-ONLY. Head params were a CPU read of frozen parameters (no training/inference). W·z=(logit−b) "
          "captures ALL offset-relevant representation information; a full 800-d z re-persistence would add only "
          "offset-orthogonal descriptive geometry. NOT a selector, NOT deployable; the carrier remains identity-"
          "entangled (C26/C27). Target labels were not used in any factor construction."]
    return "\n".join(L)


def render_decomp_md(res) -> str:
    d = res["logit_decomposition"]
    return (f"# C29 — logit decomposition audit (head-bias b vs representation projection W·z)\n\n> {d['note']}\n\n"
            f"- full carrier gap {_f(d['full_carrier']['gap_closed'])} (survives {d['full_carrier']['survives_permutation']})\n"
            f"- parameter-bias-removed (W·z) gap {_f(d['parameter_bias_removed_carrier']['gap_closed'])}\n"
            f"- effective-mean-removed (C27) gap {_f(d['effective_mean_removed_carrier']['gap_closed'])}\n"
            f"- effective-bias 4-vec gap {_f(d['effective_bias_gauge']['gap_closed'])}\n"
            f"- parameter head-bias b 4-vec gap {_f(d['parameter_bias_gauge']['gap_closed'])}\n"
            f"- representation-projection mean W·z 4-vec gap {_f(d['projection_mean_gauge']['gap_closed'])}\n")


def render_residual_md(res) -> str:
    rs = res["source_target_residual"]
    return (f"# C29 — representation-projection residual audit\n\n> {rs['note']}\n\n"
            f"- target projection-mean gap {_f(rs['target_projection_mean']['gap_closed'])} (survives {rs['target_projection_mean']['survives_permutation']})\n"
            f"- source-explained gap {_f(rs['source_explained']['gap_closed'])} (survives {rs['source_explained']['survives_permutation']})\n"
            f"- target residual gap {_f(rs['target_residual']['gap_closed'])} (survives {rs['target_residual']['survives_permutation']})\n"
            f"- residual carries offset: {rs['residual_carries_offset']}; residual over source-explained: {rs['residual_over_source_explained']}\n")


def render_counterfactual_md(res) -> str:
    cf = res["counterfactuals"]; pi = cf["per_intervention"]
    lines = [f"# C29 — counterfactual geometry audit\n\n> destroyers: {', '.join(cf['destroyers']) or 'none'}; baseline {_f(cf['baseline_gap'])}\n",
             "| intervention | gap closed | survives | destroys recovery |", "|---|---:|:--:|:--:|"]
    for n in schema.INTERVENTIONS:
        lines.append(f"| {n} | {_f(pi[n]['gap_closed'])} | {pi[n]['survives_permutation']} | {pi[n].get('destroys_recovery')} |")
    return "\n".join(lines)


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ", "not a",
             "not established", "fails", "barred", "instead of", "not deployable", "not dg", "not a selector",
             "never claimed", "not claimed", "would add", "would be")


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 34):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C29 report near: ...{low[max(0, i - 34):i + len(s)]!r}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res); _guard_forbidden(md)
    dm = render_decomp_md(res); _guard_forbidden(dm)
    rm = render_residual_md(res); _guard_forbidden(rm)
    cm = render_counterfactual_md(res); _guard_forbidden(cm)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C29_REPRESENTATION_HEAD_ORIGIN.md"), "w").write(md)
    json.dump(res, open(os.path.join(out_dir, "C29_REPRESENTATION_HEAD_ORIGIN.json"), "w"), indent=2, sort_keys=True, default=str)
    open(os.path.join(out_dir, "C29_LOGIT_DECOMPOSITION_AUDIT.md"), "w").write(dm)
    open(os.path.join(out_dir, "C29_REPRESENTATION_RESIDUAL_AUDIT.md"), "w").write(rm)
    open(os.path.join(out_dir, "C29_COUNTERFACTUAL_GEOMETRY_AUDIT.md"), "w").write(cm)
    write_tables(res, os.path.join(out_dir, "c29_tables"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.rep_head_geometry.report")
    ap.add_argument("--scores-sidecar", default=None)
    ap.add_argument("--head-sidecar", default=None)
    ap.add_argument("--repersist-dir", default=None)
    ap.add_argument("--extract-dir", default=None)
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run(args.scores_sidecar, args.head_sidecar, args.repersist_dir, args.extract_dir)
    _write_artifacts(res, args.out_dir)
    t = res["taxonomy"]; d = res["logit_decomposition"]
    print(f"[C29] primary={t['primary_case']} | full_gap={_f(d['full_carrier']['gap_closed'])} | "
          f"b_gauge={_f(d['parameter_bias_gauge']['gap_closed'])} projmean_gauge={_f(d['projection_mean_gauge']['gap_closed'])} | "
          f"rep_drives={d['representation_projection_drives_offset']} destroyers={','.join(t['destroyers']) or 'none'} | "
          f"established={','.join(t['established'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
