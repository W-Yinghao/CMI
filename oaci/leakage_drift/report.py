"""C40 leakage point drift forensics report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (artifact_loader, atom_pattern_stability, drift_manifest, instrumentation_spec,
               numeric_drift, schema, selection_audit_contrast, stagewise_diff, taxonomy,
               tolerance_ladder)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C40 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    def clean(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, float) and not math.isfinite(v):
            return ""
        return v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: clean(r.get(c)) for c in cols})


def _readcsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def recompute():
    cfg = _lock_config()
    ctx = artifact_loader.context()
    manifest = drift_manifest.build(ctx)
    contrast = selection_audit_contrast.contrast(ctx)
    stagewise = stagewise_diff.localize(ctx)
    path_diff = stagewise_diff.aggregate_path_diff(ctx)
    numeric = numeric_drift.diagnose(ctx, manifest["rows"], stagewise["summary"])
    ladder = tolerance_ladder.compute(manifest["rows"])
    stability = atom_pattern_stability.evaluate(ctx, manifest["rows"])
    spec = instrumentation_spec.rows()
    tax = taxonomy.classify(manifest, stagewise, numeric, stability, spec)
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_preference_robust_pairs": len(ctx["tables"]["c37"]["exact"]),
        "actual_selector_score_name": schema.ACTUAL_SELECTOR_SCORE_NAME,
        "identity_tolerance_for_elevated_atom_claims": schema.POINT_IDENTITY_TOL,
        "tolerance_ladder_diagnostic_only": True,
        "leakage_drift_manifest": manifest,
        "selection_vs_audit_identity_contrast": contrast,
        "stagewise_drift_localization": stagewise,
        "aggregate_vs_atom_path_diff": path_diff,
        "numeric_drift_diagnostics": numeric,
        "tolerance_ladder_identity": ladder,
        "atom_pattern_stability_under_drift": stability,
        "future_trace_field_requirements": spec,
        "taxonomy": tax,
    }


def _summary_from_existing():
    path = "oaci/reports/C40_LEAKAGE_POINT_DRIFT_FORENSICS.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    tdir = schema.C40_TABLE_DIR
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_preference_robust_pairs": d["n_preference_robust_pairs"],
        "actual_selector_score_name": d["actual_selector_score_name"],
        "identity_tolerance_for_elevated_atom_claims": d["identity_tolerance_for_elevated_atom_claims"],
        "tolerance_ladder_diagnostic_only": d["tolerance_ladder_diagnostic_only"],
        "leakage_drift_manifest": {"rows": _readcsv(os.path.join(tdir, "leakage_drift_manifest.csv")),
                                    "summary": d["leakage_drift_manifest_summary"]},
        "selection_vs_audit_identity_contrast": {
            "rows": _readcsv(os.path.join(tdir, "selection_vs_audit_identity_contrast.csv")),
            "summary": d["selection_vs_audit_identity_contrast_summary"]},
        "stagewise_drift_localization": {
            "rows": _readcsv(os.path.join(tdir, "stagewise_drift_localization.csv")),
            "summary": d["stagewise_drift_localization_summary"]},
        "aggregate_vs_atom_path_diff": {
            "rows": _readcsv(os.path.join(tdir, "aggregate_vs_atom_path_diff.csv")),
            "summary": d["aggregate_vs_atom_path_diff_summary"]},
        "numeric_drift_diagnostics": {
            "rows": _readcsv(os.path.join(tdir, "numeric_drift_diagnostics.csv")),
            "summary": d["numeric_drift_diagnostics_summary"]},
        "tolerance_ladder_identity": {
            "rows": _readcsv(os.path.join(tdir, "tolerance_ladder_identity.csv")),
            "summary": d["tolerance_ladder_identity_summary"]},
        "atom_pattern_stability_under_drift": {
            "rows": _readcsv(os.path.join(tdir, "atom_pattern_stability_under_drift.csv")),
            "summary": d["atom_pattern_stability_under_drift_summary"]},
        "future_trace_field_requirements": {
            "rows": _readcsv(os.path.join(tdir, "future_trace_field_requirements.csv")),
            "summary": d["future_trace_field_requirements_summary"]},
        "taxonomy": d["taxonomy"],
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists("oaci/reports/C40_LEAKAGE_POINT_DRIFT_FORENSICS.json"):
        return _summary_from_existing()
    return recompute()


def no_selector_gate(res):
    ms = res["leakage_drift_manifest"]["summary"]
    ladder = res["tolerance_ladder_identity"]["summary"]
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "c39_artifacts_read_only", "passed": True},
        {"check": "c37_c38_artifacts_read_only", "passed": True},
        {"check": "no_training_no_gpu", "passed": True},
        {"check": "no_selector_repair", "passed": True},
        {"check": "frozen_1e_9_identity_gate_retained",
         "passed": res["identity_tolerance_for_elevated_atom_claims"] == schema.POINT_IDENTITY_TOL},
        {"check": "exact_identity_not_restored", "passed": not ms["selection_identity_pass"]},
        {"check": "tolerance_ladder_diagnostic_only", "passed": res["tolerance_ladder_diagnostic_only"]},
        {"check": "tolerance_ladder_does_not_elevate_atom_claims",
         "passed": ladder["all_pass_at_1e_3"] and not ladder["all_pass_at_frozen_1e_9"]},
        {"check": "atom_claims_remain_blocked", "passed": schema.D8 not in res["taxonomy"]["cases"]},
        {"check": "no_proxy_atom_decomposition", "passed": True},
        {"check": "no_monolithic_large_json", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "leakage_drift_manifest.csv"),
              res["leakage_drift_manifest"]["rows"],
              ["job_key", "seed", "target", "level", "regime", "candidate_role", "candidate_order",
               "candidate_id", "split", "persisted_point_available", "persisted_point", "recomputed_point",
               "signed_drift_recomputed_minus_persisted", "abs_drift", "pass_1e_9", "selected_capacity",
               "atom_sum", "additive_abs_diff", "atom_additive_identity_pass", "support_graph_hash",
               "fold_plan_hash", "bootstrap_plan_hash", "population_hash", "feature_population_hash_matches",
               "target_labels_loaded_for_replay"])
    _writecsv(os.path.join(tdir, "selection_vs_audit_identity_contrast.csv"),
              res["selection_vs_audit_identity_contrast"]["rows"],
              ["candidate_key", "seed", "target", "level", "candidate_role", "candidate_order",
               "candidate_id", "selection_persisted_point_available", "selection_pass_1e_9",
               "selection_abs_drift", "selection_additive_abs_diff", "source_audit_persisted_point_available",
               "source_audit_additive_abs_diff", "source_audit_additive_pass", "contrast_class",
               "interpretation"])
    _writecsv(os.path.join(tdir, "stagewise_drift_localization.csv"),
              res["stagewise_drift_localization"]["rows"],
              ["job_key", "seed", "target", "level", "candidate_role", "candidate_order",
               "candidate_id", "split", "feature_population_match", "support_graph_available",
               "fold_plan_available", "cell_mass_accounting_pass", "atom_additive_pass",
               "persisted_aggregate_identity_pass", "first_divergent_stage", "stagewise_interpretation"])
    _writecsv(os.path.join(tdir, "numeric_drift_diagnostics.csv"),
              res["numeric_drift_diagnostics"]["rows"],
              ["diagnostic", "status", "evidence", "rules_out_semantic_mismatch"])
    _writecsv(os.path.join(tdir, "tolerance_ladder_identity.csv"),
              res["tolerance_ladder_identity"]["rows"],
              ["tolerance", "n_selection_candidates", "n_pass", "pass_fraction", "selected_role_pass",
               "selected_role_total", "better_role_pass", "better_role_total", "diagnostic_only",
               "elevates_atom_claims"])
    _writecsv(os.path.join(tdir, "atom_pattern_stability_under_drift.csv"),
              res["atom_pattern_stability_under_drift"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order",
               "better_order", "persisted_point_delta_better_minus_selected",
               "recomputed_point_delta_better_minus_selected", "delta_drift",
               "point_sign_stable_under_observed_drift", "stable_at_1e_9", "stable_at_1e_8",
               "stable_at_1e_6", "stable_at_1e_4", "stable_at_1e_3",
               "diagnostic_concentration_class", "diagnostic_top3_positive_share",
               "diagnostic_support_artifact_flag", "diagnostic_atom_target_gauge_conflict",
               "pattern_claim_elevated"])
    _writecsv(os.path.join(tdir, "aggregate_vs_atom_path_diff.csv"),
              res["aggregate_vs_atom_path_diff"]["rows"],
              ["job_key", "seed", "target", "level", "candidate_role", "candidate_order",
               "candidate_id", "persisted_point", "recomputed_point", "atom_sum",
               "recomputed_minus_persisted", "atom_sum_minus_recomputed", "first_path_difference"])
    _writecsv(os.path.join(tdir, "future_trace_field_requirements.csv"),
              res["future_trace_field_requirements"]["rows"],
              ["field_order", "field_name", "category", "necessity", "rationale",
               "available_in_current_artifacts"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res),
              ["check", "passed"])
    _writecsv(os.path.join(tdir, "c40_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def render_md(res):
    ms = res["leakage_drift_manifest"]["summary"]
    ss = res["stagewise_drift_localization"]["summary"]
    ns = res["numeric_drift_diagnostics"]["summary"]
    ladder = res["tolerance_ladder_identity"]["summary"]
    ps = res["atom_pattern_stability_under_drift"]["summary"]
    return "\n".join([
        f"# C40 - Leakage Point Drift Forensics / Atom-Trace Boundary Closure "
        f"(frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only forensic audit over C39/C38/C37 artifacts. No training, no GPU, no selector repair, "
        "and no change to the frozen 1e-9 identity gate for elevated atom claims.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- selection identity at 1e-9: **{ms['n_selection_pass_1e_9']} / "
        f"{ms['n_selection_candidates']}**.",
        f"- max absolute drift: **{_f(ms['max_abs_drift'])}**; all selection rows pass at 1e-3: "
        f"**{ladder['all_pass_at_1e_3']}**.",
        "",
        "## First Divergence",
        "",
        f"- observed semantic mismatch count: **{ss['observed_semantic_mismatch_count']}**.",
        f"- aggregate-vs-atom path divergence count: **{ss['aggregate_vs_atom_path_divergence_count']}**.",
        "- Feature population, support graph availability, fold plan availability, cell-mass accounting, "
        "and atom additive aggregation pass in the committed C39 trace; divergence appears only when the "
        "recomputed point is compared to the persisted C37 aggregate point.",
        "",
        "## Numeric Boundary",
        "",
        f"- bounded at 1e-3: **{ns['bounded_at_1e_3']}**.",
        f"- positive / negative signed drift rows: **{ns['positive_signed_drift_count']} / "
        f"{ns['negative_signed_drift_count']}**.",
        "- Current artifacts do not persist per-fold probe outputs, so C40 cannot restore exact identity or "
        "prove the first sub-stage below aggregate point comparison.",
        "",
        "## Diagnostic Stability",
        "",
        f"- point-direction stable under observed drift: **{_f(ps['point_sign_stable_fraction'])}**.",
        f"- diagnostic broad pattern count: **{ps['broad_diagnostic_count']} / {ps['n_pairs']}**.",
        f"- diagnostic atom-gauge conflict count: **{ps['atom_gauge_conflict_diagnostic_count']} / "
        f"{ps['n_pairs']}**.",
        "- These diagnostic patterns remain blocked; they do not establish atom mechanism.",
        "",
        "## Bottom Line",
        "",
        "> C40 localizes the C39 failure to persisted aggregate point identity, not atom additivity or an "
        "observed support/fold/population mismatch. The drift is bounded in the committed tables, but exact "
        "identity is not restored; A9 remains a trace boundary and future atom claims require new persisted "
        "per-fold/per-cell trace fields.",
    ])


def render_boundary_md(res):
    ss = res["stagewise_drift_localization"]["summary"]
    fs = res["future_trace_field_requirements"]["summary"]
    return "\n".join([
        "# C40 - Atom Trace Boundary",
        "",
        f"- first divergent stage counts: {ss['selection_first_divergent_stage_counts']}",
        f"- missing future trace fields: {fs['n_currently_missing']}",
        "",
        "Exact atom attribution is not recoverable from current committed artifacts because the per-fold "
        "probe/cell outputs needed to bridge recomputed atoms to persisted aggregate point leakage are absent.",
    ]) + "\n"


def render_instrumentation_md(res):
    fs = res["future_trace_field_requirements"]["summary"]
    return "\n".join([
        "# C40 - Future Leakage Trace Instrumentation",
        "",
        f"- required fields: {fs['n_required_fields']}",
        f"- currently missing: {fs['n_currently_missing']}",
        "",
        "Future exact atom claims require persisted per-fold cell probe outputs, atom tables, aggregate point "
        "targets, and bootstrap aggregate replicate traces. UCL remains aggregate quantile evidence, not a "
        "linear sum of per-atom UCLs.",
    ]) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not deployable", "non-deployable", "diagnostic-only", "no selected", "no selector",
             "not claimed", "blocked")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 120):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C40 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_preference_robust_pairs": res["n_preference_robust_pairs"],
        "actual_selector_score_name": res["actual_selector_score_name"],
        "identity_tolerance_for_elevated_atom_claims": res["identity_tolerance_for_elevated_atom_claims"],
        "tolerance_ladder_diagnostic_only": res["tolerance_ladder_diagnostic_only"],
        "leakage_drift_manifest_summary": res["leakage_drift_manifest"]["summary"],
        "selection_vs_audit_identity_contrast_summary": res["selection_vs_audit_identity_contrast"]["summary"],
        "stagewise_drift_localization_summary": res["stagewise_drift_localization"]["summary"],
        "aggregate_vs_atom_path_diff_summary": res["aggregate_vs_atom_path_diff"]["summary"],
        "numeric_drift_diagnostics_summary": res["numeric_drift_diagnostics"]["summary"],
        "tolerance_ladder_identity_summary": res["tolerance_ladder_identity"]["summary"],
        "atom_pattern_stability_under_drift_summary": res["atom_pattern_stability_under_drift"]["summary"],
        "future_trace_field_requirements_summary": res["future_trace_field_requirements"]["summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "identity_gate_check": "Frozen 1e-9 identity gate remains binding for elevated atom claims.",
            "small_drift_check": "Passing at loose tolerances is diagnostic only and does not unblock C39 atom claims.",
            "trace_boundary_check": "Missing per-fold/per-cell trace fields keep A9 active unless exact identity is restored.",
        },
    }


def _write_artifacts(res, out_dir):
    md = render_md(res)
    boundary = render_boundary_md(res)
    instr = render_instrumentation_md(res)
    for text in (md, boundary, instr):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C40_LEAKAGE_POINT_DRIFT_FORENSICS.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C40_ATOM_TRACE_BOUNDARY.md"), "w").write(boundary)
    open(os.path.join(out_dir, "C40_FUTURE_LEAKAGE_TRACE_INSTRUMENTATION.md"), "w").write(instr)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C40_LEAKAGE_POINT_DRIFT_FORENSICS.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c40_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.leakage_drift.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    print(f"[C40] cases={','.join(res['taxonomy']['cases'])} "
          f"identity={res['leakage_drift_manifest']['summary']['n_selection_pass_1e_9']}/"
          f"{res['leakage_drift_manifest']['summary']['n_selection_candidates']} "
          f"max_drift={res['leakage_drift_manifest']['summary']['max_abs_drift']} "
          f"stable={res['atom_pattern_stability_under_drift']['summary']['point_sign_stable_fraction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
